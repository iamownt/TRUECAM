from __future__ import annotations

import argparse
import gc
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel


def parse_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TITAN slide embeddings, optionally with EAT filtering.")
    parser.add_argument("--tile_dir", type=Path, default=Path("/home/user/sngp/TCGA-OT/Patch512/TITAN/h5_files"))
    parser.add_argument("--output_dir", type=Path,
                        default=Path("/home/user/sngp/TCGA-OT/Patch512/TITAN/h5_files_slide_entropy"))
    parser.add_argument("--mask_pkl_path", type=Path, default=None,
                        help="Ambiguity pickle from revision_may19/automl_eat.py.")
    parser.add_argument("--subtyping_task", type=str, default="NSCLC")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name used in merged pickle. Defaults to --subtyping_task.")
    parser.add_argument("--proxy_train_fraction", type=float, default=0.4)
    parser.add_argument("--keep_ratios", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    parser.add_argument("--start_trial", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--start_fold", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=20)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)
    parser.add_argument("--feature_key", type=str, default="features")
    parser.add_argument("--ambiguity_strategy", choices=["stored", "max_prob", "entropy"], default="entropy")
    parser.add_argument("--min_tiles", type=int, default=10,
                        help="Keep all tiles if filtering would leave this many or fewer.")
    parser.add_argument("--max_tiles", type=int, default=0,
                        help="Optional cap after EAT filtering; 0 disables capping.")
    parser.add_argument("--auto_skip", type=parse_bool, default=True)
    parser.add_argument("--merge", action=argparse.BooleanOptionalAction, default=True,
                        help="Merge generated per-slide H5 files into a pickle after each split.")
    parser.add_argument("--merge_only", action="store_true",
                        help="Only merge existing per-slide H5 files for the requested splits; do not encode slides.")
    return parser.parse_args()


def feature_from_mapping(obj: Dict, requested_key: str, source: Path):
    for key in [requested_key, "features", "tile_embeds"]:
        if key in obj:
            return obj[key]
    raise KeyError(f"No feature key found in {source}. Tried: {requested_key}, features, tile_embeds")


def load_tile_data(path: Path, feature_key: str = "features") -> Tuple[torch.Tensor, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix in [".h5", ".hdf5"]:
        with h5py.File(path, "r") as handle:
            features = feature_from_mapping(handle, feature_key, path)[:]
            coords = handle["coords"][:]
    elif suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if not isinstance(obj, dict):
            raise ValueError(f"PT files must contain features and coords for TITAN slide encoding: {path}")
        features = feature_from_mapping(obj, feature_key, path)
        coords = obj.get("coords")
        if coords is None:
            raise KeyError(f"Missing coords in {path}")
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
        if torch.is_tensor(coords):
            coords = coords.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported tile feature file: {path}")

    features = torch.as_tensor(features).float()
    coords = torch.as_tensor(coords).long()
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords[0]
    if features.shape[0] == 1:
        features = features.repeat(2, 1)
        coords = coords.repeat(2, 1)
        coords[1, 0] = coords[0, 0] + 512
    return features.unsqueeze(0), coords.unsqueeze(0)


def list_tile_files(tile_dir: Path) -> List[Path]:
    suffixes = {".h5", ".hdf5", ".pt"}
    files = [path for path in tile_dir.iterdir() if path.suffix.lower() in suffixes]
    files.sort(key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No TITAN tile files found in {tile_dir}")
    return files


def load_ambiguity(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def split_key_candidates(trial_idx: int, fold_idx: int) -> List[str]:
    return [f"split{fold_idx}", f"t{trial_idx}f{fold_idx}", f"split{trial_idx}_{fold_idx}"]


def get_split_ambiguity(mask_pkl: Optional[dict], trial_idx: int, fold_idx: int) -> Optional[dict]:
    if mask_pkl is None:
        return None
    for key in split_key_candidates(trial_idx, fold_idx):
        if key in mask_pkl:
            return mask_pkl[key]
    available = ", ".join(list(mask_pkl.keys())[:10])
    raise KeyError(f"No ambiguity key found for trial={trial_idx}, fold={fold_idx}. First keys: {available}")


def compute_ambiguity(entry, strategy: str) -> np.ndarray:
    if isinstance(entry, dict):
        if strategy == "stored" and "ambiguity" in entry:
            return np.asarray(entry["ambiguity"], dtype=np.float32)
        if strategy == "max_prob":
            probabilities = np.asarray(entry["probabilities"], dtype=np.float32)
            return 1.0 - np.max(probabilities, axis=1)
        if strategy == "entropy":
            probabilities = np.clip(np.asarray(entry["probabilities"], dtype=np.float32), 1e-7, 1.0)
            return -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        if "ambiguity" in entry:
            return np.asarray(entry["ambiguity"], dtype=np.float32)
    return np.asarray(entry, dtype=np.float32)


def apply_eat_filter(
    features: torch.Tensor,
    coords: torch.Tensor,
    slide_entry,
    keep_ratio: float,
    strategy: str,
    min_tiles: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if slide_entry is None:
        return features, coords
    ambiguity = compute_ambiguity(slide_entry, strategy)
    if features.shape[1] == 2 and ambiguity.shape[0] == 1:
        ambiguity = np.repeat(ambiguity, 2)
    if ambiguity.shape[0] != features.shape[1]:
        raise ValueError(f"Ambiguity length {ambiguity.shape[0]} does not match tile count {features.shape[1]}.")
    threshold = np.quantile(ambiguity, keep_ratio)
    keep = ambiguity <= threshold
    if int(keep.sum()) <= min_tiles:
        return features, coords
    return features[:, keep], coords[:, keep]


def cap_tiles(
    features: torch.Tensor,
    coords: torch.Tensor,
    max_tiles: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_tiles <= 0 or features.shape[1] <= max_tiles:
        return features, coords
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(features.shape[1], generator=generator)[:max_tiles]
    indices, _ = torch.sort(indices)
    return features[:, indices], coords[:, indices]


def encode_slide(
    model,
    features: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
    patch_size_level0: torch.Tensor,
) -> torch.Tensor:
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else torch.cpu.amp.autocast(), torch.inference_mode():
            return model.encode_slide_from_patch_features(
                features.to(device, non_blocking=True),
                coords.to(device, non_blocking=True),
                patch_size_level0,
            )
    except torch.cuda.OutOfMemoryError:
        cpu = torch.device("cpu")
        model.to(cpu)
        with torch.cpu.amp.autocast(), torch.inference_mode():
            embedding = model.encode_slide_from_patch_features(features.to(cpu), coords.to(cpu), patch_size_level0.cpu())
        model.to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return embedding


def split_output_dir(args: argparse.Namespace, fold_idx: int, keep_ratio: Optional[float]) -> Path:
    if keep_ratio is None:
        return args.output_dir
    return (
        args.output_dir
        / args.subtyping_task
        / f"prop{args.proxy_train_fraction}_split{fold_idx}_{keep_ratio}"
    )


def assigned_files(files: Iterable[Path], args: argparse.Namespace, split_ambiguity: Optional[dict]) -> List[Path]:
    if split_ambiguity is not None:
        valid = {key for key in split_ambiguity.keys() if key not in {"train_quantile_list", "val_quantile_list"}}
        files = [path for path in files if path.stem in valid]
    files = list(files)
    return [path for i, path in enumerate(files) if i % args.total_gpus == args.gpu_id]


def generate_for_split(
    model,
    files: List[Path],
    split_ambiguity: Optional[dict],
    args: argparse.Namespace,
    trial_idx: int,
    fold_idx: int,
    keep_ratio: Optional[float],
) -> Path:
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    patch_size_level0 = torch.tensor([512], device=device)
    out_dir = split_output_dir(args, fold_idx, keep_ratio)
    out_dir.mkdir(parents=True, exist_ok=True)

    work_files = assigned_files(files, args, split_ambiguity)
    desc = "baseline" if keep_ratio is None else f"split{fold_idx}@{keep_ratio}"
    for path in tqdm(work_files, desc=f"Encoding {desc}"):
        out_path = out_dir / f"{path.stem}.h5"
        if out_path.exists() and args.auto_skip:
            continue
        features, coords = load_tile_data(path, args.feature_key)
        slide_entry = None if split_ambiguity is None else split_ambiguity.get(path.stem)
        features, coords = apply_eat_filter(
            features,
            coords,
            slide_entry,
            keep_ratio if keep_ratio is not None else 1.0,
            args.ambiguity_strategy,
            args.min_tiles,
        )
        features, coords = cap_tiles(features, coords, args.max_tiles, seed=trial_idx * 10_000 + fold_idx)
        embedding = encode_slide(model, features, coords, device, patch_size_level0)
        with h5py.File(out_path, "w") as handle:
            handle.create_dataset("slide_embedding", data=embedding.detach().float().cpu().numpy())
        del features, coords, embedding
        gc.collect()
    return out_dir


def merge_slide_embeddings(out_dir: Path, dataset_name: str) -> Path:
    slide_embeddings = []
    slide_ids = []
    files = sorted(path for path in out_dir.iterdir() if path.suffix == ".h5")
    if not files:
        raise FileNotFoundError(f"No per-slide embeddings found in {out_dir}")
    for path in files:
        with h5py.File(path, "r") as handle:
            slide_embeddings.append(handle["slide_embedding"][:])
        slide_ids.append(path.stem)
    merged = {
        "embeddings": np.concatenate(slide_embeddings, axis=0),
        "filenames": slide_ids,
    }
    output_path = out_dir / f"{dataset_name}_titan_slide_embedding.pkl"
    with open(output_path, "wb") as handle:
        pickle.dump(merged, handle)
    print(f"Merged {len(slide_ids)} slide embeddings: {output_path}")
    return output_path


def main() -> None:
    args = get_args()
    dataset_name = args.dataset_name or args.subtyping_task

    if args.merge_only:
        for trial_idx in range(args.start_trial, args.start_trial + args.num_trials):
            first_fold = args.start_fold if trial_idx == args.start_trial else 0
            for fold_idx in range(first_fold, args.num_folds):
                for keep_ratio in args.keep_ratios:
                    merge_slide_embeddings(split_output_dir(args, fold_idx, keep_ratio), dataset_name)
        return

    files = list_tile_files(args.tile_dir)
    mask_pkl = load_ambiguity(args.mask_pkl_path)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device).eval()

    if mask_pkl is None:
        out_dir = generate_for_split(model, files, None, args, args.start_trial, args.start_fold, None)
        if args.merge and args.total_gpus == 1:
            merge_slide_embeddings(out_dir, dataset_name)
        return

    for trial_idx in range(args.start_trial, args.start_trial + args.num_trials):
        first_fold = args.start_fold if trial_idx == args.start_trial else 0
        for fold_idx in range(first_fold, args.num_folds):
            split_ambiguity = get_split_ambiguity(mask_pkl, trial_idx, fold_idx)
            for keep_ratio in args.keep_ratios:
                out_dir = generate_for_split(model, files, split_ambiguity, args, trial_idx, fold_idx, keep_ratio)
                if args.merge and args.total_gpus == 1:
                    merge_slide_embeddings(out_dir, dataset_name)


if __name__ == "__main__":
    main()
