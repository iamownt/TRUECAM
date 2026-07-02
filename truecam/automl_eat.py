from __future__ import annotations

import argparse
import gc
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm

sys.path.extend(["/home/user/wangtao/prov-gigapath"])

from train_eat import (  # noqa: E402
    binary_search_for_best_threshold,
    evaluate_eat_patient_metric,
)
from diverse_mil.utils import get_dataset_type, load_data_from_split, save_pickle_data, seed_everything  # noqa: E402


@dataclass(frozen=True)
class FeatureConfig:
    model_name: str
    dataset_h5: Path
    embed_dim: int


BASE_FEATURE_CONFIGS = {
    "uni": FeatureConfig("uni", Path("/home/user/TCGA-OT/Patch256/UNI/pt_files"), 1024),
    "conch": FeatureConfig("conch", Path("/home/user/TCGA-OT/Patch256/CONCH/pt_files"), 512),
    "titan": FeatureConfig("titan", Path("/home/user/sngp/TCGA-OT/Patch512/TITAN/pt_files"), 768),
}


def parse_int_or_none(value: str) -> Optional[int]:
    if value.lower() == "none":
        return None
    return int(value)


def get_eat_lr_params() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description="Train a weakly supervised LightGBM proxy for EAT tile ambiguity."
    )
    parser.add_argument("--model_name", choices=["uni", "conch", "titan"], default="titan")
    parser.add_argument("--subtyping_task", type=str, default="NSCLC")
    parser.add_argument("--sample_fraction", type=float, default=0.4,
                        help="Fraction of each training slide's tiles used to train the proxy.")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--evaluate_only", action="store_true",
                        help="Load saved LightGBM models and regenerate ambiguity/evaluation outputs.")

    parser.add_argument("--num_runs", type=int, default=20,
                        help="Number of folds. Kept for compatibility with older scripts.")
    parser.add_argument("--num_folds", type=int, default=None,
                        help="Overrides --num_runs when set.")
    parser.add_argument("--start_fold", type=int, default=0,
                        help="First fold index to process.")
    parser.add_argument("--start_trial", type=int, default=0,
                        help="First trial index. Usually 0 for the revision split files.")
    parser.add_argument("--num_trials", type=int, default=1,
                        help="Number of trial seeds to run.")

    parser.add_argument("--save_destination", type=Path, default=Path("/home/user/sngp/TCGA-OT/models"))
    parser.add_argument("--model_output_dir", type=Path, default=Path("/home/user/sngp/TCGA-OT/models/lightgbm_proxy"))
    parser.add_argument("--results_file_path", type=str, default="final_results_lightgbm_proxy.csv")
    parser.add_argument("--skip_existing_models", action="store_true",
                        help="Skip training when a fold model already exists; still scores the fold.")
    parser.add_argument("--write_trial_fold_aliases", action=argparse.BooleanOptionalAction, default=True,
                        help="Also save t{trial}f{fold} keys for legacy slide-embedding scripts.")

    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument("--use_balance_weight", dest="use_balance_weight", action="store_true", default=True,
                               help="Use class_weight='balanced' in LightGBM. Default: enabled.")
    balance_group.add_argument("--no_balance_weight", dest="use_balance_weight", action="store_false",
                               help="Disable LightGBM class balancing.")

    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--num_leaves", type=int, default=64)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--reg_alpha", type=float, default=0.1)
    parser.add_argument("--reg_lambda", type=float, default=0.1)
    parser.add_argument("--n_jobs", type=int, default=16)
    parser.add_argument("--predict_batch_size", type=int, default=250_000)
    parser.add_argument("--feature_key", type=str, default="features",
                        help="Feature key for H5/dict PT files; falls back to features/tile_embeds.")

    # Accepted for old launch scripts; intentionally unused in this LightGBM-only refactor.
    parser.add_argument("--num_gpus", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--presets", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--time_limit", type=parse_int_or_none, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()
    args.num_folds = args.num_folds if args.num_folds is not None else args.num_runs
    args.num_runs = args.num_folds
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("--sample_fraction must be in (0, 1].")
    if args.start_fold < 0 or args.start_fold >= args.num_folds:
        raise ValueError("--start_fold must be within [0, num_folds).")
    if args.num_trials < 1:
        raise ValueError("--num_trials must be >= 1.")
    if "bracs" in args.subtyping_task:
        assert args.subtyping_task in ["bracs-coarse", "bracs-finegrain"]

    args.save_destination.mkdir(parents=True, exist_ok=True)
    args.model_output_dir.mkdir(parents=True, exist_ok=True)
    args.weight_suffix = "balanced" if args.use_balance_weight else "unbalanced"
    return parser, args


def load_training_config(model_name: str, subtyping_task: Optional[str] = None) -> FeatureConfig:
    base = BASE_FEATURE_CONFIGS[model_name]
    dataset_type = get_dataset_type(subtyping_task)
    middle = "Patch512" if model_name == "titan" else "Patch256"

    if dataset_type in ["bracs-coarse", "bracs-finegrain"]:
        root = Path("/home/user/sngp/BRACS")
    elif dataset_type == "dhmc_luad":
        root = Path("/home/user/sngp/DHMC/LUAD")
    elif dataset_type == "tcga_ot":
        root = Path("/home/user/sngp/TCGA-OT")
    elif dataset_type == "qmh_lung":
        root = Path("/home/user/sngp/QMH")
    elif dataset_type == "dhmc_pathben":
        root = Path("/home/user/sngp/DHMC/LUAD")
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

    dataset_h5 = root / middle / model_name.upper() / "pt_files"
    config = FeatureConfig(model_name=model_name, dataset_h5=dataset_h5, embed_dim=base.embed_dim)
    print(f"Using tile feature path: {config.dataset_h5}")
    return config


def _feature_from_mapping(obj: Dict, requested_key: str, source: Path):
    for key in [requested_key, "features", "tile_embeds"]:
        if key in obj:
            return obj[key]
    raise KeyError(f"No feature key found in {source}. Tried: {requested_key}, features, tile_embeds")


def read_feature_matrix(path: Path, feature_key: str = "features") -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in [".h5", ".hdf5"]:
        with h5py.File(path, "r") as handle:
            features = _feature_from_mapping(handle, feature_key, path)[:]
    elif suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            features = _feature_from_mapping(obj, feature_key, path)
        else:
            features = obj
        if torch.is_tensor(features):
            features = features.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported feature file extension: {path}")

    features = np.asarray(features)
    if features.ndim == 3 and features.shape[0] == 1:
        features = features[0]
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D tile feature matrix in {path}, got shape {features.shape}")
    return np.asarray(features, dtype=np.float32, order="C")


def iter_feature_paths(dataset_h5: Path, filter_mapping: Dict[str, int]) -> List[Path]:
    suffixes = {".pt", ".h5", ".hdf5"}
    paths = [
        dataset_h5 / name
        for name in sorted(p.name for p in dataset_h5.iterdir() if p.suffix.lower() in suffixes)
        if Path(name).stem in filter_mapping
    ]
    if not paths:
        raise FileNotFoundError(f"No feature files in {dataset_h5} matched the split mapping.")
    return paths


def sample_tiles(features: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    if fraction >= 1:
        return features
    n_tiles = features.shape[0]
    keep = max(1, int(np.ceil(n_tiles * fraction)))
    indices = np.sort(rng.choice(n_tiles, size=keep, replace=False))
    return features[indices]


def build_training_tiles(
    dataset_h5: Path,
    filter_mapping: Dict[str, int],
    sample_fraction: float,
    rng: np.random.Generator,
    feature_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    x_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []

    paths = iter_feature_paths(dataset_h5, filter_mapping)
    for path in tqdm(paths, desc="Loading sampled train tiles"):
        slide_id = path.stem
        features = sample_tiles(read_feature_matrix(path, feature_key), sample_fraction, rng)
        x_chunks.append(features)
        y_chunks.append(np.full(features.shape[0], filter_mapping[slide_id], dtype=np.int64))

    x_train = np.concatenate(x_chunks, axis=0)
    y_train = np.concatenate(y_chunks, axis=0)
    print(f"train tiles={x_train.shape}, classes={dict(zip(*np.unique(y_train, return_counts=True)))}")
    return x_train, y_train


def make_lightgbm(args: argparse.Namespace, num_classes: int, random_state: int) -> lgb.LGBMClassifier:
    objective = "binary" if num_classes == 2 else "multiclass"
    params = {
        "boosting_type": "gbdt",
        "objective": objective,
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "class_weight": "balanced" if args.use_balance_weight else None,
        "random_state": random_state,
        "n_jobs": args.n_jobs,
        "device_type": "cpu",
        "verbosity": -1,
    }
    if num_classes > 2:
        params["num_class"] = num_classes
    return lgb.LGBMClassifier(**params)


def model_path(args: argparse.Namespace, trial_idx: int, fold_idx: int) -> Path:
    run_dir = args.model_output_dir / (
        f"{args.model_name}_{args.subtyping_task}_frac{args.sample_fraction}_{args.weight_suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / f"trial{trial_idx}_fold{fold_idx}.pkl"


def save_model(model: lgb.LGBMClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load_model(path: Path) -> lgb.LGBMClassifier:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def train_lightgbm_proxy(
    config: FeatureConfig,
    args: argparse.Namespace,
    train_mapping: Dict[str, int],
    num_classes: int,
    trial_idx: int,
    fold_idx: int,
) -> lgb.LGBMClassifier:
    path = model_path(args, trial_idx, fold_idx)
    if args.evaluate_only or (args.skip_existing_models and path.exists()):
        print(f"Loading LightGBM proxy: {path}")
        return load_model(path)

    rng = np.random.default_rng(args.seed + trial_idx * 10_000 + fold_idx)
    x_train, y_train = build_training_tiles(
        config.dataset_h5, train_mapping, args.sample_fraction, rng, args.feature_key
    )
    model = make_lightgbm(args, num_classes=num_classes, random_state=args.seed + trial_idx * 10_000 + fold_idx)
    print("Training LightGBM proxy...")
    model.fit(x_train, y_train)
    save_model(model, path)
    del x_train, y_train
    gc.collect()
    print(f"Saved LightGBM proxy: {path}")
    return model


def predict_proba_batched(model: lgb.LGBMClassifier, features: np.ndarray, batch_size: int) -> np.ndarray:
    if features.shape[0] <= batch_size:
        return np.asarray(model.predict_proba(features), dtype=np.float32)
    chunks = []
    for start in range(0, features.shape[0], batch_size):
        chunks.append(np.asarray(model.predict_proba(features[start:start + batch_size]), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def quantiles_from_ambiguity(ambiguity_chunks: Iterable[np.ndarray]) -> np.ndarray:
    ambiguity = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in ambiguity_chunks], axis=0)
    return np.quantile(ambiguity, np.linspace(0.1, 0.9, 9))


def score_split(
    split_name: str,
    dataset_h5: Path,
    filter_mapping: Dict[str, int],
    patient_mapping: Dict[str, str],
    model: lgb.LGBMClassifier,
    args: argparse.Namespace,
    keep_eval_arrays: bool,
) -> Tuple[Dict[str, dict], np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    ambiguity_dict: Dict[str, dict] = defaultdict(dict)
    ambiguity_chunks: List[np.ndarray] = []
    prediction_chunks: List[np.ndarray] = []
    label_chunks: List[np.ndarray] = []
    patient_tag_chunks: List[np.ndarray] = []

    paths = iter_feature_paths(dataset_h5, filter_mapping)
    for path in tqdm(paths, desc=f"Scoring {split_name} tiles"):
        slide_id = path.stem
        features = read_feature_matrix(path, args.feature_key)
        probabilities = predict_proba_batched(model, features, args.predict_batch_size)
        ambiguity = 1.0 - np.max(probabilities, axis=1)

        ambiguity_dict[slide_id]["ambiguity"] = ambiguity.astype(np.float16)
        ambiguity_dict[slide_id]["probabilities"] = probabilities.astype(np.float16)
        ambiguity_chunks.append(ambiguity)

        if keep_eval_arrays:
            labels = np.full(probabilities.shape[0], filter_mapping[slide_id], dtype=np.int64)
            patient_id = slide_id if "bracs" in args.subtyping_task else patient_mapping.get(slide_id, slide_id)
            prediction_chunks.append(probabilities)
            label_chunks.append(labels)
            patient_tag_chunks.append(np.full(probabilities.shape[0], patient_id, dtype=object))

    quantile_list = quantiles_from_ambiguity(ambiguity_chunks)
    if not keep_eval_arrays:
        return ambiguity_dict, quantile_list, None, None, None

    return (
        ambiguity_dict,
        quantile_list,
        np.concatenate(prediction_chunks, axis=0),
        np.concatenate(label_chunks, axis=0),
        np.concatenate(patient_tag_chunks, axis=0),
    )


def mapping_from_split(df: pd.DataFrame) -> Dict[str, int]:
    return df.set_index("slide_id")["cohort"].to_dict()


def evaluate_predictions(
    val_predictions: np.ndarray,
    val_labels: np.ndarray,
    val_patient_tags: np.ndarray,
    val_quantiles: np.ndarray,
    test_predictions: np.ndarray,
    test_labels: np.ndarray,
    test_patient_tags: np.ndarray,
) -> Dict[str, float]:
    val_tile_acc = accuracy_score(val_labels, val_predictions.argmax(axis=1))
    val_tile_bacc = balanced_accuracy_score(val_labels, val_predictions.argmax(axis=1))
    best_threshold, best_accuracy = binary_search_for_best_threshold(
        val_quantiles[0],
        val_quantiles[-1],
        val_predictions,
        val_labels,
        val_patient_tags,
        metric="balanced_accuracy",
        print_str=False,
    )
    test_acc, test_bacc, test_eat_acc, test_eat_bacc, proportion = evaluate_eat_patient_metric(
        best_threshold,
        test_predictions,
        test_labels,
        test_patient_tags,
    )
    return {
        "val_tile_acc": float(val_tile_acc),
        "val_tile_bacc": float(val_tile_bacc),
        "best_threshold": float(best_threshold),
        "best_val_bacc": float(best_accuracy),
        "test_acc": float(test_acc),
        "test_bacc": float(test_bacc),
        "test_eat_acc": float(test_eat_acc),
        "test_eat_bacc": float(test_eat_bacc),
        "proportion": float(proportion),
    }


def run_fold(
    config: FeatureConfig,
    args: argparse.Namespace,
    trial_idx: int,
    fold_idx: int,
) -> Tuple[Dict[str, float], Dict[str, dict]]:
    args.split_idx = fold_idx
    train_df, val_df, test_df, label_dict, slide_to_patient = load_data_from_split(args, fold_idx)
    num_classes = len(np.unique(list(label_dict.values())))
    train_mapping = mapping_from_split(train_df)
    val_mapping = mapping_from_split(val_df)
    test_mapping = mapping_from_split(test_df)

    print(
        f"trial={trial_idx} fold={fold_idx} | "
        f"train={train_df.shape} val={val_df.shape} test={test_df.shape} classes={num_classes}"
    )
    print("train labels:", train_df["cohort"].value_counts().sort_index().to_dict())
    print("val labels:", val_df["cohort"].value_counts().sort_index().to_dict())
    print("test labels:", test_df["cohort"].value_counts().sort_index().to_dict())

    model = train_lightgbm_proxy(config, args, train_mapping, num_classes, trial_idx, fold_idx)

    train_amb, train_quantiles, *_ = score_split(
        "train", config.dataset_h5, train_mapping, slide_to_patient, model, args, keep_eval_arrays=False
    )
    val_amb, val_quantiles, val_predictions, val_labels, val_patient_tags = score_split(
        "val", config.dataset_h5, val_mapping, slide_to_patient, model, args, keep_eval_arrays=True
    )
    test_amb, _, test_predictions, test_labels, test_patient_tags = score_split(
        "test", config.dataset_h5, test_mapping, slide_to_patient, model, args, keep_eval_arrays=True
    )

    metrics = evaluate_predictions(
        val_predictions,
        val_labels,
        val_patient_tags,
        val_quantiles,
        test_predictions,
        test_labels,
        test_patient_tags,
    )
    print(
        f"fold={fold_idx} test bacc={metrics['test_bacc']:.4f} "
        f"eat bacc={metrics['test_eat_bacc']:.4f} remove={metrics['proportion']:.4f}"
    )

    fold_ambiguity: Dict[str, dict] = defaultdict(dict)
    fold_ambiguity.update(train_amb)
    fold_ambiguity.update(val_amb)
    fold_ambiguity.update(test_amb)
    fold_ambiguity["train_quantile_list"] = train_quantiles
    fold_ambiguity["val_quantile_list"] = val_quantiles

    del model, val_predictions, val_labels, val_patient_tags, test_predictions, test_labels, test_patient_tags
    gc.collect()
    return metrics, fold_ambiguity


def ambiguity_output_path(args: argparse.Namespace) -> Path:
    return (
        args.save_destination
        / "ambpkl"
        / f"{args.model_name}_{args.subtyping_task}_ambiguity_dict_lightgbm_{args.sample_fraction}_{args.weight_suffix}.pkl"
    )


def summarize_metrics(args: argparse.Namespace, metric_rows: List[Dict[str, float]]) -> pd.DataFrame:
    metric_df = pd.DataFrame(metric_rows)
    run_columns = [col for col in ["trial", "fold"] if col in metric_df.columns]
    metric_columns = [col for col in metric_df.columns if col not in run_columns]
    summary_values = {}
    for metric in metric_columns:
        mean = metric_df[metric].mean()
        std = metric_df[metric].std()
        summary_values[metric] = f"{mean:.4f}+-{std:.4f}"
    summary = pd.DataFrame([summary_values])
    for metric in metric_columns:
        summary[f"{metric}_list"] = [metric_df[metric].tolist()]
    for run_column in run_columns:
        summary[f"{run_column}_list"] = [metric_df[run_column].astype(int).tolist()]
    summary["tag"] = f"{args.model_name}_{args.subtyping_task}_lightgbm_frac{args.sample_fraction}_{args.weight_suffix}"
    summary["start_trial"] = args.start_trial
    summary["num_trials"] = args.num_trials
    summary["start_fold"] = args.start_fold
    summary["num_folds"] = args.num_folds
    return summary


def append_results(args: argparse.Namespace, summary: pd.DataFrame) -> None:
    results_path = args.save_destination / args.results_file_path
    if results_path.exists():
        existing = pd.read_csv(results_path)
        summary = pd.concat([existing, summary], ignore_index=True)
    summary.to_csv(results_path, index=False)
    print(f"Saved metrics: {results_path}")


def main_spawn(args: argparse.Namespace) -> None:
    config = load_training_config(args.model_name, args.subtyping_task)
    start = time.time()
    metric_rows: List[Dict[str, float]] = []
    ambiguity_keeper: Dict[str, dict] = {}
    output_path = ambiguity_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running LightGBM EAT proxy: model={args.model_name}, task={args.subtyping_task}, "
        f"folds={args.start_fold}..{args.num_folds - 1}, trials={args.start_trial}..{args.start_trial + args.num_trials - 1}"
    )

    for trial_idx in range(args.start_trial, args.start_trial + args.num_trials):
        first_fold = args.start_fold if trial_idx == args.start_trial else 0
        for fold_idx in range(first_fold, args.num_folds):
            metrics, fold_ambiguity = run_fold(config, args, trial_idx, fold_idx)
            metrics["trial"] = trial_idx
            metrics["fold"] = fold_idx
            metric_rows.append(metrics)

            split_key = f"split{fold_idx}" if args.num_trials == 1 else f"split{trial_idx}_{fold_idx}"
            ambiguity_keeper[split_key] = fold_ambiguity
            if args.write_trial_fold_aliases:
                ambiguity_keeper[f"t{trial_idx}f{fold_idx}"] = fold_ambiguity
            save_pickle_data(ambiguity_keeper, output_path)
            print(f"Updated ambiguity pickle: {output_path}")

    summary = summarize_metrics(args, metric_rows)
    print(summary)
    append_results(args, summary)
    print(f"Duration: {(time.time() - start) / 60:.2f} min")


if __name__ == "__main__":
    _, parsed_args = get_eat_lr_params()
    print(parsed_args)
    seed_everything(parsed_args.seed)
    main_spawn(parsed_args)
