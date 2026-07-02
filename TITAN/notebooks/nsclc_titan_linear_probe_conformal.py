"""Evaluate official, custom, and EAT TITAN embeddings on the 20 NSCLC splits."""

import argparse
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

warnings.filterwarnings("ignore", message="A NumPy version .* is required")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


LABEL_CSV_CANDIDATES = (
    Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv"),
)
CUSTOM_FEATURES = Path("/home/user/sngp/TCGA-OT/Patch512/TITAN/h5_files_slide/TCGA_TITAN_custom_features.pkl")
EAT_ROOT = Path("/home/user/sngp/TCGA-OT/Patch512/TITAN/h5_files_slide_entropy/NSCLC")
DEFAULT_KEEP_RATIOS = (0.4, 0.6, 0.8)

SLIDE_COL = "slide"
PATIENT_COL = "patient"
TARGET_COL = "cohort"
FOLD_RE = re.compile(r"^tao_split_trial_(\d+)_fold(\d+)$")


@dataclass(frozen=True)
class EmbeddingSource:
    name: str
    kind: str
    path: Optional[Path] = None
    keep_ratio: Optional[float] = None


class ConformalPrediction:
    """Split conformal prediction using scores from the TRUECAM demo."""

    def __init__(self, alpha: float, force_nonempty: bool = True):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.force_nonempty = force_nonempty
        self.qhat = None

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> None:
        n = probs.shape[0]
        scores = 1.0 - probs[np.arange(n), labels]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(scores, min(q_level, 1.0), method="higher")

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.qhat is None:
            raise RuntimeError("call calibrate before predict")
        pred_sets = probs >= (1.0 - self.qhat)
        if self.force_nonempty:
            empty = pred_sets.sum(axis=1) == 0
            if empty.any():
                pred_sets[empty, np.argmax(probs[empty], axis=1)] = True
        return pred_sets

    def evaluate(self, probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        pred_sets = self.predict(probs)
        coverage = pred_sets[np.arange(pred_sets.shape[0]), labels].mean()
        avg_size = pred_sets.sum(axis=1).mean()
        return float(coverage), float(avg_size)


def first_existing(paths: Tuple[Path, ...]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of these paths exist: " + ", ".join(map(str, paths)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 20-fold NSCLC TITAN linear probing and conformal prediction."
    )
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--official-features", type=Path, default=None)
    parser.add_argument("--custom-features", type=Path, default=CUSTOM_FEATURES)
    parser.add_argument("--eat-root", type=Path, default=EAT_ROOT)
    parser.add_argument("--train-data-proportion", type=float, default=0.4)
    parser.add_argument("--keep-ratios", type=float, nargs="+", default=list(DEFAULT_KEEP_RATIOS))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--c-steps", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutation-size", type=int, default=100)
    parser.add_argument("--calibration-size", type=int, default=100)
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--skip-custom", action="store_true")
    parser.add_argument("--skip-eat", action="store_true")
    parser.add_argument(
        "--allow-empty-sets",
        action="store_true",
        help="Keep raw conformal sets from the TRUECAM demo; default fills empty sets with argmax.",
    )
    return parser.parse_args()


def official_feature_path(path: Optional[Path]) -> Path:
    if path is not None:
        return path

    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("MahmoodLab/TITAN", filename="TCGA_TITAN_features.pkl"))


def load_embeddings(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        data = pickle.load(f)

    embeddings = np.asarray(data["embeddings"])
    df = pd.DataFrame(
        {
            SLIDE_COL: data["filenames"],
            "embedding": list(embeddings),
        }
    )
    return df.drop_duplicates(subset=SLIDE_COL, keep="first")


def ratio_label(value: Optional[float]) -> str:
    if value is None:
        return "baseline"
    return f"{value:g}"


def eat_embedding_path(eat_root: Path, train_data_proportion: float, fold_idx: int, keep_ratio: float) -> Path:
    return (
        eat_root
        / f"prop{ratio_label(train_data_proportion)}_split{fold_idx}_{ratio_label(keep_ratio)}"
        / "NSCLC_titan_slide_embedding.pkl"
    )


def fold_index_from_position(fold_col: str, folds: List[str]) -> int:
    try:
        return folds.index(fold_col)
    except ValueError as exc:
        raise ValueError(f"{fold_col} is not in the fold list") from exc


def source_dataframe(
    source: EmbeddingSource,
    labels: pd.DataFrame,
    fold_idx: int,
    eat_root: Path,
    train_data_proportion: float,
    fixed_cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if source.path is not None:
        if source.name not in fixed_cache:
            embeddings = load_embeddings(source.path)
            fixed_cache[source.name] = merge_labels_embeddings(labels, embeddings, source.name)
        return fixed_cache[source.name]

    if source.keep_ratio is None:
        raise ValueError(f"{source.name}: split-specific sources require keep_ratio")

    path = eat_embedding_path(eat_root, train_data_proportion, fold_idx, source.keep_ratio)
    if not path.exists():
        raise FileNotFoundError(f"{source.name}: missing split embedding file: {path}")
    embeddings = load_embeddings(path)
    return merge_labels_embeddings(labels, embeddings, f"{source.name}:split{fold_idx}")


def load_labels(path: Optional[Path]) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    label_path = path if path is not None else first_existing(LABEL_CSV_CANDIDATES)
    labels = pd.read_csv(label_path)
    missing_cols = {SLIDE_COL, PATIENT_COL, TARGET_COL} - set(labels.columns)
    if missing_cols:
        raise ValueError(f"{label_path} is missing columns: {sorted(missing_cols)}")

    class_names = sorted(labels[TARGET_COL].dropna().unique())
    label_map = {name: idx for idx, name in enumerate(class_names)}
    labels = labels.copy()
    labels["target"] = labels[TARGET_COL].map(label_map).astype(int)

    print(f"Labels: {label_path} ({len(labels)} slides, {labels[PATIENT_COL].nunique()} patients)")
    print(f"Label map: {label_map}")
    return labels, label_map, class_names


def fold_columns(labels: pd.DataFrame) -> List[str]:
    cols = [col for col in labels.columns if FOLD_RE.match(col)]
    cols = sorted(cols, key=lambda col: tuple(map(int, FOLD_RE.match(col).groups())))
    if len(cols) != 20:
        raise ValueError(f"expected 20 fold columns, found {len(cols)}")
    return cols


def merge_labels_embeddings(labels: pd.DataFrame, embeddings: pd.DataFrame, name: str) -> pd.DataFrame:
    merged = labels.merge(embeddings, on=SLIDE_COL, how="left", validate="one_to_one")
    missing = merged["embedding"].isna()
    if missing.any():
        examples = merged.loc[missing, SLIDE_COL].head(10).tolist()
        raise ValueError(f"{name}: missing {missing.sum()} embeddings; examples: {examples}")
    print(f"{name}: matched {len(merged)}/{len(labels)} slides")
    return merged


def split_arrays(df: pd.DataFrame, fold_col: str, split: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    split_df = df[df[fold_col] == split].copy()
    x = np.stack(split_df["embedding"].to_numpy()).astype(np.float32)
    y = split_df["target"].to_numpy(dtype=np.int64)
    return split_df, x, y


def c_grid(n_steps: int) -> np.ndarray:
    # Matches linear_probe_stacy.ipynb: log_spaced_values=0.1..1000, model C=1/log_spaced_value.
    return 1.0 / np.logspace(np.log10(10e-2), np.log10(10e2), num=n_steps)


def fit_best_lr(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    classes: np.ndarray,
    c_values: np.ndarray,
    max_iter: int,
) -> Tuple[LogisticRegression, float, float]:
    best_loss = np.inf
    best_model = None
    best_c = np.nan

    for c_value in c_values:
        model = LogisticRegression(
            C=float(c_value),
            fit_intercept=True,
            max_iter=max_iter,
            random_state=0,
            solver="lbfgs",
            class_weight=None,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(train_x, train_y)
        val_probs = model.predict_proba(val_x)
        val_loss = log_loss(val_y, val_probs, labels=classes)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_c = float(c_value)

    if best_model is None:
        raise RuntimeError("failed to fit logistic regression")
    return best_model, best_c, float(best_loss)


def aggregate_by_patient(
    split_df: pd.DataFrame,
    probs: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_df = pd.DataFrame(
        {
            PATIENT_COL: split_df[PATIENT_COL].to_numpy(),
            "target": labels,
            "probs": list(probs),
        }
    )
    label_counts = pred_df.groupby(PATIENT_COL)["target"].nunique()
    mixed = label_counts[label_counts > 1]
    if len(mixed):
        raise ValueError(f"found patients with conflicting labels: {mixed.index[:5].tolist()}")

    rows = []
    for patient_id, patient_df in pred_df.groupby(PATIENT_COL, sort=False):
        rows.append(
            {
                PATIENT_COL: patient_id,
                "target": int(patient_df["target"].iloc[0]),
                "probs": np.mean(np.stack(patient_df["probs"].to_numpy()), axis=0),
            }
        )
    aggregated = pd.DataFrame(rows)
    return (
        aggregated["target"].to_numpy(dtype=np.int64),
        np.stack(aggregated["probs"].to_numpy()),
        aggregated[PATIENT_COL].to_numpy(),
    )


def metric_dict(labels: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    preds = probs.argmax(axis=1)
    metrics = {
        "acc": accuracy_score(labels, preds),
        "bacc": balanced_accuracy_score(labels, preds),
        "kappa": cohen_kappa_score(labels, preds, weights="quadratic"),
        "nw_kappa": cohen_kappa_score(labels, preds, weights="linear"),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "loss": log_loss(labels, probs, labels=classes),
    }
    if len(classes) == 2:
        metrics["auroc"] = roc_auc_score(labels, probs[:, 1])
    else:
        metrics["auroc"] = roc_auc_score(labels, probs, labels=classes, multi_class="ovo", average="macro")
    return {key: float(value) for key, value in metrics.items()}


def conformal_metrics(
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    seed: int,
    permutation_size: int,
    calibration_size: int,
    force_nonempty: bool,
) -> Dict[str, float]:
    if calibration_size >= len(test_labels):
        raise ValueError(
            f"calibration_size={calibration_size} must be smaller than test cohort size={len(test_labels)}"
        )

    results = {
        coverage: {"coverage": [], "avg_size": []}
        for coverage in (0.95, 0.99)
    }
    for permutation_seed in range(seed, seed + permutation_size):
        rng = np.random.RandomState(permutation_seed)
        idx = rng.permutation(len(test_labels))
        cal_idx = idx[:calibration_size]
        eval_idx = idx[calibration_size:]

        cal_probs = test_probs[cal_idx]
        cal_labels = test_labels[cal_idx]
        eval_probs = test_probs[eval_idx]
        eval_labels = test_labels[eval_idx]

        for coverage, alpha in ((0.95, 0.05), (0.99, 0.01)):
            cp = ConformalPrediction(alpha=alpha, force_nonempty=force_nonempty)
            cp.calibrate(cal_probs, cal_labels)
            empirical_coverage, avg_size = cp.evaluate(eval_probs, eval_labels)
            results[coverage]["coverage"].append(empirical_coverage)
            results[coverage]["avg_size"].append(avg_size)

    metrics = {
        "cp_permutations": float(permutation_size),
        "cp_calibration_size": float(calibration_size),
    }
    for coverage in (0.95, 0.99):
        coverage_values = np.asarray(results[coverage]["coverage"])
        size_values = np.asarray(results[coverage]["avg_size"])
        metrics[f"cp_{coverage:.2f}_coverage"] = float(coverage_values.mean())
        metrics[f"cp_{coverage:.2f}_coverage_perm_std"] = float(coverage_values.std(ddof=0))
        metrics[f"cp_{coverage:.2f}_avg_size"] = float(size_values.mean())
        metrics[f"cp_{coverage:.2f}_avg_size_perm_std"] = float(size_values.std(ddof=0))
    return metrics


def prefixed_metrics(prefix: str, labels: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metric_dict(labels, probs, classes).items()}


def evaluate_source(
    source: EmbeddingSource,
    labels: pd.DataFrame,
    folds: List[str],
    classes: np.ndarray,
    c_values: np.ndarray,
    max_iter: int,
    seed: int,
    permutation_size: int,
    calibration_size: int,
    force_nonempty: bool,
    eat_root: Path,
    train_data_proportion: float,
    fixed_cache: Dict[str, pd.DataFrame],
) -> List[Dict[str, Union[float, str, int]]]:
    rows = []
    for fold_col in folds:
        fold_idx = fold_index_from_position(fold_col, folds)
        df = source_dataframe(source, labels, fold_idx, eat_root, train_data_proportion, fixed_cache)
        train_df, train_x, train_y = split_arrays(df, fold_col, "train")
        val_df, val_x, val_y = split_arrays(df, fold_col, "val")
        test_df, test_x, test_y = split_arrays(df, fold_col, "test")

        model, selected_c, val_selection_loss = fit_best_lr(
            train_x,
            train_y,
            val_x,
            val_y,
            classes,
            c_values,
            max_iter,
        )
        val_probs = model.predict_proba(val_x)
        test_probs = model.predict_proba(test_x)

        val_labels_pat, val_probs_pat, _ = aggregate_by_patient(val_df, val_probs, val_y)
        test_labels_pat, test_probs_pat, _ = aggregate_by_patient(test_df, test_probs, test_y)

        row = {
            "embedding": source.name,
            "source_type": source.kind,
            "keep_ratio": ratio_label(source.keep_ratio),
            "fold": fold_col,
            "fold_idx": fold_idx,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "selected_c": selected_c,
            "val_selection_loss": val_selection_loss,
        }
        row.update(prefixed_metrics("val", val_labels_pat, val_probs_pat, classes))
        row.update(metric_dict(test_labels_pat, test_probs_pat, classes))
        row.update(
            conformal_metrics(
                test_labels_pat,
                test_probs_pat,
                seed=seed,
                permutation_size=permutation_size,
                calibration_size=calibration_size,
                force_nonempty=force_nonempty,
            )
        )
        rows.append(row)

        print(
            f"{source.name:16s} {fold_col}: "
            f"val_acc={row['val_acc']:.4f}, val_bacc={row['val_bacc']:.4f}, "
            f"test_acc={row['acc']:.4f}, test_bacc={row['bacc']:.4f}, auroc={row['auroc']:.4f}, "
            f"cp95={row['cp_0.95_coverage']:.4f}/{row['cp_0.95_avg_size']:.2f}, "
            f"cp99={row['cp_0.99_coverage']:.4f}/{row['cp_0.99_avg_size']:.2f}"
        )
    return rows


def summarize(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "val_acc",
        "val_bacc",
        "val_kappa",
        "val_nw_kappa",
        "val_weighted_f1",
        "val_loss",
        "val_auroc",
        "acc",
        "bacc",
        "kappa",
        "nw_kappa",
        "weighted_f1",
        "loss",
        "auroc",
        "cp_0.95_coverage",
        "cp_0.95_coverage_perm_std",
        "cp_0.95_avg_size",
        "cp_0.95_avg_size_perm_std",
        "cp_0.99_coverage",
        "cp_0.99_coverage_perm_std",
        "cp_0.99_avg_size",
        "cp_0.99_avg_size_perm_std",
    ]
    rows = []
    group_cols = ["embedding", "source_type", "keep_ratio"]
    for keys, group in fold_results.groupby(group_cols, sort=False):
        embedding, source_type, keep_ratio = keys
        for metric in metric_cols:
            rows.append(
                {
                    "embedding": embedding,
                    "source_type": source_type,
                    "keep_ratio": keep_ratio,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "std": group[metric].std(ddof=0),
                    "n_folds": len(group),
                }
            )
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    print("\n20-fold summary (mean +/- std)")
    for keys, group in summary.groupby(["embedding", "source_type", "keep_ratio"], sort=False):
        embedding, source_type, keep_ratio = keys
        print(f"\n{embedding} ({source_type}, keep_ratio={keep_ratio})")
        for row in group.itertuples(index=False):
            value_fmt = "{:.2f}" if row.metric.endswith("avg_size") else "{:.4f}"
            print(
                f"  {row.metric:<20} "
                f"{value_fmt.format(row.mean)} +/- {value_fmt.format(row.std)}"
            )


def validation_accuracy_table(fold_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["embedding", "source_type", "keep_ratio"]
    for keys, group in fold_results.groupby(group_cols, sort=False):
        embedding, source_type, keep_ratio = keys
        rows.append(
            {
                "embedding": embedding,
                "source_type": source_type,
                "keep_ratio": keep_ratio,
                "val_acc_mean": group["val_acc"].mean(),
                "val_acc_std": group["val_acc"].std(ddof=0),
                "val_bacc_mean": group["val_bacc"].mean(),
                "val_bacc_std": group["val_bacc"].std(ddof=0),
                "test_acc_mean": group["acc"].mean(),
                "test_acc_std": group["acc"].std(ddof=0),
                "test_bacc_mean": group["bacc"].mean(),
                "test_bacc_std": group["bacc"].std(ddof=0),
                "n_folds": len(group),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["val_acc_mean", "val_bacc_mean"], ascending=False).reset_index(drop=True)


def print_validation_accuracy_table(table: pd.DataFrame) -> None:
    print("\nValidation accuracy table (sorted by validation accuracy; test metrics are shown only for reference)")
    display_cols = [
        "embedding",
        "source_type",
        "keep_ratio",
        "val_acc_mean",
        "val_acc_std",
        "val_bacc_mean",
        "val_bacc_std",
        "test_acc_mean",
        "test_bacc_mean",
        "n_folds",
    ]
    printable = table[display_cols].copy()
    for col in printable.select_dtypes(include=[np.number]).columns:
        if col != "n_folds":
            printable[col] = printable[col].map(lambda value: f"{value:.4f}")
    print(printable.to_string(index=False))


def build_embedding_sources(args: argparse.Namespace) -> List[EmbeddingSource]:
    sources: List[EmbeddingSource] = []
    if not args.skip_official:
        sources.append(
            EmbeddingSource(
                name="official",
                kind="official",
                path=official_feature_path(args.official_features),
            )
        )
    if not args.skip_custom:
        sources.append(EmbeddingSource(name="custom", kind="custom", path=args.custom_features))
    if not args.skip_eat:
        for keep_ratio in args.keep_ratios:
            sources.append(
                EmbeddingSource(
                    name=f"custom_eat_{ratio_label(keep_ratio)}",
                    kind="custom_eat",
                    keep_ratio=float(keep_ratio),
                )
            )
    if not sources:
        raise ValueError("no embedding sources selected")
    return sources


def main() -> None:
    args = parse_args()
    labels, _, class_names = load_labels(args.labels_csv)
    folds = fold_columns(labels)
    classes = np.arange(len(class_names))
    c_values = c_grid(args.c_steps)

    all_rows = []
    fixed_cache: Dict[str, pd.DataFrame] = {}
    for source in build_embedding_sources(args):
        all_rows.extend(
            evaluate_source(
                source,
                labels,
                folds,
                classes,
                c_values,
                args.max_iter,
                seed=args.seed,
                permutation_size=args.permutation_size,
                calibration_size=args.calibration_size,
                force_nonempty=not args.allow_empty_sets,
                eat_root=args.eat_root,
                train_data_proportion=args.train_data_proportion,
                fixed_cache=fixed_cache,
            )
        )

    fold_results = pd.DataFrame(all_rows)
    summary = summarize(fold_results)
    val_table = validation_accuracy_table(fold_results)
    print_summary(summary)
    print_validation_accuracy_table(val_table)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "nsclc_titan_linear_probe_conformal_fold_results.csv"
    summary_path = args.output_dir / "nsclc_titan_linear_probe_conformal_summary.csv"
    val_table_path = args.output_dir / "nsclc_titan_linear_probe_conformal_validation_accuracy.csv"
    fold_results.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    val_table.to_csv(val_table_path, index=False)
    print(f"\nSaved fold results: {fold_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved validation accuracy table: {val_table_path}")


if __name__ == "__main__":
    main()
