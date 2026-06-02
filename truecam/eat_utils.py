"""Lightweight, multi-class-capable EAT helpers.

Self-contained version of the helpers used in ``train_eat.py`` /
``notebooks/fast_lgb_validation.py``, with no heavy third-party imports
so it can be reused from notebooks safely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_tile_average_accuracy(predictions, labels, tag, level="tile",
                                   metric="accuracy", print_str=True):
    n_classes = predictions.shape[1]
    cols = {f"prediction_{c}": predictions[:, c] for c in range(n_classes)}
    df = pd.DataFrame({"tag": tag, "label": labels, **cols})
    if level != "tile":
        agg = {f"prediction_{c}": "mean" for c in range(n_classes)}
        agg["label"] = "first"
        df = df.groupby("tag").agg(agg).reset_index()
    if print_str:
        print("agg shape", df.shape)
    pred_cols = [f"prediction_{c}" for c in range(n_classes)]
    y_true = df["label"].values
    y_pred = df[pred_cols].values
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred.argmax(axis=1))
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred.argmax(axis=1))
    if metric == "auroc":
        if n_classes == 2:
            return roc_auc_score(y_true, y_pred[:, 1])
        return roc_auc_score(label_binarize(y_true, classes=range(n_classes)),
                             y_pred, multi_class="ovr", average="macro")
    raise ValueError(f"Unknown metric: {metric}")


def search_for_best_removing(amb_thres, predictions, labels, tag,
                             metric="accuracy", print_str=True):
    ambiguity = 1 - np.max(predictions, axis=1)
    keep = ambiguity < amb_thres
    # if any tile of a slide is masked, drop the whole slide
    dropped_slides = set(tag) - set(tag[keep])
    keep = keep | np.isin(tag, list(dropped_slides))
    score = evaluate_tile_average_accuracy(
        predictions[keep], labels[keep], tag[keep],
        level="patient", metric=metric, print_str=print_str)
    return score, 1 - keep.mean()


def binary_search_for_best_threshold(start, end, predictions, labels, tag,
                                     metric="accuracy", print_str=False):
    if start is None and end is None:
        amb = 1 - np.max(predictions, axis=1)
        start, end = np.quantile(amb, 0.01), np.quantile(amb, 0.98)
    best_threshold, best_score = start, -float("inf")
    while end - start > 1e-3:
        mid = (start + end) / 2
        score, _ = search_for_best_removing(
            mid, predictions, labels, tag, metric, print_str)
        if score > best_score:
            best_score, best_threshold, end = score, mid, mid
        else:
            start = mid
    return best_threshold, best_score


def create_ambiguity_dict(predictions, tag, start_quantile=0.1, end_quantile=0.9):
    quantile_list = np.quantile(
        1 - np.max(predictions, axis=1),
        np.linspace(start_quantile, end_quantile, 9))
    ambiguity_dict = {
        s: 1 - np.max(predictions[tag == s], axis=1) for s in np.unique(tag)
    }
    return ambiguity_dict, quantile_list


def evaluate_eat_patient_metric(best_threshold, predictions, labels, tag,
                                metrics=("accuracy", "balanced_accuracy")):
    out = []
    for m in metrics:
        out.append(evaluate_tile_average_accuracy(
            predictions, labels, tag, level="patient", metric=m, print_str=False))
    eat_scores = []
    proportion = None
    for m in metrics:
        s, p = search_for_best_removing(
            best_threshold, predictions, labels, tag, metric=m, print_str=False)
        eat_scores.append(s)
        proportion = p
    return (*out, *eat_scores, proportion)
