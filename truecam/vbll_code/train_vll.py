import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import vbll
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import os
import time
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from utils import AverageMeter, seed_everything, null_output_metric, load_pickle_data
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, classification_report
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from revision_may19.revision_utils.vbll_utils import VBLLMLP

TCGA_INT_TO_PATIENT = load_pickle_data(".../tcga_int_to_p.pkl")
CPTAC_INT_TO_PATIENT = load_pickle_data(".../cptac_int_to_p.pkl")




def load_data_from_h5(folder_path, prefix):
    """Load features, labels, and other data from h5 files."""
    with h5py.File(os.path.join(folder_path, f'{prefix}_rff.h5'), 'r') as f:
        features = f['rff'][:]

    with h5py.File(os.path.join(folder_path, f'{prefix}_y_true.h5'), 'r') as f:
        labels = f['y_true'][:]

    with h5py.File(os.path.join(folder_path, f'{prefix}_tile_to_slides.h5'), 'r') as f:
        tile_to_slides = f['tile_to_slides'][:]
    return features, labels, tile_to_slides


def get_vbll_params():
    parser = argparse.ArgumentParser(description='Train VBLL models on H5 data')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--data_path', type=str, default="/home/hdd1/sngp/images_r4_d",
                        help='Path to the folder containing trial and fold directories')
    parser.add_argument('--model_type', type=str, default='discriminative', choices=['discriminative', 'generative'],
                        help='Type of VBLL model to use')
    parser.add_argument('--balance_sampling', action='store_true', help='balance the sampling')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--reg_weight', type=float, default=1/1000000., help='Regularization weight')
    parser.add_argument('--prior_scale', type=float, default=1.0, help='Prior scale')
    parser.add_argument('--parameterization', type=str, default='diagonal', choices=['diagonal', 'full'],
                        help='Parameterization type')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
    parser.add_argument('--num_trials', type=int, default=4, help='Number of trials to run')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds per trial')
    parser.add_argument('--save_destination', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--results_file', type=str, default='vbll_results.csv', help='CSV file to save results')
    parser.add_argument('--save_model', action='store_true', help='Save trained models')
    # New arguments for patient-level evaluation
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions to file')
    parser.add_argument('--predictions_dir', type=str, default='predictions', help='Directory to save predictions')
    args = parser.parse_args()

    return parser, args


def aggregate_to_patient_level(predictions, probabilities, uncertainty, tile_to_slides, dataset_type, labels, num_classes):
    """
    Aggregate tile-level predictions to patient-level with configurable strategies.

    Args:
        predictions: Tile-level predicted classes
        probabilities: Tile-level class probabilities
        uncertainty: Tile-level uncertainty values
        tile_to_slides: Mapping from tiles to slides
        dataset_type: Either 'tcga' or 'cptac' to determine the mapping dictionary
        labels: Ground truth labels
        num_classes: Number of classes

    Returns:
        Dictionary containing tile, slide, and patient-level results
    """
    # Map slides to patients based on dataset type
    patient_map = TCGA_INT_TO_PATIENT if dataset_type == 'tcga' else CPTAC_INT_TO_PATIENT

    # Tile-level results
    tile_results = {
        'probability': probabilities,
        'uncertainty': uncertainty,
        'label': labels,
        'slide_id': tile_to_slides
    }

    # First aggregate to slide level
    slide_data = {'slide_id': [], 'patient_id': [], 'labels': []}
    for i in range(num_classes):
        slide_data[f'prob_{i}'] = []  # Using 'prob' consistently instead of 'logit'

    if uncertainty is not None:
        slide_data['uncertainty'] = []

    # Get unique slides
    unique_slides = np.unique(tile_to_slides)

    for slide_id in unique_slides:
        mask = (tile_to_slides == slide_id)
        slide_data['slide_id'].append(slide_id)
        if slide_id in patient_map:
            slide_data['patient_id'].append(patient_map[slide_id])
        else:
            raise ValueError(f"Slide ID {slide_id} not found in patient mapping for dataset {dataset_type}")
        slide_data['labels'].append(labels[mask][0])  # Use first label
        for i in range(num_classes):
            class_probs = probabilities[mask, i]
            slide_data[f'prob_{i}'].append(np.mean(class_probs))

        # Aggregate uncertainty if available
        if uncertainty is not None:
            slide_data['uncertainty'].append(np.mean(uncertainty[mask]))

    # Create slide-level DataFrame
    slide_df = pd.DataFrame(slide_data)
    agg_dict = {}
    for i in range(num_classes):
        agg_dict[f'prob_{i}'] = 'mean'

    agg_dict['labels'] = 'first'  # Use first label for patient

    if 'uncertainty' in slide_df:
        agg_dict['uncertainty'] = 'mean'

    patient_df = slide_df.groupby('patient_id').agg(agg_dict).reset_index()
    prob_cols = [f'prob_{i}' for i in range(num_classes)]

    slide_results = {
        'prob': np.array(slide_df[prob_cols]),
        'label': np.array(slide_df['labels']),
        'slide_id': np.array(slide_df['slide_id']),
        'patient_id': np.array(slide_df['patient_id'])
    }
    if 'uncertainty' in slide_df:
        slide_results['uncertainty'] = np.array(slide_df['uncertainty'])

    patient_results = {
        'prob': np.array(patient_df[prob_cols]),
        'label': np.array(patient_df['labels']),
        'patient_id': np.array(patient_df['patient_id'])
    }
    if 'uncertainty' in patient_df:
        patient_results['uncertainty'] = np.array(patient_df['uncertainty'])

    return {
        'tile': tile_results,
        'slide': slide_results,
        'patient': patient_results
    }

def calculate_metrics(labels, probabilities, num_classes=2):
    """Calculate performance metrics."""
    predictions = np.argmax(probabilities, axis=1)
    acc = accuracy_score(labels, predictions)
    bacc = balanced_accuracy_score(labels, predictions) if len(labels) > 1 else 0
    cls_rep = classification_report(labels, predictions, output_dict=True, zero_division=0)
    weighted_f1 = cls_rep["weighted avg"]["f1-score"]

    try:
        if num_classes == 2:
            auroc = roc_auc_score(labels, probabilities[:, 1])
        else:
            y_true_onehot = label_binarize(labels, classes=range(num_classes))
            auroc = roc_auc_score(y_true_onehot, probabilities, multi_class='ovr', average='macro')
    except Exception as e:
        print(f"Error calculating AUROC: {e}")
        auroc = 0

    return {
        'acc': acc,
        'bacc': bacc,
        'auroc': auroc,
        'weighted_f1': weighted_f1
    }


def evaluate_model(model, features, labels, tile_to_slides=None, dataset_type='tcga'):
    """Evaluate model performance and return metrics at tile, slide, and patient levels."""
    predictions, probs, uncertainty = model.predict(
        features, return_probs=True, return_uncertainty=True
    )

    num_classes = probs.shape[1]

    # Calculate tile-level metrics
    tile_metrics = calculate_metrics(labels, probs, num_classes)

    results = {
        'tile_metrics': tile_metrics,
        'predictions': predictions,
        'probabilities': probs,
        'uncertainty': uncertainty
    }

    # If tile_to_slides is provided, calculate slide and patient level metrics
    if tile_to_slides is not None:
        multi_level_results = aggregate_to_patient_level(
            predictions, probs, uncertainty, tile_to_slides, dataset_type, labels, num_classes
        )

        # Calculate slide-level metrics
        slide_metrics = calculate_metrics(
            multi_level_results['slide']['label'],
            multi_level_results['slide']['prob'],
            num_classes=num_classes
        )

        # Calculate patient-level metrics
        patient_metrics = calculate_metrics(
            multi_level_results['patient']['label'],
            multi_level_results['patient']['prob'],
            num_classes=num_classes
        )

        results.update({
            'slide_metrics': slide_metrics,
            'patient_metrics': patient_metrics,
            'multi_level_results': multi_level_results
        })

    return results


def train_fold(args, trial_idx, fold_idx):
    """Train and evaluate a model on a single fold."""
    cptac_prefix_name = "cptac_original_deterministic" if "images_r4_d" in args.data_path else "cptac_original_sngp"
    folder_path = os.path.join(args.data_path, f'trial_{trial_idx}_fold{fold_idx}')
    train_features, train_labels, train_tile_to_slides = load_data_from_h5(folder_path, 'train')
    val_features, val_labels, val_tile_to_slides = load_data_from_h5(folder_path, 'val')
    test_features, test_labels, test_tile_to_slides = load_data_from_h5(folder_path, 'test')
    cptac_test_features, cptac_test_labels, cptac_tile_to_slides = load_data_from_h5(folder_path, cptac_prefix_name)

    num_classes = len(np.unique(train_labels))
    input_dim = train_features.shape[1]

    print(f"Data loaded - Train: {train_features.shape}, Val: {val_features.shape}, Test: {test_features.shape}, CPTAC Test: {cptac_test_features.shape}")
    print(f"Number of classes: {num_classes}")

    model = VBLLMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        num_layers=args.num_layers,
        model_type=args.model_type,
        reg_weight=args.reg_weight,
        parameterization=args.parameterization,
        prior_scale=args.prior_scale
    )

    history = model.fit(
        train_features,
        train_labels,
        val_data=(val_features, val_labels),
        monitor='val_acc',
        save_best=True,
        balance_sampling=args.balance_sampling,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        verbose=1
    )

    if args.save_model:
        save_dir = os.path.join(args.save_destination, 'models')
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, f'vbll_model_trial{trial_idx}_fold{fold_idx}.pt'))

    # Evaluate at multiple levels
    val_results = evaluate_model(model, val_features, val_labels, val_tile_to_slides, 'tcga')
    test_results = evaluate_model(model, test_features, test_labels, test_tile_to_slides, 'tcga')
    cptac_results = evaluate_model(model, cptac_test_features, cptac_test_labels, cptac_tile_to_slides, 'cptac')

    # Save detailed predictions if requested
    if args.save_predictions:
        save_dir = os.path.join(args.save_destination, args.predictions_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Save test set predictions
        if 'multi_level_results' in test_results:
            save_predictions(test_results['multi_level_results'], num_classes,
                            f"test_trial{trial_idx}_fold{fold_idx}", save_dir)

        # Save CPTAC test set predictions
        if 'multi_level_results' in cptac_results:
            save_predictions(cptac_results['multi_level_results'], num_classes,
                            f"cptac_trial{trial_idx}_fold{fold_idx}", save_dir)

    # Combine metrics at all levels
    metrics = {
        # Validation metrics
        'val_tile_acc': val_results['tile_metrics']['acc'],
        'val_tile_bacc': val_results['tile_metrics']['bacc'],
        'val_tile_auroc': val_results['tile_metrics']['auroc'],
        'val_tile_weighted_f1': val_results['tile_metrics']['weighted_f1'],

        # Test metrics (tile level)
        'test_tile_acc': test_results['tile_metrics']['acc'],
        'test_tile_bacc': test_results['tile_metrics']['bacc'],
        'test_tile_auroc': test_results['tile_metrics']['auroc'],
        'test_tile_weighted_f1': test_results['tile_metrics']['weighted_f1'],

        # CPTAC metrics (tile level)
        'cptac_tile_acc': cptac_results['tile_metrics']['acc'],
        'cptac_tile_bacc': cptac_results['tile_metrics']['bacc'],
        'cptac_tile_auroc': cptac_results['tile_metrics']['auroc'],
        'cptac_tile_weighted_f1': cptac_results['tile_metrics']['weighted_f1'],
    }

    # Add slide-level metrics if available
    if 'slide_metrics' in test_results:
        metrics.update({
            'test_slide_acc': test_results['slide_metrics']['acc'],
            'test_slide_bacc': test_results['slide_metrics']['bacc'],
            'test_slide_auroc': test_results['slide_metrics']['auroc'],
            'test_slide_weighted_f1': test_results['slide_metrics']['weighted_f1'],
        })

    if 'slide_metrics' in cptac_results:
        metrics.update({
            'cptac_slide_acc': cptac_results['slide_metrics']['acc'],
            'cptac_slide_bacc': cptac_results['slide_metrics']['bacc'],
            'cptac_slide_auroc': cptac_results['slide_metrics']['auroc'],
            'cptac_slide_weighted_f1': cptac_results['slide_metrics']['weighted_f1'],
        })

    # Add patient-level metrics if available
    if 'patient_metrics' in test_results:
        metrics.update({
            'test_patient_acc': test_results['patient_metrics']['acc'],
            'test_patient_bacc': test_results['patient_metrics']['bacc'],
            'test_patient_auroc': test_results['patient_metrics']['auroc'],
            'test_patient_weighted_f1': test_results['patient_metrics']['weighted_f1'],
        })

    if 'patient_metrics' in cptac_results:
        metrics.update({
            'cptac_patient_acc': cptac_results['patient_metrics']['acc'],
            'cptac_patient_bacc': cptac_results['patient_metrics']['bacc'],
            'cptac_patient_auroc': cptac_results['patient_metrics']['auroc'],
            'cptac_patient_weighted_f1': cptac_results['patient_metrics']['weighted_f1'],
        })

    print(f"\nTrial {trial_idx} Fold {fold_idx} Results:")
    print(f"Val (tile): Acc={metrics['val_tile_acc']:.4f}, BAcc={metrics['val_tile_bacc']:.4f}, AUROC={metrics['val_tile_auroc']:.4f}")
    print(f"Test (tile): Acc={metrics['test_tile_acc']:.4f}, BAcc={metrics['test_tile_bacc']:.4f}, AUROC={metrics['test_tile_auroc']:.4f}")

    if 'test_patient_acc' in metrics:
        print(f"Test (patient): Acc={metrics['test_patient_acc']:.4f}, BAcc={metrics['test_patient_bacc']:.4f}, AUROC={metrics['test_patient_auroc']:.4f}")

    print(f"CPTAC (tile): Acc={metrics['cptac_tile_acc']:.4f}, BAcc={metrics['cptac_tile_bacc']:.4f}, AUROC={metrics['cptac_tile_auroc']:.4f}")

    if 'cptac_patient_acc' in metrics:
        print(f"CPTAC (patient): Acc={metrics['cptac_patient_acc']:.4f}, BAcc={metrics['cptac_patient_bacc']:.4f}, AUROC={metrics['cptac_patient_auroc']:.4f}")

    return metrics, model


def save_predictions(multi_level_results, num_classes, tag, save_dir):
    """Save predictions at different levels to parquet files."""
    # Save patient-level predictions
    patient_data = {}
    patient_results = multi_level_results['patient']

    for i in range(num_classes):
        patient_data[f'prob_{i}'] = patient_results['prob'][:, i]

    patient_data['true_label'] = patient_results['label']
    patient_data['patient_id'] = patient_results['patient_id']
    patient_data['pred_label'] = np.argmax(patient_results['prob'], axis=1)

    if 'uncertainty' in patient_results:
        patient_data['uncertainty'] = patient_results['uncertainty']

    patient_df = pd.DataFrame(patient_data)
    patient_df.to_parquet(os.path.join(save_dir, f"{tag}_patient_predictions.parquet.gzip"), compression="gzip")

    # Save slide-level predictions
    slide_data = {}
    slide_results = multi_level_results['slide']

    for i in range(num_classes):
        slide_data[f'prob_{i}'] = slide_results['prob'][:, i]

    slide_data['true_label'] = slide_results['label']
    slide_data['slide_id'] = slide_results['slide_id']
    slide_data['patient_id'] = slide_results['patient_id']
    slide_data['pred_label'] = np.argmax(slide_results['prob'], axis=1)

    if 'uncertainty' in slide_results:
        slide_data['uncertainty'] = slide_results['uncertainty']

    slide_df = pd.DataFrame(slide_data)
    slide_df.to_parquet(os.path.join(save_dir, f"{tag}_slide_predictions.parquet.gzip"), compression="gzip")


def main_spawn(args):
    os.makedirs(args.save_destination, exist_ok=True)

    all_metrics = {}
    start_time = time.time()

    for trial_idx in range(args.num_trials):
        for fold_idx in range(args.num_folds):
            print(f"\n===== Running Trial {trial_idx} Fold {fold_idx} =====")
            metrics, _ = train_fold(args, trial_idx, fold_idx)

            if not all_metrics:
                for key in metrics:
                    all_metrics[key] = []

            for key, value in metrics.items():
                all_metrics[key].append(value)

    df_metrics = pd.DataFrame(all_metrics)
    summary_metrics = df_metrics.agg(['mean', 'std']).transpose()
    summary_metrics.columns = ['mean', 'std']
    summary_metrics["metrics"] = summary_metrics.index
    summary_metrics["mean-std"] = (summary_metrics["mean"].round(4).astype(str) + "±" +
                                   summary_metrics["std"].round(4).astype(str))

    summary_metrics.drop(columns=["mean", "std"], inplace=True)
    tag = f"vbll_{args.model_type}_{args.hidden_dim}_{args.num_layers}"
    summary_metrics['tag'] = tag
    summary_metrics = summary_metrics.pivot(index='tag', columns='metrics', values='mean-std').reset_index()

    for metric in all_metrics.keys():
        summary_metrics[metric + "_list"] = [all_metrics[metric]]

    results_file_path = os.path.join(args.save_destination, args.results_file)
    if os.path.exists(results_file_path):
        existing_data = pd.read_csv(results_file_path)
        updated_data = pd.concat([existing_data, summary_metrics], ignore_index=True)
        updated_data.to_csv(results_file_path, index=False)
    else:
        summary_metrics.to_csv(results_file_path, index=False)

    # Print summary grouped by level
    print("\n===== Summary Results =====")
    print("\nTile-Level Metrics:")
    for key in sorted([k for k in all_metrics if 'tile' in k]):
        print(f"{key}: {np.mean(all_metrics[key]):.4f}±{np.std(all_metrics[key]):.4f}")

    slide_keys = [k for k in all_metrics if 'slide' in k]
    if slide_keys:
        print("\nSlide-Level Metrics:")
        for key in sorted(slide_keys):
            print(f"{key}: {np.mean(all_metrics[key]):.4f}±{np.std(all_metrics[key]):.4f}")

    patient_keys = [k for k in all_metrics if 'patient' in k]
    if patient_keys:
        print("\nPatient-Level Metrics:")
        for key in sorted(patient_keys):
            print(f"{key}: {np.mean(all_metrics[key]):.4f}±{np.std(all_metrics[key]):.4f}")

    print(f"\nTotal runtime: {(time.time() - start_time) / 60:.2f} minutes")
    print(f"Results saved to {results_file_path}")


if __name__ == "__main__":
    parser, args = get_vbll_params()
    print(args)
    seed_everything(args.seed)
    main_spawn(args)
