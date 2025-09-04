import os
import sys
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import h5py
import random
from collections import defaultdict
import argparse
# warnings.filterwarnings('error')
import operator
import wandb
from functools import partial
import copy
import yaml
from utils import TISSUE_SITE_TO_SUBTYPING, load_tcga_ot_titan_setting, convert_threshold, seed_everything
from utils import load_pickle_data, save_pickle_data, load_data_from_split, none_or_float
from utils import CLAM_BAG_WEIGHT, GP_KWARGS_CONCH, GP_KWARGS_UNI, create_model_path, get_dataset_type
from downstream_models.abmil import GatedABMIL
from downstream_models.clam import CLAM_SB
from downstream_models.transmil import TransMIL
from sklearn.preprocessing import label_binarize
from smooth_topk.topk.svm import SmoothTop1SVM
from sngp_wrapper_compatible.covert_utils import convert_to_sn_my, replace_layer_with_gaussian



def get_abmil_params():
    parser = argparse.ArgumentParser(description='MIL on downstream tasks')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--subtyping_task', type=str, default="Brain", help='Subtyping task')
    parser.add_argument('--model_name', type=str, default="conch", help='Model name')
    parser.add_argument('--mil_type', type=str, default="clam", choices=["clam", "abmil", "transmil"],)
    parser.add_argument('--spec_norm_bound', type=none_or_float, default=None,
                        help='Spectral norm bound, if set not None, will use spectral normalization')
    parser.add_argument('--spec_norm_replace_list', type=str, default='Linear,Conv2D',
                        help='List of layers to replace with spectral normalization')
    parser.add_argument('--gaussian_process', action='store_true',
                        help='If set True, will use Laplace Approximization of Gaussian process to estimate the uncertainty')
    parser.add_argument('--gp_kernel_type', type=str, default=None, help='Type of GP kernel')

    parser.add_argument('--num_runs', type=int, default=2, help='Number of runs')
    parser.add_argument('--save_to_parquet', action='store_true',
                        help='If set True, will save the results to parquet file')
    parser.add_argument('--save_destination', type=str, default="/home/user/sngp/TCGA-OT/models_ot",
                        help='Model and parquet save path')
    parser.add_argument('--mask_tile', action='store_true', help='whether to mask the tiles')
    parser.add_argument('--mask_tile_category', type=str, default="rand",
                        choices=["rand", "in_slide", "all_slide"], help='whether to mask the tiles')
    parser.add_argument('--ambiguity_strategy', type=str, default="entropy", help='ambiguity strategy',)
    parser.add_argument('--mask_tile_threshold', type=float, default=0.4, help='mask tile threshold')
    parser.add_argument('--invert_threshold', action='store_true', help='whether to invert the threshold')
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate the model')
    parser.add_argument('--save_probs', action='store_true', help='Save probabilities for train/val/test sets (only valid with --evaluate_only)')
    parser.add_argument('--balance_sampling', action='store_true', help='balance the sampling')
    parser.add_argument('--results_file_path', type=str, default="final_results.csv", help='final_results.csv')
    parser.add_argument('--proxy_model', type=str, default="autogluon", choices=["autogluon", "lightgbm"], help='proxy  model to use for eat')
    parser.add_argument('--proxy_training_proportion', type=float, default=0.4, help="Proxy model training proportion")
    parser.add_argument('--no_patient_aggregation', action='store_true',
                        help='If set, predictions will be evaluated at slide level instead of aggregating by patient')
    parser.add_argument('--uncertainty_type', type=str, default=None,
                        choices=[None, "total_uncertainty", "aleatoric_uncertainty", "epistemic_uncertainty"],
                        help='Type of uncertainty to use for ambiguity')
    args = parser.parse_args()
    args.spec_norm_replace_list = args.spec_norm_replace_list.split(',')
    if "bracs" in args.subtyping_task: # asssert only two kind of bracs input, bracs-coarse, bracs-finegrain
        assert args.subtyping_task in ["bracs-coarse", "bracs-finegrain"]
    if args.save_probs and not args.evaluate_only:
        parser.error("--save_probs can only be used with --evaluate_only")
    args.save_destination = Path(args.save_destination)
    assert args.results_file_path.endswith(
        ".csv"), f"results_file_path should be a csv file, get {args.results_file_path}"
    return parser, args

def amb_function(args):
    if not hasattr(args, 'uncertainty_type') or args.uncertainty_type is None:
        return f"/home/user/sngp/TCGA-OT/models/ambpkl/{args.model_name}_{args.subtyping_task}_ambiguity_dict_{args.proxy_model}_{args.proxy_training_proportion}_balanced.pkl"
    else:
        return f"/home/user/sngp/TCGA-OT/models/ambpkl_decouple/{args.model_name}_{args.subtyping_task}_ambiguity_{args.uncertainty_type}_split{args.split_idx}.pkl"
# def amb_function(args):
#     return f"/home/user/sngp/TCGA-OT/models/ambpkl/{args.model_name}_{args.subtyping_task}_ambiguity_dict_{args.proxy_model}_{args.proxy_training_proportion}_balanced.pkl"

class TITANTrainingConfig:
    model_name = "titan"
    dataset_h5 = Path("/home/user/sngp/TCGA-OT/Patch512/TITAN/pt_files")
    mask_func = amb_function

    embed_dim = 768
    batch_size = 1
    num_workers = 4
    num_classes = None
    epochs = 20

class UNITrainingConfig:
    model_name = "uni"
    dataset_h5 = Path("/home/user/sngp/TCGA-OT/Patch256/UNI/pt_files")
    mask_func = amb_function

    embed_dim = 1024
    batch_size = 1
    num_workers = 4
    num_classes = None
    epochs = 20

class CONCHTrainingConfig:
    model_name = "conch"
    dataset_h5 = Path("/home/user/sngp/TCGA-OT/Patch256/CONCH/pt_files")
    mask_func = amb_function

    embed_dim = 512
    batch_size = 1
    num_workers = 4
    num_classes = None
    epochs = 20

class ConvNextAttoTrainingConfig:
    model_name = "convnext_atto_d2_in1k"
    dataset_h5 = Path("/home/user/sngp/TCGA-OT/Patch256/convnext_atto_d2_in1k/pt_files")
    mask_func = amb_function

    embed_dim = 320
    batch_size = 1
    num_workers = 4 # 0: 8225it [04:31, 30.34it/s]
    num_classes = None
    epochs = 20


def load_training_config(model_name, subtyping_task=None):
    """
    Load appropriate training config based on model name and dataset type.

    Args:
        model_name: The feature extractor model (titan, uni, conch, convnext_atto_d2_in1k)
        subtyping_task: The dataset/task (TCGA-OT subtypes or "bracs" or "dhmc_luad")
    """
    # First, determine which dataset we're working with
    dataset_type = get_dataset_type(subtyping_task)

    # Create base configuration by model name
    if model_name == "titan":
        config = TITANTrainingConfig
    elif model_name == "uni":
        config = UNITrainingConfig
    elif model_name == "convnext_atto_d2_in1k":
        config = ConvNextAttoTrainingConfig
    elif model_name == "conch":
        config = CONCHTrainingConfig
    else:
        raise ValueError(f"Invalid model name {model_name}")

    # Now adjust the dataset_h5 path and other settings based on dataset type
    middle_prefix = "Patch512" if model_name == "titan" else "Patch256"
    if dataset_type == "bracs-coarse" or dataset_type == "bracs-finegrain": # BRACS dataset paths
        base_path = Path("/home/user/sngp/BRACS")
    elif dataset_type == "dhmc_luad": # DHMC-LUAD dataset paths
        base_path = Path("/home/user/sngp/DHMC/LUAD")
    elif dataset_type == "tcga_ot": # TCGA-OT paths (keep original structure)
        base_path = Path("/home/user/sngp/TCGA-OT")
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")
    config.dataset_h5 = base_path / middle_prefix / f"{model_name.upper()}/pt_files"

    # Create a copy of the config to avoid modifying the class
    config_instance = type('DynamicConfig', (), {
        'model_name': model_name,
        'dataset_h5': config.dataset_h5,
        'mask_func': config.mask_func,
        'embed_dim': config.embed_dim,
        'batch_size': config.batch_size,
        'num_workers': config.num_workers,
        'num_classes': None,  # Will be set dynamically
        'epochs': config.epochs
    })

    print(f"Using dataset path: {config_instance.dataset_h5}")
    return config_instance


def get_downstream_model(mil_type, embed_dim, n_classes):
    if mil_type == "clam": # set instance loss's classes to 2
        model = CLAM_SB(embed_dim=embed_dim, n_classes=n_classes, instance_loss_fn=SmoothTop1SVM(n_classes=2).cuda())
    elif mil_type == "abmil":
        model = GatedABMIL(embed_dim=embed_dim, n_classes=n_classes)
    elif mil_type == "transmil":
        model = TransMIL(embed_dim=embed_dim, n_classes=n_classes)
    else:
        raise ValueError(f"Invalid mil_type {mil_type}, should be one of clam, abmil, transmil")
    return model


class PtDataset(Dataset):

    def __init__(self, pt_path, filter_mapping, s_to_p_mapping=None,
                 mask_pkl=None, mask_tile_threshold=None, args=None):
        self.pt_path = pt_path
        self.args = args
        self.pt_paths = os.listdir(pt_path)
        self.pt_paths = [pt_path / path for path in self.pt_paths if path[:-3] in filter_mapping.keys()]
        self.pt_labels = [filter_mapping[pt_path.name[:-3]] for pt_path in self.pt_paths]
        self.s_to_p_mapping = s_to_p_mapping
        if mask_pkl is None:
            self.mask_pkl = None
        else:
            self.mask_pkl = self._ambiguity_strategy(strategy=args.ambiguity_strategy, mask_pkl=copy.deepcopy(mask_pkl))
            self.mask_threshold = mask_tile_threshold
            self.mask_tile_category = args.mask_tile_category
            self.comp_func = operator.gt if args.invert_threshold else operator.lt
            self.invert_threshold = args.invert_threshold

    def __len__(self):
        return len(self.pt_paths)

    def _ambiguity_strategy(self, strategy: str, mask_pkl: dict):
        # Check if we're using a pre-computed uncertainty metric
        if hasattr(self.args, 'uncertainty_type') and self.args.uncertainty_type is not None:
            # Just validate that the uncertainty data is available
            for slide_id in mask_pkl.keys():
                if slide_id in ["train_quantile_list", "val_quantile_list"]:
                    continue
                if "ambiguity" not in mask_pkl[slide_id]:
                    # Look for the uncertainty field in the mask_pkl
                    if self.args.uncertainty_type in mask_pkl[slide_id]:
                        mask_pkl[slide_id]["ambiguity"] = mask_pkl[slide_id][self.args.uncertainty_type]
                    else:
                        raise ValueError(f"Uncertainty type '{self.args.uncertainty_type}' not found in mask_pkl for slide {slide_id}")
            return mask_pkl

        # Original ambiguity computation strategies
        if strategy == "max_prob":
            for slide_id in mask_pkl.keys():
                if slide_id in ["train_quantile_list", "val_quantile_list"]:
                    continue
                mask_pkl[slide_id]["ambiguity"] = 1 - np.max(mask_pkl[slide_id]["probabilities"], axis=1)
            return mask_pkl
        elif strategy == "entropy":
            for slide_id in mask_pkl.keys():
                if slide_id in ["train_quantile_list", "val_quantile_list"]:
                    continue
                epsilon = 1e-7  # Small value to prevent log(0)
                probabilities = np.clip(mask_pkl[slide_id]["probabilities"], epsilon, 1.0).astype(np.float32)
                mask_pkl[slide_id]["ambiguity"] = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
            return mask_pkl
        else:
            raise ValueError(f"Invalid ambiguity strategy {strategy}")

    def __getitem__(self, idx):
        assets = self.read_assets_from_pt(self.pt_paths[idx], self.mask_pkl)
        # print("after", assets["features"].shape)
        assets["labels"] = self.pt_labels[idx]
        if self.s_to_p_mapping is None:
            assets["patient"] = self.pt_paths[idx].name[:-3]
        else:
            assets["patient"] = self.s_to_p_mapping[self.pt_paths[idx].name[:-3]]
        return assets

    def read_assets_from_pt(self, pt_path: str, mask_pkl=None) -> tuple:
        '''Read the assets from the pt file'''
        features = torch.load(pt_path)
        if isinstance(features, dict):
            features = features["features"]

        if mask_pkl is not None:
            if self.mask_tile_category == "rand":
                mask_bool = self.comp_func(np.random.rand(features.shape[0]), self.mask_threshold)
            elif self.mask_tile_category == "in_slide":
                in_slide_threshold = np.quantile(mask_pkl[pt_path.stem]["ambiguity"], self.mask_threshold)
                mask_bool = self.comp_func(mask_pkl[pt_path.stem]["ambiguity"], in_slide_threshold)
            else:
                mask_bool = self.comp_func(mask_pkl[pt_path.stem]["ambiguity"], self.mask_threshold)
            assets = {"features": features[mask_bool]}
        else:
            assets = {"features": features}
        return assets


def create_balanced_sampler(dataset):
    """Create a weighted sampler to balance classes in each batch"""
    labels = [dataset[i]['labels'] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    return sampler

def train(config, args):
    args.to_destination = create_model_path(config, args)
    args.to_destination.mkdir(parents=True, exist_ok=True)
    with open(args.to_destination.parent / "args.yaml", "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    # create a dictionary mapping slide names to cohort
    train_df, val_df, test_df, label_dict, s_to_p_mapping = load_data_from_split(args, args.split_idx)
    config.num_classes = len(np.unique(list(label_dict.values())))
    train_df_mapping = train_df.set_index("slide_id")["cohort"].to_dict()
    val_df_mapping = val_df.set_index("slide_id")["cohort"].to_dict()
    test_df_mapping = test_df.set_index("slide_id")["cohort"].to_dict()
    print("Length test patient", test_df["case_id"].nunique(), "Length test slide", len(test_df_mapping))
    if args.mask_tile:
        test_mask_pkl = load_pickle_data(config.mask_func(args))[f"split{args.split_idx}"]
        if args.mask_tile_category == "rand" or args.mask_tile_category == "in_slide":  # float
            mask_tile_threshold = args.mask_tile_threshold
            val_mask_tile_threshold = args.mask_tile_threshold
        else:
            raise NotImplementedError(f"mask_tile_category {args.mask_tile_category} not implemented")
        print("Removing Train tiles with quantile larger than", mask_tile_threshold)
        print("Removing Test tiles with quantile larger than", val_mask_tile_threshold)
    else:
        test_mask_pkl, mask_tile_threshold, val_mask_tile_threshold = None, None, None


    train_dataset, val_dataset, test_dataset = (PtDataset(config.dataset_h5, train_df_mapping,
                                                                         s_to_p_mapping=s_to_p_mapping,
                                                                         mask_pkl=test_mask_pkl,
                                                                         mask_tile_threshold=mask_tile_threshold,
                                                                         args=args),
                                                               PtDataset(config.dataset_h5, val_df_mapping,
                                                                         s_to_p_mapping=s_to_p_mapping,
                                                                         mask_pkl=test_mask_pkl,
                                                                         mask_tile_threshold=val_mask_tile_threshold,
                                                                         args=args),
                                                               PtDataset(config.dataset_h5, test_df_mapping,
                                                                         s_to_p_mapping=s_to_p_mapping,
                                                                         mask_pkl=test_mask_pkl,
                                                                         mask_tile_threshold=val_mask_tile_threshold,
                                                                         args=args))
    print(f"train dataset size: {len(train_dataset)}, val dataset size: {len(val_dataset)}, "
          f"test dataset size: {len(test_dataset)}")

    def collate_fn(batch):
        return {
            'features': torch.stack([x['features'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch]),
            "patient": np.array([x['patient'] for x in batch])
        }
    train_sampler = create_balanced_sampler(train_dataset) if args.balance_sampling else None

    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                                                     num_workers=config.num_workers,
                                                                     collate_fn=collate_fn,
                                                       sampler=train_sampler,
                                                       pin_memory=True, shuffle=True if not args.evaluate_only else False), \
        DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn,
                   pin_memory=True, shuffle=False), \
        DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn,
                   pin_memory=True, shuffle=False)
    model = get_downstream_model(args.mil_type, config.embed_dim, config.num_classes).cuda()
    if args.spec_norm_bound is not None:
        model = convert_to_sn_my(model, args.spec_norm_replace_list, args.spec_norm_bound)
    if args.gaussian_process:
        GP_KWARGS = GP_KWARGS_CONCH if args.model_name == "conch" else GP_KWARGS_UNI
        if args.gp_kernel_type is not None:
            GP_KWARGS['gp_kernel_type'] = args.gp_kernel_type
        print(GP_KWARGS)
        replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    print("parameter", sum(p.numel() for p in model.parameters()))
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    total_iter = len(train_loader) * config.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_iter, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy, best_val_loss = 0.0, float('inf')
    if args.evaluate_only:
        print("args.to_destination", args.to_destination)
        model.load_state_dict(torch.load(args.to_destination / "best_model.pth"))
        val_acc, val_bacc, val_auroc, val_loss, val_weighted_f1 = evaluate(model, val_loader, args, "val", label_dict=label_dict)
        print(
            f"Val Accuracy: {val_acc:.4f}, Val Balanced Accuracy: {val_bacc:.4f}, Val AUROC: {val_auroc:.4f}, Val Loss: {val_loss:.4f}")
        test_acc, test_bacc, test_auroc, test_loss, test_weighted_f1 = evaluate(model, test_loader, args, "test", label_dict=label_dict)
        print(f"Test Accuracy: {test_acc:.4f}, Test Balanced Accuracy: {test_bacc:.4f}, Test AUROC: {test_auroc:.4f}, Test Loss: {test_loss:.4f}")
        # Evaluate train and val sets if save_probs is enabled
        if args.save_probs:
            train_acc, train_bacc, train_auroc, train_loss, train_weighted_f1 = evaluate(model, train_loader, args, "train", label_dict=label_dict)
            print(f"Train Accuracy: {train_acc:.4f}, Train Balanced Accuracy: {train_bacc:.4f}, Train AUROC: {train_auroc:.4f}, Train Loss: {train_loss:.4f}")
        return val_acc, val_bacc, val_auroc, val_weighted_f1, test_acc, test_bacc, test_auroc, test_weighted_f1
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        if args.gaussian_process:
            # GP_KWARGS["gp_cov_discount_factor"] == -1, in fact, it is not necessary when momentum != -1
            model.classifier.reset_covariance_matrix()
            kwargs = {'return_random_features': False, 'return_covariance': False,
                      'update_precision_matrix': True, 'update_covariance_matrix': False}
        else:
            kwargs = {}
        for idx, assets in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            H = assets['features'].float().cuda(non_blocking=True)
            labels = assets['labels'].long().cuda(non_blocking=True)
            if args.mil_type == "clam":
                preds, inst_loss = model(H, label=labels, instance_eval=True, **kwargs)
                loss = criterion(preds, labels)
                loss = loss * CLAM_BAG_WEIGHT + inst_loss * (1 - CLAM_BAG_WEIGHT)
            else:
                preds = model(H, **kwargs)
                loss = criterion(preds, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {total_loss / len(train_loader)}")
        val_accuracy, val_bacc, val_auroc, val_loss, val_weighted_f1 = evaluate(model, val_loader, args)
        print(f"Validation Accuracy after Epoch {epoch + 1}: {val_accuracy:.4f}, Validation AUROC: {val_auroc:.4f}, Validation Loss: {val_loss:.4f}")
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss < best_val_loss):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            os.makedirs(args.to_destination, exist_ok=True)
            torch.save(model.state_dict(), args.to_destination / "best_model.pth")

    model.load_state_dict(torch.load(args.to_destination / f"best_model.pth"))
    val_acc, val_bacc, val_auroc, val_loss, val_weighted_f1 = evaluate(model, val_loader, args, "val")
    test_acc, test_bacc, test_auroc, test_loss, test_weighted_f1= evaluate(model, test_loader, args, "test")
    print(
        f"Val Accuracy: {val_acc:.4f}, Val Balanced Accuracy: {val_bacc:.4f}, Val AUROC: {val_auroc:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Test Balanced Accuracy: {test_bacc:.4f}, Test AUROC: {test_auroc:.4f}, Test Loss: {test_loss:.4f}")
    return val_acc, val_bacc, val_auroc, val_weighted_f1, test_acc, test_bacc, test_auroc, test_weighted_f1


def evaluate(model, loader, args, tag=None, label_dict=None):
    model.eval()
    labels = []
    patient_id = []
    logits_list = []
    uncertainty_list = []
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gaussian_process:
        model.classifier.update_covariance_matrix()
        eval_kwargs = {'return_random_features': False, 'return_covariance': True,
                       'update_precision_matrix': False, 'update_covariance_matrix': False}
    else:
        eval_kwargs = {}
    with torch.inference_mode():
        for idx, assets in tqdm(enumerate(loader)):
            H = assets['features'].float().cuda()
            output = model(H, **eval_kwargs)
            if isinstance(output, tuple):
                logits, covariance = output
                val_loss += criterion(output[0], assets['labels'].long().cuda()).item()
                logits = logits.cpu().detach().numpy()
                uncertainty = torch.diagonal(covariance).cpu().detach().numpy()
                uncertainty_list.extend(uncertainty)
            else:
                val_loss += criterion(output, assets['labels'].long().cuda()).item()
                logits = output.cpu().detach().numpy()
            logits_list.extend(logits)
            labels.extend(assets['labels'].long())
            patient_id.extend(assets['patient'])

    logits = np.stack(logits_list, axis=0)
    labels = np.stack(labels, axis=0)
    val_loss = val_loss / len(loader)
    num_classes = logits.shape[1]

    # Build dataframe with dynamic logit columns.
    data_dict = {'patient_id': np.array(patient_id), 'labels': labels}
    for i in range(num_classes):
        data_dict[f'logit_{i}'] = logits[:, i]

    df = pd.DataFrame(data_dict)

    if uncertainty_list:
        df['uncertainty'] = uncertainty_list
    # Skip patient-level aggregation if no_patient_aggregation is True
    if hasattr(args, 'no_patient_aggregation') and args.no_patient_aggregation:
        agg_df = df  # Use slide-level data directly
        print(f"Using slide-level evaluation without patient aggregation: {len(agg_df)} slides")
    else:
        # Aggregate by patient
        agg_dict = {f'logit_{i}': 'mean' for i in range(num_classes)}
        agg_dict['labels'] = 'first'
        if uncertainty_list:
            agg_dict['uncertainty'] = 'mean'

        agg_df = df.groupby('patient_id').agg(agg_dict).reset_index()
        print(f"Aggregated from {len(df)} slides to {len(agg_df)} patients")

    logit_cols = [f'logit_{i}' for i in range(num_classes)]
    results = {
        'logit': np.array(agg_df[logit_cols]),
        'label': np.array(agg_df['labels']),
        'prob': softmax(np.array(agg_df[logit_cols]), axis=1),
        'patient': np.array(agg_df['patient_id'])
    }
    if uncertainty_list:
        results['uncertainty'] = np.array(agg_df['uncertainty'])

    print(f"agg from {len(df)} to {len(agg_df)}")
    unique_classes = num_classes
    targets_all = results['label']
    preds_all = results['prob'].argmax(axis=1)
    acc = accuracy_score(targets_all, preds_all)
    bacc = balanced_accuracy_score(targets_all, preds_all) if len(targets_all) > 1 else 0
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)
    weighted_f1 = cls_rep["weighted avg"]["f1-score"]

    try:
        if num_classes == 2:
            auroc = roc_auc_score(results['label'], results['prob'][:, 1])
        else:
            y_true_onehot = label_binarize(results['label'], classes=range(num_classes))
            auroc = roc_auc_score(y_true_onehot, results['prob'], multi_class='ovr', average='macro')
    except Exception as e:
        print(f"Error: {e}")
        auroc = 0

    if args.save_to_parquet and tag is not None:
        save_data = {}
        for i in range(num_classes):
            save_data[f'Outcome {i}-y_pred'] = results['logit'][:, i]
            save_data[f'prob_{i}'] = results['prob'][:, i]
        save_data['Outcome y_true'] = results['label']
        save_data['patient'] = results['patient']
        if uncertainty_list:
            save_data['Outcome uncertainty'] = results['uncertainty']
        save_df = pd.DataFrame(save_data)
        if args.mask_tile:
            save_df_name = f"patient_predictions_{tag}_split{args.split_idx}_mask{args.mask_tile_category}_thres{args.mask_tile_threshold}_invert{int(args.invert_threshold)}.parquet.gzip"
        else:
            save_df_name = f"patient_predictions_{tag}_split{args.split_idx}.parquet.gzip"
        if args.uncertainty_type:
            save_df_name = save_df_name.replace(".parquet.gzip", f"_{args.uncertainty_type}_{args.ambiguity_strategy}.parquet.gzip")
        elif args.ambiguity_strategy == "max_prob":
            save_df_name = save_df_name.replace(".parquet.gzip", f"_maxprob.parquet.gzip")
        save_df.to_parquet(args.to_destination / save_df_name, compression="gzip")
    if label_dict is not None:
        invert_label_dict = {v:k for k, v in label_dict.items()}
        per_class_correct = np.zeros(num_classes)
        per_class_total = np.zeros(num_classes)
        predictions = results['prob'].argmax(axis=1)

        for pred, label in zip(predictions, results['label']):
            per_class_correct[label] += (pred == label)
            per_class_total[label] += 1

        per_class_acc = per_class_correct / np.maximum(per_class_total, 1)

        print("\nPer-class accuracies:")
        for i in range(num_classes):
            if per_class_total[i] > 0:
                print(f"Class {invert_label_dict[i]}: {per_class_acc[i]:.3f} ({int(per_class_correct[i])}/{int(per_class_total[i])})")
        print(per_class_acc)

    return acc, bacc, auroc, val_loss, weighted_f1

def main_spawn(args):
    config = load_training_config(model_name=args.model_name, subtyping_task=args.subtyping_task)
    start_time = time.time()
    print(f"Running {args.num_runs} splits for {config.model_name} on {args.subtyping_task}")
    metric_dict = defaultdict(list)
    for split_idx in range(args.num_runs):
        print(f"\n===== Running Split {split_idx}/{args.num_runs - 1} =====")
        args.split_idx = split_idx
        val_acc, val_bacc, val_auroc, val_weighted_f1, test_acc, test_bacc, test_auroc, test_weighted_f1 = train(config, args)
        metric_dict['val_acc'].append(val_acc)
        metric_dict['val_bacc'].append(val_bacc)
        metric_dict['val_auroc'].append(val_auroc)
        metric_dict['val_weighted_f1'].append(val_weighted_f1)
        metric_dict['test_acc'].append(test_acc)
        metric_dict['test_bacc'].append(test_bacc)
        metric_dict['test_auroc'].append(test_auroc)
        metric_dict['test_weighted_f1'].append(test_weighted_f1)

        print(f"Split {split_idx} results - Val Accuracy: {val_acc:.4f}, Val Balanced Accuracy: {val_bacc:.4f}, Val AUROC: {val_auroc:.4f}")
        print(f"Split {split_idx} results - Test Accuracy: {test_acc:.4f}, Test Balanced Accuracy: {test_bacc:.4f}, Test AUROC: {test_auroc:.4f}")

    df_metrics = pd.DataFrame(metric_dict)
    summary_metrics = df_metrics.agg(['mean', 'std']).transpose()
    summary_metrics.columns = ['mean', 'std']
    summary_metrics["metrics"] = summary_metrics.index  # Use actual metric names from index
    summary_metrics["mean-std"] = (summary_metrics["mean"].round(4).astype(str) + "+-" +
                                   summary_metrics["std"].round(4).astype(str))
    summary_metrics.drop(columns=["mean", "std"], inplace=True)
    summary_metrics['tag'] = args.to_destination.parent.name
    summary_metrics = summary_metrics.pivot(index='tag', columns='metrics', values='mean-std').reset_index()
    for metric in metric_dict.keys():
        summary_metrics[metric + "_list"] = [metric_dict[metric]]
    if args.mask_tile is False:
        mask_ratio = "no_mask"
    else:
        mask_ratio = f"mask_category{args.mask_tile_category}_thres{args.mask_tile_threshold}"
    summary_metrics["mask_ratio"] = mask_ratio
    print(summary_metrics)

    results_file_path = args.save_destination / args.results_file_path

    if results_file_path.exists():
        existing_data = pd.read_csv(results_file_path)
        updated_data = pd.concat([existing_data, summary_metrics], ignore_index=True)
        updated_data.to_csv(results_file_path, index=False)
    else:
        summary_metrics.to_csv(results_file_path, index=False)
    for key in metric_dict:
        print(f"{key}: {np.mean(metric_dict[key])}+-{np.std(metric_dict[key])}, runs {len(metric_dict[key])}")
    torch.cuda.empty_cache()
    print("Duration: (s)", time.time() - start_time, "minutes: (m)", (time.time() - start_time) / 60)





if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    _, args = get_abmil_params()
    print(args)
    seed_everything(args.seed)
    main_spawn(args)



