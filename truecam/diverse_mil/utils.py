import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import argparse
import random
import pickle
import torch
from collections import defaultdict

CLAM_BAG_WEIGHT = 0.7

## Important note: the Bowel subtyping seems to be a bit difficult, so that following previous work, we try COADREAD typing


TISSUE_SITE_TO_SUBTYPING = {
    "Adrenal gland": ["ACC", "PHC"],
    'COADREAD': ["COAD", "READ"],
    "Bowel": ["COAD", "READ", "MACR"],
    'GBMODGAASTR': ["GBM", "ODG", "AASTR"], # cancer vs cancer: OASTAOASTASTR
    'OASTAOASTASTR': ["OAST", "AOAST", "ASTR"], # cancer vs cancer: GBMODGAASTR
    "Brain": ["GBM", "OAST", "ODG", "AASTR", "AOAST", "ASTR"],
    "BRCA": ["IDC", "ILC"],
    "RCC": ["CCRCC", "PRCC", "CHRCC"],
    "NSCLC": ["LUAD", "LUSC"],
    "Soft tissue": ["MFH", "LMS", "MFS", "DDLS", "THYM"],
    "Stomach": ["STAD", "ESCC", "TSTAD", "DSTAD", "ESCA"],
    "Testis": ["NSGCT", "SEM"],
    "Thyroid": ["THPA", "THFO"],
    "Uterus": ["UEC", "USC", "UCS"]
}

GP_KWARGS_CONCH = {
    'num_inducing': 512,
    'gp_scale': 1.0,
    'gp_bias': 0.,
    'gp_kernel_type': 'linear',
    'gp_input_normalization': False,
    'gp_cov_discount_factor': -1,
    'gp_cov_ridge_penalty': 1.,
    'gp_output_bias_trainable': True,
    'gp_scale_random_features': False,
    'gp_use_custom_random_features': True,
    'gp_random_feature_type': 'orf',
    'gp_output_imagenet_initializer': False,
}


GP_KWARGS_UNI = {
    'num_inducing': 1024,
    'gp_scale': 1.0,
    'gp_bias': 0.,
    'gp_kernel_type': 'gaussian',
    'gp_input_normalization': True,
    'gp_cov_discount_factor': -1,
    'gp_cov_ridge_penalty': 1.,
    'gp_output_bias_trainable': False,
    'gp_scale_random_features': False,
    'gp_use_custom_random_features': True,
    'gp_random_feature_type': 'orf',
    'gp_output_imagenet_initializer': False,
}


def load_pickle_data(path: str = "kmeans_100w.pkl"):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model


def save_pickle_data(data, path: str = "kmeans_100w.pkl"):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def read_assets_from_h5(h5_path: str) -> tuple:
    '''Read the assets from the h5 file'''
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            assets[key] = f[key][:]
            if f[key].attrs is not None:
                attrs[key] = dict(f[key].attrs)
    return assets

def read_assets_from_pt(pt_path: str) -> tuple:
    pt_data = torch.load(pt_path)
    if isinstance(pt_data, dict):
        pt_data = pt_data["features"]
    return pt_data.half()


def read_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':2f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"



def none_or_float(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float or 'None'")


def convert_threshold(value):
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid threshold value: {value}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_strictly_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def safe_list_to(data_list, device):
    """Move a list of tensors/data to the specified device"""
    return [item.to(device) if isinstance(item, torch.Tensor) else item for item in data_list]


def _read_gene(omics_dir):
    df = pd.read_csv(omics_dir / "rna_clean.csv", engine='python', index_col=0)
    assert 'Unnamed: 0' not in df.columns, "Found 'Unnamed: 0' column in RNA data, potential index issue."
    print(df.columns)
    df = df.reset_index()
    id_col = "sample" # f"Assuming id_col is the case/sample ID column in RNA data."
    df = df.rename(columns={id_col: 'case_id'})
    return df


def load_pickle_data(path: str = "kmeans_100w.pkl"):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def save_pickle_data(data, path: str = "kmeans_100w.pkl"):
    with open(path, "wb") as file:
        pickle.dump(data, file)

def read_survival_splits(csv_splits_fold, omics_dir, split_names=["train", "test"]):
    csv_splits_processed = {}
    for key in split_names:
        histo_df = csv_splits_fold[key]
        gene_df = _read_gene(omics_dir)
        csv_splits_processed[key] = {"histo": histo_df, "gene": gene_df}
        print(f"  Split '{key}': Histo={len(histo_df)}, Gene={len(gene_df)}")
    return csv_splits_processed


def get_dataset_type(subtyping_task):
    """Determine the dataset type based on subtyping_task"""
    tcga_ot_list = ["tcga_ot", "NSCLCv0"]
    if subtyping_task in ["bracs-coarse", "bracs-finegrain", "dhmc_luad", "cptac_nsclc"]:
        return subtyping_task
    elif subtyping_task in TISSUE_SITE_TO_SUBTYPING or subtyping_task  in tcga_ot_list:
        return "tcga_ot"
    else:
        raise ValueError(f"Unknown subtyping_task: {subtyping_task}")


def load_data_from_multi_split(args, split_idx):
    """Load data from a specific split file"""
    split_dir = Path("/home/user/wangtao/prov-gigapath/revision_may19/revision_utils/data_splits")
    prefix_path = Path("/home/user/wangtao/prov-gigapath/TITAN/datasets")
    dataset_type = get_dataset_type(args.subtyping_task)

    if dataset_type == "tcga_ot": # Original TCGA-OT logic
        task_config = read_yaml(prefix_path / 'config_tcga-ot.yaml')
        target = task_config['target']
        if args.subtyping_task is None:
            prefix = f"split_all_{split_idx}"
        else:
            prefix = f"split_{args.subtyping_task}_{split_idx}"

        split_path = split_dir / prefix

    elif dataset_type == "bracs-coarse" or dataset_type == "bracs-finegrain": # BRACS dataset logic
        target = "cohort"  # Use cohort as the target column
        prefix = f"bracs/split_all_{split_idx}"
        split_path = split_dir / prefix

    elif dataset_type == "dhmc_luad": # DHMC_LUAD dataset logic
        target = "cohort"  # Use cohort as the target column
        prefix = f"dhmc_luad/split_all_{split_idx}"
        split_path = split_dir / prefix

    if not split_path.exists():
        raise ValueError(f"Split path not found: {split_path}")

    train_df = pd.read_csv(split_path / "train.csv")
    val_df = pd.read_csv(split_path / "val.csv")
    test_df = pd.read_csv(split_path / "test.csv")
    concat_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    slide_to_patient = dict(zip(concat_df["slide_id"], concat_df["case_id"]))
    all_targets = set(concat_df[target].unique())
    label_dict = {label: idx for idx, label in enumerate(sorted(all_targets))}
    if dataset_type == "bracs-coarse": # Coarse-grained BRACS subtyping
        label_dict = {"N": 0, "PB": 0, "UDH": 0, "FEA": 1, "ADH": 1, "DCIS": 2, "IC": 2}
    for df in [train_df, val_df, test_df]:
        df["cohort"] = df[target].map(label_dict)
    return train_df, val_df, test_df, label_dict, slide_to_patient


def create_model_path(config, args):
    """Create a clean hierarchical path structure for model saving."""
    components = [config.model_name, args.subtyping_task, args.mil_type]
    if args.balance_sampling:
        components.append("balanced")
    if args.spec_norm_bound is not None:
        components.append(f"sn{args.spec_norm_bound}")
    if args.gaussian_process:
        components.append("gp")
        if args.gp_kernel_type is not None:
            components.append(f"kernel_{args.gp_kernel_type}")

    model_id = "_".join(components)
    model_path = args.save_destination / model_id / f"run{args.split_idx}"
    return model_path

def load_tcga_ot_titan_setting():
    prefix_path = Path("/home/user/wangtao/prov-gigapath/TITAN/datasets")
    task_config = read_yaml(prefix_path / 'config_tcga-ot.yaml')
    target = task_config['target']
    label_dict = task_config['label_dict']
    train_df = pd.read_csv(prefix_path / 'tcga-ot_train.csv')
    val_df = pd.read_csv(prefix_path / 'tcga-ot_val.csv')
    test_df = pd.read_csv(prefix_path / 'tcga-ot_test.csv')
    concat_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    slide_to_patient = dict(zip(concat_df["slide_id"], concat_df["case_id"]))

    # if subtyping_task is not None:
    #     assert subtyping_task in TISSUE_SITE_TO_SUBTYPING, f"subtyping_task {subtyping_task} not in TISSUE_SITE_TO_SUBTYPING"
    #     # filter the df with only the involved subtypes
    #     subtyping_list = TISSUE_SITE_TO_SUBTYPING[subtyping_task]
    #     print(f"subtyping_list: {subtyping_list}")
    #
    #     train_df = train_df[train_df[target].isin(subtyping_list)]
    #     val_df = val_df[val_df[target].isin(subtyping_list)]
    #     test_df = test_df[test_df[target].isin(subtyping_list)]
    #     label_dict = {k: v for k, v in zip(subtyping_list, range(len(subtyping_list)))}

    print(f"len(train_df): {len(train_df)}, len(val_df): {len(val_df)}, len(test_df): {len(test_df)}")

    for df in [train_df, val_df, test_df]:
        df["cohort"] = df[target].map(label_dict)
    return train_df, val_df, test_df, label_dict, slide_to_patient

def load_data_from_split(args, split_idx):
    print("subtyping_task", args.subtyping_task)
    if args.subtyping_task == "tcga_ot":
        return load_tcga_ot_titan_setting()
    else:
        return load_data_from_multi_split(args, split_idx)


def load_data_from_cptac(args):
    assert args.subtyping_task == "cptac_nsclc", "load_data_from_cptac should only be called for cptac_nsclc subtyping task"
    csv_path = "/home/user/sngp/CPTAC/cptac.csv"
    df = pd.read_csv(csv_path) #         patient         slide cohort
    df = df.rename(columns={"patient": "patient_id", "slide": "slide_id"})
    label_dict = df['cohort'].unique()
    label_dict = {label: idx for idx, label in enumerate(sorted(label_dict))}
    df["cohort"] = df["cohort"].map(label_dict)
    slide_to_patient = dict(zip(df["slide_id"], df["patient_id"]))

    return df, label_dict, slide_to_patient

def null_output_metric():
    return {
        'val_acc': 0.,
        'val_bacc': 0,
        'val_auroc': 0,
        'val_weighted_f1': 0,
        'test_acc': 0,
        'test_bacc': 0,
        'test_auroc': 0,
        'test_weighted_f1': 0
    }

def evaluate_ood_datasets(model, args, config, ood_dataset_name=None):
    """
    Evaluate model on out-of-distribution datasets and save predictions and features.

    Args:
        model: The trained model
        args: Command-line arguments
        config: Model configuration
        label_dict: Dictionary mapping class names to indices
    """
    # Import required modules from the referenced file
    from revision_may19.train_tile_models import (
        WSISamplingDataset, get_tile_model_augmentation, DataLoader
    )

    # Path to OOD H5 files - update these paths as needed
    ood_dataset_paths = {
        "gtex": Path("/home/user/Patchfy-Data/GTEX-PATCH-512-IMGS/h5_files"),
        "OASTAOASTASTR": Path("/home/user/Patchfy-Data/TCGA-OT-PATCH-512-IMGS/h5_files/"),
    }

    # OOD labels mapping (can be arbitrary since we don't use the labels)
    # Each slide ID maps to a dummy class (0)
    _, eval_transform = get_tile_model_augmentation(config.model_name, config.pixel_size)

    print(f"Evaluating model on {ood_dataset_name} OOD datasets")

    # Create model's target directory if it doesn't exist
    args.to_destination.mkdir(parents=True, exist_ok=True)

    print(f"\nEvaluating on OOD dataset: {ood_dataset_name}")
    h5_path = ood_dataset_paths[ood_dataset_name]
    h5_files = [f.stem for f in h5_path.glob("*.h5")]
    if ood_dataset_name == "gtex":
        ood_mapping = {slide_id: 0 for slide_id in h5_files}
        s_to_p_mapping = None
    elif ood_dataset_name == "OASTAOASTASTR":
        bk_subtyping_task = args.subtyping_task
        args.subtyping_task = ood_dataset_name
        train_df, val_df, test_df, _, s_to_p_mapping = load_data_from_split(args, split_idx=0)
        concat_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        ood_mapping = concat_df.set_index("slide_id")["cohort"].to_dict()
        args.subtyping_task = bk_subtyping_task  # Restore original subtyping task

    # Create dataset and dataloader
    ood_dataset = WSISamplingDataset(
        h5_path=h5_path,
        filter_mapping=ood_mapping,
        transform=eval_transform,
        s_to_p_mapping=s_to_p_mapping,
    )

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=config.batch_size * 2,  # Can use larger batch for inference
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
        drop_last=False
    )

    # Run evaluation
    from revision_may19.train_tile_models import evaluate
    print(f"Running evaluation on {ood_dataset_name} with {len(ood_dataset)} samples")
    evaluate(
        model=model,
        data_loader=ood_loader,
        args=args,
        tag=ood_dataset_name,
        label_dict=None,
        extract_features=True  # Always extract features for OOD
    )

    print(f"Completed evaluation on {ood_dataset_name}")
    print(
        f"Saved predictions to: {args.to_destination}/tile_predictions_{ood_dataset_name}_split{args.split_idx}.parquet.gzip")
    print(f"Saved RFF features to: {args.to_destination}/rff_features_{ood_dataset_name}_split{args.split_idx}.h5")
    return null_output_metric()


if __name__ == "__main__":
    class args:
        subtyping_task = "tcga_ot"

    for split_idx in range(5):
        print("############################# Loading data for split index:", split_idx, "#############################")
        train_df, val_df, test_df, label_dict, slide_to_patient = load_data_from_split(args, split_idx=split_idx)
        concat_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        print(f"WSIs {len(concat_df['slide_id'].unique())}, Patients {len(concat_df['case_id'].unique())}, Labels {len(label_dict)}")
        # print(test_df.groupby("case_id")["cohort"].first().value_counts())
        # each df has a column "cohort" which is the slide-level label index, and can be aggregate by case_id for patient-level label
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print("Slide val_df cohort distribution:")
        # print(val_df["cohort"].value_counts())
        print(val_df.groupby("case_id")["cohort"].first().value_counts())
        print("Slide test_df cohort distribution:")
        # print(test_df["cohort"].value_counts())
        print(test_df.groupby("case_id")["cohort"].first().value_counts())