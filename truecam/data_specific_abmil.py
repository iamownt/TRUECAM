import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
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
from sklearn.metrics import roc_auc_score
import argparse
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian
# warnings.filterwarnings('error')
import operator
import wandb
from functools import partial
import copy
import fcntl
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d

lazy_mapping = {"tcga": "itest", "cptac": "etest"}

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


def load_pickle_model(path: str = "kmeans_100w.pkl"):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model


def save_pickle_data(data, path: str = "kmeans_100w.pkl"):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def get_abmil_params():
    parser = argparse.ArgumentParser(description='ABMIL on downstream tasks')
    parser.add_argument('--trial', type=int, default=4, help='Set trial')
    parser.add_argument('--fold', type=int, default=5, help='Set fold')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--training_dataset', type=str, default=None, choices=["tcga", "cptac"], help='Training dataset')
    parser.add_argument('--model_name', type=str, default="uni", help='Model name')
    parser.add_argument('--spec_norm_bound', type=none_or_float, default=None,
                        help='Spectral norm bound, if set not None, will use spectral normalization')
    parser.add_argument('--gaussian_process', action='store_true',
                        help='If set True, will use Laplace Approximization of Gaussian process to estimate the uncertainty')
    parser.add_argument('--num_inducing', type=int, default=None,
                        help='Number of inducing points for Gaussian process')
    parser.add_argument('--gp_kernel_type', type=str, default=None, help='Type of GP kernel')
    parser.add_argument('--gp_input_normalization', type=bool, help='Enable input normalization (True/False)')

    parser.add_argument('--spec_norm_replace_list', type=str, default='Linear,Conv2D',
                        help='List of layers to replace with spectral normalization')
    parser.add_argument('--save_to_parquet', action='store_true',
                        help='If set True, will save the results to parquet file')
    parser.add_argument('--save_destination', type=str, default="/home/user/sngp/UniConch/data_specific_models/",
                        help='Model and parquet save path')
    parser.add_argument('--csv_destination', type=str, default="/home/user/wangtao/prov-gigapath/data_specific_training_csv",
                        help='Save csv for better remote sync')
    parser.add_argument('--retrain', action='store_true', help='retrain the model')
    parser.add_argument('--mask_tile', action='store_true', help='whether to mask the tiles')
    parser.add_argument('--mask_tile_category', type=str, default="rand",
                        choices=["rand", "in_slide", "in_slide_weight", "all_slide"], help='whether to mask the tiles')
    parser.add_argument('--mask_tile_threshold', type=convert_threshold, default=0.4, help='mask tile threshold')
    parser.add_argument('--nearby_tiles', type=int, default=1, help='number of nearby tiles to consider')
    parser.add_argument('--invert_threshold', action='store_true', help='whether to invert the threshold')
    parser.add_argument('--evaluate_only', action='store_true', help='evaluate the model')
    parser.add_argument('--results_file_path', type=str, default="final_results.csv", help='final_results.csv')
    parser.add_argument('--force_middle_prefix', type=str, default=None, help='force middle prefix')
    parser.add_argument('--set', dest='set_gp', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()
    args.spec_norm_replace_list = args.spec_norm_replace_list.split(',')
    args.save_destination = Path(args.save_destination)
    args.csv_destination = Path(args.csv_destination)
    assert args.results_file_path.endswith(
        ".csv"), f"results_file_path should be a csv file, get {args.results_file_path}"
    if args.retrain:
        assert args.mask_tile, "If retrain, mask_tile should be set"
        assert args.mask_tile_category == "in_slide", "If retrain, mask_tile_category should be in_slide"
    return parser, args


def amb_function(model_name, dataset_name, tuning_method):
    return f"/home/user/sngp/UniConch/models/ambpkl/newambk/{model_name}_{dataset_name}_ambiguity_dict_autogluon_0.2_tuning{tuning_method}.pkl"


class UniTrainingConfig:
    model_name = "uni"
    dataset_h5 = Path("/home/user/sngp/UniConch/uni_tcga_h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    etest_h5 = Path("/home/user/sngp/UniConch/uni_cptac_h5file")
    external_dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labelsv2.csv")
    mask_func = amb_function

    embed_dim = 1024
    batch_size = 1
    num_workers = 8
    epochs = 20
    label_dict = {"LUAD": 0, "LUSC": 1}


class ConchTrainingConfig:
    model_name = "conch"
    dataset_h5 = Path("/home/user/sngp/UniConch/conch_tcga_h5file")
    etest_h5 = Path("/home/user/sngp/UniConch/conch_cptac_h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    external_dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labelsv2.csv")
    mask_func = amb_function

    embed_dim = 512
    batch_size = 1
    num_workers = 8
    epochs = 20
    label_dict = {"LUAD": 0, "LUSC": 1}


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
    'num_inducing': 2048,
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

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionMIL, self).__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 2)

    def forward(self, H):
        tanh_VH = torch.tanh(self.V(H))
        attention_scores = self.w(tanh_VH)
        attention_weights = torch.softmax(attention_scores, dim=0)
        return attention_weights


class GatedAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_input, dropout_hidden):
        super(GatedAttention, self).__init__()
        assert 0 <= dropout_input <= 1 and 0 <= dropout_hidden <= 1
        self.attention_a = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_input))
        self.attention_b = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout_hidden))
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        c = a.mul(b)
        c = self.w(c)
        prob = F.softmax(c, dim=1)  # abmil likes to use batch size 1
        return (prob * x).sum(dim=1)


class GatedABMIL(nn.Module):
    """https://github.com/mahmoodlab/CLAM/models/model_clam.py
       The only differences is that we use single mapping to enable uni and conch with the same hidden state for attention
    """

    def __init__(self, embed_dim: int = 1024, hdim1: int = 512, hdim2: int = 384, n_classes: int = 2):
        super(GatedABMIL, self).__init__()
        if embed_dim == 512:
            self.fair_proj = nn.Linear(embed_dim, 1024)
            print("use fair projection")
            embed_dim = 1024
        else:
            self.fair_proj = nn.Identity()
        self.feature_extractor = nn.Sequential(nn.Linear(embed_dim, hdim1), nn.ReLU(), nn.Dropout(0.1))
        self.attention_layer = GatedAttention(hdim1, hdim2, dropout_input=0.25, dropout_hidden=0.25)
        self.classifier = nn.Linear(hdim1, n_classes)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, **kwargs):
        x = self.fair_proj(x)
        x = self.feature_extractor(x)
        x = self.attention_layer(x)
        return self.classifier(x, **kwargs)


class H5Dataset(Dataset):

    def __init__(self, h5_path, filter_mapping, s_to_p_mapping=None,
                 mask_pkl=None, mask_tile_category=None, mask_tile_threshold=None, invert_threshold=None):
        self.h5_path = h5_path
        self.h5_paths = os.listdir(h5_path)
        self.h5_paths = [h5_path / path for path in self.h5_paths if int(path[:-3]) in filter_mapping.keys()]
        self.h5_labels = [filter_mapping[int(h5_path.name[:-3])] for h5_path in self.h5_paths]
        if s_to_p_mapping is None:
            self.s_to_p_mapping = None
        else:
            self.s_to_p_mapping = s_to_p_mapping
        if mask_pkl is None:
            self.mask_pkl = None
        else:
            self.mask_pkl = mask_pkl
            self.mask_threshold = mask_tile_threshold
            self.mask_tile_category = mask_tile_category
            self.comp_func = operator.gt if invert_threshold else operator.lt
            self.invert_threshold = invert_threshold

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        assets, attrs = self.read_assets_from_h5(self.h5_paths[idx], self.mask_pkl)
        # print("after", assets["tile_embeds"].shape)
        assets["labels"] = self.h5_labels[idx]
        if self.s_to_p_mapping is None:
            assets["patient_int"] = int(self.h5_paths[idx].name[:-3])
        else:
            assets["patient"] = self.s_to_p_mapping[int(self.h5_paths[idx].name[:-3])]
        return assets

    def read_assets_from_h5(self, h5_path: str, mask_pkl=None) -> tuple:
        '''Read the assets from the h5 file'''
        assets = {}
        attrs = {}
        with h5py.File(h5_path, 'r') as f:
            if mask_pkl is not None:
                if self.mask_tile_category == "rand":
                    mask_bool = self.comp_func(np.random.rand(f["tile_embeds"].shape[0]), self.mask_threshold)
                elif self.mask_tile_category == "in_slide":
                    coords = f["coords"][:]
                    if self.mask_threshold != 1.0 and args.nearby_tiles > 1:
                        tree = KDTree(coords)
                        _, indices = tree.query(coords, k=args.nearby_tiles)
                        ambiguity_value = mask_pkl[h5_path.stem].reshape(-1, 1)
                        gathered_points = np.take_along_axis(ambiguity_value, indices, axis=0)
                        # ambiguity_value = gaussian_filter1d(gathered_points, sigma=1., axis=1)
                        ambiguity_value = np.mean(gathered_points, axis=1)
                    else:
                        ambiguity_value = mask_pkl[h5_path.stem]
                    # print("ambiguity_value", ambiguity_value.shape)
                    # print("indices", indices.shape)
                    # ambiguity_value = np.mean(gathered_points, axis=1, keepdims=True)
                    in_slide_threshold = np.quantile(ambiguity_value, self.mask_threshold)
                    mask_bool = self.comp_func(ambiguity_value, in_slide_threshold)
                elif self.mask_tile_category == "in_slide_weight":
                    slide_ambiguity = mask_pkl[h5_path.stem]
                    # Normalize the ambiguity scores to [0, 1]
                    # slide_ambiguity = (slide_ambiguity - slide_ambiguity.min()) / (slide_ambiguity.max() - slide_ambiguity.min())
                    # Calculate the selection probability inversely proportional to the ambiguity scores
                    selection_probabilities = 1 - slide_ambiguity
                    selection_probabilities /= np.sum(selection_probabilities)
                    mask_bool = np.zeros(len(selection_probabilities), dtype=bool)
                    if self.invert_threshold:
                        num_selected = int((1 - self.mask_threshold) * len(selection_probabilities))
                    else:
                        num_selected = int(self.mask_threshold * len(selection_probabilities))
                    # print("selection_probabilities", selection_probabilities)
                    selected_indices = np.random.choice(len(selection_probabilities), num_selected, replace=False,
                                                        p=selection_probabilities)
                    mask_bool[selected_indices] = True
                    mask_bool = ~mask_bool if self.invert_threshold else mask_bool
                else:
                    mask_bool = self.comp_func(mask_pkl[h5_path.stem], self.mask_threshold)
            for key in f.keys():
                if mask_pkl is not None:

                    assets[key] = f[key][:][mask_bool]
                else:
                    assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs


def create_slide_int_to_patient_mapping(df):
    slide_int_to_patient = dict(zip(df["slide_int"], df["patient"]))
    return slide_int_to_patient


def train(config, args, i, j):
    if args.spec_norm_bound is None:
        middle_prefix = f"{config.model_name}"
    else:
        middle_prefix = f"{config.model_name}_sn{args.spec_norm_bound}_gp{int(args.gaussian_process)}"
    if args.retrain:
        middle_prefix += f"_retrain_{args.mask_tile_category}_thres{args.mask_tile_threshold}"

    if args.num_inducing is not None or args.gp_kernel_type is not None or args.gp_input_normalization is not None:
        middle_prefix += f"_kernel_{args.gp_kernel_type}_inducing{args.num_inducing}_inorm{int(args.gp_input_normalization)}"
    print("middle_prefix", middle_prefix)
    if hasattr(args, "force_middle_prefix") and args.force_middle_prefix is not None:
        middle_prefix = args.force_middle_prefix
        print("force_middle_prefix", middle_prefix)
    args.to_destination = args.save_destination / middle_prefix / f"{args.training_dataset}" / f"t{i}f{j}"
    training_csv_path = config.dataset_csv if args.training_dataset == "tcga" else config.external_dataset_csv
    df = pd.read_csv(training_csv_path)
    df["cohort"] = df["cohort"].apply(lambda x: config.label_dict[x])
    split_key = f"split{args.fold * i + j}"
    train_df, val_df, test_df = df[df[split_key] == "train"], df[df[split_key] == "val"], df[df[split_key] == "test"]
    train_df_mapping = train_df.set_index("slide_int")["cohort"].to_dict()
    val_df_mapping = val_df.set_index("slide_int")["cohort"].to_dict()
    test_df_mapping = test_df.set_index("slide_int")["cohort"].to_dict()

    def get_quantile_removal(mask_pkl, df_mapping, quantile_point):
        amb_list = []
        for key in df_mapping.keys():
            amb_list.append(mask_pkl[str(key)])
        tau = np.quantile(np.concatenate(amb_list), quantile_point)
        return tau

    if args.mask_tile:
        if args.training_dataset == "tcga":
            mask_pkl = load_pickle_model(config.mask_func(args.model_name, "itest", 0))[f"t{i}f{j}"]
        else:
            mask_pkl = load_pickle_model(config.mask_func(args.model_name, "etest", 2))[f"t{i}f{j}"]

        if args.mask_tile_category == "rand" or args.mask_tile_category == "in_slide" or args.mask_tile_category == "in_slide_weight":  # float
            mask_tile_threshold = args.mask_tile_threshold
            val_mask_tile_threshold = args.mask_tile_threshold
        else:
            # train_mask_tile_threshold = mask_pkl["train_quantile_list"][args.mask_tile_threshold]
            # val_mask_tile_threshold = mask_pkl["val_quantile_list"][args.mask_tile_threshold]

            mask_tile_threshold = get_quantile_removal(mask_pkl, test_df_mapping, args.mask_tile_threshold)
            val_mask_tile_threshold = mask_tile_threshold

        print("Train Quantile", mask_pkl["train_quantile_list"])
        print("Val Quantile", mask_pkl["val_quantile_list"])
        print("Removing Train tiles with quantile larger than", mask_tile_threshold)
        print("Removing Val tiles with quantile larger than", val_mask_tile_threshold)
    else:
        mask_pkl = None
        mask_tile_threshold = None
        val_mask_tile_threshold = None

    s_to_p_mapping = create_slide_int_to_patient_mapping(df)
    dataset_h5 = config.dataset_h5 if args.training_dataset == "tcga" else config.etest_h5
    train_dataset, val_dataset, test_dataset = (H5Dataset(dataset_h5, train_df_mapping,
                                                                         s_to_p_mapping=s_to_p_mapping,
                                                                         mask_pkl=mask_pkl,
                                                                         mask_tile_category=args.mask_tile_category,
                                                                         mask_tile_threshold=mask_tile_threshold,
                                                                         invert_threshold=args.invert_threshold),
                                                               H5Dataset(dataset_h5, val_df_mapping,
                                                                         s_to_p_mapping=s_to_p_mapping,
                                                                         mask_pkl=mask_pkl,
                                                                         mask_tile_category=args.mask_tile_category,
                                                                         mask_tile_threshold=val_mask_tile_threshold,
                                                                         invert_threshold=args.invert_threshold),
                                                               H5Dataset(dataset_h5, test_df_mapping,
                                                                         s_to_p_mapping=s_to_p_mapping,
                                                                         mask_pkl=mask_pkl,
                                                                         mask_tile_category=args.mask_tile_category,
                                                                         mask_tile_threshold=val_mask_tile_threshold,
                                                                         invert_threshold=args.invert_threshold))
    print(f"train dataset size: {len(train_dataset)}, val dataset size: {len(val_dataset)}, "
          f"test dataset size: {len(test_dataset)}")

    def collate_fn(batch):
        return {
            'tile_embeds': torch.stack([torch.from_numpy(x['tile_embeds']) for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch]),
            "patient": np.array([x['patient'] for x in batch])
        }

    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                                                     num_workers=config.num_workers,
                                                                     collate_fn=collate_fn), \
        DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn), \
        DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=collate_fn), \

    model = GatedABMIL(embed_dim=config.embed_dim).cuda()
    if args.spec_norm_bound is not None:
        model = convert_to_sn_my(model, args.spec_norm_replace_list, args.spec_norm_bound)
    if args.gaussian_process:
        GP_KWARGS = GP_KWARGS_CONCH if args.model_name == "conch" else GP_KWARGS_UNI
        if args.num_inducing is not None:
            GP_KWARGS['num_inducing'] = args.num_inducing
        if args.gp_kernel_type is not None:
            GP_KWARGS['gp_kernel_type'] = args.gp_kernel_type
        if args.gp_input_normalization is not None:
            GP_KWARGS['gp_input_normalization'] = args.gp_input_normalization
        print(GP_KWARGS)
        replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
    print("parameter", sum(p.numel() for p in model.parameters()))
    print(model)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    total_iter = len(train_dataset) // config.batch_size * config.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_iter, eta_min=0)
    criterion = nn.CrossEntropyLoss()
    best_val_accuracy, best_val_loss = 0.0, float('inf')
    if args.evaluate_only:
        print("args.to_destination", args.to_destination)
        model.load_state_dict(torch.load(args.to_destination / "best_model.pth"))
        test_acc, test_bacc, test_auc, test_loss = evaluate(model, test_loader, args, i, j, lazy_mapping[args.training_dataset])
        print(
            f"Trial {i} Fold {j} Test Accuracy: {test_acc}, Test Balanced Accuracy: {test_bacc}, Test AUC: {test_auc}, Test Loss: {test_loss}")
        return test_acc, test_bacc, test_auc
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
            H = assets['tile_embeds'].float().cuda()
            labels = assets['labels'].long().cuda()
            preds = model(H, **kwargs)
            loss = criterion(preds, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"{i * 4 + j} Epoch {epoch + 1}/{config.epochs}, Loss: {total_loss / len(train_loader)}")
        val_accuracy, _, _, val_loss = evaluate(model, val_loader, args, i, j)
        print(f"Validation Accuracy after Epoch {epoch + 1}: {val_accuracy}")
        if val_accuracy > best_val_accuracy or (val_accuracy == best_val_accuracy and val_loss < best_val_loss):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            os.makedirs(args.to_destination, exist_ok=True)
            torch.save(model.state_dict(), args.to_destination / "best_model.pth")

    model.load_state_dict(torch.load(args.to_destination / f"best_model.pth"))
    val_accuracy, _, _, _ = evaluate(model, val_loader, args, i, j, "ival")
    test_acc, test_bacc, test_auc, test_loss = evaluate(model, test_loader, args, i, j, lazy_mapping[args.training_dataset])
    print(
        f"Trial {i} Fold {j} Test Accuracy: {test_acc}, Test Balanced Accuracy: {test_bacc}, Test AUC: {test_auc}, Test Loss: {test_loss}")
    return test_acc, test_bacc, test_auc


def evaluate(model, loader, args, i, j, tag=None):
    labels = []
    patient_id = []
    logits_list = []
    uncertainty_list = []
    model.eval()
    if args.gaussian_process:
        model.classifier.update_covariance_matrix()
        eval_kwargs = {'return_random_features': False, 'return_covariance': True,
                       'update_precision_matrix': False, 'update_covariance_matrix': False}
    else:
        eval_kwargs = {}
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for idx, assets in tqdm(enumerate(loader)):
        H = assets['tile_embeds'].float().cuda()
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
    df = pd.DataFrame({
        'patient_id': np.array(patient_id),
        'logit_0': logits[:, 0],
        'logit_1': logits[:, 1],
        'labels': labels,
    })

    agg_dict = {
        'logit_0': 'mean',
        'logit_1': 'mean',
        'labels': 'first',
    }

    if uncertainty_list:
        df['uncertainty'] = uncertainty_list
        agg_dict['uncertainty'] = 'mean'

    agg_df = df.groupby('patient_id').agg(agg_dict).reset_index()

    results = {
        'logit': np.array(agg_df[['logit_0', 'logit_1']]),
        'label': np.array(agg_df['labels']),
        'prob': softmax(np.array(agg_df[['logit_0', 'logit_1']]), axis=1),
        'patient': np.array(agg_df['patient_id'])
    }

    if uncertainty_list:
        results['uncertainty'] = np.array(agg_df['uncertainty'])

    print(f"agg from {len(df)} to {len(agg_df)}")
    try:
        acc = accuracy_score(results['label'], results['prob'].argmax(axis=1))
        bacc = balanced_accuracy_score(results['label'], results['prob'].argmax(axis=1))
        auc = roc_auc_score(results['label'], results['prob'][:, 1])
    except Exception as e:
        print(f"Error: {e}")
        acc, bacc, auc = 0, 0, 0
    if args.save_to_parquet and tag is not None:
        save_df = pd.DataFrame({
            'Outcome 0-y_pred0': results['logit'][:, 0],
            'Outcome 0-y_pred1': results['logit'][:, 1],
            'Outcome 0-y_true': results['label'],
            'prob_0': results['prob'][:, 0],
            'prob_1': results['prob'][:, 1],
            'patient': results['patient']
        })
        if uncertainty_list:
            save_df['Outcome 0-uncertainty0'] = results['uncertainty']
        if args.mask_tile:
            save_df_name = f"patient_predictions_{tag}_t{i}f{j}_mask{args.mask_tile_category}_thres{args.mask_tile_threshold}_invert{int(args.invert_threshold)}_nearby_tiles{args.nearby_tiles}.parquet.gzip"
        else:
            save_df_name = f"patient_predictions_{tag}_t{i}f{j}.parquet.gzip"
        save_df.to_parquet(args.to_destination / save_df_name, compression="gzip")

    return acc, bacc, auc, val_loss


def load_training_config(model_name):
    if model_name == "uni":
        return UniTrainingConfig
    elif model_name == "conch":
        return ConchTrainingConfig
    else:
        raise ValueError(f"Invalid model name {model_name}")

def main_spawn(args):
    training_config = load_training_config(model_name=args.model_name)
    start_time = time.time()
    for config in [training_config]:
        print(config.model_name)
        metric_dict = defaultdict(list)
        for i in range(args.trial):
            for j in range(args.fold):
                test_acc, test_bacc, test_auc = train(config, args, i, j)
                metric_dict['test_acc'].append(test_acc)
                metric_dict['test_bacc'].append(test_bacc)
                metric_dict['test_auc'].append(test_auc)

        df_metrics = pd.DataFrame(metric_dict)
        summary_metrics = df_metrics.agg(['mean', 'std']).transpose()
        summary_metrics.columns = ['mean', 'std']
        summary_metrics["metrics"] = ["test_acc", "test_bacc", "test_auc"]
        summary_metrics["mean-std"] = (summary_metrics["mean"].round(4).astype(str) + "+-" +
                                       summary_metrics["std"].round(4).astype(str))
        summary_metrics.drop(columns=["mean", "std"], inplace=True)
        summary_metrics['tag'] = args.to_destination.parents[1].name
        summary_metrics = summary_metrics.pivot(index='tag', columns='metrics', values='mean-std').reset_index()
        for metric in metric_dict.keys():
            summary_metrics[metric + "_list"] = [metric_dict[metric]]  # additional original value for each metric
        if args.mask_tile is False:
            mask_ratio = "no_mask"
        else:
            mask_ratio = f"mask_category{args.mask_tile_category}_thres{args.mask_tile_threshold}"
        summary_metrics["retrain"] = "retrain" if args.retrain else "no_retrain"
        summary_metrics["mask_ratio"] = mask_ratio
        summary_metrics["nearby_tiles"] = args.nearby_tiles
        summary_metrics["dataset"] = args.training_dataset
        print(summary_metrics)

        results_file_path = args.csv_destination / args.results_file_path
        if results_file_path.exists():
            existing_data = pd.read_csv(results_file_path)
            updated_data = pd.concat([existing_data, summary_metrics],
                                     ignore_index=True)  # Concatenating with ignore_index=True
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