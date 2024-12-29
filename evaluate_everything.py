import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
from ast import literal_eval

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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import argparse
from sngp_wrapper.covert_utils import convert_to_sn_my, replace_layer_with_gaussian
sys.path.append('..')
from biscuit.compatible_slideflowv3 import InitEvaluator, compute_internal_external_acc_auc_metrics
from easydict import EasyDict as edict
from ABMIL import (H5Dataset, GatedABMIL, none_or_float, load_training_config, seed_everything, GP_KWARGS_UNI, GP_KWARGS_CONCH,
                   load_pickle_model)
from data_specific_abmil import evaluate
from typing import Union, Optional, Any
import shutil
import scipy
import operator
from ABMIL import convert_threshold
# warnings.filterwarnings('error')
import scipy.stats as stats



def save_model(model, name_space):
    with open(name_space, "wb") as file:
        pickle.dump(model, file)


def load_model(path: Union[str, Path] = "kmeans_100w.pkl"):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model



def compute_mean_std_new(array: list, round_num: int = 4):
    mean_test_acc = np.mean(array).round(round_num)
    std_test_acc = np.std(array).round(round_num)
    return f"{mean_test_acc}+-{std_test_acc}"


def get_prov_gigapath_file_info_dict(folds: int = 5, save_destination: str = "", compute_prefix: str = "titan",
                                     ispecial_prefix: str = None, especial_prefix: str = None):
    # ispecial_prefix and especial_prefix shoule be all None or all with values
    assert (ispecial_prefix is None and especial_prefix is None) or (ispecial_prefix is not None and especial_prefix is not None)
    all_file_info_dict = dict()
    for fold in range(folds):
        i, j = fold // 5, fold % 5
        all_file_info_dict[f"t{i}_f{j}"] = {}
        if ispecial_prefix is not None:
            itest_csv_name = f'test_pair_fold{fold}_masktile1_thres{ispecial_prefix}.parquet.gzip'
            etest_csv_name = f'test_pair_fold{fold}_masktile1_thres{especial_prefix}.parquet.gzip'
        else:
            itest_csv_name = f'test_pair_fold{fold}.parquet.gzip'
            etest_csv_name = f'test_pair_fold{fold}.parquet.gzip'
        if compute_prefix == "titan":
            current_file_info_dict = {"patient_predictions": {
                "itest": os.path.join(save_destination, "tcga/nsclc", f"fold_{fold}", itest_csv_name),
                "ival": None,
                "etest": os.path.join(save_destination, "cptac/nsclc", f"fold_{fold}", etest_csv_name),
            }, "meta_one_for_all": False, "remove_cat": False, "vanilla_average": False, "remove_cat_for_etest": False}
        else:
            current_file_info_dict = {"patient_predictions": {
                "itest": os.path.join(save_destination, "tcga/nsclc/sngp/eval_pretrained_nsclc", f"fold_{fold}", itest_csv_name),
                "ival": None,
                "etest": os.path.join(save_destination, "cptac/nsclc/sngp-cptac/eval_pretrained_nsclc", f"fold_{fold}", etest_csv_name),
            }, "meta_one_for_all": False, "remove_cat": False, "vanilla_average": False, "remove_cat_for_etest": False}
        all_file_info_dict[f"t{i}_f{j}"] = current_file_info_dict
    return all_file_info_dict


def get_foundation_file_info_dict(trials: int, folds: int, compute_prefix: str = "uni",
                                  ispecial_prefix: str = None, especial_prefix: str = None, model_path: Any = None):
    save_destination = model_path / f"{compute_prefix}"
    all_file_info_dict = dict(meta_info=compute_prefix)
    if ispecial_prefix is not None:
        ispecial_prefix = f"_{ispecial_prefix}"
    else:
        ispecial_prefix = ""
    if especial_prefix is not None:
        especial_prefix = f"_{especial_prefix}"
    else:
        especial_prefix = ""
    for i in range(trials):
        for j in range(folds):
            all_file_info_dict[f"t{i}_f{j}"] = {}
            current_file_info_dict = {"patient_predictions": {
                "itest": str(save_destination / "tcga" / f"t{i}f{j}" / f"patient_predictions_itest_t{i}f{j}{ispecial_prefix}.parquet.gzip"),
                "ival": str(save_destination / "tcga" / f"t{i}f{j}" / f"patient_predictions_ival_t{i}f{j}.parquet.gzip"),
                "etest": str(save_destination / "cptac" / f"t{i}f{j}" / f"patient_predictions_etest_t{i}f{j}{especial_prefix}.parquet.gzip"),
            }, "meta_one_for_all": False, "remove_cat": False, "vanilla_average": False, "remove_cat_for_etest": False}

            all_file_info_dict[f"t{i}_f{j}"] = current_file_info_dict
    return all_file_info_dict


def load_ood_config(model_name, ood_dataset):
    assert model_name in ["titan", "prov-gigapath", "uni", "conch", "titan-cptac"]
    assert ood_dataset in ["blca", "ucs", "uvm", "acc", "unified"]
    if model_name == "prov-gigapath":
        embed_d = 1536
    elif model_name == "uni":
        embed_d = 1024
    elif model_name == "conch":
        embed_d = 512
    elif model_name == "titan":
        embed_d = 768
    elif model_name == "titan-cptac":
        embed_d = 768
        model_name = "titan"
    class cfg:
        h5_path = f"/home/user/sngp/UniConch/{model_name}_{ood_dataset}_h5file"
        embed_dim = embed_d
        pkl_path = f"/home/user/sngp/UniConch/ood_pkl_folder/{ood_dataset}_mapping.pkl"

        num_workers = 8
        batch_size = 1
    return cfg


def get_extended_abmil_params():
    parser = argparse.ArgumentParser(description='ABMIL on downstream tasks')
    parser.add_argument('--trial',                  type=int,  default=4,  help='Set trial')
    parser.add_argument('--fold',                   type=int,  default=5,  help='Set fold')
    parser.add_argument('--seed',                   type=int,  default=2024,  help='Random seed')
    parser.add_argument('--training_dataset', type=str, default=None, choices=["tcga", "cptac"], help='Training dataset')
    parser.add_argument('--model_name',             type=str,  default="uni", help='Model name')
    parser.add_argument('--spec_norm_bound',        type=none_or_float,  default=None,  help='Spectral norm bound, if set not None, will use spectral normalization')
    parser.add_argument('--gaussian_process',       action='store_true', help='If set True, will use Laplace Approximization of Gaussian process to estimate the uncertainty')
    parser.add_argument('--spec_norm_replace_list', type=str, default='Linear,Conv2D', help='List of layers to replace with spectral normalization')
    parser.add_argument('--save_to_parquet',        action='store_true', help='If set True, will save the results to parquet file')
    parser.add_argument('--save_destination',       type=str, default="/home/user/sngp/UniConch/data_specific_models/", help='Model and parquet save path')
    parser.add_argument('--csv_destination', type=str, default="/home/user/wangtao/prov-gigapath/data_specific_training_csv",
                        help='Save csv for better remote sync')
    parser.add_argument('--ood_dataset_name', type=str, default="ucs", choices=["ucs", "uvm", "blca", "acc", "unified"], help='An additional argument')
    parser.add_argument('--generate_ood',        action='store_true', help='If set True, will generate the ood data')
    parser.add_argument('--gp_num_inducing',        type=int, default=None, help='Number of inducing points for Gaussian process')
    parser.add_argument('--ispecial_prefix',        type=str, default=None, help='special prefix for eat itest')
    parser.add_argument('--especial_prefix',        type=str, default=None, help='special prefix for eat etest')
    parser.add_argument('--eval_type',        type=str, default="cp", choices=["cp", "ood"], help='evaluation type')
    parser.add_argument('--ood_detection_type', type=str, default="uncertainty", choices=["uncertainty", "probability"], help="incertainty_type")
    parser.add_argument('--mask_tile',              action='store_true', help='whether to mask the tiles')
    parser.add_argument('--mask_tile_category',     type=str, default="rand",  help='whether to mask the tiles')
    parser.add_argument('--mask_tile_threshold', type=convert_threshold, default=None, help='mask tile threshold')
    parser.add_argument('--nearby_tiles', type=int, default=1, help='number of nearby tiles to consider')
    parser.add_argument('--invert_threshold',    action='store_true', help='whether to invert the threshold')
    parser.add_argument('--results_file_path', type=str, default="final_results.csv", help='final_results.csv')
    parser.add_argument('--force_middle_prefix', type=str, default=None, help='force middle prefix')

    args = parser.parse_args()
    args.spec_norm_replace_list = args.spec_norm_replace_list.split(',')
    args.save_destination = Path(args.save_destination)    # Add additional arguments
    args.csv_destination = Path(args.csv_destination)

    return args


class OODH5Dataset(H5Dataset):
   def __init__(self, h5_path, filter_mapping, s_to_p_mapping=None,
                mask_pkl=None, mask_tile_category=None, mask_tile_threshold=None, invert_threshold=None):
        self.h5_path = h5_path
        self.h5_paths = os.listdir(h5_path)
        self.h5_paths = [h5_path / path for path in self.h5_paths if filter_mapping is None or
                         int(path[:-3]) in filter_mapping.keys()]
        self.h5_labels = [0] * len(self.h5_paths)
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

class GigaPathFoundationEvaluator(InitEvaluator):

    def __init__(self, args, set_tpr: float = 0.95, compute_prefix: str = "gigapath",
                 conformal_key_param: dict = None, force_alpha_list: list = None, level_list: list = None):

        self.trials = args.trial
        self.folds = args.fold
        self.set_tpr = set_tpr
        self.compute_prefix = compute_prefix
        print(args)
        self.file_info_dict = get_prov_gigapath_file_info_dict(args.trial*args.fold, args.save_destination, compute_prefix=compute_prefix,
                                                               ispecial_prefix=args.ispecial_prefix,
                                                               especial_prefix=args.especial_prefix)
        self.detect_cptac_ood_measure = None
        self.detect_cptac_ood_threshold = None
        self.uncertainty_topk = None

        # we will need the special prefix to load the special file for cp evaluation
        self.level_list = level_list
        if conformal_key_param is None:
            if force_alpha_list is None:
                force_alpha_list = [0.1, 0.05, 0.01]

            self.ckp = edict({
                "tile_n": 20000,
                "slide_n": 100,
                "mean_field_factor": None,
                "use_val_qhat": False,
                "replicated_num": 500,
                "alpha_list": force_alpha_list,
            })
        else:
            self.ckp = edict(conformal_key_param)


class FoundationEvaluator(InitEvaluator):

    def __init__(self, args, set_tpr: float = 0.95, compute_prefix: str = "uni",
                 conformal_key_param: dict = None, force_alpha_list: list = None, level_list: list = None):

        self.trials = args.trial
        self.folds = args.fold
        self.set_tpr = set_tpr
        self.compute_prefix = compute_prefix
        print(args)
        self.file_info_dict = get_foundation_file_info_dict(args.trial, args.fold, compute_prefix=compute_prefix,
                                                            ispecial_prefix=args.ispecial_prefix,
                                                            especial_prefix=args.especial_prefix, model_path=args.save_destination)
        self.detect_cptac_ood_measure = None
        self.detect_cptac_ood_threshold = None
        self.uncertainty_topk = None

        # we will need the special prefix to load the special file for cp evaluation
        self.level_list = level_list
        if conformal_key_param is None:
            if force_alpha_list is None:
                force_alpha_list = [0.1, 0.05, 0.01]

            self.ckp = edict({
                "tile_n": 20000,
                "slide_n": 100,
                "mean_field_factor": None,
                "use_val_qhat": False,
                "replicated_num": 500,
                "alpha_list": force_alpha_list,
            })
        else:
            self.ckp = edict(conformal_key_param)


    def deprecated_extend_classification_compute_pvalue(self, save_name: str = None):
        # remind 1: in compute_internal_external_acc_auc_metrics, softmax -> 0.5 threshold is not strictly equal to
        # argmax of the logits, so the results may be slightly different from the original results
        # remind 2: in the beginning, the auroc is computed with average the 0, 1 label, but now is computed with 1 label
        # so the results may be slightly different from the original results
        if "sn" in self.compute_prefix:
            baseline_prefix = self.compute_prefix.split("_sn")[0]
        baseline_info_dict = get_foundation_file_info_dict(self.trials, self.folds, compute_prefix=baseline_prefix)
        sngp_file_info = get_foundation_file_info_dict(self.trials, self.folds, compute_prefix=self.compute_prefix)

        baseline_level_info_dict, pvalue_baseline_dict = \
            compute_internal_external_acc_auc_metrics(baseline_info_dict, trials=self.trials, folds=self.folds,
                                                      return_all_list=True, return_pvalue_dict=True,
                                                      pred_thresh=None,
                                                      set_tpr=None)
        sngp_level_info_dict, pvalue_sngp_dict = \
            compute_internal_external_acc_auc_metrics(sngp_file_info, trials=self.trials, folds=self.folds,
                                                      return_all_list=True, return_pvalue_dict=True,
                                                      pred_thresh=None,
                                                      set_tpr=None)
        print("Internal and External metrics are computed")
        print("baseline_level_info_dict", baseline_level_info_dict)
        print("sngp_level_info_dict", sngp_level_info_dict)
        level = "patient"
        pvalue_list = []
        for idx, namespace in zip([0, 1, 2, 3], ["iacc", "iauc", "eacc", "eauc"]):
            t_statistic, p_value = scipy.stats.wilcoxon(x=np.array(baseline_level_info_dict[level][idx]),
                                                        y=np.array(sngp_level_info_dict[level][idx]),
                                                        alternative='less')
            print(f"level: {level} {namespace}", "baseline to sngp", t_statistic, p_value)
            pvalue_list.append(p_value)
        if save_name is not None:
            plot_info_interval = {
                "baseline": baseline_level_info_dict,
                'sngp': sngp_level_info_dict,
                'pvalue_list': pvalue_list,
            }
            plot_info_path = f"/home/user/wangtao/biscuit/jul9/{save_name}.pkl"
            save_model(plot_info_interval, plot_info_path)

def amb_mask_function(model_name, dataset_name):
    return f"/home/user/sngp/UniConch/models/ambpkl/newambk/{model_name}_{dataset_name}_ambiguity_dict_autogluon_0.2_tuning0.pkl"

def load_all_ood_list(args: Any, i: int, j: int, ood_exp_name_list: list = None):
    ood_df_list = []
    for ood_exp_name in ood_exp_name_list:
        if args.mask_tile:
            save_ood_df_name = f"patient_predictions_{ood_exp_name}_t{i}f{j}_mask{args.mask_tile_category}_thres{args.mask_tile_threshold}_invert{int(args.invert_threshold)}_nearby_tiles{args.nearby_tiles}.parquet.gzip"
        else:
            save_ood_df_name = f"patient_predictions_{ood_exp_name}_t{i}f{j}.parquet.gzip"

        ood_df = pd.read_parquet(args.to_destination / save_ood_df_name)
        ood_df_list.append(ood_df)
    return pd.concat(ood_df_list, axis=0)
class FoundationOODEvaluator:
    # save ood data to self.model_name / t{i}f{j} / ood_dataset / patient_predictions_t{i}f{j}.parquet.gzip
    # ood_dataset -> ucs, uvm, acc, blca
    def __init__(self, args: Any):
        init_model_path = args.save_destination
        self.trial = args.trial
        self.fold = args.fold
        self.args = args
        self.model_name = init_model_path / args.model_name
        self.config = load_ood_config(model_name=self.args.model_name, ood_dataset=self.args.ood_dataset_name)
        self.config.mask_func = amb_mask_function

    def generate_ood_data(self):
        def collate_fn(batch):
            return {
                'tile_embeds': torch.stack([torch.from_numpy(x['tile_embeds']) for x in batch]),
                'labels': torch.tensor([x['labels'] for x in batch]),
                "patient": np.array([x['patient'] for x in batch])
            }

        tbar = tqdm(self.trial*self.fold, desc="Generating OOD data")
        for i in range(self.trial):
            for j in range(self.fold):
                args.to_destination = args.save_destination / args.compute_prefix / f"{args.training_dataset}" / f"t{i}f{j}"
                print("\033[1;32;40m", "WARNINGS, OOD HYPEROPT VERSION", args.compute_prefix, "\033[0m")

                model_path = args.to_destination / "best_model.pth"
                model = GatedABMIL(embed_dim=self.config.embed_dim).cuda()
                if args.spec_norm_bound is not None:
                    model = convert_to_sn_my(model, args.spec_norm_replace_list, args.spec_norm_bound)
                if args.gaussian_process:
                    GP_KWARGS = GP_KWARGS_UNI if args.model_name == "uni" else GP_KWARGS_CONCH
                    GP_KWARGS["num_inducing"] = args.gp_num_inducing if args.gp_num_inducing is not None else GP_KWARGS[
                        "num_inducing"]
                    replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS)
                print(model_path)
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.eval()

                if self.args.mask_tile:
                    mask_pkl = load_pickle_model(self.config.mask_func(args.model_name, args.ood_dataset_name))[f"t{i}f{j}"]
                    if self.args.mask_tile_category == "rand" or self.args.mask_tile_category == "in_slide":  # float
                        mask_tile_threshold = self.args.mask_tile_threshold
                    else:
                        mask_tile_threshold = mask_pkl["val_quantile_list"][self.args.mask_tile_threshold]
                    print("Test Quantile", mask_pkl["val_quantile_list"])
                    print("Removing  val tiles with quantile larger than", mask_tile_threshold)
                else:
                    mask_pkl = None
                    mask_tile_threshold = None
                ood_dataset = OODH5Dataset(Path(self.config.h5_path), None,
                                           s_to_p_mapping=load_pickle_model(self.config.pkl_path)["itop"],
                                           mask_pkl=mask_pkl,
                                           mask_tile_category=self.args.mask_tile_category,
                                           mask_tile_threshold=mask_tile_threshold,
                                           invert_threshold=self.args.invert_threshold,
                                           )
                print(f"{self.args.ood_dataset_name} dataset size: {len(ood_dataset)}")

                ood_loader = DataLoader(ood_dataset, batch_size=self.config.batch_size,
                                        num_workers=self.config.num_workers,
                                        collate_fn=collate_fn, shuffle=False)
                evaluate(model, ood_loader, args, i, j, self.args.ood_dataset_name)
                tbar.update(1)

    def evaluate_ood(self):
        tbar = tqdm(self.trial*self.fold, desc="Generating OOD data")
        auroc_list, aupr_list = [], []
        for i in range(self.trial):
            for j in range(self.fold):
                args.to_destination = args.save_destination / args.compute_prefix /f"{args.training_dataset}" / f"t{i}f{j}"
                print("\033[1;32;40m", "WARNINGS, OOD HYPEROPT VERSION", args.compute_prefix, "\033[0m")
                test_type = "itest" if args.training_dataset == "tcga" else "etest"
                if args.mask_tile:
                    save_itest_df_name = f"patient_predictions_{test_type}_t{i}f{j}_mask{args.mask_tile_category}_thres{args.mask_tile_threshold}_invert{int(args.invert_threshold)}_nearby_tiles{args.nearby_tiles}.parquet.gzip"
                    save_ood_df_name = f"patient_predictions_{self.args.ood_dataset_name}_t{i}f{j}_mask{args.mask_tile_category}_thres{args.mask_tile_threshold}_invert{int(args.invert_threshold)}_nearby_tiles{args.nearby_tiles}.parquet.gzip"
                else:
                    save_itest_df_name = f"patient_predictions_{test_type}_t{i}f{j}.parquet.gzip"
                    save_ood_df_name = f"patient_predictions_{self.args.ood_dataset_name}_t{i}f{j}.parquet.gzip"

                itest_path = args.to_destination / save_itest_df_name

                if self.args.ood_dataset_name == "unified":
                    ood_df = load_all_ood_list(args, i, j, ["ucs", "uvm", "acc", "blca"])
                else:
                    ood_path = args.to_destination / save_ood_df_name
                    ood_df = pd.read_parquet(ood_path)

                itest_df = pd.read_parquet(itest_path)
                if args.ood_detection_type == "uncertainty":
                    iid_u, ood_u = itest_df["Outcome 0-uncertainty0"].values,  ood_df["Outcome 0-uncertainty0"].values
                else:
                    iid_u, ood_u = 1 - np.max(itest_df[["prob_0", "prob_1"]].values, axis=1),  1 - np.max(ood_df[["prob_0", "prob_1"]].values, axis=1)
                    # iid_u, ood_u = 1 - np.max(softmax(itest_df[["Outcome 0-y_pred0", "Outcome 0-y_pred1"]].values, axis=1), axis=1),  1 - np.max(softmax(ood_df[["Outcome 0-y_pred0", "Outcome 0-y_pred1"]].values, axis=1), axis=1)
                    # print("here")
                label = np.concatenate([np.zeros_like(iid_u), np.ones_like(ood_u)])
                pred = np.concatenate([iid_u, ood_u])
                auroc_list.append(roc_auc_score(label, pred))
                precision, recall, _ = precision_recall_curve(label, pred)
                aupr_list.append(auc(recall, precision))
                tbar.update(1)
        print(f"Dataset {self.args.ood_dataset_name} AUROC: {np.mean(auroc_list)} +- {np.std(auroc_list)}, "
              f"AUPR: {np.mean(aupr_list)} +- {np.std(aupr_list)}")
        return auroc_list, aupr_list


def evaluate_cp_performance_print(args, compute_prefix):
    #this part is to evalute the CP performance and the fairness of the model
    def save_data_to_csv_foundation(args, data, compute_prefix, cp_more_info=None):
        results_file_path = args.csv_destination / args.results_file_path
        rows = []
        for run, metrics in data.items():
            row = {'tag': run}
            for condition, levels in metrics.items():
                for level, stats in levels.items():
                    for stat_name, value in stats.items():
                        col_name = f"{condition}_{level}_{stat_name}_list"
                        row[col_name] = value
            rows.append(row)
        df = pd.DataFrame(rows)
        summary_metrics = df.groupby(lambda x: 0).agg(list)
        summary_metrics = summary_metrics.reset_index(drop=True)
        summary_metrics["tag"] = compute_prefix
        if args.force_middle_prefix is not None:
            summary_metrics["hyperopt"] = args.force_middle_prefix
        summary_metrics["ispecial_prefix"] = args.ispecial_prefix
        summary_metrics["especial_prefix"] = args.especial_prefix
        summary_metrics["nearby_tiles"] = args.nearby_tiles

        if cp_more_info is not None:
            cp_more_info = cp_more_info["patient"]
            for dataset_type in cp_more_info.keys():
                for alpha in cp_more_info[dataset_type].keys():
                    for stats in cp_more_info[dataset_type][alpha].keys():
                        if stats == "qhat_permutation_list":
                            continue
                        col_name = f"patient_{alpha}_{dataset_type}_{stats}"
                        summary_metrics[col_name] = str(cp_more_info[dataset_type][alpha][stats])
                        summary_metrics[col_name] = summary_metrics[col_name].apply(literal_eval)


        for col in summary_metrics.columns:
            if "list" in col:
                summary_metrics[col.replace("_list", "")] = summary_metrics[col].apply(
                    lambda x: compute_mean_std_new(x, 5))


        if results_file_path.exists():
            existing_data = pd.read_csv(results_file_path)
            updated_data = pd.concat([existing_data, summary_metrics],
                                     ignore_index=True)  # Concatenating with ignore_index=True
            updated_data.to_csv(results_file_path, index=False)
        else:
            summary_metrics.to_csv(results_file_path, index=False)

    tpr, alpha_list = 0.95, [0.05, 0.01]
    if args.model_name == "prov-gigapath" or args.model_name == "titan":
        init_evaluator = GigaPathFoundationEvaluator(args, set_tpr=tpr, compute_prefix=compute_prefix,
                                         force_alpha_list=alpha_list, level_list=["patient"])
    else:
        init_evaluator = FoundationEvaluator(args, set_tpr=tpr, compute_prefix=compute_prefix,
                                             force_alpha_list=alpha_list, level_list=["patient"])
    output = init_evaluator.compute_internal_and_external_conformal_metrics(fairness=False,
                                                                                          return_full_info=True)
    save_data_to_csv_foundation(args, output[0], compute_prefix, output[1])



def evaluate_ood_performance(args, compute_prefix):
    ######################################## OOD Evaluation ################################################
    foo_evaluator = FoundationOODEvaluator(args=args)
    if args.generate_ood and args.ood_dataset_name != "unified":
        foo_evaluator.generate_ood_data()
    ood_auroc, ood_aupr = foo_evaluator.evaluate_ood()
    ood_auroc_mean, ood_auroc_std = np.mean(ood_auroc), np.std(ood_auroc)
    ood_aupr_mean, ood_aupr_std = np.mean(ood_aupr), np.std(ood_aupr)

    summary_metrics = pd.DataFrame({"tag": [compute_prefix]*2,
                                    "metrics": [f"{args.ood_dataset_name}_auroc", f"{args.ood_dataset_name}_aupr"],
                                    "mean-std": [(np.round(ood_auroc_mean, 4).astype(str) + "+-" + np.round(ood_auroc_std, 4).astype(str)),
                                                 (np.round(ood_aupr_mean, 4).astype(str) + "+-" + np.round(ood_aupr_std, 4).astype(str))], })
    summary_metrics = summary_metrics.pivot(index='tag', columns='metrics', values='mean-std').reset_index()
    if args.mask_tile is False:
        mask_ratio = "no_mask"
    else:
        mask_ratio = f"mask_category{args.mask_tile_category}_thres{args.mask_tile_threshold}"
    summary_metrics["mask_ratio"] = mask_ratio
    summary_metrics["nearby_tiles"] = args.nearby_tiles
    summary_metrics["ood_detection_type"] = args.ood_detection_type
    summary_metrics["dataset"] = args.training_dataset
    summary_metrics[f"{args.ood_dataset_name}_auroc_list"] = [ood_auroc]
    summary_metrics[f"{args.ood_dataset_name}_aupr_list"] = [ood_aupr]

    print(summary_metrics)
    ####################################### OOD Evaluation ################################################
    results_file_path = args.csv_destination / args.results_file_path
    if results_file_path.exists():
        existing_data = pd.read_csv(results_file_path)
        # Check for matches based on tag, mask_ratio, and dataset
        for index, row in summary_metrics.iterrows():
            match = ((existing_data['tag'] == row['tag']) &
                     (existing_data['mask_ratio'] == row['mask_ratio']) &
                     (existing_data['dataset'] == row['dataset']) &
                     (existing_data['ood_detection_type'] == row['ood_detection_type']))

            if match.any():
                if f"{args.ood_dataset_name}_auroc_list" not in existing_data.columns:
                    existing_data[f"{args.ood_dataset_name}_auroc_list"] = pd.Series(dtype=object)
                if f"{args.ood_dataset_name}_aupr_list" not in existing_data.columns:
                    existing_data[f"{args.ood_dataset_name}_aupr_list"] = pd.Series(dtype=object)

                existing_data.loc[match, f"{args.ood_dataset_name}_auroc"] = row[f"{args.ood_dataset_name}_auroc"]
                existing_data.loc[match, f"{args.ood_dataset_name}_aupr"] = row[f"{args.ood_dataset_name}_aupr"]
                existing_data.loc[match, f"{args.ood_dataset_name}_auroc_list"] = str(row[f"{args.ood_dataset_name}_auroc_list"])
                existing_data.loc[match, f"{args.ood_dataset_name}_aupr_list"] = str(row[f"{args.ood_dataset_name}_aupr_list"])
                existing_data.loc[match, f"ood_detection_type"] = row["ood_detection_type"]
            else:
                existing_data = pd.concat([existing_data, row.to_frame().T], ignore_index=True)

        existing_data.to_csv(results_file_path, index=False)
    else:
        summary_metrics.to_csv(results_file_path, index=False)



def evalaute_ood_prov_gigapath(args, folds: int = 5):
    def load_all_ood_list_prov(args: Any, fold: int, ood_exp_name_list: list = None):
        ood_df_list = []
        for ood_exp_name in ood_exp_name_list:
            if args.mask_tile:
                save_ood_df_name = f'{ood_exp_name}_pair_fold{fold}_masktile1_thres{args.mask_tile_threshold}.parquet.gzip'
            else:
                save_ood_df_name = f"{ood_exp_name}_pair_fold{fold}.parquet.gzip"

            ood_df = pd.read_parquet(args.to_destination / save_ood_df_name)
            ood_df_list.append(ood_df)
        return pd.concat(ood_df_list, axis=0)

    assert args.ood_dataset_name == "unified"
    args.save_destination = Path(args.save_destination)
    tbar = tqdm(folds, desc="Generating OOD data")
    auroc_list, aupr_list = [], []
    factor_ci = stats.t.ppf((1 + 0.95) / 2, 19) / np.sqrt(20)

    for fold in range(folds):
        args.to_destination = args.save_destination / f"fold_{fold}"
        if args.mask_tile_threshold is not None:
            test_csv_name = f'test_pair_fold{fold}_masktile1_thres{args.mask_tile_threshold}.parquet.gzip'
        else:
            test_csv_name = f'test_pair_fold{fold}.parquet.gzip'
        itest_df = pd.read_parquet(args.save_destination / f"fold_{fold}" / test_csv_name)
        ood_df = load_all_ood_list_prov(args, fold, ["ucs", "uvm", "acc", "blca"])

        if args.ood_detection_type == "uncertainty":
            iid_u, ood_u = itest_df["Outcome 0-uncertainty0"].values,  ood_df["Outcome 0-uncertainty0"].values
        else:
            iid_u, ood_u = 1 - np.max(itest_df[["prob_0", "prob_1"]].values, axis=1),  1 - np.max(ood_df[["prob_0", "prob_1"]].values, axis=1)

        label = np.concatenate([np.zeros_like(iid_u), np.ones_like(ood_u)])
        pred = np.concatenate([iid_u, ood_u])
        auroc_list.append(roc_auc_score(label, pred))
        precision, recall, _ = precision_recall_curve(label, pred)
        aupr_list.append(auc(recall, precision))
        tbar.update(1)
    print(f"Dataset {args.ood_dataset_name} AUROC: {np.mean(auroc_list)} +- {np.std(auroc_list)*factor_ci}, "
          f"AUPR: {np.mean(aupr_list)} +- {np.std(aupr_list)*factor_ci}")
    return auroc_list, aupr_list


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_extended_abmil_params()
    args.save_destination = Path(args.save_destination)
    print(args)
    seed_everything(args.seed)
    if args.spec_norm_bound is None:
        compute_prefix = f"{args.model_name}"
    else:
        compute_prefix = f"{args.model_name}_sn{args.spec_norm_bound}_gp{int(args.gaussian_process)}"
        # make this print function colorful
        print("\033[1;32;40m", "WARNINGS, HYPEROPT VERSION", compute_prefix, "\033[0m")
    if args.force_middle_prefix is not None:
        compute_prefix = args.force_middle_prefix
        print("force middle prefix", compute_prefix)
    args.compute_prefix = compute_prefix

    if args.eval_type == "cp":
        evaluate_cp_performance_print(args=args, compute_prefix=compute_prefix)
    elif args.eval_type == "ood":
        if args.model_name == "prov-gigapath" or args.model_name == "titan":
            evalaute_ood_prov_gigapath(args=args, folds=args.trial*args.fold)
        else:
            evaluate_ood_performance(args, compute_prefix=compute_prefix)
    else:
        raise NotImplementedError(f"Only support class, cp, ood but get {args.eval_type}",)

