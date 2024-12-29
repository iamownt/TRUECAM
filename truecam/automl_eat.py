import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import glob
from tqdm import tqdm
import os
from PIL import Image
import pickle
import timm
import h5py
from functools import partial
import gc
from ABMIL import load_pickle_model, save_pickle_data
import pandas as pd
import copy
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
from ABMIL import seed_everything, save_pickle_data
import argparse
from collections import defaultdict
from easydict import EasyDict as edict
from scipy.special import softmax
from torchvision.transforms import v2
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor
from train_eat import evaluate_eat_patient_metric, binary_search_for_best_threshold, create_ambiguity_dict, load_lr_dataset_and_predict
from sklearn.decomposition import PCA
from autogluon.core.metrics import make_scorer
from train_eat import evaluate_tile_average_accuracy


def get_eat_lr_params():
    parser = argparse.ArgumentParser(description='ABMIL on downstream tasks')
    parser.add_argument('--trial',                  type=int,  default=4,  help='Set trial')
    parser.add_argument('--fold',                   type=int,  default=5,  help='Set fold')
    parser.add_argument('--start_trial',            type=int,  default=0,  help='Set Start trial')
    parser.add_argument('--start_fold',             type=int,  default=0,  help='Set Start trial')
    parser.add_argument('--sample_fraction',        type=float,  default=0.01,  help='Set fold')

    parser.add_argument('--seed',                   type=int,  default=2024,  help='Random seed')
    parser.add_argument('--model_name',             type=str,  default="uni", help='Model name')
    parser.add_argument('--save_mask_tile',       action='store_true', help='whether to save tiles')
    parser.add_argument('--evaluate_only',       action='store_true', help='whether to evaluate the model')
    parser.add_argument('--ood_dataset_name', type=str, default="ucs", choices=["ucs", "uvm", "acc", "blca"], help='An additional argument')
    parser.add_argument('--generate_ood',        action='store_true', help='If set True, will generate the ood data')
    parser.add_argument('--pca',                 action='store_true', help='whether to use pca')
    parser.add_argument('--pca_dim',             type=int, default=None, help='pca dimension')
    parser.add_argument('--tuning_method',       type=int, default=0, help='tuning method')
    parser.add_argument('--save_destination',       type=str, default="/home/user/sngp/UniConch/models/", help='Model and parquet save path')
    args = parser.parse_args()
    args.save_destination = Path(args.save_destination)
    if args.pca and args.pca_dim is None:
        raise ValueError("PCA dimension is not set")
    if args.generate_ood:
        assert args.evaluate_only is True, "Generate ood data should be used with evaluate only mode"
    return parser, args



def create_slide_int_to_patient_mapping(df):
    slide_int_to_patient = dict(zip(df["slide_int"], df["patient"]))
    return slide_int_to_patient


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

def load_lr_dataset(dataset_h5, filter_mapping, sample_size=None, sample_fraction=None, mapping=None, embed_dim=1024):
    h5_paths = os.listdir(dataset_h5)
    h5_paths = [dataset_h5 / path for path in h5_paths if int(path[:-3]) in filter_mapping.keys()]
    h5_labels = [filter_mapping[int(h5_path.name[:-3])] for h5_path in h5_paths]

    # memory_list = []
    # label_list = []
    tag, tag_p = [], []
    tbar = tqdm(h5_paths, desc="Reading h5 files")
    features, labels = np.empty((10000000, embed_dim), dtype=np.float16), np.empty((10000000,), dtype=np.int8)
    start_idx = 0
    for i, path in enumerate(h5_paths):
        assets = read_assets_from_h5(path)["tile_embeds"]
        if sample_fraction is not None:
            select_bool = np.random.rand(len(assets)) < sample_fraction
            assets = assets[select_bool]
        features[start_idx:start_idx + len(assets)] = assets
        labels[start_idx:start_idx + len(assets)] = h5_labels[i]
        tag.append(len(assets) * [path.stem])
        if mapping is not None:
            tag_p.append(len(assets)*[mapping[int(path.stem)]])
        start_idx += len(assets)
        tbar.update(1)
    features = features[:start_idx]
    labels = labels[:start_idx]
    # features = np.concatenate(memory_list, axis=0)
    # labels = np.concatenate(label_list, axis=0)
    tag = np.concatenate(tag, axis=0)
    if len(tag_p) > 0:
        tag_p = np.concatenate(tag_p, axis=0)
    print("features.shape", features.shape, "labels.shape", labels.shape, "sample_size", sample_size)
    if sample_size is None:
        base_output = tuple([features, labels, tag])
        if len(tag_p) > 0:
            base_output += (tag_p,)
    base_output = tuple([features[:sample_size], labels[:sample_size], tag[:sample_size]])
    if len(tag_p) > 0:
        base_output += (tag_p[:sample_size],)
    return base_output



class UniLRConfig:
    dataset_h5 = Path("/home/user/sngp/UniConch/uni_tcga_h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    etest_h5 = Path("/home/user/sngp/UniConch/uni_cptac_h5file")
    external_dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labels.csv")

    embed_dim = 1024
    label_dict = {"LUAD": 0, "LUSC": 1}

class ConchLRConfig:
    dataset_h5 = Path("/home/user/sngp/UniConch/conch_tcga_h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    etest_h5 = Path("/home/user/sngp/UniConch/conch_cptac_h5file")
    external_dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labels.csv")

    embed_dim = 512
    label_dict = {"LUAD": 0, "LUSC": 1}


class ProvConfig:
    dataset_h5 = Path("/home/user/sngp/project/destination_20X/h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    etest_h5 = Path("/home/user/sngp/project/cptac_destination_20X/h5file")
    external_dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labels.csv")

    embed_dim = 1536
    label_dict = {"LUAD": 0, "LUSC": 1}


class TITANConfig:
    dataset_h5 = Path("/home/user/sngp/project/titan_destination_20X/h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv")
    etest_h5 = None
    external_dataset_csv = None

    embed_dim = 768
    label_dict = {"LUAD": 0, "LUSC": 1}

class TITANCPTACConfig:
    dataset_h5 = Path("/home/user/sngp/project/titan_cptac_destination_20X/h5file")
    dataset_csv = Path("/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labelsv2.csv")
    etest_h5 = None
    external_dataset_csv = None

    embed_dim = 768
    label_dict = {"LUAD": 0, "LUSC": 1}

def load_lr_config(model_name):
    if model_name == "uni":
        return UniLRConfig
    elif model_name == "conch":
        return ConchLRConfig
    elif model_name == "prov-gigapath":
        return ProvConfig
    elif model_name == "titan":
        return TITANConfig
    elif model_name == "titan-cptac":
        return TITANCPTACConfig
    else:
        raise ValueError("Model name not found")


def memory_restrict_amb_generate_function(args):
    """if the machine can not train on full data, this function can post generate ambiguity score for the dataset"""
    seed_everything(args.seed)
    config = load_lr_config(args.model_name)
    if config.etest_h5 is None or config.external_dataset_csv is None:
        has_etest = False
        print("Warning: No external test set")
    else:
        has_etest = True
    itest_ambiguity_dict = defaultdict(lambda: defaultdict(list))
    etest_ambiguity_dict = defaultdict(lambda: defaultdict(list))
    metric_dict = defaultdict(list)
    for i in range(args.trial):
        for j in range(args.fold):
            print(f"Trial {i}, Fold {j}")
            df = pd.read_csv(config.dataset_csv)
            df["cohort"] = df["cohort"].apply(lambda x: config.label_dict[x])
            if has_etest:
                external_df = pd.read_csv(config.external_dataset_csv)
                external_df["cohort"] = external_df["cohort"].apply(lambda x: config.label_dict[x])
            split_key = f"split{args.fold * i + j}"

            train_df, val_df, test_df = df[df[split_key] == "train"], df[df[split_key] == "val"], df[df[split_key] == "test"]
            train_df_mapping = train_df.set_index("slide_int")["cohort"].to_dict()
            val_df_mapping = val_df.set_index("slide_int")["cohort"].to_dict()
            test_df_mapping = test_df.set_index("slide_int")["cohort"].to_dict()
            i_s_to_p_mapping = create_slide_int_to_patient_mapping(df)

            if has_etest:
                external_df_mapping = external_df.set_index("slide_int")["cohort"].to_dict()
                e_s_to_p_mapping = create_slide_int_to_patient_mapping(external_df)

            save_gluon_model_path = Path("/home/user/wangtao/prov-gigapath/AutogluonModels/")
            save_gluon_model_path = save_gluon_model_path / f"{args.model_name}_t{i}f{j}_frac{args.sample_fraction}_tuning{args.tuning_method}_seed{args.seed}"

            predictor = TabularPredictor.load(str(save_gluon_model_path))

            train_predictions, train_labels, train_tag = load_lr_dataset_and_predict(config.dataset_h5, train_df_mapping, predictor, np_input=False)
            val_predictions, val_labels, val_tag = load_lr_dataset_and_predict(config.dataset_h5, val_df_mapping, predictor, np_input=False)
            test_predictions, test_labels, test_tag = load_lr_dataset_and_predict(config.dataset_h5, test_df_mapping, predictor, np_input=False)
            print("train_predictions", train_predictions.shape, "val_predictions", val_predictions.shape)
            print("test_predictions", test_predictions.shape)
            if has_etest:
                etest_predictions, etest_labels, etest_tag, etest_tagp = load_lr_dataset_and_predict(config.etest_h5,
                                                                                                     external_df_mapping,
                                                                                                     predictor,
                                                                                                     np_input=False,
                                                                                                     mapping=e_s_to_p_mapping)
                print("etest_predictions", etest_predictions.shape)
            train_ambiguity_dict, train_quantile_list = create_ambiguity_dict(train_predictions, train_tag)
            val_ambiguity_dict, val_quantile_list = create_ambiguity_dict(val_predictions, val_tag)

            itest_ambiguity_dict[f"t{i}f{j}"].update(train_ambiguity_dict)
            itest_ambiguity_dict[f"t{i}f{j}"].update(val_ambiguity_dict)
            itest_ambiguity_dict[f"t{i}f{j}"]["train_quantile_list"] = train_quantile_list
            itest_ambiguity_dict[f"t{i}f{j}"]["val_quantile_list"] = val_quantile_list

            test_ambiguity_dict, _ = create_ambiguity_dict(test_predictions, test_tag)
            itest_ambiguity_dict[f"t{i}f{j}"].update(test_ambiguity_dict)
            if has_etest:
                etest_ambiguity_dict_g, _ = create_ambiguity_dict(etest_predictions, etest_tag)
                etest_ambiguity_dict[f"t{i}f{j}"].update(etest_ambiguity_dict_g)

            best_threshold, best_accuracy = binary_search_for_best_threshold(val_quantile_list[0], val_quantile_list[-1], val_predictions, val_labels, val_tag)


            itest_acc, itest_auc, itest_eat_acc, itest_eat_auc, proportion = (
                evaluate_eat_patient_metric(best_threshold, test_predictions, test_labels, test_tag))
            print(f"Itest Patient Accuracy = {itest_acc:.3f}, Patient AUC = {itest_auc:.3f}")
            print(
                f"Itest EAT Patient Accuracy = {itest_eat_acc:.3f}, EAT Patient AUC = {itest_eat_auc:.3f}, Removing {proportion}")
            if has_etest:
                etest_acc, etest_auc, etest_eat_acc, etest_eat_auc, etest_proportion = (
                    evaluate_eat_patient_metric(best_threshold, etest_predictions, etest_labels, etest_tagp))
                print(f"Etest Patient Accuracy = {etest_acc:.3f}, Patient AUC = {etest_auc:.3f}")
                print(
                    f"Etest EAT Patient Accuracy = {etest_eat_acc:.3f}, EAT Patient AUC = {etest_eat_auc:.3f}, Removing {etest_proportion}")
                metric_dict["etest_acc"].append(etest_acc)
                metric_dict["etest_auc"].append(etest_auc)
                metric_dict["etest_eat_acc"].append(etest_eat_acc)
                metric_dict["etest_eat_auc"].append(etest_eat_auc)
            metric_dict["itest_acc"].append(itest_acc)
            metric_dict["itest_auc"].append(itest_auc)
            metric_dict["itest_eat_acc"].append(itest_eat_acc)
            metric_dict["itest_eat_auc"].append(itest_eat_auc)

    if args.save_mask_tile:
        save_pickle_data(edict(itest_ambiguity_dict), args.save_destination / "ambpkl" / "newambk" / f"{args.model_name}_itest_ambiguity_dict_autogluon_{args.sample_fraction}_tuning{args.tuning_method}.pkl")
        if has_etest:
            save_pickle_data(edict(etest_ambiguity_dict),
                             args.save_destination / "ambpkl" / "newambk" / f"{args.model_name}_etest_ambiguity_dict_autogluon_{args.sample_fraction}_tuning{args.tuning_method}.pkl")

    df_metrics = pd.DataFrame(metric_dict)
    summary_metrics = df_metrics.agg(['mean', 'std']).transpose()
    print(summary_metrics)


def memory_restrict_amb_generate_ood_function(args):
    """if the machine can not train on full data, this function can post generate ambiguity score for the dataset"""
    from evaluate_everything import load_ood_config
    seed_everything(args.seed)
    config = load_ood_config(model_name=args.model_name, ood_dataset=args.ood_dataset_name)
    ood_ambiguity_dict = defaultdict(lambda: defaultdict(list))
    for i in range(args.trial):
        for j in range(args.fold):
            print(f"Trial {i}, Fold {j}")
            save_gluon_model_path = Path("/home/user/wangtao/prov-gigapath/AutogluonModels/")
            save_gluon_model_path = save_gluon_model_path / f"{args.model_name}_t{i}f{j}_frac{args.sample_fraction}_tuning{args.tuning_method}_seed{args.seed}"
            predictor = TabularPredictor.load(str(save_gluon_model_path))

            predictions, labels, tag = load_lr_dataset_and_predict(Path(config.h5_path),
                                                                   load_pickle_model(config.pkl_path)["itop"],
                                                                   predictor,
                                                                   np_input=False)
            print("predictions", predictions.shape, "labels", labels.shape)

            ambiguity_dict, quantile_list = create_ambiguity_dict(predictions, tag)

            ood_ambiguity_dict[f"t{i}f{j}"].update(ambiguity_dict)
            ood_ambiguity_dict[f"t{i}f{j}"]["val_quantile_list"] = quantile_list
    try:
        if args.save_mask_tile:
            save_pickle_data(edict(ood_ambiguity_dict), args.save_destination / "ambpkl" / "newambk" / f"{args.model_name}_{args.ood_dataset_name}_ambiguity_dict_autogluon_{args.sample_fraction}_tuning{args.tuning_method}.pkl")
            pass
    except Exception as e:
        print(e)
        print(ood_ambiguity_dict)
        print(ood_ambiguity_dict.keys())
    print("OOD Generation Finished")



if __name__ == "__main__":
    _, args = get_eat_lr_params()
    print(args)
    if args.evaluate_only and not args.generate_ood:
        memory_restrict_amb_generate_function(args)
        exit(0)

    if args.evaluate_only and args.generate_ood:
        memory_restrict_amb_generate_ood_function(args)
        exit(0)

    seed_everything(args.seed)
    config = load_lr_config(args.model_name)
    if config.etest_h5 is None or config.external_dataset_csv is None:
        has_etest = False
        print("Warning: No external test set")
    else:
        has_etest = True
    itest_ambiguity_dict = defaultdict(lambda: defaultdict(list))
    etest_ambiguity_dict = defaultdict(lambda: defaultdict(list))
    metric_dict = defaultdict(list)
    for i in range(args.start_trial, args.trial):
        for j in range(args.start_fold, args.fold):
            print(f"Trial {i}, Fold {j}")
            df = pd.read_csv(config.dataset_csv)
            df["cohort"] = df["cohort"].apply(lambda x: config.label_dict[x])
            if has_etest:
                external_df = pd.read_csv(config.external_dataset_csv)
                external_df["cohort"] = external_df["cohort"].apply(lambda x: config.label_dict[x])
            split_key = f"split{args.fold * i + j}"

            train_df, val_df, test_df = df[df[split_key] == "train"], df[df[split_key] == "val"], df[df[split_key] == "test"]
            train_df_mapping = train_df.set_index("slide_int")["cohort"].to_dict()
            val_df_mapping = val_df.set_index("slide_int")["cohort"].to_dict()
            test_df_mapping = test_df.set_index("slide_int")["cohort"].to_dict()
            i_s_to_p_mapping = create_slide_int_to_patient_mapping(df)

            if has_etest:
                external_df_mapping = external_df.set_index("slide_int")["cohort"].to_dict()
                e_s_to_p_mapping = create_slide_int_to_patient_mapping(external_df)

            train_features, train_labels, train_tag = load_lr_dataset(config.dataset_h5, train_df_mapping,
                                                                      sample_size=None, sample_fraction=args.sample_fraction, embed_dim=config.embed_dim)
            if args.pca:
                pca = PCA(n_components=args.pca_dim)
            train_features = pca.fit_transform(train_features) if args.pca else train_features
            val_features, val_labels, val_tag = load_lr_dataset(config.dataset_h5, val_df_mapping,
                                                                sample_size=None, sample_fraction=0.01, embed_dim=config.embed_dim)
            val_features = pca.transform(val_features) if args.pca else val_features
            test_features, test_labels, test_tag = load_lr_dataset(config.dataset_h5, test_df_mapping,
                                                                   sample_size=None, sample_fraction=0.01, embed_dim=config.embed_dim)
            test_features = pca.transform(test_features) if args.pca else test_features
            train_data, val_data, test_data = (pd.DataFrame(train_features), pd.DataFrame(val_features),
                                                           pd.DataFrame(test_features))

            if has_etest:
                etest_features, etest_labels, etest_tag, etest_tagp = load_lr_dataset(config.etest_h5, external_df_mapping,
                                                                                      sample_size=None, sample_fraction=0.08,
                                                                                      mapping=e_s_to_p_mapping, embed_dim=config.embed_dim)
                etest_features = pca.transform(etest_features) if args.pca else etest_features
                etest_data = pd.DataFrame(etest_features)
                del etest_features
                etest_data['label'] = etest_labels
            del train_features, val_features, test_features
            gc.collect()
            train_data['label'], val_data['label'], test_data['label'], = train_labels, val_labels, test_labels

            hyperparameters = {
                'NN_TORCH': {},
                'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, 'GBMLarge'],
                'CAT': {},
                'XGB': {},
                # 'FASTAI': {},
            }
            if args.tuning_method == 0:
                tuning_data = val_data
                tuning_tag = val_tag
            else:
                tuning_data = etest_data
                tuning_tag = etest_tagp

            # def confusion_removal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            #     y_temp = np.stack([1 - y_pred, y_pred], axis=1)
            #
            #     try:
            #         _, accuracy = binary_search_for_best_threshold(None, None,
            #                                                        y_temp, y_true,
            #                                                        tuning_tag, print_str=False)
            #     except Exception as e:
            #         _, accuracy = binary_search_for_best_threshold(None, None,
            #                                                        y_temp, y_true,
            #                                                        etest_tagp, print_str=False)
            #     return accuracy
            # patient_accuracy_fn = make_scorer(name='patient_accuracy',
            #                                   score_func=confusion_removal,
            #                                   optimum=1,
            #                                   needs_proba=True,
            #                                   greater_is_better=True)

            save_gluon_model_path = Path("/home/user/wangtao/prov-gigapath/AutogluonModels/")
            save_gluon_model_path = save_gluon_model_path / f"{args.model_name}_t{i}f{j}_frac{args.sample_fraction}_tuning{args.tuning_method}_seed{args.seed}"
            excluded_model_types = ['KNN'] # eval_metric=patient_accuracy_fn,). parameters in the init func
            predictor = TabularPredictor(label='label', path=str(save_gluon_model_path)).fit(train_data,
                                                                                            tuning_data=tuning_data,
                                                                                            excluded_model_types=excluded_model_types,
                                                                                            num_gpus=1, hyperparameters=hyperparameters)
            # train_predictions = predictor.predict_proba(train_data.drop(columns=['label'])).values
            val_predictions = predictor.predict_proba(val_data.drop(columns=['label'])).values
            test_predictions = predictor.predict_proba(test_data.drop(columns=['label'])).values
            print("test_predictions", test_predictions.shape)
            # performance = predictor.evaluate(test_data)
            # print("Test", performance)
            if has_etest:
                etest_predictions = predictor.predict_proba(etest_data.drop(columns=['label'])).values
                performance = predictor.evaluate(etest_data)
                print("External Test", performance)

            # train_ambiguity_dict, train_quantile_list = create_ambiguity_dict(train_predictions, train_tag)
            val_ambiguity_dict, val_quantile_list = create_ambiguity_dict(val_predictions, val_tag)

            # itest_ambiguity_dict[f"t{i}f{j}"].update(train_ambiguity_dict)
            itest_ambiguity_dict[f"t{i}f{j}"].update(val_ambiguity_dict)
            # itest_ambiguity_dict[f"t{i}f{j}"]["train_quantile_list"] = train_quantile_list
            itest_ambiguity_dict[f"t{i}f{j}"]["val_quantile_list"] = val_quantile_list

            test_ambiguity_dict, _ = create_ambiguity_dict(test_predictions, test_tag)
            itest_ambiguity_dict[f"t{i}f{j}"].update(test_ambiguity_dict)
            if has_etest:
                etest_ambiguity_dict_g, etest_quantile_list = create_ambiguity_dict(etest_predictions, etest_tag, start_quantile=0.1, end_quantile=0.9)
                etest_ambiguity_dict[f"t{i}f{j}"].update(etest_ambiguity_dict_g)

            best_threshold, best_accuracy = binary_search_for_best_threshold(val_quantile_list[0], val_quantile_list[-1], val_predictions, val_labels, val_tag)
            # best_threshold, best_accuracy = binary_search_for_best_threshold(etest_quantile_list[0], etest_quantile_list[-1], etest_predictions, etest_labels, etest_tagp)


            itest_acc, itest_auc, itest_eat_acc, itest_eat_auc, proportion = (
                evaluate_eat_patient_metric(best_threshold, test_predictions, test_labels, test_tag))
            print(f"Itest Patient Accuracy = {itest_acc:.3f}, Patient AUC = {itest_auc:.3f}")
            print(
                f"Itest EAT Patient Accuracy = {itest_eat_acc:.3f}, EAT Patient AUC = {itest_eat_auc:.3f}, Removing {proportion}")
            if has_etest:
                etest_acc, etest_auc, etest_eat_acc, etest_eat_auc, etest_proportion = (
                    evaluate_eat_patient_metric(best_threshold, etest_predictions, etest_labels, etest_tagp))
                print(f"Etest Patient Accuracy = {etest_acc:.3f}, Patient AUC = {etest_auc:.3f}")
                print(
                    f"Etest EAT Patient Accuracy = {etest_eat_acc:.3f}, EAT Patient AUC = {etest_eat_auc:.3f}, Removing {etest_proportion}")
                metric_dict["etest_acc"].append(etest_acc)
                metric_dict["etest_auc"].append(etest_auc)
                metric_dict["etest_eat_acc"].append(etest_eat_acc)
                metric_dict["etest_eat_auc"].append(etest_eat_auc)
            metric_dict["itest_acc"].append(itest_acc)
            metric_dict["itest_auc"].append(itest_auc)
            metric_dict["itest_eat_acc"].append(itest_eat_acc)
            metric_dict["itest_eat_auc"].append(itest_eat_auc)
    if args.save_mask_tile:
        save_pickle_data(edict(itest_ambiguity_dict), args.save_destination / "ambpkl" / f"{args.model_name}_itest_ambiguity_dict_autogluon_{args.sample_fraction}.pkl")
        if has_etest:
            save_pickle_data(edict(etest_ambiguity_dict), args.save_destination / "ambpkl" / f"{args.model_name}_etest_ambiguity_dict_autogluon_{args.sample_fraction}.pkl")
    df_metrics = pd.DataFrame(metric_dict)
    summary_metrics = df_metrics.agg(['mean', 'std']).transpose()
    print(summary_metrics)


