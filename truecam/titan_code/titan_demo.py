import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
# from scipy.special import softmax
# from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
# import matplotlib.pyplot as plt
# from pathlib import Path
import pickle
import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# import torch.optim as optim
# import h5py
# import random
# from collections import defaultdict
# from sklearn.metrics import roc_auc_score
# import argparse
# # warnings.filterwarnings('error')
# import operator
# import wandb
# from functools import partial
# import copy
import h5py
from huggingface_hub import login
from transformers import AutoModel
from titan.eval_linear_probe import train_and_evaluate_logistic_regression_with_val
from sklearn.model_selection import train_test_split
from pathlib import Path
from sngp_wrapper.covert_utils import replace_layer_with_gaussian



class MapOodSlideToPatient:

    def __init__(self):
        ood_dataset_list = ["acc", "blca", "ucs", "uvm"]
        mapping_path = "/home/user/sngp/UniConch/ood_pkl_folder"
        for ood_dataset in ood_dataset_list:
            with open(f"{mapping_path}/{ood_dataset}_mapping.pkl", 'rb') as file:
                current_mapping = pickle.load(file)
            stoi = current_mapping["stoi"] # slide to int
            itop = current_mapping["itop"] # int to patient
            # construct the mapping from slide to patient
            slide_to_patient = {}
            for slide, int_id in stoi.items():
                slide_to_patient[slide] = itop[int_id]
            setattr(self, f"{ood_dataset}_slide_to_patient", slide_to_patient)

    def map_slide_to_patient(self, slide_id, ood_dataset_name):
        return getattr(self, f"{ood_dataset_name}_slide_to_patient")[slide_id]


GP_KWARGS_TITAN = {
    'num_inducing': 768,
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

class TitanGaussianProcess(nn.Module):
    
    def __init__(self):
        super(TitanGaussianProcess, self).__init__()
        self.classifier = nn.Linear(768, 1)
        
    def forward(self, x, **kwargs):
        return self.classifier(x, **kwargs)
    
    

def titan_demo_function():
    model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    conch, eval_transform = model.return_conch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # load TCGA sample data
    from huggingface_hub import hf_hub_download
    demo_h5_path = hf_hub_download(
        "MahmoodLab/TITAN",
        filename="TCGA_demo_features/TCGA-PC-A5DK-01Z-00-DX1.C2D3BC09-411F-46CF-811B-FDBA7C2A295B.h5",
    )
    file = h5py.File(demo_h5_path, 'r')
    features = torch.from_numpy(file['features'][:])
    coords = torch.from_numpy(file['coords'][:])
    patch_size_lv0 = file['coords'].attrs['patch_size_level0']

    # extract slide embedding
    with torch.autocast('cuda', torch.float16), torch.inference_mode():
        features = features.to(device)
        coords = coords.to(device)
        slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)

    print("slide_embedding", slide_embedding.shape)


def load_titan_slide_embed_on_our_setting(folds: int = 5):
    tcga_slide_path = "/home/user/sngp/tcga_slides/slides"
    tcga_label_path = "/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv"
    tcga_slide_list = os.listdir(tcga_slide_path)
    tcga_slide_name = [slide.strip('.svs') for slide in tcga_slide_list]
    titan_official_pkl = '/home/user/.cache/huggingface/hub/models--MahmoodLab--TITAN/snapshots/b2fb4f475256eb67c6e9ccbf2d6c9c3f25f20791/TCGA_TITAN_features.pkl'
    with open(titan_official_pkl, 'rb') as file:
        data = pickle.load(file)
    embeddings_df = pd.DataFrame({'slide_id': data['filenames'], 'embeddings': list(data['embeddings'][:])})
    embedding_df = embeddings_df[embeddings_df['slide_id'].isin(tcga_slide_name)]

    print(embedding_df.shape)
    print(embedding_df.head())
    #
    lung_labels = pd.read_csv(tcga_label_path)
    print(lung_labels["slide"].iloc[0])
    lung_labels["cohort"] = lung_labels["cohort"].apply(lambda x: {"LUAD": 0, "LUSC": 1}[x])
    slide_name_to_label_zip = dict(zip(lung_labels["slide"], lung_labels["cohort"]))
    # train the model
    cross_validation_dict = {}

    for fold in range(folds):
        train_df = lung_labels[lung_labels["split{}".format(fold)] == "train"]
        val_df = lung_labels[lung_labels["split{}".format(fold)] == "val"]
        test_df = lung_labels[lung_labels["split{}".format(fold)] == "test"]
        train_slide_id, val_slide_id, test_slide_id = train_df["slide"], val_df["slide"], test_df["slide"]

        train_data_titan_format = embedding_df[embedding_df["slide_id"].isin(train_slide_id)]
        val_data_titan_format = embedding_df[embedding_df["slide_id"].isin(val_slide_id)]
        test_data_titan_format = embedding_df[embedding_df["slide_id"].isin(test_slide_id)]
        train_labels = train_data_titan_format["slide_id"].apply(lambda x: slide_name_to_label_zip[x]).values
        val_labels = val_data_titan_format["slide_id"].apply(lambda x: slide_name_to_label_zip[x]).values
        test_labels = test_data_titan_format["slide_id"].apply(lambda x: slide_name_to_label_zip[x]).values
        test_slide_id = test_data_titan_format["slide_id"]
        test_patient_id = test_slide_id
    

        train_embedding = np.stack(train_data_titan_format["embeddings"].values)
        val_embedding = np.stack(val_data_titan_format["embeddings"].values)
        test_embedding = np.stack(test_data_titan_format["embeddings"].values)
        # print(train_embedding.shape)
        # log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
        results, outputs = train_and_evaluate_logistic_regression_with_val(train_embedding, train_labels, val_embedding,
                                                                           val_labels, test_embedding, test_labels, test_slide_id, test_patient_id,
                                                                           log_spaced_values=None)
        # to use the default setting from our paper use the default value for searching C (log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45))
        # results = train_and_evaluate_logistic_regression_with_val(train_data, train_labels, val_data, val_labels, test_data, test_labels)
        # for key, value in results.items():
        #     print(f"{key.split('/')[-1]: <12}: {value:.4f}")
        cross_validation_dict[fold] = results
    # average the results of the fold
    for key in cross_validation_dict[0].keys():
        print(key, np.mean([cross_validation_dict[i][key] for i in range(folds)]))


def load_titan_custom_slide_embed_on_our_setting(folds: int = 5, keep_ratio: float = None, save_dir = None,
                                                 generate_ood: bool = False, evalute_and_save_uncertainty: bool = True,
                                                 dataset: str = "tcga"):
    """ for eat, we should know that when we apply eat to the test, we should not have data leak
        therefore, the slide embedding should be generated from each folds.
    """
    label_path_dict = {
        "tcga": "/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/nsclc_labels.csv",
        "cptac": "/home/user/wangtao/prov-gigapath/dataset_csv/nsclc/external_nsclc_labelsv2.csv"
    }
    titan_official_pkl_dict = {
        "tcga": '/home/user/sngp/project/titan_destination_20X/h5file_slide/tcga_nsclc_titan_slide_embedding.pkl',
        "cptac": "/home/user/sngp/project/titan_cptac_destination_20X/h5file_slide/cptac_nsclc_titan_slide_embedding.pkl"
    }

    eat_embedding_path_dict = {
        "tcga": f"/home/user/sngp/project/titan_destination_20X/h5file_slide_eat{keep_ratio}",
        "cptac": f"/home/user/sngp/project/titan_cptac_destination_20X/h5file_slide_eat{keep_ratio}",
    }
    if dataset == "cptac" and keep_ratio != 0.4:
        eat_embedding_path_dict["cptac"] = f"/home/user/sngp/project/titan_cptac_destination_20X/h5file_slide_eat{keep_ratio}"

    lung_labels = pd.read_csv(label_path_dict[dataset])
    slide_name = lung_labels["slide"].to_list()
    ind_slide_to_patient_map = dict(zip(lung_labels["slide"], lung_labels["patient"]))
    if keep_ratio is None:
        titan_official_pkl = titan_official_pkl_dict[dataset]
        with open(titan_official_pkl, 'rb') as file:
            data = pickle.load(file)
        embeddings_df = pd.DataFrame({'slide_id': data['filenames'], 'embeddings': list(data['embeddings'][:]),
                                      "patient_id": [ind_slide_to_patient_map[slide] for slide in data['filenames']]})
        embedding_df = embeddings_df[embeddings_df['slide_id'].isin(slide_name)]
        print(embedding_df.shape)
        print(embedding_df.head())
        # normalized embedding
        embedding_df["embeddings"] = embedding_df["embeddings"].apply(lambda x: x / np.linalg.norm(x))
    else:
        print("load eat embedding from each trial and fold")
    #
    lung_labels["cohort"] = lung_labels["cohort"].apply(lambda x: {"LUAD": 0, "LUSC": 1}[x])
    slide_name_to_label_zip = dict(zip(lung_labels["slide"], lung_labels["cohort"]))
    # train the model
    cross_validation_dict = {}
    load_embedding_path = Path(eat_embedding_path_dict[dataset])
    for fold in range(folds):
        i, j = fold // 5, fold % 5
        if keep_ratio is not None:
            titan_official_pkl = load_embedding_path / f't{i}f{j}/{dataset}_nsclc_titan_slide_embedding.pkl'
            with open(titan_official_pkl, 'rb') as file:
                data = pickle.load(file)
            embeddings_df = pd.DataFrame({'slide_id': data['filenames'], 'embeddings': list(data['embeddings'][:]),
                                          "patient_id": [ind_slide_to_patient_map[slide] for slide in data['filenames']]})
            embedding_df = embeddings_df[embeddings_df['slide_id'].isin(slide_name)]
            embedding_df["embeddings"] = embedding_df["embeddings"].apply(lambda x: x / np.linalg.norm(x))


        train_df = lung_labels[lung_labels["split{}".format(fold)] == "train"]
        val_df = lung_labels[lung_labels["split{}".format(fold)] == "val"]
        test_df = lung_labels[lung_labels["split{}".format(fold)] == "test"]
        train_slide_id, val_slide_id, test_slide_id = train_df["slide"], val_df["slide"], test_df["slide"]


        train_data_titan_format = embedding_df[embedding_df["slide_id"].isin(train_slide_id)]
        val_data_titan_format = embedding_df[embedding_df["slide_id"].isin(val_slide_id)]
        test_data_titan_format = embedding_df[embedding_df["slide_id"].isin(test_slide_id)]
        train_labels = train_data_titan_format["slide_id"].apply(lambda x: slide_name_to_label_zip[x]).values
        val_labels = val_data_titan_format["slide_id"].apply(lambda x: slide_name_to_label_zip[x]).values
        test_labels = test_data_titan_format["slide_id"].apply(lambda x: slide_name_to_label_zip[x]).values
        truly_train_slide_id, truly_val_slide_id, truly_test_slide_id = (train_data_titan_format["slide_id"],
                                                                         val_data_titan_format["slide_id"], test_data_titan_format["slide_id"])
        truly_train_patient_id, truly_val_patient_id, truly_test_patient_id = (train_data_titan_format["patient_id"],
                                                                                val_data_titan_format["patient_id"],
                                                                                test_data_titan_format["patient_id"])

        train_embedding = np.stack(train_data_titan_format["embeddings"].values)
        val_embedding = np.stack(val_data_titan_format["embeddings"].values)
        test_embedding = np.stack(test_data_titan_format["embeddings"].values)
        print(train_embedding.shape, val_embedding.shape, test_embedding.shape)

        results, outputs = train_and_evaluate_logistic_regression_with_val(train_embedding, train_labels, val_embedding,
                                                                           val_labels, test_embedding, test_labels,
                                                                           truly_test_slide_id, truly_test_patient_id,
                                                                           log_spaced_values=None)
        if evalute_and_save_uncertainty:
            model = TitanGaussianProcess()
            replace_layer_with_gaussian(container=model, signature="classifier", **GP_KWARGS_TITAN)
            model.classifier.reset_covariance_matrix()
            kwargs = {'return_random_features': False, 'return_covariance': False,
                      'update_precision_matrix': True, 'update_covariance_matrix': False}
            _ = model(torch.from_numpy(train_embedding), **kwargs) # update the precision metrix
            # udpate the covariance matrix
            model.classifier.update_covariance_matrix()
            eval_kwargs = {'return_random_features': False, 'return_covariance': True,
                           'update_precision_matrix': False, 'update_covariance_matrix': False}
            output = model(torch.from_numpy(test_embedding), **eval_kwargs)
            _, covariance = output
            uncertainty = torch.diagonal(covariance).numpy()
            outputs["uncertainty"] = uncertainty
        else:
            outputs["uncertainty"] = np.zeros(test_labels.shape)
            
            

        cross_validation_dict[fold] = results
        if save_dir is not None:
            save_dir = Path(save_dir)

            def create_csv(outputs, save_dir, test_csv_name, slide_id, patient_id):
                """Outcome 0-y_pred0  Outcome 0-y_pred1  Outcome 0-y_true    prob_0    prob_1"""
                pair_csv = pd.DataFrame({
                    'slide_id': slide_id.values,
                    "patient": patient_id.values,
                    'Outcome 0-uncertainty0': outputs["uncertainty"],
                })
                if len(np.unique(patient_id)) == len(slide_id):
                    pair_csv["Outcome 0-y_pred0"] = 1 - outputs["probs"]
                    pair_csv["Outcome 0-y_pred1"] = outputs["probs"]
                    pair_csv["prob_0"] = 1 - outputs["probs"]
                    pair_csv["prob_1"] = outputs["probs"]
                    pair_csv["Outcome 0-y_true"] = outputs["targets"]
                else:
                    # in ood scenario, the prob has not been aggregate
                    if len(outputs["probs"]) != len(np.unique(patient_id)):
                        pair_csv["Outcome 0-y_pred0"] = 1 - outputs["probs"]
                        pair_csv["Outcome 0-y_pred1"] = outputs["probs"]
                        pair_csv["prob_0"] = 1 - outputs["probs"]
                        pair_csv["prob_1"] = outputs["probs"]
                        pair_csv["Outcome 0-y_true"] = outputs["targets"]
                        pair_csv = pair_csv.groupby("patient").agg(
                            {'Outcome 0-uncertainty0': 'mean', 'Outcome 0-y_true': 'first', 'Outcome 0-y_pred0': 'mean',
                             'Outcome 0-y_pred1': 'mean', 'prob_0': 'mean', 'prob_1': 'mean'}).reset_index()

                    else:
                        pair_csv = pair_csv.groupby("patient").agg(
                            {'Outcome 0-uncertainty0': 'mean'}).reset_index()
                        print(f"aggregation from {len(slide_id)} to {len(pair_csv)}")
                        pair_csv["Outcome 0-y_pred0"] = 1 - outputs["probs"]
                        pair_csv["Outcome 0-y_pred1"] = outputs["probs"]
                        pair_csv["prob_0"] = 1 - outputs["probs"]
                        pair_csv["prob_1"] = outputs["probs"]
                        pair_csv["Outcome 0-y_true"] = outputs["targets"]

                (save_dir / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
                pair_csv.to_parquet(save_dir / f"fold_{fold}" / test_csv_name, compression='gzip')

            mask_tile = 1 if keep_ratio is not None else 0

            if generate_ood:
                stop_ood_mapper = MapOodSlideToPatient()
                lr_model = outputs["lr_model"]
                for ood_dataset in ["blca", "uvm", "ucs", "acc"]:
                    slide_ood_path = ood_path_f(ood_dataset, keep_ratio=keep_ratio, fold=fold)
                    with open(slide_ood_path, 'rb') as file:
                        slide_ood_data = pickle.load(file)
                    ood_embedding_df = pd.DataFrame(
                        {'slide_id': slide_ood_data['filenames'], 'embeddings': list(slide_ood_data['embeddings'][:])})
                    ood_embedding_df["embeddings"] = ood_embedding_df["embeddings"].apply(
                        lambda x: x / np.linalg.norm(x))
                    ood_prediction = lr_model.predict_proba(np.stack(ood_embedding_df["embeddings"].values, axis=0))
                    if evalute_and_save_uncertainty:
                        output = model(torch.from_numpy(np.stack(ood_embedding_df["embeddings"].values, axis=0)), **eval_kwargs)
                        _, covariance = output
                        ood_uncertainty = torch.diagonal(covariance).numpy()
                        ood_outputs = {
                            "probs": ood_prediction[:, 1],
                            "targets": np.ones(ood_prediction.shape[0]),
                            'uncertainty': ood_uncertainty,
                        }
                    
                    if keep_ratio:
                        test_csv_name = f"{ood_dataset}_pair_fold{fold}_masktile{int(mask_tile)}_thres{keep_ratio}.parquet.gzip"
                    else:
                        test_csv_name = f'{ood_dataset}_pair_fold{fold}.parquet.gzip'
                    if evalute_and_save_uncertainty:
                        create_csv(ood_outputs, save_dir, test_csv_name, ood_embedding_df["slide_id"],
                                   ood_embedding_df["slide_id"].apply(lambda x: stop_ood_mapper.map_slide_to_patient(x, ood_dataset)))
            else:
                if keep_ratio:
                    test_csv_name = f'test_pair_fold{fold}_masktile{int(mask_tile)}_thres{keep_ratio}.parquet.gzip'
                else:
                    test_csv_name = f'test_pair_fold{fold}.parquet.gzip'
                outputs["probs"] =  outputs["probs"].astype(np.float32)
                create_csv(outputs, save_dir, test_csv_name, truly_test_slide_id, truly_test_patient_id)


    for key in cross_validation_dict[0].keys():
        print(key, np.mean([cross_validation_dict[i][key] for i in range(folds)]))

def ood_path_f(ood_dataset_name: str = "ucs", keep_ratio: float = None, fold: int = 0):
    i, j = fold // 5, fold % 5
    if keep_ratio is None:
        return f"/home/user/sngp/UniConch/titan_{ood_dataset_name}_h5file_slide/tcga_{ood_dataset_name}_titan_slide_embedding.pkl"
    return f"/home/user/sngp/UniConch/titan_{ood_dataset_name}_h5file_slide_eat{keep_ratio}/t{i}f{j}/{ood_dataset_name}_nsclc_titan_slide_embedding.pkl"


# load_titan_slide_embed_on_our_setting(folds=20)


# generate the ind dataset

dataset = "tcga" # assess on the tcga and cptac dataset
save_dir = f"/home/user/wangtao/prov-gigapath/TITAN/outputs/{dataset}/nsclc"
# os.makedirs(save_dir, exist_ok=True)
load_titan_custom_slide_embed_on_our_setting(folds=20, keep_ratio=None, dataset=dataset, save_dir=save_dir,
                                             generate_ood=True, evalute_and_save_uncertainty=True)


# generate ood for slide_embedding
# load_titan_custom_slide_embed_on_our_setting(folds=5, keep_ratio=None,
#                                              save_dir="/home/user/wangtao/prov-gigapath/TITAN/outputs/nsclc",
#                                              generate_ood=True, evalute_and_save_uncertainty=True)














