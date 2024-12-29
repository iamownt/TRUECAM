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
import pyDeepInsight


try:
    from cuml.linear_model import LogisticRegression
except Exception as e:
    print(e)
    from sklearn.linear_model import LogisticRegression


def get_eat_lr_params():
    parser = argparse.ArgumentParser(description='ABMIL on downstream tasks')
    parser.add_argument('--trial',                  type=int,  default=1,  help='Set trial')
    parser.add_argument('--fold',                   type=int,  default=5,  help='Set fold')
    parser.add_argument('--sample_fraction',        type=float,  default=0.1,  help='Set fold')
    parser.add_argument('--seed',                   type=int,  default=2024,  help='Random seed')
    parser.add_argument('--model_name',             type=str,  default="uni", help='Model name')
    parser.add_argument('--classifier',             type=str,  default="convnext", help='Model name')
    parser.add_argument('--save_mask_tile',       action='store_true', help='whether to save tiles')
    parser.add_argument('--save_destination',       type=str, default="/home/user/sngp/UniConch/models/", help='Model and parquet save path')

    args = parser.parse_args()
    args.save_destination = Path(args.save_destination)
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

def load_lr_dataset_and_predict(dataset_h5, filter_mapping, classifier, np_input: bool = True, mapping: dict = None):
    h5_paths = os.listdir(dataset_h5)
    h5_paths = [dataset_h5 / path for path in h5_paths if int(path[:-3]) in filter_mapping.keys()]
    h5_labels = [filter_mapping[int(h5_path.name[:-3])] for h5_path in h5_paths]
    tag, tag_p = [], []
    predictions = []
    label_list = []
    for i, path in tqdm(enumerate(h5_paths), desc="Reading h5 files", total=len(h5_paths)):
        assets = read_assets_from_h5(path)["tile_embeds"]
        if np_input:
            prediction = classifier.predict_proba(assets)
        else:
            prediction = classifier.predict_proba(pd.DataFrame(assets)).values
        predictions.append(prediction)
        label_list.append(len(assets)*[h5_labels[i]])
        tag.append(len(assets)*[path.stem])
        if mapping is not None:
            tag_p.append(len(assets)*[mapping[int(path.stem)]])

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(label_list, axis=0)
    tag = np.concatenate(tag, axis=0)
    if len(tag_p) > 0:
        tag_p = np.concatenate(tag_p, axis=0)
        print("tag_p agg", tag_p.shape)
        return predictions, labels, tag, tag_p
    return predictions, labels, tag


def evaluate_tile_average_accuracy(predictions, labels, tag, level="tile", metric="accuracy", print_str=True):
    '''Evaluate the average accuracy of the classifier on the features'''
    df = pd.DataFrame({"tag": tag, "label": labels, "prediction_0": predictions[:, 0], "prediction_1": predictions[:, 1]})
    if level != "tile":
        agg_dict = {
            'prediction_0': 'mean',
            'prediction_1': 'mean',
            'label': 'first',
        }
        df = df.groupby('tag').agg(agg_dict).reset_index()
    if print_str:
        print("agg shape", df.shape)
    if metric == "accuracy":
        outcome = np.mean((df["label"].values == df[["prediction_0", "prediction_1"]].values.argmax(axis=1)))
    elif metric == "auroc":
        outcome = roc_auc_score(df["label"].values, df["prediction_1"].values)
    return outcome


# def search_for_best_removing(amb_thres, val_predictions, val_labels, val_tag, metric="accuracy"):
#     val_ambiguity = 1 - np.max(val_predictions, axis=1)
#     filter_bool = val_ambiguity < amb_thres
#     return evaluate_tile_average_accuracy(val_predictions[filter_bool], val_labels[filter_bool],
#                                           val_tag[filter_bool], level="patient", metric=metric), 1 - filter_bool.mean()

def search_for_best_removing(amb_thres, val_predictions, val_labels, val_tag, metric="accuracy", print_str=True):
    val_ambiguity = 1 - np.max(val_predictions, axis=1)
    filter_bool = val_ambiguity < amb_thres

    # Ensure full tag removal
    tag_filter = val_tag[filter_bool]
    tags_to_remove = set(val_tag) - set(tag_filter)
    complete_filter_bool = np.isin(val_tag, list(tags_to_remove))
    complete_filter_bool = complete_filter_bool | filter_bool

    return evaluate_tile_average_accuracy(val_predictions[complete_filter_bool], val_labels[complete_filter_bool],
                                          val_tag[complete_filter_bool], level="patient",
                                          metric=metric, print_str=print_str), 1 - complete_filter_bool.mean()

def binary_search_for_best_threshold(start, end, val_predictions, val_labels, val_tag, metric="accuracy", print_str=True):
    if start is None and end is None:
        temp_prob = 1 - np.max(val_predictions, axis=1)
        start = np.quantile(temp_prob, 0.01)
        end = np.quantile(temp_prob, 0.98)
    best_threshold = start
    best_accuracy = 0
    while end - start > 1e-3:
        mid = (start + end) / 2
        accuracy, remove_proportion = search_for_best_removing(mid, val_predictions, val_labels, val_tag, metric, print_str)
        if print_str:
            print(f"Threshold {mid}, Accuracy {accuracy}, Remove Proportion {remove_proportion}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = mid
            end = mid
        else:
            start = mid

    return best_threshold, best_accuracy

class LogisticRegressionCV:
    def __init__(self, cv=5, **kwargs):
        self.cv = cv
        self.classifiers = []
        self.kwargs = kwargs

    def fit(self, X, y):
        kf = KFold(n_splits=self.cv)
        tbar = tqdm(kf.split(X), desc="Fitting classifiers")
        TRAIN_PROB = np.zeros((X.shape[0], len(np.unique(y))))
        for train_index, val_index in tbar:
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            classifier = LogisticRegression(**self.kwargs)
            classifier.fit(X_train, y_train)
            self.classifiers.append(copy.deepcopy(classifier))
            TRAIN_PROB[val_index] = classifier.predict_proba(X_val)

            del X_train, X_val, y_train, y_val, classifier
            gc.collect()
        del X, y
        gc.collect()
        return TRAIN_PROB

    def predict_proba(self, X):
        probas = np.mean([clf.predict_proba(X) for clf in self.classifiers], axis=0)
        return probas

class ConvNextClassifier:

    def __init__(self, num_classes, embed_dim):
        self.model = torchvision.models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.IMAGENET1K_V1")
        # self.model = torchvision.models.convnext_base(weights="ConvNeXt_Base_Weights.IMAGENET1K_V1")
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)
        self.model.requires_grad_(False)
        self.model.features[-6:].requires_grad_(True)
        self.model.classifier.requires_grad_(True)
        print("grad parameter", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        self.embed_dim = embed_dim
        self.epochs = 5
        self.tabular_transform = None

    def preprocess_input(self, X, y):
        if self.tabular_transform is None:
            self.tabular_transform = pyDeepInsight.ImageTransformer(feature_extractor='tsne', discretization='bin', pixels=(32, 32))
            self.tabular_transform.fit(X, y)
            X = self.tabular_transform.transform(X)
        else:
            X = self.tabular_transform.transform(X)
        # if self.embed_dim == 512:
        #     X = np.concatenate([X, np.zeros((X.shape[0], 512), dtype=np.float16)], axis=1).astype(np.float16)

        # X = torch.from_numpy(X).reshape(-1, 1, 32, 32).repeat(1, 3, 1, 1).half()
        X = torch.from_numpy(X).half().permute(0, 3, 1, 2)
        y = torch.from_numpy(y).long()
        print(X.shape, y.shape)
        return X, y


    def fit(self, X_train, y_train, X_train_tag, X_val, y_val, X_val_tag):
        X_train, y_train = self.preprocess_input(X_train, y_train)
        X_val, y_val = self.preprocess_input(X_val, y_val)

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers=0, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, num_workers=0, shuffle=False)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(train_loader)))
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0
        self.model.cuda()
        scaler = torch.cuda.amp.GradScaler()
        cutmix = v2.CutMix(num_classes=2)
        mixup = v2.MixUp(num_classes=2)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
        for epoch in range(self.epochs):
            self.model.train()

            for i, (inputs, labels) in enumerate(tqdm(train_loader)):
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = cutmix_or_mixup(inputs, labels)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()


            train_acc = self.evaluate(train_loader, X_train_tag, level="tile")
            val_acc = self.evaluate(val_loader, X_val_tag, level="tile")
            print(f"Epoch {epoch}, train_acc {train_acc} val_acc {val_acc}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_model = copy.deepcopy(self.model)

        self.model = self.best_model
        del X_train, y_train, X_val, y_val, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, loader, X_val_tag, level="patient"):
        self.model.eval()
        predictions_list = []
        label_list = []
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                predictions_list.append(outputs.detach().cpu().numpy())
                label_list.append(labels.cpu().numpy())
        predictions = np.concatenate(predictions_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        tag = X_val_tag
        df = pd.DataFrame(
            {"tag": tag, "label": labels, "prediction_0": predictions[:, 0], "prediction_1": predictions[:, 1]})
        if level != "tile":
            agg_dict = {
                'prediction_0': 'mean',
                'prediction_1': 'mean',
                'label': 'first',
            }
            df = df.groupby('tag').agg(agg_dict).reset_index()
        print("agg shape", df.shape)
        outcome = np.mean((df["label"].values == df[["prediction_0", "prediction_1"]].values.argmax(axis=1)))
        return outcome

    def predict_proba(self, x):
        x, _ = self.preprocess_input(x, np.zeros(x.shape[0]))
        self.model.eval()
        predictions = []
        dataset = torch.utils.data.TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=0, shuffle=False)
        with torch.no_grad():
            for inputs in tqdm(loader):
                inputs = inputs[0].cuda()
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    outputs = nn.Softmax(dim=1)(outputs)
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions, axis=0)



class MLPClassifier:
    def __init__(self, num_classes, embed_dim, hidden_dim=512):
        self.model = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.input_dim = embed_dim
        self.num_classes = num_classes
        self.epochs = 1
        self.scaler = None

    def preprocess_input(self, X, fit_scaler=False):
        X = X.astype(np.float16)
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        X = torch.from_numpy(X).half()
        return X

    def fit(self, X_train, y_train, X_train_tag, X_val, y_val, X_val_tag):
        X_train = self.preprocess_input(X_train, fit_scaler=True)
        X_val = self.preprocess_input(X_val, fit_scaler=False)
        y_train = torch.from_numpy(y_train).long()
        y_val = torch.from_numpy(y_val).long()

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.epochs * len(train_loader)))

        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0
        self.model.cuda()
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.epochs):
            self.model.train()

            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.cuda(), labels.cuda()

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            val_acc = self.evaluate(val_loader, X_val_tag)
            print(f"Epoch {epoch}, Val Accuracy {val_acc}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_model = copy.deepcopy(self.model)

        self.model = self.best_model
        del X_train, y_train, X_val, y_val, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()

    def evaluate(self, loader, X_val_tag, level="patient"):
        self.model.eval()
        predictions_list = []
        label_list = []
        for inputs, labels in tqdm(loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                predictions_list.append(outputs.detach().cpu().numpy())
                label_list.append(labels.cpu().numpy())
        predictions = np.concatenate(predictions_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        tag = X_val_tag
        df = pd.DataFrame(
            {"tag": tag, "label": labels, "prediction_0": predictions[:, 0], "prediction_1": predictions[:, 1]})
        if level != "tile":
            agg_dict = {
                'prediction_0': 'mean',
                'prediction_1': 'mean',
                'label': 'first',
            }
            df = df.groupby('tag').agg(agg_dict).reset_index()
        print("agg shape", df.shape)
        outcome = np.mean((df["label"].values == df[["prediction_0", "prediction_1"]].values.argmax(axis=1)))
        return outcome

    def predict_proba(self, x):
        x = self.preprocess_input(x, fit_scaler=False)
        self.model.eval()
        predictions = []
        dataset = torch.utils.data.TensorDataset(x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1024, num_workers=0, shuffle=False)
        with torch.no_grad():
            for inputs in tqdm(loader):
                inputs = inputs[0].cuda()
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    outputs = nn.Softmax(dim=1)(outputs)
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions, axis=0)



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

def load_lr_config(model_name):
    if model_name == "uni":
        return UniLRConfig
    elif model_name == "conch":
        return ConchLRConfig
    else:
        raise ValueError("Model name not found")


def evaluate_eat_patient_metric(best_threshold, predictions, labels, tag):
    patient_acc = evaluate_tile_average_accuracy(predictions, labels, tag, level="patient",
                                                 metric="accuracy")
    patient_auc = evaluate_tile_average_accuracy(predictions, labels, tag, level="patient",
                                                 metric="auroc")
    eat_patient_acc, proportion = search_for_best_removing(best_threshold, predictions, labels, tag,
                                                           metric="accuracy")
    eat_patient_auc, _ = search_for_best_removing(best_threshold, predictions, labels, tag,
                                                  metric="auroc")
    return patient_acc, patient_auc, eat_patient_acc, eat_patient_auc, proportion


def create_ambiguity_dict(predictions, tag, start_quantile=0.1, end_quantile=0.9):
    all_ambiguity = 1 - np.max(predictions, axis=1)
    quantile_list = np.quantile(all_ambiguity, np.linspace(start_quantile, end_quantile, 9))
    print("Create Quantile List from", start_quantile, "to", end_quantile, "with length", len(quantile_list))
    ambiguity_dict = defaultdict(list)
    for tag_id in np.unique(tag):
        tag_bool = tag == tag_id
        ambiguity = 1 - np.max(predictions[tag_bool], axis=1)
        ambiguity_dict[tag_id] = ambiguity
    return ambiguity_dict, quantile_list


if __name__ == "__main__":
    _, args = get_eat_lr_params()
    print(args)
    seed_everything(args.seed)
    config = load_lr_config(args.model_name)
    itest_ambiguity_dict = defaultdict(lambda: defaultdict(list))
    etest_ambiguity_dict = defaultdict(lambda: defaultdict(list))
    metric_dict = defaultdict(list)
    for i in range(args.trial):
        for j in range(args.fold):
            print(f"Trial {i}, Fold {j}")
            df = pd.read_csv(config.dataset_csv)
            external_df = pd.read_csv(config.external_dataset_csv)
            df["cohort"] = df["cohort"].apply(lambda x: config.label_dict[x])
            external_df["cohort"] = external_df["cohort"].apply(lambda x: config.label_dict[x])
            split_key = f"split{args.fold * i + j}"

            train_df, val_df, test_df = df[df[split_key] == "train"], df[df[split_key] == "val"], df[df[split_key] == "test"]
            train_df_mapping = train_df.set_index("slide_int")["cohort"].to_dict()
            val_df_mapping = val_df.set_index("slide_int")["cohort"].to_dict()
            test_df_mapping = test_df.set_index("slide_int")["cohort"].to_dict()
            external_df_mapping = external_df.set_index("slide_int")["cohort"].to_dict()

            i_s_to_p_mapping = create_slide_int_to_patient_mapping(df)
            e_s_to_p_mapping = create_slide_int_to_patient_mapping(external_df)

            train_features, train_labels, train_tag = load_lr_dataset(config.dataset_h5, train_df_mapping,
                                                                      sample_size=None, sample_fraction=args.sample_fraction, embed_dim=config.embed_dim)
            val_features, val_labels, val_tag = load_lr_dataset(config.dataset_h5, val_df_mapping,
                                                                sample_size=None, sample_fraction=args.sample_fraction, embed_dim=config.embed_dim)
            test_features, test_labels, test_tag = load_lr_dataset(config.dataset_h5, test_df_mapping,
                                                                   sample_size=None, sample_fraction=args.sample_fraction, embed_dim=config.embed_dim)
            etest_features, etest_labels, etest_tag, etest_tagp = load_lr_dataset(config.etest_h5, external_df_mapping,
                                                                                  sample_size=None, sample_fraction=args.sample_fraction,
                                                                                  mapping=e_s_to_p_mapping, embed_dim=config.embed_dim)
            if args.classifier == "lr":
                classifier = LogisticRegressionCV(random_state=0, C=0.316, max_iter=1000, verbose=1)
                train_predictions = classifier.fit(train_features, train_labels)  # this is for kfold val prediction
            elif args.classifier == "convnext":
                classifier = ConvNextClassifier(num_classes=2, embed_dim=config.embed_dim)
                classifier.fit(train_features, train_labels, train_tag, etest_features, etest_labels, etest_tagp)
                train_predictions = classifier.predict_proba(train_features)
            elif args.classifier == "mlp":
                classifier = MLPClassifier(num_classes=2, embed_dim=config.embed_dim)
                classifier.fit(train_features, train_labels, train_tag, val_features, val_labels, val_tag)
                train_predictions = classifier.predict_proba(train_features)

            train_ambiguity_dict, train_quantile_list = create_ambiguity_dict(train_predictions, train_tag)
            del train_features, train_labels, train_tag
            gc.collect()
            # Evaluate using the logistic regression classifier
            val_predictions = classifier.predict_proba(val_features)
            val_ambiguity_dict, val_quantile_list = create_ambiguity_dict(val_predictions, val_tag)

            best_threshold, best_accuracy = binary_search_for_best_threshold(val_quantile_list[0], val_quantile_list[-1], val_predictions, val_labels, val_tag)
            itest_ambiguity_dict[f"t{i}f{j}"].update(train_ambiguity_dict)
            itest_ambiguity_dict[f"t{i}f{j}"].update(val_ambiguity_dict)
            itest_ambiguity_dict[f"t{i}f{j}"]["train_quantile_list"] = train_quantile_list
            itest_ambiguity_dict[f"t{i}f{j}"]["val_quantile_list"] = val_quantile_list

            del val_features, val_labels, val_tag
            gc.collect()

            def evaluate_and_clear(ambiguity_dict, i, j, name_space="itest"):
                if name_space == "itest":
                    test_features, test_labels, test_tag = load_lr_dataset(config.dataset_h5, test_df_mapping,
                                                                           sample_size=None, embed_dim=config.embed_dim)
                    test_predictions = classifier.predict_proba(test_features)
                    # test_predictions, test_labels, test_tag = load_lr_dataset_and_predict(config.dataset_h5, test_df_mapping, classifier)

                    patient_acc, patient_auc, eat_patient_acc, eat_patient_auc, proportion = (
                        evaluate_eat_patient_metric(best_threshold, test_predictions, test_labels, test_tag))
                    print(f"Itest Patient Accuracy = {patient_acc:.3f}, Patient AUC = {patient_auc:.3f}")
                    print(
                        f"Itest EAT Patient Accuracy = {eat_patient_acc:.3f}, EAT Patient AUC = {eat_patient_auc:.3f}, Removing {proportion}")
                    test_ambiguity_dict, _ = create_ambiguity_dict(test_predictions, test_tag)
                    ambiguity_dict[f"t{i}f{j}"].update(test_ambiguity_dict)
                    del test_predictions, test_labels, test_tag
                    return patient_acc, patient_auc, eat_patient_acc, eat_patient_auc

                elif name_space == "etest":
                    etest_features, etest_labels, etest_tag, etest_tagp = load_lr_dataset(config.etest_h5, external_df_mapping, sample_size=None,
                                                                              mapping=e_s_to_p_mapping, embed_dim=config.embed_dim)
                    etest_predictions = classifier.predict_proba(etest_features)
                    # etest_predictions, etest_labels, etest_tag = load_lr_dataset_and_predict(config.etest_h5, external_df_mapping, classifier)

                    etest_patient_acc, etest_patient_auc, etest_eat_patient_acc, etest_eat_patient_auc, etest_proportion = (
                        evaluate_eat_patient_metric(best_threshold, etest_predictions, etest_labels, etest_tagp))
                    print(f"Etest Patient Accuracy = {etest_patient_acc:.3f}, Patient AUC = {etest_patient_auc:.3f}")
                    print(
                        f"Etest EAT Patient Accuracy = {etest_eat_patient_acc:.3f}, EAT Patient AUC = {etest_eat_patient_auc:.3f}, Removing {etest_proportion}")
                    etest_ambiguity_dict, _ = create_ambiguity_dict(etest_predictions, etest_tag)
                    ambiguity_dict[f"t{i}f{j}"].update(etest_ambiguity_dict)
                    del etest_predictions, etest_labels, etest_tag
                    return etest_patient_acc, etest_patient_auc, etest_eat_patient_acc, etest_eat_patient_auc
                else:
                    raise ValueError(f"Name space not found, {name_space}")
                gc.collect()
                torch.cuda.empty_cache()


            itest_acc, itest_auc, itest_eat_acc, itest_eat_auc = evaluate_and_clear(itest_ambiguity_dict, i, j, "itest")
            etest_acc, etest_auc, etest_eat_acc, etest_eat_auc = evaluate_and_clear(etest_ambiguity_dict, i, j, "etest")
            metric_dict["itest_acc"].append(itest_acc)
            metric_dict["itest_auc"].append(itest_auc)
            metric_dict["itest_eat_acc"].append(itest_eat_acc)
            metric_dict["itest_eat_auc"].append(itest_eat_auc)
            metric_dict["etest_acc"].append(etest_acc)
            metric_dict["etest_auc"].append(etest_auc)
            metric_dict["etest_eat_acc"].append(etest_eat_acc)
            metric_dict["etest_eat_auc"].append(etest_eat_auc)
    if args.save_mask_tile:
        save_pickle_data(edict(itest_ambiguity_dict), args.save_destination / "ambpkl" / f"{args.model_name}_itest_ambiguity_dict.pkl")
        save_pickle_data(edict(etest_ambiguity_dict), args.save_destination / "ambpkl" / f"{args.model_name}_etest_ambiguity_dict.pkl")
    df_metrics = pd.DataFrame(metric_dict)
    summary_metrics = df_metrics.agg(['mean', 'std']).transpose()
    print(summary_metrics)




