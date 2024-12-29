import os
import torch
import pandas as pd
import numpy as np

from training import train
from params import get_finetune_params
from task_configs.utils import load_task_config
from utils import seed_torch, get_exp_code, get_splits, get_loader, save_obj
from datasets.slide_datatset import SlideDataset
import pickle

def load_pickle_model(path: str = "kmeans_100w.pkl"):
    with open(path, "rb") as file:
        model = pickle.load(file)
    return model

def create_csv(record, save_dir, file_name):
    """Outcome 0-y_pred0  Outcome 0-y_pred1  Outcome 0-y_true    prob_0    prob_1"""
    pair_csv = pd.DataFrame({
        'patient': record["patient"],
        'Outcome 0-uncertainty0': record["uncertainty"],
        "prob_0": record["prob"][:, 0],
        "prob_1": record["prob"][:, 1],
        "Outcome 0-y_true": record["label"].argmax(axis=1),
        "Outcome 0-y_pred0": record["logit"][:, 0],
        "Outcome 0-y_pred1": record["logit"][:, 1]
    })
    if len(record["patient"]) != len(np.unique(record["patient"])):
        pair_csv = pair_csv.groupby("patient").agg(
            {'Outcome 0-uncertainty0': 'mean', 'Outcome 0-y_true': 'first', 'Outcome 0-y_pred0': 'mean',
             'Outcome 0-y_pred1': 'mean', 'prob_0': 'mean', 'prob_1': 'mean'}).reset_index()
        print(f"agg from {len(record['patient'])} to {len(pair_csv)}")

    pair_csv.to_parquet(os.path.join(save_dir, file_name), compression='gzip')


if __name__ == '__main__':
    args = get_finetune_params()
    print(args)

    # set the devicereplace_layer_with_gaussianga
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # set the random seed
    seed_torch(device, args.seed)

    # load the task configuration
    print('Loading task configuration from: {}'.format(args.task_cfg_path))
    args.task_config = load_task_config(args.task_cfg_path)
    print(args.task_config)
    args.task = args.task_config.get('name', 'task')
    
    # set the experiment save directory
    args.save_dir = os.path.join(args.save_dir, args.task, args.exp_name)
    args.model_code, args.task_code, args.exp_code = get_exp_code(args) # get the experiment code
    args.save_dir = os.path.join(args.save_dir, args.exp_code)
    os.makedirs(args.save_dir, exist_ok=True)
    print('Experiment code: {}'.format(args.exp_code))
    print('Setting save directory: {}'.format(args.save_dir))

    # set the learning rate
    eff_batch_size = args.batch_size * args.gc
    if args.lr is None or args.lr < 0:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.gc)
    print("effective batch size: %d" % eff_batch_size)

    # set the split key
    if args.pat_strat:
        args.split_key = 'pat_id'
    else:
        args.split_key = 'slide_id'

    # set up the dataset
    args.split_dir = os.path.join(args.split_dir, args.task_code) if not args.pre_split_dir else args.pre_split_dir
    os.makedirs(args.split_dir, exist_ok=True)
    print('Setting split directory: {}'.format(args.split_dir))
    dataset = pd.read_csv(args.dataset_csv) # read the dataset csv file

    # use the slide dataset
    DatasetClass = SlideDataset

    # set up the results dictionary
    results = {}

    # start cross validation
    # assert args.folds <=5
    fold_to_tf_map = {0: "t0f0", 1: "t0f1", 2: "t0f2", 3: "t0f3", 4: "t0f4",
                      5: "t1f0", 6: "t1f1", 7: "t1f2", 8: "t1f3", 9: "t1f4",
                      10: "t2f0", 11: "t2f1", 12: "t2f2", 13: "t2f3", 14: "t2f4",
                      15: "t3f0", 16: "t3f1", 17: "t3f2", 18: "t3f3", 19: "t3f4"}
    for fold in range(args.start_folds, args.folds):
        # set up the fold directory
        save_dir = os.path.join(args.save_dir, f'fold_{fold}')
        os.makedirs(save_dir, exist_ok=True)
        # get the splits
        split_key = f"split{fold}"
        if args.mask_tile:
            mask_dict = {
                "mask_pkl": load_pickle_model(args.mask_pkl_path)[fold_to_tf_map[fold]],
                "mask_tile": args.mask_tile,
                "mask_tile_category": args.mask_tile_category,
                "mask_tile_threshold": args.mask_tile_threshold,
            }
        else:
            mask_dict = None

        train_data, val_data, test_data = (DatasetClass(dataset, args.root_path, "train", args.task_config, split_key=split_key, mask_dict=mask_dict),
                                           DatasetClass(dataset, args.root_path, "val", args.task_config, split_key=split_key, mask_dict=mask_dict),
                                           DatasetClass(dataset, args.root_path, "test", args.task_config, split_key=split_key, mask_dict=mask_dict))
        args.n_classes = train_data.n_classes
        # get the dataloader
        train_loader, val_loader, test_loader = get_loader(train_data, val_data, test_data, **vars(args))
        # start training
        val_records, test_records = train((train_loader, val_loader, test_loader), fold, args)

        # update the results
        records = {'val': val_records, 'test': test_records}
        if args.evaluate_only:
            records.pop('val')
        for record_ in records:
            for key in records[record_]:
                if 'prob' in key or 'label' in key or 'logit' in key  or key in 'uncertainty' or key in "patient":
                    continue
                key_ = record_ + '_' + key
                if key_ not in results:
                    results[key_] = []
                results[key_].append(records[record_][key])
        if args.mask_tile:
            val_csv_name = f'val_pair_fold{fold}_masktile{int(args.mask_tile)}_thres{args.mask_tile_threshold}.parquet.gzip'
            test_csv_name = f'test_pair_fold{fold}_masktile{int(args.mask_tile)}_thres{args.mask_tile_threshold}.parquet.gzip'
        else:
            val_csv_name = f'val_pair_fold{fold}.parquet.gzip'
            test_csv_name = f'test_pair_fold{fold}.parquet.gzip'
        if not args.evaluate_only:
            create_csv(val_records, save_dir, val_csv_name)
        create_csv(test_records, save_dir, test_csv_name)

    # save the results into a csv file
    if args.mask_tile:
        summary_name = f"summary_masktile{int(args.mask_tile)}_thres{args.mask_tile_threshold}.csv"
    else:
        summary_name = "summary.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.save_dir, summary_name), index=False)

    # print the results, mean and std
    for key in results_df.columns:
        print('{}: {:.4f} +- {:.4f}'.format(key, np.mean(results_df[key]), np.std(results_df[key])))
    print('Results saved in: {}'.format(os.path.join(args.save_dir, 'summary.csv')))
    print('Done!')