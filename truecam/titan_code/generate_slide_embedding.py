import os
import h5py
import torch
import pickle
from pathlib import Path
from transformers import AutoModel
import numpy as np
from tqdm import tqdm


def inference_slide_embedding(titan_tile_path: str = "/home/user/sngp/project/titan_destination_20X/h5file",
                              titan_slide_path: str = "/home/user/sngp/project/titan_destination_20X/h5file_slide"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    model = model.to(device)
    model.eval()



    titan_tile_path = Path(titan_tile_path)
    titan_slide_path = Path(titan_slide_path)
    titan_slide_path.mkdir(parents=True, exist_ok=True)
    patch_size_level0 = torch.tensor([512]).to(device)
    file_path = os.listdir(titan_tile_path)
    for file in tqdm(file_path, desc="Processing Slide embedding"):
        demo_h5_path = titan_tile_path / file
        print(demo_h5_path)
        with h5py.File(demo_h5_path, 'r') as f:
            features = torch.from_numpy(f['tile_embeds'][:]).unsqueeze(dim=0)
            coords = torch.from_numpy(f['coords'][:]).unsqueeze(dim=0).long()
        print("features", features.shape)
        print("coords", coords.shape)

        # extract slide embedding
        with torch.autocast(device_type='cuda', dtype=torch.float16), torch.inference_mode():
            features = features.to(device)
            coords = coords.to(device)
            slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_level0)
            print("slide_embedding", slide_embedding.shape)

        slide_h5_path = titan_slide_path / file
        with h5py.File(slide_h5_path, 'w') as f:
            f.create_dataset('slide_embedding', data=slide_embedding.detach().float().cpu().numpy())

def merge_slide_embedding(pkl_path: str = "/home/user/sngp/project/destination/pickle_data.pkl",
                          titan_slide_path: str = "/home/user/sngp/project/titan_destination_20X/h5file_slide",
                          dataset_name: str = "tcga", subtyping: str = "nsclc"):
    with open(pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)
    invert_pkl_dict = {v: k for k, v in pkl_dict.items()}
    titan_slide_path = Path(titan_slide_path)
    slide_embedding, slide_id = [], []

    for file in os.listdir(titan_slide_path):
        slide_h5_path = titan_slide_path / file
        with h5py.File(slide_h5_path, 'r') as f:
            slide_embedding.append(f['slide_embedding'][:])
            slide_id.append(invert_pkl_dict[int(file[:-3])])
    slide_embedding = np.concatenate(slide_embedding, axis=0)
    with open(titan_slide_path / f"{dataset_name}_{subtyping}_titan_slide_embedding.pkl", "wb") as f:
        pickle.dump({"embeddings": slide_embedding, "filenames": slide_id}, f)


def inference_slide_embedding_eat(trials: int = 1, folds: int = 5,
                                  keep_ratio: float = 0.4,
                                  titan_tile_path: str = "/home/user/sngp/project/titan_destination_20X/h5file",
                                  titan_slide_path: str = "/home/user/sngp/project/titan_destination_20X/h5file_slide_eat",
                                  titan_amb_path: str = "/home/user/sngp/UniConch/models/ambpkl/newambk/titan_itest_ambiguity_dict_autogluon_0.2_tuning0.pkl"):
    with open(titan_amb_path, "rb") as f:
        titan_amb = pickle.load(f)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    model = model.to(device)
    model.eval()


    titan_tile_path = Path(titan_tile_path)
    titan_slide_path = Path(titan_slide_path)
    titan_slide_path.mkdir(parents=True, exist_ok=True)
    patch_size_level0 = torch.tensor([512]).to(device)
    file_path = os.listdir(titan_tile_path)
    for trial in range(trials):
        for fold in range(folds):
            for file in tqdm(file_path, desc="Processing Slide embedding"):
                demo_h5_path = titan_tile_path / file
                print(demo_h5_path)
                with h5py.File(demo_h5_path, 'r') as f:
                    features = torch.from_numpy(f['tile_embeds'][:]).unsqueeze(dim=0)
                    coords = torch.from_numpy(f['coords'][:]).unsqueeze(dim=0).long()
                print("features", features.shape)
                print("coords", coords.shape)
                amb_array = titan_amb[f"t{trial}f{fold}"][file[:-3]]
                in_slide_threshold = np.quantile(amb_array, keep_ratio)
                mask_bool = amb_array <= in_slide_threshold
                # extract slide embedding
                with torch.autocast(device_type='cuda', dtype=torch.float16), torch.inference_mode():
                    features = features.to(device)
                    coords = coords.to(device)
                    features = features[:, mask_bool]
                    coords = coords[:, mask_bool]
                    print("reduce from {} to {}".format(len(mask_bool), len(mask_bool[mask_bool])))
                    slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_level0)
                    print("slide_embedding", slide_embedding.shape)

                slide_h5_path = titan_slide_path / f"t{trial}f{fold}"
                slide_h5_path.mkdir(parents=True, exist_ok=True)
                with h5py.File(slide_h5_path / file, 'w') as f:
                    f.create_dataset('slide_embedding', data=slide_embedding.detach().float().cpu().numpy())

def merge_slide_embedding_eat(trials: int = 1, folds: int = 5,
                              pkl_path: str = "/home/user/sngp/project/destination/pickle_data.pkl",
                              titan_slide_path: str = "/home/user/sngp/project/titan_destination_20X/h5file_slide_eat",
                              dataset_name: str = "tcga", subtyping: str = "nsclc"):
    with open(pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)
    invert_pkl_dict = {v: k for k, v in pkl_dict.items()}
    titan_slide_path = Path(titan_slide_path)
    for trial in range(trials):
        for fold in range(folds):
            slide_embedding, slide_id = [], []
            for file in os.listdir(titan_slide_path / f"t{trial}f{fold}"):
                if "slide_embedding" in file:
                    continue
                slide_h5_path = titan_slide_path / f"t{trial}f{fold}" / file
                with h5py.File(slide_h5_path, 'r') as f:
                    slide_embedding.append(f['slide_embedding'][:])
                    slide_id.append(invert_pkl_dict[int(file[:-3])])

            slide_embedding = np.concatenate(slide_embedding, axis=0)
            with open(titan_slide_path / f"t{trial}f{fold}" / f"{dataset_name}_{subtyping}_titan_slide_embedding.pkl", "wb") as f:
                pickle.dump({"embeddings": slide_embedding, "filenames": slide_id}, f)




if __name__ == "__main__":
    # normal generation
    # inference_slide_embedding(titan_tile_path="/home/user/sngp/project/titan_cptac_destination_20X/h5file",
    #                           titan_slide_path="/home/user/sngp/project/titan_cptac_destination_20X/h5file_slide")
    # merge_slide_embedding(pkl_path="/home/user/sngp/project/cptac_destination/pickle_data.pkl",
    #                         titan_slide_path="/home/user/sngp/project/titan_cptac_destination_20X/h5file_slide",
    #                         dataset_name="cptac", subtyping="nsclc")
    # eat version
    for tune_mask_ratio in [0.4]:
        inference_slide_embedding_eat(trials=4, folds=5, keep_ratio=tune_mask_ratio,
                                      titan_tile_path="/home/user/sngp/project/titan_destination_20X/h5file",
                                      titan_slide_path=f"/home/user/sngp/project/titan_destination_20X/h5file_slide_eat{tune_mask_ratio}",
                                      titan_amb_path="/home/user/sngp/UniConch/models/ambpkl/newambk/titan_itest_ambiguity_dict_autogluon_1.0_tuning0.pkl")
        merge_slide_embedding_eat(trials=4, folds=5, pkl_path=f"/home/user/sngp/project/destination/pickle_data.pkl",
                                  titan_slide_path=f"/home/user/sngp/project/titan_destination_20X/h5file_slide_eat{tune_mask_ratio}",
                                  dataset_name="tcga", subtyping="nsclc")



    # -------------------------------------- previous tcga version --------------------------------------
    # non-eat version, deal with the ood dataset
    # for ood_dataset in ["blca", "ucs", "uvm", "acc"]:
    #     inference_slide_embedding(titan_tile_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file",
    #                               titan_slide_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file_slide")
    #     merge_slide_embedding(pkl_path=f"/home/user/sngp/UniConch/ood_pkl_folder/{ood_dataset}_pickle_data.pkl",
    #                           titan_slide_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file_slide",
    #                           dataset_name=ood_dataset)
    # eat version, deal with the ood dataset
    # tune_mask_ratio = 0.8
    # for ood_dataset in ["blca", "ucs", "uvm", "acc"]:
    #     inference_slide_embedding_eat(trials=4, folds=5, keep_ratio=tune_mask_ratio,
    #                                   titan_tile_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file",
    #                                   titan_slide_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file_slide_eat{tune_mask_ratio}",
    #                                   titan_amb_path=f"/home/user/sngp/UniConch/models/ambpkl/newambk/titan_{ood_dataset}_ambiguity_dict_autogluon_1.0_tuning0.pkl")
    #     merge_slide_embedding_eat(trials=4, folds=5, pkl_path=f"/home/user/sngp/UniConch/ood_pkl_folder/{ood_dataset}_pickle_data.pkl",
    #                               titan_slide_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file_slide_eat{tune_mask_ratio}",
    #                               dataset_name=ood_dataset)