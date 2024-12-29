import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from pathlib import Path
import time
import glob
from tqdm import tqdm
import os
from PIL import Image
# from gigapath.pipeline import run_inference_with_tile_encoder
# from gigapath.pipeline import load_tile_slide_encoder, load_tile_encoder_transforms
from torchvision import transforms

import pickle
import timm
import h5py
from CONCH.conch.open_clip_custom import create_model_from_pretrained
from functools import partial
import gc
from ABMIL import load_pickle_model, save_pickle_data
from transformers import AutoModel

os.environ["HF_TOKEN"] = "YOUR HUGGINGFACE TOKEN"


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class AverageMeter:
    def __init__(self):
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


def _parse_function(record):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'slide': tf.io.FixedLenFeature([], tf.string),
        'loc_x': tf.io.FixedLenFeature([], tf.int64),
        'loc_y': tf.io.FixedLenFeature([], tf.int64),
    }
    return tf.io.parse_single_example(record, feature_description)

def decode_example(example):
    image = tf.io.decode_image(example['image_raw'], channels=3)
    slide = example['slide']
    loc_x = example['loc_x']
    loc_y = example['loc_y']
    return image, slide, loc_x, loc_y


def load_tile_encoder_transforms(model_name) -> transforms.Compose:
    """Load the transforms for the tile encoder"""
    if model_name != "titan":
        transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    else:
        img_size = 448 # performance of 512 is badder than 448
        transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    return transform


class MixedDataset(Dataset):
    def __init__(self, tfrecord_files, transform=None, pkl_path=None):
        self.tfrecord_files = tfrecord_files
        init_record_time = time.time()
        self.dataset = tf.data.TFRecordDataset(tfrecord_files)
        self.records = list(self.dataset)
        print("Time to read records:", time.time() - init_record_time)
        self.transform = transform
        if pkl_path:
            self.pkl_dict = load_pickle_model(pkl_path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        example = _parse_function(record)
        image, slide, loc_x, loc_y = decode_example(example)

        image = image.numpy()
        slide = slide.numpy().decode('utf-8')
        loc_x = int(loc_x.numpy())
        loc_y = int(loc_y.numpy())

        # Convert image to PyTorch tensor
        # image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
        if self.transform:
            image_pil = Image.fromarray(image, 'RGB')
            image_tensor = self.transform(image_pil)
        coord = torch.tensor([loc_x, loc_y], dtype=torch.float32)
        return image_tensor, self.pkl_dict[slide], coord


class ProvTCGAConfig:
    model_name = "prov-gigapath"
    tfrecord_path = Path("/home/user/sngp/project/destination_20X/tfrecords/256px_128um/")
    h5_des = Path("/home/user/sngp/project/destination_20X/h5file")
    pkl_path = "/home/user/sngp/project/destination/pickle_data.pkl"


class ProvCPTACConfig:
    model_name = "prov-gigapath"
    tfrecord_path = Path("/home/user/sngp/project/cptac_destination_20X/tfrecords/256px_128um/")
    h5_des = Path("/home/user/sngp/project/cptac_destination_20X/h5file")
    pkl_path = "/home/user/sngp/project/cptac_destination/pickle_data.pkl"

class TITANConfig:
    model_name = "titan"
    tfrecord_path = Path("/home/user/sngp/project/titan_destination_20X/tfrecords/512px_256um/")
    h5_des = Path("/home/user/sngp/project/titan_destination_20X/h5file")
    pkl_path = "/home/user/sngp/project/destination/pickle_data.pkl"


class TITANCPTACConfig:
    model_name = "titan"
    tfrecord_path = Path("/home/user/sngp/project/titan_cptac_destination_20X/tfrecords/512px_256um/")
    h5_des = Path("/home/user/sngp/project/titan_cptac_destination_20X/h5file")
    pkl_path = "/home/user/sngp/project/cptac_destination/pickle_data.pkl"


class UniTCGAConfig:
    model_name = "uni"
    tfrecord_path = Path("/home/user/sngp/project/destination_20X/tfrecords/256px_128um/")
    h5_des = Path("/home/user/sngp/UniConch/uni_tcga_h5file")
    pkl_path = "/home/user/sngp/project/destination/pickle_data.pkl"

class UniCPTACConfig:
    model_name = "uni"
    tfrecord_path = Path("/home/user/sngp/project/cptac_destination_20X/tfrecords/256px_128um/")
    h5_des = Path("/home/user/sngp/UniConch/uni_cptac_h5file")
    pkl_path = "/home/user/sngp/project/cptac_destination/pickle_data.pkl"

class ConchTCGAConfig:
    model_name = "conch"
    tfrecord_path = Path("/home/user/sngp/project/destination_20X/tfrecords/256px_128um/")
    h5_des = Path("/home/user/sngp/UniConch/conch_tcga_h5file")
    pkl_path = "/home/user/sngp/project/destination/pickle_data.pkl"

class ConchCPTACConfig:
    model_name = "conch"
    tfrecord_path = Path("/home/user/sngp/project/cptac_destination_20X/tfrecords/256px_128um/")
    h5_des = Path("/home/user/sngp/UniConch/conch_cptac_h5file")
    pkl_path = "/home/user/sngp/project/cptac_destination/pickle_data.pkl"


def create_ood_config(model_name_str: str, ood_dataset: str):
    assert model_name_str in ["prov-gigapath", "uni", "conch", "titan"]
    assert ood_dataset in ["blca", "ucs", "uvm", "acc"]
    if model_name_str != "titan":
        class cfg:
            model_name = model_name_str
            tfrecord_path = f"/home/user/sngp/project/{ood_dataset}_destination_20X/tfrecords/256px_128um/"
            h5_des = Path(f"/home/user/sngp/UniConch/{model_name_str}_{ood_dataset}_h5file")
            pkl_path = Path(f"/home/user/sngp/UniConch/ood_pkl_folder/{ood_dataset}_pickle_data.pkl")
    else:
        class cfg:
            model_name = model_name_str
            tfrecord_path = f"/home/user/sngp/project/titan_{ood_dataset}_destination_20X/tfrecords/512px_256um/"
            h5_des = Path(f"/home/user/sngp/UniConch/{model_name_str}_{ood_dataset}_h5file")
            pkl_path = Path(f"/home/user/sngp/UniConch/ood_pkl_folder/{ood_dataset}_pickle_data.pkl")
    return cfg




def load_model_and_transform(model_name: str):
    if model_name == "prov-gigapath":
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    elif model_name == "uni":
        local_dir = "/home/user/wangtao/assets/uni/vit_large_patch16_224.dinov2.uni_mass100k"
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    elif model_name == "conch":
        local_dir = "/home/user/wangtao/assets/conch/"
        model, _ = create_model_from_pretrained("conch_ViT-B-16", os.path.join(local_dir, "pytorch_model.bin"))
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
    elif model_name == 'titan':
        titan = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        model, _ = titan.return_conch()
    else:
        raise ValueError("Invalid model name for tile encoder. Must be one of 'prov-gigapath', 'uni', 'conch'")
    transform = load_tile_encoder_transforms(model_name=model_name)
    print("Tile encoder param #", sum(p.numel() for p in model.parameters()))

    return model, transform

pkl_path_dict = {
    "tcga": "/home/user/sngp/project/destination/pickle_data.pkl",
    "cptac": "/home/user/sngp/project/cptac_destination/pickle_data.pkl",
}

# config_list = []
# for model in ["conch", "uni", "prov-gigapath", "titan"]:
#     for ood_dataset in ["blca", "ucs", "uvm", "acc"]:
#         config = create_ood_config(model, ood_dataset)
#         config_list.append(config)

for config in [ConchTCGAConfig, ConchCPTACConfig, UniTCGAConfig, UniCPTACConfig, TITANConfig, TITANCPTACConfig]:
    batch_size = 1
    num_workers = 0
    tfrecord_path = config.tfrecord_path
    tfrecord_files = glob.glob(os.path.join(tfrecord_path, '*.tfrecords'))
    tfrecord_files_size = len(tfrecord_files)
    print("Number of tfrecords:", tfrecord_files_size)

    tile_encoder, transform = load_model_and_transform(model_name=config.model_name)
    if config.model_name != "conch":
        tile_encoder = torch.compile(tile_encoder, mode="max-autotune")
    tile_encoder = tile_encoder.cuda()
    tile_encoder.eval()
    tile_encoder.half()

    h5_des = config.h5_des
    os.makedirs(h5_des, exist_ok=True)

    with open(pkl_path_dict["cptac"], "rb") as f:
        pkl_dict = pickle.load(f)
    invert_pkl_dict = {v: k for k, v in pkl_dict.items()}

    tfrecord_files_size = len(tfrecord_files)
    for i in range(tfrecord_files_size):
        print("tfrecord_files[i:i+1]", tfrecord_files[i:i+1])
        dataset = MixedDataset(tfrecord_files[i:i+1], transform=transform,
                               pkl_path=config.pkl_path)
        print("Length of dataset:", len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

        collated_outputs = {"tile_embeds": [], "coords": []}
        # compute the data time and model time, keep moving average
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()
        start_time = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for batch in tqdm(dataloader, desc="Running inference with tile encoder"):
                data_time_meter.update(time.time() - start_time)
                images, slides, coord = batch
                # print("images.shape:", images.shape)
                images = images.to(dtype=torch.float16)
                collated_outputs["tile_embeds"].append(tile_encoder(images.cuda()).detach().cpu())
                collated_outputs["coords"].append(coord)
                batch_time_meter.update(time.time() - start_time)
                start_time = time.time()
                # print("Average batch_time", batch_time_meter.avg, "Average data_time", data_time_meter.avg, flush=True)
                # print("tile_embeds", collated_outputs["tile_embeds"][0].shape)
        with h5py.File(h5_des / f"{slides[0].item()}.h5", "w") as h5_file:
            h5_file.create_dataset("tile_embeds", data=torch.cat(collated_outputs["tile_embeds"], dim=0).numpy())
            h5_file.create_dataset("coords", data=torch.cat(collated_outputs["coords"], dim=0).numpy())

    torch.cuda.empty_cache()
    gc.collect()

