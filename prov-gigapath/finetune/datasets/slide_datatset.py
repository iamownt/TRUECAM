import os
import h5py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class SlideDatasetForTasks(Dataset):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 splits: list,
                 task_config: dict,
                 slide_key: str='slide_id',
                 split_key: str='pat_id',
                 evaluate_level: str="patient_level",
                 **kwargs
                 ):
        '''
        This class is used to set up the slide dataset for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide data
        root_path: str
            The root path of the tile embeddings
        splits: list
            The list of splits to use
        task_config: dict
            The task configuration dictionary
        slide_key: str
            The key that contains the slide id
        split_key: str
            The key that specifies the column for splitting the data
        '''
        self.root_path = root_path
        self.split_key = split_key
        self.slide_key = slide_key
        self.task_cfg = task_config

        # get slides that have tile encodings
        valid_slides = self.get_valid_slides(root_path, data_df[slide_key].values)
        # filter out slides that do not have tile encodings
        data_df = data_df[data_df[slide_key].isin(valid_slides)]
        if "patient" not in data_df.columns and "case_id" in data_df.columns:
            data_df.rename(columns={'case_id': 'patient'}, inplace=True)
        if "patient" in data_df.columns and "slide_id" in data_df.columns and evaluate_level == "patient_level":
            self.s_to_p_mapping = dict(zip(data_df["slide_id"], data_df["patient"]))
        else:
            self.s_to_p_mapping = None
        print("Evaluate level: ", evaluate_level)
        data_df.rename(columns={'cohort': 'label'}, inplace=True)
        # set up the task
        self.setup_data(data_df, splits, task_config.get('setting', 'multi_class'))
        
        self.max_tiles = task_config.get('max_tiles', 1000)
        self.shuffle_tiles = task_config.get('shuffle_tiles', False)
        print('Dataset has been initialized!')
        
    def get_valid_slides(self, root_path: str, slides: list) -> list:
        '''This function is used to get the slides that have tile encodings stored in the tile directory'''
        valid_slides = []
        for i in range(len(slides)):
            if 'pt_files' in root_path.split('/')[-1]:
                sld = slides[i].replace(".svs", "") + '.pt'
            else:
                sld = slides[i].replace(".svs", "") + '.h5'
            sld_path = os.path.join(root_path, sld)
            if not os.path.exists(sld_path):
                print('Missing: ', sld_path)
            else:
                valid_slides.append(slides[i])
        return valid_slides
    
    def setup_data(self, df: pd.DataFrame, splits: list, task: str='multi_class'):
        '''Prepare the data for multi-class setting or multi-label setting'''
        # Prepare slide data
        if task == 'multi_class' or task == 'binary':
            prepare_data_func = self.prepare_multi_class_or_binary_data
        elif task == 'multi_label':
            prepare_data_func = self.prepare_multi_label_data
        else:
            raise ValueError('Invalid task: {}'.format(task))
        self.slide_data, self.images, self.labels, self.n_classes = prepare_data_func(df, splits)
    
    def prepare_multi_class_or_binary_data(self, df: pd.DataFrame, splits: list):
        '''Prepare the data for multi-class classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert  label_dict, 'No label_dict found in the task configuration'
        # set up the mappings
        assert 'label' in df.columns, 'No label column found in the dataframe'
        if df['label'].dtype == 'object':
            df['label'] = df['label'].map(label_dict)

        n_classes = len(label_dict)
        unique_labels = len(df['label'].unique())
        # assert n_classes == unique_labels, f'Mismatch: label_dict has {n_classes} classes but data has {unique_labels} unique labels'

        # get the corresponding splits
        if self.split_key is not None:
            assert self.split_key in df.columns, 'No {} column found in the dataframe'.format(self.split_key)
            df = df[df[self.split_key].isin(splits)]
        images = df[self.slide_key].to_list()
        labels = df[['label']].to_numpy().astype(int)
        
        return df, images, labels, n_classes
        
    def prepare_multi_label_data(self, df: pd.DataFrame, splits: list):
        '''Prepare the data for multi-label classification'''
        # set up the label_dict
        label_dict = self.task_cfg.get('label_dict', {})
        assert label_dict, 'No label_dict found in the task configuration'
        # Prepare mutation data
        label_keys = label_dict.keys()
        # sort key using values
        label_keys = sorted(label_keys, key=lambda x: label_dict[x])
        n_classes = len(label_dict)

        # get the corresponding splits
        assert self.split_key in df.columns, 'No {} column found in the dataframe'.format(self.split_key)
        df = df[df[self.split_key].isin(splits)]
        images = df[self.slide_key].to_list()
        labels = df[label_keys].to_numpy().astype(int)
            
        return df, images, labels, n_classes
    
    
class SlideDataset(SlideDatasetForTasks):
    def __init__(self,
                 data_df: pd.DataFrame,
                 root_path: str,
                 splits: list,
                 task_config: dict,
                 slide_key='slide_id',
                 split_key='pat_id',
                 evaluate_level="slide_level",
                 mask_dict=None,
                 **kwargs
                 ):
        '''
        The slide dataset class for retrieving the slide data for different tasks

        Arguments:
        ----------
        data_df: pd.DataFrame
            The dataframe that contains the slide data
        root_path: str
            The root path of the tile embeddings
        splits: list
            The list of splits to use
        task_config_path: dict
            The task configuration dictionary
        slide_key: str
            The key that contains the slide id
        split_key: str
            The key that specifies the column for splitting the data
        '''
        super(SlideDataset, self).__init__(data_df, root_path, splits, task_config, slide_key, split_key, evaluate_level, **kwargs)
        self.mask_dict = mask_dict

    def shuffle_data(self, images: torch.Tensor, coords: torch.Tensor) -> tuple:
        '''Shuffle the serialized images and coordinates'''
        indices = torch.randperm(len(images))
        images_ = images[indices]
        coords_ = coords[indices]
        return images_, coords_

    def read_assets_from_h5(self, h5_path: str) -> tuple:
        '''Read the assets from the h5 file'''
        assets = {}
        attrs = {}
        h5_path = Path(h5_path)
        slide_id = h5_path.stem
        with h5py.File(h5_path, 'r') as f:
            if self.mask_dict is not None:
                epsilon = 1e-7  # Small value to prevent log(0)
                probabilities = np.clip(self.mask_dict[slide_id]["probabilities"], epsilon, 1.0).astype(np.float32)
                self.mask_dict[slide_id]["ambiguity"] = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

                in_slide_threshold = np.quantile(self.mask_dict["mask_pkl"][slide_id]["ambiguity"], self.mask_dict["mask_tile_threshold"])
                mask_bool = self.mask_dict["mask_pkl"][slide_id]["ambiguity"] <= in_slide_threshold
            for key in f.keys():
                if self.mask_dict is not None:
                    assets[key] = f[key][:][mask_bool]
                else:
                    assets[key] = f[key][:]
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
        return assets, attrs

    def read_assets_from_pt(self, pt_path: str) -> dict:
        '''Read the assets from the pt file'''
        assets = torch.load(pt_path)
        pt_path = Path(pt_path)
        if self.mask_dict is not None:
            slide_id = pt_path.stem
            epsilon = 1e-7  # Small value to prevent log(0)
            probabilities = np.clip(self.mask_dict["mask_pkl"][slide_id]["probabilities"], epsilon, 1.0).astype(np.float32)
            self.mask_dict["mask_pkl"][slide_id]["ambiguity"] = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

            in_slide_threshold = np.quantile(self.mask_dict["mask_pkl"][slide_id]["ambiguity"],
                                             self.mask_dict["mask_tile_threshold"])
            mask_bool = self.mask_dict["mask_pkl"][slide_id]["ambiguity"] <= in_slide_threshold
            # print("mask", len(mask_bool), mask_bool.sum())
        for key in assets.keys():
            if self.mask_dict is not None:
                assets[key] = assets[key][mask_bool]
            else:
                assets[key] = assets[key]
        return assets, {}

    def get_sld_name_from_path(self, sld: str) -> str:
        '''Get the slide name from the slide path'''
        sld_name = os.path.basename(sld).split('.h5')[0]
        return sld_name
    
    def get_images_from_path(self, img_path: str) -> dict:
        '''Get the images from the path'''
        if '.pt' in img_path:
            assets, _ = self.read_assets_from_pt(img_path)
            images = assets['features']
            coords = assets['coords']
        elif '.h5' in img_path:
            assets, _ = self.read_assets_from_h5(img_path)
            images = torch.from_numpy(assets['tile_embeds'])
            coords = torch.from_numpy(assets['coords'])

        if self.shuffle_tiles:
            images, coords = self.shuffle_data(images, coords)

        if images.size(0) > self.max_tiles:
            images = images[:self.max_tiles, :]
        if coords.size(0) > self.max_tiles:
            coords = coords[:self.max_tiles, :]
        # set the input dict

        data_dict = {'imgs': images,
                'img_lens': images.size(0),
                'pad_mask': 0,
                'coords': coords}
        return data_dict
    
    def get_one_sample(self, idx: int) -> dict:
        '''Get one sample from the dataset'''
        # get the slide id
        # slide_id = self.images[idx]
        slide_id = str(self.images[idx]) + ".svs"
        # get the slide path
        if 'pt_files' in self.root_path.split('/')[-1]:
            slide_path = os.path.join(self.root_path, slide_id.replace(".svs", "") + '.pt')
        else:
            slide_path = os.path.join(self.root_path, slide_id.replace(".svs", "") + '.h5')
        # get the slide images
        data_dict = self.get_images_from_path(slide_path)
        # get the slide label
        label = torch.from_numpy(self.labels[idx])
        # set the sample dict
        sample = {'imgs': data_dict['imgs'],
                  'img_lens': data_dict['img_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords': data_dict['coords'],
                  'slide_id': self.images[idx] if self.s_to_p_mapping is None else self.s_to_p_mapping[self.images[idx]],
                  'labels': label}
        return sample
    
    def get_sample_with_try(self, idx, n_try=3):
        '''Get the sample with n_try'''
        for _ in range(n_try):
            sample = self.get_one_sample(idx)
            try:
                sample = self.get_one_sample(idx)
                return sample
            except:
                print('Error in getting the sample, try another index')
                idx = np.random.randint(0, len(self.slide_data))
        print('Error in getting the sample, skip the sample')
        return None
        
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        sample = self.get_sample_with_try(idx)
        return sample

    
