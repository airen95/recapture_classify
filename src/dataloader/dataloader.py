import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from omegaconf import OmegaConf


from .process import read_image, preprocess_pipeline
from .transform import *

class IdDataSet(Dataset):
    def __init__(self, data, input_size, method, transform):

        self.data = data
        self.input_size = input_size
        self.method = method
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        image = read_image(filename)
        image = preprocess_pipeline(image, self.input_size, self.method)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}

        return sample



# config = OmegaConf.load('/home/le/capture_classify/config/config.yaml')
# sample = IdDataSet(config, transform_train)
# for i, s in enumerate(sample):
#     print(s)

class CustomDataset:
    def __init__(self, config):
        self.cfg = config
        self.data = pd.read_csv(self.cfg.data['data_path'])
        self.input_size = self.cfg.data['input_size']
        self.method = self.cfg.data['method']
        self.batch_size = self.cfg.data['batch_size']
        self.num_workers = self.cfg.data['num_workers']
        self.device = self.cfg.device
        self.mean, self.std = None, None
        self.method = self.cfg.data['method']

    def get_dataset(self, transforms):
        sample = IdDataSet(self.data, self.input_size, self.method, transforms)
        return sample

    def get_dataloader(self, mode: str = 'train'):
        data = self.data.loc[self.data["usage"] == mode].reset_index(drop=True)
        data = data.to_dict("records")
        if mode == "train":
            transforms = transform_train(mean=self.mean, std=self.std) if (self.mean is not None and self.std is not None) else transform_train(self.method)
            # transforms = data_transforms[mode]
            shuffle = True

        elif mode in ["valid", "test"]:
            transforms = transform_val(mean=self.mean, std=self.std) if (self.mean is not None and self.std is not None) else transform_val(self.method)
            # transforms = data_transforms[mode]
            shuffle = False

        datasets = self.get_dataset(transforms=transforms)

        dataloader = DataLoader(datasets,
                        batch_size=self.batch_size,
                        shuffle=shuffle,
                        num_workers=self.num_workers,
                        drop_last=True,
                        pin_memory=True)

        return dataloader



