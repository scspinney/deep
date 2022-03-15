# -*- coding: utf-8 -*-
"""dl1_enigma_drug.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o7ohIUvnkFyBrRMyNWm1TGrLhXvw1wpj
"""

from nilearn.image import index_img, smooth_img
from nilearn.masking import apply_mask
from nibabel.nifti1 import Nifti1Image
import os
import sys
import torch
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import json

import monai
from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ResizeWithPadOrCrop, ScaleIntensity, EnsureType, RandGaussianNoise, RandAffine, Rotate

import nibabel as nib
from monai.data.image_reader import NibabelReader, has_nib
from monai.data.utils import (correct_nifti_header_if_necessary, is_supported_format)
from monai.transforms import LoadImage
from monai.utils import ensure_tuple
from nibabel.nifti1 import Nifti1Image
from nilearn.image import crop_img, resample_to_img

from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    Any
)


class Subsample(object):
    """subsample the 3D image of a timestep.

    Args:
    """

    def __init__(self, subsample_rate_x, subsample_rate_y, subsample_rate_z):
        self.rate_x = subsample_rate_x
        self.rate_y = subsample_rate_y
        self.rate_z = subsample_rate_z

    def __call__(self, sample):
        dim_x = sample.shape[0]
        dim_y = sample.shape[1]
        dim_z = sample.shape[2]

        indexes_x = range(0, dim_x, self.rate_x)
        indexes_y = range(0, dim_y, self.rate_y)
        indexes_z = range(0, dim_z, self.rate_z)

        sample = sample[indexes_x, :, :]
        sample = sample[:, indexes_y, :]
        sample = sample[:, :, indexes_z]

        return sample



def class_imbalance_sampler(labels):

    class_sample_count = torch.tensor(
        [(labels == t).sum() for t in torch.unique(labels, sorted=True)])

    print(f"Class_count: {class_sample_count}")

    weight = 1. / class_sample_count.float()

    samples_weight = torch.tensor([weight[t] for t in labels])

    # Create sampler, dataset, loader
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)

    return sampler, weight


class MRIDataModuleIO(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 labels: List[int],
                 format: str,
                 batch_size: int,
                 augment: List[str],
                 mask: str = '',
                 file_paths: List[str] = None,
                 num_workers: int = 1,
                 input_shape: Tuple = (96,96,96),
                 sampler: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.labels = torch.tensor(labels)
        self.format = format
        self.n = len(labels)
        self.n_train = int(.8 * self.n)
        self.n_test = self.n - self.n_train
        self.mask = mask
        self.batch_size = batch_size
        self.augment = augment
        self.file_paths = file_paths
        self.num_workers = num_workers
        self.input_shape = input_shape

        shuffled_ind = np.random.choice(range(self.n), len(range(self.n)), replace=False)

        self.train_labels = self.labels[shuffled_ind[:self.n_train]]
        self.test_labels = self.labels[shuffled_ind[self.n_train:]]
        self.train_paths = self.file_paths[shuffled_ind[:self.n_train]]
        self.test_paths = self.file_paths[shuffled_ind[self.n_train:]]
        self.sampler = sampler

        # check test distribution
        class_sample_count_test = torch.tensor(
            [(self.test_labels == int(t)).sum() for t in torch.unique(self.labels, sorted=True)])

        class_sample_count_train = torch.tensor(
            [(self.train_labels == int(t)).sum() for t in torch.unique(self.labels, sorted=True)])

        print(f"Class distribution in test set: {class_sample_count_test}")
        print(f"Class distribution in train set: {class_sample_count_train}")

    # def get_max_shape(self, subjects):
    #
    #     dataset = tio.SubjectsDataset(subjects, transform=preprocess)
    #     dataset = ImageDataset(image_files=images, labels=labels, transform=train_transforms)
    #     shapes = np.array([s.spatial_shape for s in dataset])
    #     self.max_shape = shapes.max(axis=0)
    #     return self.max_shape


    def get_preprocessing_transform(self):
        preprocess = Compose([ScaleIntensity(),
                              AddChannel(),
                              ResizeWithPadOrCrop(self.input_shape),
                              EnsureType(),
        ])
        return preprocess

    def get_augmentation_transform(self):

        if self.augment:
            augment = []
            for a in self.augment:
                if a == 'affine':
                    augment.append(RandAffine(0.1))
                elif a == 'noise':
                    augment.append(RandGaussianNoise(0.3))

                elif a == 'rotate':
                    augment.append(Rotate(20))

            augment = Compose(augment)
            return augment
        else:
            return None

    def setup(self, stage=None):
        image_training_paths = self.train_paths
        label_training = self.train_labels
        image_test_paths = self.test_paths
        label_test = self.test_labels

        indices = range(self.n_train)  # np.random.choice(range(self.n_train), range(self.n_train), replace=False)
        split = int(np.floor(.2 * self.n_train))
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        if self.sampler is not True:
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.val_sampler = SubsetRandomSampler(val_indices)
        else:

            self.train_sampler, self.weight = class_imbalance_sampler(self.train_labels[train_indices])


        image_train_paths_subset = [image_training_paths[i] for i in train_indices]
        label_train_paths_subset = [label_training[i] for i in train_indices]
        image_val_paths_subset = [image_training_paths[i] for i in val_indices]
        label_val_paths_subset = [label_training[i] for i in val_indices]

        self.preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        if augment is not None:
            self.transform = Compose([self.preprocess, augment])
        else:
            self.transform = self.preprocess

        self.train_set = ImageDataset(image_files=image_train_paths_subset, labels=label_train_paths_subset,
                                transform=self.transform, reader="NibabelReader")
        self.val_set = ImageDataset(image_files=image_val_paths_subset, labels=label_val_paths_subset,
                                transform=self.preprocess, reader="NibabelReader")
        self.test_set = ImageDataset(image_files=image_test_paths, labels=label_test,
                                transform=self.preprocess, reader="NibabelReader")

        # save train/test sets
        with open(os.path.join(self.data_dir,"train_fnames.txt"), "w") as fp:
            json.dump(list(image_training_paths), fp)

        with open(os.path.join(self.data_dir,"test_fnames.txt"), "w") as fp:
            json.dump(list(image_test_paths), fp)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, sampler=self.train_sampler, num_workers=0,#self.num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=0, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=0)


def get_mri_data_beta(num_samples,num_classes, data_dir, cropped=False):

    name = f"data_split_c.csv"
    class_name = "dep" if num_classes == 2 else "class"
    N = num_samples // num_classes
    df = pd.read_csv(os.path.join(data_dir, name))
    labels = []

    dfg = df.groupby(class_name)
    data = []
    for name, subdata in dfg:
        print(f"Group: {name}")
        # shuffle
        K = subdata.shape[0]
        shuffled_ind = np.random.choice(range(K), len(range(K)), replace=False)
        # subsample
        shuffled_ind = shuffled_ind[:N]
        data.extend(subdata["filename"].values[shuffled_ind])
        labels.extend(subdata[class_name].values[shuffled_ind])

    # shuffle
    data = np.array(data).reshape(-1)
    labels = np.array(labels).reshape(-1)
    ind = np.random.choice(len(labels), len(labels), replace=False)

    labels = labels[ind]
    data = data[ind]

    assert data.shape[0] == labels.shape[0]

    return data, labels

def make_environment(flags):
    # dep,drug,age,sex,filename,study,class
    data_dir = flags.data_dir
    input_shape = flags.input_shape
    batch_size = flags.batch_size
    name = f"data_split_c.csv"
    class_name = "dep"    
    df = pd.read_csv(os.path.join(data_dir, name))
    K = df.shape[0]
    shuffled_ind = np.random.choice(range(K), len(range(K)), replace=False)
    image_train_paths = np.array(df["filename"].values[shuffled_ind])
    label_train = np.array(df[class_name].values[shuffled_ind])
    drug = np.array(df["drug"].values[shuffled_ind])
    age = np.array(df["age"].values[shuffled_ind])
    class_type = np.array(df["class"].values[shuffled_ind])
    sex = np.array(df["sex"].values[shuffled_ind])
    envs = [
        {
                'images': image_train_paths[:K//2-50],
                'labels': label_train[:K//2-50],
                'drug': drug[:K//2-50],
                'age': age[:K//2-50],
                'sex': sex[:K//2-50],
                'class': class_type[:K//2-50]
        },
        {
                'images': image_train_paths[K//2-50:-50],
                'labels': label_train[K//2-50:-50],
                'drug': drug[K//2-50:-50],
                'age': age[K//2-50:-50],
                'sex': sex[K//2-50:-50],
                'class': class_type[K//2-50:-50]                
        },
        {
                'images': image_train_paths[-100:],
                'labels': label_train[-100:],
                'drug': drug[-100:],
                'age': age[-100:],
                'sex': sex[-100:],
                'class': class_type[-100:]                
        }
        ]
            
    return envs


def simple_dataloader(image_paths,labels,batch_size,transform):
    if isinstance(labels,np.ndarray):
        labels = torch.tensor(labels)

    train_sampler, weight = class_imbalance_sampler(labels)
    train_set = ImageDataset(
        image_files=image_paths, 
        labels=labels,
        transform=transform, 
        reader="NibabelReader"
        )
    
    dataloader = DataLoader(
        train_set, 
        batch_size, 
        sampler=train_sampler, 
        num_workers=0,
        drop_last=True,
    )
    return dataloader, weight

