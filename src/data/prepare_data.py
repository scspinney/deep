from nilearn.image import index_img, smooth_img
from nilearn.masking import apply_mask
from nibabel.nifti1 import Nifti1Image
import os
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import glob
import pandas as pd
import math
from functools import partial
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import wandb

from nilearn.image import crop_img, resample_to_img

import warnings
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


def random_split(dataset, lengths):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def class_imbalance_sampler(labels):
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    sampler = WeightedRandomSampler(sample_weights, len(labels))
    return sampler


class MRIDataset(Dataset):

    def __init__(self, images: Tensor, labels: Tensor, train: bool, transform: List[Callable[[Tensor], Tensor]]):
        self.labels = labels
        self.images = images
        self.transform = transform
        self.train = train

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if type(idx) == torch.Tensor:
            idx = idx.item()

        sample = self.images[idx, None, :, :, :]
        # reshape for channel

        if self.transform:
            for t in self.transform:
                sample = t(sample)

        return sample, self.labels[idx]



def prepare_data(data_dir, file_paths, mask):


    # take the first image, crop, then use to resample all others
    # load image and remove nan and inf values.
    path = os.path.join(data_dir, file_paths[0])
    ref_image = crop_img(smooth_img(path, fwhm=None))
    # ref_image = torch.tensor(ref_image.get_fdata(), dtype=torch.float64)
    print(f"Reference image shape after cropping : {ref_image.shape}")



    for index, image_path in enumerate(file_paths):
        print(f"Processing {index/len(file_paths)}...")
        # load image and remove nan and inf values.
        path = os.path.join(data_dir, image_path)
        image = resample_to_img(smooth_img(path, fwhm=None), ref_image)

        if mask:
            raise NotImplementedError
            #image = apply_mask(path, mask)

        # save
        print(f"Out shape: {image.shape}")
        outname = path.split('.')[0] + '_cr.mgz'
        image.to_filename(outname)

    return ref_image.shape



def prepare_data_c(data_dir, file_paths, mask):

    max_shape = (0,0,0)
    filenames = []
    for index, image_path in enumerate(file_paths):
        print(f"Processing {index/len(file_paths)}...")
        # load image and remove nan and inf values.
        path = os.path.join(data_dir, image_path)
        image = crop_img(smooth_img(path, fwhm=None))

        if np.prod(image.shape) > np.prod(max_shape):
            max_shape = image.shape
        if mask:
            raise NotImplementedError
            #image = apply_mask(path, mask)

        # save
        print(f"Out shape: {image.shape}")
        outname = path.split('.')[0] + '_c.mgz'
        image.to_filename(outname)
        filenames.append(outname)

    return max_shape, filenames


def get_mri_data(data_dir,label):
    df = pd.read_csv(os.path.join(data_dir, 'data_split.csv'))
    file_paths = list(df["filename"].values)
    labels = list(df[label].values)
    return file_paths, labels




def main(test=False):
    # pl.seed_everything(1234)

    # ------------
    # args
    # ------------

    # parser = ArgumentParser()
    # parser.add_argument('--data_dir', type=str,
    #                     default='/Users/sean/Projects/MRI_Deep_Learning/Kamran_Montreal_Data_Share/')
    # parser.add_argument('--batch_size', default=4, type=int)
    # parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--num_workers', type=int, default=0)
    # parser.add_argument('--format', type=str, default='nifti')
    # parser.add_argument('--test', type=bool, default=True)
    # parser = pl.Trainer.add_argparse_args(parser)
    # # parser = DL1Classifier.add_model_specific_args(parser)
    # args = parser.parse_args()

    #data_dir = '/scratch/spinney/enigma_drug/data/'
    data_dir = '/Users/sean/Projects/MRI_Deep_Learning/Kamran_Montreal_Data_Share/'
    label = "class"
    file_paths, labels = get_mri_data(data_dir,label)
    mask = ''

    out_dims, file_paths_c = prepare_data_c(data_dir,file_paths,mask)

    # create new data file with croppped
    print(f"Creating new cropped data_split file in {data_dir}")
    df = pd.read_csv(os.path.join(data_dir, 'data_split.csv'))
    df["filename"] = file_paths_c
    df.to_csv(os.path.join(data_dir,"data_split_c.csv"))


    print(f"Completed resizing/cropping images to dimensions: {out_dims}")


if __name__ == '__main__':
    main(test=True)
