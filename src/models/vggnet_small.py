

import sys
sys.path.append('../')

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
import torchio as tio
from nilearn.image import crop_img, resample_to_img
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

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

from dataloaders import *


class VGGNet(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        #self.save_hyperparameters(kwargs)


        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0))

        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.hparams.dropout),
                         )

        # self.block4 = nn.Sequential(
        #     nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=2, stride=2, padding=0),
        #     nn.BatchNorm3d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=self.hparams.dropout),
        # )


        n_size = self._get_block_output(self.hparams.input_shape)

        self.fc = nn.Sequential(nn.Linear(in_features=n_size, out_features=128),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(128),
                                nn.Dropout(p=self.hparams.dropout),
                                nn.Linear(in_features=128, out_features=64),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=64, out_features=self.hparams.num_classes))

        #self.loss = nn.BCEWithLogitsLoss(pos_weight=self.hparams.pos_weight)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self._block_pass(x.float())
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        #x, y = batch
        x = batch['image'][tio.DATA]
        y = batch['label']
        y_hat = self.forward(x.float())
        # loss = F.cross_entropy(y_hat, y)
        #print(f"ytrain: {y_hat.squeeze().float()}")
        loss = self.loss(y_hat.squeeze(), y)
        acc = self.accuracy(y_hat.squeeze().float(), y)
        y_hat_bin = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
        #print(f"Train Loss: {loss}")
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc)
        wandb.log({"conf_mat_train": wandb.plot.confusion_matrix(probs = None,
                                                            y_true = y.cpu().detach().numpy(),
                                                            preds = y_hat_bin.cpu().detach().numpy(),
                                                            class_names = self.hparams.class_names)})
        return loss



    def validation_step(self, batch, batch_idx):
        #x, y = batch
        x = batch['image'][tio.DATA]
        y = batch['label']
        y_hat = self.forward(x.float())
        # loss = F.cross_entropy(y_hat, y)
        loss = self.loss(y_hat.squeeze(), y)
        acc = self.accuracy(y_hat.squeeze().float(), y.float())
        y_hat_bin = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)

        self.log('valid_loss', loss, on_step=True)
        self.log('valid_acc', acc)

        wandb.log({"conf_mat_valid": wandb.plot.confusion_matrix(probs = None,
                                                            y_true = y.cpu().detach().numpy(),
                                                            preds = y_hat_bin.cpu().detach().numpy(),
                                                            class_names = self.hparams.class_names)})

    def test_step(self, batch, batch_idx):
        #x, y = batch
        x = batch['image'][tio.DATA]
        y = batch['label']
        y_hat = self.forward(x.float())
        # loss = F.cross_entropy(y_hat, y)
        #print(f"ytest: {y_hat.squeeze().float()}")
        loss = self.loss(y_hat.squeeze(), y)
        acc = self.accuracy(y_hat.squeeze().float(), y.float())
        y_hat_bin = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        wandb.log({"conf_mat_test": wandb.plot.confusion_matrix(probs=None,
                                                                 y_true=y.cpu().detach().numpy(),
                                                                 preds=y_hat_bin.cpu().detach().numpy(),
                                                                 class_names=self.hparams.class_names)})


    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        if self.hparams.optim == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4)
        elif self.hparams.optim == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        elif  self.hparams.optim == 'adamw':
            optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        else:
            raise NotImplementedError

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optim, gamma=0.9), #ReduceLROnPlateau(optim, ...),
                "monitor": "valid_loss",
            },
        }


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VGGNet")
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--name', type=str, default='vggnet')
        parser.add_argument('--optim', type=str, default='adam')
        return parent_parser



    def _block_pass(self,input):
        x = self.block1(input)
        #print(f"Block 1 shape: {x.shape}")
        x = self.block2(x)
        #print(f"Block 2 shape: {x.shape}")
        x = self.block3(x)
        #print(f"Block 3 shape: {x.shape}")
        # x = self.block4(x)
        #print(f"Block 4 shape: {x.shape}")
        return x

    def _get_block_output(self,shape):
        self.eval()
        batch_size=1
        input = torch.autograd.Variable(torch.rand(batch_size,*shape))
        input = torch.unsqueeze(input,0)
        output_feat = self._block_pass(input)
        n_size = output_feat.data.view(batch_size,-1).size(1)
        print(f"n_size: {n_size}")
        return n_size

    def accuracy(self, y_hat, y):

        targets = torch.argmax(torch.softmax(y_hat,dim=1),dim=1)
        return (targets == y).sum() / y.size(0)





#####################################################################
def main(test=False):


    # ------------
    # args
    # ------------
    print("Parsing arguments...")
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/sean/Projects/MRI_Deep_Learning/Kamran_Montreal_Data_Share/')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--format', type=str, default='nifti')
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--cropped', type=bool, default=True)
    parser.add_argument('--augment', nargs='*')

    # trainer specific args
    parser = pl.Trainer.add_argparse_args(parser)

    # model specific args
    parser = VGGNet.add_model_specific_args(parser)
    args = parser.parse_args()

    # set global seed
    pl.seed_everything(args.seed)

    print("Starting Wandb...")

    wandb.init(project="mri_classification_debug", name=args.name)

    wandb_logger = WandbLogger()

    #file_paths, labels = get_mri_data_(args.data_dir, cropped=args.cropped, test=args.test)

    mask = ''


    if args.num_samples == -1:
        args.num_samples = -1*args.num_classes


    file_paths, labels = get_mri_data_beta(args.num_samples//args.num_classes, args.data_dir, cropped=args.cropped, test=False)

    dm = MRIDataModuleIO(args.data_dir, labels, args.format, args.batch_size, args.augment, mask, file_paths, args.num_workers)
    dm.prepare_data()
    dm.setup(stage='fit')

    print(f"Input shape used: {dm.max_shape}")
    dict_args = vars(args)
    dict_args['pos_weight'] = dm.pos_weight
    dict_args['input_shape'] = dm.max_shape
    dict_args['class_names'] = ["control","ALC","ATS","COC","NIC"] if args.num_classes == 5 else ["control","ALC"]

    model = VGGNet(dict_args)

    slurm = os.environ.get("SLURM_JOB_NUM_NODES")
    num_nodes = int(slurm) if slurm else 1
    trainer = pl.Trainer(default_root_dir="/scratch/spinney/enigma_drug/checkpoints/",
                         gpus=torch.cuda.device_count(),
                         num_nodes=num_nodes,
                         accelerator='ddp' if num_nodes > 1 else 'dp',
                         max_epochs=args.max_epochs,
                         log_every_n_steps=10,
                         logger=wandb_logger,
                         replace_sampler_ddp=False)#,
                         #precision=16)
                         #early_stop_callback=False)
                         #callbacks=[early_stopping_callback])

    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------

    dm.setup(stage='test')
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    main(test=False)
