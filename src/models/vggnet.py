

import sys
sys.path.append('../')

import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
import wandb
import torchio as tio
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
from torchmetrics.functional import accuracy

from dataloader import *


class VGG(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(self.hparams.cfg[-2] * 7 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.hparams.num_classes),
        )
        # if init_weights:
        self.loss = nn.CrossEntropyLoss(self.hparams.weight)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def training_step(self, batch, batch_idx):
        x = batch['image'][tio.DATA]
        y = batch['label']
        raw_out = self(x)
        loss = self.loss(raw_out, y)
        preds = torch.argmax(torch.softmax(raw_out, dim=1), dim=1)
        acc = accuracy(preds, y)

        # print(f"Train Loss: {loss}")
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        lr_scheduler = self.lr_schedulers()
        if lr_scheduler:
            lr_scheduler.step()
        return loss

    def evaluate(self, batch, stage=None):
        x = batch['image'][tio.DATA]
        y = batch['label']
        raw_out = self(x)
        loss = self.loss(raw_out, y)
        preds = torch.argmax(torch.softmax(raw_out, dim=1), dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True)
            self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True)
            wandb.log({"conf_mat_val": wandb.plot.confusion_matrix(probs = None,
                                                                   y_true = y.cpu().detach().numpy(),
                                                                   preds = preds.cpu().detach().numpy(),
                                                                   class_names = self.hparams.class_names)})
#        return torch.cat([y[:,None],preds[:,None]],axis=-1)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

  #  def validation_epoch_end(self, validation_step_outputs):

 #       all_preds = torch.vstack(validation_step_outputs)

 #       assert all_preds.shape[-1] == 2

   #     wandb.log({"conf_mat_val": wandb.plot.confusion_matrix(probs = None,
    #                                                        y_true = all_preds[:,0].cpu().detach().numpy(),
     #                                                       preds = all_preds[:,1].cpu().detach().numpy(),
      #                                                      class_names = self.hparams.class_names)})


    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        if self.hparams.optim == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        elif self.hparams.optim == 'sgd':
            optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        elif self.hparams.optim == 'adamw':
            optim = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        else:
            raise NotImplementedError

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(optim,T_max=10, eta_min=0), #ExponentialLR(optim, gamma=0.9), ReduceLROnPlateau(optim, ...),
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

    def make_layers(self):
        layers = []
        in_channels = 1
        for v in self.hparams.cfg:
            if v == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
            else:
                conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                if self.hparams.batch_norm:
                    layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv3d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


cfgs = {
    'A': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M'],
    'B': [8, 8,'M', 8,16, 'M', 16, 32, 32, 'M'],
    'C': [8, 8, 'M', 16, 16,'M', 32, 32, 'M', 64, 64, 'M'],
    #'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    #'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'E':[32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
}


#####################################################################
def main(test=False):


    # ------------
    # args
    # ------------
    print("Parsing arguments...")
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/spinney/enigma_drug/data')
    parser.add_argument('--batch_size', default=16, type=int)
    #parser.add_argument('--max_epochs', default=15, type=int)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--format', type=str, default='nifti')
    parser.add_argument('--test', type=bool, default=True)
    parser.add_argument('--cropped', type=bool, default=True)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--augment', nargs='*')
    parser.add_argument('--cfg_name', type=str, default='A')


    # trainer specific args
    parser = pl.Trainer.add_argparse_args(parser)

    # model specific args
    parser = VGG.add_model_specific_args(parser)
    args = parser.parse_args()

    # set global seed
    pl.seed_everything(args.seed)

    print("Starting Wandb...")

    wandb.init(project="deep", name=f"{args.name}-{args.cfg_name}",settings=wandb.Settings(start_method="fork"))

    wandb_logger = WandbLogger()

    mask = ''


    if args.num_samples == -1:
        args.num_samples = -1*args.num_classes


    # these are returned shuffled
    file_paths, labels = get_mri_data_beta(args.num_samples//args.num_classes, args.data_dir, cropped=args.cropped, test=False)

    dm = MRIDataModuleIO(args.data_dir, labels, args.format, args.batch_size, args.augment, mask, file_paths, args.num_workers)
    dm.prepare_data()
    dm.setup(stage='fit')

    print(f"Input shape used: {dm.max_shape}")
    dict_args = vars(args)
    dict_args['weight'] = dm.weight
    dict_args['input_shape'] = dm.max_shape
    dict_args['class_names'] = ["control","ALC","ATS","COC","NIC"]
    dict_args['cfg'] = cfgs[dict_args['cfg_name']]
    
    model = VGG(**dict_args)


    slurm = os.environ.get("SLURM_JOB_NUM_NODES")
    num_nodes = int(slurm) if slurm else 1

    default_root_dir = f"/scratch/spinney/enigma_drug/checkpoints/vggnet/"

    trainer = pl.Trainer(default_root_dir=default_root_dir,
                         gpus=torch.cuda.device_count(),
                         num_nodes=num_nodes,
                         strategy='ddp' if num_nodes > 1 else 'dp',
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

    trainer.test(datamodule=dm)


if __name__ == '__main__':
    main(test=False)

