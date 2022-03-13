import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import pdb
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataloader import *

def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

# Define loss function helpers
def nll(logits, y, reduction='mean'):
    #TODO: cross entropy not binary
    return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def pretrain_model(flags,envs,model,optimizer_pre,batch_size,transform):
    n_env = len(envs)
    for step in range(flags.steps):
        for env in envs:
            train_dataloader = simple_dataloader(env['images'],env['labels'],batch_size,transform)
            logits_env = []
            labels_env = []
            for i, (images, labels) in enumerate(train_dataloader):
                logits = model(images)
                logits_env.append(logits)
                labels_env.append(labels)
            logits = torch.cat(logits_env,0)
            labels = torch.cat(labels_env,0)
            env['nll'] = nll(logits, labels)
            env['acc'] = mean_accuracy(logits, labels)
            env['penalty'] = penalty(logits, labels)

        train_nll = torch.stack([envs[i]['nll'] for i in range(n_env-1)]).mean()
        train_acc = torch.stack([envs[i]['acc'] for i in range(n_env-1)]).mean()
        train_penalty = torch.stack([envs[i]['penalty'] for i in range(n_env-1)]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm
        # NOTE: IRM penalties not used in pre-training

        optimizer_pre.zero_grad()
        loss.backward()
        optimizer_pre.step()

        test_acc = envs[-1]['acc']
        if step % 100 == 0:
            pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            train_penalty.detach().cpu().numpy(),
            test_acc.detach().cpu().numpy()
            )
    return model   


class VGG(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.n_size = self._get_block_output(self.hparams.input_shape)
        self.classifier = self.make_classifier()
        # if init_weights:        
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
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
                "scheduler": CosineAnnealingLR(optim, T_max=10, eta_min=0),
                # ExponentialLR(optim, gamma=0.9), ReduceLROnPlateau(optim, ...),
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


    def make_classifier(self):
        return nn.Sequential(  # nn.Linear(self.hparams.cfg[-2] * 7 * 7 * 7, self.hparams.classifier_cfg[0]),
            nn.Linear(self.n_size, self.hparams.classifier_cfg[0]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.hparams.classifier_cfg[0], self.hparams.classifier_cfg[1]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.hparams.classifier_cfg[1], self.hparams.num_classes))


    def _get_block_output(self, shape):
        self.eval()
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        input = torch.unsqueeze(input, 0)
        output_feat = self.features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        print(f"Block output size: {n_size}")
        return n_size


def split_data_opt(envs, model, n_steps=10000, n_samples=-1, transform=None, batch_size=8):
    """Learn soft environment assignment."""
    n_env = len(envs)    
    image_train_paths = torch.cat([envs[i]['images'][:n_samples] for i in range(n_env-1)],0)    
    label_train = torch.cat([envs[i]['labels'][:n_samples] for i in range(n_env-1)],0)
    print('size of pooled envs: '+str(len(image_train_paths)))
    
    train_dataloader = simple_dataloader(image_train_paths,label_train,batch_size,transform)
    scale = torch.tensor(1.).cuda().requires_grad_()

    #TODO: batch loop training
    logits_all = []
    loss_all = []
    for i, (images, labels) in enumerate(train_dataloader):
        logits = model(images)
        loss = nll(logits * scale, labels, reduction='none')
        logits_all.append(logits)
        loss_all.append(loss)
    
    logits = torch.cat(logits_all,0)
    loss = torch.cat(loss_all,0)
    env_w = torch.randn(len(logits)).cuda().requires_grad_()
    optimizer = optim.Adam([env_w], lr=0.001)

    print('learning soft environment assignments')
    penalties = []
    #TODO: make multinomial instead of binomial
    for i in tqdm(range(n_steps)):
        # penalty for env a
        lossa = (loss.squeeze() * env_w.sigmoid()).mean()
        grada = autograd.grad(lossa, [scale], create_graph=True)[0]
        penaltya = torch.sum(grada**2)
        # penalty for env b
        lossb = (loss.squeeze() * (1-env_w.sigmoid())).mean()
        gradb = autograd.grad(lossb, [scale], create_graph=True)[0]
        penaltyb = torch.sum(gradb**2)
        # negate
        npenalty = - torch.stack([penaltya, penaltyb]).mean()

        optimizer.zero_grad()
        npenalty.backward(retain_graph=True)
        optimizer.step()

    # split envs based on env_w threshold
    new_envs = []
    idx0 = (env_w.sigmoid()>.5)
    idx1 = (env_w.sigmoid()<=.5)
    # train envs
    for idx in (idx0, idx1):
        new_envs.append(dict(images=images[idx], labels=labels[idx]))
    # test env
    new_envs.append(dict(images=envs[-1]['images'],
                        labels=envs[-1]['labels']))

    print('size of env0: '+str(len(new_envs[0]['images'])))
    print('size of env1: '+str(len(new_envs[1]['images'])))
    print('size of env2: '+str(len(new_envs[2]['images'])))
    return new_envs


def run_eiil(flags, train_dataloader, val_dataloader, test_dataloader, transform):
    final_train_accs = []
    final_test_accs = []
    for restart in range(flags.n_restarts):
        print("Restart", restart)

        # Build environments: if training vgg_pre then use single env with all data        
        envs = make_environment(flags.data_dir)
        
        # Instantiate the model
        vgg_pre = VGG().cuda()
        vgg = VGG().cuda()

        optimizer_pre = optim.Adam(vgg_pre.parameters(), lr=flags.lr)
        optimizer = optim.Adam(vgg.parameters(), lr=flags.lr)

        pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

        if flags.eiil:
            vgg_pre = pretrain_model(envs,vgg_pre, optimizer_pre)
            envs = split_data_opt(envs, vgg_pre, transform)

        for step in range(flags.steps):
            for env in envs:
                #TODO: create batch loop
                logits = vgg(env['images'])
                env['nll'] = nll(logits, env['labels'])
                env['acc'] = mean_accuracy(logits, env['labels'])
                env['penalty'] = penalty(logits, env['labels'])

            train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
            train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
            train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

            weight_norm = torch.tensor(0.).cuda()
            for w in vgg.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm
            penalty_weight = (flags.penalty_weight
                if step >= flags.penalty_anneal_iters else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_acc = envs[2]['acc']
            if step % 100 == 0:
                pretty_print(
                    np.int32(step),
                    train_nll.detach().cpu().numpy(),
                    train_acc.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                    test_acc.detach().cpu().numpy()
                )

        final_train_accs.append(train_acc.detach().cpu().numpy())
        final_test_accs.append(test_acc.detach().cpu().numpy())
        print('Final train acc (mean/std across restarts so far):')
        print(np.mean(final_train_accs), np.std(final_train_accs))
        print('Final test acc (mean/std across restarts so far):')
        print(np.mean(final_test_accs), np.std(final_test_accs))



def main():

    print("Parsing arguments...")
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/sean/Projects/deep/dataset')
    parser.add_argument('--batch_size', default=8, type=int)
    # parser.add_argument('--max_epochs', default=15, type=int)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--input_shape', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--format', type=str, default='nifti')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--cropped', type=bool, default=True)
    parser.add_argument('--batch_norm', type=bool, default=False)
    parser.add_argument('--augment', nargs='*')
    parser.add_argument('--cfg_name', type=str, default='A')
    parser.add_argument('--classifier_cfg', type=str, default='A')

    # EIIL params
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=5001)
    parser.add_argument('--grayscale_model', action='store_true')
    parser.add_argument('--eiil', action='store_true')

    args = parser.parse_args()

    print("Starting Wandb...")
    # project_name = f"deep-{'multi_class' if args.num_classes > 2 else 'binary'}"

    # mode = "disabled" if args.debug else "dryrun"
    # wandb.init(mode=mode, project=project_name, name=f"{args.name}-{args.cfg_name}",
    #            settings=wandb.Settings(start_method="fork"))
    # wandb_logger = WandbLogger()
    # mask = ''

    if args.num_samples == -1:
        args.num_samples = -1 * args.num_classes

    if isinstance(args.input_shape,int):
        input_shape = args.input_shape
        args.input_shape = (args.input_shape,
                            args.input_shape,
                            args.input_shape)

    # these are returned shuffled
    file_paths, labels = get_mri_data_beta(args.num_samples, args.num_classes, args.data_dir, cropped=args.cropped)
    mask = ''
    
    dm = MRIDataModuleIO(args.data_dir, labels, args.format, args.batch_size, args.augment, mask, file_paths,
                         args.num_workers, args.input_shape)
    dm.setup(stage='fit')
    train_dataloader = dm.train_dataloader
    val_dataloader = dm.val_dataloader
    test_dataloader = dm.test_dataloader

    preprocess = Compose(
    [
        ScaleIntensity(),
        AddChannel(),
        ResizeWithPadOrCrop(input_shape),
        EnsureType(),
    ])
    run_eiil(args,train_dataloader, val_dataloader, test_dataloader, preprocess)

if __name__ == '__main__':
    main()
