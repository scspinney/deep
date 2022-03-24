import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

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
def nll(logits, y, pos_weight, reduction='mean'):    
    logits = torch.squeeze(logits)
    y = torch.squeeze(y).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction=reduction)

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y, pos_weight, device):
    scale = torch.tensor(1.).to(device).requires_grad_()
    #loss = nll(logits * scale, y, pos_weight)
    loss = nll(logits * scale, y, pos_weight)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def pretrain_model(flags,envs,model,optimizer_pre,transform):
    n_env = len(envs)
    train_dataloader1, pos_weight1 = simple_dataloader(envs[0]['images'],envs[0]['labels'],flags.batch_size//2,transform)
    train_dataloader2, pos_weight2 = simple_dataloader(envs[1]['images'],envs[1]['labels'],flags.batch_size//2,transform)
    test_dataloader, pos_weight = simple_dataloader(envs[-1]['images'],envs[-1]['labels'],flags.batch_size,transform)
    for step in range(flags.pretrain_steps):    
        
        for i, ((images1, labels1), (images2, labels2)) in enumerate(zip(train_dataloader1,train_dataloader2)):
            #print(f"Batch num: {i}")
            images1, labels1 = images1.to(flags.device), labels1.to(flags.device)
            images2, labels2 = images2.to(flags.device), labels2.to(flags.device)
            logits1 = model(images1)
            logits2 = model(images2)
        
            envs[0]['nll'] = nll(logits1, labels1, pos_weight1)
            envs[0]['acc'] = mean_accuracy(logits1, labels1)
            envs[0]['penalty'] = penalty(logits1, labels1, pos_weight1, flags.device)
            envs[1]['nll'] = nll(logits2, labels2, pos_weight2)
            envs[1]['acc'] = mean_accuracy(logits2, labels2)
            envs[1]['penalty'] = penalty(logits2, labels2, pos_weight2, flags.device)

            train_nll = torch.stack([envs[i]['nll'] for i in range(n_env-1)]).mean()
            train_acc = torch.stack([envs[i]['acc'] for i in range(n_env-1)]).mean()
            train_penalty = torch.stack([envs[i]['penalty'] for i in range(n_env-1)]).mean()

            weight_norm = torch.tensor(0.).to(flags.device)
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm
            # NOTE: IRM penalties not used in pre-training

            optimizer_pre.zero_grad()
            loss.backward()
            optimizer_pre.step()

        # Test
        if True:
        #if step % 10 == 0:
            test_acc = []
            for i, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(flags.device), labels.to(flags.device)            
                logits = model(images)
                test_acc.append( mean_accuracy(logits, labels).detach().cpu())
                
            #test_acc = envs[-1]['acc']
            test_acc = np.mean(test_acc)
            
            pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            train_penalty.detach().cpu().numpy(),
            test_acc
            )
        
            wandb.log({
                "pre_vgg": 
                {"train_nll": train_nll.detach().cpu().numpy(), 
                "train_acc": train_acc.detach().cpu().numpy(),
                "train_penalty": train_penalty.detach().cpu().numpy(),
                "test_acc": test_acc}}, 
                step=step)

    return model   


class VGG(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.n_size = self._get_block_output(self.hparams.input_shape)

        if self.hparams.num_classes == 2:
            self.hparams.num_classes = 1

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
        for v in cfgs[self.hparams.cfg_name]:
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
        layers = classifiers[self.hparams.classifier_cfg]
        return nn.Sequential(  # nn.Linear(self.hparams.cfg[-2] * 7 * 7 * 7, self.hparams.classifier_cfg[0]),
            nn.Linear(self.n_size, layers[0]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(layers[0], layers[1]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(layers[1], self.hparams.num_classes))


    def _get_block_output(self, shape):
        self.eval()
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        input = torch.unsqueeze(input, 0)
        output_feat = self.features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        print(f"Block output size: {n_size}")
        return n_size


def split_data_opt(envs, model, device, batch_size, n_steps=10000, n_samples=-1, transform=None):
    """Learn soft environment assignment."""
    n_env = len(envs)    
    #image_train_paths = torch.cat([envs[i]['images'][:n_samples] for i in range(n_env-1)],0)    
    #label_train = torch.cat([envs[i]['labels'][:n_samples] for i in range(n_env-1)],0)
    image_train_paths = np.concatenate([envs[i]['images'][:n_samples] for i in range(n_env-1)],0)
    label_train = np.concatenate([envs[i]['labels'][:n_samples] for i in range(n_env-1)],0)
    print('size of pooled envs: '+str(len(image_train_paths)))
    
    train_dataloader, pos_weight = simple_dataloader(image_train_paths,label_train,batch_size,transform)
    scale = torch.tensor(1.).to(device).requires_grad_()
    
    logits_all = []
    loss_all = []
    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = nll(logits * scale, labels, pos_weight,reduction='none')
        logits_all.append(logits.detach().cpu())
        loss_all.append(loss.detach().cpu())
    
    logits = torch.cat(logits_all,0)
    loss = torch.cat(loss_all,0).to(device).requires_grad_()
    env_w = torch.randn(len(logits)).to(device).requires_grad_()
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


def run_eiil(flags, transform):
    final_train_accs = []
    final_test_accs = []
    print("Beginning EIIL")
    for restart in range(flags.n_restarts):
        print("Restart", restart)

        # Build environments: two groups with binary labelled data randomly assigned
        envs = make_environment(flags)
        
        # Instantiate the model
        vgg_pre = VGG(flags).to(flags.device)
        #vgg = VGG(flags).to(flags.device)

        optimizer_pre = optim.Adam(vgg_pre.parameters(), lr=flags.lr)
        #optimizer = optim.Adam(vgg.parameters(), lr=flags.lr)

        pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

        #if flags.eiil:
        if True: # flags,envs,model,optimizer_pre,batch_size,transform
            vgg_pre = pretrain_model(flags,envs,vgg_pre, optimizer_pre,transform)
            envs = split_data_opt(envs, vgg_pre, flags.device, flags.batch_size, 10, -1, transform)
      
        torch.cuda.empty_cache()
        vgg = VGG(flags).to(flags.device)
        optimizer = optim.Adam(vgg.parameters(), lr=flags.lr)
        train_dataloader1, pos_weight1 = simple_dataloader(envs[0]['images'],envs[0]['labels'],flags.batch_size//2,transform)
        train_dataloader2, pos_weight2 = simple_dataloader(envs[1]['images'],envs[1]['labels'],flags.batch_size//2,transform)
        test_dataloader, pos_weight = simple_dataloader(envs[-1]['images'],envs[-1]['labels'],flags.batch_size,transform)
        for step in range(flags.steps):                
            for i, ((images1, labels1), (images2, labels2)) in enumerate(zip(train_dataloader1,train_dataloader2)):
                #print(f"Batch num: {i}")
                images1, labels1 = images1.to(flags.device), labels1.to(flags.device)
                images2, labels2 = images2.to(flags.device), labels2.to(flags.device)
                logits1 = vgg(images1)
                logits2 = vgg(images2)
            
                envs[0]['nll'] = nll(logits1, labels1, pos_weight1)
                envs[0]['acc'] = mean_accuracy(logits1, labels1)
                envs[0]['penalty'] = penalty(logits1, labels1, pos_weight1, flags.device)
                envs[1]['nll'] = nll(logits2, labels2, pos_weight2)
                envs[1]['acc'] = mean_accuracy(logits2, labels2)
                envs[1]['penalty'] = penalty(logits2, labels2, pos_weight2, flags.device)

                train_nll = torch.stack([envs[i]['nll'] for i in range(n_env-1)]).mean()
                train_acc = torch.stack([envs[i]['acc'] for i in range(n_env-1)]).mean()
                train_penalty = torch.stack([envs[i]['penalty'] for i in range(n_env-1)]).mean()


                weight_norm = torch.tensor(0.).to(flags.device)
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

            if step % 10 == 0:
                test_acc = []
                for i, (images, labels) in enumerate(test_dataloader):
                    images, labels = images.to(flags.device), labels.to(flags.device)            
                    logits = vgg(images)
                    test_acc.append( mean_accuracy(logits, labels).detach().cpu())
                    
                #test_acc = envs[-1]['acc']
                test_acc = np.mean(test_acc)
                
                pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
                )
            
                wandb.log({
                    "vgg": 
                    {"train_nll": train_nll.detach().cpu().numpy(), 
                    "train_acc": train_acc.detach().cpu().numpy(),
                    "train_penalty": train_penalty.detach().cpu().numpy(),
                    "test_acc": test_acc.detach().cpu().numpy()}}, 
                    step=step)

        final_train_accs.append(train_acc.detach().cpu().numpy())
        final_test_accs.append(test_acc.detach().cpu().numpy())
        print('Final train acc (mean/std across restarts so far):')
        print(np.mean(final_train_accs), np.std(final_train_accs))
        print('Final test acc (mean/std across restarts so far):')
        print(np.mean(final_test_accs), np.std(final_test_accs))

        wandb.log({
            "vgg": 
            {"mean_final_train_acc": np.mean(final_train_accs), 
            "std_final_train_acc": np.std(final_train_accs),
            "mean_final_test_acc": np.mean(final_test_accs), 
            "std_final_test_acc": np.std(final_test_accs)}}, 
            step=step)


cfgs = {
    'A': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M'],
    'B': [8, 8, 'M', 8, 16, 'M', 16, 32, 32, 'M'],
    'C': [8, 8, 'M', 16, 16, 'M', 32, 32, 'M', 64, 64, 'M'],
    # 'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    # 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M'],
    'E': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
}

classifiers = {'A': [128, 64],
               'B': [256, 256],
               'C': [4096, 4096]}


if __name__ == '__main__':
    print("Parsing arguments...")
    parser = ArgumentParser()
    #parser.add_argument('--data_dir', type=str, default='/Users/sean/Projects/deep/dataset')
    parser.add_argument('--data_dir', type=str, default='/scratch/spinney/enigma_drug/data')
    parser.add_argument('--batch_size', default=16, type=int)
    # parser.add_argument('--max_epochs', default=15, type=int)
    parser.add_argument('--num_classes', type=int, default=2)
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
    parser.add_argument('--max_epochs', default=40, type=int)

    # model params
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--name', type=str, default='vggnet')
    parser.add_argument('--optim', type=str, default='adam')

    # EIIL params
    #parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--penalty_anneal_iters', type=int, default=100)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--pretrain_steps', type=int, default=1)
    parser.add_argument('--grayscale_model', action='store_true')
    parser.add_argument('--eiil', action='store_true')

    args = parser.parse_args()

    print("Starting Wandb...")
    project_name = f"deep-{'multi_class' if args.num_classes > 2 else 'binary'}"

    mode = "disabled" if args.debug else "online"
    wandb.init(mode=mode, entity="scspinney", project=project_name, name=f"{args.name}-{args.cfg_name}",
               settings=wandb.Settings(start_method="fork"))

    if args.num_samples == -1:
        args.num_samples = -1 * args.num_classes

    if isinstance(args.input_shape,int):
        input_shape = args.input_shape
        args.input_shape = (args.input_shape,
                            args.input_shape,
                            args.input_shape)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preprocess = Compose(
    [
        ScaleIntensity(),
        AddChannel(),
        ResizeWithPadOrCrop(input_shape),
        EnsureType(),
    ])

    run_eiil(args, preprocess)
