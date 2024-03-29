import argparse
import numpy as np
import torch
#from torchvision import datasets
from torch import nn, optim, autograd
import pdb
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import gc

from dataloader import *

def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if isinstance(v, int):
            v = str(v)

        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))

# Define loss function helpers
def nll(logits, y, pos_weight, reduction='mean'):
    #TODO: cross entropy not binary
    logits = torch.squeeze(logits)
    y = torch.squeeze(y).float()
    #print(f"Logits shape: {logits.shape}")
    #print(f"y shape: {y.shape}")
    #print(f"pos_weight shape: {pos_weight.shape}")
    #return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction=reduction)
    return nn.functional.binary_cross_entropy_with_logits(logits, y,  pos_weight=pos_weight, reduction=reduction)

def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y, pos_weight, device):
    scale = torch.tensor(1.).to(device).requires_grad_()
    #loss = nll(logits * scale, y, pos_weight)
    loss = nll(logits * scale, y, pos_weight)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def pretrain_model(flags,envs,model,optimizer_pre,scheduler_pre,batch_size,transform):
    n_env = len(envs)
    all_images = np.concatenate([envs[0]['images'],envs[1]['images']])
    all_labels = np.concatenate([envs[0]['labels'],envs[1]['labels']])
    train_dataloader, pos_weight = simple_dataloader(all_images,all_labels,batch_size,transform,flags.num_workers)
    #train_dataloader1, pos_weight1 = simple_dataloader(envs[0]['images'],envs[0]['labels'],batch_size//2,transform,flags.num_workers)
    #train_dataloader2, pos_weight2 = simple_dataloader(envs[1]['images'],envs[1]['labels'],batch_size//2,transform,flags.num_workers)
    test_dataloader, pos_weight_test = simple_dataloader(envs[-1]['images'],envs[-1]['labels'],batch_size,transform,flags.num_workers,True)
    gradient_accumulations = 1
    for step in range(1):            
        for i, (images1, labels1) in enumerate(train_dataloader):
            print(f"Batch num: {i}")
            images1, labels1 = images1.to(flags.device), labels1.to(flags.device)
            #images2, labels2 = images2.to(flags.device), labels2.to(flags.device)
            logits1 = model(images1)
            #logits2 = model(images2)

            #TODO: fix test set    
            #logits_env = [logits1,logits2,logits2]
            #labels_env = [labels1,labels2,labels2]
                     
        
            # logits = torch.cat([logits1,logits2],0)
            # labels = torch.cat([labels1,labels2],0)
            # for e in range(n_env):
            #     envs[e]['nll'] = nll(logits_env[e], labels_env[e], pos_weight[e])
            #     envs[e]['acc'] = mean_accuracy(logits_env[e], labels_env[e])
            #     envs[e]['penalty'] = penalty(logits_env[e], labels_env[e], pos_weight[e], flags.device)
            train_nll = nll(logits1, labels1, pos_weight)
            train_acc = mean_accuracy(logits1, labels1)
            #train_penalty = penalty(logits1, labels1, pos_weight, flags.device)
            
            #train_nll = torch.stack([envs[i]['nll'] for i in range(n_env-1)]).mean()
            #train_acc = torch.stack([envs[i]['acc'] for i in range(n_env-1)]).mean()
            #train_penalty = torch.stack([envs[i]['penalty'] for i in range(n_env-1)]).mean()

            weight_norm = torch.tensor(0.).to(flags.device)
            for w in model.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm
            # NOTE: IRM penalties not used in pre-training
            
            
            optimizer_pre.zero_grad()
            #train_nll.backward()          
            loss.backward()          
            optimizer_pre.step()     
            scheduler_pre.step()                           
            
            pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            #train_penalty.detach().cpu().numpy(),
            #test_acc#.detach().cpu().numpy()
            )

        # test 
        if (step+1) % gradient_accumulations == 0:                 
            for i, (images, labels) in enumerate(test_dataloader):     
                images, labels = images.to(flags.device), labels.to(flags.device)  
                logits = model(images)
                test_acc = mean_accuracy(logits, labels)
                print(f"Batch: {i}, Test acc: {test_acc}")
                print(f"Labels: {labels}")
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


def split_data_opt(envs, model, device, n_steps=10000, n_samples=-1, transform=None, batch_size=8, num_workers=0):
    """Learn soft environment assignment."""
    n_env = len(envs)    
    image_train_paths = np.concatenate([envs[i]['images'][:n_samples] for i in range(n_env-1)],0)    
    label_train = np.concatenate([envs[i]['labels'][:n_samples] for i in range(n_env-1)],0)
    print('size of pooled envs: '+str(len(image_train_paths)))
    
    train_dataloader, pos_weight = simple_dataloader(image_train_paths,label_train,batch_size,transform,num_workers,test=True)
    scale = torch.tensor(1.).to(device).requires_grad_()
    
    logits_all = []
    loss_all = []
    
    # logits = torch.cat(logits_all,0).to(device).requires_grad_()
    # loss = torch.cat(loss_all,0).to(device).requires_grad_()
    gradient_accumulation_steps = 32    
    w = []
    for i, (images, labels) in enumerate(train_dataloader):
        print(f"Batch : {i} of {len(train_dataloader)}")
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = nll(logits * scale, labels, pos_weight,reduction='none')
        #loss = loss / gradient_accumulation_steps
#        logits_all.append(logits.detach().cpu())
#        loss_all.append(loss.detach().cpu())
        env_w = torch.randn(len(logits)).to(device).requires_grad_()
        optimizer = optim.Adam([env_w], lr=0.001, weight_decay=1e-4)
        print('learning soft environment assignments')
        for k in tqdm(range(n_steps)):
            print(f"Step: {k}")
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
        w.append(env_w.detach().cpu().numpy())
        #if i % gradient_accumulation_steps == 0:
            
    # split envs based on env_w threshold
    w = torch.tensor(np.concatenate(w,0))
    new_envs = []
    idx0 = (w.sigmoid()>.5)
    idx1 = (w.sigmoid()<=.5)
    # train envs
    for idx in (idx0, idx1):
        new_envs.append(dict(images=image_train_paths[idx], labels=label_train[idx]))
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

        optimizer_pre = optim.Adam(vgg_pre.parameters(), lr=flags.lr, weight_decay=1e-4)
        scheduler_pre = CosineAnnealingLR(optimizer_pre, T_max=10, eta_min=0)
        #optimizer = optim.Adam(vgg.parameters(), lr=flags.lr)       

        #if flags.eiil:
        if True: # flags,envs,model,optimizer_pre,batch_size,transform
            vgg_pre = pretrain_model(flags,envs,vgg_pre, optimizer_pre, scheduler_pre, flags.batch_size, transform)
            envs = split_data_opt(envs, vgg_pre, flags.device, flags.steps, -1,transform, flags.batch_size, flags.num_workers)
      
        torch.cuda.empty_cache()
        vgg = VGG(flags).to(flags.device)
        optimizer = optim.Adam(vgg.parameters(), lr=flags.lr)
        scheduler = CosineAnnealingLR(optimizer_pre, T_max=10, eta_min=0)   

        train_dataloader1, pos_weight1 = simple_dataloader(envs[0]['images'],envs[0]['labels'],flags.batch_size//2,transform,flags.num_workers)
        train_dataloader2, pos_weight2 = simple_dataloader(envs[1]['images'],envs[1]['labels'],flags.batch_size//2,transform,flags.num_workers)
        test_dataloader, pos_weight_test = simple_dataloader(envs[-1]['images'],envs[-1]['labels'],flags.batch_size,transform,flags.num_workers,test=True)
        pos_weight = [pos_weight1,pos_weight2,pos_weight_test]
        for step in range(flags.steps):
            print(f"IRM Step: {step}")
            train_acc_batch=[]
            for i, ((images1, labels1),(images2, labels2)) in enumerate(zip(train_dataloader1,train_dataloader2)):
                print(f"Batch: {i} of {len(train_dataloader1)}")
                images1, labels1 = images1.to(flags.device), labels1.to(flags.device)
                images2, labels2 = images2.to(flags.device), labels2.to(flags.device)
                logits1 = vgg(images1)
                logits2 = vgg(images2)

                logits_env = [logits1,logits2,logits2]
                labels_env = [labels1,labels2,labels2]
                        
                # logits = torch.cat([logits1,logits2],0)
                # labels = torch.cat([labels1,labels2],0)
                
                for e in range(len(envs)):
                    envs[e]['nll'] = nll(logits_env[e], labels_env[e], pos_weight[e])
                    envs[e]['acc'] = mean_accuracy(logits_env[e], labels_env[e])
                    envs[e]['penalty'] = penalty(logits_env[e], labels_env[e], pos_weight[e], flags.device)
                                
                logits = torch.cat(logits_env,0)
                labels = torch.cat(labels_env,0)
                # env['nll'] = nll(logits, labels, pos_weight)
                # env['acc'] = mean_accuracy(logits, labels)
                # env['penalty'] = penalty(logits, labels, pos_weight,flags.device)

                train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
                train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
                train_acc_batch.append(train_acc.detach().cpu().numpy())
                train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

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
                scheduler.step()

                pretty_print(
                    np.int32(step),
                    train_nll.detach().cpu().numpy(),
                    train_acc.detach().cpu().numpy(),
                    train_penalty.detach().cpu().numpy(),
                    #test_acc.detach().cpu().numpy()
                )

            final_train_accs.append(train_acc_batch)        
            print('Final train acc (mean/std across restarts so far):')
            print(np.mean(final_train_accs), np.std(final_train_accs))
            # test 
            if (step+1) % 1 == 0:                 
                test_acc_batch=[]
                for i, (images, labels) in enumerate(test_dataloader):     
                    images, labels = images.to(flags.device), labels.to(flags.device)  
                    logits = vgg(images)
                    test_acc = mean_accuracy(logits, labels)
                    print(f"Batch: {i}, Test acc: {test_acc}")
                    print(f"Labels: {labels}")                
                final_test_accs.append(test_acc_batch)
                print('Final test acc (mean/std across restarts so far):')
                print(np.mean(final_test_accs), np.std(final_test_accs))

        


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

def main():

    print("Parsing arguments...")
    parser = ArgumentParser()
    #parser.add_argument('--data_dir', type=str, default='/Users/sean/Projects/deep/dataset')
    #parser.add_argument('--data_dir', type=str, default='/scratch/spinney/enigma_drug/data')
    parser.add_argument('--data_dir', type=str, default='/network/scratch/s/sean.spinney/deep/data')
    parser.add_argument('--batch_size', default=32, type=int)
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
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--penalty_anneal_iters', type=int, default=10)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--steps', type=int, default=20)
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

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preprocess = Compose(
    [
        ScaleIntensity(),
        AddChannel(),
        ResizeWithPadOrCrop(input_shape),
        EnsureType(),
    ])
    
    run_eiil(args, preprocess)

if __name__ == '__main__':
    main()
