from cv2 import cv2
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser
from skimage.io import imread, imshow
from skimage.transform import resize
import torchvision.transforms as transforms
from nipy.labs.viz import plot_map, mni_sform, coord_transform

sys.path.insert(1, '/Users/sean/Projects/deep/src/models')
#sys.path.insert(1, '../models')

from eiil import *
from dataloader import *

def get_activations(name):
    def hook(model, input, output):
        activation = output.detach()
        return activation
    return hook

def gradcam(test_data_loader,model,device,N=5):
    fig, ax = plt.subplots(nrows = 40, ncols = 1, figsize = (300,300))
    counter_rows = 0

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for i in range(N):
        model.train()
        # get the image from the dataloader
        obj = next(iter(test_data_loader))
        image = obj[0]
        label = obj[1]
        #print("Target:", target)
        image.requires_grad=True
        
        # register forward hook        
        model.features.register_forward_hook(get_activations('features'))

        # get the most likely prediction of the model
        pred = model(image.to(device),1)

        # get the gradient of the output with respect to the parameters of the model
        pred.backward()

        # pull the gradients out of the model
        gradients = model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])

        # get the activations of the last convolutional layer
        activations = model.get_activations(image.to(device)).detach()

        # weight the channels by corresponding gradients
        for i in range(64):
            activations[:, i, :, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        #heatmap = np.maximum(heatmap.cpu(), 0)
        
        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        fig, ax = plt.subplots(nrows = 4, ncols = 3, figsize = (20,10))
        fig.tight_layout()        
        heatmap = torch.mean(heatmap,0)
        heatmap_2d = heatmap           
        for i in range(4):
            slice_2d = image[0,:,:,:,np.random.randint(40,70)].detach().cpu()             
            temp_heat = cv2.resize(np.float32(heatmap_2d), (128, 128))
            temp_heat = np.uint8(255 * temp_heat)
            temp_heat = cv2.applyColorMap(temp_heat, cv2.COLORMAP_JET)
            #superimposed_img = temp_heat * 0.001 + inv_normalize(img[0].cpu()).permute(1,2,0).numpy() * 1.5
            superimposed_img = temp_heat* 0.001 + slice_2d.permute(1,2,0).numpy() * 1.5
            superimposed_img = np.clip(superimposed_img, 0, 1)
            ax[i,0].imshow(superimposed_img)#.astype(np.uint8))
            ax[i,1].set_title("Actual:%i | Predicted:%i" %(np.int(label), np.int(np.round(pred[0].cpu().detach()))), fontsize=12)
            ax[i,1].matshow(heatmap_2d.squeeze())
            ax[i,2].imshow(slice_2d.permute(1,2,0))           
        plt.show()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/sean/Projects/deep/dataset')
    parser.add_argument('--checkpointpath', type=str, default='/Users/sean/Projects/deep/models')
    #parser.add_argument('--data_dir', type=str, default='/scratch/spinney/enigma_drug/data')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_classes', type=int, default=2)
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

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(args.input_shape,int):
        input_shape = args.input_shape
        args.input_shape = (args.input_shape,
                            args.input_shape,
                            args.input_shape)

    # load model 
    model = VGG(args).to(device)
    if args.checkpointpath:
        model_path = os.path.join(args.checkpointpath,'vgg.pt')
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])        
    
    # the preprocessing when loading images
    transform = Compose(
    [
        ScaleIntensity(),
        AddChannel(),
        ResizeWithPadOrCrop(input_shape),
        EnsureType(),
    ])

    # load test data loader 
    envs = make_environment(args)
    test_dataloader, pos_weight = simple_dataloader(envs[-1]['images'],envs[-1]['labels'],1,transform)
    model.eval()

    gradcam(test_dataloader,model,device)


