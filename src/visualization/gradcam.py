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

#sys.path.insert(1, '/Users/sean/Projects/deep/src/models')
sys.path.insert(1, '../models')

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
    
        # set the evaluation mode
        model.eval()

        # get the image from the dataloader
        img, target = next(iter(test_data_loader))
        #print("Target:", target)

        # register forward hook        
        model.features.register_forward_hook(get_activations('features'))

        # get the most likely prediction of the model
        pred = model(img.to(device),0)

        # get the gradient of the output with respect to the parameters of the model
        pred.backward()

        # pull the gradients out of the model
        gradients = model.get_activations_gradient()

        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.get_activations(img.to(device)).detach()

        # weight the channels by corresponding gradients
        for i in range(32):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)
        
        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        temp_heat = cv2.resize(np.float32(heatmap), (224, 224))
        temp_heat = np.uint8(255 * temp_heat)
        temp_heat = cv2.applyColorMap(temp_heat, cv2.COLORMAP_JET)
        superimposed_img = temp_heat * 0.001 + inv_normalize(img[0].cpu()).permute(1,2,0).numpy() * 1.5
        superimposed_img = np.clip(superimposed_img, 0, 1)
        ax[counter_rows].imshow(superimposed_img)#.astype(np.uint8))
        ax[counter_rows].set_title("Actual:%i | Predicted:%i" %(np.int(target), np.int(np.round(pred[0].cpu().detach()))), fontsize=20)
        #ax[counter_rows,0].matshow(heatmap.squeeze())
        #ax[counter_rows,1].imshow(inv_normalize(img[0]).permute(1,2,0))
        #ax[counter_rows,0].set_title("Predicted value:%i" %np.int(np.round(pred[0].cpu().detach())),fontsize=300)
        #ax[counter_rows,1].set_title("Actual value:%i" %np.int(target),fontsize=300)
        
        # draw the heatmap
        #plt.matshow(heatmap.squeeze())

        # plot original image
        #plt.imshow(inv_normalize(img[0]).permute(1,2,0))

        counter_rows = counter_rows + 1

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/sean/Projects/deep/dataset')
    parser.add_argument('--checkpointpath', type=str, default='~/somewhere')
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

    checkpointpath = ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if isinstance(args.input_shape,int):
        input_shape = args.input_shape
        args.input_shape = (args.input_shape,
                            args.input_shape,
                            args.input_shape)

    # load model 
    model = VGG(args).to(device)
    if checkpointpath:
        checkpoint = torch.load(checkpointpath, map_location="cpu")
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
    test_dataloader, pos_weight = simple_dataloader(envs[-1]['images'],envs[-1]['labels'],args.batch_size,transform)
    model.eval()

    gradcam(test_dataloader,model,device)


