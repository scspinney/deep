from cv2 import cv2
import torch
import numpy as np
import matplotlib.plt as plt

def gradcam(test_data_loader,model,device):
    fig, ax = plt.subplots(nrows = 40, ncols = 1, figsize = (300,300))
    counter_rows = 0
    for i in range(0,40):
    
        # set the evaluation mode
        model.eval()

        # get the image from the dataloader
        img, target = next(iter(test_data_loader))
        #print("Target:", target)

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