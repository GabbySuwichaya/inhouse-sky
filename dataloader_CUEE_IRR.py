# %%
import h5py
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
#from warp import *
import pdb
import glob
from math import log10, sqrt 
import matplotlib.pyplot as plt
import math
import os
import random

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def crop_center(img,cropx,cropy):
    _, y, x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    
    return img[:, starty:starty+cropy,startx:startx+cropx, :]


class Dataset(data.Dataset):

    def __init__(self, input_folder, model="PhyDNet", start_file_index = None, Image_list_file = None, subsample=None, is_crop=True, cropx=64, cropy=64):
        super(Dataset, self).__init__()

        if Image_list_file == None:
            input_files = glob.glob(input_folder + "/*.h5")
        else:
            my_file =  open(Image_list_file, "r")  
            data = my_file.read()  
            data_into_list = data.split("\n")  
            my_file.close()   
            input_files = [os.path.join(input_folder, "%s" % file_name )   for file_name in data_into_list if len(file_name) > 0] 

        input_files.sort() 

        if start_file_index is not None:
            input_files = input_files[start_file_index:]
        else:
            print("Restart to the input_files to run all %d samples" % len(input_files)) 

        total_length  = len(input_files)

        if subsample is not None:
            num_subsample          = int(subsample*total_length)   
            self.input_files       = random.choices(input_files, k = num_subsample) 
        else:
            self.input_files = input_files

        self.n_images = len(self.input_files) 


        self.crop = is_crop
        self.cropx = cropx
        self.cropy = cropy
        self.model = model
        self.channel = 3

    def dim_transform(self, X_image):
        N, H, W, C = X_image.shape

        if self.model == "PhyDNet":
            X_image = X_image.permute(0, 3, 1, 2) # from (B x N x H x W x C ) to (B x N x C x H x W ) 
        
        elif self.model == "SkyNet":
            X_image = X_image.permute(0, 3, 1, 2).view(-1, H, W) # from (B x N x H x W x C ) to (B x N x C x H x W ) 

        return X_image
    
    def rev_dim_transform(self, X_image):
        if self.model == "PhyDNet":
            if X_image.dim() > 4:
                B, N, C, H, W = X_image.shape
                X_image = X_image.permute(0, 1,  3, 4, 2) # from (B x N x C x H x W )  to (B x N x H x W x C ) 
            else: 
                N, C, H, W = X_image.shape
                X_image = X_image.permute(0, 2, 3, 1) # from (B x N x C x H x W )  to (B x N x H x W x C ) 
        
        elif self.model == "SkyNet":
            if X_image.dim() > 4:
                B, NC, H, W = X_image.shape
                X_image = X_image.view(B, -1, self.channel, H, W).permute(0, 1,  3, 4, 2)  # from (B x N x H x W x C ) to (B x N x C x H x W ) 
            else:
                NC, H, W = X_image.shape
                X_image = X_image.view(-1, self.channel, H, W).permute(0, 2, 3, 1) # from (B x N x H x W x C ) to (B x N x C x H x W ) 

        return X_image        

        
    def __getitem__(self, idx):
        try:
            h5file = h5py.File(self.input_files[idx], 'r') 
        except:  
            h5py._errors.unsilence_errors() 
            pdb.set_trace()
            print(self.input_files[idx] + " has issues \n")  
            pdb.set_trace()
            return None  
  
        X_image = h5file["X"] 
        Y_image = h5file["Y"]  

        X_image  = np.array(X_image)
        Y_image  = np.array(Y_image)  

        if self.crop:
            X_image = crop_center(X_image,self.cropx,self.cropy)
            Y_image = crop_center(Y_image,self.cropx,self.cropy)

 
        X_image = X_image.astype('float')/256     
        Y_image = Y_image.astype('float')/256     
 
        N, H, W, C = X_image.shape
 
        assert H == Y_image.shape[1]
        assert W == Y_image.shape[2]
        assert C == Y_image.shape[3]

        X_Irr = h5file["X_Irr"] 
        Y_Irr = h5file["Y_Irr"]  

        X_Irr  = np.array(X_Irr)
        Y_Irr  = np.array(Y_Irr)

        X_DateTime_Array = h5file["X_DateTime_Array"] 
        Y_DateTimeArray  = h5file["Y_DateTimeArray"]

        X_DateTime_Array  = np.array(X_DateTime_Array)
        Y_DateTimeArray   = np.array(Y_DateTimeArray)

        # convert to torch

        X_image = torch.from_numpy(X_image)
        Y_image = torch.from_numpy(Y_image) 

        X_Irr = torch.from_numpy(X_Irr)
        Y_Irr = torch.from_numpy(Y_Irr) 

        X_DateTime_Array = torch.from_numpy(X_DateTime_Array)
        Y_DateTimeArray = torch.from_numpy(Y_DateTimeArray)  

        X_image = self.dim_transform(X_image)
        Y_image = self.dim_transform(Y_image)

        X_image = X_image.contiguous().float()
        Y_image = Y_image.contiguous().float()  

        out = [idx, X_image, Y_image, X_Irr, Y_Irr, X_DateTime_Array, Y_DateTimeArray]
        return out

    def __len__(self):
        return self.n_images
     
    

def plot_patch(ax, X, title, color=None): 
    ax.imshow(X)  
    if color is not None:
        ax.set_title(title, color=color)  
    else:
        ax.set_title(title)
    ax.axis('equal') 
    ax.axis('off')

   


def plotXY(input_X, input_Y, Y_predict=None, savepath=None, text_description=None):
    # input_X  = B x N x H x W x 3
    # input_Y  = B x N x H x W x 3
 
    num_x = input_X.shape[0]
    num_y = input_Y.shape[0]

    num_row_x = math.ceil(num_x / 4) 
    num_row_y = math.ceil(num_y / 4)  
 

    if text_description is not None:    
        if len(text_description) == 0:
            txt_list = []
            for key, value in text_description.items():
                if key == "id":
                    txt_list.append( "%s: %d" % (key, value))
                else:
                    txt_list.append( "%s: %.2f" % (key, value)) 
            textstr_sampled = '\n'.join(txt_list)   
        else:
            x_description = text_description[0]
            y_description = text_description[1]

            if Y_predict is not None: 
                ypred_description = text_description[2]

    # check input
        
    plt.close("all")   
    fig, axs = plt.subplots(num_row_x, 4, figsize=(15, 15))   
    d = 1 
    for i in range(num_row_x):
        for j in range(4): 
            plot_patch( axs[i,j], input_X[d-1,:,:,:] , r"$X[t=%d], I_{irrad}=%s$" % (d, x_description[d-1])  ) 
            d= d + 1 
 
    plt.tight_layout()
    plt.savefig(savepath + "-X.png")   
    plt.close("all")   

    if Y_predict is None: 
        d = 1 
        fig, axs = plt.subplots(num_row_y, 4, figsize=(15, 15))   
        for i in range(num_row_y):
            for j in range(4): 
                if d <= input_Y.shape[0]:
                    plot_patch(axs[i,j], input_Y[d-1,:,:,:], r"$Y_{gt}[t=%d], I_{irrad}=%s$" % (d, y_description[d-1]) )  
                    d= d + 1
                else:
                    axs[i,j].axis('off')
        fig.tight_layout()
        plt.savefig(savepath + "-Y.png") 
        plt.close("all")  
         

    else: 

        d = 1 
        fig, axs = plt.subplots(2*num_row_y, 4, figsize=(15, 30))   
        for i in range(0, 2*num_row_x, 2):
            for j in range(4): 
                if d <= input_Y.shape[0]:
                    plot_patch(axs[i,j], input_Y[d-1,:,:,:], r"$Y_{gt}[t=%d], I_{irrad}=%s$" % (d, y_description[d-1]) )  
                    d= d + 1
                else:
                    axs[i,j].axis('off')

        d = 1 
        for i in range(1, 2*num_row_x, 2):
            for j in range(4): 
                if d <= input_Y.shape[0]:
                    plot_patch(axs[i,j], input_Y[d-1,:,:,:], r"$Y_{prd}[t=%d], I_{irrad}=%s$" % (d, ypred_description[d-1]), color="red")  
                    d= d + 1
                else:
                    axs[i,j].axis('off')
        fig.tight_layout()
        plt.savefig(savepath + "-Y.png")  


if __name__ == "__main__":

    h5path = "h5files-IRR-Frame-1x16-to-1x15-Mins_IMS-64-2024-03-15"
    INPUTS_PATH   = os.path.join("CUEE_preprocessing", h5path) 

    image_list_file_train  = os.path.join("CUEE_preprocessing", "Train_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt") 
    train_dataset    = Dataset(INPUTS_PATH, Image_list_file = image_list_file_train)
    train_dataloader      = DataLoader(train_dataset)

    image_list_file_valid  = os.path.join("CUEE_preprocessing", "Valid_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt") 
    valid_dataset    = Dataset(INPUTS_PATH, Image_list_file = image_list_file_valid)
    valid_dataloader = DataLoader(valid_dataset) 
    
    image_list_file_test   = os.path.join("CUEE_preprocessing", "Test_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt") 
    test_dataset    = Dataset(INPUTS_PATH, Image_list_file = image_list_file_test)
    test_dataloader = DataLoader(test_dataset)

    for data_ in test_dataloader:
        out_list = data_   
        
        ind = out_list[0]
        X_image  = out_list[1]
        Y_image  = out_list[2]

        X_Irr = out_list[3]
        Y_Irr = out_list[4]

        x_description = {}
        y_description = {} 

        X_Irr = X_Irr.squeeze(0)
        Y_Irr = Y_Irr.squeeze(0)
 
        for i in range(X_Irr.shape[0]):
            x_description[i] = "%d" % X_Irr[i]

        for i in range(Y_Irr.shape[0]):
            y_description[i] = "%d" % Y_Irr[i]
 
 
        X_image = test_dataset.rev_dim_transform(X_image)
        Y_image = test_dataset.rev_dim_transform(Y_image)   

        plotXY(X_image.squeeze(0), Y_image.squeeze(0), Y_image.squeeze(0), savepath="h2files-sample", text_description=[x_description, y_description, y_description])  # from 1 x N x H x W x 3  ==> N x H x W x 3 

        pdb.set_trace()