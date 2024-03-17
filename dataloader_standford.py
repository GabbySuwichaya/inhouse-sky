# %%
import torch
import torchvision
import torch.nn as nn 
import numpy as np 
import torch.utils.data as data 
import numpy as np
import os 
import pdb
import h5py


class Dataset(data.Dataset):
    def __init__(self, data_set, transform=None):
        self.data_set = data_set
        self.transform = transform
        self.length = self.data_set[0].shape[0]
        
    def __getitem__(self, idx):
        input_data = self.data_set[0][idx]
        output_data = self.data_set[1][idx]
        length = len(input_data)

        input_data = input_data.transpose(0, 3, 1, 2)
        output_data = output_data.transpose(0, 3, 1, 2) 

        output_data = torch.from_numpy(output_data / 255.0).contiguous().float()
        input_data = torch.from_numpy(input_data / 255.0).contiguous().float()
         

        out = [idx,input_data,output_data]
        return out

    def __len__(self):
        return self.length
     
    


def train_test_split(data_folder, data_path):

    print("data_folder:", data_folder)
    print("data_path:", data_path)

    # %%
    f = h5py.File(data_path, 'r')

    # %%
    f.keys()

    # %%
    trainval = f['trainval']
    test     = f['test']

    images_log_train = trainval['images_log'][:,::2,:,:,:]
    images_pred_train = trainval['images_pred'][:,::2,:,:,:]

    images_log_test = test['images_log'][:,::2,:,:,:]
    images_pred_test = test['images_pred'][:,::2,:,:,:]

    times_curr_train = np.load(os.path.join(data_folder,"times_curr_trainval.npy"),allow_pickle=True)
    times_curr_test = np.load(os.path.join(data_folder,"times_curr_test.npy"),allow_pickle=True)
 
    print('-'*50)
    print("times_curr_train.shape:", times_curr_train.shape)
    print("images_log_train.shape:", images_log_train.shape)
    print("images_pred_train.shape:", images_pred_train.shape) 
    print("times_curr_test.shape:", times_curr_test.shape)
    print("images_log_test.shape:", images_log_test.shape)
    print("images_pred_test.shape:", images_pred_test.shape)
    print('-'*50)
    # get the input dimension for constructing the model
    num_log_frame = images_log_train.shape[1]
    img_side_len  = images_log_train.shape[2]

    num_color_channel = images_log_train.shape[4]
    num_pred_frame = images_pred_train.shape[1]

    image_log_dim  = [num_log_frame,img_side_len,img_side_len,num_color_channel]
    image_pred_dim = [num_pred_frame,img_side_len,img_side_len,num_color_channel]

    print("image side length:", img_side_len)
    print("number of log frames:", num_log_frame)
    print("number of pred frames:", num_pred_frame)
    print("number of color channels:", num_color_channel)
    print("context(log) image dimension:", image_log_dim)
    print("future(pred) image dimension:", image_pred_dim)

    return images_log_train, images_pred_train, images_log_test, images_pred_test, times_curr_test
  