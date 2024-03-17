import torch.nn.functional as F 
import numpy as np  
import math
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import itertools
from tqdm import tqdm
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# day block shuffling of the time stamps, and return shuffled indices
def day_block_shuffle(times_trainval):
    
    # Only keep the date of each time point
    dates_trainval = np.zeros_like(times_trainval, dtype=datetime.date)
    for i in range(len(times_trainval)):
        dates_trainval[i] = times_trainval[i].date()

    # Chop the indices into blocks, so that each block contains the indices of the same day
    unique_dates = np.unique(dates_trainval)
    blocks = []
    for i in range(len(unique_dates)):
        blocks.append(np.where(dates_trainval == unique_dates[i])[0])

    # shuffle the blocks, and chain it back together
    np.random.seed(1)
    np.random.shuffle(blocks)
    shuffled_indices = np.asarray(list(itertools.chain.from_iterable(blocks)))

    return shuffled_indices

 
def trainval_split(split_data, split_ratio):
    '''
    input:
    split_data: the dayblock shuffled indices to be splitted
    fold_index: the ith fold chosen as the validation, used for generating the seed for random shuffling
    num_fold: N-fold cross validation
    output:
    data_train: the train data indices
    data_val: the validation data indices
    '''
    # randomly divides into a training set and a validation set
    num_samples = len(split_data[0])
    indices = np.arange(num_samples)

    # finding training and validation indices
    val_mask = np.zeros(len(indices), dtype=bool)
    val_mask[:int(split_ratio * num_samples)] = True
    val_indices = indices[val_mask]
    train_indices = indices[np.logical_not(val_mask)]

    # shuffle indices
    np.random.seed(0)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    # Initialize the training and validation data set list
    data_train = []
    data_val = []
    # obtain training and validation data
    for one_data in split_data:
        one_train, one_val = one_data[train_indices], one_data[val_indices]
        data_train.append(one_train)
        data_val.append(one_val)

    return data_train,data_val 
 
# %%
def plot_gen_images(predictions, images_log, images_pred, select_idx, epoch, times_curr=None, name=None):
    predictions = predictions.transpose((0,1,3,4,2))
    images_log = images_log.transpose((0,1,3,4,2))
    images_pred = images_pred.transpose((0,1,3,4,2))

    for i in range(len(select_idx)):
        #print("-"*50,"sample ",str(i+1), "-"*50)
        f, ax = plt.subplots(2,8)
        f.subplots_adjust(wspace=0, hspace=0)
        f.set_size_inches(24,6)
        if times_curr is not None:
            ax[0,0].imshow(images_log[select_idx[i]][0][:,:,::-1])
            ax[0,0].set_title(times_curr[select_idx[i]]-datetime.timedelta(minutes=14))
            ax[0,1].imshow(images_log[select_idx[i]][2][:,:,::-1])
            ax[0,1].set_title(times_curr[select_idx[i]]-datetime.timedelta(minutes=10))
            ax[0,2].imshow(images_log[select_idx[i]][4][:,:,::-1])
            ax[0,2].set_title(times_curr[select_idx[i]]-datetime.timedelta(minutes=6))
            ax[0,3].imshow(images_log[select_idx[i]][7][:,:,::-1])
            ax[0,3].set_title(times_curr[select_idx[i]])
            ax[0,4].imshow(images_pred[select_idx[i]][0][:,:,::-1])
            ax[0,4].set_title(times_curr[select_idx[i]]+datetime.timedelta(minutes=1))
            ax[0,5].imshow(images_pred[select_idx[i]][2][:,:,::-1])
            ax[0,5].set_title(times_curr[select_idx[i]]+datetime.timedelta(minutes=5))
            ax[0,6].imshow(images_pred[select_idx[i]][4][:,:,::-1])
            ax[0,6].set_title(times_curr[select_idx[i]]+datetime.timedelta(minutes=9))
            ax[0,7].imshow(images_pred[select_idx[i]][7][:,:,::-1])
            ax[0,7].set_title(times_curr[select_idx[i]]+datetime.timedelta(minutes=15))

        else: 
            ax[0,0].imshow(images_log[select_idx[i],0,:,:,:])
            ax[0,0].set_title("$X[t=0]$")
            ax[0,1].imshow(images_log[select_idx[i],2,:,:,:])
            ax[0,1].set_title("$X[t=2]$")
            ax[0,2].imshow(images_log[select_idx[i],4,:,:,:])
            ax[0,2].set_title("$X[t=4]$")
            ax[0,3].imshow(images_log[select_idx[i],7,:,:,:])
            ax[0,3].set_title("$X[t=7]$")
            ax[0,4].imshow(images_pred[select_idx[i],0,:,:,:])
            ax[0,4].set_title("$Y_{gt}[t=0]$")
            ax[0,5].imshow(images_pred[select_idx[i],2,:,:,:])
            ax[0,5].set_title("$Y_{gt}[t=2]$")
            ax[0,6].imshow(images_pred[select_idx[i],4,:,:,:])
            ax[0,6].set_title("$Y_{gt}[t=4]$")
            ax[0,7].imshow(images_pred[select_idx[i],7,:,:,:])
            ax[0,7].set_title("$Y_{gt}[t=7]$")         

        ax[1,4].imshow(predictions[select_idx[i], 0, :, :, :])
        ax[1,4].set_title("$Y_{pred}[t=0]$")
        ax[1,5].imshow(predictions[select_idx[i], 2, :, :, :])
        ax[1,5].set_title("$Y_{pred}[t=2]$")
        ax[1,6].imshow(predictions[select_idx[i], 4, :, :, :])
        ax[1,6].set_title("$Y_{pred}[t=4]$")
        ax[1,7].imshow(predictions[select_idx[i], 7, :, :, :])
        ax[1,7].set_title("$Y_{pred}[t=7]$")

   
        ax[0,0].axis('off')
        ax[0,1].axis('off')
        ax[0,2].axis('off')
        ax[0,3].axis('off')
        ax[0,4].axis('off')
        ax[0,5].axis('off')
        ax[0,6].axis('off')
        ax[0,7].axis('off')
        ax[1,0].axis('off')
        ax[1,1].axis('off')
        ax[1,2].axis('off')
        ax[1,3].axis('off')
        ax[1,4].axis('off')
        ax[1,5].axis('off')
        ax[1,6].axis('off')
        ax[1,7].axis('off')

        if name is not None:
            f.savefig("save/%s/figures/image_%d.png" % (name, epoch)) 
        else:
            f.savefig("save/image_%d.png" % epoch) 




 