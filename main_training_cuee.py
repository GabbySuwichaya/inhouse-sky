import torch 
import torch.nn.functional as F 
import numpy as np
import random 
import os
import h5py 
import numpy as np
import os 
import matplotlib.pyplot as plt
import datetime 
import pdb

from dataloader_CUEE_IRR import Dataset
from models_v2 import ConvLSTM, PhyCell, EncoderRNN 
from gan_utils import Discr_frame
from training_utils import  trainIters, reserve_schedule_sampling_exp, schedule_sampling, evaluate 
from utils import count_parameters
 


h5files_rootpath = "CUEE_preprocessing" 
h5path = "h5files-IRR-Frame-1x16-to-1x15-Mins_IMS-64-2024-03-15"

device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cwd         = os.getcwd() 

batch_size  = 16
nepochs     = 10
print_every = 1 
eval_every  = 1  

save_name   = 'PhyDNetGAN_CUEE'

os.makedirs('save', exist_ok=True)
os.makedirs('save/{}'.format(save_name), exist_ok=True)
os.makedirs('save/{}/figures'.format(save_name), exist_ok=True)
 

lamda = 0.01 # weight for generator adversarial loss
training_discriminator_every=1  

INPUTS_PATH     = os.path.join(h5files_rootpath,h5path)   

image_list_file_train  = os.path.join("CUEE_preprocessing", "Train_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt") 
train_dataset    = Dataset(INPUTS_PATH, Image_list_file = image_list_file_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

image_list_file_valid  = os.path.join("CUEE_preprocessing", "Valid_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt") 
valid_dataset    = Dataset(INPUTS_PATH, Image_list_file = image_list_file_valid)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0) 

num_log_frame  = 16
num_pred_frame = 15

r_eta = np.zeros(nepochs)
eta   = np.zeros(nepochs)
for epoch in range(nepochs):
    r_eta[epoch],real_input_flag_encoder = reserve_schedule_sampling_exp(epoch,num_log_frame)
    eta[epoch],real_input_flag_decoder = schedule_sampling(epoch,num_pred_frame)
plt.plot(range(nepochs),r_eta,label='r_eta')
plt.plot(range(nepochs),eta,label='eta')
plt.legend()
 
phycell  = PhyCell(input_shape=(16,16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
convcell = ConvLSTM(input_shape=(16,16), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)   
encoder  = EncoderRNN(phycell, convcell, device)
discriminator_frame = Discr_frame().to(device)

print('phycell ' , count_parameters(phycell))    
print('convcell ' , count_parameters(convcell)) 
print('encoder ' , count_parameters(encoder)) 
print('discriminator_frame ', count_parameters(discriminator_frame)) 

print("batchsize : %d" % batch_size)

testing_input = [None,  None,  None ]
# %%
encoder_total_train_losses, encoder_ad_train_losses, discr_train_losses = trainIters(encoder,discriminator_frame, nepochs, train_loader, valid_loader, testing_input, print_every=print_every, eval_every=eval_every, name=save_name)

# %% [markdown]
# ### Save Predicted Images from Validation Set

# %%



