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

from dataloader_standford import Dataset, train_test_split
from models_v2 import ConvLSTM, PhyCell, EncoderRNN 
from gan_utils import Discr_frame
from training_utils import  trainIters, reserve_schedule_sampling_exp, schedule_sampling, evaluate 
from utils import count_parameters

root_path = "/mnt/4E280AEC280AD33F/Projects/CU/CUEE_SkyImager/SkyGPT"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()
data_folder = os.path.join(root_path,"data") 
batch_size = 16
nepochs = 10
print_every = 1 
eval_every = 1  
save_name = 'PhyDNetGAN'

os.makedirs('save', exist_ok=True)
os.makedirs('save/{}'.format(save_name), exist_ok=True)
os.makedirs('save/{}/figures'.format(save_name), exist_ok=True)

lamda = 0.01 # weight for generator adversarial loss
training_discriminator_every=1 


data_folder = os.path.join(root_path,'data')
data_path   = os.path.join(data_folder,'video_prediction_dataset.hdf5')

images_log_train, images_pred_train, images_log_test, images_pred_test, times_curr_test  = train_test_split(data_folder,data_path) 
data_set     = Dataset([images_log_train, images_pred_train])
train_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, num_workers=0) 
data_set     = Dataset([images_log_test, images_pred_test])
test_loader  = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers=0)

num_log_frame  = images_log_train.shape[1]  
num_pred_frame = images_pred_train.shape[1]

  
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

testing_input = [times_curr_test, images_log_test, images_pred_test ]
# %%
encoder_total_train_losses, encoder_ad_train_losses, discr_train_losses = trainIters(encoder,discriminator_frame,nepochs,train_loader, test_loader, testing_input, print_every=print_every,eval_every=eval_every,name=save_name)

# %% [markdown]
# ### Save Predicted Images from Validation Set

# %%



