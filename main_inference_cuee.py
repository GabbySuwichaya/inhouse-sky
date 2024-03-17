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

 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_name = 'PhyDNetGAN_CUEE'

h5path = "h5files-IRR-Frame-1x16-to-1x15-Mins_IMS-64-2024-03-15"
INPUTS_PATH   = os.path.join("CUEE_preprocessing", h5path) 


image_list_file_test   = os.path.join("CUEE_preprocessing", "Test_IRR_Tr0p80-Val0p05-Test0p15-Frame-1x16-to-1x15.txt") 
test_dataset    = Dataset(INPUTS_PATH, Image_list_file = image_list_file_test)  
test_loader      = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
 

phycell  = PhyCell(input_shape=(16,16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
convcell = ConvLSTM(input_shape=(16,16), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)   
encoder  = EncoderRNN(phycell, convcell, device)
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

encoder.load_state_dict(torch.load('save/{0}/encoder.pth'.format(save_name)))
encoder.eval()
mse, mae, predictions, indices, target_samples, input_samples = evaluate(encoder,test_loader) 

print(predictions.shape)
predictions = predictions.transpose((0,1,3,4,2))
target_samples = target_samples.transpose((0,1,3,4,2))
input_samples = input_samples.transpose((0,1,3,4,2))

print(predictions.shape)

np.save('save/{0}/predicted_images.npy'.format(save_name), predictions)
 
select_num_samples = 30
select_idx = random.sample(np.arange(len(target_samples)).tolist(),select_num_samples)

# %%
for i in range(select_num_samples):
    print("-"*50,"sample ",str(i+1), "-"*50)
    f, ax = plt.subplots(2,8)
    f.set_size_inches(24,6)
    ax[0,0].imshow(input_samples[select_idx[i]][0][:,:,::-1])
    ax[0,0].set_title("$X_0$")
    ax[0,1].imshow(input_samples[select_idx[i]][2][:,:,::-1])
    ax[0,1].set_title("$X_2$")
    ax[0,2].imshow(input_samples[select_idx[i]][4][:,:,::-1])
    ax[0,2].set_title("$X_4$")
    ax[0,3].imshow(input_samples[select_idx[i]][7][:,:,::-1])
    ax[0,3].set_title("$X_7$")
    ax[0,4].imshow(target_samples[select_idx[i]][0][:,:,::-1])
    ax[0,4].set_title("$Y_0$")
    ax[0,5].imshow(target_samples[select_idx[i]][2][:,:,::-1])
    ax[0,5].set_title("$Y_2$")
    ax[0,6].imshow(target_samples[select_idx[i]][4][:,:,::-1])
    ax[0,6].set_title("$Y_4$")
    ax[0,7].imshow(target_samples[select_idx[i]][7][:,:,::-1])
    ax[0,7].set_title("$Y_7$")
    
    ax[1,4].imshow(predictions[select_idx[i]][0][:,:,::-1])
    ax[1,5].imshow(predictions[select_idx[i]][2][:,:,::-1])
    ax[1,6].imshow(predictions[select_idx[i]][4][:,:,::-1])
    ax[1,7].imshow(predictions[select_idx[i]][7][:,:,::-1])
    
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
    
    if save_name is not None:
        f.savefig("save/%s/figures/prediction_%d.png" % (save_name, i)) 
    else:
        f.savefig("save/prediction_%d.png" % i) 