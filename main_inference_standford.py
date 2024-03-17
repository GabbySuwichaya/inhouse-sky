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

cwd = "/mnt/4E280AEC280AD33F/Projects/CU/CUEE_SkyImager/SkyGPT"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_name = 'PhyDNetGAN'


data_folder = os.path.join(cwd,'data')
data_path   = os.path.join(data_folder,'video_prediction_dataset.hdf5')

images_log_train, images_pred_train, images_log_test, images_pred_test, times_curr_test  = train_test_split(data_folder,data_path) 
data_set    = Dataset([images_log_test, images_pred_test])
test_loader = torch.utils.data.DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=0)
 

phycell  = PhyCell(input_shape=(16,16), input_dim=64, F_hidden_dims=[49], n_layers=1, kernel_size=(7,7), device=device) 
convcell = ConvLSTM(input_shape=(16,16), input_dim=64, hidden_dims=[128,128,64], n_layers=3, kernel_size=(3,3), device=device)   
encoder  = EncoderRNN(phycell, convcell, device)
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

encoder.load_state_dict(torch.load('save/{0}/encoder.pth'.format(save_name)))
encoder.eval()
mse, mae, predictions, indices = evaluate(encoder,test_loader)

print(predictions.shape)
predictions = predictions.transpose((0,1,3,4,2))
print(predictions.shape)

np.save('save/{0}/predicted_images.npy'.format(save_name), predictions)

random.seed(0)
select_num_samples = 30
select_idx = random.sample(np.arange(len(times_curr_test)).tolist(),select_num_samples)

# %%
for i in range(select_num_samples):
    print("-"*50,"sample ",str(i+1), "-"*50)
    f, ax = plt.subplots(2,8)
    f.set_size_inches(24,6)
    ax[0,0].imshow(images_log_test[select_idx[i]][0][:,:,::-1])
    ax[0,0].set_title(times_curr_test[select_idx[i]]-datetime.timedelta(minutes=15))
    ax[0,1].imshow(images_log_test[select_idx[i]][2][:,:,::-1])
    ax[0,1].set_title(times_curr_test[select_idx[i]]-datetime.timedelta(minutes=11))
    ax[0,2].imshow(images_log_test[select_idx[i]][4][:,:,::-1])
    ax[0,2].set_title(times_curr_test[select_idx[i]]-datetime.timedelta(minutes=7))
    ax[0,3].imshow(images_log_test[select_idx[i]][7][:,:,::-1])
    ax[0,3].set_title(times_curr_test[select_idx[i]]-datetime.timedelta(minutes=1))
    ax[0,4].imshow(images_pred_test[select_idx[i]][0][:,:,::-1])
    ax[0,4].set_title(times_curr_test[select_idx[i]]+datetime.timedelta(minutes=1))
    ax[0,5].imshow(images_pred_test[select_idx[i]][2][:,:,::-1])
    ax[0,5].set_title(times_curr_test[select_idx[i]]+datetime.timedelta(minutes=5))
    ax[0,6].imshow(images_pred_test[select_idx[i]][4][:,:,::-1])
    ax[0,6].set_title(times_curr_test[select_idx[i]]+datetime.timedelta(minutes=9))
    ax[0,7].imshow(images_pred_test[select_idx[i]][7][:,:,::-1])
    ax[0,7].set_title(times_curr_test[select_idx[i]]+datetime.timedelta(minutes=15))
    
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