import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import time 
from constrain_moments import K2M 
import math
import numpy as np
import os 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pdb

from utils import plot_gen_images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
constraints = torch.zeros((49,7,7)).to(device)
ind = 0
for i in range(0,7):
    for j in range(0,7):
        constraints[ind,i,j] = 1
        ind +=1   

def reserve_schedule_sampling_exp(epoch, log_length, sampling_step_1 = 15, sampling_step_2 = 30, r_exp_alpha = 2.5):

    real_input_flag_encoder = np.zeros(log_length, dtype=bool)
    
    if epoch < sampling_step_1:
        r_eta = 0.5
    elif epoch < sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(epoch - sampling_step_1) / r_exp_alpha)
    else:
        r_eta = 1.0

    for i in range(log_length):
        real_input_flag_encoder[i] = True if random.random() < r_eta else False
    
    return r_eta, real_input_flag_encoder

def schedule_sampling(epoch,pred_length, sampling_step_1 = 15, sampling_step_2 = 30):
    
    real_input_flag_decoder = np.zeros(pred_length, dtype=bool)
    
    if epoch < sampling_step_1:
        eta = 0.5
    elif epoch < sampling_step_2:
        eta = 0.5 - (0.5 / (sampling_step_2 - sampling_step_1)) * (epoch - sampling_step_1)
    else:
        eta = 0
    
    for i in range(pred_length):
        real_input_flag_decoder[i] = True if random.random() < eta else False
    
    return eta, real_input_flag_decoder



def train_on_batch(epoch, input_tensor, target_tensor, encoder, encoder_optimizer, criterion_mae, criterion_mse, discr_frame, discr_frame_optimizer,training_discriminator_every=1, lamda = 0.01 ):                
    
    encoder_optimizer.zero_grad()
    
    # input_tensor : torch.Size([batch_size, input_length, channels, cols, rows])
    curr_batch_size = input_tensor.size(0)    
    input_length    = input_tensor.size(1)
    target_length   = target_tensor.size(1)
    loss = 0
    encoder_frame_ad_loss = 0
    real_label = torch.full((curr_batch_size,), 1, dtype=torch.float, device=device)
    fake_label = torch.full((curr_batch_size,), 1, dtype=torch.float, device=device)

    discr_frame_loss = 0
    r_eta,real_input_flag_encoder = reserve_schedule_sampling_exp(epoch,input_length)
    eta,real_input_flag_decoder   = schedule_sampling(epoch,target_length)
    
    encoder_input = input_tensor[:,0,:,:,:]
    for ei in range(input_length-1): 
        encoder_output, encoder_hidden, encoder_output_image,_,_ = encoder(encoder_input, (ei==0))
        encoder_target   = input_tensor[:,ei+1,:,:,:]
        loss            += criterion_mae(encoder_output_image,encoder_target)
        
        if real_input_flag_encoder[ei]:
            encoder_input = encoder_target # Teacher forcing    
        else:
            encoder_input = encoder_output_image
    
    if real_input_flag_encoder[-1]:        
        decoder_input = input_tensor[:,-1,:,:,:] 
    else:
        decoder_input = encoder_output_image # first decoder input = last image of input sequence
    
    for di in range(target_length):
        decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input)
        target = target_tensor[:,di,:,:,:]
        loss += criterion_mae(output_image,target)
        if (epoch+1)% training_discriminator_every==0:
            discr_frame_out_fake = discr_frame(output_image).view(-1)
            encoder_frame_ad_loss += 0.5 * torch.mean((discr_frame_out_fake - 1)**2)
        
        if real_input_flag_decoder[di]:
            decoder_input = target # Teacher forcing    
        else:
            decoder_input = output_image
    
    if (epoch+1)% training_discriminator_every==0:
        loss += lamda*encoder_frame_ad_loss
    
    # Moment regularization  # encoder.phycell.cell_list[0].F.conv1.weight # size (nb_filters,in_channels,7,7)
    k2m = K2M([7,7]).to(device)
    for b in range(0,encoder.phycell.cell_list[0].input_dim):
        filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:] # (nb_filters,7,7)     
        m = k2m(filters.double()) 
        m  = m.float()   
        loss += criterion_mse(m, constraints) # constrains is a precomputed matrix   
    loss.backward()
    encoder_optimizer.step()
    
    if (epoch+1)%training_discriminator_every==0:
        discr_frame_optimizer.zero_grad()
        if real_input_flag_encoder[-1]:        
            decoder_input = input_tensor[:,-1,:,:,:] 
        else:
            decoder_input = encoder_output_image # first decoder input = last image of input sequence
        for di in range(target_length):
            decoder_output, decoder_hidden, output_image,_,_ = encoder(decoder_input)
            target = target_tensor[:,di,:,:,:]
            discr_frame_out_real = discr_frame(target).view(-1)
            discr_frame_out_fake = discr_frame(output_image.detach()).view(-1)
            discr_frame_loss += 0.5 * (torch.mean((discr_frame_out_real - 1)**2) + torch.mean(discr_frame_out_fake**2))

            if real_input_flag_decoder[di]:
                decoder_input = target # Teacher forcing    
            else:
                decoder_input = output_image

        discr_frame_loss.backward()
        discr_frame_optimizer.step()
        
        return discr_frame_loss.item()/target_length, encoder_frame_ad_loss.item()/target_length, loss.item() / target_length
    
    else:
        return discr_frame_loss/target_length, encoder_frame_ad_loss/target_length, loss.item() / target_length



def trainIters(encoder, discr_frame, nepochs, train_loader, test_loader, testing_input = None, print_every=10, eval_every=10, name='', training_discriminator_every = 1, save_model_every = 1):
    plot_every                  = 1*eval_every
    encoder_total_train_losses  = []
    encoder_ad_train_losses     = []
    discr_train_losses          = []
    

    encoder_optimizer     = torch.optim.Adam(encoder.parameters(),    lr=0.001, betas=(0.5,0.99))
    discr_frame_optimizer = torch.optim.Adam(discr_frame.parameters(),lr=0.0002,betas=(0.5,0.99))
    scheduler_enc         = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=5, factor=0.1, verbose=True)
    scheduler_discr       = ReduceLROnPlateau(discr_frame_optimizer, mode='min', patience=5, factor=0.1, verbose=True)

    criterion_mae         = nn.L1Loss()
    criterion_mse         = nn.MSELoss()

    if testing_input is not None:
        times_curr_test,images_log_test,images_pred_test  = testing_input[0],testing_input[1], testing_input[2]

    for epoch in range(0, nepochs):
        t0 = time.time()
        encoder_total_loss_epoch = 0
        discr_loss_epoch = 0
        encoder_ad_loss_epoch = 0
        
        pbar = tqdm(train_loader)
        for i, out in enumerate(pbar, 0):
            input_tensor = out[1].to(device)
            target_tensor = out[2].to(device)
            discr_frame_loss, encoder_frame_ad_loss, loss = train_on_batch(epoch, input_tensor, target_tensor, encoder, encoder_optimizer, criterion_mae, criterion_mse, discr_frame, discr_frame_optimizer)                                   
            encoder_total_loss_epoch += loss
            discr_loss_epoch += discr_frame_loss
            encoder_ad_loss_epoch += encoder_frame_ad_loss
            pbar.set_description("ep:%d, i:%d: Total Loss: %0.4f, dis loss %0.4f, encd loss %0.4f" % (epoch, i, (encoder_total_loss_epoch)/(i+1), (discr_loss_epoch)/(i+1), (encoder_ad_loss_epoch)/(i+1) ))
        
        encoder_total_train_losses.append(encoder_total_loss_epoch)     
        encoder_ad_train_losses.append(encoder_ad_loss_epoch)
        discr_train_losses.append(discr_loss_epoch)
        
        
        if (epoch+1) % print_every == 0:
            print('training epoch {0}/{1}'.format(epoch+1,nepochs))
            print('encoder total loss:{0:.3f}'.format(encoder_total_loss_epoch))
            print('time epoch:{0:.3f}s'.format(time.time()-t0))
            
        if (epoch+1) % save_model_every == 0:
            print('saving the model...')
            torch.save(encoder.state_dict(),'save/{0}/encoder.pth'.format(name))
            torch.save(discr_frame.state_dict(),'save/{0}/discriminator.pth'.format(name))
            
        if (epoch+1) % training_discriminator_every == 0:  
            print('encoder adversarial loss:{0:.3f}'.format(encoder_ad_loss_epoch))
            print('discriminator loss:{0:.3f}'.format(discr_loss_epoch)) 
            f,ax=plt.subplots()
            ax.plot(range(len(encoder_ad_train_losses)),encoder_ad_train_losses,label="gen_loss")
            ax.plot(range(len(discr_train_losses)),discr_train_losses,label='disc_loss')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.legend()
            f.tight_layout()
            f.savefig("save/%s/figures/loss_%d.png" % (name, epoch)) 
            
        if (epoch+1) % eval_every == 0: 
            mse, mae, predictions,_, target_samples, input_samples = evaluate(encoder, test_loader)
            scheduler_enc.step(mae)
            scheduler_discr.step(mae)
        
        if ((epoch+1) % plot_every == 0):
            select_idx = [500] # 3445
            plot_gen_images(predictions,  input_samples, target_samples, select_idx, epoch, times_curr=times_curr_test, name=name)
        
    return encoder_total_train_losses,encoder_ad_train_losses,discr_train_losses



def evaluate(encoder, loader):
    total_mse, total_mae = 0,0
    t0 = time.time()
    predictions = []
    target_samples = []
    input_samples = []
    indices = []
    num_samples_ = 0
    num_color_channel = 3
    img_side_len = 64

    pbar = tqdm(loader)
    with torch.no_grad():
        for i, out in enumerate(pbar, 0):

            indices.append(out[0])
            input_tensor  = out[1].to(device)
            target_tensor = out[2].to(device)
            input_length = input_tensor.size()[1]
            target_length = target_tensor.size()[1]

            for ei in range(input_length-1):
                encoder_output, encoder_hidden, _,_,_  = encoder(input_tensor[:,ei,:,:,:], (ei==0))

            decoder_input = input_tensor[:,-1,:,:,:] # first decoder input= last image of input sequence
            prediction = []
            
            for di in range(target_length):
                decoder_output, decoder_hidden, output_image, _, _ = encoder(decoder_input, False, False)
                decoder_input = output_image
                prediction.append(output_image.cpu())
            
            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
            prediction =  np.stack(prediction) # (8, batch_size, 3, 64, 64)
            prediction = prediction.swapaxes(0,1)  # (batch_size, 8, 3, 64, 64)
            
            
            mse_batch = np.mean((prediction-target)**2 , axis=1).sum()
            mae_batch = np.mean(np.abs(prediction-target) ,  axis=1).sum() 
            total_mse += mse_batch
            total_mae += mae_batch
            
            predictions.append(prediction)
            target_samples.append(target)
            input_samples.append(input)

            num_samples_ +=1

            pbar.set_description("[%d] averaged MSE:%.2f MAE:%2f" % (i, total_mse/num_samples_, total_mae/num_samples_))
     
    target_samples =  np.concatenate(target_samples,axis=0)  # (NxB, PredLen, 3, 64, 64) 
    input_samples  =  np.concatenate(input_samples,axis=0) # (NxB, PredLen, 3, 64, 64) 
    predictions    =  np.concatenate(predictions,axis=0) # (NxB, PredLen, 3, 64, 64) 
    print("validation...")    
    print('mse per frame:{0:.3f}'.format(total_mse/num_samples_))  
    print('mae per frame:{0:.3f}'.format(total_mae/num_samples_))
    print('mse per pixel:{0:.3f}'.format(total_mse/num_samples_/(img_side_len*img_side_len*num_color_channel)))  
    print('mae per pixel:{0:.3f}'.format(total_mae/num_samples_/(img_side_len*img_side_len*num_color_channel)))
    print('time:{0:.3f}s'.format(time.time()-t0))
    print('-'*40)
    return total_mse/num_samples_,  total_mae/num_samples_, predictions,  indices, target_samples, input_samples