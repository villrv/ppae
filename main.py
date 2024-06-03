import sys
import os
import json
import logging
import pickle
from pytorch_lightning.loggers import WandbLogger
import configargparse
root_dir = "/nobackup/users/yankeson/Astronomy" 
sys.path.insert(0, f"{root_dir}/ppae/")

from utils import *
from autoencoder import *
from dataset import *


device = 'cuda'

if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    
    ################# All Parameters ####################
    # Very important ones (values required)
    p.add_argument('--model_name', type=str, required=True,
                   help='Name of the model. Relevant for saving and plotting')
    p.add_argument('--num_epochs', type=int, required=True,
                   help='Number of epochs to train')
    p.add_argument('--lam_TV', type=float, required=True,
                   help='Penalty for total variation')
    p.add_argument('--model_type', type=str, required=True,
                   help='Type of the model. decoder: decoder only. lstm: lstm encoder. transformer: vanilla transformer encoder')
    p.add_argument('--starting_epoch', type=int, required=True,
                   help='Which epoch to start training')
    p.add_argument('--data_type', type=str, required=True,
                   help='Whether to use large or small dataset')
    
    
    # Important ones (but with default values)
    p.add_argument('--discrete', action='store_true',
               help='Whether to use discrete version of the decoder (not a neural field)')
    p.add_argument('--random_shift', action='store_true',
               help='Whether to random shift the first event in the dataset')
    p.add_argument('--filter', action='store_true',
               help='Whether to filter the large dataset for balance')
    p.add_argument('--more', action='store_true',
               help='Whether to filter more the large dataset for testing')
    p.add_argument('--B', type=int, default=32, required=False,
                   help='Batch size')
    p.add_argument('--lr', type=float, default=0.001, required=False,
                   help='Learning rate')
    p.add_argument('--num_freqs', type=int, default=12, required=False,
                   help='Number of frequencies for the positional encoding')
    p.add_argument('--num_latent', type=int, default=64, required=False,
                   help='Dimensionality of latent space')
    p.add_argument('--hidden_size', type=int, default=512, required=False,
                   help='hidden size for the decoder ResNet')
    p.add_argument('--lam_latent', type=float, default=0.0, required=False,
                   help='Penalty for norm of latents')
    p.add_argument('--d_transformer_token', type=int, default=48, required=False,
                   help='Dimensionality of transformer token')
    p.add_argument('--n_transformer_head', type=int, default=4, required=False,
                   help='Number of transformer heads')
    p.add_argument('--n_encoder_layers', type=int, default=1, required=False,
                   help='Number of encoder layers for transformer/LSTM')
    p.add_argument('--d_encoder_forward', type=int, default=512, required=False,
                   help='Number of feedforward neurons in transformer/LSTM')
    
    

    
    # Less important ones
    p.add_argument('--T_threshold', type=int, default=43200, required=False,
                   help='Maximum T of eventfiles we consider. Longer eventfiles are truncated')
    p.add_argument('--num_workers', type=int, default=4, required=False,
                   help='Number of workers')
    p.add_argument('--E_bins', type=int, default=3, required=False,
                   help='Number of energy bins to discretize for')
    p.add_argument('--plotting_nbins', type=int, default=100, required=False,
                   help='Scale to normalize times for plottng')
    p.add_argument('--plotting_E_index', type=int, default=1, required=False,
                   help='Which energy bin to plot')
    

    opt = p.parse_args()
    os.chdir(root_dir)
    
    folder_path = f'{root_dir}/experiments/{opt.model_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Create relevant files
    with open(f'{folder_path}/arguments.json', 'w') as file:
        json.dump(vars(opt), file)
    
    #### Load data
    # Load and deserialize the list from the file
    if opt.data_type == 'large':
        if not opt.filter:
            filename = f'{root_dir}/Chandra_data/large_eventfiles_lifetime28800.pkl'
        else:
            if not opt.more:
                filename = f'{root_dir}/Chandra_data/large_eventfiles_filtered_lifetime28800.pkl'
            else:
                filename = f'{root_dir}/Chandra_data/large_eventfiles_filteredmore_lifetime28800.pkl'
                
    else:
        filename = f'{root_dir}/Chandra_data/small_eventfiles_lifetime43200.pkl'
    
    if opt.random_shift:
        filename = filename[:-4] + '_randomshift.pkl'
    
    with open(filename, 'rb') as file: 
        data_lst = pickle.load(file)

    # Load into dataset and dataloader
    if opt.data_type == 'small':
        t_scale = 43200
    else:
        t_scale = 28800
    data = RealEventsDataset(data_lst,E_bins=opt.E_bins,t_scale=t_scale)
    loader = DataLoader(data, batch_size=opt.B, shuffle=True, num_workers=opt.num_workers, collate_fn=padding_collate_fn)
    
    ##### Set up validation plotting
    if opt.data_type == 'large':
        if not opt.filter:
            plotting_inds = [17219, 34935, 36634, 54247, 88181, 88609,  # flares
                     49551, 51095, 1970, 4424, 42866, 74778, # dips
                    71, 159, 304, 381]               # other random ids
        else:
            if opt.more:
                plotting_inds = [709, # flares
                 64, 333, 363, 397, 412, 591, 592, # dips
                71, 159, 304, 381, 118, 42, 513, 832]               # other random ids
            else:
                # Lightly filtered
                plotting_inds = [297, 1940, 6294, 11123, 11197,  # flares
                     290, 558, 911, 4683, 5587, 4997, # dips
                    71, 159, 304, 381, 2024]               # other random ids
    else:
        plotting_inds = [i for i in range(8)] + [j for j in range(600,608)]
        
    # Define the test set
    batch = [data[i] for i in plotting_inds]
    batch = padding_collate_fn(batch)

    ################## Create, train and save the NN model
    wandb_logger = WandbLogger(project='ppad', name=opt.model_name)

    clip_val = 1
    encoding = PositionalEncoding(num_freqs=opt.num_freqs)
    trainer = pl.Trainer(max_epochs=opt.starting_epoch+opt.num_epochs, 
                 accelerator=device, 
                 devices=1, 
                 plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                 logger=wandb_logger)
                 # callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])
                 # accumulate_grad_batches=5)
                 
                 # gradient_clip_val=clip_val)
                # precision="16-mixed")
                
    
    if opt.starting_epoch == 0:
        modelclass = DiscreteAutoEncoder if opt.discrete else AutoEncoder
        model = modelclass(opt.model_type, opt.num_latent, encoding, latent_num=len(data), hidden_size=opt.hidden_size, E_bins=opt.E_bins, lam_TV=opt.lam_TV,lam_latent=opt.lam_latent, d_encoder_model=opt.d_transformer_token, nhead=opt.n_transformer_head, num_encoder_layers=opt.n_encoder_layers, dim_feedforward=opt.d_encoder_forward, lr=opt.lr, test_batch=batch)
        history = trainer.fit(model, loader)
    else:
        modelclass = DiscreteAutoEncoder if opt.discrete else AutoEncoder
        model = modelclass.load_from_checkpoint(f'{folder_path}/model_{opt.starting_epoch}epochs.ckpt', model_type=opt.model_type, latent_size=opt.num_latent, encoding=encoding, latent_num=len(data), hidden_size=opt.hidden_size, E_bins=opt.E_bins, lam_TV=opt.lam_TV, lam_latent=opt.lam_latent, d_encoder_model=opt.d_transformer_token, nhead=opt.n_transformer_head,num_encoder_layers=opt.n_encoder_layers, dim_feedforward=opt.d_encoder_forward, lr=opt.lr, test_batch=batch)
        history = trainer.fit(model, loader, ckpt_path=f'{folder_path}/model_{opt.starting_epoch}epochs.ckpt')

    
    trainer.save_checkpoint(f'{folder_path}/model_{opt.starting_epoch+opt.num_epochs}epochs.ckpt')
    
    
    
    ################ Inference
#     model.to(device)
#     B_test = 16
#     test_loader = DataLoader(data, batch_size=B_test, collate_fn=padding_collate_fn)
#     outputs = []
#     for idx, batch in enumerate(test_loader):
#         batch = todevice(batch, device)
#         outputs.append(todevice(model(batch),'cpu'))
    
#     # Plot total rates
    
#     if opt.data_type == 'large':
#         Tmax = 28800
#     else:
#         Tmax = 43200
#     plt.figure(figsize=(12,9))
#     for i, total_index in enumerate(plotting_inds):
#         batch_index = total_index // B_test
#         if opt.data_type == 'large':
#             # index = total_index % 2 # Kind of random!
#             index = 0
#         else:
#             index = total_index % B_test
#         batch = outputs[batch_index]
#         mask = batch['mask'][index]
#         times = batch['event_list'][index,mask,0] * t_scale / 3600
#         # rates = batch['rates'][index,mask] * Tmax / t_scale / opt.plotting_nbins
        
#         total_mask = batch['total_mask'][index]
#         total_times = batch['total_list'][index,total_mask,0] * t_scale / 3600
#         total_rates = batch['total_rates'][index,total_mask] * Tmax / t_scale / opt.plotting_nbins
        
#         plt.subplot(4,4,i+1)
#         plt.hist(times, bins = opt.plotting_nbins)
#         plt.plot(total_times, torch.sum(total_rates,dim=-1))
#     plt.suptitle('Fitted vs true total rates',size=20)
#     plt.tight_layout()
#     plt.savefig(f'{folder_path}/total_rates_{opt.starting_epoch+opt.num_epochs}epochs.png')

