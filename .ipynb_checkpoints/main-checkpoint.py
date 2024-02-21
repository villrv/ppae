import sys
import os
import json
import logging
import pickle
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
    p.add_argument('--random_shift', action='store_true',
               help='Whether to random shift the first event in the dataset')
    p.add_argument('--B', type=int, default=64, required=False,
                   help='Batch size')
    p.add_argument('--lr', type=float, default=0.001, required=False,
                   help='Learning rate')
    p.add_argument('--num_freqs', type=int, default=12, required=False,
                   help='Number of frequencies for the positional encoding')
    p.add_argument('--num_latent', type=int, default=30, required=False,
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
    p.add_argument('--t_scale', type=int, default=5000, required=False,
                   help='The scale we normalize times with. This is to ensure NN inputs are within reasonable range')
    p.add_argument('--num_workers', type=int, default=4, required=False,
                   help='Number of workers')
    p.add_argument('--E_bins', type=int, default=13, required=False,
                   help='Number of energy bins to discretize for')
    p.add_argument('--plotting_t_scale', type=int, default=10, required=False,
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
    logging.basicConfig(filename=f'{folder_path}/logfile.log')
    
    #### Load data
    # Load and deserialize the list from the file
    if opt.data_type == 'large':
        with open(f'{root_dir}/Chandra_data/large_eventfiles_lifetime28800.pkl', 'rb') as file: 
            data_lst = pickle.load(file)
    else:
        with open(f'{root_dir}/Chandra_data/small_eventfiles_lifetime43200.pkl', 'rb') as file: 
            data_lst = pickle.load(file)
#         true_flares_df = pd.read_csv(f'{root_dir}/Chandra_data/trueflares.csv')
#         false_flares_df = pd.read_csv(f'{root_dir}/Chandra_data/falseflares.csv')
#         true_flares_df = true_flares_df[['time','energy','obsreg_id']]
#         false_flares_df = false_flares_df[['time','energy','obsreg_id']]

#         # Convert to data dictionary
#         d = true_flares_df.groupby('obsreg_id').apply(lambda group: np.array(group[['time', 'energy']])).to_dict()
#         d.update(false_flares_df.groupby('obsreg_id').apply(lambda group: np.array(group[['time', 'energy']])).to_dict())

#         # Convert to data list and drop outliers
#         data_lst = []
#         lengths = []
#         Ts = []
#         length_threshold = 5000
#         T_threshold = 250000
#         for key in list(d.keys()):
#             length = len(d[key])
#             T = max(d[key][:,0]) - min(d[key][:,0])
#             if length > length_threshold or T > T_threshold:
#                 continue
#             else:
#                 lengths.append(length)
#                 Ts.append(T)
#                 data_lst.append({'event_list':d[key]})

    # Load into dataset and dataloader
    data = RealEventsDataset(data_lst,E_bins=opt.E_bins,t_scale=opt.t_scale, random_shift=opt.random_shift)
    loader = DataLoader(data, batch_size=opt.B, shuffle=True, num_workers=opt.num_workers, collate_fn=padding_collate_fn)

    ################## Create, train and save the NN model
    # torch.autograd.set_detect_anomaly(True)
    clip_val = 1
    encoding = PositionalEncoding(num_freqs=opt.num_freqs)
    trainer = pl.Trainer(max_epochs=opt.starting_epoch+opt.num_epochs, 
                 accelerator=device, 
                 devices=1, 
                 plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                 log_every_n_steps = 2,
                 gradient_clip_val=clip_val)
                # precision="16-mixed")
    if opt.starting_epoch == 0:
        model = AutoEncoder(opt.model_type, opt.num_latent, encoding, hidden_size=opt.hidden_size, E_bins=opt.E_bins,
                        lam_TV=opt.lam_TV,lam_latent=opt.lam_latent, d_encoder_model=opt.d_transformer_token, nhead=opt.n_transformer_head,
                        num_encoder_layers=opt.n_encoder_layers, dim_feedforward=opt.d_encoder_forward, lr=opt.lr)
        history = trainer.fit(model, loader)
    else:
        model = AutoEncoder.load_from_checkpoint(f'{folder_path}/model_{opt.starting_epoch}epochs.ckpt',
                                          model_type=opt.model_type, latent_size=opt.num_latent, encoding=encoding,
                                          hidden_size=opt.hidden_size, E_bins=opt.E_bins, lam_TV=opt.lam_TV, 
                                           lam_latent=opt.lam_latent, d_encoder_model=opt.d_transformer_token,
                                           nhead=opt.n_transformer_head,num_encoder_layers=opt.n_encoder_layers,
                                           dim_feedforward=opt.d_encoder_forward, lr=opt.lr)
        history = trainer.fit(model, loader, ckpt_path=f'{folder_path}/model_{opt.starting_epoch}epochs.ckpt')

    
    trainer.save_checkpoint(f'{folder_path}/model_{opt.starting_epoch+opt.num_epochs}epochs.ckpt')
    
    # Plot training history
    plt.figure(figsize=(12,9))
    plt.plot(torch.arange(opt.starting_epoch, opt.starting_epoch+opt.num_epochs), [l.cpu().detach() for l in model.losses[-opt.num_epochs:]])
    if opt.data_type == 'small':
        plt.ylim([-1200,-300]);
    plt.savefig(f'{folder_path}/training_history_{opt.starting_epoch+opt.num_epochs}epochs.png')
    
    
    ################ Inference
    if opt.data_type == 'large':
        plotting_inds = [2088, 17743, 36200, 56507, 92786, # flares
             7751, 44535, 51575, 78283, 95302, # dips
            71, 159, 304, 381,                 # other large eventfiles
            4, 11]                             # other small eventfiles
    else:
        plotting_inds = [i for i in range(16,24)] + [j for j in range(500,508)]
    
    model.to(device)
    B_test = 16
    test_loader = DataLoader(data, batch_size=B_test, collate_fn=padding_collate_fn)
    outputs = []
    for idx, batch in enumerate(test_loader):
        batch = todevice(batch, device)
        outputs.append(todevice(model(batch),'cpu'))
        
    # Plot total rates
    plt.figure(figsize=(12,9))
    for i, total_index in enumerate(plotting_inds):
        batch_index = total_index // B_test
        if opt.data_type == 'large':
            index = total_index % 2 # Kind of random!
        else:
            index = total_index % B_test
        batch = outputs[batch_index]
        mask = batch['mask'][index]
        times = batch['event_list'][index,mask,0] * opt.plotting_t_scale
        rates = batch['rates'][index,mask] / opt.plotting_t_scale
        T = batch['T'][index] * opt.plotting_t_scale
        plt.subplot(4,4,i+1)
        plt.hist(times, bins = torch.arange(torch.ceil(T)))
        plt.plot(times, torch.sum(rates,dim=-1))
    plt.suptitle('Fitted vs true total rates',size=20)
    plt.tight_layout()
    plt.savefig(f'{folder_path}/total_rates_{opt.starting_epoch+opt.num_epochs}epochs.png')

