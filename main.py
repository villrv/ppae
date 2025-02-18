import sys
import os
import json
import logging
import pickle
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import configargparse
root_dir = "/nobackup/users/yankeson/Astronomy" 
sys.path.insert(0, f"{root_dir}/ppae/")

from utils import *
from autodecoder import *
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
    p.add_argument('--starting_epoch', type=int, required=True,
                   help='Which epoch to start training')
    p.add_argument('--checkpoint_every', type=int, required=True,
                   help='How often to save checkpoints')
    p.add_argument('--data_type', type=str, required=True,
                   help='Whether to use large or small dataset')
    
    
    # Important ones (but with default values)
    p.add_argument('--finetune_checkpoint', type=str, default=None, required=False,
               help='Whether to finetune a model with larger dataset with one from smaller')
    p.add_argument('--finetune_checkpoint2', type=str, default=None, required=False,
               help='Whether to resume full training after latent only training')
    p.add_argument('--latent_only', action='store_true',
               help='Whether to only optimize latents')

    p.add_argument('--B', type=int, default=32, required=False,
                   help='Batch size')
    p.add_argument('--lr', type=float, default=0.001, required=False,
                   help='Learning rate')
    p.add_argument('--num_freqs', type=int, default=12, required=False,
                   help='Number of frequencies for the positional encoding')
    p.add_argument('--latent_size', type=int, default=64, required=False,
                   help='Dimensionality of latent space')
    p.add_argument('--hidden_size', type=int, default=512, required=False,
                   help='hidden size for the decoder ResNet')
    p.add_argument('--hidden_blocks', type=int, default=5, required=False,
                   help='number of hidden blocks for the decoder ResNet')
    p.add_argument('--lam_latent', type=float, default=0.0, required=False,
                   help='Penalty for norm of latents')
    p.add_argument('--clip_val', type=float, default=10.0, required=False,
                   help='Value to clip gradient')
    
    # Less important ones
    p.add_argument('--num_workers', type=int, default=4, required=False,
                   help='Number of workers')
    p.add_argument('--resolution', type=int, default=2048, required=False,
                   help='Resolution for mesh')
    

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
        filename = f'{root_dir}/Chandra_data/large_eventfiles_lifetime28800_randomshift.pkl'
    elif opt.data_type == 'large_filter':
        filename = f'{root_dir}/Chandra_data/large_eventfiles_filtered_lifetime28800_randomshift.pkl'

    with open(filename, 'rb') as file: 
        data_lst = pickle.load(file)

    # Load into dataset and dataloader
    t_scale = 28800
    data = RealEventsDataset(data_lst, t_scale=t_scale)
    loader = DataLoader(data, batch_size=opt.B, shuffle=True, num_workers=opt.num_workers, collate_fn=padding_collate_fn)
    
    ##### Set up validation plotting
    if opt.data_type == 'large':
        plotting_inds = [17219, 34935, 36634, 54247, 88181, 88609,  # flares
                 49551, 51095, 1970, 4424, 42866, 74778, # dips
                71, 159, 304, 381]               # other random ids
    elif opt.data_type == 'large_filter':
        # Lightly filtered
        plotting_inds = [297, 1940, 6294, 11123, 11197,  # flares
             290, 558, 911, 4683, 5587, 4997, # dips
            71, 159, 304, 381, 2024]               # other random ids
        
    batch = [data[i] for i in plotting_inds]
    batch = padding_collate_fn(batch)

    ################## Create, train and save the NN model
    
    # wandb logger
    def find_id_by_name(project_name, run_name):
        api = wandb.Api()
        runs = api.runs(path=f"{api.default_entity}/{project_name}")
        for run in runs:
            if run.name == run_name:
                return run.id
        return None

    # Search for the project by name and get its ID
    run_id = find_id_by_name('ppad', f"{opt.model_name}_lr00001")
    if run_id:
        wandb_logger = WandbLogger(project='ppad', name=f"{opt.model_name}_lr00001", id=run_id, resume='allow')
    else:
        wandb_logger = WandbLogger(project='ppad', name=f"{opt.model_name}_lr00001")

    # prepare models and trainers
    encoding = PositionalEncoding(num_freqs=opt.num_freqs)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=folder_path,
        filename='model_{epoch}',  # Customize the checkpoint filename
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=opt.checkpoint_every  # Save a checkpoint every 10 epochs
        )

    trainer = pl.Trainer(max_epochs=opt.starting_epoch+opt.num_epochs, 
                 accelerator=device, 
                 devices=1, 
                 plugins=[DisabledSLURMEnvironment(auto_requeue=False)],
                 logger=wandb_logger,
                 callbacks=checkpoint_callback,
                 gradient_clip_val=opt.clip_val)
    
    # Start training
    if opt.starting_epoch == 0:
        if opt.finetune_checkpoint is not None:
            model = AutoDecoder(opt, encoding, latent_num=len(data), test_batch=batch)
            assert opt.data_type == 'large'
            filename = f'{root_dir}/Chandra_data/large_eventfiles_filtered_lifetime28800.pkl'
            with open(filename, 'rb') as file: 
                small_data_lst = pickle.load(file)
            model = load_from_less_latents(model, small_data_lst, data_lst, opt.finetune_checkpoint)
            del small_data_lst
        elif opt.finetune_checkpoint2 is not None:
            model = AutoDecoder.load_from_checkpoint(opt.finetune_checkpoint2, opt=opt, encoding=encoding, latent_num=len(data), test_batch=batch)
        history = trainer.fit(model, loader)
    else:
        search_pattern = os.path.join(folder_path, f'model_{opt.starting_epoch}epochs*.ckpt')
        matching_files = glob(search_pattern)
        if len(matching_files) == 0:
            raise FileNotFoundError(f"No checkpoint files starting with 'model_{opt.starting_epoch}epochs' found in {folder_path}")
        checkpoint_file = matching_files[0]
        model = AutoDecoder.load_from_checkpoint(checkpoint_file, opt=opt, encoding=encoding, latent_num=len(data), test_batch=batch)
        history = trainer.fit(model, loader, ckpt_path=checkpoint_file)

    lrstr = f"{opt.lr:.0e}"
    trainer.save_checkpoint(f'{folder_path}/model_{opt.starting_epoch+opt.num_epochs}epochs_lr{lrstr}.ckpt')
