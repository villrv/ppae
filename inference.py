import sys
import os
import json
import logging
import pickle
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import configargparse

from utils import *
from autoencoder import *
from dataset import *
import pickle
import json


device = 'cuda'

if __name__ == "__main__":
    p = configargparse.ArgumentParser()
    
    ################# All Parameters ####################
    # Very important ones (values required)
    p.add_argument('--checkpoint', type=str, required=True,
                   help='path to model checkpoint')
    p.add_argument('--data', type=str, required=True,
                   help='path to data (in pkl format)')
    p.add_argument('--save_location', type=str, required=True,
                   help='place to save computed latents')
    p.add_argument('--root_dir', type=str, required=True,
                   help='the root directory where ppad lies')
    
    # Optional ones
    p.add_argument('--optimization_epochs', type=int, required=False,
                   help='How many iterations to do optimization')
    p.add_argument('--lr', type=float, required=False,
                   help='learning rate for test time optimization')

    opt = p.parse_args()
    
    
    

    class DotDict:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = DotDict(value)
                self.__dict__[key] = value
    device = "cuda"
    B = 32
    num_freqs = 12
    latent_num = 109656

    
    # Load model
    encoding = PositionalEncoding(num_freqs=num_freqs,include_input=True)
    with open('arguments.json', 'r') as file:
        opt = json.load(file)
    opt = DotDict(opt)
    model = AutoEncoder.load_from_checkpoint(args.checkpoint, opt=opt, encoding=encoding, latent_num=latent_num)
    model.to(device);
    
    # Load your data
    with open(args.data, 'rb') as f:
    data_list = pickle.load(f)
    
    # The following line of code optionally prunes your data if the event files are not yet in 8 hour lifetime.
    pruned_list = prune(data_list)
    
    data = RealEventsDataset(pruned_list,E_bins=3,t_scale=28800)
    test_loader = DataLoader(data, batch_size=B, collate_fn=padding_collate_fn)
    
    
    
    
    outputs = []
    for idx, batch in enumerate(test_loader):
        temp = idx * B
        batch = todevice(batch, device)
        batch = model.optimize_new_latent(batch, optimization_epochs=args.optimization_epochs, lr=args.lr, init='center', neg_likelihood_only=False, verbose=False)
        outputs.append(todevice(batch,'cpu'))

    def output_collate_fn(outputs):
        '''
        Outputs is a list of dic
        '''
        output = {}
        for key in outputs[0].keys():
            if torch.is_tensor(outputs[0][key]):
                output[key] = torch.cat([o[key] for o in outputs], dim=0)
            else:
                output[key] = [o[key] for o in outputs]
        return output

    collated_outputs = output_collate_fn(outputs)

    np.save(args.save_location, collated_outputs['latent'].detach().cpu().numpy())