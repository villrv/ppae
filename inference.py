import sys
import os
import json
import logging
import pickle
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import configargparse

from utils import *
from autodecoder import *
from dataset import *
import pickle
import json

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
    
    # Optional ones
    p.add_argument('--optimization_epochs', type=int, default=400,
                   help='How many iterations to do optimization')
    p.add_argument('--num_freqs', type=int, default=12, required=False,
                   help='Number of frequencies for the positional encoding')
    p.add_argument('--B', type=int, default=32, required=False,
                   help='Batch size')
    p.add_argument('--latent_num', type=int, default=109656,
                   help='Latent num in the pretrained checkpoint. Depends on your training dataset size')
    p.add_argument('--lr', type=float, default=0.01,
                   help='learning rate for test time optimization')

    args = p.parse_args()
    
    class DotDict:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    value = DotDict(value)
                self.__dict__[key] = value
                
    device = "cuda"

    # Load model
    with open('arguments.json', 'r') as file:
        opt = json.load(file)
    opt = DotDict(opt)
    
    encoding = PositionalEncoding(num_freqs=opt.num_freqs,include_input=True)
    
    model = AutoDecoder.load_from_checkpoint(args.checkpoint, opt=opt, encoding=encoding, latent_num=args.latent_num)
    model.to(device);
    
    # Load your data
    with open(args.data, 'rb') as f:
        data_list = pickle.load(f)
    
    # The following line of code optionally prunes your data if the event files are not yet in 8 hour lifetime.
    pruned_list = prune(data_list)
    
    data = RealEventsDataset(pruned_list, t_scale=28800)
    test_loader = DataLoader(data, batch_size=args.B, collate_fn=padding_collate_fn)
    
    print('finished preparing data')
    
    
    outputs = []
    for idx, batch in enumerate(test_loader):
        batch = todevice(batch, device)
        batch = model.optimize_new_latent(batch, optimization_epochs=args.optimization_epochs, lr=args.lr, init='center', neg_likelihood_only=False, verbose=False)
        outputs.append(todevice(batch,'cpu'))
        print(f'finished {(idx+1)*args.B} sources')

    def output_collate_fn(outputs):
        '''
        Outputs is a list of dic
        '''
        output = {}
        for key in outputs[0].keys():
            if key in ['latent']:
                output[key] = torch.cat([o[key] for o in outputs], dim=0)
            else:
                output[key] = [o[key] for o in outputs]
        return output

    collated_outputs = output_collate_fn(outputs)

    collated_original_idx = torch.cat(collated_outputs['original_idx'],dim=0)
    output_d = {'latents': collated_outputs['latent'].detach().cpu().numpy(),
                'original_idx': collated_original_idx.detach().cpu().numpy()}
    output_d.keys()
    with open(args.save_location, 'wb') as f:
        pickle.dump(output_d, f)