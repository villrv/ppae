from utils import *
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

def poisson_process_with_count(lam, n):
    '''This function samples a Poisson process with rate lambda until n arrivals'''
    times = []
    t = 0
    c = 0
    while c < n:
        # Time between events follows exponential distribution
        dt = np.random.exponential(1/lam)
        t += dt
        c += 1
        times.append(t)
    return times

def poisson_process_with_time(lam, T):
    '''This function samples a Poisson process with rate lambda until time T'''
    n = np.random.poisson(lam * T)

    if n > 0:
        return np.sort(np.random.uniform(0, T, n))
    else:
        return []
    
def onehot_code_energy(lst, E_min=500, E_max=7000, E_bins=13):
    '''
    This function takes in a list of eventlists. Each eventlist has shape (n_i, 2) where the second column contains evergy.
    The function one-hot encode the energy into E_bins number of bins and concat that to the original first column (time).
    '''
    bin_edges = np.linspace(E_min, E_max, E_bins+1)
    return np.concatenate((lst[:,0:1], np.eye(E_bins)[np.digitize(lst[:,1], bin_edges)-1]), axis = 1)

class BaseEventsDataset(torch.utils.data.Dataset):
    '''
    This is the abstract class for events dataset
    '''
    def __init__(self, N):
        # Each entry of the list below should be a dictionary containing the event list, the source type label, and potentially the hyperparameters of the source
        self.data = [None] * N
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    
class RealEventsDataset(torch.utils.data.Dataset):
    '''
    The class that handles real dataset. 
    Input should be a list of dictionary, each dictionary is a source, containing 'event_list', and other keys
    '''
    def __init__(self, lst, E_min=500, E_max=7000, E_bins=13, t_scale = 500):
        # Each entry of the list below should be a dictionary containing the event list, the source type label, and potentially the hyperparameters of the source
        self.data = [None] * len(lst)
        for i in range(len(lst)):
            d = lst[i]
            d['event_list'] = onehot_code_energy(d['event_list'], E_min=E_min, E_max=E_max, E_bins=E_bins)
            d['event_list'][:,0] = (d['event_list'][:,0] - np.min(d['event_list'][:,0])) / t_scale
            d['event_list'] = torch.tensor(d['event_list']).float()
            d['event_list_len'] = len(d['event_list'])
            d['idx'] = i
            self.data[i] = d
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
            
class StepFunctionEventsDatasetFixedTime(BaseEventsDataset):
    '''
    This class generates event lists from sources of the following type: Pos(10) for 0 <= t < 1, Pos(20/k) for 1 <= t < 1+k, and Pos(10) afterwards. It generate 40 arrivals and stops. The input k_list provides k values for different sources.
    '''
    def __init__(self, N, k_set):
        super().__init__(N)
        
        k_set = torch.tensor(k_set)
        
        # Generate source types
        type_list = torch.randint(0,len(k_set), (N,))
        k_list = k_set[type_list]
        
        for i in range(N):
            d = {}
            k = k_list[i]
            d['k'] = k
            d['type'] = type_list[i]
            
            l1 = poisson_process_with_time(10, 1)
            l2 = poisson_process_with_time(20.0/k, k)
            l3 = poisson_process_with_time(10,1)
            t_only_lst = torch.concat((torch.tensor(l1),1+torch.tensor(l2),1+k+torch.tensor(l3))).float().unsqueeze(-1)
            d['event_list'] = torch.concat((t_only_lst, torch.ones_like(t_only_lst)), dim=1)
            d['event_list_len'] = len(d['event_list'])
            self.data[i] = d
                                      
def padding_collate_fn(batch):
    batch.sort(key=lambda x: x['event_list_len'], reverse=True)
    
    superbatch = {}
    for key in batch[0].keys():
        if key == 'event_list':
            superbatch[key] = pad_sequence([x[key] for x in batch], batch_first=True, padding_value=0) # (B, n, E_bins+1)
        elif isinstance(batch[0][key], str):
            superbatch[key] = [x[key] for x in batch]
        else:
            superbatch[key] = torch.tensor([x[key] for x in batch])
            
    # Create a mask
    superbatch['mask'] = torch.arange(torch.max(superbatch['event_list_len'])).expand(len(batch), -1) < superbatch['event_list_len'].unsqueeze(1) # (B, n)
    return superbatch


    