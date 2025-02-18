from utils import *
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
    
def onehot_code_energy(lst, E_grid):
    '''
    This function takes in a list of eventlists. Each eventlist has shape (n_i, 2) where the second column contains evergy.
    The function one-hot encode the energy into E_bins number of bins and concat that to the original first column (time).
    '''
    return np.concatenate((lst[:,0:1], np.eye(len(E_grid)-1)[np.digitize(lst[:,1], E_grid)-1]), axis = 1)

def random_shift(data_lst):
    '''
    Randomly shift each event file in the data_lst so that they don't all start at t=0.
    '''
    for i in range(len(data_lst)):
        d = data_lst[i]
        T = np.max(d['event_list'][:,0])
        shift = np.random.uniform(0, d['event_list'][1,0] - d['event_list'][0,0])
        d['event_list'][:,0] = d['event_list'][:,0] + shift
        temp_ind = np.where(d['event_list'][:,0] <= T)[0]
        d['event_list'] = d['event_list'][temp_ind,:]
        data_lst[i] = d
    return data_lst

def prune(data_lst, T=28800):
    '''
    Prune a data_lst to remove all event files shorter than T lifetime, and divide longer event files into multiple chunks with lifetime T.
    '''
    pruned_lst = []
    for i in range(len(data_lst)):
        d = data_lst[i]
        event_list = d['event_list']
        
        # normalize
        if len(event_list) == 0:
            continue
        else:
            event_list[:,0] = event_list[:,0] - np.min(event_list[:,0])
        
        # Cut
        valid_indices = np.where((event_list[:,1] <= 7000) & (event_list[:,1] >= 500))[0]
        event_list = event_list[valid_indices,:]
        if event_list[-1,0] < T:
            continue
        else:
            valid_indices = np.where(event_list[:,0] <= T)[0]
            event_list = event_list[valid_indices,:]
            if event_list.shape[0] < 2:
                continue

            # Random shift
            shift = np.random.uniform(0, event_list[1,0] - event_list[0,0])
            event_list[:,0] = event_list[:,0] + shift
            temp_ind = np.where(event_list[:,0] <= T)[0]
            event_list = event_list[temp_ind,:]
            d['event_list'] = event_list
            d['original_idx'] = i
            pruned_lst.append(d)
    return pruned_lst


def glvary(lst, T, m_max=14, resolution=10000):
    '''
    Implementation of Gregory Loredo variability index calculation
    Input:
        lst: list of arrival times
        m_max: maximum number of bins to consider
    Output:
        O - Odds ratio
        P - Probability
        f3 - f3
        f5 - f5
        var_index - var_index
    '''
    N = len(lst)
    Om_list = np.zeros(m_max-1)  # ith index means i+2 bins
    ratesm_list = np.zeros((m_max-1, resolution-1))
    mesh = np.linspace(T/resolution, T-T/resolution, resolution-1)
    counts_list = np.zeros((m_max-1, m_max))
    for m in range(2,m_max+1):
        bin_edges = np.linspace(0, T, m + 1)
        counts = np.histogram(lst, bins=m, range=(0, T))[0]
        bin_rates = m * (counts+1) / (N+m)
        temp = np.digitize(mesh, bin_edges, right=True)-1
        ratesm_list[m-2,:] = bin_rates[temp]
        Om_list[m-2] = N*np.log(m) + gammaln(m) - gammaln(N+m) + np.sum(gammaln(counts+1))
        # Om_list[m-2] = -N*np.log(m)  - np.sum(gammaln(counts+1)) + gammaln(N+1)

    Om_list = np.exp(Om_list)
    O = np.mean(Om_list)
    P = O / (1+O)
    O = np.log(O) - np.log(10)
    
    weights = Om_list / np.sum(Om_list)
    rates_list = np.sum(ratesm_list * weights[:, np.newaxis], axis=0)
    rates_mean = np.mean(rates_list)
    rates_sd = np.mean(np.sqrt(np.sum(ratesm_list**2 * weights[:, np.newaxis], axis=0) - rates_list**2))
    f3 = np.sum((rates_list < rates_mean + 3*rates_sd) & (rates_list > rates_mean - 3*rates_sd)) / (resolution-1)
    f5 = np.sum((rates_list < rates_mean + 5*rates_sd) & (rates_list > rates_mean - 5*rates_sd)) / (resolution-1)
    
    if P<=0.5:
        var_index = 0
    elif P < 2/3 and f3 > 0.997 and f5 == 1:
        var_index = 1
    elif P < 0.9 and f3 > 0.997 and f5 == 1:
        var_index = 2
    elif P < 0.6:
        var_index = 3
    elif P < 2/3:
        var_index = 4
    elif P < 0.9:
        var_index = 5
    elif O < 2:
        var_index = 6
    elif O < 4:
        var_index = 7
    elif O < 10:
        var_index = 8
    elif O < 30:
        var_index = 9
    else:
        var_index = 10
        
    return P, var_index

def compute_statistics(data_lst, t_scale):
    '''
    Compute different statistics for a list of data.
    Input:
        - data_lst: a list containing dictionaries. Each dic should have a "event_list" key.
            - event_list: (N, 2)
    Output: data_lst, with additional summary statistics computed
    '''
    for d in data_lst:
        event_list = d['event_list']
        
        # Normalize
        d['event_list'][:,0] = d['event_list'][:,0] - np.min(d['event_list'][:,0])
        event_list = d['event_list']
        
        # Three different lists
        soft_list = event_list[(event_list[:,1]>500) & (event_list[:,1]<=1200),0]
        medium_list = event_list[(event_list[:,1]>1200) & (event_list[:,1]<=2000),0]
        high_list = event_list[(event_list[:,1]>2000) & (event_list[:,1]<=7000),0]
        event_list = event_list[:,0]
        
        # Hardness ratio
        soft_flux = np.array(len(soft_list))
        med_flux = np.array(len(medium_list))
        high_flux = np.array(len(high_list))
        d['hard_hm'] = (high_flux - med_flux) / (high_flux + med_flux)
        d['hard_ms'] = (med_flux - soft_flux) / (med_flux + soft_flux)
        d['hard_hs'] = (high_flux - soft_flux) / (high_flux + soft_flux)

        # Variability Index!
        P, var_index = glvary(event_list, t_scale)
        d['var_prob_b'] = P
        d['var_index_b'] = var_index
        d['var_prob_s'] = glvary(soft_list, t_scale)[0]
        d['var_prob_m'] = glvary(medium_list, t_scale)[0]
        d['var_prob_h'] = glvary(high_list, t_scale)[0]
        
    return data_lst
    
class RealEventsDataset(torch.utils.data.Dataset):
    '''
    The class that handles real dataset. 
    Input should be a list of dictionary, each dictionary is a source, containing 'event_list', and other keys
    '''
    def __init__(self, lst, t_scale = 28800):
        # Each entry of the list below should be a dictionary containing the event list, the source type label, and potentially the hyperparameters of the source
        self.data = [None] * len(lst)
        for i in range(len(lst)):
            d = lst[i]
            E_grid = np.asarray([5,12,20,70]) * 100
            
            # One hot encode energy
            d['event_list'] = onehot_code_energy(d['event_list'], E_grid)
            
            # Scale to [0,1]
            d['event_list'][:,0] = d['event_list'][:,0] / t_scale
            d['event_list'] = torch.tensor(d['event_list']).float()
            d['event_list_len'] = len(d['event_list'])
            d['idx'] = i
            self.data[i] = d
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
                                      
def padding_collate_fn(batch):
    '''
    This function collate data in a batch, while using padding to make each data having the same length.
    '''
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


    