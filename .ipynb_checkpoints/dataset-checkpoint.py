from utils import *

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
    
class StepFunctionEventsDataset(BaseEventsDataset):
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
            
            # Generate an event list
            while True:
                l1 = poisson_process_with_time(10, 1)
                if len(l1) > 40:
                    continue
                l2 = poisson_process_with_time(20.0/k, k)
                if len(l1) + len(l2) > 40:
                    continue
                l3 = poisson_process_with_count(10,40-len(l1)-len(l2))
                d['event_list'] = torch.concat((torch.tensor(l1),1+torch.tensor(l2),1+k+torch.tensor(l3))).float()
                break
            self.data[i] = d