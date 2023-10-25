from utils import *


def loglikelihood_single(log_event_rate_list, log_mesh_rate_list, T):
    '''
    Likelihood of an event list (t1,...,tn) with Poisson rate function r(t) is:
        r(t1) * ... * r(tn) * exp(-integral(r(t)))
    We take the log likelihood for better computational performance
    log likelihood of a single event list. Needed when we hav event lists of different length
    '''
    integral = 0.5 * (torch.sum(torch.exp(log_mesh_rate_list[1:])) + torch.sum(torch.exp(log_mesh_rate_list[:-1]))) * T / (len(log_mesh_rate_list)-1)
    return torch.sum(log_event_rate_list) - integral

def loglikelihood(log_event_rate_list, log_mesh_rate_list, T):
    '''
    log likelihood of a batch of event list with the same length.
    Input:
        event_rate_list: (B, n_event)
        mesh_rate_list: (B, n_mesh)
        T: (B,)
    '''
    B, n_mesh = log_mesh_rate_list.shape
    integral = 0.5 * (torch.sum(torch.exp(log_mesh_rate_list[:,1:]), dim=1) + torch.sum(torch.exp(log_mesh_rate_list[:,:-1]), dim=1)) * T / (n_mesh-1)   # (B,)
    return torch.mean(torch.sum(log_event_rate_list, dim=1) - integral)
    

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the input t.
    """

    def __init__(self, num_freqs=6, freq_factor=np.pi, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs) # (num_freqs, )
        self.include_input = include_input
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, 1, -1) # (1, 1, 2*num_freqs)
        )
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, 1, -1))

    def forward(self, t_list):
        """
        Apply positional encoding
        :param t_list (B, n)
        :return (B, n, d_out)
        """
        B, n = t_list.shape
        t_list = t_list.unsqueeze(-1) # (B, n, 1)
        coded_t_list = t_list.repeat(1, 1, self.num_freqs * 2) # (B, n, 2*num_freqs)
        coded_t_list = torch.sin(torch.addcmul(self._phases, coded_t_list, self._freqs))
        if self.include_input:
            coded_t_list = torch.cat((t_list, coded_t_list), dim=-1)
        return coded_t_list

class NaiveEncoding(nn.Module):
    '''
    A naive encoding that just adds a new dimension to t_list
    '''
    def __init__(self):
        super(NaiveEncoding, self).__init__()
        
    def forward(self, t_list):
        return t_list.unsqueeze(-1)

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 64, activation='relu'):
        super(MLP, self).__init__()
        if activation=='none':
            self.layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, output_size))
        else:
            self.layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, output_size),
                                       nn.ReLU())

    def forward(self, x):
        return self.layer(x)
    

class Autoencoder(pl.LightningModule):
    '''
    Input: Undecided
    Output: the log of the rate function at the query point x.
    '''
    def __init__(self, input_size, latent_size, pe_size, output_size, resolution=128, lam = 1):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.pe_size = pe_size
        self.output_size = output_size
        self.encoder = MLP(self.input_size, self.latent_size)
        self.decoder = MLP(self.latent_size+self.pe_size, self.output_size, activation='relu')
        self.code = PositionalEncoding()
        self.resolution = resolution
        self.lam = lam
    
    def encode(self, encoder_input):
        self.latent = self.encoder(encoder_input) # (B, latent_size)
        
    def decode(self, coded_t_list): # coded_t_list has shape (B, n, pe_size)
        '''
        Input: et_list, the positional encoded t_list with shape (B, n, pe_size)
        Output: log rate function values at those t, with shape (B, n)
        '''
        # Broadcast and concat
        B, n_t, _ = coded_t_list.shape
        decoder_input = torch.cat((self.latent.unsqueeze(1).expand(B,n_t,-1), coded_t_list), dim=-1)  # (B, n, latent_size+pe_size)
        return self.decoder(decoder_input)
    
    def forward(self, x, t_list):
        # Not sure of the use case yet.
        pass
    
    def training_step(self, batch, batch_idx):
        '''
        Input: a batch of size B containing the following keys:
            k: (B,) value of k for step function sources
            type: (B,) type of sources
            event_list: (B, n)
        '''
        self.encode(batch['event_list'])
        T, _ = torch.max(batch['event_list'], dim=1)
        coded_event_t_list = self.code(batch['event_list'])
        coded_mesh_t_list = self.code(T.unsqueeze(1) * torch.linspace(0, 1, self.resolution+1).unsqueeze(0))
        log_event_rate_list = self.decode(coded_event_t_list).squeeze()
        log_mesh_rate_list = self.decode(coded_mesh_t_list).squeeze()
        loss = -loglikelihood(log_event_rate_list, log_mesh_rate_list, T) + self.lam * torch.norm(self.latent, p=2)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
