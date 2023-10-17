from utils import *


def loglikelihood_single(event_rate_list, mesh_rate_list, T):
    '''
    log likelihood of a single event list. Needed when we hav event lists of different length
    '''
    integral = 0.5 * (torch.sum(mesh_rate_list[1:]) + torch.sum(mesh_rate_list[:-1])) * T / (len(mesh_rate_list)-1)
    return torch.sum(event_rate_list) - integral

def loglikelihood(event_rate_list, mesh_rate_list, T):
    '''
    log likelihood of a batch of event list with the same length.
    Input:
        event_rate_list: (B, n_event)
        mesh_rate_list: (B, n_mesh)
        T: (B,)
    '''
    B, n_mesh = mesh_rate_list.shape
    integral = 0.5 * (torch.sum(mesh_rate_list[:,1:], dim=1) + torch.sum(mesh_rate_list[:,:-1], dim=1)) * T / (n_mesh-1)   # (B,)
    return torch.mean(torch.sum(event_rate_list, dim=1) - integral)
    

# The following is a positional encoding module that potentially helps
# class PositionalEncoding(nn.Module):
#     """
#     Positional Encoding for the input t.
#     """

#     def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
#         super().__init__()
#         self.num_freqs = num_freqs
#         self.d_in = d_in
#         self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
#         self.d_out = self.num_freqs * 2 * d_in
#         self.include_input = include_input
#         if include_input:
#             self.d_out += d_in
#         # f1 f1 f2 f2 ... to multiply x by
#         self.register_buffer(
#             "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
#         )
#         # 0 pi/2 0 pi/2 ... so that
#         # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
#         _phases = torch.zeros(2 * self.num_freqs)
#         _phases[1::2] = np.pi * 0.5
#         self.register_buffer("_phases", _phases.view(1, -1, 1))

#     def forward(self, x):
#         """
#         Apply positional encoding (new implementation)
#         :param x (batch, self.d_in)
#         :return (batch, self.d_out)
#         """
#         with profiler.record_function("positional_enc"):
#             embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
#             embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
#             embed = embed.view(x.shape[0], -1)
#             if self.include_input:
#                 embed = torch.cat((x, embed), dim=-1)
#             return embed

# Currently just a naive positional encoding
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
        if activation=='relu':
            last_layer = nn.ReLU()
        else:
            last_layer = nn.LeakyReLU(negative_slope=0.1)
        self.layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size),
                                  last_layer)

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
        self.decoder = MLP(self.latent_size+self.pe_size, self.output_size, activation='leakyrelu')
        self.code = NaiveEncoding()
        self.resolution = resolution
        self.lam = lam
    
    def encode(self, encoder_input):
        self.latent = self.encoder(encoder_input) # (B, latent_size)
        
    def decode(self, coded_t_list): # et_list has shape (B, n, pe_size)
        '''
        Input: et_list, the positional encoded t_list with shape (B, n, pe_size)
        Output: rate function values at those t, with shape (B, n)
        '''
        # Broadcast and concat
        B, n_t, _ = coded_t_list.shape
        decoder_input = torch.cat((self.latent.unsqueeze(1).expand(B,n_t,-1), coded_t_list), dim=-1)  # (B, n, hidden_size+pe_size)
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
        event_rate_list = self.decode(coded_event_t_list).squeeze()
        mesh_rate_list = self.decode(coded_mesh_t_list).squeeze()
        loss = loglikelihood(event_rate_list, mesh_rate_list, T) + self.lam * torch.norm(self.latent, p=2)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
