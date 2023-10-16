from utils import *

# Loss function: likelihood of an event list given an input
def loglikelihood(event_rate_list, mesh_rate_list, T):
    integral = 0.5 * (torch.sum(mesh_rate_list[1:]) + torch.sum(mesh_rate_list[:-1])) * T / (len(mesh_rate_list)-1)
    return torch.sum(event_rate_list) - integral

# Need modification
class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the input t.
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)
    

class Autoencoder(nn.Module):
    '''
    Input: Undecided
    Output: the log of the rate function at the query point x.
    '''
    def __init__(self, input_size, hidden_size, pe_size, output_size):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pe_size = pe_size
        self.output_size = output_size
        self.encoder = MLP(self.input_size, self.hidden_size)
        self.decoder = MLP(self.hidden_size+self.pe_size, self.output_size)
        self.code = PositionalEncoding()
    
    def encode(self, encoder_input):
        self.latent = self.encoder(encoder_input)
        
    def decode(self, et_list): # et_list has shape (B, n, pe_size)
        '''
        Input: et_list, the positional encoded t_list with shape (B, n, pe_size)
        Output: rate function values at those t, with shape (B, n)
        '''
        # Broadcast and concat
        n, _ = t_list.shape
        decoder_input = torch.cat((self.latent.reshape(1,1,-1).expand(B,n,-1), t_list), dim=-1)  # (B, n, hidden_size+pe_size)
        return self.decoder(decoder_input)
    
    def forward(self, x, t_list):
        et_list = self.code(t_list)
        self.encode(x)
        return self.decode(et_list)
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = nn.MSELoss()(x_hat, x)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer