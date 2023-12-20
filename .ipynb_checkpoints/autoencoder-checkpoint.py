from utils import *
    

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the input t
    """

    def __init__(self, num_freqs=6, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs) # (num_freqs, )
        self.include_input = include_input
        self.code_size = 2 * num_freqs + 1 * (self.include_input)
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
        :param t_list (B, n, d_in)
        :return (B, n, d_out)
        """
        
        B, n, _ = t_list.shape
        coded_t_list = t_list[:,:,0:1].repeat(1, 1, self.num_freqs * 2) # (B, n, 2*num_freqs)
        coded_t_list = torch.sin(torch.addcmul(self._phases, coded_t_list, self._freqs))

        if self.include_input:
            coded_t_list = torch.cat((t_list[:,:,0:1], coded_t_list), dim=-1)
        return torch.cat((coded_t_list, t_list[:,:,1:]), dim=-1)
    
# class DiffEncoding(nn.Module):
#     '''
#     This encoding just append the sequential difference of t to t in the new dimension
#     '''
#     def __init__(self, include_input=True):
#         super().__init__()
#         self.include_input = include_input
#         self.code_size = 1 + 1 * (self.include_input)
        
#     def forward(self, t_list):
#         B, n, _ = t_list.shape
#         coded_t_list = torch.zeros(B, n, 1).to(t_list.device)
#         coded_t_list[:,:-1,:] = t_list[:,1:,0:1] - t_list[:,:-1,0:1]
#         coded_t_list[coded_t_list <= 0] = torch.min(t_list)
#         if self.include_input:
#             coded_t_list = torch.cat((t_list[:,:,0:1], coded_t_list), dim=-1)
#         return torch.cat((coded_t_list, t_list[:,:,1:]), dim=-1)

# class NaiveEncoding(nn.Module):
#     '''
#     A naive encoding that just adds a new dimension to t_list
#     '''
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, t_list):
#         return t_list.unsqueeze(-1)
    
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.bias, 0.0)
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
    elif type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.kaiming_normal_(param.data, a=0, mode="fan_in")
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0.0)

# class MLP(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size = 64, activation='relu'):
#         super().__init__()
#         if activation=='none':
#             self.layer = nn.Sequential(nn.Linear(input_size, hidden_size),
#                                        nn.ReLU(),
#                                        nn.Linear(hidden_size, output_size))
#         else:
#             self.layer = nn.Sequential(nn.Linear(input_size, hidden_size),
#                                        nn.ReLU(),
#                                        nn.Linear(hidden_size, output_size),
#                                        nn.ReLU())

#     def forward(self, x):
#         return self.layer(x)
    
    
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    
    Comment: we are using pre-activation (activation before dense layers), mainly to allow flexible output activation when putting these layers together.
    """

    def __init__(self, size_in, size_out=None, size_h=None, activation='ReLU'):
        super().__init__()
        
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)


        # Init
        init_weights(self.fc_0)
        nn.init.constant_(self.fc_1.bias, 0.0)
        nn.init.zeros_(self.fc_1.weight)

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = torch.sin

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
            init_weights(self.shortcut)
            
    def forward(self, x):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        return x_s + dx
    
class ResnetFC(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        n_blocks=3,
        d_hidden=64,
        activation='ReLU'
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_hidden hidden dimension throughout network
        """
        super().__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)
            init_weights(self.lin_in)

        self.lin_out = nn.Linear(d_hidden, d_out)
        init_weights(self.lin_out)

        self.n_blocks = n_blocks
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, activation=activation) for i in range(n_blocks)]
        )

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = torch.sin

    def forward(self, x):
        x = self.lin_in(self.activation(x))
            
        for blkid in range(self.n_blocks):
            x = self.blocks[blkid](x)
            
        out = self.lin_out(self.activation(x))
        return out
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_input, d_model, nhead, num_encoder_layers, d_latent, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Transformer Encoder
        self.input_linear = nn.Linear(d_input, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_linear = nn.Linear(d_model, d_latent)

    def forward(self, src, src_key_padding_mask):
        '''
        src has dimension: (B, n, k)
        mask has dimension: (B, n)
        '''
        src = self.input_linear(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.output_linear(output)  # (B, n, d_latent)

        # Average Pooling
        # Use the mask to exclude padding for averaging
        mask_expanded = src_key_padding_mask.unsqueeze(-1).expand(-1, -1, output.shape[-1])
        masked_output = output.masked_fill(mask_expanded, 0)
        sum_output = torch.sum(masked_output, dim=1)
        num_tokens = torch.sum(~src_key_padding_mask, dim=1, keepdim=True)
        latent = sum_output / num_tokens

        return latent


# class AutoencoderFCFixedLength(pl.LightningModule):
#     '''
#     Input: batches of data containing the following:
#         event_list: each event list has fixed length
#     Output: the log of the rate function at the query point x.
#     '''
#     def __init__(self, input_size, latent_size, num_freqs_pe, output_size, E_bins=13, resolution=128, lam_latent = 0, lam_TV = 0, lam_gradient = 0):
#         super().__init__()
#         self.input_size = input_size
#         self.latent_size = latent_size
#         self.pe_size = 2 * num_freqs_pe + 1
#         self.output_size = output_size
#         self.encoder = ResnetFC(self.input_size, self.latent_size)
#         self.decoder = ResnetFC(self.latent_size+self.pe_size, self.output_size)
#         self.code = PositionalEncoding()
#         self.resolution = resolution
#         self.lam_latent = lam_latent
#         self.lam_TV = lam_TV
#         self.lam_gradient = lam_gradient # Gradient loss is more involved.
    
#     def encode(self, encoder_input):
#         self.latent = self.encoder(encoder_input) # (B, latent_size)
        
#     def decode(self, coded_t_list): # coded_t_list has shape (B, n, pe_size)
#         '''
#         Input: et_list, the positional encoded t_list with shape (B, n, pe_size)
#         Output: log rate function values at those t, with shape (B, n)
#         '''
#         # Broadcast and concat
#         B, n_t, _ = coded_t_list.shape
#         decoder_input = torch.cat((self.latent.unsqueeze(1).expand(B,n_t,-1), coded_t_list), dim=-1)  # (B, n, latent_size+pe_size)
#         return self.decoder(decoder_input)
    
#     def forward(self, x, t_list):
#         # Not sure of the use case yet.
#         pass
    
#     def training_step(self, batch, batch_idx):
#         '''
#         Input: a batch of size B containing the following keys:
#             k: (B,) value of k for step function sources
#             type: (B,) type of sources
#             event_list: (B, n)
#         '''
#         # code t_list
#         event_t_list = batch['event_list']
#         coded_event_t_list = self.code(event_t_list)
#         B, n, _ = coded_event_t_list.shape
        
#         # encode
#         self.encode(coded_event_t_list.reshape(B,-1))
#         T, _ = torch.max(batch['event_list'], dim=1)
        
#         # decode, for both the event list and a mesh (for integration)
#         coded_mesh_t_list = self.code(T.unsqueeze(1) * torch.linspace(0, 1, self.resolution+1).unsqueeze(0).to(coded_event_t_list.device))
#         log_event_rate_list = self.decode(coded_event_t_list).squeeze()
#         log_mesh_rate_list = self.decode(coded_mesh_t_list).squeeze()
        
#         # Compute the loss
#         loss = -loglikelihood(log_event_rate_list, log_mesh_rate_list, T) + self.lam_latent * torch.norm(self.latent, p=2)
#         if self.lam_TV > 0:
#             loss += self.lam_TV * loss_TV(log_event_rate_list)
#         self.log('train_loss', loss)
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
#         return optimizer
    
# class AutoencoderRNN(pl.LightningModule):
#     '''
#     Input: batches of data containing the following:
#         event_list: (B, n), where n might change
#     Output: the log of the rate function at the query point x.
#     '''
#     def __init__(self, latent_size, encoding, hidden_size=128, E_bins=13, resolution=4096, lam_latent = 0, lam_TV = 0, lam_gradient = 0, lr=0.001):
#         super().__init__()
#         self.latent_size = latent_size
#         self.hidden_size = hidden_size
#         self.E_bins = E_bins
#         self.code = encoding
#         self.code_size = encoding.code_size
#         self.encoder_lstm = nn.LSTM(input_size=self.code_size+self.E_bins, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
#         self.encoder_linear = nn.Linear(self.hidden_size, self.latent_size)
#         init_weights(self.encoder_lstm)
#         init_weights(self.encoder_linear)
#         self.decoder = ResnetFC(self.latent_size+self.code_size, self.E_bins, d_hidden=self.hidden_size)

        
#         self.resolution = resolution
#         self.lam_latent = lam_latent
#         self.lam_TV = lam_TV
#         self.lam_gradient = lam_gradient # Gradient loss is more involved.
        
#         self.lr = lr
    
#     def encode(self, encoder_input):
#         lstm_output, (_, _) = self.encoder_lstm(encoder_input)
#         self.latent = self.encoder_linear(lstm_output[:,-1,:]) # (B, latent_size)
        
#     def decode(self, coded_t_list): # coded_t_list has shape (B, n, code_size)
#         '''
#         Input: et_list, the positional encoded t_list with shape (B, n, code_size)
#         Output: log rate function values at those t, with shape (B, n)
#         '''
#         # Broadcast and concat
#         B, n_t, _ = coded_t_list.shape
#         decoder_input = torch.cat((self.latent.unsqueeze(1).expand(B,n_t,-1), coded_t_list), dim=-1)  # (B, n, latent_size+code_size)
#         return self.decoder(decoder_input)
    
#     def training_step(self, batch, batch_idx):
#         '''
#         Input: a batch of size B containing the following keys:
#             k: (B,) value of k for step function sources
#             type: (B,) type of sources
#             event_list: (B, n)
#         '''
#         # code t_list
#         event_t_list = batch['event_list']
#         coded_event_t_list = self.code(event_t_list) # (B, n, code_size+E_bins)
#         B, n, _ = coded_event_t_list.shape
        
#         # encode
#         self.encode(coded_event_t_list)
#         T, _ = torch.max(event_t_list[:,:,0], dim=1)
        
#         # decode, for both the event list and a mesh (for integration)
#         coded_mesh_t_list = self.code((T.unsqueeze(-1) * torch.linspace(0, 1, self.resolution+1).to(coded_event_t_list.device).unsqueeze(0)).unsqueeze(-1))
#         # print(coded_mesh_t_list.shape)
#         # print(coded_event_t_list.shape)
#         log_event_rate_list = self.decode(coded_event_t_list[:,:,:self.code_size]) # (B, n, E_bins)
#         log_mesh_rate_list = self.decode(coded_mesh_t_list) # (B, n, E_bins)
        
#         # Compute the loss
#         loss = -loglikelihood(log_event_rate_list, batch['mask'], event_t_list[:,:,-self.E_bins:], log_mesh_rate_list, T) + self.lam_latent * torch.norm(self.latent, p=2)
#         if self.lam_TV > 0:
#             loss += self.lam_TV * loss_TV(log_event_rate_list)
#         self.log('train_loss', loss, prog_bar=True)
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
#         return optimizer
    
class AutoencoderTransformer(pl.LightningModule):
    '''
    Input: batches of data containing the following:
        event_list: (B, n, k), where n is the event list length that might change, and k is the dimension of each event, containing the positional encoding and the energy one-hot encoding
    Output: the log of the rate function at the query point x.
    '''
    def __init__(self, latent_size, encoding, hidden_size=128, E_bins=13, d_encoder_model=32, nhead=4, num_encoder_layers=1, resolution=4096, lam_latent = 0, lam_TV = 0, lam_gradient = 0):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.E_bins = E_bins
        self.code = encoding
        self.code_size = encoding.code_size
        self.encoder = TransformerEncoder(self.code_size+self.E_bins, d_model=d_encoder_model, nhead=nhead, num_encoder_layers=num_encoder_layers, d_latent=latent_size, dim_feedforward=128, dropout=0.1)
        # Weight initialization?
        self.decoder = ResnetFC(self.latent_size+self.code_size, self.E_bins, d_hidden=self.hidden_size)

        
        self.resolution = resolution
        self.lam_latent = lam_latent
        self.lam_TV = lam_TV
        self.lam_gradient = lam_gradient # Gradient loss is more involved.

        self.losses_in_epoch = []
        self.losses = []
    
    def encode(self, encoder_input, mask):
        self.latent = self.encoder(encoder_input, mask) # (B, latent_size)
        
    def decode(self, coded_t_list): # coded_t_list has shape (B, n, code_size)
        '''
        Input: et_list, the positional encoded t_list with shape (B, n, code_size)
        Output: log rate function values at those t, with shape (B, n)
        '''
        # Broadcast and concat
        B, n_t, _ = coded_t_list.shape
        decoder_input = torch.cat((self.latent.unsqueeze(1).expand(B,n_t,-1), coded_t_list), dim=-1)  # (B, n, latent_size+code_size)
        return self.decoder(decoder_input)
    
    def training_step(self, batch, batch_idx):
        '''
        Input: a batch of size B containing the following keys:
            event_list: (B, n, k)
            mask: (B, n)
        '''
        # code t_list
        event_t_list = batch['event_list']
        coded_event_t_list = self.code(event_t_list) # (B, n, code_size+E_bins)
        B, n, _ = coded_event_t_list.shape
        
        # encode
        self.encode(coded_event_t_list, ~batch['mask'])
        T, _ = torch.max(event_t_list[:,:,0], dim=1)
        
        # decode, for both the event list and a mesh (for integration)
        coded_mesh_t_list = self.code((T.unsqueeze(-1) * torch.linspace(0, 1, self.resolution+1).to(coded_event_t_list.device).unsqueeze(0)).unsqueeze(-1))
        log_event_rate_list = self.decode(coded_event_t_list[:,:,:self.code_size]) # (B, n, E_bins)
        log_mesh_rate_list = self.decode(coded_mesh_t_list) # (B, n, E_bins)
        
        # Compute the loss
        loss = -loglikelihood(log_event_rate_list, batch['mask'], event_t_list[:,:,-self.E_bins:], log_mesh_rate_list, T) + self.lam_latent * torch.norm(self.latent, p=2)
        if self.lam_TV > 0:
            loss += self.lam_TV * loss_TV(log_event_rate_list)
            
        self.log('train_loss', loss, prog_bar=True)
        self.losses_in_epoch.append(loss)
        
        return loss
    
    def on_train_epoch_end(self):
        self.losses.append(torch.stack(self.losses_in_epoch).mean())
        self.losses_in_epoch.clear()
        
    def forward(self, batch):
        self.train()
        with torch.no_grad():
            event_t_list = batch['event_list']
            coded_event_t_list = self.code(event_t_list) # (B, n, code_size+E_bins)
            B, n, _ = coded_event_t_list.shape

            # encode
            self.encode(coded_event_t_list, ~batch['mask'])
            T, _ = torch.max(event_t_list[:,:,0], dim=1)
            log_event_rate_list = self.decode(coded_event_t_list[:,:,:self.code_size])
            
            batch.update({
                'latent': self.latent, 
                'rates': torch.exp(log_event_rate_list), 
                'T': T,
                'num_events': torch.sum(batch['mask'], dim=-1)
            })

            return batch
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5),
            'interval': 'epoch',
        }
        return [optimizer], [scheduler]