from utils import *
from io import BytesIO
from PIL import Image
import wandb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    

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
    
class ResnetBlockFC(nn.Module):
    """
    Fully connected ResNet Block class.
    Taken from DVR code.
    :param size_in (int): input dimension
    :param size_out (int): output dimension
    :param size_h (int): hidden dimension
    
    Comment: we are using pre-activation (activation before dense layers), mainly to allow flexible output activation when putting these layers together.
    """

    def __init__(self, size_in, size_h = None, size_out = None):
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

        self.activation = nn.ReLU()

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
        d_latent,
        n_blocks=5,
        d_hidden=64,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_hidden hidden dimension throughout network
        """
        super().__init__()
        if d_in > 0:
            self.lin_in = nn.Linear(d_in+d_latent, d_hidden)
            init_weights(self.lin_in)

        self.lin_out = nn.Linear(d_hidden, d_out)
        init_weights(self.lin_out)

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden) for i in range(n_blocks)]
        )

        self.activation = nn.ReLU()

    def forward(self, zx):
        zx = self.lin_in(zx)
            
        for blkid in range(self.n_blocks):
            zx = self.blocks[blkid](zx)
            
        out = self.lin_out(self.activation(zx))
        return out


    
class AutoDecoder(pl.LightningModule):
    '''
    Input: batches of data containing the following:
        event_list: (B, n, k), where n is the event list length that might change, and k is the dimension of each event, containing the positional encoding and the energy one-hot encoding
    Output: the log of the rate function at the query point x.
    '''
    def __init__(self, opt, encoding, latent_num, test_batch=[]):
        super().__init__()
        self.latent_size = opt.latent_size
        self.latent_only = opt.latent_only if hasattr(opt, 'latent_only') else False
        self.latent_num = latent_num
        self.hidden_size = opt.hidden_size
        self.hidden_blocks = opt.hidden_blocks
        self.code = encoding
        self.E_bins = 3
        self.code_size = encoding.code_size

        # Initialize latent variables
        latent_variables = torch.randn(latent_num, opt.latent_size, requires_grad=True)
        self.latent = nn.Parameter(latent_variables)
        
        # Initialize decoder network
        self.decoder = ResnetFC(self.code_size, self.E_bins, self.latent_size, d_hidden=self.hidden_size, n_blocks=self.hidden_blocks)

        self.lr = opt.lr
        self.resolution = opt.resolution
        self.lam_latent = opt.lam_latent
        self.lam_TV = opt.lam_TV
        self.test_batch = test_batch

        self.loss_map = {'loss_total':[],'neg_loglikelihood':[],'loss_TV':[],'loss_latent':[]}
        
    def decode(self, coded_t_list, indices=None, new_latents=None): # coded_t_list has shape (B, n, code_size)
        '''
        Input: coded_t_list, the positional encoded t_list with shape (B, n, code_size)
        Output: log rate function values at those t, with shape (B, n)
        '''
        # Broadcast and concat
        B, n_t, _ = coded_t_list.shape
        
        if new_latents is not None:
            decoder_input = torch.cat((new_latents.unsqueeze(1).expand(B,n_t,-1), coded_t_list), dim=-1)  # (B, n, latent_size+code_size)
        else:
            decoder_input = torch.cat((self.latent[indices].unsqueeze(1).expand(B,n_t,-1), coded_t_list), dim=-1)  # (B, n, latent_size+code_size)

        return self.decoder(decoder_input)
    
    def training_step(self, batch, batch_idx):
        '''
        Input: a batch of size B containing the following keys:
            event_list: (B, n, k)
            mask: (B, n)
        '''
        # code event_t_list
        event_t_list = batch['event_list']
        coded_event_t_list = self.code(event_t_list) # (B, n, code_size+E_bins)
        B, n, _ = coded_event_t_list.shape
        
        T, _ = torch.max(event_t_list[:,:,0], dim=1)
        
        # decode, for both the event list and a mesh (for integration)
        mesh_t_list = (T.unsqueeze(-1) * torch.linspace(0, 1, self.resolution+1).to(coded_event_t_list.device).unsqueeze(0)).unsqueeze(-1)
        coded_mesh_t_list = self.code(mesh_t_list)
        indices = batch['idx']
        log_event_rate_list = self.decode(coded_event_t_list[:,:,:self.code_size], indices) # (B, n, E_bins)
        log_mesh_rate_list = self.decode(coded_mesh_t_list, indices) # (B, n, E_bins)
        
        # Neg likelihood loss
        neg_loglikelihood = -loglikelihood(log_event_rate_list, batch['mask'], event_t_list[:,:,-self.E_bins:], log_mesh_rate_list, T)
        
        # Total variation loss
        TV1 = total_variation_normalized(log_mesh_rate_list.exp(), T_mask=None)
        TV2 = total_variation_normalized(log_event_rate_list.exp(), T_mask=batch['mask'])
        loss_TV = TV1 + TV2

        # Regularization on latents
        loss_latent = self.latent[indices].square().sum(dim=1).mean()

        loss_total = self.lam_TV * loss_TV + self.lam_latent * loss_latent + neg_loglikelihood
        self.loss_map['loss_total'].append(loss_total)
        self.loss_map['loss_TV'].append(loss_TV)
        self.loss_map['loss_latent'].append(loss_latent)
        self.loss_map['neg_loglikelihood'].append(neg_loglikelihood)
        
        return loss_total
    
    
    def on_train_epoch_end(self):
        for k in self.loss_map.keys():
            self.log(f'loss/{k}', torch.stack(self.loss_map[k]).mean(), on_step=False, on_epoch=True)
            self.loss_map[k] = []
        if not self.latent_only:
            scheduler = self.lr_schedulers()
            self.log('optim/model_lr', scheduler.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True)
            self.log('optim/latent_lr', scheduler.optimizer.param_groups[1]['lr'], on_step=False, on_epoch=True)
        else:
            scheduler = self.lr_schedulers()
            self.log('optim/latent_lr', scheduler.optimizer.param_groups[0]['lr'], on_step=False, on_epoch=True)
        
        # Reconstruction plot
        if (self.current_epoch+1) % 10 == 0:
            t_scale = 28800
            batch = self.forward(todevice(self.test_batch, self.device))

            fig = plt.figure(figsize=(6,9))
            for ii, index in enumerate([0,1,2,3,4,5,14,15]):
                mask = batch['mask'][index].cpu()
                times = batch['event_list'][index,mask,0].cpu() * t_scale / 3600
                rates = batch['rates'][index,mask].cpu() / 100
                total_mask = batch['total_mask'][index].cpu()
                total_times = batch['total_list'][index,total_mask,0].cpu() * t_scale / 3600
                total_rates = batch['total_rates'][index,total_mask].cpu() / 100

                plt.subplot(4,2,ii+1)
                plt.hist(times, bins = 100)
                plt.plot(total_times, torch.sum(total_rates,dim=-1))
                plt.title(f"source id: {int(batch['idx'][index].cpu().numpy())}")
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            image = Image.open(buf)
            self.logger.experiment.log({"recon/recon": wandb.Image(image)})   
        
    def forward(self, batch):
        event_t_list = batch['event_list']
        coded_event_t_list = self.code(event_t_list) # (B, n, code_size+E_bins)
        B, n, _ = coded_event_t_list.shape

        T, _ = torch.max(event_t_list[:,:,0], dim=1)
        mesh_t_list = (T.unsqueeze(-1) * torch.linspace(0, 1, self.resolution+1).to(coded_event_t_list.device).unsqueeze(0)).unsqueeze(-1)
        coded_mesh_t_list = self.code(mesh_t_list)

        total_t_list, total_mask = merge_with_mask(event_t_list[:,:,0:1], batch['mask'], mesh_t_list[:,:,0:1])
        coded_total_t_list = self.code(total_t_list)

        with torch.no_grad():
            log_event_rate_list = self.decode(coded_event_t_list[:,:,:self.code_size], batch['idx'])
            log_total_rate_list = self.decode(coded_total_t_list[:,:,:self.code_size], batch['idx'])
            log_mesh_rate_list = self.decode(coded_mesh_t_list[:,:,:self.code_size], batch['idx'])
            batch['latent'] = self.latent[batch['idx']].detach()
                
        batch.update({
            'rates': torch.exp(log_event_rate_list), 
            'total_rates': torch.exp(log_total_rate_list), 
            'total_list': total_t_list,
            'T': T,
            'num_events': torch.sum(batch['mask'], dim=-1),
            'total_mask': total_mask
            })

        return batch

    def optimize_new_latent(self, batch, optimization_epochs=200, lr=0.01, init='random', neg_likelihood_only=False, verbose=False):
        '''
        Test time optimization
        '''
        self.decoder.requires_grad_(False)
        
        # Prepare encodings of t_lists
        event_t_list = batch['event_list']
        T, _ = torch.max(event_t_list[:,:,0], dim=1)
        coded_event_t_list = self.code(event_t_list)
        B, n, _ = coded_event_t_list.shape
        mesh_t_list = (T.unsqueeze(-1) * torch.linspace(0, 1, self.resolution+1).to(coded_event_t_list.device).unsqueeze(0)).unsqueeze(-1)
        coded_mesh_t_list = self.code(mesh_t_list)
        
        total_t_list, total_mask = merge_with_mask(event_t_list[:,:,0:1], batch['mask'], mesh_t_list[:,:,0:1])
        coded_total_t_list = self.code(total_t_list)
        
        
        if init == 'random':
            new_latents = torch.randn_like(self.latent[0]).repeat(B, 1).requires_grad_(True).to(coded_event_t_list.device)
        elif init == 'center':
            new_latents = torch.mean(self.latent, dim=0).repeat(B, 1).clone().detach().requires_grad_(True).to(coded_event_t_list.device)
            
        # Prepare optimizer
        new_optimizer = torch.optim.Adam([new_latents], lr=lr)

        for epoch in range(optimization_epochs):
            new_optimizer.zero_grad(set_to_none=True)
            log_event_rate_list = self.decode(coded_event_t_list[:,:,:self.code_size], new_latents=new_latents)
            log_mesh_rate_list = self.decode(coded_mesh_t_list, new_latents=new_latents)
            
            neg_loglikelihood = -loglikelihood(log_event_rate_list, batch['mask'], event_t_list[:,:,-self.E_bins:], log_mesh_rate_list, T)

            TV1 = total_variation_normalized(log_mesh_rate_list.exp(), T_mask=None)
            TV2 = total_variation_normalized(log_event_rate_list.exp(), T_mask=batch['mask'])
            loss_TV = TV1 + TV2

            loss_latent = new_latents.square().sum(dim=1).mean()

            if neg_likelihood_only:
                loss = neg_loglikelihood
            else:
                loss = self.lam_TV * loss_TV + self.lam_latent * loss_latent + neg_loglikelihood
            loss.backward()

            if verbose:
                if epoch % 10 == 0:
                    print(f'after {epoch} epochs, loss are {loss.detach().item():.2f}')
            new_optimizer.step()

        with torch.no_grad():
            log_event_rate_list = self.decode(coded_event_t_list[:,:,:self.code_size], new_latents=new_latents)
            log_total_rate_list = self.decode(coded_total_t_list[:,:,:self.code_size], new_latents=new_latents)
                
        batch.update({
            'latent': new_latents,
            'rates': torch.exp(log_event_rate_list), 
            'total_rates': torch.exp(log_total_rate_list), 
            'total_list': total_t_list,
            'T': T,
            'num_events': torch.sum(batch['mask'], dim=-1),
            'total_mask': total_mask
            })
        torch.cuda.empty_cache()
        del log_event_rate_list, log_mesh_rate_list, log_total_rate_list
        del new_optimizer
        
        self.decoder.requires_grad_(True)
        
        return batch

    
    def configure_optimizers(self):
        if self.latent_only:
            optimizer = torch.optim.Adam([self.latent], lr=self.lr * 10)
        else:
            grouped_parameters = [
                {"params": self.decoder.parameters(), 'lr': self.lr},
                {"params": [self.latent], 'lr': self.lr * 10},
            ]

            optimizer = torch.optim.Adam(grouped_parameters)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True),
            'interval': 'epoch',
            'monitor': 'loss/loss_total'
        }
        return [optimizer], [scheduler]