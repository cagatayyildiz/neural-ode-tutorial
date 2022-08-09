import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

    
class MNIST_UnFlatten(nn.Module):
    def __init__(self,w):
        super().__init__()
        self.w = w
    def forward(self, input):
        nc = input[0].numel()//(self.w**2)
        return input.view(input.size(0), nc, self.w, self.w)

    
class MNIST_Encoder(nn.Module):
    def __init__(self, q, n_filt=8):
        ''' Inputs
                q      - latent dimensionality
                n_filt - number of filters in the first CNN layer
        '''
        super().__init__()
        h_dim  = n_filt*4**3
        self.q = q
        self.encoder = nn.Sequential(
            nn.Conv2d(1, n_filt, kernel_size=5, stride=2, padding=(2,2)), # 14,14
            nn.BatchNorm2d(n_filt),
            nn.ReLU(),
            nn.Conv2d(n_filt, n_filt*2, kernel_size=5, stride=2, padding=(2,2)), # 7,7
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.Conv2d(n_filt*2, n_filt*4, kernel_size=5, stride=2, padding=(2,2)),
            nn.ReLU(),
            MNIST_Flatten(),
            nn.Linear(h_dim, 2*q)
        )
        
    def __transform_sig(self,sig):
        return torch.log(1 + torch.exp(sig))
    
    def forward(self, X):
        ''' Encodes the initial values for input trajectores
            Input:
                X - [T,N,1,28,28] image sequences
            Returns:
                mu  - [N,q] encoder mean
                sig - [N,q] encoder std
        '''
        q_z0 = self.encoder(X[0]) # N,2q
        q_z0_mu, q_z0_logsig = q_z0[:,:self.q], q_z0[:,:self.q] # N,q & N,q
        return q_z0_mu, self.__transform_sig(q_z0_logsig)

    
class MNIST_Decoder(nn.Module):
    def __init__(self, q, n_filt=8):
        ''' Inputs
                q      - latent dimensionality
                n_filt - number of filters in the first CNN layer
        '''
        super().__init__()
        h_dim = n_filt*4**3
        self.decoder = nn.Sequential(
            nn.Linear(q, h_dim),
            MNIST_UnFlatten(4),
            nn.ConvTranspose2d(h_dim//16, n_filt*8, kernel_size=3, stride=1, padding=(0,0)),
            nn.BatchNorm2d(n_filt*8),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*8, n_filt*4, kernel_size=5, stride=2, padding=(1,1)),
            nn.BatchNorm2d(n_filt*4),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*4, n_filt*2, kernel_size=5, stride=2, padding=(1,1), output_padding=(1,1)),
            nn.BatchNorm2d(n_filt*2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filt*2, 1, kernel_size=5, stride=1, padding=(2,2)),
            nn.Sigmoid(),
        )
    
    def forward(self, zt):
        ''' Decodes a set of latent trajectores
            Input:
                zt - [T,N,q] latent sequences
            Returns:
                X  - [T,N,1,28,28] decoded images
        '''
        [T,N,q] = zt.shape
        zt = zt.reshape([T*N,q])
        Xhat = self.decoder(zt).reshape([T,N,1,28,28]) # T,N,nc,d,d
        return Xhat

        