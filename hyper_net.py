import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.
    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.width = width
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.blocksize  = blocksize

    def get_weights(self, t):
        ''' Computes hypernetwork weights. See the forward() function for hypernet output
            Inputs
                t - current time
            Outputs
                W - [width,d,1]
                B - [width,1,d]
                U - [width,1,d]
        '''
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)
        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        U = U * torch.sigmoid(G)
        return [W, B, U]
    
    def forward(self, t, z):
        ''' takes current time and state as input and computes hypernet output: U * h(W.T*Z+B) '''
        # time dependent weights
        W, B, U = self.get_weights(t) # [width,d,1], [width,1,1], [width,1,d]
        # copy the state for each hidden unit
        Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1) # [width,N,d]
        # compute function output
        h = torch.tanh(Z@W+B) # [width,N,1]
        return (h@U).mean(0) # [N,d] - mean over hidden units