import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions import MultivariateNormal, Normal, Bernoulli, kl_divergence as kl
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


def get_activation(act="relu"):
    if act=="relu":         return nn.ReLU()
    elif act=="elu":        return nn.ELU()
    elif act=="celu":       return nn.CELU()
    elif act=="leaky_relu": return nn.LeakyReLU()
    elif act=="sigmoid":    return nn.Sigmoid()
    elif act=="tanh":       return nn.Tanh()
    elif act=="sin":        return torch.sin
    elif act=="linear":     return nn.Identity()
    elif act=='softplus':   return nn.modules.activation.Softplus()
    elif act=='swish':      return lambda x: x*torch.sigmoid(x)
    else:                   return None


class BNN(nn.Module):
    def __init__(self, n_in: int, n_out: int, n_hid_layers: int=2, n_hidden: int=100, act: str='relu', \
                        requires_grad=True, logsig0=-3):
        super().__init__()
        layers_dim = [n_in] + n_hid_layers*[n_hidden] + [n_out]
        self.weight_mus = nn.ParameterList([])
        self.bias_mus   = nn.ParameterList([])
        self.weight_logsigs = nn.ParameterList([])
        self.bias_logsigs   = nn.ParameterList([])
        self.acts = []
        self.act  = act 
        self.logsig0 = logsig0
        for i,(n_in,n_out) in enumerate(zip(layers_dim[:-1],layers_dim[1:])):
            self.weight_mus.append(Parameter(torch.Tensor(n_in, n_out),requires_grad=requires_grad))
            self.bias_mus.append(Parameter(torch.Tensor(1,n_out),requires_grad=requires_grad))
            self.weight_logsigs.append(Parameter(torch.Tensor(n_in, n_out),requires_grad=requires_grad))
            self.bias_logsigs.append(Parameter(torch.Tensor(1,n_out),requires_grad=requires_grad))
            self.acts.append(get_activation(act) if i<n_hid_layers else get_activation('linear')) # no act. in final layer
        self.reset_parameters()

    @property
    def device(self):
        return self.weight_mus[0].device

    def __transform_sig(self,sig):
        return torch.log(1 + torch.exp(sig))

    def reset_parameters(self,gain=1.0):
        for i,(weight,bias) in enumerate(zip(self.weight_mus,self.bias_mus)):
            nn.init.xavier_uniform_(weight,gain)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
        for w,b in zip(self.weight_logsigs,self.bias_logsigs):
            nn.init.uniform_(w,self.logsig0-1,self.logsig0+1)
            nn.init.uniform_(b,self.logsig0-1,self.logsig0+1)

    def __sample_weights(self, L):
        weights = [w_mu+self.__transform_sig(w_sig)*torch.randn(L,*w_mu.shape,device=self.device) \
                   for w_mu,w_sig in zip(self.weight_mus,self.weight_logsigs)]
        biases = [b_mu+self.__transform_sig(b_sig)*torch.randn(L,*b_mu.shape,device=self.device) \
                   for b_mu,b_sig in zip(self.bias_mus,self.bias_logsigs)]
        return weights,biases

    def draw_f(self, L=1):
        """ 
            x=[N,n] & L=1 ---> out=[N,n]
            x=[N,n] & L>1 ---> out=[L,N,n]
        """
        weights,biases = self.__sample_weights(L)
        def f(x):
            x2d = x.ndim==2
            if x2d:
                x = torch.stack([x]*L) # [L,N,n]
            for (weight,bias,act) in zip(weights,biases,self.acts):
                x = act(torch.baddbmm(bias, x, weight))
            return x.squeeze(0) if x2d and L==1 else x
        return f
    
    def forward(self, x, L=1):
        '''Draws L samples from the BNN output'''  
        return self.draw_f(L)(x)

    def kl(self):
        mus =  [weight_mu.view([-1]) for weight_mu in self.weight_mus]
        mus += [bias_mu.view([-1]) for bias_mu in self.bias_mus]
        logsigs =  [weight_logsig.view([-1]) for weight_logsig in self.weight_logsigs]
        logsigs += [bias_logsigs.view([-1]) for bias_logsigs in self.bias_logsigs]
        mus  = torch.cat(mus)
        sigs = self.__transform_sig(torch.cat(logsigs))
        q = Normal(mus,sigs)
        N = Normal(torch.zeros_like(mus),torch.ones_like(mus))
        return kl(q,N).sum()
    
    def __repr__(self):
        str_ = 'BNN\n'
        for i,(weight,act) in enumerate(zip(self.weight_mus,self.acts)):
            str_ += 'Layer-{:d}: '.format(i+1) + ''.join(str([*weight.shape][::-1])) \
                + '\t' + str(act) + '\n'
        return str_


