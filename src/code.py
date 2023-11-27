# Code implementation for Conv-TasNet

import torch
import torch.nn as nn

'''
Although we reformulate the enc/dec operations as matrix multiplications,
the term "convolutional autoencoder" is still used in the code.
Because convolutional and transposed convolutional layers can more easily handle
the overlap between segments and thus enable faster training and better convergence.
- From Conv-TasNet paper

For now, I used linear layers instead of convolutional layers.
'''
class Encoder(nn.Module):
    def __init__(self,
                 L: int,
                 N: int,
                 ):
        super().__init__()
        
        self.linear = nn.Linear(L, N, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self,
                x: torch.Tensor,
                ):
        x = self.linear(x)
        x = self.relu(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self,
                 N: int,
                 L: int,
                 ):
        super().__init__()
        
        self.linear = nn.Linear(N, L, bias=False)
        
    def forward(self,
                x: torch.Tensor,
                ):
        x = self.linear(x)
        
        return x