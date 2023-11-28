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
                ) -> torch.Tensor:
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
                ) -> torch.Tensor:
        
        x = self.linear(x)
        
        return x

class Norm(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        
        self.epsilon = 0
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        batch_size = x.size(0)
        N = x.size(1)
        L = x.size(2)
        org_x = x.clone()
        
        mean = torch.zeros(batch_size, L)
        variance = torch.zeros(batch_size, L)
        
        gamma = nn.Parameter(torch.ones(batch_size, N))
        beta = nn.Parameter(torch.zeros(batch_size, N))
        
        for k in range(L):
            mean[:, k] = 1 / (N * (k+1)) * torch.sum(org_x[:, :, :k+1])
            variance[:, k] = 1 / (N * (k+1)) * torch.sum((org_x[:, :, :k+1] - mean[:, k]) ** 2)
            
            x[:, :, k] = gamma * (x[:, :, k] - mean[:, k]) / torch.sqrt(variance[:, k] + self.epsilon) + beta
        
        return x

class OneDConvBlock(nn.Module):
    def __init__(self,
                 B: int,
                 H: int,
                 P: int,
                 Sc: int,
                 dilation: int,
                 ):
        super().__init__()
        
        self.normalize = Norm()
        
        self.conv1x1 = nn.Conv1d(in_channels=B,
                                 out_channels=H,
                                 kernel_size=1,
                                 stride=1,
                                 dilation=2 ** (dilation - 1),
                                 )
        
        self.prelue1 = nn.PReLU(num_parameters=1,
                                init=0.25,
                                )
        
        self.prelue2 = nn.PReLU(num_parameters=1,
                                init=0.25,
                                )
        
        self.d_conv = nn.ModuleList(
            [nn.Conv1d(in_channels=H,
                       out_channels=H,
                       kernel_size=P,
                       stride=1,
                       dilation=2 ** (dilation - 1),
                       padding='same',
                       padding_mode='zeros'
                       ) for _ in range(H)]
        )
        
        self.conv1x1_out = nn.Conv1d(in_channels=H,
                                     out_channels=B,
                                     kernel_size=1,
                                     stride=1,
                                     dilation=2 ** (dilation - 1),
                                     )
        
        self.conv1x1_skip = nn.Conv1d(in_channels=H,
                                      out_channels=Sc,
                                      kernel_size=1,
                                      stride=1,
                                      dilation=2 ** (dilation - 1),
                                      )
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        res_x = x
        x = self.conv1x1(x)
        x = self.prelue1(x)
        x = self.normalize(x)
        
        for i in range(x.size(1)):
            x[:,i] = self.d_conv[i](x[:,i])
        x = self.prelue2(x)
        x = self.normalize(x)
        
        out = self.conv1x1_out(x) + res_x
        skip = self.conv1x1_skip(x)
        
        return out, skip