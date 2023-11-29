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
        
        mean = torch.zeros(batch_size, 1, L)
        variance = torch.zeros(batch_size, 1, L)
        
        gamma = nn.Parameter(torch.ones(batch_size, N, 1))
        beta = nn.Parameter(torch.zeros(batch_size, N, 1))
        
        for k in range(L):
            mean[:, :, k] = 1 / (N * (k+1)) * torch.sum(org_x[:, :, :k+1], dim=(1,2)).unsqueeze(dim=1)
            variance[:, :, k] = 1 / (N * (k+1)) * torch.sum((org_x[:, :, :k+1] - mean[:, :, k:k+1]) ** 2, dim=(1,2)).unsqueeze(dim=1)
            
            x[:, :, k:k+1] = gamma * (org_x[:, :, k:k+1] - mean[:, :, k:k+1]) / torch.sqrt(variance[:, :, k:k+1] + self.epsilon) + beta
            
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
            [nn.Conv1d(in_channels=1,
                       out_channels=1,
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
            x[:, i:i+1] = self.d_conv[i](x[:,i:i+1])
        x = self.prelue2(x)
        x = self.normalize(x)
        
        out = self.conv1x1_out(x) + res_x
        skip = self.conv1x1_skip(x)
        
        return out, skip

class OneDStack(nn.Module):
    def __init__(self,
                 B: int,
                 H: int,
                 P: int,
                 Sc: int,
                 dilate_X : int,
                 ):
        super().__init__()
        
        self.Sc = Sc
        
        self.dilate_X = dilate_X
        
        self.convstack = nn.ModuleList(
            [OneDConvBlock(B=B,
                           H=H,
                           P=P,
                           Sc=Sc,
                           dilation=i+1
                           ) for i in range(dilate_X)]
        )
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        sum = torch.zeros(x.size(0), self.Sc, x.size(2))
        
        for k in range(self.dilate_X):
            x, skip = self.convstack[k](x)
            sum += skip
        
        return x, sum

class Separation(nn.Module):
    def __init__(self,
                 B: int,
                 H: int,
                 P: int,
                 Sc: int,
                 L: int,
                 dilate_X: int,
                 ):
        super().__init__()
        
        self.dilate_X = dilate_X
        self.Sc =Sc
        
        self.layer_norm = nn.LayerNorm([B, L])
        
        self.conv1x1_in = nn.Conv1d(in_channels=B,
                                    out_channels=B,
                                    kernel_size=1,
                                    stride=1,
                                    )
        
        self.one_d_stack = OneDStack(B=B,
                                     H=H,
                                     P=P,
                                     Sc=Sc,
                                     dilate_X=dilate_X,
                                     )
        
        self.prelu = nn.PReLU(num_parameters=1,
                              init=0.25,
                              )
        
        self.conv1x1_out = nn.Conv1d(in_channels=Sc,
                                     out_channels=B,
                                     kernel_size=1,
                                     stride=1,
                                     )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        org_x = x.clone()
        sum = torch.zeros(x.size(0), self.Sc, x.size(2))
        
        x = self.layer_norm(x)
        x = self.conv1x1_in(x)
        for _ in range(self.dilate_X):
            x, skip = self.one_d_stack(x)
            sum += skip
        x = self.prelu(skip)
        x = self.conv1x1_out(x)
        x = self.sigmoid(x)
        
        x = org_x * x
        
        return x
