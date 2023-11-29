# Code implementation for Conv-TasNet

import torch
import torch.nn as nn

'''
Although we reformulate the enc/dec operations as matrix multiplications,
the term "convolutional autoencoder" is still used in the code.
Because convolutional and transposed convolutional layers can more easily handle
the overlap between segments and thus enable faster training and better convergence.
- From Conv-TasNet paper

For now, I used linear layers instead of convolutional layers.  # First enc-dec commit

Last update: conv1d implemented.    # Overall Tasnet commit
'''
# batch x L -> batch x N x L
class Encoder(nn.Module):
    def __init__(self,
                 N: int,    # Number of filters in autoencoder
                 ):
        super().__init__()
        
        self.encode = nn.Conv1d(in_channels=1,
                                out_channels=N,
                                stride=1,
                                kernel_size=1,
                                )
        self.relu = nn.ReLU()
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        # x: batch x 1 x L
        batch_size = x.size(0)
        x = x.reshape(batch_size, 1, -1)
        
        # x: batch x 1 x L -> batch x N x L
        x = self.encode(x)
        x = self.relu(x)
        
        return x    # batch x N x L

# batch x N x L -> batch x 1 x L
class Decoder(nn.Module):
    def __init__(self,
                 N: int,
                 ):
        super().__init__()
        
        self.decode = nn.Conv1d(in_channels=N,
                                out_channels=1,
                                stride=1,
                                kernel_size=1,
                                )
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        x = self.decode(x)
        
        return x

class Norm(nn.Module):
    def __init__(self,
                 ):
        super().__init__()
        
        self.epsilon = 1e-8
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        batch_size = x.size(0)
        M = x.size(1)
        L = x.size(2)
        org_x = x.clone()
        
        mean = torch.zeros(batch_size, 1, L)
        variance = torch.zeros(batch_size, 1, L)
        
        gamma = nn.Parameter(torch.ones(batch_size, M, 1))
        beta = nn.Parameter(torch.zeros(batch_size, M, 1))
        
        for k in range(L):
            mean[:, :, k] = 1 / (M * (k+1)) * torch.sum(org_x[:, :, :k+1], dim=(1,2)).unsqueeze(dim=1)
            variance[:, :, k] = 1 / (M * (k+1)) * torch.sum((org_x[:, :, :k+1] - mean[:, :, k:k+1]) ** 2, dim=(1,2)).unsqueeze(dim=1)
            
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
        
        # batch x B x L -> batch x B x L
        self.normalize = Norm()
        
        # batch x B x L -> batch x H x L
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
        
        # batch x H x L -> batch x H x L
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
        
        # batch x H x L -> batch x B x L
        self.conv1x1_out = nn.Conv1d(in_channels=H,
                                     out_channels=B,
                                     kernel_size=1,
                                     stride=1,
                                     dilation=2 ** (dilation - 1),
                                     )
        
        # batch x H x L -> batch x Sc x L
        self.conv1x1_skip = nn.Conv1d(in_channels=H,
                                      out_channels=Sc,
                                      kernel_size=1,
                                      stride=1,
                                      dilation=2 ** (dilation - 1),
                                      )
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        # x: batch x B x L -> x: batch x H x L
        res_x = x
        x = self.conv1x1(x)
        x = self.prelue1(x)
        x = self.normalize(x)

        # x: batch x H x L -> x: batch x H x L
        for i in range(x.size(1)):
            x[:, i:i+1] = self.d_conv[i](x[:,i:i+1])
        x = self.prelue2(x)
        x = self.normalize(x)
        
        # x: batch x H x L -> x: batch x B x L
        out = self.conv1x1_out(x) + res_x
        skip = self.conv1x1_skip(x)
        
        return out, skip

class OneDStack(nn.Module):
    def __init__(self,
                 B: int,
                 H: int,
                 P: int,
                 Sc: int,
                 Y: int,
                 ):
        super().__init__()
        
        self.Sc = Sc
        
        self.Y = Y
        
        self.convstack = nn.ModuleList(
            [OneDConvBlock(B=B,
                           H=H,
                           P=P,
                           Sc=Sc,
                           dilation=i+1
                           ) for i in range(Y)]
        )
        
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        sum = torch.zeros(x.size(0), self.Sc, x.size(2))
        
        for k in range(self.Y):
            x, skip = self.convstack[k](x)
            sum += skip
        
        return x, sum

class Separation(nn.Module):
    def __init__(self,
                 L: int,
                 N: int,    # Length of encoder output
                 B: int,    # Number of channels in bottleneck layer
                 H: int,    # Number of channels in convolutional layers
                 P: int,    # Kernel size of convolutional layers
                 Sc: int,   # Number of channels in skip-connection layers
                 Y: int,
                 R: int,
                 C: int,
                 ):
        super().__init__()
        
        self.Sc =Sc
        self.R = R
        self.C = C
        
        self.layer_norm = nn.LayerNorm([N, L])
        
        self.conv1x1_in = nn.Conv1d(in_channels=N,
                                    out_channels=B,
                                    kernel_size=1,
                                    stride=1,
                                    )
        
        self.one_d_stack = OneDStack(B=B,
                                     H=H,
                                     P=P,
                                     Sc=Sc,
                                     Y=Y,
                                     )
        
        self.prelu = nn.PReLU(num_parameters=1,
                              init=0.25,
                              )
        
        self.conv1x1_out = nn.Conv1d(in_channels=Sc,
                                     out_channels=N * C,
                                     kernel_size=1,
                                     stride=1,
                                     )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,
                x: torch.Tensor,
                ) -> torch.Tensor:
        
        # x: batch x N x L
        org_x = x.clone()
        sum = torch.zeros(x.size(0), self.Sc, x.size(2))
        
        # x: batch x N x L -> skip: batch x Sc x L
        x = self.layer_norm(x)
        x = self.conv1x1_in(x)
        for _ in range(self.R):
            x, skip = self.one_d_stack(x)
            sum += skip
        
        # skip: batch x Sc x L -> x: batch x N*C x L
        x = self.prelu(skip)
        x = self.conv1x1_out(x)
        x = self.sigmoid(x)
        
        # x: batch x N*C x L -> x_temp: C x batch x N x L tuple
        x_temp = torch.chunk(x, chunks=self.C, dim=1)
        
        # x: C x batch x N x L
        x = torch.zeros(self.C, x.size(0), x.size(1) // self.C, x.size(2))
        
        # x_temp: C x batch x N x L, org_x: batch x N x L -> x: C x batch x N x L
        for i in range(self.C):
            x[i] = org_x * x_temp[i]
        
        return x    # C x batch x N x L

class Conv_TasNet(nn.Module):
    def __init__(self,
                 L:int,
                 N: int,
                 B: int,
                 H: int,
                 P: int,
                 Sc: int,
                 Y: int,
                 R: int,
                 C: int,
                 ):
        super().__init__()
        
        self.C = C
        
        self.encoder = Encoder(N=N)
        self.decoder = Decoder(N=N)
        self.separation = Separation(L=L,
                                     N=N,
                                     B=B,
                                     H=H,
                                     P=P,
                                     Sc=Sc,
                                     Y=Y,
                                     R=R,
                                     C=C,
                                     )
        
    def forward(self,
                x: torch.Tensor,    # batch x L
                ) -> torch.Tensor:
        
        # batch x L -> batch x N x L
        x = self.encoder(x)
        
        # batch x N x L -> C x batch x N x L
        temp_x = self.separation(x)
        
        # Empty C x batch x 1 x L
        x = torch.zeros(self.C, temp_x.size(1), 1, temp_x.size(3))
        
        # C x batch x N x L -> C x batch x L
        for i in range(self.C):
            x[i] = self.decoder(temp_x[i])
        x = x.squeeze()
        
        return x
