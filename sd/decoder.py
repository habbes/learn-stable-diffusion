import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # The residual block is made up of a convolutions and group normalizations
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # The last layer will be a residual layer that adds the input to the output.
        # But the in_channels might differ from out_channels. We create a helper
        # residual layer to convert the input to the output shape if necessary.
        if in_channels == out_channels:
            # if the channels match, the residual layer won't perform any transformation
            self.residual_layer = nn.Identity()
        else:
            # if the channels don't match, the residual layer will transform the input to match the output number of channels
            # using a convolution
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width)
        
        residue = x
        
        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        # In the residual layer we want to add the input to the output
        # But since the input and output can have different shapes (difference in number of channels),
        # we use the residual layer to transform the input to the output shape.
        return x + self.residual_layer(residue)

    

class VAE_AttentionBlock:
    pass
