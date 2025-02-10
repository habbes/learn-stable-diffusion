import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        # normalization helps ensure that output of one layer is within a certain range
        # before it reaches the next layer, the output of each layer
        # is normalizerd to N(0, 1) to avoid having each layer output a different
        # distribution, which could cascade up to the final layer and cause the loss
        # to oscillate widely and can slow down training.
        # There are different types of normalization techniques
        # (batch norm, layer norm, instance nor,, group norm, etc.)
        # group normalization is like layer norm but
        # it normalizes the input in groups, instead of all the features.
        # Each group spans a subset of features and the data is normalized
        # in each group independently.
        # We use group norm becauses values in nearby channels are more likely
        # to be related due to the fact that they were generated by convolution.
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels, height, width)
        residue = x
        batch_size, channels, height, width = x.shape

        # we perform self-attention between all the pixels of the image
        # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        x = x.view(batch_size, channels, height * width)

        # (batch_size, channels, height * width) -> (batch_size, height * width, channels)
        # We consider each batch to contain a sequence of height * width pixels where each pixel has an embedding vector of size channels.
        # We want to find the relationships between each pixel and other pixels in the image using attention.
        x = x.transpose(-1, -2)

        # (batch_size, height * width, channels) -> (batch_size, height * width, channels)
        x = self.attention(x)

        # go back to original shape
        # (batch_size, height * width, channels) -> (batch_size, channels, height * width)
        x = x.transponse(-1, -2)

        x = x.view(batch_size, channels, height, width)

        x += residue

        return x

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

    

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # The decoder is roughly the inverse of the encoder. We start with a latent space
            # and progress to reconstruct the image.
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 8, width / 8) => (batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 8, width / 8) => (batch_size, 512, height / 4, width / 4)
            nn.Upsample(scale_factor=2), # scales image up by a factor of 2 by replicating pixels

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 4, width / 4) => (batch_size, 512, height / 2, width / 2)
            nn.Upsample(scale_factor=2),

             nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # reduce number of features
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, height / 2, width / 2) => (batch_size, 256, height, width)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128),

            nn.SiLu(),

            # (batch_size, 128, height, width) => (batch_size, 3, height, width)
            # convert back to image
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, height / 8, width / 8)

        x /= 0.18215 # reverse the scaling done in the encoder

        for module in self:
            x = module(x)
        
        # (batch_size, 3, height, width)
        return x