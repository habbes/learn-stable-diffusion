import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # Each image in the batch is a height * width * channels tensor
            # 3 channels for 3 color values per pixel.
            # The convolution layer will output 128 channels, these are features we want to learn from the image.
            # (batch_size, channels, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # Combination of volutions and normalizations,
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

             # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

            # Stride=2 means the convolution skips every other pixel, therefore
            # the output image size is halved
            # (batch_size, 128, height, width) -> (batch_size, 128, height/2, width/2)
            # We'll continue with the trend of reducing the size of the image while increasing the number of features.
            # The image becomes smaller but each pixel contains more information.
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # Increase number of features to 256
            # (batch_size, 128, height / 2, width / 2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),

            # (batch_256, 128, height / 2, width / 2) -> (batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256)

            # Shrink the image again
            # (batch_size, 256, height / 2, width / 2) -> (batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # Increase number of features to 512
            # (batch_size, 256, height / 4, width / 4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),

            # (batch_size, 512, height / 4, width / 4) -> (batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),

            # Shrink the image again
            # (batch_size, 512, height / 4, width / 4) -> (batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, height / 8, width / 8) -> (batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),

            # It will run attention on each pixel. If we consider pixels as a sequence of tokens,
            # then this will relate pixels to each other. While convolution does this to some extent,
            # it only relates pixels close to each other in space, attention is more "global". It can
            # find a relationship between pixels that are far apart.

            # (batch_size, 512, height / 8, width / 8) -> (batch_size, 512, height / 8, width / 8)
            VAE_AttentionBlock(512),

            # (batch_size, 512, height / 8, width / 8) -> (batch_size, 512, height / 8, width / 8)
            VAE_ResidualBlock(512, 512),
            
            # 32 groups, 512 channels
            nn.GroupNorm(32, 512),

            # Activation function, sigmoid Linear Unit. It was found to perform better
            # in practice than others.
            nn.SiLU(),

            # Decreases the number of features, retains the same size.
            # (batch_size, 512, height / 8, width / 8) -> (batch_size, 8, height / 8, width / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (batch_size, 8, height / 8, width / 8) -> (batch_size, 8, height / 8, width / 8)
            # Q: what's the value of kernel size 1? What operation does it perform?
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )