import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):

    def __init__(self):
        super().__init__(
            # You may wonder, why this specific architecture? In a lot of DL research,
            # it is common to borrow from prior work, models and architecture that worked
            # well in a similar domain. So a lot of this architecture could be explained
            # by prior art. The sequence of layers reduces the image size while
            # increasing the number of features.
            # In a variational autoencoder we not just compressing the image, but we
            # are learning a latent space that represents the parameters of a (multivariate) (gaussian) distribution,
            # the mean and variance of this distribution. (todo: Need to learn more about VAE)

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
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, channels:3, height:512, width:512):
        # noise: (batch_size, out_channels, height / 8, width / 8): same as the the output tensor

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # for each layer with a stride=2, we want to apply a custom assymetric padding
                # i.e. adding a layer of pixes to the right and bottom sides only. If we used the 
                # padding parameter un Conv2d, it would apply the padding to each side, and that's not
                # what we want in this case.
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        
        # (batch_size, 8, height / 8, width / 8) -> two tensors of shape (batch_size, 4, width / 8, height / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # (batch_size, 4, height / 8, width / 8) -> (batch_size, 4, height / 8, width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # (batch_size, 4, height / 8, width / 8)
        variance = log_variance.exp();

        # (batch_size, 4, height / 8, width / 8)
        stdev = variance.sqrt()

        # Now that we have mean and variance of the latent space distribution
        # we want to "sample" from this distribution.
        # If we have a normal distribution with mean=0 and variance=1, Z=N(0, 1) we can sample
        # by transforming to a distribution X = N(mean + variance)
        # by X = mean + stdev * Z
        x = mean + stdev * noise # we accept the noise as a parameter to allow the caller to initialize with a specific seed

        # Scale the output by a constant
        # (not sure why this is here, the instructor didn't know either but found it in the original repo, should look it up later)
        x *= 0.18215

        return x