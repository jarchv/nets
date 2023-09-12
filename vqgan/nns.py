import torch
import torch.nn  as nn
from .modules import ResidualBlock, AttnBlock, Downsample, GroupNorm, Upsample

class Encoder(nn.Module):
    def __init__(
        self,
        img_channels,
        hid_channels,
        f_channels,
        num_res_layers,
        attn_resolutions,
        resolution,
        latent_dim):
        super(Encoder, self).__init__()

        prev_resolutions = (1,) + f_channels
        # Downsampling
        layers = [nn.Conv2d(img_channels, hid_channels, 3, stride=1, padding=1)]
        for i in range(len(f_channels)):
            in_channels  = hid_channels * prev_resolutions[i]
            out_channels = hid_channels * f_channels[i]

            for j in range(num_res_layers):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i != len(f_channels) - 1:
                layers.append(Downsample(in_channels))
                resolution //= 2

        # Middle
        layers.append(ResidualBlock(out_channels, out_channels))
        layers.append(AttnBlock(out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))
        
        # End
        layers.append(GroupNorm(out_channels))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(out_channels, latent_dim, 3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        # x: (B,C,H,W)
        h = self.model(x)        # h: (B,N,H,W)
        return h

class Decoder(nn.Module):
    def __init__(
        self,
        img_channels,
        hid_channels,
        f_channels,
        num_res_layers,
        attn_resolutions,
        resolution,
        latent_dim):
        super(Decoder, self).__init__()

        in_channels  = latent_dim
        out_channels = f_channels[-1] * hid_channels
        resolution   = resolution // 2 ** (len(f_channels)-1)

        # Middle
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            ResidualBlock(out_channels, out_channels),
            AttnBlock(out_channels),
            ResidualBlock(out_channels, out_channels)]
        
        # Upsampling
        for i in reversed(range(len(f_channels))):
            in_channels = out_channels
            out_channels = hid_channels * f_channels[i]

            for j in range(num_res_layers):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i != 0:
                layers.append(Upsample(in_channels, True))
                resolution *= 2

        # End
        layers.append(GroupNorm(out_channels))
        layers.append(nn.Conv2d(out_channels, img_channels, 3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (B,N,H,W)
        h = self.model(x)        # h: (B,C,H,W)
        return h

class Discriminator(nn.Module):
    def __init__(
        self,
        img_channels,
        hid_channels=32,
        num_layers=3):
        super(Discriminator, self).__init__()
        
        out_channels = hid_channels
        layers = [nn.Conv2d(img_channels, out_channels, 3, stride=1, padding=1)]

        for i in range(1, num_layers):
            in_channels  = out_channels
            out_channels = hid_channels * min(2**i, 8)
            layers += [
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size = 4, 
                    stride= 2 if i < num_layers else 1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers.append(nn.Conv2d(out_channels, 1, 3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B,C,H,W)
        h = self.model(x)        # h: (B,1,H,W)
        return h

if __name__ == '__main__':
    x = torch.randn(10, 3, 64, 64)
    E = Encoder(3)
    y = E(x)
    print(y.shape)

    G = Decoder(3)
    x_hat = G(y)
    print(x_hat.shape)

    D = Discriminator(3)
    z = D(x_hat)
    print(z.shape)