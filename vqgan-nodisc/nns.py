import torch
import torch.nn  as nn
import os
import yaml
import argparse
import importlib
import sys

from torchsummary import summary
from .modules import ResidualBlock, AttnBlock, Downsample, GroupNorm, Upsample

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        prev_resolutions = (1,) + tuple(config.channels_mult)
        resolution = config.resolution
        in_channels  = config.in_channels
        out_channels = config.hid_channels

        # Downsampling
        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)]
        for i in range(len(config.channels_mult)):
            in_channels  = config.hid_channels * prev_resolutions[i]
            out_channels = config.hid_channels * config.channels_mult[i]
            for j in range(config.num_res_layers):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in config.attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i != len(config.channels_mult) - 1:
                layers.append(Downsample(in_channels))
                resolution //= 2

        # Middle
        layers.append(ResidualBlock(out_channels, out_channels))
        layers.append(AttnBlock(out_channels))
        layers.append(ResidualBlock(out_channels, out_channels))
        
        # End
        layers.append(GroupNorm(out_channels))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(out_channels, config.embed_dim, 3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B,C,H,W)
        h = self.model(x)        # h: (B,N,H,W)
        return h

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        in_channels  = config.embed_dim
        out_channels = config.channels_mult[-1] * config.hid_channels
        resolution   = config.resolution // 2 ** (len(config.channels_mult)-1)

        # Middle
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            ResidualBlock(out_channels, out_channels),
            AttnBlock(out_channels),
            ResidualBlock(out_channels, out_channels)]
        
        # Upsampling
        for i in reversed(range(len(config.channels_mult))):
            in_channels = out_channels
            out_channels = config.hid_channels * config.channels_mult[i]

            for j in range(config.num_res_layers):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in config.attn_resolutions:
                    layers.append(AttnBlock(in_channels))
            if i != 0:
                layers.append(Upsample(in_channels, True))
                resolution *= 2

        # End
        layers.append(GroupNorm(out_channels))
        layers.append(nn.Conv2d(out_channels, config.in_channels, 3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # x: (B,N,H,W)
        h = self.model(x)        # h: (B,C,H,W)
        return h

if __name__ == '__main__':
    sys.path.append("..")
    utils = importlib.import_module("utils")
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--config', default='vqgan.yaml', help='config file')
    args = parser.parse_args()
    
    with open(os.path.join("../configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict_to_namespace(config)

    x = torch.randn(10, 3, 64, 64).to(config.log.device)

    print("\nEncoder:")
    E = Encoder(config.vqgan.hyp).to(x.device)
    summary(E, (3, 64, 64))
    y = E(x)

    print("\nDecoder:")
    G = Decoder(config.vqgan.hyp).to(x.device)
    summary(G, (64, 16, 16))
    x_hat = G(y)

    print("\nDiscriminator:")
    D = NLayerDiscriminator(config.vqgan.hyp).to(x.device)
    summary(D, (3, 64, 64))
    z = D(x_hat)