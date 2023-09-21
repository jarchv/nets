import torch
import torch.nn  as nn

from .nns import Encoder, Decoder
from .modules import Codebook
from torchsummary import summary

def normalized(t):
    return t * 2 - 1

class VQGAN(nn.Module):
    def __init__(self, config, device):
        super(VQGAN, self).__init__()

        self.E = Encoder(config).to(device)
        self.G = Decoder(config).to(device)
        self.q = Codebook(config).to(device)
        
        summary(self.E, (3,64,64))
        self.quantize_conv = nn.Conv2d(
            in_channels=config.embed_dim,
            out_channels=config.embed_dim,
            kernel_size=1).to(device)
        self.post_quant_conv = nn.Conv2d(
            in_channels=config.embed_dim,
            out_channels=config.embed_dim,
            kernel_size=1).to(device)

    def encode(self, imgs):
        encoded_imgs = self.E(imgs)
        quantize_conv_imgs = self.quantize_conv(encoded_imgs)
        codebook_map, codebook_ind, q_loss = self.q(quantize_conv_imgs)
        return codebook_map, codebook_ind, q_loss

    def decode(self, imgs):
        post_quant_conv_imgs = self.post_quant_conv(imgs)
        decoded_imgs = self.G(post_quant_conv_imgs)
        return decoded_imgs

    def get_lambda(self, perceptual_loss, gan_loss):
        last_layer_weight = self.G.model[-1].weight
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True)[0]
        lambda_ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()
        return lambda_ * 0.8

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def forward(self, x):
        codebook_map, codebook_ind, q_loss = self.encode(x)
        decoded_imgs = self.decode(codebook_map)
        return decoded_imgs, codebook_ind, q_loss
        
if __name__ == "__main__":
    import importlib
    import argparse
    import os
    import sys
    import yaml

    sys.path.append("..")
    utils = importlib.import_module("utils")
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--config', default='vqgan.yaml', help='config file')
    args = parser.parse_args()
    
    with open(os.path.join("../configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict_to_namespace(config)
    x = torch.randn(10, 3, 64, 64).to(config.log.device)
    model = VQGAN(config.vqgan.hyp, x.device).to(x.device)
    y, ind, q_loss = model(x)
    print(y.shape, ind.shape, q_loss.shape)