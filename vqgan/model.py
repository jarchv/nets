import torch
import torch.nn  as nn

from .nns import Encoder, Decoder, Discriminator
from .modules import Codebook

def normalized(t):
    return t * 2 - 1

class VQGAN(nn.Module):
    def __init__(
        self,
        img_channels,
        hid_channels,
        f_channels,
        num_res_layers,
        attn_resolutions,
        resolution,
        num_embeddings,
        embedding_dim,
        beta,
        device):
        super(VQGAN, self).__init__()

        self.E = Encoder(
            img_channels,
            hid_channels,
            (1,2,4),
            num_res_layers,
            attn_resolutions,
            resolution,
            embedding_dim).to(device)
        self.G = Decoder(
            img_channels,
            hid_channels,
            (1,2,4),
            num_res_layers,
            attn_resolutions,
            resolution,
            embedding_dim).to(device)
        self.D = Discriminator(
            img_channels).to(device)

        self.q = Codebook(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            beta=beta).to(device)
        
        self.quantize_conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1).to(device)
        self.post_quant_conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
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

    def get_x_hat(self, x, verbose=False):
        x     = normalized(x)
        x_hat, _, _ = self(x)
        return x_hat

    def forward(self, x):
        codebook_map, codebook_ind, q_loss = self.encode(x)
        decoded_imgs = self.decode(codebook_map)
        return decoded_imgs, codebook_ind, q_loss
        