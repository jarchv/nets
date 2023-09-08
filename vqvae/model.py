import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .quantizer import VectorQuantizer

def unnormalized(t):
    return (t + 1) * 0.5

def normalized(t):
    return t * 2 - 1

class ResidualLayer(nn.Module):
    def __init__(self, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(h_dim, res_h_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, bias=False)
        )
    def forward(self, x):
        return x + self.res_block(x)

class ResidualStack(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_layers):
        super(ResidualStack, self).__init__()
        self.res_stack = nn.ModuleList(
            [ResidualLayer(h_dim, res_h_dim) for _ in range(n_layers)])
    def forward(self, x):
        for layer in self.res_stack:
            x = layer(x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, res_h_dim, n_layers):
        super(Encoder, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim // 2, out_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualStack(out_dim, res_h_dim, n_layers),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(out_dim, res_h_dim, n_layers)
        )  
    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, res_h_dim, n_layers):
        super(Decoder, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(out_dim, res_h_dim, n_layers),
            nn.ConvTranspose2d(out_dim, out_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_dim // 2, 3, kernel_size=4, stride=2, padding=1),
        )
    def forward(self, x):
        return self.block(x)

class VQVAE(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, K, D, beta, img_size, device):
        super(VQVAE, self).__init__()
        self.K = K
        self.D = D
        self.img_size = img_size
        self.device   = device
        self.encoder = Encoder(in_dim, h_dim, res_h_dim, n_res_layers)
        self.decoder = Decoder(D, h_dim, res_h_dim, n_res_layers)

        self.pre_vq_conv = nn.Conv2d(h_dim, D, kernel_size=1, stride=1)
        self.quantizer = VectorQuantizer(K, D, beta)
        

    def __uniform_samples(self, n_samples, indices):
        min_embed = torch.zeros(indices.shape[0], self.K).to(self.device)
        min_embed.scatter_(1, indices, 1)
        W = self.quantizer.embedding.weight
        z_q = torch.matmul(min_embed, W)
        z_q = z_q.view(n_samples, self.img_size // 4, self.img_size // 4, self.D)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        x_hat = self.decoder(z_q)
        return x_hat, z_q

    def sample(self, n_samples):
        f    = (self.img_size // 4) * (self.img_size // 4)
        rand = np.random.randint(self.K, size=(n_samples * f,1))
        min_embed_ind = torch.from_numpy(rand).long().to(self.device)
        
        x_hat, z_q = self.__uniform_samples(n_samples, min_embed_ind)
        return x_hat

    def forward(self, x, verbose=False):
        x   = normalized(x)
        z_e = self.encoder(x)
        z_e = self.pre_vq_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.quantizer(z_e)

        x_hat = self.decoder(z_q)

        if verbose:
            print("input shape: ", x.shape)
            print("encoded(z_e) shape: ", z_e.shape)
            print("decoded(x_hat) shape: ", x_hat.shape)
        return embedding_loss, x_hat, perplexity
        
if __name__ == "__main__":
    x = np.random.random_sample((16, 3, 8, 8))
    x = torch.from_numpy(x).float()

    res = ResidualLayer(3, 20)
    res_out = res(x)
    print("ResidualLayer output shape: ", res_out.shape)

    res_stack = ResidualStack(3, 20, 5)
    res_stack_out = res_stack(x)
    print("ResidualStack output shape: ", res_stack_out.shape)

    encoder = Encoder(3, 20, 40, 5)
    out     = encoder(x)
    print("Encoder output shape: ", out.shape)

    decoder = Decoder(20, 3, 40, 5)
    y     = decoder(out)
    print("Decoder output shape: ", y.shape)

