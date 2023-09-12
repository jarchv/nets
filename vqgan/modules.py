import torch
import torch.nn as nn
import torch.nn.functional as F

def GroupNorm(in_channels):
    return torch.nn.GroupNorm(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    if embedding_dim % 2 == 1:      # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0, 0, 0), 'constant', 0)
    return emb 

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)
    
    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = F.pad(x, pad, mode='constant', value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = GroupNorm(in_channels)
        self.swish = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)

        self.norm2 = GroupNorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,padding=1)

        if in_channels != out_channels:
            self.conv_up = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x                                       # h: (B,C,H,W)
        h = self.norm1(h)                           
        h = self.swish(h)
        h = self.conv1(h)                           # h: (B,N,H,W)
        
        
        h = self.norm2(h)                           
        h = self.swish(h)
        h = self.conv2(h)                           # h: (B,N,H,W)

        if self.in_channels != self.out_channels:
            x = self.conv_up(x)            # x: (B,N,H,W)
        return x + h            

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = GroupNorm(in_channels)
        self.q    = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.k    = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.v    = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0)

        self.swish = nn.SiLU(inplace=True)

    def forward(self, x):
        h = x
        h = self.norm(h)
        h = self.swish(h)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        # Compute Attention Map
        B, C, H, W = q.shape
        q = q.reshape(B, C, -1)                     # q: (B,C,H*W)
        q = q.permute(0, 2, 1)                      # q: (B,H*W,C)
        
        k = k.reshape(B, C, -1)                     # k: (B,C,H*W)
        w = torch.bmm(q, k)                         # w: (B,H*W,H*W)
        w = w / (C ** 0.5)
        w = F.softmax(w, dim=-1)
        
        # Compute Context Vector
        v = v.reshape(B, C, -1)
        w = w.permute(0, 2, 1)                      # w: (B,H*W,H*W)
        c = torch.bmm(v, w)                         # c: (B,C,H*W)
        c = c.reshape(B, C, H, W)                   # c: (B,C,H,W)
        c = self.proj(c)                            # c: (B,C,H,W)

        return x + c

class Codebook(nn.Module):
    def __init__(self, 
        num_embeddings, 
        embedding_dim, 
        beta):
        super(Codebook, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self, z):
        # z: (B,C,H,W)
        z.permute(0, 2, 3, 1)                           # z: (B,H,W,C)
        z_flatten = z.reshape(-1, self.embedding_dim)   # z: (B*H*W,C)

        # Compute L2 distance
        dist = torch.sum(z_flatten**2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight**2, dim=1) - \
               2 * torch.matmul(z_flatten, self.embedding.weight.t()) # dist: (B*H*W,N)

        min_embed_ind = torch.argmin(dist, dim=1)        # min_embed_ind: (B*H*W)
        min_embed_ind = min_embed_ind.unsqueeze(1)       # min_embed_ind: (B*H*W,1)
        # Compute Quantize Loss
        min_embed = torch.zeros(min_embed_ind.shape[0], self.num_embeddings, device=z.device)
        min_embed.scatter_(1, min_embed_ind, 1)          # min_embed: (B*H*W,N)

        # Compute Quantize Latent Vectors
        z_q = torch.matmul(min_embed, self.embedding.weight) # z_q: (B*H*W,C)
        z_q = z_q.reshape(z.shape)                           # z_q: (B,C,H,W)

        # Compute Commitment (Embedding)
        loss  = torch.mean((z_q.detach() - z)**2)             # loss: (1)
        loss += self.beta * torch.mean((z_q - z.detach())**2) # loss: (2)
        
        # Preserve Gradient
        z_q = z + (z_q - z).detach()                          # z_q: (B,C,H,W)

        return z_q, min_embed_ind, loss

if __name__ == "__main__":
    # test
    x = torch.randn(7, 32, 256, 256)
    model = ResidualBlock(32, 64, dropout=0.1)
    y = model(x)
    print(y.shape)
