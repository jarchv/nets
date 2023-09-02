import torch
from torch import nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer,self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        # Se define la capa de embedding con los parametros de entrada
        # y se inicializa con valores aleatorios entre -1/K y 1/K
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1/self.K, 1/self.K)

    def forward(self, latents: Tensor) -> Tensor:
        # latents: (batch_size, D, H, W)
        latents = latents.permute(0, 2, 3, 1).contiguous()
        # latents: (batch_size, H, W, D)
        latents_shape = latents.shape
        latents = latents.view(-1, self.D)
        # latents: (batch_size * H * W, D)

        dist = torch.sum(latents**2, dim=1, keepdim=True) \
               - 2 * torch.matmul(latents, self.embedding.weight.t()) \
               + torch.sum(self.embedding.weight**2, dim=1)
        # dist: (batch_size * H * W, K)
        _, embed_ind = (-dist).max(1)
        # embed_ind: (batch_size * H * W,)

        embed_onehot = F.one_hot(embed_ind, self.K).type(latents.dtype)
        # embed_onehot: (batch_size * H * W, K)
        embed_ind = embed_ind.view(*latents_shape[:-1])
        # embed_ind: (batch_size, H, W)

        quantized = torch.matmul(embed_onehot, self.embedding.weight)
        # quantized: (batch_size * H * W, D)
        quantized = quantized.view(*latents_shape)
        # quantized: (batch_size, H, W, D)
        diff = self.beta * (quantized.detach() - latents)
        # diff: (batch_size, H, W, D)
        quantized = latents + diff
        # quantized: (batch_size, H, W, D)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # quantized: (batch_size, D, H, W)
        return quantized, embed_ind