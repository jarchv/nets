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

    def forward(self, latents):
        # latents: (batch_size, D, H, W)
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape

        # latents_flattened: (batch_size * H * W, D)
        latents_flattened = latents.view(-1, self.D)

        # self.embedding.weight: (K, D)
        # dist: (batch_size * H * W, K)
        dist = torch.sum(latents_flattened**2, dim=1, keepdim=True) \
               - 2 * torch.matmul(latents_flattened, self.embedding.weight.t()) \
               + torch.sum(self.embedding.weight**2, dim=1)

        # min_embed_ind: (batch_size * H * W, 1)
        min_embed_ind = torch.argmin(dist, dim=1).unsqueeze(1) 

        # min_embed: (batch_size * H * W, K)
        min_embed = torch.zeros(min_embed_ind.shape[0], self.K).to(latents.device)

        # 'scatter_' reemplaza los valores de min_embed en las posiciones indicadas
        # por min_embed_ind con 1
        min_embed.scatter_(1, min_embed_ind, 1)
    
        # z_q: (batch_size, H, W, D)
        z_q = torch.matmul(min_embed, self.embedding.weight).view(latents_shape)

        # loss: (batch_size * H * W,). Al colocar z_q.detach() se evita que el
        # gradiente fluya por z_q y se calcula el gradiente solo por z_e (latents)
        loss = torch.mean((z_q.detach() - latents)**2) \
                    + self.beta * torch.mean((z_q - latents.detach())**2)

        # Permite que el gradiente fluya por z_q pero no por z_e (latents)
        z_q = latents + (z_q - latents).detach()
        
        # e_mean: (K,)
        e_mean = torch.mean(min_embed, dim=0)
       
        # 'perplexity' es una medida de la calidad de la compresion de los datos
        # mientras mas bajo, mejor es la compresion (idealmente 1).
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # z_q: (batch_size, D, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_embed, min_embed_ind

if __name__ == "__main__":
    vq = VectorQuantizer(8, 10)
    x = torch.randn(2, 10, 4, 4)
    l, z_q, p, min_embed, min_embed_ind = vq(x)
