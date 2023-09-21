import torch
import time
import itertools
import os
import importlib
import numpy as np
from torchsummary import summary

utils = importlib.import_module('utils')
model = importlib.import_module('ldm.ddpm')
vqgan = importlib.import_module('ldm.model')

from .train import Train

class TrainLDM(Train):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.timesteps = config.models.ddim.timesteps
        self.in_resolution = config.models.ddim.in_resolution
        self.set_model(config, config.log.device)
        self.vq_gan = vqgan.VQGAN(self.config.models.vqgan, self.device).to(self.device).eval()
        self.vq_gan.load_epoch(config.models.ddim.vqgan_epoch, config.models.ddim.vqgan_log)

    def set_model(self, config, device):
        self.model = model.GaussianDiffusion(config.models.ddim, device).to(device)
        self.opt_model = torch.optim.Adam(
            itertools.chain(self.model.parameters()), 
            lr=self.lr, betas=(config.train.beta1, config.train.beta2))
        
        summary(self.model, (3, self.in_resolution, self.in_resolution))   
        self.load_model()

    def optimize_parameters(self, x):
        loss = self.model(x)
        self.opt_model.zero_grad()
        loss.backward()
        self.opt_model.step()

        return loss.item()

    def save_model(self, ep):
        print('Saving "model-{:d}"... '.format(ep), end='')

        file_model = 'model-{:d}.pth'.format(ep)
        save_path  = os.path.join(self.checkpoints_path, file_model)
		
        checkpoint = {}

        checkpoint['state_dict_net'] = self.model.state_dict()   
        checkpoint['opt_model'] = self.opt_model.state_dict()

        torch.save(checkpoint, save_path)
        print("Done.")

    def load_model(self):
        if self.load_epoch <= 0:
            return
        print('\nLoading "model-{:d}"...'.format(self.load_epoch), end='')
        file_model = 'model-{:d}.pth'.format(self.load_epoch)

        load_path  = os.path.join(self.checkpoints_path, file_model)
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint['state_dict_net'])
        self.opt_model.load_state_dict(checkpoint['opt_model'])
        print("Done.")
    
    def sample(self, batch):
        self.model.eval()
        z_hat = self.model.sample(min(32,self.batch_size)).detach()

        z_shape = z_hat.shape
        embed_dim = z_shape[1]
        z_hat.permute(0, 2, 3, 1)                        # z: (B,H,W,C)
        z_flatten = z_hat.reshape(-1, embed_dim)       # z: (B*H*W,C)

        # Compute L2 distance
        dist = torch.sum(z_flatten**2, dim=1, keepdim=True) + \
               torch.sum(self.vq_gan.q.embedding.weight**2, dim=1) - \
               2 * torch.matmul(z_flatten, self.vq_gan.q.embedding.weight.t()) # dist: (B*H*W,N)

        min_embed_ind = torch.argmin(dist, dim=1)        # min_embed_ind: (B*H*W)
        min_embed_ind = min_embed_ind.unsqueeze(1)       # min_embed_ind: (B*H*W,1)
        print(min_embed_ind.shape)
        # Compute Quantize Loss
        min_embed = torch.zeros(min_embed_ind.shape[0], self.config.models.vqgan.num_embed, device=z_hat.device)
        min_embed.scatter_(1, min_embed_ind, 1)          # min_embed: (B*H*W,N)

        # Compute Quantize Latent Vectors
        z_q = torch.matmul(min_embed, self.vq_gan.q.embedding.weight) # z_q: (B*H*W,C)
        z_q = z_q.reshape(z_shape)                           # z_q: (B,C,H,W)

        x_hat = self.vq_gan.decode(z_q)
        return x_hat
    def train_step(self):
        x_hat = self.sample(32)
        utils.save_batch(x_hat, f"batch-{ep}.png")  
        return

        for ep in range(self.load_epoch + 1, self.epochs + 1):
            start_t = time.time()
            self.model.train()
            train_loss = []

            it = 0
            for image_batch, _ in self.train_load:
                z_q, codebook_ind, _ = vq_gan.encode(image_batch.to(self.device))
                loss = self.optimize_parameters(x=z_q.detach())

                it += image_batch.size(0)

                train_loss.append(loss)
                print(f"\rEpoch {ep:4d}[{it:5d}/{self.train_size:5d}] loss={train_loss[-1]:.3f}", end="")

            train_loss_mean = np.mean(train_loss)
            end_t = time.time()   
            print(f"\rEpoch {ep:4d}[{it:5d}/{self.train_size:5d}] loss={train_loss[-1]:.3f} in {end_t-start_t:.1f} sec.")
            
            self.model.eval()
            # Show sample
            for image_batch_v, _ in self.valid_load:
                x_hat = self.sample(32)
                utils.save_batch(x_hat, f"batch-{ep}.png")  
                break

            if ep % self.save_freq == 0: 
                #for image_batch_v, _ in self.valid_load:
                #    loss = self.model(image_batch.to(self.device))
                #    valid_loss.append(loss.item())
                #    break
                #valid_loss_mean = np.mean(valid_loss)   

                #print(f"\tloss(val)={valid_loss_mean:.3f}")
                self.save_model(ep)