import torch
import time
import itertools
import importlib
import numpy as np
from torchsummary import summary
import torch.nn.functional as F

utils = importlib.import_module('utils')
model = importlib.import_module('vqvae.model')

from .trainer import Trainer

class TrainerVQVAE(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.in_dim = config.hyp.in_dim
        self.h_dim = config.hyp.h_dim
        self.res_h_dim = config.hyp.res_h_dim
        self.n_res_layers = config.hyp.n_res_layers
        self.num_embeddings = config.hyp.num_embeddings
        self.embedding_dim = config.hyp.embedding_dim
        self.beta = config.hyp.beta
        
    def set_model(self):
        self.model = model.VQVAE(
            in_dim          = self.in_dim,
            h_dim           = self.h_dim,
            res_h_dim       = self.res_h_dim,
            n_res_layers    = self.n_res_layers,
            K               = self.num_embeddings,
            D               = self.embedding_dim,
            beta            = self.beta,
            img_size        = self.img_size,
            device          = self.device).to(self.device)
        
        #summary(self.model, (3, self.img_size, self.img_size))   
        self.opt_model = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2), amsgrad=True)
        self.load_model()
    
    def __get_loss(self, x):
        embed_loss, x_hat, perplexity = self.model(x)
        rec_loss = F.mse_loss(x_hat, x)
        loss = rec_loss + embed_loss
        return loss
        
    def optimize_parameters(self, x):
        loss = self.__get_loss(x)

        self.opt_model.zero_grad()
        loss.backward()
        self.opt_model.step()

        return loss.item()

    def train_step(self):
        self.set_model()

        x_hat = self.model.sample(min(16,self.batch_size)).cpu().detach()
        utils.save_batch(x_hat, f"batch-{0}.png") 

        for ep in range(self.load_epoch + 1, self.epochs + 1):
            start_t = time.time()
            self.model.train()
            train_loss = []

            it = 0
            for image_batch, _ in self.train_load:
                loss = self.optimize_parameters(x=image_batch.to(self.device))

                it += image_batch.size(0)

                train_loss.append(loss)
                print(f"\rEpoch {ep:4d}[{it:5d}/{self.train_size:5d}] loss={train_loss[-1]:.3f}", end="")

            train_loss_mean = np.mean(train_loss)
            end_t = time.time()   
            print(f"\rEpoch {ep:4d}[{it:5d}/{self.train_size:5d}] loss={train_loss[-1]:.3f} in {end_t-start_t:.1f} sec.")
            
            self.model.eval()
            # Show sample
            x_hat = self.model.sample(min(16,self.batch_size)).cpu().detach()
            utils.save_batch(x_hat, f"batch-{ep}.png")  

            if ep % self.save_freq == 0: 
                valid_loss = []

                for image_batch_v, _ in self.valid_load:
                    loss = self.__get_loss(image_batch.to(self.device))
                    valid_loss.append(loss.item())
                    #break
                valid_loss_mean = np.mean(valid_loss)   

                print(f"\tloss(val)={valid_loss_mean:.3f}")
                self.save_model(ep)