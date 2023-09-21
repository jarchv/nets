import torch
import time
import itertools
import importlib
import numpy as np
from torchsummary import summary

utils = importlib.import_module('utils')
model = importlib.import_module('ddpm.model')

from .trainer import Trainer

class TrainDDPM(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.timesteps = config.hyp.timesteps

    def set_model(self):
        self.model = model.GaussianDiffusion(
            img_ch=3,
            img_size=self.img_size,
            init_dim=64,
            timesteps=self.timesteps,
            device=self.device).to(self.device)
        self.opt_model = torch.optim.Adam(
            itertools.chain(self.model.parameters()), 
            lr=self.lr, betas=(self.beta1, self.beta2))
        
        summary(self.model, (3, self.img_size, self.img_size))   
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
                    loss = self.model(image_batch.to(self.device))
                    valid_loss.append(loss.item())
                    #break
                valid_loss_mean = np.mean(valid_loss)   

                print(f"\tloss(val)={valid_loss_mean:.3f}")
                self.save_model(ep)