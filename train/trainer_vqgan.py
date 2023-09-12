import torch
import time
import os
import itertools
import importlib
import numpy as np
from torchsummary import summary
import torch.nn.functional as F

utils = importlib.import_module('utils')
model = importlib.import_module('vqgan.model')
from .trainer import Trainer

class TrainerVQGAN(Trainer):
    def __init__(self, config):
        super().__init__(config)
        
        self.lr = config.hyp.lr
        self.beta1 = config.hyp.beta1
        self.beta2 = config.hyp.beta2
        self.epochs = config.hyp.epochs

        self.in_dim = config.hyp.in_dim
        self.hid_dim = config.hyp.hid_dim
        self.f_channels = config.hyp.f_dims
        self.num_res_layers = config.hyp.num_res_layers
        self.attn_resolutions = config.hyp.attn_resolutions
        self.resolution = config.hyp.resolution

        self.num_embeddings = config.hyp.num_embeddings
        self.embedding_dim = config.hyp.embedding_dim
        self.beta = config.hyp.beta

        self.save_freq = config.log.save_freq
        self.load_epoch = config.log.load_epoch
        self.device = config.log.device
    def set_model(self):
        self.model = model.VQGAN(
            img_channels     = self.in_dim,
            hid_channels     = self.hid_dim,
            f_channels       = self.f_channels,
            num_res_layers   = self.num_res_layers,
            attn_resolutions = self.attn_resolutions,
            resolution       = self.resolution,
            num_embeddings   = self.num_embeddings,
            embedding_dim    = self.embedding_dim,
            beta             = self.beta,
            device           = self.device).to(self.device)
        
        #summary(self.model, (3, self.img_size, self.img_size))   
        self.opt_vq = torch.optim.Adam(
            itertools.chain(
                self.model.E.parameters(),
                self.model.G.parameters(),
                self.model.q.parameters(),
                self.model.quantize_conv.parameters(),
                self.model.post_quant_conv.parameters()
            ),
            lr=self.lr,
            betas=(self.beta1, self.beta2))

        self.opt_dis = torch.optim.Adam(
            self.model.D.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2))
        #self.perceptual_loss = LPIPS().eval().to(self.device)
        self.load_model()

    def __get_vq_loss(self, x):
        x_hat, codebook_ind, q_loss = self.model(x)

        # Compute GAN loss
        d_real = self.model.D(x)
        d_fake = self.model.D(x_hat)

        # Compute VQ Loss
        rec_loss = F.mse_loss(x, x_hat)
        g_loss = -torch.mean(d_fake)

        lambda_ = self.model.get_lambda(rec_loss, g_loss)
        vq_loss = rec_loss + q_loss + lambda_ * g_loss
        return vq_loss, d_real, d_fake

    def __get_dis_loss(self, d_real, d_fake):
        d_loss_real = torch.mean(F.relu(1. - d_real))
        d_loss_fake = torch.mean(F.relu(1. + d_fake))
        dis_loss = d_loss_real + d_loss_fake
        return dis_loss * 0.5 
        
    def optimize_parameters(self, x):
        vq_loss, d_real, d_fake = self.__get_vq_loss(x)

        self.opt_vq.zero_grad()
        vq_loss.backward(retain_graph=True)
        
        dis_loss = self.__get_dis_loss(d_real, d_fake)
        self.opt_dis.zero_grad()
        dis_loss.backward()

        self.opt_vq.step()
        self.opt_dis.step()

        return vq_loss.item(), dis_loss.item()

    def save_model(self, ep):
        print('Saving "model-{:d}"... '.format(ep), end='')

        file_model = 'model-{:d}.pth'.format(ep)
        save_path  = os.path.join(self.checkpoints_path, file_model)
		
        checkpoint = {}

        checkpoint['state_dict_net'] = self.model.state_dict()   
        checkpoint['opt_vq'] = self.opt_vq.state_dict()
        checkpoint['opt_dis'] = self.opt_dis.state_dict()

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
        self.opt_vq.load_state_dict(checkpoint['opt_vq'])
        self.opt_dis.load_state_dict(checkpoint['opt_dis'])
        print("Done.")

    def train_step(self):
        self.set_model()

        for ep in range(self.load_epoch + 1, self.epochs + 1):
            start_t = time.time()
            self.model.train()
            train_loss = []

            it = 0
            for image_batch, _ in self.train_load:
                loss, _ = self.optimize_parameters(x=image_batch.to(self.device))

                it += image_batch.size(0)

                train_loss.append(loss)
                print(f"\rEpoch {ep:4d}[{it:5d}/{self.train_size:5d}] loss={train_loss[-1]:.3f}", end="")

            train_loss_mean = np.mean(train_loss)
            end_t = time.time()   
            print(f"\rEpoch {ep:4d}[{it:5d}/{self.train_size:5d}] loss={train_loss[-1]:.3f} in {end_t-start_t:.1f} sec.")
            
            self.model.eval()
            # Show sample
            x_hat = self.model.get_x_hat(image_batch[:16,:,:,:].to(self.device)).cpu().detach()
            utils.save_batch(x_hat, f"batch-{ep}.png")  

            if ep % self.save_freq == 0: 
                valid_loss = []

                for image_batch_v, _ in self.valid_load:
                    loss, *rest = self.__get_vq_loss(image_batch.to(self.device))
                    valid_loss.append(loss.item())
                    #break
                valid_loss_mean = np.mean(valid_loss)   

                print(f"\tloss(val)={valid_loss_mean:.3f}")
                self.save_model(ep)