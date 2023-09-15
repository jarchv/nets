import torch
import time
import os
import itertools
import importlib
import numpy as np
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from torchsummary import summary
from tqdm import tqdm


utils = importlib.import_module('utils')
model = importlib.import_module('vqgan.model')
lpips = importlib.import_module('vqgan.lpips')
nns   = importlib.import_module('vqgan.nns')

from .train import Train

class TrainVQGAN(Train):
    def __init__(self, config):
        super().__init__(config)
        self.device = config.log.device
        self.disc_factor = config.train.disc_factor

        self.D = nns.NLayerDiscriminator(config.vqgan.model).to(self.device)
        self.D.apply(utils.weights_init)
        self.perceptual_loss = lpips.LPIPS().eval().to(self.device)
        self.set_model(config)

    def set_model(self, config):
        self.vqgan = model.VQGAN(config.vqgan.model, self.device).to(self.device)
        #summary(self.vqgan, (3, self.img_size, self.img_size))   
        self.opt_vq = torch.optim.Adam(
            itertools.chain(
                self.vqgan.E.parameters(),
                self.vqgan.G.parameters(),
                self.vqgan.q.parameters(),
                self.vqgan.quantize_conv.parameters(),
                self.vqgan.post_quant_conv.parameters()
            ),
            lr=config.train.lr,
            betas=(config.train.beta1, config.train.beta2))
        self.opt_dis = torch.optim.Adam(
            self.D.parameters(),
            lr=config.train.lr,
            betas=(config.train.beta1, config.train.beta2))

        self.scheduler1 = lr_scheduler.LinearLR(self.opt_vq, start_factor=1.0, end_factor=0.01, total_iters=10)
        self.scheduler2 = lr_scheduler.LinearLR(self.opt_dis, start_factor=1.0, end_factor=0.01, total_iters=10)
        self.load_model()

    def __get_vq_loss(self, x, epoch, train_length, i_batch):
        # Compute VQGAn outputs
        x_hat, codebook_ind, q_loss = self.vqgan(x)

        # Compute Discriminator outputs
        disc_real = self.D(x)
        disc_fake = self.D(x_hat)

        # When to use Discriminator
        disc_factor = self.vqgan.adopt_weight(
            self.disc_factor, (epoch-1) * train_length + i_batch, 1000)

        # Compute Perceptual Loss and Reconstruction Loss
        perceptual_loss = self.perceptual_loss(x, x_hat)
        rec_loss = torch.abs(x - x_hat)
        perceptual_rec_loss = perceptual_loss + rec_loss
        perceptual_rec_loss = perceptual_rec_loss.mean()

        # Compute Generator Loss
        g_loss = -torch.mean(disc_fake)

        # Compute VQGAN Loss: Encoder-Decoder Loss
        lambda_ = self.vqgan.get_lambda(perceptual_rec_loss, g_loss)
        vq_loss = perceptual_rec_loss + q_loss + disc_factor * lambda_ * g_loss

        return vq_loss, disc_real, disc_fake, disc_factor

    def __get_dis_loss(self, d_real, d_fake, disc_factor):
        disc_loss_real = torch.mean(F.relu(1. - d_real))
        disc_loss_fake = torch.mean(F.relu(1. + d_fake))
        dis_loss = disc_factor * 0.5 * (disc_loss_real + disc_loss_fake)
        return dis_loss, disc_loss_real, disc_loss_fake
        
    def optimize_parameters(self, x, epoch, train_length, it):
        vq_loss, d_real, d_fake, disc_factor = self.__get_vq_loss(x, epoch, train_length, it)

        self.opt_vq.zero_grad()
        vq_loss.backward(retain_graph=True)
        
        dis_loss, disc_loss_real, disc_loss_fake = self.__get_dis_loss(d_real, d_fake.detach(), disc_factor)
        self.opt_dis.zero_grad()
        dis_loss.backward()

        self.opt_vq.step()
        self.opt_dis.step()

        return vq_loss, dis_loss, disc_loss_real, disc_loss_fake

    def save_model(self, ep):
        print('Saving "model-{:d}"... '.format(ep), end='')

        file_model = 'model-{:d}.pth'.format(ep)
        save_path  = os.path.join(self.checkpoints_path, file_model)
		
        checkpoint = {}

        checkpoint['state_dict_net'] = self.vqgan.state_dict()   
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

        self.vqgan.load_state_dict(checkpoint['state_dict_net'])
        self.opt_vq.load_state_dict(checkpoint['opt_vq'])
        self.opt_dis.load_state_dict(checkpoint['opt_dis'])
        print("Done.")

    def train_step(self):
        for ep in range(self.load_epoch + 1, self.epochs + 1):
            start_t = time.time()
            self.vqgan.train()
            vq_loss_record = []
            dis_loss_record = []
            it = 0
            train_length = len(self.train_load)
            with tqdm(range(train_length), total=train_length, desc=f"Epoch {ep:3d}: ", ncols = 140) as pbar:
                for i_batch, (image_batch, _) in zip(pbar, self.train_load):
                    vq_loss, dis_loss, disc_loss_real, disc_loss_fake = self.optimize_parameters(
                        x=image_batch.to(self.device), epoch=ep, train_length=train_length, it=i_batch)
                    it += image_batch.size(0)

                    vq_loss_record.append(vq_loss.cpu().detach().numpy().item())
                    pbar.set_postfix(
                        VQ_Loss=f"{vq_loss.cpu().detach().numpy().item():.3f}",
                        DIS_Loss=f"{dis_loss.cpu().detach().numpy().item():.3f}",
                        DIS_Real=f"{disc_loss_real.cpu().detach().numpy().item():.3f}",
                        DIS_Fake=f"{disc_loss_fake.cpu().detach().numpy().item():.3f}"
                    )
                    pbar.update(0)
                vq_loss_mean = np.mean(vq_loss_record)
                end_t = time.time()   
                print(f"\rEpoch {ep:3d}: VQ_Loss={vq_loss_mean:.3f} in {(end_t-start_t)/60:.1f} min.")
                
                self.vqgan.eval()
                # Show sample
                x_hat  = self.vqgan(image_batch[:16,:,:,:].to(self.device))[0].cpu().detach()
                merged = torch.cat((image_batch[:16,:,:,:], x_hat), dim=0)
                
                utils.save_batch(merged, f"batch-{ep}.png")  

                if ep % self.save_freq == 0:
                    self.save_model(ep)
        self.scheduler1.step()
        self.scheduler2.step()