import os
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt

utils = importlib.import_module('utils')

class Trainer():
    def __init__(self, config):
        self.loader_dict = utils.get_celeb_data(
            img_size=config.data.img_size, batch_size=config.hyp.batch_size)
        
        self.train_load = self.loader_dict['train_loader'][0]
        self.train_size = self.loader_dict['train_loader'][1]
        self.valid_load = self.loader_dict['valid_loader'][0]

        checkpoints_path = os.path.join(
                config.log.save_dir,
                config.log.name,
                'log-%d' % config.log.log_num,
                'checkpoints')

        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        self.device = config.log.device
        self.lr     = config.hyp.lr

        self.checkpoints_path = checkpoints_path
        self.beta1  = config.hyp.beta1
        self.beta2  = config.hyp.beta2
        self.img_size = config.data.img_size
        self.batch_size = config.hyp.batch_size
        self.load_epoch = config.log.load_epoch
        self.save_freq = config.log.save_freq
        self.epochs = config.hyp.epochs
        self.model = None
        self.opt_model = None

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
        
    def show_inputs(self):
        fig = plt.figure(figsize=(5, 8), dpi=80)
        
        for image_batch, _ in self.train_load:
            utils.save_batch(image_batch, "inputs.png")
            break

    def sample_batch(self):
        self.__set_model()
        self.model.eval()
        x_hat = self.model.sample(self.batch_size).cpu().detach()
        utils.save_batch(x_hat, "sample.png")    



