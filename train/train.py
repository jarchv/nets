import os
import importlib
import numpy as np
import torch
import matplotlib.pyplot as plt

utils = importlib.import_module('utils')

class Train():
    def __init__(self, config):
        self.loader_dict = utils.get_celeb_data(
            img_size=config.data.img_size, batch_size=config.train.batch_size)
        
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
        self.lr     = config.train.lr

        self.checkpoints_path = checkpoints_path
        self.beta1  = config.train.beta1
        self.beta2  = config.train.beta2
        self.img_size = config.data.img_size
        self.batch_size = config.train.batch_size
        self.load_epoch = config.log.load_epoch
        self.save_freq = config.log.save_freq
        self.epochs = config.train.epochs


