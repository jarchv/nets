import torch
import torchvision
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn  as nn

from torchvision import transforms
from torchvision.utils import make_grid

def dict_to_namespace(d):
    n = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(n, key, dict_to_namespace(value))
        else:
            setattr(n, key, value)
    return n
    
def to_net(tensor):
    return tensor * 2 - 1

def to_view(tensor):
    return tensor * 0.5 + 0.5

def save_batch(data, filename=None):
    plt.figure(figsize=(5, 5), dpi=80)

    plt.xticks([])
    plt.yticks([])
    
    m = torch.clamp(to_view(data.detach().cpu()), 0, 1)
    m = make_grid(m, padding=1, nrow=4, pad_value=100)
    m = m.permute(1,2,0)

    m = m * 255
    m = np.uint8(m.to(torch.int32).cpu().numpy())
    plt.imsave(filename, m, vmin=0, vmax=255)
    plt.close()

def get_celeb_data(img_size=64, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        #transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        to_net])
    celeb_dataset = torchvision.datasets.ImageFolder(
        root ='celeba',
        transform = transform)

    train_idx = range(162770)
    valid_idx = range(162770, 182637)
    test_idx  = range(182637,len(celeb_dataset))

    train_set = torch.utils.data.Subset(celeb_dataset, train_idx)
    valid_set = torch.utils.data.Subset(celeb_dataset, valid_idx)
    test_set  = torch.utils.data.Subset(celeb_dataset, test_idx)

    train_size = len(train_set)
    valid_size = len(valid_set)
    test_size  = len(test_set)
    
    assert (train_size + valid_size + test_size) == len(celeb_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False)

    data_dict = {}
    data_dict['train_loader'] = (train_loader, train_size)
    data_dict['valid_loader'] = (valid_loader, valid_size)
    data_dict['test_loader']  = (test_loader , test_size )

    return data_dict


# Discriminator

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)