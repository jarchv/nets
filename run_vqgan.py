import os
import yaml
import argparse
from train.trainer_vqgan import TrainerVQGAN as Trainer

def dict_to_namespace(d):
    n = argparse.Namespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(n, key, dict_to_namespace(value))
        else:
            setattr(n, key, value)
    return n

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--config', default='vqgan.yaml', help='config file')
    args = parser.parse_args()
    
    
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict_to_namespace(config)
    train_op = Trainer(config)
    #train_op.sample_batch()
    #train_op.show_inputs()
    train_op.train_step()