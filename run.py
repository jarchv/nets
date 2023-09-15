import os
import yaml
import argparse
import importlib

utils = importlib.import_module("utils")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--model', default="vqgan", help='config file')
    args = parser.parse_args()
    
    if args.model == "vqgan":
        config = "vqgan.yaml"
        Trainer = importlib.import_module("train.train_vqgan").TrainVQGAN
    elif args.model == "ddpm":
        config = "ddpm.yaml"
        Trainer = importlib.import_module("train.train_ddpm").TrainDDPM
    elif args.model == "vqvae":
        config = "vqvae.yaml"
        Trainer = importlib.import_module("train.train_vqvae").TrainVQVAE
    else:
        print("Please specify a model to train.")
        exit()

    with open(os.path.join("configs", config), "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict_to_namespace(config)
    train_op = Trainer(config)
    #train_op.sample_batch()
    #train_op.show_inputs()
    train_op.train_step()