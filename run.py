import os
import yaml
import argparse
import importlib

utils = importlib.import_module("utils")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Network")
    parser.add_argument('--model', default="ldm", help='config file')
    args = parser.parse_args()
    
    if args.model == "vqgan":
        config = "vqgan.yaml"
        Train = importlib.import_module("train.train_vqgan").TrainVQGAN
    elif args.model == "ldm":
        config = "ldm.yaml"
        Train = importlib.import_module("train.train_ldm").TrainLDM
    elif args.model == "vqgan-nodisc":          
        config = "vqgan-nodisc.yaml"
        Train = importlib.import_module("train.train_vqgan-nodisc").TrainVQGANNoDisc
    elif args.model == "ddpm":
        config = "ddpm.yaml"
        Train = importlib.import_module("train.train_ddpm").TrainDDPM
    elif args.model == "vqvae":
        config = "vqvae.yaml"
        Train = importlib.import_module("train.train_vqvae").TrainVQVAE
    else:
        print("Please specify a model to train.")
        exit()

    with open(os.path.join("configs", config), "r") as f:
        config = yaml.safe_load(f)
    config = utils.dict_to_namespace(config)
    train_op = Train(config)
    #train_op.sample_batch()
    #train_op.show_inputs()
    train_op.train_step()