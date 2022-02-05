from pytorch_lightning.loggers import WandbLogger
import pickle
import sys
import argparse
import torch
import pytorch_lightning as pl
import yaml
from experiments.fully_connected_video_ae import FCBaseline
from data.datamodule import StaticDataModule
import numpy as np
import os

def parse():
    parser = argparse.ArgumentParser('Fully Connected VAE for Videos')
    parser.add_argument('-g', '--gpus', type=int, required=True, help='gpu/gpus to run on e.g. -g 5')
    parser.add_argument('-c', '--config', default='config/baseline_fc.yaml', metavar='config_path')
    # parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-e', '--epoch', type=int, default=10, metavar='n_epochs')
    parser.add_argument('--offline', action='store_true', help='dont sync wandb logger')
    parser.add_argument('-r', '--resume', action='store_true', help='resume from local checkpoint')
    parser.add_argument('--skip_save', action='store_true')
    parser.add_argument('--z_dim', type=int, default=None, help='latent dimension of the autoencoder')

    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # with open('/export/scratch/compvis/datasets/plants/processed_256_resized/plants_dataset_daniel.p', 'rb') as infile:
    #     data_frame = pickle.load(infile)
    args = parse()
    # batch_size = args.batch_size
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
    os.environ["WANDB_DIR"] = "/export/compvis-nfs/user/daltmann/scratch/FCAEModelVideos"
    print('gpu training on ', str(gpus))
    config_path = args.config
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    if args.z_dim is not None:
        config['architecture']['z_dim'] = args.z_dim
    batch_size = config['data']['batch_size']
    z_dim = config['architecture']['z_dim']
    config['general']['run_name'] = config['general']['run_name'].replace('z_dim', str(z_dim))

    datakeys = ['images']
    # datakeys += ['poke']
    config['data']['batch_size'] = batch_size
    datamod = StaticDataModule(config['data'], datakeys=datakeys)
    # model
    model = FCBaseline
    if args.resume:
        assert args.resume != True, "resume not implemented"
        # model = model.load_from_checkpoint(f'scratch/FCAEModel_{z_dim}.ckpt', strict=False, config=config)
    else:
        model = model(config)
    datamod.setup()

    # Logger
    wandb_logger = WandbLogger(project='Fully Connected BigGAN Autoencoder Video',
                               offline=args.offline,
                               notes=str(sys.argv),
                               name=config["general"]["run_name"]
                               )
    # training
    trainer = pl.Trainer(gpus=1, # int(1) = one gpu | [1] = '1' = gpu number 1
                         logger=wandb_logger,
                         max_epochs=args.epoch,
                         default_root_dir='/export/home/daltmann/master_project/tmp/ipoke-main-opticalflow/scratch/FCAEModelVideos'
                         )
    #if (args.resume) or (input("start from scratch? (y,n) ") == 'y'): # make sure new initialization is wanted

    trainer.fit(model, datamod.train_dataloader(), datamod.val_dataloader())
    log_config = trainer.__getattribute__('default_root_dir')
    log_config += wandb_logger.experiment.__getattribute__('path')[7:]
    log_config += '/config.yaml'
    with open(log_config, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    if not args.skip_save:
        trainer.save_checkpoint(f'scratch/FCAEModelVideos/FCAEModelVideos_{z_dim}.ckpt')
