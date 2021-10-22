from pytorch_lightning.loggers import WandbLogger
import pickle
import sys
import argparse
import torch
import pytorch_lightning as pl
import yaml
from experiments.fully_connected_ae import FCAEModel
from data.datamodule import StaticDataModule
import numpy as np
import os

def parse():
    parser = argparse.ArgumentParser('Fully Connected VAE')
    parser.add_argument('-g', '--gpus', type=int, required=True, help='gpu/gpus to run on e.g. -g 5')
    parser.add_argument('-c', '--config', default='config/BigGanAE.yaml', metavar='config_path')
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('-e', '--epoch', type=int, default=10, metavar='n_epochs')
    parser.add_argument('--offline', action='store_true', help='dont sync wandb logger')
    parser.add_argument('-r', '--resume', action='store_true', help='resume from local checkpoint')
    parser.add_argument('--skip_save', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # with open('/export/scratch/compvis/datasets/plants/processed_256_resized/plants_dataset_daniel.p', 'rb') as infile:
    #     data_frame = pickle.load(infile)
    args = parse()
    batch_size = args.batch_size
    gpus = args.gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
    os.environ["WANDB_DIR"] = "/export/data/daltmann/FCAEModel"
    print('gpu training on ', str(gpus))
    config_path = args.config
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # plants_train, plants_val = random_split(dataset, [42221 - 6000, 6000])
    # train_loader = DataLoader(plants_train,
    #                           batch_size=batch_size,
    #                           num_workers=len(gpus)*3,
    #                           shuffle=True,
    #                           pin_memory=True,
    #                           drop_last=True)
    # val_loader = DataLoader(plants_val,
    #                         batch_size=batch_size,
    #                         num_workers=len(gpus)*3,
    #                         pin_memory=True,
    #                         drop_last=True)
    datakeys = ['flow']
    # datakeys += ['poke']
    config['data']['batch_size'] = batch_size
    datamod = StaticDataModule(config['data'], datakeys=datakeys)


    # Logger
    wandb_logger = WandbLogger(project='Fully Connected BigGAN Autoencoder',
                               offline=args.offline,
                               notes=str(sys.argv),
                               name=config["general"]["run_name"]
                               )
    # training
    trainer = pl.Trainer(gpus=1, # int(1) = one gpu | [1] = '1' = gpu number 1
                         logger=wandb_logger,
                         max_epochs=args.epoch,
                         default_root_dir='/export/home/daltmann/master_project/tmp/ipoke-main-opticalflow/wandb/FCAEModel')
                         # resume_from_checkpoint='')
    if (args.resume) or (input("start from scratch? (y,n) ") == 'y'): # make sure new initialization is wanted
        # model
        model = FCAEModel
        if args.resume:
            model = model.load_from_checkpoint('wandb/FCAEModel.ckpt', strict=False, config=config)
        else:
            model = model(config)
        datamod.setup()
        trainer.fit(model, datamod.train_dataloader(), datamod.val_dataloader())
        if not args.skip_save:
            trainer.save_checkpoint('wandb/FCAEModel.ckpt')

