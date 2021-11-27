from pytorch_lightning.loggers import WandbLogger
import sys
import argparse
import torch
import yaml
from data.datamodule import StaticDataModule
from models.modules.autoencoders.big_ae import BigAE
from collections import OrderedDict

import numpy as np
import os
import json

def angular_error(gt, pred):
    gt = np.pad(gt, [[0,0],[0,1],[0,0],[0,0]], 'constant', constant_values=1)
    pred = np.pad(pred, [[0,0],[0,1],[0,0],[0,0]], 'constant', constant_values=1)

    # u, v = pred[0], pred[1]
    # u_gt, v_gt = gt[0], gt[1]
    # AE = np.arccos((1. + u*u_gt + v*v_gt)/(np.sqrt(1.+u**2+v**2)*np.sqrt(1.+u_gt**2+v_gt**2)))
    AE = np.arccos(np.sum(gt*pred, axis=1)/(np.linalg.norm(gt, axis=1) * np.linalg.norm(pred, axis=1)))
    return AE

def endpoint_error(gt, pred):
    EE = np.linalg.norm(gt-pred, axis=1)
    return EE

class BigAEfixed(BigAE):

    def __init__(self, config):
        super(BigAEfixed, self).__init__(config)
        checkpoint = torch.load(config["checkpoint_ae"], map_location='cpu')
        new_state_dict = OrderedDict()
        for key, value in checkpoint['state_dict'].items():
            # if key[3:] == 'ae.':
            new_key = key[3:]  # trimming keys by removing "ae."
            new_state_dict[new_key] = value
        m, u = self.load_state_dict(new_state_dict, strict=False)
        assert len(m) == 0, "BigAE state_dict is missing pretrained params"
        del checkpoint

    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        """ avoid pytorch lighting auto save params """
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination


def parse():
    parser = argparse.ArgumentParser('Fully Connected VAE')
    parser.add_argument('-g', '--gpus', type=int, required=True, help='gpu/gpus to run on e.g. -g 5')
    parser.add_argument('-c', '--config', default='config/BigGanAE.yaml', metavar='config_path')
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    parser.add_argument('--offline', action='store_true', help='dont sync wandb logger')
    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    args = parse()
    batch_size = args.batch_size
    gpus = args.gpus
    device = f"cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
    os.environ["WANDB_DIR"] = "/export/compvis-nfs/user/daltmann/scratch/FCAEModel"
    print('gpu: ', str(gpus))
    config_path = args.config
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    datakeys = ['flow']
    config['data']['batch_size'] = batch_size
    config["architecture"]["in_size"] = config["data"]["spatial_size"][0]
    datamod = StaticDataModule(config['data'], datakeys=datakeys)

    # Logger
    wandb_logger = WandbLogger(project='Fully Connected BigGAN Autoencoder',
                               offline=args.offline,
                               notes=str(sys.argv),
                               name='evaluation'
                               )
    datamod.setup()

    z_dim_list = [256, 512, 1024]
    R = {}
    for z_dim in z_dim_list:
        config['architecture']['z_dim'] = z_dim
        path = f"scratch/FCAEModel_{z_dim}.ckpt"
        config['architecture']['checkpoint_ae'] = path
        R[path] = {'angular_error': {'2.5'  : [],
                                      '5.0' : [],
                                      '10.0': []},
                   'endpoint_error': {'0.5' : [],
                                      '1.0' : [],
                                      '2.0' : []}}
        model = BigAEfixed
        model = model(config['architecture']).eval()
        model.to(device)

        for batch in datamod.val_dataloader():
            flow = batch['flow'].to(device)
            rec, _, _ = model(flow)
            rec_cpu = np.array(rec.detach().cpu())
            batch_cpu = np.array(flow.detach().cpu())


            AE = angular_error(batch_cpu, rec_cpu)
            EE = endpoint_error(batch_cpu, rec_cpu)
            normalization = np.minimum(np.linalg.norm(rec_cpu,axis=1), np.linalg.norm(batch_cpu, axis=1))
            normalization = np.minimum(normalization, np.zeros_like(normalization) + 1e-4)
            NEE = EE/normalization
            R_temp = AE[AE>2.5*np.pi/180]
            R[path]['angular_error']['2.5'] += [R_temp.size/AE.size]
            R_temp = R_temp[R_temp>5.*np.pi/180]
            R[path]['angular_error']['5.0'] += [R_temp.size/AE.size]
            R_temp = R_temp[R_temp>10.*np.pi/180]
            R[path]['angular_error']['10.0'] += [R_temp.size/AE.size]

            R_temp = EE[EE>0.5]
            R[path]['endpoint_error']['0.5'] += [R_temp.size/EE.size]
            R_temp = R_temp[R_temp>1.0]
            R[path]['endpoint_error']['1.0'] += [R_temp.size/EE.size]
            R_temp = R_temp[R_temp>2.0]
            R[path]['endpoint_error']['2.0'] += [R_temp.size/EE.size]

            R_temp = NEE[NEE>0.5]
            R[path]['endpoint_error_normalized']['0.5'] += [R_temp.size/NEE.size]
            R_temp = R_temp[R_temp>1.0]
            R[path]['endpoint_error_normalized']['1.0'] += [R_temp.size/NEE.size]
            R_temp = R_temp[R_temp>2.0]
            R[path]['endpoint_error_normalized']['2.0'] += [R_temp.size/NEE.size]
        print(path)
        print(np.mean(R[path]['endpoint_error']['2.0']))

    with open('scratch/FCAE_eval.json', 'w+') as f:
        json.dump(R, f, indent=4)
    wandb_logger.experiment.log(R)

