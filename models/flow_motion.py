from models.modules.INN.INN import UnsupervisedMaCowTransformer3
from models.modules.INN.loss import FlowLoss
from models.opticalFlow.models import FlowVAE
from models.second_stage_video import PokeMotionModel
from utils.evaluation import color_fig, fig_matrix
from functools import partial
import wandb
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
import cv2
import yaml
from glob import glob
import os



class PokeMotionModelFixed(PokeMotionModel):
    def __init__(self, config, dirs):
        super(PokeMotionModelFixed, self).__init__(config, dirs)

    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        """ avoid pytorch lighting auto save params """
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

    def setup(self, device: torch.device):
        self.freeze()

class FlowVAEFixed(FlowVAE):
    def __init__(self, config):
        super(FlowVAEFixed, self).__init__(config)

    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        """ avoid pytorch lighting auto save params """
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

    def setup(self, device: torch.device):
        self.freeze()


class FlowMotion(pl.LightningModule):

    def __init__(self, config):

        super(FlowMotion, self).__init__()
        self.config = config
        self.VAE = FlowVAEFixed(config).eval()
        self.INN = UnsupervisedMaCowTransformer3(self.config["architecture"])
        motion_model = PokeMotionModelFixed
        lr = config["training"]["lr"]


        # os.system('ln -s /export/scratch3/ablattma/ipoke logs')
        ckpt_path = '/export/scratch3/ablattma/ipoke/second_stage/ckpt/plants_64/0/'
        ckpt_path = 'logs/second_stage/ckpt/plants_64/0/'
        ckpt_path = glob(ckpt_path + '*.ckpt')
        assert len(ckpt_path) == 1, 'checkpoints error for PokeMotionModel (i.e. second stage)'
        ckpt_path = ckpt_path[0]

        config_path = 'config/pretrained_models/plants_64.yaml'
        with open(config_path, 'r') as stream:
            config_motion = yaml.load(stream)
        self.dirs = create_dir_structure(config_motion['general'], 'flow_motion_16x8x8')
        self.motion_model = motion_model.load_from_checkpoint(ckpt_path, map_location="cpu", config=config_motion, strict=False, dirs=self.dirs)
        self.loss_func = FlowLoss()

        checkpoint = torch.load(config["checkpoint"]["VAE"], map_location='cpu')
        new_state_dict = OrderedDict()
        for key, value in checkpoint['state_dict'].items():
            new_key = key[6:] # trimming keys by removing "model."
            new_state_dict[new_key] = value

        m, u = self.VAE.load_state_dict(new_state_dict, strict=False)
        assert len(m) == 0, "VAE state_dict is missing pretrained params"
        del checkpoint


        self.apply_lr_scaling = "lr_scaling" in self.config["training"] and self.config["training"]["lr_scaling"]
        if self.apply_lr_scaling:
            end_it = self.config["training"]["lr_scaling_max_it"]

            self.lr_scaling = partial(linear_var, start_it=0, end_it=end_it, start_val=0., end_val=lr, clip_min=0.,
                                      clip_max=lr)
        # configure custom lr decrease
        self.custom_lr_decrease = self.config['training']['custom_lr_decrease']
        if self.custom_lr_decrease:
            start_it = 500  # 1000
            self.lr_adaptation = partial(linear_var, start_it=start_it, end_it=1000 * 1759, start_val=lr, end_val=0.,
                                         clip_min=0.,
                                         clip_max=lr)

        self.VAE.setup(self.device)
        self.VAE.eval()
        self.motion_model.setup(self.device)
        self.motion_model.eval()

    def forward_density_video(self, batch):
        out, logdet = self.motion_model.forward_density(batch)
        return out, logdet

    def forward(self, batch): # for early testing purposes
        batch = batch
        out_hat, _ = self.forward_density_video(batch)
        out, logdet = self.forward_density(batch)
        loss, loss_dict = self.loss_func(out, logdet)
        return loss  # + F.mse_loss(out, out_hat, reduction='sum')

    # def on_fit_start(self) -> None:
    #     self.VAE.setup(self.device)

    def forward_sample(self, batch, n_samples=1, n_logged_imgs=1):
        image_samples = []

        with torch.no_grad():

            for n in range(n_samples):
                flow_input, _, _ = self.VAE.encoder(batch['flow'])
                flow_input = torch.randn_like(torch.cat((flow_input, flow_input), 1)).detach()
                out = self.INN(flow_input, reverse=True)
                out = self.VAE.decoder([out[:,:16]], del_shape=False)
                image_samples.append(out[:n_logged_imgs])

        return image_samples

    def forward_density(self, batch):
        X = batch['flow']
        with torch.no_grad():
            encv, _, _ = self.VAE.encoder(X)
            rand = torch.randn_like(encv)
            # rand = torch.zeros_like(encv)
            encv = torch.cat((encv, rand), 1)

        out, logdet = self.INN(encv, reverse=False)

        return out, logdet


    def configure_optimizers(self):
        trainable_params = [{"params": self.INN.parameters(), "name": "flow"}, ]

        optimizer = torch.optim.Adam(trainable_params, lr=self.config["training"]['lr'], betas=(0.9, 0.999),
                                     weight_decay=self.config["training"]['weight_decay'], amsgrad=True)

        return [optimizer]

    def training_step(self,batch, batch_idx):

        out_hat, _ = self.forward_density_video(batch)
        out, logdet = self.forward_density(batch)
        loss, loss_dict = self.loss_func(out, logdet)
        loss_recon = F.mse_loss(out, out_hat, reduction='sum')*0.01
        loss_dict['reconstruction loss'] = loss_recon
        loss += loss_recon
        loss_dict["flow_loss"] = loss

        self.log_dict(loss_dict,prog_bar=True,on_step=True,logger=False)
        self.log_dict({"train/"+key: loss_dict[key] for key in loss_dict},logger=True,on_epoch=True,on_step=True)
        self.log("global_step",self.global_step)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate",lr,on_step=True,on_epoch=False,prog_bar=True,logger=True)

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            self.INN.eval()
            n_samples = self.config["logging"]["n_samples"]
            n_logged_imgs = self.config["logging"]["n_log_images"]
            with torch.no_grad():
                image_samples = self.forward_sample(batch,n_samples-1,n_logged_imgs)
                tgt_imgs = batch['flow'][:n_logged_imgs]
                image_samples.insert(0, tgt_imgs)

                enc, *_ = self.VAE.encoder(tgt_imgs)
                rec = self.VAE.decoder([enc], del_shape=False)
                image_samples.insert(1, rec)

                captions = ["target", "rec"] + ["sample"] * (n_samples - 1)
                img = fig_matrix(image_samples, captions)

            self.logger.experiment.history._step=self.global_step
            self.logger.experiment.log({"Image Grid train set":wandb.Image(img,
                                                                caption=f"Image Grid train @ it #{self.global_step}")}
                                        ,step=self.global_step, commit=False)

        #     if self.global_step % (self.config["logging"]["log_train_prog_at"]*10) == 0:
        #         img = color_fig(image_samples, captions)
        #         self.logger.experiment.log({"Image sample": wandb.Image(img, caption=f"Image sample @ it #{self.global_step}")}
        #                                    , step=self.global_step, commit=False)

        return loss

    def validation_step(self, batch, batch_id):

        with torch.no_grad():
            out_hat, _ = self.forward_density_video(batch)
            out, logdet = self.forward_density(batch)
            loss, loss_dict = self.loss_func(out, logdet)
            loss_recon = F.mse_loss(out, out_hat, reduction='sum')*0.01
            loss_dict['reconstruction loss'] = loss_recon
            loss += loss_recon
            loss_dict["flow_loss"] = loss

            self.log_dict({"val/" + key: loss_dict[key] for key in loss_dict}, logger=True, on_epoch=True)

        return {"loss": loss, "batch_idx": batch_id, "loss_dict": loss_dict}

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.apply_lr_scaling and self.global_step <= self.config["training"]["lr_scaling_max_it"]:
            # adjust learning rate
            lr = self.lr_scaling(self.global_step)
            opt = self.optimizers()
            for pg in opt.param_groups:
                pg["lr"] = lr

        if self.custom_lr_decrease and self.global_step >= 500:
            lr = self.lr_adaptation(self.global_step)
            # self.console_logger.info(f'global step is {self.global_step}, learning rate is {lr}\n')
            opt = self.optimizers()
            for pg in opt.param_groups:
                pg["lr"] = lr



def linear_var(
    act_it, start_it, end_it, start_val, end_val, clip_min, clip_max
):
    act_val = (
        float(end_val - start_val) / (end_it - start_it) * (act_it - start_it)
        + start_val
    )
    return np.clip(act_val, a_min=clip_min, a_max=clip_max)


def create_dir_structure(config, model_name):
    subdirs = ["ckpt", "config", "generated", "log"]


    # model_name = config['model_name'] if model_name is None else model_name
    structure = {subdir: os.path.join('wandb/logs_altmann',config["experiment"],subdir,model_name) for subdir in subdirs}
    return structure


if __name__ == '__main__':
    from data.datamodule import StaticDataModule
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    config_data = {
        'aug_deg': 15,
        'aug_trans': [0.1, 0.1],
        'augment_b': 0.4,
        'augment_c': 0.5,
        'augment_h': 0.15,
        'augment_s': 0.4,
        'augment_wo_dis': True,
        'batch_size': 2,
        'dataset': 'PlantDataset',
        'flow_weights': False,
        'max_frames': 10,
        'n_pokes': 5,
        'n_workers': 1,
        'normalize_flows': False,
        'object_weighting': False,
        'p_col': 0.8,
        'p_geom': 0.8,
        'poke_size': 5,
        'scale_poke_to_res': True,
        'spatial_size': [64,64],
        'split': 'official',
        'val_obj_weighting': False,
        'yield_videos': True,
        'zero_poke': True,
        'zero_poke_amount': 12,
        'filter': 'all'}

    datakeys = ['images', 'flow', 'poke']
    with open('config/VAE_INN.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    datamod = StaticDataModule(config['data'], datakeys=datakeys)
    datamod.setup()
    flowmotion = FlowMotion(config).cuda()

    for idx, batch in enumerate(datamod.train_dataloader()):
        batch['images'] = batch['images'].cuda()
        batch['flow'] = batch['flow'].cuda()
        batch['poke'][0] = batch['poke'][0].cuda()
        batch['poke'][1] = batch['poke'][1].cuda()

        print(flowmotion(batch))
        break
