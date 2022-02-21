import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam, lr_scheduler
import wandb
from os import path,makedirs
from collections import OrderedDict
import yaml
import numpy as np
from functools import partial
from lpips import LPIPS as lpips_net
import os
import cv2
from tqdm import tqdm
import pandas as pd
import time
import math

from utils.evaluation import fig_matrix
from models.modules.autoencoders.big_ae import BigAE
# from models.first_stage_motion_model import FCBaseline
from models.second_stage_video_fc import PokeMotionModelFC
# from models.first_stage_motion_model import SpadeCondMotionModel
from models.pretrained_models_fc import second_stage_models, flow_encoder_models
# from models.modules.autoencoders.baseline_fc_models import FirstStageFCWrapper
from models.modules.INN.INN import UnsupervisedTransformer3
from models.modules.INN.loss import FlowLoss
# from models.modules.autoencoders.util import Conv2dTransposeBlock
# from models.modules.INN.coupling_flow_alternative import AdaBelief
from utils.logging import make_flow_video_with_samples, log_umap, make_samples_and_samplegrid, save_video, make_transfer_grids_new, make_multipoke_grid
from utils.general import linear_var, get_logger
from utils.metrics import FVD, calculate_FVD, LPIPS,PSNR_custom,SSIM_custom, KPSMetric, metric_vgg16, compute_div_score,SampleLPIPS, SampleSSIM, compute_div_score_mse, compute_div_score_lpips
# from utils.posenet_wrapper import PoseNetWrapper
# from models.pose_estimator.tools.infer import save_batch_image_with_joints

class FlowINNFC(pl.LightningModule):

    def __init__(self,config,dirs):
        super().__init__()
        #self.automatic_optimization=False
        self.config = config
        self.embed_poke = True
        self.dirs = dirs
        self.test_mode = self.config['general']['test']
        self.n_test_samples = self.config['testing']['n_samples_per_data_point']


        self.console_logger = get_logger()

        # configure learning rate scheduling, if intended
        self.apply_lr_scaling = "lr_scaling" in self.config["training"] and self.config["training"]["lr_scaling"]
        if self.apply_lr_scaling:
            end_it = self.config["training"]["lr_scaling_max_it"]
            lr = self.config["training"]["lr"]
            self.lr_scaling = partial(linear_var, start_it=0, end_it=end_it, start_val=0., end_val=lr, clip_min=0.,
                                      clip_max=lr)

        #self.embed_poke_and_image = self.config["poke_embedder"]["embed_poke_and_image"]
        self.__initialize_second_stage()
        self.first_stage_config = self.second_stage_model.first_stage_config # to make 4D flow sample input
        self.config["architecture"]["flow_in_channels"] = self.first_stage_config["architecture"]["flow_in_channels"]

        self.metrics_dir = path.join(self.dirs['generated'],'metrics')
        os.makedirs(self.metrics_dir,exist_ok=True)


        self.FVD = FVD(n_samples=self.config['logging']['n_fvd_samples'] if 'n_fvd_samples' in  self.config['logging'] else 1000)
        if self.test_mode == 'none' or self.test_mode=='accuracy':
            self.lpips_metric = LPIPS()
            self.ssim = SSIM_custom()
            self.psnr = PSNR_custom()
            self.lpips_net = lpips_net()


        self.__initialize_flow_encoder()

        self.config["architecture"]["flow_mid_channels"] = int(self.config["architecture"]["flow_mid_channels_factor"] * \
                                                               self.config["architecture"]["flow_in_channels"])
        self.config["architecture"]["flow_in_channels"] = self.first_stage_config["architecture"]["z_dim"]
        self.INN = UnsupervisedTransformer3(**config['architecture'])

        self.loss_func = FlowLoss(spatial_mean=self.spatial_mean_for_loss,logdet_weight=1.)
        self.apply_lr_scaling = "lr_scaling" in self.config["training"] and self.config["training"]["lr_scaling"]
        lr = self.config["training"]["lr"]
        if self.apply_lr_scaling:
            end_it = self.config["training"]["lr_scaling_max_it"]
            self.lr_scaling = partial(linear_var, start_it=0, end_it=end_it, start_val=0., end_val=lr, clip_min=0., clip_max=lr)

        self.custom_lr_decrease = self.config['training']['custom_lr_decrease']
        if self.custom_lr_decrease:
            start_it = self.config["training"]["lr_scaling_max_it"]  # 1000
            self.lr_adaptation = partial(linear_var, start_it=start_it, start_val=lr, end_val=0., clip_min=0.,
                                         clip_max=lr)


    def __initialize_second_stage(self):
        dic = second_stage_models[self.config['second_stage']['name']]
        second_stage_ckpt = dic['ckpt']

        second_stage_config = path.join(*second_stage_ckpt.split('/')[:-2],'config.yaml').replace('ckpt','config')

        with open(second_stage_config) as f:
            self.second_stage_config = yaml.load(f, Loader=yaml.FullLoader)


        self.second_stage_model = PokeMotionModelFCFixed.load_from_checkpoint(second_stage_ckpt,config=self.second_stage_config,
                                                                        train=False,strict=False,dirs=self.dirs)

    def __initialize_flow_encoder(self):
        dic = flow_encoder_models[self.config['flow_encoder']['name']]
        flow_enc_ckpt = dic['ckpt']

        flow_enc_config = path.join(*flow_enc_ckpt.split('/')[:-2],'config.yaml').replace('ckpt','config')

        with open(flow_enc_config) as f:
            self.flow_enc_config = yaml.load(f, Loader=yaml.FullLoader)
        self.flow_encoder = FlowEncoderFixed(self.flow_enc_config)

        state_dict = torch.load(flow_enc_ckpt, map_location="cpu")
        # remove keys from checkpoint which are not required
        state_dict = {key[3:]: state_dict["state_dict"][key] for key in state_dict["state_dict"] if
                      key[:2] == 'ae'}
        # load first stage model
        m, u = self.flow_encoder.load_state_dict(state_dict, strict=False)
        assert len(m) == 0, f'poke_embedder is missing keys {m}'
        del state_dict

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.apply_lr_scaling and self.global_step < self.config["training"]["lr_scaling_max_it"]:
            # adjust learning rate
            lr = self.lr_scaling(self.global_step)
            opt = self.optimizers()
            self.log("learning_rate",lr,prog_bar=True,logger=True,on_step=True,on_epoch=False)
            for pg in opt.param_groups:
                pg["lr"] = lr


        if self.custom_lr_decrease and self.global_step >= self.config["training"]["lr_scaling_max_it"]:
            lr = self.lr_adaptation(self.global_step)
            # self.console_logger.info(f'global step is {self.global_step}, learning rate is {lr}\n')
            opt = self.optimizers()
            for pg in opt.param_groups:
                pg["lr"] = lr

    def make_flow_input(self,batch,reverse=False):
        X = batch['flow']
        if reverse:
            shape = [self.config['data']['batch_size'], self.config["architecture"]["flow_in_channels"], 1, 1]
            flow_input = torch.randn(shape, dtype=X.dtype, layout=X.layout, device=X.device).detach()
        else:
            with torch.no_grad():
                enc_vec = self.flow_encoder.encode(X).sample()
                motion_rest_shape = enc_vec.shape
                motion_rest_shape[1] = self.config['architecture']['z_dim'] - self.flow_enc_config['architecture']['z_dim']
                assert motion_rest_shape[1] >= 0, f'{self.config["architecture"]["z_dim"]} < {self.flow_enc_config["architecture"]["z_dim"]}'
                if motion_rest_shape[1] != 0:
                    motion_rest = torch.randn(motion_rest_shape, dtype=enc_vec.dtype, layout=enc_vec.layout, device=enc_vec.device)
                    flow_input = torch.cat((enc_vec, motion_rest), 1)
                else:
                    flow_input = enc_vec

        return flow_input

    def on_train_epoch_start(self):

        if self.custom_lr_decrease:
            n_train_iterations = self.config['training']['n_epochs'] * self.trainer.num_training_batches
            self.lr_adaptation = partial(self.lr_adaptation, end_it=n_train_iterations)

    def forward_sample(self, batch, n_samples=1, n_logged_imgs=1):
        image_samples = []

        with torch.no_grad():
            for n in range(n_samples):
                flow_input = self.make_flow_input(batch, reverse=True)
                out = self.INN(flow_input, reverse=True)
                out = out[:n_logged_imgs,:self.flow_enc_config['architecture']['z_dim']]
                if len(out.shape) == 4:
                    out = out.squeeze(-1).squeeze(-1)
                out = self.flow_encoder.decode(out)
                image_samples.append(out)

        return image_samples

    def forward_density(self, batch):
        flow_input = self.make_flow_input(batch)

        out, logdet = self.INN(flow_input.detach(), reverse=False)

        return out, logdet

    def training_step(self, batch, batch_idx):

        out, logdet = self.forward_density(batch)
        with torch.no_grad:
            out_hat, _  = self.second_stage_model.forward_density(batch)

        loss, loss_dict = self.loss_func(out, logdet)
        loss_recon = torch.nn.functionals.smooth_l1_loss(out, out_hat, reduction='mean')
        loss_dict['reconstruction loss'] = loss_recon.detach()
        loss += loss_recon * self.weight_recon
        loss_dict["flow_loss"] = loss

        self.log_dict(loss_dict, prog_bar=True, on_step=True, logger=False)
        self.log_dict({"train/" + key: loss_dict[key] for key in loss_dict}, logger=True, on_epoch=True, on_step=True)
        self.log("global_step", self.global_step, on_epoch=False, logger=True, on_step=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            n_samples = self.config["logging"]["n_samples"]
            n_logged_images = self.config["logging"]["n_log_images"]

            optical_flow = self.forward_sample(batch, n_samples, n_logged_images)
            tgt_imgs = batch['flow'][:n_logged_images] # not target but ground truth
            optical_flow.insert(0, tgt_imgs)
            captions = ["ground truth"] + ["sample"] * n_samples
            img = fig_matrix(optical_flow, captions)

            self.logger.experiment.history._step = self.global_step
            self.logger.experiment.log({"Image Grid train set": wandb.Image(img,
                                                                            caption=f"Image Grid train @ it #{self.global_step}")}
                                       , step=self.global_step, commit=False)

        return loss

    # def __make_umap_samples(self,key,batch):
    #     X = batch["images"]
    #     with torch.no_grad():
    #         # posterior
    #         z_p, z_m = self.encode_first_stage(X)
    #
    #         if self.augment_input:
    #             # scale samples with samples
    #             input_augment = torch.randn(
    #                 (z_p.size(0), self.config['architecture']['augment_channels'], *z_p.shape[-2:]),
    #                 device=self.device)
    #             input_augment_p = self.scale_augment[None, :, None, None] * input_augment + self.shift_augment[None,:,None,None]
    #             z_p = torch.cat([z_p,input_augment_p],dim=1)
    #             # add the mean to the means, which is zero for the augmented space
    #             z_m = torch.cat([z_m,self.shift_augment],dim=1)
    #
    #
    #         self.log_samples[key]["z_m"].append(z_m.detach().cpu().numpy())
    #         self.log_samples[key]["z_p"].append(z_p.detach().cpu().numpy())
    #         # from residual
    #         flow_input, cond = self.make_flow_input(batch, reverse=True)
    #         z_s = self.flow(flow_input, cond, reverse=True)
    #         if not torch.isnan(z_s).any():
    #             self.log_samples[key]["z_s"].append(z_s.detach().cpu().numpy())
    #             return 0
    #         else:
    #             self.console_logger.info("NaN encountered in umap samples.")
    #             return 1

    def training_epoch_end(self, outputs):
        self.log("epoch",self.current_epoch)

        # if self.current_epoch % 3 == 0:
        #     self.log_umap(train=True)


    def validation_step(self, batch, batch_id):

        out, logdet = self.forward_density(batch)
        out_hat, _ = self.second_stage_model.forward_density(batch)

        loss, loss_dict = self.loss_func(out, logdet)
        loss_recon = torch.nn.functionals.smooth_l1_loss(out, out_hat, reduction='mean')
        loss_dict['reconstruction loss'] = loss_recon.detach()
        loss += loss_recon * self.weight_recon
        loss_dict["flow_loss"] = loss

        self.log_dict({"val/" + key: loss_dict[key] for key in loss_dict}, logger=True, on_epoch=True)

        # X = batch['images']
        if batch_id <= int(self.config["logging"]["n_val_img_batches"]):
            n_samples = self.config["logging"]["n_samples"]
            n_logged_images = self.config["logging"]["n_log_images"]

            optical_flow = self.forward_sample(batch, n_samples, n_logged_images)
            tgt_imgs = batch['flow'][:n_logged_images]  # not target but ground truth
            optical_flow.insert(0, tgt_imgs)
            captions = ["ground truth"] + ["sample"] * n_samples
            img = fig_matrix(optical_flow, captions)

            self.logger.experiment.history._step = self.global_step
            self.logger.experiment.log({"Image Grid train set": wandb.Image(img,
                                                                            caption=f"Image Grid train @ it #{self.global_step}")}
                                       , step=self.global_step, commit=False)


        return {"loss":loss, "val-batch": batch, "batch_idx": batch_id, "loss_dict":loss_dict}



    def configure_optimizers(self):
        trainable_params = [{"params": self.INN.parameters(), "name": "flow"}, ]

        optim_type = Adam
        optimizer = optim_type(trainable_params, lr=self.config["training"]['lr'], betas=(0.9, 0.999),
                         weight_decay=self.config["training"]['weight_decay'], amsgrad=True)
        # optimizer = RMSprop(trainable_params, lr=self.config["training"]['lr'],
        #                  weight_decay=self.config["training"]['weight_decay'],alpha=0.9)
        if "gamma" not in self.config["training"] and not self.custom_lr_decrease:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            return [optimizer], [scheduler]
        elif not self.custom_lr_decrease:
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.config["training"]["gamma"],
                                                   last_epoch=self.current_epoch - 1)
            return [optimizer], [scheduler]

        else:
            return [optimizer]

class FlowEncoderFixed(BigAE):

    def __init__(self, config):
        super(FlowEncoderFixed, self).__init__(config)

    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        """ avoid pytorch lighting auto save params """
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination


class PokeMotionModelFCFixed(PokeMotionModelFC):

    def __init__(self,config,dirs):
        super(PokeMotionModelFCFixed, self).__init__(config,dirs)

    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        """ avoid pytorch lighting auto save params """
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

