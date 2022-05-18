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
from models.modules.autoencoders.baseline_fc_models import FirstStageFCWrapper
from models.modules.INN.INN import SupervisedTransformer
from models.modules.INN.loss import FlowLoss
# from models.modules.autoencoders.util import Conv2dTransposeBlock
# from models.modules.INN.coupling_flow_alternative import AdaBelief
# from utils.logging import make_flow_video_with_samples, log_umap, make_samples_and_samplegrid, save_video, make_transfer_grids_new, make_multipoke_grid
from utils.general import linear_var, get_logger
# from utils.metrics import FVD, calculate_FVD, LPIPS,PSNR_custom,SSIM_custom, KPSMetric, metric_vgg16, compute_div_score,SampleLPIPS, SampleSSIM, compute_div_score_mse, compute_div_score_lpips
from utils.metrics import LPIPS,PSNR_custom,SSIM_custom,optical_flow_metric
# from utils.posenet_wrapper import PoseNetWrapper
# from models.pose_estimator.tools.infer import save_batch_image_with_joints
import matplotlib.pyplot as plt

class ThirdStageFlowFCConditional(pl.LightningModule):

    def __init__(self,config,dirs):
        super().__init__()
        #self.automatic_optimization=False
        self.config = config
        self.embed_poke = True
        self.dirs = dirs

        # normal or radial_gaussian distribution
        if 'base_distribution' in self.config['architecture']:
            self.base_distribution = self.config['architecture']['base_distribution']
        else:
            self.base_distribution = 'normal'

        # check if reconstuction weight increase is needed
        self.weight_recon = config['training']['weight_recon']
        self.apply_recon_scaling = "recon_scaling" in self.config["training"] and self.config["training"]["recon_scaling"]

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

        # first stage and embedder models for poke and image are part of second stage
        self.__initialize_second_stage()
        self.first_stage_config = self.second_stage_model.first_stage_config # to make 4D flow sample input
        self.config["architecture"]["flow_in_channels"] = self.first_stage_config["architecture"]["z_dim"]

        # self.metrics_dir = path.join(self.dirs['generated'],'metrics')
        # os.makedirs(self.metrics_dir,exist_ok=True)


        # self.FVD = FVD(n_samples=self.config['logging']['n_fvd_samples'] if 'n_fvd_samples' in  self.config['logging'] else 1000)
        # if self.test_mode == 'none' or self.test_mode=='accuracy':
        #     self.lpips_metric = LPIPS()
        #     self.ssim = SSIM_custom()
        #     self.psnr = PSNR_custom()
        #     self.lpips_net = lpips_net()


        self.__initialize_flow_encoder()

        self.config["architecture"]["flow_embedding_channels"] = self.second_stage_model.poke_emb_config["architecture"]["nf_max"]
        self.config["architecture"]["flow_embedding_channels"] = 64
        self.config["architecture"]["flow_mid_channels"] = int(config["architecture"]["flow_mid_channels_factor"] * \
                                                               self.config["architecture"]["flow_in_channels"])

        self.INN = SupervisedTransformer(config['architecture'])

        self.loss_func = FlowLoss(spatial_mean=False, logdet_weight=1., radial=self.base_distribution=='radial')

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
    def __initialize_poke_embedder(self):
        dic = poke_embedder_models[self.config['poke_embedder']['name']]
        # model_name = dic['model_name']
        emb_ckpt = dic['ckpt']

        # emb_config = path.join(self.config["general"]["base_dir"], "poke_encoder", "config", model_name, "config.yaml")
        emb_config = path.join(*emb_ckpt.split('/')[:-2],'config.yaml').replace('ckpt','config')

        with open(emb_config) as f:
            self.poke_emb_config = yaml.load(f, Loader=yaml.FullLoader)
        self.poke_embedder = FirstStageFCWrapperFixed(self.poke_emb_config)
        assert self.poke_embedder.be_deterministic
        state_dict = torch.load(emb_ckpt, map_location="cpu")
        # remove keys from checkpoint which are not required
        state_dict = {key[6:]: state_dict["state_dict"][key] for key in state_dict["state_dict"] if
                      key[:5] == 'model'}
        # load first stage model
        m, u = self.poke_embedder.load_state_dict(state_dict, strict=False)
        assert len(m) == 0, f'poke_embedder is missing keys {m}'
        del state_dict

    def __initialize_flow_encoder(self):
        dic = flow_encoder_models[self.config['flow_encoder']['name']]
        flow_enc_ckpt = dic['ckpt']

        flow_enc_config = path.join(*flow_enc_ckpt.split('/')[:-2],'config.yaml').replace('ckpt','config')

        with open(flow_enc_config) as f:
            self.flow_enc_config = yaml.load(f, Loader=yaml.FullLoader)
        self.flow_enc_config["architecture"]["in_size"] = self.flow_enc_config["data"]["spatial_size"][0]

        self.flow_encoder = FlowEncoderFixed(self.flow_enc_config["architecture"])

        state_dict = torch.load(flow_enc_ckpt, map_location="cpu")
        # remove keys from checkpoint which are not required
        state_dict = {key[3:]: state_dict["state_dict"][key] for key in state_dict["state_dict"] if
                      key[:2] == 'ae'}
        # load first stage model
        m, u = self.flow_encoder.load_state_dict(state_dict, strict=False)
        assert len(m) == 0, f'flow_embedder is missing keys {m}'
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

            if self.base_distribution == 'radial':
                flow_input = torch.nn.functional.normalize(flow_input.view(shape[0], -1))
                flow_input = flow_input.T * torch.abs(torch.randn(shape[0]).type_as(X))
                flow_input = flow_input.T.view(shape).detach()
        else:
            with torch.no_grad():
                enc_vec = self.flow_encoder.encode(X).sample()
                motion_rest_shape = list(enc_vec.shape)
                motion_rest_shape[1] = self.config['architecture']['flow_in_channels'] - self.flow_enc_config['architecture']['z_dim']
                assert motion_rest_shape[1] >= 0, f'{self.config["architecture"]["flow_in_channels"]} < {self.flow_enc_config["architecture"]["z_dim"]}'
                if motion_rest_shape[1] != 0:
                    motion_rest = torch.randn(motion_rest_shape, dtype=enc_vec.dtype, layout=enc_vec.layout, device=enc_vec.device)
                    flow_input = torch.cat((enc_vec, motion_rest), 1)
                else:
                    flow_input = enc_vec

        return flow_input

    def on_train_epoch_start(self):
        plt.close('all')
        if self.custom_lr_decrease:
            n_train_iterations = self.config['training']['n_epochs'] * self.trainer.num_training_batches
            self.lr_adaptation = partial(self.lr_adaptation, end_it=n_train_iterations)
        # reconsturcion weight sweep

        if self.apply_recon_scaling:
            if self.current_epoch % 10 == 9:
                self.weight_recon = self.weight_recon * 2

    def forward_sample(self, batch, cond, n_samples=1, n_logged_imgs=1, flow_input=None):
        image_samples = []
        opt_flow_dim = self.flow_enc_config['architecture']['z_dim']
        with torch.no_grad():
            for n in range(n_samples):
                if flow_input is None:
                    flow_input = self.make_flow_input(batch, reverse=True)
                else:
                    assert n_samples==1, 'second pass will random sample and overwrite given input'
                out = self.INN(flow_input[:n_logged_imgs],cond[:n_logged_imgs], reverse=True)
                flow_input = None

                out, residual = out[:,:opt_flow_dim], out[:,opt_flow_dim:]
                if len(out.shape) == 4:
                    out = out.squeeze(-1).squeeze(-1)
                out = self.flow_encoder.decode(out)
                image_samples.append(out)

        return image_samples, residual

    def forward_density(self, batch, cond):
        flow_input = self.make_flow_input(batch, reverse=False)
        optical_flow_vector = flow_input.clone().detach() # hopefully triggers correctnesschecks if flow_input is changed inplace in INN
        out, logdet = self.INN(flow_input.detach(), cond, reverse=False)
        return out, logdet, optical_flow_vector

    def training_step(self, batch, batch_idx):

        with torch.no_grad():
            out_hat, _, cond  = self.second_stage_model.forward_density(batch, return_cond=True)
        # alternatively self.config["architecture"]["flow_embedding_channels"] for poke_embedder z_dim
        poke_encoding = cond[:,-self.second_stage_model.poke_embedder.config['architecture']['z_dim']:]
        out, logdet, optical_flow_vec = self.forward_density(batch, poke_encoding.detach())

        loss, loss_dict = self.loss_func(out, logdet)
        loss_recon = torch.nn.functional.mse_loss(out, out_hat, reduction='mean')
        loss_dict['reconstruction loss'] = loss_recon.detach()
        loss += loss_recon * self.weight_recon
        loss_dict["flow_loss"] = loss

        self.log_dict(loss_dict, prog_bar=True, on_step=True, logger=False)
        self.log_dict({"train/" + key: loss_dict[key] for key in loss_dict}, logger=True, on_epoch=True, on_step=True)
        self.log("global_step", self.global_step, on_epoch=False, logger=True, on_step=True)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)


        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            with torch.no_grad():
                n_samples = self.config["logging"]["n_samples"]
                n_logged_images = self.config["logging"]["n_log_images"]

                optical_flow_imgs, latent_residual = self.forward_sample(batch, poke_encoding.detach(), n_samples, n_logged_images)
                optical_flow_hat, _ = self.forward_sample(None, poke_encoding.detach(), 1, 64, out_hat) # take 64 logged images for optical flow metric
                optical_flow_imgs.insert(0, optical_flow_hat[0]) # reconstructed extracted optical flow
                tgt_imgs = batch['flow'][:n_logged_images]
                optical_flow_imgs.insert(0, tgt_imgs)
                captions = ["ground truth"] + ["extracted"] + ["sample"] * n_samples

                ############################################################################################################
                if isinstance(batch['poke'], list):
                    poke = batch["poke"][0]
                else:
                    poke = batch['poke']

                ### optical flow reconstruction image
                if len(poke_encoding.shape) == 4:
                    poke_encoding = poke_encoding.squeeze(-1).squeeze(-1)
                poke_enc_reconstruction = self.second_stage_model.poke_embedder.decoder([poke_encoding], None)
                captions += ["poke_recon"]
                optical_flow_imgs.append(poke_enc_reconstruction[:n_logged_images])

                ### poke image
                captions += ["poke"]
                optical_flow_imgs += [poke[:n_logged_images]]

                #### optical flow metric
                if len(optical_flow_vec.shape) == 4:
                    optical_flow_vec = optical_flow_vec.squeeze(-1).squeeze(-1)
                optical_flow = self.flow_encoder.decode(optical_flow_vec[:,:self.flow_enc_config['architecture']['z_dim']])
                # unpack optical_flow_hat since it is a list of batches
                optical_flow_errors = optical_flow_metric(optical_flow_hat[0][:64], optical_flow[:64])
                optical_flow_errors_poke_recon = optical_flow_errors = optical_flow_metric(poke_enc_reconstruction[:64], optical_flow[:64])

                self.log_dict({"train/" + key: optical_flow_errors[key] for key in optical_flow_errors}, logger=True, on_epoch=True, on_step=True)
                self.log_dict({"train/" + key + "poke_recon": optical_flow_errors_poke_recon[key] for key in optical_flow_errors_poke_recon}, logger=True, on_epoch=True, on_step=True)

                ############################################################################################################
                img = fig_matrix(optical_flow_imgs, captions)
                self.logger.experiment.log({"Image Grid train set": wandb.Image(img,
                                                                                caption=f"Image Grid train @ it #{self.global_step}")}
                                           , step=self.global_step, commit=False)
                self.log("train/latent_residual_mean", torch.mean(latent_residual), on_step=False, on_epoch=True, logger=True)
                self.log("train/latent_residual_std", torch.std(latent_residual), on_step=False, on_epoch=True, logger=True)

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
        out_hat, _, cond = self.second_stage_model.forward_density(batch, return_cond=True)
        # TODO only works for poke and image conditioning in second stage for now
        poke_encoding = cond[:,-self.second_stage_model.poke_embedder.config['architecture']['z_dim']:]

        out, logdet, optical_flow_vec = self.forward_density(batch, poke_encoding.detach())

        loss, loss_dict = self.loss_func(out, logdet)
        loss_recon = torch.nn.functional.mse_loss(out, out_hat, reduction='mean')
        loss_dict['reconstruction loss'] = loss_recon.detach()
        loss += loss_recon * self.weight_recon
        loss_dict["flow_loss"] = loss

        self.log_dict({"val/" + key: loss_dict[key] for key in loss_dict}, logger=True, on_epoch=True)

        # X = batch['images']
        if batch_id <= int(self.config["logging"]["n_val_img_batches"]):
            n_samples = self.config["logging"]["n_samples"]
            n_logged_images = self.config["logging"]["n_log_images"]

            optical_flow_imgs, latent_residual = self.forward_sample(batch, poke_encoding.detach(), n_samples, n_logged_images)
            optical_flow_hat, _ = self.forward_sample(None, poke_encoding.detach(), 1, 64, out_hat) # take 64 logged images for optical flow metric
            optical_flow_imgs.insert(0, optical_flow_hat[0][:n_logged_images])
            tgt_imgs = batch['flow'][:n_logged_images]
            optical_flow_imgs.insert(0, tgt_imgs)
            captions = ["ground truth"] + ["extracted"] + ["sample"] * n_samples
            ############################################################################################################
            if isinstance(batch['poke'], list):
                poke = batch["poke"][0]
            else:
                poke = batch['poke']

            ### optical flow poke encoder model
            if len(poke_encoding.shape) == 4:
                poke_encoding = poke_encoding.squeeze(-1).squeeze(-1)
            poke_enc_reconstruction = self.second_stage_model.poke_embedder.decoder(
                [poke_encoding], None)
            captions += ["poke_recon"]
            optical_flow_imgs.append(poke_enc_reconstruction[:n_logged_images])

            ### poke image
            captions += ["poke"]
            optical_flow_imgs += [poke[:n_logged_images]]
            ### optical flow metric
            if len(optical_flow_vec.shape) == 4:
                optical_flow_vec = optical_flow_vec.squeeze(-1).squeeze(-1)
            optical_flow = self.flow_encoder.decode(optical_flow_vec[:,:self.flow_enc_config['architecture']['z_dim']])
            # unpack optical_flow_hat since it is a list of batches
            optical_flow_errors = optical_flow_metric(optical_flow_hat[0][:64], optical_flow[:64])
            # optical_flow_errors_poke_recon = optical_flow_metric(poke_enc_reconstruction[:64], optical_flow[:64])
            self.log_dict({"val/" + key: optical_flow_errors[key] for key in optical_flow_errors}, logger=True, on_epoch=True, on_step=False)
            self.log("val-EE_R3", optical_flow_errors['endpoint_error']['3.0']) # for checkpoint saving
            ############################################################################################################
            img = fig_matrix(optical_flow_imgs, captions)
            # self.logger.experiment.history._step = self.global_step
            self.logger.experiment.log({"Image Grid val set": wandb.Image(img,
                                                                            caption=f"Image Grid val @ it #{self.global_step}")}
                                       , step=self.global_step, commit=False)
            self.log("val/latent_residual_mean", torch.mean(latent_residual), on_step=False, on_epoch=True, logger=True)
            self.log("val/latent_residual_std", torch.std(latent_residual), on_step=False, on_epoch=True, logger=True)

        self.log('val-loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('weight_recon', self.weight_recon, on_epoch=True, on_step=False, logger=True)
        return {"loss":loss, "val-batch": batch, "batch_idx": batch_id, "loss_dict":loss_dict}

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""

    def test_step(self, batch, batch_id):
        self.eval()
        n_logged_images = self.config["logging"]["n_log_images"]
        with torch.no_grad():
            out_hat, _, cond = self.second_stage_model.forward_density(batch, return_cond=True)
            poke_encoding = cond[:, -self.second_stage_model.poke_embedder.config['architecture']['z_dim']:]

            out, logdet, optical_flow_vec = self.forward_density(batch, poke_encoding.detach())

            if self.test_mode == 'accuracy':
                # n_samples = self.config["logging"]["n_samples"]
                # n_logged_images = self.config["logging"]["n_log_images"]

                optical_flow_imgs, latent_residual = self.forward_sample(batch, poke_encoding.detach(), n_samples,
                                                                         n_logged_images)
                optical_flow_hat, _ = self.forward_sample(None, poke_encoding.detach(), 1, 64,
                                                          out_hat)  # take 64 logged images for optical flow metric
                optical_flow_imgs.insert(0, optical_flow_hat[0][:n_logged_images])
                tgt_imgs = batch['flow'][:n_logged_images]
                optical_flow_imgs.insert(0, tgt_imgs)
                captions = ["ground truth"] + ["extracted"] + ["sample"] * n_samples
                ############################################################################################################
                if isinstance(batch['poke'], list):
                    poke = batch["poke"][0]
                else:
                    poke = batch['poke']

                ### optical flow poke encoder model
                if len(poke_encoding.shape) == 4:
                    poke_encoding = poke_encoding.squeeze(-1).squeeze(-1)
                poke_enc_reconstruction = self.second_stage_model.poke_embedder.decoder(
                    [poke_encoding], None)
                captions += ["poke_recon"]
                optical_flow_imgs.append(poke_enc_reconstruction[:n_logged_images])

                ### poke image
                captions += ["poke"]
                optical_flow_imgs += [poke[:n_logged_images]]
                ### optical flow metric
                if len(optical_flow_vec.shape) == 4:
                    optical_flow_vec = optical_flow_vec.squeeze(-1).squeeze(-1)
                optical_flow = self.flow_encoder.decode(
                    optical_flow_vec[:, :self.flow_enc_config['architecture']['z_dim']])
                # unpack optical_flow_hat since it is a list of batches
                optical_flow_errors = optical_flow_metric(optical_flow_hat[0][:], optical_flow[:])
                ############################################################################################################

                plt.close('all')
                return optical_flow_errors, optical_flow_imgs
            if self.test_mode == '':
                pass
            else:
                raise ValueError(f'No such test {self.test_mode}')
    def test_epoch_end(self, outputs):
        n_logged_images = self.config["logging"]["n_log_images"]
        self.print(f'******************* TEST SUMMARY on {self.trainer.datamodule.dset_val.__class__.__name__} FOR {n_logged_images} SAMPLES *******************')
        if self.test_mode == "accuracy":
            errs = [o[0] for o in outputs]
            exmpls = np.concatenate([o[1] for o in outputs])

            for error_type in errs[0].keys():
                print(f"{error_type}")
                for key in errs[0][error_type].keys():
                    result = 0.
                    for error in errs:
                        result += error[error_type][key]
                    print(f"    R{key} = {result / len(errs)}")

            n_pokes = self.trainer.datamodule.dset_val.config['n_pokes']

            exmpls = exmpls.cpu().numpy()
            savepath = path.join(self.dirs['generated'], 'diversity')
            makedirs(savepath, exist_ok=True)

            np.save(path.join(savepath, f'samples_diversity_{n_pokes}_pokes.npy'), exmpls)

        # if self.test_mode == 'fvd':
        #
        #     savedir_vid_samples = path.join(self.dirs['generated'],'fvd_vid_examples')
        #
        #     makedirs(self.savedir_fvd, exist_ok=True)
        #     makedirs(savedir_vid_samples, exist_ok=True)
        #
        #     real_samples = np.stack(self.first_stage_model.fvd_features_true_x0, axis=0)
        #     fake_samples = np.stack(self.first_stage_model.fvd_features_fake_x0, axis=0)
        #
        #     self.console_logger.info(f"Generating example videos")
        #     for i, (r, f) in enumerate(zip(real_samples, fake_samples)):
        #         savename = path.join(savedir_vid_samples, f"sample{i}.mp4")
        #         r = np.concatenate([v for v in r], axis=2)
        #         f = np.concatenate([v for v in f], axis=2)
        #         all = np.concatenate([r, f], axis=1)
        #
        #         save_video(all, savename)
        #
        #         if i >= 4:
        #             break
        #
        #     self.console_logger.info(f"Saving samples to {self.savedir_fvd}")
        #     np.save(path.join(self.savedir_fvd, "real_samples.npy"), real_samples)
        #     np.save(path.join(self.savedir_fvd, "fake_samples.npy"), fake_samples)
        #
        #     self.console_logger.info(f'Finish generation of vid samples.')

    def on_test_end(self) -> None:
        """Called at the end of testing."""

    def configure_optimizers(self):
        trainable_params = [{"params": self.INN.parameters(), "name": "flow"}, ]

        optim_type = Adam
        optimizer = optim_type(trainable_params, lr=self.config["training"]['lr'], betas=(0.9, 0.999),
                         weight_decay=self.config["training"]['weight_decay'], amsgrad=True)
        # optimizer = RMSprop(trainable_params, lr=self.config["training"]['lr'],
        #                  weight_decay=self.config["training"]['weight_decay'],alpha=0.9)
        if "gamma" not in self.config["training"] and not self.custom_lr_decrease:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
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
        self.eval()

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
        config["general"]["test"] = 'none'
        config["general"]["third_stage_training"] = True
        super(PokeMotionModelFCFixed, self).__init__(config,dirs)
        self.eval()

    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        """ avoid pytorch lighting auto save params """
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

class FirstStageFCWrapperFixed(FirstStageFCWrapper):

    def __init__(self, config):
        super(FirstStageFCWrapperFixed, self).__init__(config)
        self.eval()

    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        """ avoid pytorch lighting auto save params """
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination