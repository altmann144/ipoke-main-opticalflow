import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam, lr_scheduler
import wandb
from os import path
import yaml

from models.modules.autoencoders.baseline_fc_models import BaselineFCEncoder,BaselineFCGenerator
from models.modules.autoencoders.big_ae import BigAE
from models.modules.autoencoders.LPIPS import LPIPS as PerceptualLoss
from models.modules.INN.INN import UnsupervisedTransformer3
from models.modules.discriminators.disc_utils import calculate_adaptive_weight, adopt_weight, hinge_d_loss
from models.modules.discriminators.patchgan import define_D
# from utils.metrics import LPIPS
# from lpips import LPIPS as lpips_net
from utils.logging import batches2flow_grid
from collections import OrderedDict

class BigAEfixed(BigAE):

    def __init__(self, config, dirs):
        super(BigAE, self).__init__(config, dirs)
        checkpoint = torch.load(config["checkpoint_AE"], map_location='cpu')
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

    def setup(self, device: torch.device):
        self.freeze()

class FCAEINNModel(pl.LightningModule):

    def __init__(self,config):
        super().__init__()
        self.automatic_optimization = False
        config_ae_path = config['general']['config_ae']
        with open(config_ae_path, 'r') as stream:
            config_ae = yaml.safe_load(stream)
        self.config_ae = config_ae

        self.n_logged_imgs = self.config_ae["logging"]["n_log_images"]
        self.be_deterministic = self.config_ae["architecture"]["deterministic"]
        self.config_ae["architecture"]["in_size"] = self.config_ae["data"]["spatial_size"][0]

        # ae

        self.ae = BigAEfixed(self.config_ae["architecture"])

        # INN
        self.config["architecture"]["flow_in_channels"] = self.first_stage_config["architecture"]["z_dim"]
        self.INN = UnsupervisedTransformer3(**config['architecture'])

    def setup(self, stage: str):
        assert isinstance(self.logger, WandbLogger)
        self.logger.watch(self,log=None)

    # def forward_sample(self, batch, n_samples=1, n_logged_imgs=1):
    #     image_samples = []
    #
    #     with torch.no_grad():
    #
    #         for n in range(n_samples):
    #             flow_input, _, _ = self.VAE.encoder(batch['flow'])
    #             flow_input = torch.randn_like(flow_input).detach()
    #             out = self.INN(flow_input, reverse=True)
    #             out = self.VAE.decoder([out], del_shape=False)
    #             image_samples.append(out[:n_logged_imgs])
    #
    #     return image_samples

    def forward_sample(self, batch, n_samples=1, n_logged_imgs=1):
        image_samples = []

        with torch.no_grad():
            for n in range(n_samples):
                flow_input, _, _ = self.VAE.encoder(batch['flow'])
                flow_input = torch.randn_like(flow_input).detach()
                out = self.INN(flow_input, reverse=True)
                out = self.VAE.decoder([out], del_shape=False)
                image_samples.append(out[:n_logged_imgs])

        return image_samples

    def forward_density(self, batch):
        X = batch['flow']
        with torch.no_grad():
            encv, _, _ = self.VAE.encoder(X)
            # other = torch.randn_like(torch.cat((encv, encv, encv), 1))
            # rand = torch.zeros_like(encv)
            # encv = torch.cat((encv, other), 1)

        out, logdet = self.INN(encv, reverse=False)

        return out, logdet

    def configure_optimizers(self):
        trainable_params = [{"params": self.INN.parameters(), "name": "flow"}, ]

        optimizer = torch.optim.Adam(trainable_params, lr=self.config["training"]['lr'], betas=(0.9, 0.999),
                                     weight_decay=self.config["training"]['weight_decay'], amsgrad=True)

        return [optimizer]

    def training_step(self, batch, batch_id):

        # out_hat, _ = self.forward_density_video(batch)
        out, logdet = self.forward_density(batch)
        loss, loss_dict = self.loss_func(out, logdet)
        # loss_recon = F.smooth_l1_loss(out, out_hat, reduction='sum')
        # loss_dict['reconstruction loss'] = loss_recon.detach()
        # loss += loss_recon * self.weight_recon
        # loss_dict["flow_loss"] = loss

        self.log_dict(loss_dict, prog_bar=True, on_step=True, logger=False)
        self.log_dict({"train/" + key: loss_dict[key] for key in loss_dict}, logger=True, on_epoch=True,
                      on_step=True)
        self.log("global_step", self.global_step)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            self.INN.eval()
            n_samples = self.config["logging"]["n_samples"]
            n_logged_imgs = self.config["logging"]["n_log_images"]
            with torch.no_grad():
                # image_samples = self.forward_sample(batch,n_samples-1,n_logged_imgs)
                # tgt_imgs = batch['flow'][:n_logged_imgs]
                # image_samples.insert(0, tgt_imgs)
                #
                # enc, *_ = self.VAE.encoder(tgt_imgs)
                # rec = self.VAE.decoder([enc], del_shape=False)
                # image_samples.insert(1, rec)
                #
                # captions = ["target", "rec"] + ["sample"] * (n_samples - 1)
                # img = fig_matrix(image_samples, captions)
                optical_flow = self.forward_sample(batch, 2, 8)
                tgt_imgs = batch['flow'][:8]
                optical_flow.insert(0, tgt_imgs)
                captions = ["ground truth"] + ["sample"] * 2
                img = fig_matrix(optical_flow, captions)

            self.logger.experiment.history._step = self.global_step
            self.logger.experiment.log({"Image Grid train set": wandb.Image(img,
                                                                            caption=f"Image Grid train @ it #{self.global_step}")}
                                       , step=self.global_step, commit=False)


        return loss

    def validation_step(self, batch, batch_id):

        with torch.no_grad():
            out, logdet = self.forward_density(batch)
            loss, loss_dict = self.loss_func(out, logdet)

            if batch_id < self.config["logging"]["n_val_img_batches"]:
                optical_flow = self.forward_sample(batch, 2, 8)
                tgt_imgs = batch['flow'][:8]
                optical_flow.insert(0, tgt_imgs)
                captions = ["ground truth"] + ["sample"] * 2
                img = fig_matrix(optical_flow, captions)

                self.logger.experiment.log({"Image Grid val set": wandb.Image(img,
                                                                              caption=f"Image Grid val @ it #{self.global_step}")}
                                           , step=self.global_step, commit=False)

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

    def on_fit_start(self):
        self.ae.setup(self.device)

    def training_epoch_end(self, outputs):
        self.log("epoch",self.current_epoch)

def linear_var(
        act_it, start_it, end_it, start_val, end_val, clip_min, clip_max
):
    act_val = (
            float(end_val - start_val) / (end_it - start_it) * (act_it - start_it)
            + start_val
    )
    return np.clip(act_val, a_min=clip_min, a_max=clip_max)

# def create_dir_structure(config, model_name):
#     subdirs = ["ckpt", "config", "generated", "log"]
#
#     # model_name = config['model_name'] if model_name is None else model_name
#     structure = {subdir: os.path.join('wandb/logs_altmann', config["experiment"], subdir, model_name) for subdir in
#                  subdirs}
#     return structure