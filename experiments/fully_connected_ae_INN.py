import torch
import numpy as np
import pytorch_lightning as pl
import wandb
import yaml
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from functools import partial
from collections import OrderedDict

from models.modules.autoencoders.big_ae import BigAE
from models.modules.INN.INN import UnsupervisedTransformer3
from models.modules.INN.loss import FlowLoss
from utils.evaluation import fig_matrix


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


class FCAEINNModel(pl.LightningModule):

    def __init__(self,config):
        super().__init__()
        self.config = config
        # self.automatic_optimization = False
        config_ae_path = config['general']['config_ae']
        with open(config_ae_path, 'r') as stream:
            config_ae = yaml.safe_load(stream)
        self.config_ae = config_ae
        self.config_ae['architecture']['checkpoint_ae'] = self.config_ae['general']['checkpoint_ae']

        self.n_logged_imgs = self.config_ae["logging"]["n_log_images"]
        self.be_deterministic = self.config_ae["architecture"]["deterministic"]
        self.config_ae["architecture"]["in_size"] = self.config_ae["data"]["spatial_size"][0]

        # lr warmup
        lr = config["training"]["lr"]
        self.apply_lr_scaling = "lr_scaling" in self.config["training"] and self.config["training"]["lr_scaling"]
        if self.apply_lr_scaling:
            end_it = self.config["training"]["lr_scaling_max_it"]

            self.lr_scaling = partial(linear_var, start_it=0, end_it=end_it, start_val=0., end_val=lr, clip_min=0.,
                                      clip_max=lr)
        # configure custom lr decrease
        self.custom_lr_decrease = self.config['training']['custom_lr_decrease']
        if self.custom_lr_decrease:
            start_it = 500  # 1000
            self.lr_adaptation = partial(linear_var, start_it=start_it, end_it=50000, start_val=lr, end_val=1e-4,
                                         clip_min=1e-4,
                                         clip_max=1)


        # ae

        self.ae = BigAEfixed(self.config_ae["architecture"])

        # INN
        self.config["architecture"]["flow_in_channels"] = self.config_ae["architecture"]["z_dim"]
        self.INN = UnsupervisedTransformer3(**config['architecture'])

        # loss
        self.loss_func = FlowLoss()

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
                flow_input = self.ae.encode(batch['flow']).mode()
                flow_input = torch.randn_like(flow_input).detach()
                out = self.INN(flow_input, reverse=True)
                out = self.ae.decode(out)
                image_samples.append(out[:n_logged_imgs])

        return image_samples

    def forward_density(self, batch):
        X = batch['flow']
        with torch.no_grad():
            encv = self.ae.encode(X).sample()
            # other = torch.randn_like(torch.cat((encv, encv, encv), 1))
            # rand = torch.zeros_like(encv)
            # encv = torch.cat((encv, other), 1)

        out, logdet = self.INN(encv, reverse=False)

        return out, logdet

    def configure_optimizers(self):
        # trainable_params = [{"params": self.INN.parameters(), "name": "INN"}, ]

        optimizer = torch.optim.Adam(self.INN.parameters(), lr=self.config["training"]['lr'], betas=(0.9, 0.999),
                                     weight_decay=self.config["training"]['weight_decay'], amsgrad=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        return optimizer

    def training_step(self, batch, batch_id):

        out, logdet = self.forward_density(batch)
        loss, loss_dict = self.loss_func(out, logdet)

        self.log_dict(loss_dict, prog_bar=True, on_step=True, logger=False)
        self.log_dict({"train/" + key: loss_dict[key] for key in loss_dict}, logger=True, on_epoch=True,
                      on_step=True)
        self.log("global_step", self.global_step)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            self.INN.eval()
            with torch.no_grad():
                optical_flow = self.forward_sample(batch, 2, 8)
                tgt_imgs = batch['flow'][:8]
                optical_flow.insert(0, tgt_imgs)
                captions = ["ground truth"] + ["sample"] * 2
                img = fig_matrix(optical_flow, captions)
                img = wandb.Image(img, caption=f"Image Grid train @ it #{self.global_step}")

            self.logger.experiment.history._step = self.global_step
            self.logger.experiment.log({"Image Grid train set": img}, step=self.global_step, commit=False)
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
                img = wandb.Image(img, caption=f"Image Grid train @ it #{self.global_step}")
                self.logger.experiment.log({"Image Grid val set": img}, step=self.global_step, commit=False)

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

    def training_epoch_end(self, outputs):
        self.log("epoch", self.current_epoch)
        plt.close('all')

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