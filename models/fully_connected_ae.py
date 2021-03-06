import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Adam, lr_scheduler
import wandb
from os import path
import yaml

# from models.modules.autoencoders.baseline_fc_models import BaselineFCEncoder,BaselineFCGenerator
from models.modules.autoencoders.big_ae import BigAE
from models.modules.autoencoders.LPIPS import LPIPS as PerceptualLoss
from models.modules.discriminators.disc_utils import calculate_adaptive_weight, adopt_weight, hinge_d_loss
# from models.modules.discriminators.disc_utils import MinibatchDiscrimination

from models.modules.discriminators.patchgan import define_D
from utils.metrics import LPIPS, SSIM_custom, PSNR_custom
from lpips import LPIPS as lpips_net
from utils.logging import batches2flow_grid, batches2image_grid

class FCAEModel(pl.LightningModule):

    def __init__(self,config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        kl_weight = config["training"]["w_kl"]
        self.register_buffer("disc_factor",torch.tensor(1.),persistent=True)
        self.register_buffer("previous_disc_loss",torch.tensor(1.),persistent=True)
        self.register_buffer("disc_weight",torch.tensor(1.),persistent=True)
        self.register_buffer("perc_weight",torch.tensor(1.),persistent=True)
        self.register_buffer("kl_weight", torch.tensor(kl_weight), persistent=True)
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)
        self.disc_start = self.config["training"]['pretrain']
        self.n_logged_imgs = self.config["logging"]["n_log_images"]
        self.config["architecture"]["in_size"] = self.config["data"]["spatial_size"][0]
        self.vgg_loss = PerceptualLoss()

        # ae
        self.ae = BigAE(self.config["architecture"])
        self.config['architecture']['spectral_norm'] = True


        self.be_deterministic = self.config["architecture"]["deterministic"]

        # discriminator
        self.discriminator = define_D(config['architecture']['n_out_channels'], 64, netD='basic')
        # self.minibatch_disc = MinibatchDiscrimination(self.config["architecture"]["in_size"]*self.config["architecture"]["in_size"]*config['architecture']['n_out_channels'],
        #                                               self.config["architecture"]["in_size"]*self.config["architecture"]["in_size"], 2, mean=True)

        # metrics
        self.lpips_net = lpips_net()
        for param in self.lpips_net.parameters():
            param.requires_grad = False

        self.lpips_metric = LPIPS()

        if config['architecture']['n_out_channels'] == 2:
            self.key = 'flow'
            self.batches2visual_grid = batches2flow_grid
        else:
            self.key = 'images'
            self.batches2visual_grid = batches2image_grid
    def setup(self, stage: str):
        assert isinstance(self.logger, WandbLogger)
        self.logger.watch(self,log=None)


    def forward(self,x):
        img, p_mode, p = self.ae(x)
        return img, p_mode, p

    def discriminate(self, img):
        # img = self.minibatch_disc(img)
        # img = img.view(-1, self.config['architecture']['n_out_channels'] + 1, self.config["architecture"]["in_size"], self.config["architecture"]["in_size"])
        x = self.discriminator(img)
        return x

    def training_step(self,batch, batch_idx,optimizer_idx):
        x = batch[self.key]
        if self.key == 'images':
            x = x[:, 0] # dataloader returns stacked img.append(img)

        (opt_g, opt_d) = self.optimizers()

        rec, p_mode, p = self(x)
        rec_loss = torch.mean(torch.abs(x.contiguous() - rec.contiguous()))

        # equal weighting of l1 and perceptual loss
        p_loss = torch.mean(self.vgg_loss(x.contiguous(), rec.contiguous()))

#         nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        kl_loss = p.kl().mean()

        # generator update
        logits_fake = self.discriminate(rec)
        g_loss = -torch.mean(logits_fake)

        d_weight = calculate_adaptive_weight( rec_loss + p_loss*self.perc_weight, g_loss, self.disc_weight,
                                             last_layer=list(self.ae.decoder.parameters())[-1])
        d_weight *= nn.functional.relu_(1 - nn.functional.relu_(self.previous_disc_loss))

        disc_factor = adopt_weight(self.disc_factor, self.current_epoch, threshold=self.disc_start)
        # loss = nll_loss  + d_weight * disc_factor * g_loss + kl_loss
        loss = rec_loss + p_loss*self.perc_weight + d_weight*disc_factor*g_loss + kl_loss*self.kl_weight


        opt_g.zero_grad()
        self.manual_backward(loss,opt_g)
        opt_g.step()
        for i in range(1):
            logits_real = self.discriminate(x.contiguous().detach())
            logits_fake = self.discriminate(rec.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, self.current_epoch, threshold=self.disc_start)
            d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)
            self.previous_disc_loss = d_loss.detach()
            self.previous_disc_loss.requires_grad = False

            if d_loss.item() <= 0:
                break
            opt_d.zero_grad()
            self.manual_backward(d_loss,opt_d)
            opt_d.step()

        loss_dict = {"train/loss": loss, "train/d_loss":d_loss,
                     "train/rec_loss": rec_loss,"train/d_weight":d_weight, "train/disc_factor": disc_factor,
                     "train/g_loss": g_loss, "train/logits_real": logits_real.mean(), "train/logits_fake": logits_fake.mean(),
                     "train/kl_loss": kl_loss,}

        self.log_dict(loss_dict,logger=True,on_epoch=True,on_step=True)
        self.log("global step", self.global_step)
        self.log("learning rate",opt_g.param_groups[0]["lr"],on_step=True, logger=True)

        self.log("overall_loss",loss,prog_bar=True,logger=False)
        self.log("rec_loss",rec_loss,prog_bar=True,logger=False)
        self.log("d_loss",d_loss,prog_bar=True,logger=False)
        self.log("kl_loss",kl_loss,prog_bar=True,logger=False)
        self.log("g_loss",g_loss,prog_bar=True,logger=False)

        loss_dict.update({"img_real-train": x, "img_fake-train": rec})

        return loss_dict, batch_idx

    def training_step_end(self, outputs):

        # for convenience, in case ditributed training is used
        loss_dict = outputs[0]
        x  =loss_dict["img_real-train"]
        rec = loss_dict["img_fake-train"]
        #

        if self.global_step % self.config["logging"]["log_train_prog_at"] == 0:
            imgs = [x[:self.n_logged_imgs].detach(),rec[:self.n_logged_imgs].detach()]
            captions = ["Targets", "Predictions"]
            train_grid = self.batches2visual_grid(imgs, captions)
            self.logger.experiment.log({f"Train Batch": wandb.Image(train_grid,
                                                                    caption=f"Training Images @ it #{self.global_step}")},step=self.global_step, commit=False)

    def training_epoch_end(self, outputs):
        self.log("epoch",self.current_epoch)

    def validation_step(self, batch, batch_id):
        x = batch[self.key]
        if self.key == 'images':
            x = x[:, 0] # dataloader returns stacked img.append(img)

        (opt_g, opt_d) = self.optimizers()

        rec, p_mode, p = self(x)
        rec_loss = torch.mean(torch.abs(x.contiguous() - rec.contiguous()))

        # equal weighting of l1 and perceptual loss
        p_loss = torch.mean(self.vgg_loss(x.contiguous(), rec.contiguous()))

#         nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        kl_loss = p.kl().mean()

        loss = rec_loss + p_loss*self.perc_weight + kl_loss*self.kl_weight

        log_dict = {"val/loss": loss,
                    "val/rec_loss": rec_loss,
                    "val/kl_loss": kl_loss}

        if self.key == 'flow':
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 1)).contiguous()
            rec = torch.nn.functional.pad(rec, (0, 0, 0, 0, 0, 1)).contiguous()

        self.log("lpips-val", self.lpips_metric(self.lpips_net, rec, x).cpu(), on_step=False, on_epoch=True, logger=True)

        self.log_dict(log_dict, logger=True, prog_bar=False,on_epoch=True)

        log_dict.update({"img_real-val": x, "img_fake-val": rec})

        return log_dict, batch_id

    def validation_step_end(self,val_out):
        log_dict = val_out[0]
        batch_id = val_out[1]
        x = log_dict["img_real-val"]
        rec = log_dict["img_fake-val"]

        # log train metrics
        # with torch.no_grad():
        #     self.log("ssim-val", self.ssim(rec, x).cpu().item(), on_step=False, on_epoch=True)
        #     self.log("psnr-val", self.psnr(rec, x).cpu().item(), on_step=False, on_epoch=True)
        #     self.log("val/lpips", lpips, on_step=False, on_epoch=True)
        #     self.log("lpips-val", self.lpips_metric(self.lpips_net,rec,x).cpu(), on_step=False, on_epoch=True,logger=True)

        if batch_id < self.config["logging"]["n_val_img_batches"]:
            imgs = [x[:self.n_logged_imgs].detach(),rec[:self.n_logged_imgs].detach()]
            captions = ["Targets", "Predictions"]
            val_grid = self.batches2visual_grid(imgs,captions)
            self.logger.experiment.log({f"Validation Batch #{batch_id}" : wandb.Image(val_grid,
                                                                                      caption=f"Validation Images @ it {self.global_step}")},
                                       step=self.global_step,
                                       commit=False)

    def configure_optimizers(self):
        # optimizers
        ae_params = [{"params": self.ae.parameters(), "name": "BigGAN"}
                     #{"params": self.logvar, "name": "logvar"}
                     ]
        lr = self.config["training"]["lr"]

        opt_g = Adam(ae_params, lr = lr, weight_decay=self.config["training"]["weight_decay"])
        # params = list(self.discriminator.parameters())+ list(self.minibatch_disc.parameters())
        disc_params = list(self.discriminator.parameters())
        opt_d = Adam(disc_params,lr=self.config["training"]["lr"], weight_decay=self.config["training"]["weight_decay"])


        # schedulers
        sched_g = lr_scheduler.ReduceLROnPlateau(opt_g,mode="min",factor=.5,patience=0,min_lr=1e-8,
                                                             threshold=0.001, threshold_mode='rel')
        sched_d = lr_scheduler.ReduceLROnPlateau(opt_g, mode="min", factor=.1, patience=0, min_lr=1e-8,
                                                 threshold=0.001, threshold_mode='rel')

        return [opt_g,opt_d], [{'scheduler':sched_g,'monitor':"val/loss","interval":1,'reduce_on_plateau':True,'strict':True},
                               {'scheduler':sched_d,'monitor':"val/loss","interval":1,'reduce_on_plateau':True,'strict':True}]
        # return ({'optimizer': opt_g,'lr_scheduler':sched_g,'monitor':"loss-val","interval":1,'reduce_on_plateau':True,'strict':True},
                # {'optimizer': opt_d,'lr_scheduler':sched_d,'monitor':"loss-val","interval":1,'reduce_on_plateau':True,'strict':True})
