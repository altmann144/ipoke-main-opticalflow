import torch
import torch.nn as nn
from torch.distributions import Normal
from einops import rearrange
import numpy as np
from functools import partial

class FlowLoss(nn.Module):
    def __init__(self,spatial_mean=False, logdet_weight=1., nll_weight=1., radial=False):
        super().__init__()
        # self.config = config
        self.spatial_mean = spatial_mean
        self.logdet_weight = logdet_weight
        self.nll_weight = nll_weight
        self.radial = radial

    def forward(self, sample, logdet):
        nll_loss = torch.mean(nll(sample, spatial_mean=self.spatial_mean, radial=self.radial))
        assert len(logdet.shape) == 1
        if self.spatial_mean:
            h,w = sample.shape[-2:]
            nlogdet_loss = -torch.mean(logdet) / (h*w)
        else:
            nlogdet_loss = -torch.mean(logdet)

        loss = self.nll_weight*nll_loss + self.logdet_weight*nlogdet_loss
        ref_sample = torch.randn_like(sample)
        if self.radial:
            ref_sample = torch.nn.functional.normalize(ref_sample.view(sample.shape[0], -1))
            ref_sample = ref_sample.T * torch.abs(torch.randn(sample.shape[0]).type_as(sample))
            ref_sample = ref_sample.T.view(sample.shape)
        reference_nll_loss = torch.mean(nll(ref_sample,spatial_mean=self.spatial_mean, radial=self.radial)).detach()

        log = {
            "flow_loss": loss.detach(),
            "reference_nll_loss": reference_nll_loss,
            "nlogdet_loss": nlogdet_loss.detach(),
            "nll_loss": nll_loss.detach(),
            'logdet_w': self.logdet_weight,
            'nll_w' : self.nll_weight
        }
        return loss, log

class FlowLossAlternative(nn.Module):
    def __init__(self):
        super().__init__()
        # self.config = config

    def forward(self, sample, logdet):
        nll_loss = torch.mean(torch.sum(0.5*torch.pow(sample, 2), dim=1))
        nlogdet_loss = - logdet.mean()


        loss = nll_loss + nlogdet_loss
        reference_sample = torch.randn_like(sample)
        reference_nll_loss = torch.mean(torch.sum(0.5*torch.pow(reference_sample, 2), dim=1))
        log = {
            "flow_loss": loss,
            "reference_nll_loss": reference_nll_loss,
            "nlogdet_loss": nlogdet_loss,
            "nll_loss": nll_loss
        }
        return loss, log

class ExtendedFlowLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        # self.config = config

    def forward(self, sample_x, sample_v, logdet):
        nll_loss_x = torch.mean(nll(sample_x))
        nll_loss_v = torch.mean(nll(sample_v))
        assert len(logdet.shape) == 1
        nlogdet_loss = -torch.mean(logdet)
        loss = nll_loss_x + nll_loss_v + nlogdet_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample_x)))
        log = {
            "flow_loss": loss,
            "reference_nll_loss": reference_nll_loss,
            "nlogdet_loss": nlogdet_loss,
            "nll_loss_x": nll_loss_x,
            "nll_loss_v": nll_loss_v
        }
        return loss, log

def nll(sample, spatial_mean= False, radial = False):
    if len(sample.shape) == 2:
        sample = sample[:, :, None, None]
    if radial:
        shape = list(sample.shape)
        r = sample.view(shape[0], -1).norm(2,1)
        return (sum(shape[1:]) - 1.) * torch.log(r) + 0.5 * torch.pow(r,2)

    if spatial_mean:
        return 0.5 * torch.sum(torch.mean(torch.pow(sample, 2),dim=[2,3]), dim=1)
    else:
        return 0.5 * torch.sum(torch.pow(sample, 2), dim=[1, 2, 3])


class GaussianLogP(nn.Module):

    def __init__(self,mu=0.,sigma=1.):
        super().__init__()
        self.dist = Normal(loc=mu,scale=sigma)

    def forward(self,sample,logdet):
        nll_log_loss = torch.sum(self.dist.log_prob(sample)) / sample.size(0)
        nlogdet_loss = torch.mean(logdet)
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        nll_loss = torch.mean(nll(sample))
        loss = - (nll_log_loss + nlogdet_loss)
        log = {"flow_loss":loss,
               "reference_nll_loss":reference_nll_loss,
               "nlogdet_loss":-nlogdet_loss,
               "nll_loss": nll_loss,
               "nll_log_loss":-nll_log_loss}

        return loss, log

class NLLWithTypicality(nn.Module):
    def __init__(self, dim, lambda_t=0., lambda_nll=1.,loss_type='squared',fade_start=None,
                 fade_end=False,start_val=0., entropy_factor=1.):
        super().__init__()
        # # multivariate standard normal
        self.register_buffer('loc',torch.zeros(dim))
        self.register_buffer('scale',torch.eye(dim))

        self.base_dist = None
        self.loss_type = loss_type
        self.lambda_t = lambda_t
        self.lambda_nll = lambda_nll
        self.entropy_factor = entropy_factor

        self.fade = fade_start is not None and fade_end is not None

        if self.fade:
            print(f'Scaling typicality loss from {start_val} to {lambda_t} between iterations {fade_start} and {fade_end}')
            self.scaled_lambda = partial(linear_scaling, start_it=fade_start, end_it=fade_end,
                                         start_val=start_val,end_val=lambda_t, clip_min=start_val,
                                         clip_max=lambda_t)

        print(f'Initializing {self.__class__.__name__} with lambda_t = {self.lambda_t}')

    def init_distribution(self):
        self.base_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.loc, self.scale)

    def forward(self, sample, logdet, global_step=None):
        if len(sample.shape) == 2:
            sample = sample[:, :, None, None]
        nll_loss = torch.mean(nll(sample))
        assert len(logdet.shape) == 1
        assert self.base_dist is not None, 'Forgot to initialize normal'
        nlogdet_loss = -torch.mean(logdet)
        # sample entropy
        sample_entropy = - self.base_dist.log_prob(rearrange(sample,'b c h w -> b (c h w)')).mean()
        # entropy of base distribution
        base_entropy = self.entropy_factor * self.base_dist.entropy()
        # force sample entropy to be close to entropy of the standard normal, i.e. samples should lie within the typical set of the base distribution
        if self.loss_type == 'squared':
            typicality_loss = (sample_entropy - base_entropy)**2
        elif self.loss_type == 'abs':
            typicality_loss = (sample_entropy - base_entropy).abs()
        else:
            raise NotImplementedError()

        if self.fade:
            assert global_step is not None
            lambda_t = self.scaled_lambda(global_step)
        else:
            lambda_t = self.lambda_t

        loss = self.lambda_nll * nll_loss + nlogdet_loss + lambda_t * typicality_loss
        reference_nll_loss = torch.mean(nll(torch.randn_like(sample)))
        log = {"total_loss": loss, "reference_nll_loss": reference_nll_loss,
               "nlogdet_loss": nlogdet_loss, "nll_loss": nll_loss,
               'typicality_loss': typicality_loss
               }
        return loss, log


def linear_scaling(
    act_it, start_it, end_it, start_val, end_val, clip_min, clip_max
):
    act_val = (
        float(end_val - start_val) / (end_it - start_it) * (act_it - start_it)
        + start_val
    )
    return np.clip(act_val, a_min=clip_min, a_max=clip_max)