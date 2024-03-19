# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.simclr import *
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select

import numpy as np

class SimCLR(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR (https://arxiv.org/abs/2002.05709).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
        """

        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature
        self.normalize: bool = cfg.method_kwargs.normalize
        self.learnable_temp = cfg.method_kwargs.learnable_temp
        if cfg.method_kwargs.learnable_temp:
            print('use learnable temp')
            self.temperature = torch.tensor(self.temperature)
            self.temperature = nn.Parameter(self.temperature)
        self.drop = cfg.method_kwargs.drop
        self.non_neg = cfg.method_kwargs.non_neg
        self.proj = cfg.method_kwargs.proj
        self.loss_type = cfg.method_kwargs.loss_type
        self.tau = cfg.method_kwargs.tau
        self.gsize = cfg.method_kwargs.gsize
        self.power_iter = cfg.method_kwargs.power_iter

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(SimCLR, SimCLR).add_and_assert_specific_cfg(cfg)
        cfg.method_kwargs.drop = omegaconf_select(cfg, "method_kwargs.drop", 0.0)
        cfg.method_kwargs.non_neg = omegaconf_select(cfg, "method_kwargs.non_neg", None)
        cfg.method_kwargs.proj = omegaconf_select(cfg, "method_kwargs.proj", 'vanilla')
        cfg.method_kwargs.loss_type = omegaconf_select(cfg, "method_kwargs.loss_type", 'xent')
        cfg.method_kwargs.tau = omegaconf_select(cfg, "method_kwargs.tau", 0.0)
        cfg.method_kwargs.gsize = omegaconf_select(cfg, "method_kwargs.gsize", 256)
        cfg.method_kwargs.learnable_temp = omegaconf_select(cfg, "method_kwargs.learnable_temp", 0)
        cfg.method_kwargs.normalize = omegaconf_select(cfg, "method_kwargs.normalize", False)
        cfg.method_kwargs.power_iter = omegaconf_select(cfg, "method_kwargs.power_iter", False)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        if self.proj in ['vanilla', 'gsoftmax_rep']:
            extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        else:
            extra_learnable_params = []
        if self.learnable_temp:
            extra_learnable_params += [{"name": "temp", "params": self.temperature}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().forward(X)
        if self.proj == 'vanilla':
            z = self.projector(out["feats"])
            out.update({"z": z})

        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[]: a dict containing the outputs of the parent
                and the projected features.
        """

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z = torch.cat(out["z"])

        # ------- non-negative -------
        if self.non_neg == None:
            # leave blank to restore original contrastive learning
            pass
        elif self.non_neg == 'relu': 
            # vanilla ReLU
            z = F.relu(z)
        elif self.non_neg == 'rep_relu': 
            # reparameterized ReLU: forward is ReLU and backward is GELU
            # more friendly to backpropagation and avoids dead neurons 
            gelu_z = F.gelu(z)
            z = gelu_z - gelu_z.data + F.relu(z).data
        # below are other viable options that enforce non-negativity either strictly or approximately. generally we find strict non-negativity to be more effective
        elif self.non_neg == 'gelu':
            z = F.gelu(z)
        elif self.non_neg == 'sigmoid':
            z = F.sigmoid(z)
        elif self.non_neg == 'softplus':
            z = F.softplus(z)
        elif self.non_neg == 'exp':
            z = torch.exp(z)
        elif self.non_neg == 'leakyrelu':
            z = F.leaky_relu(z)

        # ------- contrastive loss -------
        n_augs = self.num_large_crops + self.num_small_crops
        indexes = indexes.repeat(n_augs)
        # row normalize the features
        if self.normalize == 'none':
            pass
        elif self.normalize == 'dim':
            z = F.normalize(z, dim=-1)



        if self.loss_type == 'xent':
            if self.learnable_temp:
                temp = F.softplus(self.temperature).clamp(min=self.learnable_temp)
                self.log("temp", temp, on_epoch=True, sync_dist=True)
            else:
                temp = self.temperature
            nce_loss = simclr_loss_func(
                z,
                indexes=indexes,
                temperature=temp,
            )
            
            self.log("train_nce_loss", nce_loss, on_epoch=True, sync_dist=True)
            
            return nce_loss + class_loss




