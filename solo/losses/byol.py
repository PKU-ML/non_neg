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

import torch
import torch.nn.functional as F


def byol_loss_func(p: torch.Tensor, z: torch.Tensor, simplified: bool = True) -> torch.Tensor:
    """Computes BYOL's loss given batch of predicted features p and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.

    Returns:
        torch.Tensor: BYOL's loss.
    """

    if simplified:
        return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)


    return 2 - 2 * (p * z.detach()).sum(dim=1)

def byol_loss_tri(p: torch.Tensor, z: torch.Tensor, scale_param=None,simplified: bool = True) -> torch.Tensor:
    """Computes BYOL's loss given batch of predicted features p and projected momentum features z.

    Args:
        p (torch.Tensor): NxD Tensor containing predicted features from view 1
        z (torch.Tensor): NxD Tensor containing projected momentum features from view 2
        simplified (bool): faster computation, but with same result. Defaults to True.

    Returns:
        torch.Tensor: BYOL's loss.
    """

    N,D = p.size()

    if simplified:
        corr = p.T @ p
        diag = torch.eye(D, device=corr.device)
        cdif = (corr - diag).pow(2)
        dec_loss = cdif.mean()
        scale = torch.nn.functional.softplus(scale_param).unsqueeze(0)
        p = p * scale
        return 2 - 2 * F.cosine_similarity(p, z.detach(), dim=-1).mean(),dec_loss

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)

    corr = p.T @ p
    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    dec_loss = cdif.mean()
    scale = torch.nn.functional.softplus(scale_param).unsqueeze(0)
    sum_we=torch.sum(scale)/D
    scale = scale / sum_we
    p = p * scale

    return 2 - 2 * (p * z.detach()).sum(dim=1).mean(),dec_loss
