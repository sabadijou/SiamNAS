from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
        # self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, z, x):
        return self.conv2d_dw_group(x, z) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation

        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        # out = self.batch_norm(out)
        return out

    def conv2d_dw_group(self, x, kernel):
        batch, channel = kernel.shape[:2]
        x = x.view(1, batch * channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
