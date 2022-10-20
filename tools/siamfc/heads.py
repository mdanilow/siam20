from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F

from fft_conv import fft_conv


__all__ = ['SiamFC']


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
    
    def forward(self, z, x):
        # return self._fast_xcorr(z, x) * self.out_scale
        return self._faster_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out

    def _faster_xcorr(self, z, x):

        # out = F.conv2d(x, z)
        # padding = (8, 8, 8, 8)
        # z = F.pad(z, padding, "constant", 0)
        out = fft_conv(x, z)

        return out


class BBoxRegression(nn.Module):

    def __init__(self, out_scale=0.001):
        super(BBoxRegression, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_depthwise_xcorr(z, x) * self.out_scale

    def _fast_depthwise_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(1)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out
