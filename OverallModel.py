from pickle import TRUE
import torch
import math
import torch.nn as nn
from basicModule import *
import numpy as np
from VolFormer import *                      #VolFormer


class SSB(nn.Module):
    def __init__(self, n_feats, kernel_size, act, res_scale, conv=default_conv):
        super(SSB, self).__init__()
        self.spa = ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
        self.spc = ResAttentionBlock(conv, n_feats, 1, act=act, res_scale=res_scale)

    def forward(self, x):
        return self.spc(self.spa(x))

class SSPN(nn.Module):
    def __init__(self, n_feats, n_blocks, act, res_scale):
        super(SSPN, self).__init__()
        kernel_size = 3

        self.net1 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale)
        self.net2 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale)
        self.net3 = SSB(n_feats, kernel_size, act=act, res_scale=res_scale)

    def forward(self, x):
        res = self.net1(x)
        res = self.net2(res)
        res = self.net3(res)
        res += x
        return res

# a single branch of proposed SSPSR
class BranchUnit(nn.Module):
    def __init__(self, n_colors, n_feats, n_outputs, n_blocks, act, res_scale, up_scale, use_tail=True, conv=default_conv):
        super(BranchUnit, self).__init__()
        kernel_size = 3
        self.head = conv(n_colors, n_feats, kernel_size)
        self.body = SSPN(n_feats, n_blocks, act, res_scale)

        self.upsample = Upsampler(conv, up_scale, n_feats)
        self.tail = None
        if use_tail:
            self.tail = conv(n_feats, n_outputs, kernel_size)

    def forward(self, x):
        y = self.head(x)
        y = self.body(y)
        y = self.upsample(y)
        if self.tail is not None:
            y = self.tail(y)

        return y

class General_VolFormer(nn.Module):
    def __init__(self, n_subs, n_ovls, n_colors, n_blocks, n_feats, n_scale, res_scale, use_share=True, conv=default_conv) -> object:
        super(General_VolFormer, self).__init__()
        kernel_size = 3
        window_size=8 #7
        self.shared = use_share
        act = nn.ReLU(True)

        # calculate the group number (the number of branch networks)
        self.G = math.ceil((n_colors - n_ovls) / (n_subs - n_ovls))
        # calculate group indices
        self.start_idx = []
        self.end_idx = []

        for g in range(self.G):
            sta_ind = (n_subs - n_ovls) * g
            end_ind = sta_ind + n_subs
            if end_ind > n_colors:
                end_ind = n_colors
                sta_ind = n_colors - n_subs
            self.start_idx.append(sta_ind)
            self.end_idx.append(end_ind)

        self.branch1 = VolFormer(in_chans=n_subs,  out_chans=n_subs,  window_size=window_size, img_range=1, depths=[6,6,6,6,6,6,6,6],embed_dim=180, num_heads=[6,6,6,6,6,6,6,6], mlp_ratio=2, upscale=n_scale).cuda()

        self.trunk = BranchUnit(n_colors, n_feats, n_subs, n_blocks, act, res_scale, up_scale=1, use_tail=False, conv=default_conv)
        self.skip_conv = conv(n_colors, n_feats, kernel_size)
        self.final = conv(n_feats, n_colors, kernel_size)
        self.sca = n_scale
        # define decoder for RGB images
        self.trunk_RGB = BranchUnit(n_subs, n_feats, n_feats, n_blocks, act, res_scale, up_scale=1, use_tail=False, conv=default_conv)
        self.skip_conv_RGB = conv(n_subs, n_feats, kernel_size)
        self.final_RGB = conv(n_feats, n_subs, kernel_size)

    def forward(self, x, lms, modality,img_size):
        b, c, h, w = x.shape

        # the rest steps depend on the modality which could be spectral images or RGB images
        if modality == "spectral":
            # Initialize intermediate “result”, which is upsampled with n_scale times
            y = torch.zeros(b, c, self.sca * h, self.sca * w).cuda()

            channel_counter = torch.zeros(c).cuda()
            for g in range(self.G):
                sta_ind = self.start_idx[g]
                end_ind = self.end_idx[g]

                xi = x[:, sta_ind:end_ind, :, :]
                xi = self.branch1(xi,img_size)
                y[:, sta_ind:end_ind, :, :] += xi
                channel_counter[sta_ind:end_ind] = channel_counter[sta_ind:end_ind] + 1

            # intermediate “result” is averaged according to their spectral indices
            y = y / channel_counter.unsqueeze(1).unsqueeze(2)

            y = self.trunk(y)
            y = y + self.skip_conv(lms)
            y = self.final(y)

        elif modality == "rgb":

            y = self.branch1(x,img_size)
            y = self.trunk_RGB(y)
 
            y = y + self.skip_conv_RGB(lms)
            y = self.final_RGB(y)

        else:
            raise("Not implemented!!!")
        return y

