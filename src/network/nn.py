import torch
from torch import nn


class ConvNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=(1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class ASC_Module(nn.Module):
    def __init__(self, in_chns_FL, in_chns_FT, in_chns_FH):
        super().__init__()

        self.conv_FT = nn.Conv3d(in_chns_FT, in_chns_FT, 1)
        self.conv_FH = nn.Conv3d(in_chns_FH, in_chns_FH, 1)
        self.conv_HT = nn.Conv3d(in_chns_FH, in_chns_FH, 1)
        self.conv_HL = nn.Conv3d(in_chns_FH, in_chns_FH, 1)
        self.up_conv_FL = nn.ConvTranspose3d(in_chns_FL, in_chns_FL // 2, kernel_size=2, stride=2)
        self.up_conv_FT = nn.ConvTranspose3d(in_chns_FT, in_chns_FT // 2, kernel_size=2, stride=2)

    def forward(self, F_L, F_T, F_H):
        F_L = self.up_conv_FL(F_L)
        F_T = self.conv_FT(F_T)
        F_T = self.up_conv_FT(F_T)
        F_H = self.conv_FH(F_H)
        F_HT = self.conv_HT(F_H - F_T)
        F_HT = torch.sigmoid(F_HT) * F_H
        F_HT = self.conv_HL(F_HT)
        out = torch.concatenate([F_L, F_HT], dim=1)
        return out
