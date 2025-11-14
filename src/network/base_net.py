import torch
from torch import nn

from .nn import ConvNormReLU


class BaseNet(nn.Module):
    def __init__(self, in_chns, out_chns):
        super().__init__()

        bc, deep = 32, 4
        self.in_layer = nn.Sequential(
            ConvNormReLU(in_chns, bc, (3, 3, 3), (1, 1, 1)),
            ConvNormReLU(bc, bc, (3, 3, 3), (1, 1, 1))
        )

        self.encoders, self.stages, self.trans_convs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(deep):
            s, st = 2 ** i, 2 ** (deep - i)
            self.encoders.append(nn.Sequential(
                ConvNormReLU(s * bc, s * bc * 2, (3, 3, 3), (2, 2, 2)),
                ConvNormReLU(s * bc * 2, s * bc * 2, (3, 3, 3), (1, 1, 1)))
            )
            self.stages.append(nn.Sequential(
                ConvNormReLU(st * bc, st * bc // 2, (3, 3, 3), (1, 1, 1)),
                ConvNormReLU(st * bc // 2, st * bc // 2, (3, 3, 3), (1, 1, 1)))
            )
            self.trans_convs.append(nn.ConvTranspose3d(st * bc, st * bc // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        self.out_layer = nn.Conv3d(bc, out_chns, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        print('BaseNet initialized')

    def forward(self, x, **_):
        skips = [self.in_layer(x)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1]))

        lup_inp = skips.pop(-1)
        for skip, stage, trans_conv in zip(skips[::-1], self.stages, self.trans_convs):
            lup_inp = stage(torch.cat([trans_conv(lup_inp), skip], dim=1))

        return self.out_layer(lup_inp)
