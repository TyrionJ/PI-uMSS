from torch import nn

from .nn import ConvNormReLU, ASC_Module


class UMSS_Net(nn.Module):
    def __init__(self, in_chns, out_chns):
        super().__init__()

        bc, deep = 32, 4
        self.in_layer = nn.Sequential(
            ConvNormReLU(in_chns, bc, (3, 3, 3), (1, 1, 1)),
            ConvNormReLU(bc, bc, (3, 3, 3), (1, 1, 1))
        )

        s, self.encoders, self.stages, self.asc_modules = None, nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
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
            self.asc_modules.append(ASC_Module(st * bc, st * bc, st * bc // 2))

        self.mid_block = ConvNormReLU(s * bc * 2, s * bc * 2, (3, 3, 3), (1, 1, 1))
        self.out_layer = nn.Conv3d(bc, out_chns, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        print('UMSS_Net initialized')

    def forward(self, x, **_):
        skips = [self.in_layer(x)]
        for encoder in self.encoders:
            skips.append(encoder(skips[-1]))

        lup_inp = self.mid_block(skips[-1])
        skips = skips[::-1]
        for i, (stage, asc_module) in enumerate(zip(self.stages, self.asc_modules)):
            t = asc_module(lup_inp, skips[i], skips[i+1])
            lup_inp = stage(t)

        return self.out_layer(lup_inp)
