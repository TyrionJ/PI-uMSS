import torch
from torch import nn


class TVLoss(nn.Module):
    def __init__(self, L1L2flag='L2'):
        super(TVLoss, self).__init__()
        assert L1L2flag in ['L1', 'L2'], 'L1L2flag must be either L1 or L2'
        self.L1L2flag = L1L2flag

    def forward(self, x, WG=None):
        row = list(range(1, x.shape[1]))
        row.append(x.shape[1] - 1)
        col = list(range(1, x.shape[2]))
        col.append(x.shape[2] - 1)
        z = list(range(1, x.shape[3]))
        z.append(x.shape[3] - 1)

        dx = x - x[:, row, :, :]
        dy = x - x[:, :, col, :]
        dz = x - x[:, :, :, z]

        if WG is None:
            pass
        else:
            dx = WG[:, :, :, 0].unsqueeze(-1) * dx
            dy = WG[:, :, :, 1].unsqueeze(-1) * dy
            dz = WG[:, :, :, 2].unsqueeze(-1) * dz

        if self.L1L2flag == 'L1':
            dx = (torch.abs(dx))
            dy = (torch.abs(dy))
            dz = (torch.abs(dz))
            tv_loss = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3
        else:
            dx_sum = torch.pow(dx, 2).sum()
            dy_sum = torch.pow(dy, 2).sum()
            dz_sum = torch.pow(dz, 2).sum()
            count_x, count_y, count_z = self._torch_size(dx), self._torch_size(dy), self._torch_size(dz)
            tv_loss = (dx_sum / count_x + dy_sum / count_y + dz_sum / count_z) / 3

        return tv_loss

    @staticmethod
    def _torch_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GradientLoss(nn.Module):
    def __init__(self, L1L2flag='L2'):
        super(GradientLoss, self).__init__()
        assert L1L2flag in ['L1', 'L2'], 'L1L2flag must be either L1 or L2'
        self.L1L2flag = L1L2flag

    def forward(self, x, y, WG=None):

        row = list(range(1, x.shape[1]))
        row.append(x.shape[1] - 1)
        col = list(range(1, x.shape[2]))
        col.append(x.shape[2] - 1)
        z = list(range(1, x.shape[3]))
        z.append(x.shape[3] - 1)

        dx1 = x - x[:, row, :, :]
        dy1 = x - x[:, :, col, :]
        dz1 = x - x[:, :, :, z]

        dx2 = y - y[:, row, :, :]
        dy2 = y - y[:, :, col, :]
        dz2 = y - y[:, :, :, z]

        dx = dx1 - dx2
        dy = dy1 - dy2
        dz = dz1 - dz2

        if WG is None:
            pass
        else:
            dx = WG[:, :, :, 0].unsqueeze(-1) * dx
            dy = WG[:, :, :, 1].unsqueeze(-1) * dy
            dz = WG[:, :, :, 2].unsqueeze(-1) * dz

        if self.L1L2flag  == 'L1':
            dx = torch.abs(dx)
            dy = torch.abs(dy)
            dz = torch.abs(dz)

            grd_loss = (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3
        else:
            dx_sum = torch.pow(dx, 2).sum()
            dy_sum = torch.pow(dy, 2).sum()
            dz_sum = torch.pow(dz, 2).sum()

            count_x, count_y, count_z = self._torch_size(dx), self._torch_size(dy), self._torch_size(dz)
            grd_loss = (dx_sum / count_x + dy_sum / count_y + dz_sum / count_z) / 3
        return grd_loss

    @staticmethod
    def _torch_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
