import torch
from torch import nn
from torch.nn import Module
from torch.fft import fftn, ifftn

from .loss_utils import TVLoss, GradientLoss


class NetLoss(Module):
    def __init__(self, L1L2flag='L1', WeightPre=0.05):
        super(NetLoss, self).__init__()

        assert L1L2flag in ['L1', 'L2'], 'L1L2flag must be either L1 or L2'
        self.loss_fn = nn.L1Loss() if L1L2flag == 'L1' else nn.MSELoss()
        self.tv_loss = TVLoss(L1L2flag)
        self.grd_loss = GradientLoss(L1L2flag)
        self.WeightPre = WeightPre

    def forward(self, net_out, R2star, Chi, Fld, DiKnl, slicer, pre_training=False, no_PI=False):
        mask = torch.ones_like(Chi)
        mask[Chi == 0] = 0

        alpha_p, ahpla_d, k_slop, b_intercept = [net_out[:, i] for i in range(net_out.shape[1])]
        ONE = torch.ones_like(alpha_p[mask==1])

        if no_PI or self.WeightPre == 0:
            L_pre = 0
        else:
            L_alpha = self.loss_fn(alpha_p[mask==1], ONE * 137)
            L_alpha += self.loss_fn(ahpla_d[mask==1], ONE * 137)
            L_linar = self.loss_fn(k_slop[mask==1], ONE * 1.1781)
            L_linar += self.loss_fn(b_intercept[mask==1], ONE * 12.9629)
            L_pre = L_alpha + L_linar
            if pre_training:
                return L_pre

        R2_prime = (R2star - b_intercept) / k_slop * mask
        Chi_para, Chi_dia = cal_para_dia_mag(Chi, R2_prime, alpha_p, ahpla_d, mask)

        ZERO = torch.zeros_like(R2_prime[mask == 1].min())
        L_R2 = self.loss_fn(R2_prime[mask == 1].min(), ZERO)
        L_para = self.loss_fn(Chi_para[mask == 1].min(), ZERO)
        L_dia = self.loss_fn(Chi_dia[mask == 1].min(), ZERO)
        L_rng = 0.1 * L_R2 + L_para + L_dia

        Chi_para[Chi_para <= 0] = 0
        Chi_dia[Chi_dia <= 0] = 0
        pos = torch.where(((Chi_dia == 0) | (Chi_para == 0)) & (mask == 1))
        L_Chi = self.loss_fn((Chi_para - Chi_dia)[pos], Chi[pos])

        full_para = torch.zeros_like(DiKnl)
        full_dia = torch.zeros_like(DiKnl)
        full_fld = torch.zeros_like(DiKnl)
        full_para[slicer] = Chi_para
        full_dia[slicer] = Chi_dia
        full_fld[slicer] = Fld
        fld_hat = ifftn(DiKnl * fftn(full_para - full_dia))[slicer]
        L_fld = self.loss_fn(fld_hat[mask==1], Fld[mask==1])

        L_TV = self.tv_loss(Chi_para) + self.tv_loss(Chi_dia)
        abs_Chi = abs(Chi)
        L_grd = self.grd_loss(Chi_para, abs_Chi) + self.grd_loss(Chi_dia, abs_Chi)

        return self.WeightPre * L_pre + L_Chi + 0.1 * L_fld + L_rng + 0.1 * L_TV + 0.1 * L_grd


"""
R2prime = alpha_p * para + ahpla_d * dia
Chi = para - dia
para, dia >= 0
"""
def cal_para_dia_mag(Chi, R2_prime, alpha_p, ahpla_d, mask):
      para = (R2_prime + ahpla_d * Chi) / (alpha_p + ahpla_d)
      dia = (R2_prime - alpha_p * Chi) / (alpha_p + ahpla_d)

      return para * mask, dia * mask
