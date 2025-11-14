from .base_net import BaseNet
from .net_loss import NetLoss
from .uMSS_net import UMSS_Net


__all__ = ['NetLoss', 'get_net']


def get_net(net_name, **kwargs):
    in_chns = kwargs.get('in_chns', 2)
    out_chns = kwargs.get('out_chns', 4)

    if net_name == 'BaseNet':
        return BaseNet(in_chns, out_chns)
    elif net_name == 'UMSS_Net':
        return UMSS_Net(in_chns, out_chns)
    else:
        raise NotImplementedError(f'Unsupported network {net_name}.')
