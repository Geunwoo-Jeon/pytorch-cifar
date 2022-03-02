import torch
from torch.optim.optimizer import Optimizer

__all__ = ('Thanos',)


class Thanos(Optimizer):
    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0,):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(Thanos, self).__init__(params, defaults)

    def step(self):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Thanos does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(self.state) == 0:
                    state['step'] = 0

                # state['step'] += 1

                if p.data.dim() == 4:  # p is conv weight with shape (in_channels, out_channels, x, y)
                    param_norm = p.data.norm(dim=[0, 1])
                    grad_norm = grad.norm(dim=[0, 1])
                elif p.data.dim() == 2:  # p is linear weight with shape (in_channels, out_channels)
                    param_norm = p.data.abs()
                    grad_norm = grad.abs()
                elif p.data.dim() == 1:  # p is bias with shape (out_channels)
                    param_norm = p.data.abs()
                    grad_norm = grad.abs()
                else:
                    param_norm = p.data.norm()
                    grad_norm = grad.norm()

                param_norm = param_norm.clamp(max=1)
                grad_norm = grad_norm.clamp(min=0.1)

                delta = grad / grad_norm * param_norm

                if group['weight_decay'] != 0:
                    delta.add_(p.data, alpha=group['weight_decay'])

                p.data.add_(delta, alpha=-group['lr'])

        return loss
