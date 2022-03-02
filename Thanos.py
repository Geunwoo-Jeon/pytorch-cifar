import torch
from torch.optim.optimizer import Optimizer

__all__ = ('Thanos',)


class Thanos(Optimizer):
    def __init__(self, params, lr: float = 1e-4, weight_decay: float = 0, wait: int = 1):
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if weight_decay < 0:
            raise ValueError('Invalid weight_decay value: {}'.format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay, wait=wait)
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
                if len(state) == 0:
                    state['step'] = 0
                    if group['wait'] != 1:
                        state['delta_sum'] = torch.zeros_like(p)
                    # state['lr_decay'] = torch.ones_like(p)
                    # state['prev_sign'] = p.sign()

                # lr_decay, prev_sign = state['lr_decay'], state['prev_sign']
                state['step'] += 1

                if p.dim() == 4:  # p is conv weight with shape (in_channels, out_channels, x, y)
                    param_norm = p.data.norm(dim=[0, 1])
                    grad_norm = grad.norm(dim=[0, 1])
                elif p.dim() == 2:  # p is linear weight with shape (in_channels, out_channels)
                    param_norm = p.abs()
                    grad_norm = grad.abs()
                elif p.dim() == 1:  # p is bias with shape (out_channels)
                    param_norm = p.abs()
                    grad_norm = grad.abs()
                else:
                    param_norm = p.data.norm()
                    grad_norm = grad.norm()

                grad_norm = grad_norm.clamp(min=1e-8)
                param_norm = param_norm.clamp(min=1e-8)

                # sign_changed = prev_sign != p.sign()
                # lr_decay = lr_decay * (1 - 0.01 * sign_changed)

                delta = grad / grad_norm * param_norm  # * lr_decay

                if group['weight_decay'] != 0:
                    delta.add_(p.data, alpha=group['weight_decay'])

                if group['wait'] != 1:
                    state['delta_sum'] += delta
                    if state['step'] % group['wait'] == 0:
                        p.data.add_(state['delta_sum'], alpha=-group['lr'])
                        state['delta_sum'] *= 0
                else:
                    p.data.add_(delta, alpha=-group['lr'])

        return loss
