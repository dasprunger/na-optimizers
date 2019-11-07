import torch
from torch.optim.optimizer import Optimizer


class NaSGD(Optimizer):
    """Norm adapted gradient descent, for use with nonnegative loss functions with percent decrease targeting.

    This is a modified version of PyTorch's SGD implementation."""
    def __init__(self, params, pd=0.8, momentum=0):
        if pd <= 0.0:
            raise ValueError("Invalid percent decrease: {}".format(pd))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(pd=pd, momentum=momentum)
        super(NaSGD, self).__init__(params, defaults)

    def step(self, closure: callable = None) -> tuple:
        if closure is not None:
            loss = closure()
        else:
            raise RuntimeError("need a loss (closure) to do norm-adapted descent")

        squared_norm = torch.zeros([])
        for group in self.param_groups:
            for p in group['params']:
                squared_norm += torch.sum(p.grad.data.flatten() * p.grad.data.flatten())

        for group in self.param_groups:
            for p in group['params']:
                pd = group['pd']
                momentum = group['momentum']

                if p.grad is None:
                    continue

                d_p = loss / squared_norm.item() * p.grad.data

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(-pd * d_p)

        return loss, squared_norm
