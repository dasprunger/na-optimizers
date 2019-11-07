from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import SGD

from na_sgd import NaSGD


class Quadratic(nn.Module):
    def __init__(self, start=torch.tensor([1.0, 1.0])):
        super(Quadratic, self).__init__()
        self.arg = nn.Parameter(start)

    def forward(self, *input):
        return 8.0 * self.arg[0] ** 2 + 0.5 * self.arg[1] ** 2


class Rosenbrock(nn.Module):
    def __init__(self, start=torch.tensor([-3.0, -4.0])):
        super(Rosenbrock, self).__init__()
        self.arg = nn.Parameter(start)

    def forward(self, *input):
        return (1 - self.arg[0]) ** 2 + 100 * (self.arg[1] - self.arg[0] ** 2) ** 2


def test_for_levels(q, optim, levels, timeout=100000):
    """Optimizes the function q using the optimizer optim, recording the number of steps required to first go below the
    values present in the levels list. Returns a triple (steps_to_level: list, didDiverge: bool, didTimeout: bool),
    where the last two flags indicate if the optimizer diverged or timed out. If both are false, steps_to_level has the
    same shape as levels."""
    steps, min_val = 0, q()
    steps_to_level = []
    for level in levels:
        while min_val == min_val and min_val > level:  # 'min_val == min_val' catches divergences
            optim.zero_grad()
            v = q()
            v.backward()
            def cl(): return v
            optim.step(closure=cl)

            min_val = min(v, min_val)
            steps += 1

            if steps > timeout:
                return steps_to_level, False, True
        if min_val != min_val:  # true when diverged
            return steps_to_level, True, False
        else:
            steps_to_level.append(steps)
    return steps_to_level, False, False


if __name__ == '__main__':
    r = Rosenbrock()
    levels = [10.0 ** _ for _ in range(3, -8, -1)]

    q = deepcopy(r)
    pd, na_mom = 0.8, 0.0
    sgd = NaSGD(q.parameters(), pd=pd, momentum=na_mom)
    print('NaSGD(pd={:.1f}, momentum={:.1f}): '.format(pd, na_mom), end='')
    print(test_for_levels(q, sgd, levels)[0])

    q = deepcopy(r)
    lr, sgd_mom = 0.0003, 0.0
    sgd = SGD(q.parameters(), lr=lr, momentum=sgd_mom)  # learning rates >= 0.0004 diverge, 0.0002 is slower...
    print('SGD(lr={:.4f}, momentum={:.1f}): '.format(lr, sgd_mom), end='')
    print(test_for_levels(q, sgd, levels)[0])
