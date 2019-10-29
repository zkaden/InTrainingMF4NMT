#!/usr/bin/env python3

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, beginning_step=0):
        self.optimizer = optimizer
        self._step = beginning_step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def load_state_dict(self, state_dict):
        self._step = state_dict['_step']
        self.warmup = state_dict['warmup']
        self.factor = state_dict['factor']
        self.model_size = state_dict['model_size']
        self._rate = state_dict['_rate']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def state_dict(self):
        optimizer_dict = self.optimizer.state_dict()
        noam_dict = {
            '_step': self._step,
            'warmup': self.warmup,
            'factor': self.factor,
            'model_size': self.model_size,
            '_rate': self._rate,
        }
        noam_dict.update({'optimizer': optimizer_dict})
        return noam_dict


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing):
        super().__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        # print(true_dist)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # print(true_dist)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, Variable(true_dist, requires_grad=False))


class ONMTLabelSmoothing(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, size, padding_idx, smoothing):
        assert 0.0 < smoothing <= 1.0
        self.padding_idx = padding_idx
        super().__init__()

        smoothing_value = smoothing / (size - 2)
        one_hot = torch.full((size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class SimpleTransformerLoss:
    "A simple loss compute and train function."
    # def __init__(self, generator, criterion, opt=None):

    def __init__(self, criterion, optimizer=None):
        # self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, y, norm):
        # x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.data[0] * norm
