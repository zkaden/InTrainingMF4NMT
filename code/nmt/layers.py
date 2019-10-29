import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
import numpy as np

import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch
import torch.nn.functional as F

from utils import factorize_tensors


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


class AdvancedLSTM(nn.LSTM):
    """
    Wrapper on the LSTM class, with learned initial state
    """

    def __init__(self, *args, **kwargs):
        super(AdvancedLSTM, self).__init__(*args, **kwargs)
        bi = 2 if self.bidirectional else 1
        self.h0 = nn.Parameter(torch.FloatTensor(bi * self.num_layers, 1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(bi * self.num_layers, 1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(-1, n, -1).contiguous(),
            self.c0.expand(-1, n, -1).contiguous()
        )

    def forward(self, input, hx=None):
        if hx is None:
            if isinstance(input, rnn.PackedSequence):
                n = input.batch_sizes[0]
            else:
                n = input.size(1)
            hx = self.initial_state(n)
        return super(AdvancedLSTM, self).forward(input, hx=hx)


class AdvancedLSTMCell(nn.LSTMCell):
    # Extend LSTMCell to learn initial state
    def __init__(self, *args, **kwargs):
        super(AdvancedLSTMCell, self).__init__(*args, **kwargs)
        self.h0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())
        self.c0 = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())

    def initial_state(self, n):
        return (
            self.h0.expand(n, -1).contiguous(),
            self.c0.expand(n, -1).contiguous()
        )


class MultipleLSTMCells(nn.Module):

    def __init__(self, num_layers, lstm_sizes, dropout_vertical=0, dropout_horizontal=0, residual=False):
        super(MultipleLSTMCells, self).__init__()
        self.num_layers = num_layers
        self.lstm_sizes = lstm_sizes
        self.cells = nn.ModuleList([AdvancedLSTMCell(lstm_sizes[i], lstm_sizes[i + 1])
                                    for i in range(num_layers)])
        self.dropouts = nn.ModuleList(
            [VariationalDropout(dropout_horizontal, lstm_sizes[i + 1]) for i in range(num_layers)])
        self.dr = nn.Dropout(dropout_vertical)
        self.residual = residual
        if self.residual:
            self.where_residual = []
            one_residual = False
            for i in range(num_layers):
                if self.lstm_sizes[i] == self.lstm_sizes[i + 1]:
                    self.where_residual.append(True)
                    one_residual = True
                else:
                    self.where_residual.append(False)
            if not one_residual:
                print(" No skip connection : need similar layer sizes to have residual cells")
                self.residual = False

    def sample_masks(self):

        for i in range(self.num_layers):
            self.dropouts[i].sample_mask()

    def forward(self, input, previous_state=None):
        # if previous_state is None:
        #    previous_state = [c.initial_state() for c in self.cells]

        new_state = previous_state[0].new_full(previous_state[0].size(), 0), \
            previous_state[1].new_full(previous_state[1].size(), 0)

        h = input
        for i in range(self.num_layers):
            # print(h.size(), previous_state[0][i].size(), previous_state[1][i].size())
            residual = h
            h, c = self.cells[i](h, (previous_state[0][i], previous_state[1][i]))
            if self.residual and self.where_residual[i]:
                h = residual + h
            # print(i)
            # print(h)
            new_state[0][i] = self.dropouts[i](h)
            new_state[1][i] = c
            h = self.dr(h)
            # print("dropout")
            # print(h)

        return h, new_state


class VariationalDropout(nn.Module):

    def __init__(self, dropout_rate, hidden_size):
        super(VariationalDropout, self).__init__()

        self.p = dropout_rate
        self.hidden_size = hidden_size
        self.mask = torch.autograd.Variable(torch.Tensor(
            np.ones((1, hidden_size))), requires_grad=False).float().cuda()

    def sample_mask(self):
        new_mask = torch.Tensor((np.random.rand(1, self.hidden_size) < self.p) / self.p)
        self.mask = torch.nn.Parameter(new_mask, requires_grad=False).float().cuda()

    def forward(self, h):
        if self.training and self.p > 0:
            return h * self.mask.expand(h.size())
        else:
            return h


class FactorizedEmbeddings(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, rank, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(FactorizedEmbeddings, self).__init__()

        self.input_dim = num_embeddings
        self.embedding_dim = embedding_dim
        self.output_bias = nn.Parameter(torch.FloatTensor(num_embeddings,).zero_())

        if rank is None or rank >= embedding_dim:
            self.factorized = False
            self.embeddings = nn.Embedding(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                                           scale_grad_by_freq, sparse, _weight)
        else:
            self.factorized = True
            self.embeddings = nn.Embedding(num_embeddings, rank, padding_idx, max_norm, norm_type,
                                           scale_grad_by_freq, sparse, _weight)
            self.linear = nn.Linear(rank, embedding_dim, bias=False)

    def forward(self, input):

        if self.factorized:
            low_rank_hidden = self.embeddings(input)
            final_embedding = self.linear(low_rank_hidden)
            return final_embedding

        else:
            return self.embeddings(input)

    def reverse_embeddings(self, hidden):

        if self.factorized:
            hidden = F.linear(hidden, self.linear.weight.t())
            return F.linear(hidden, self.embeddings.weight, bias=self.output_bias)

        else:
            return F.linear(hidden, self.embeddings.weight, bias=self.output_bias)

    def factorize_layer(self, ratio):
        if self.factorized:
            return "already factorized !"
        else:
            t = self.embeddings.weight.data
            b = self.embeddings.bias.data
            rank = int(ratio * min(self.input_dim, self.embedding_dim))
            w, h = factorize_tensors(t, rank)
            self.embeddings = nn.Embedding(self.input_dim, rank)
            self.linear = nn.Linear(rank, self.embedding_dim, bias=False)
            self.embeddings.weight.data = w
            self.embeddings.bias.data = b
            self.linear.weight.data = h.t()
            self.factorized = True
            return "successfully factorized"
