#!/usr/bin/env python3
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nmt.layers import FactorizedEmbeddings

from utils import factorize_tensors


def clones(module, n):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, rank=None):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(FactorizedLinear(d_model, d_model, rank), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, rank=None):
        super().__init__()
        self.w_1 = FactorizedLinear(d_model, d_ff, rank)
        self.w_2 = FactorizedLinear(d_ff, d_model, rank)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TrFactorizedEmbeddings(FactorizedEmbeddings):

    def __init__(self, num_embeddings, embedding_dim, rank, padding_idx=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(TrFactorizedEmbeddings, self).__init__(num_embeddings, embedding_dim, rank, padding_idx=None,
                                                     max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                                     sparse=False, _weight=None)

    def forward(self, input):

        if self.factorized:
            low_rank_hidden = self.embeddings(input)
            final_embedding = self.linear(low_rank_hidden)

        else:
            final_embedding = self.embeddings(input)

        return final_embedding * math.sqrt(self.embedding_dim)

    def reverse_embeddings(self, hidden):

        if self.factorized:
            hidden = F.linear(hidden, self.linear.weight.t())
            proj = F.linear(hidden, self.embeddings.weight, bias=self.output_bias)

        else:
            proj = F.linear(hidden, self.embeddings.weight, bias=self.output_bias)

        return proj / math.sqrt(self.embedding_dim)


class FactorizedLinear(nn.Module):

    def __init__(self, in_size, out_size, rank):
        super(FactorizedLinear, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        if rank is None or rank >= out_size or rank >= in_size:
            self.factorized = False
            self.linear = nn.Linear(in_size, out_size)
        else:
            self.factorized = True
            self.linear = nn.Sequential(nn.Linear(in_size, rank), nn.Linear(rank, out_size))

    def forward(self, input):

        return self.linear(input)

    def factorize_layer(self, ratio):
        if self.factorized:
            return "already factorized !"
        else:
            t = self.linear.weight.data
            b = self.linear.bias.data
            rank = int(ratio * min(self.in_size, self.out_size))
            w, h = factorize_tensors(t, rank)
            self.linear = nn.Sequential(nn.Linear(self.in_size, rank, bias=False), nn.Linear(rank, self.out_size))
            self.linear[0].weight.data = h
            self.linear[1].weight.data = w
            self.linear[1].bias.data = b
            self.factorized = True
            return "successfully factorized"


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab_size, rank=None):
        super().__init__()
        self.proj = FactorizedLinear(d_model, vocab_size, rank=rank)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
