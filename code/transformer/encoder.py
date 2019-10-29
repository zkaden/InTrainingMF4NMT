#!/usr/bin/env python3
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from transformer.layers import clones, LayerNorm, EncoderLayer, PositionalEncoding
from transformer.layers import MultiHeadedAttention, PositionwiseFeedForward
from configuration import TransformerConfig as tconfig
from transformer.layers import TrFactorizedEmbeddings as FactorizedEmbeddings


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, vocab_size, embedding_rank, inner_rank=None, ffward_rank=None):
        super().__init__()
        self.vocab = vocab_size
        layer = EncoderLayer(
            tconfig.layer_dimension,
            MultiHeadedAttention(
                tconfig.num_attention_heads,
                tconfig.layer_dimension,
                rank=inner_rank
            ),
            PositionwiseFeedForward(
                tconfig.layer_dimension,
                tconfig.inner_layer_dimension,
                tconfig.dropout,
                rank=ffward_rank
            ),
            tconfig.dropout,
        )

        self.layers = clones(layer, tconfig.num_layers)
        self.norm = LayerNorm(layer.size)
        self.src_embed = nn.Sequential(
            FactorizedEmbeddings(
                vocab_size,
                tconfig.layer_dimension,
                embedding_rank
            ),
            PositionalEncoding(
                tconfig.layer_dimension,
                tconfig.dropout,
            ),
        )

    def forward(self, sequences):
        padded_seq = pad_sequence(sequences).transpose(0, 1)
        mask = (padded_seq != 0).unsqueeze(-2)
        embed = self.src_embed(padded_seq)

        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            embed = layer(embed, mask)
        return self.norm(embed), mask
