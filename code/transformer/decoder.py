#!/usr/bin/env python3
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformer.optimizer import LabelSmoothing, ONMTLabelSmoothing
from transformer.layers import clones, LayerNorm, DecoderLayer, PositionalEncoding
from transformer.layers import MultiHeadedAttention, PositionwiseFeedForward, TrFactorizedEmbeddings, Generator
from configuration import TransformerConfig as tconfig


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_rank, attention_rank=None, ffward_rank=None):
        super().__init__()
        layer = DecoderLayer(
            tconfig.layer_dimension,
            MultiHeadedAttention(
                tconfig.num_attention_heads,
                tconfig.layer_dimension,
                rank=attention_rank
            ),
            MultiHeadedAttention(
                tconfig.num_attention_heads,
                tconfig.layer_dimension,
                rank=attention_rank
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
        self.tgt_embed = nn.Sequential(
            TrFactorizedEmbeddings(
                vocab_size,
                tconfig.layer_dimension,
                embedding_rank
            ),
            PositionalEncoding(
                tconfig.layer_dimension,
                tconfig.dropout,
            ),
        )
        self.generator = Generator(
            tconfig.layer_dimension,
            vocab_size,
            rank=embedding_rank
        )
        self.train_criterion = LabelSmoothing(vocab_size, padding_idx=0, smoothing=0.1)
        # self.train_criterion = ONMTLabelSmoothing(vocab_size, padding_idx=0, smoothing=0.1)
        self.eval_criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, src_enc, src_mask, tgt):
        decode_tgt = [tensor[:-1] for tensor in tgt]  # remove EOS token
        padded_tgt_decode = pad_sequence(decode_tgt).transpose(0, 1)
        padded_tgt_loss = pad_sequence(tgt).transpose(0, 1)[:, 1:]
        tgt_mask = self.make_std_mask(padded_tgt_decode)
        embed = self.tgt_embed(padded_tgt_decode)

        for layer in self.layers:
            embed = layer(embed, src_enc, src_mask, tgt_mask)
        out = self.generator(self.norm(embed))

        norm = (padded_tgt_loss != 0).data.sum().item()

        if self.training:
            loss = self.train_criterion(out.contiguous().view(-1, out.size(-1)),
                                  padded_tgt_loss.contiguous().view(-1)) / norm

            return loss, norm
        # evaluate perplexity with cross-entropy loss
        else:
            lens = [len(tensor) for tensor in decode_tgt]
            pred_sents = [out[i, :l, :] for i, l in enumerate(lens)]
            pred_sents = torch.cat(pred_sents)
            tgt_sents = [tensor[1:len(tensor)] for tensor in tgt]
            tgt_sents = torch.cat(tgt_sents)

            loss = self.eval_criterion(pred_sents, tgt_sents)

            return loss, 1

    def make_std_mask(self, tgt, pad=0):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            self.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    def subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    def get_word_scores(self, src_enc, src_mask, tgt):
        padded_tgt = pad_sequence(tgt).transpose(0, 1)
        tgt_mask = self.make_std_mask(padded_tgt)
        embed = self.tgt_embed(padded_tgt)

        for layer in self.layers:
            embed = layer(embed, src_enc, src_mask, tgt_mask)
        return self.generator(self.norm(embed)[:, -1])
