import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.utils.rnn as rnn
import numpy as np

import configuration
from nmt.layers import init_weights, AdvancedLSTM, FactorizedEmbeddings

mconfig = configuration.LSTMConfig()


class Encoder(nn.Module):
    def __init__(self, vocab_size, context_projection=None, state_projection=None):
        super(Encoder, self).__init__()
        print(vocab_size)
        self.hidden_size = mconfig.hidden_size_encoder
        self.embed_size = mconfig.embed_size
        self.rank = mconfig.rank_encoder
        self.lookup = FactorizedEmbeddings(vocab_size, self.embed_size, self.rank)
        self.lstm = AdvancedLSTM(self.embed_size, self.hidden_size, num_layers=mconfig.num_layers_encoder,
                                 bidirectional=mconfig.bidirectional_encoder, dropout=mconfig.dropout_layers)
        self.dr = nn.Dropout(mconfig.dropout_layers)
        self.act = nn.Tanh()

        self.use_context_projection = False
        if context_projection is not None:
            self.use_context_projection = True
            self.context_projection = nn.Linear(
                self.hidden_size * (2 if mconfig.bidirectional_encoder else 1), context_projection)
        self.use_state_projection = False
        if state_projection is not None:
            self.use_state_projection = True
            self.state_projection = nn.Linear(
                self.hidden_size * (2 if mconfig.bidirectional_encoder else 1), state_projection)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.apply(init_weights)

    def forward(self, sequences):
        lens = [len(seq) for seq in sequences]
        bounds = [0]
        for l in lens:
            bounds.append(bounds[-1]+l)
        piled_sequence = torch.cat(sequences)
        piled_embeddings = self.dr(self.lookup(piled_sequence))
        embed_sequences = [piled_embeddings[bounds[i]:bounds[i+1]] for i in range(len(sequences))]
        packed_sequence = rnn.pack_sequence(embed_sequences)

        encoded, last_state = self.lstm(packed_sequence)

        encoded_pad, lengths = rnn.pad_packed_sequence(encoded)

        if not self.use_context_projection:
            context_pad = encoded_pad
        else:
            context_pad = self.act(self.context_projection(self.dr(encoded_pad)))
        state = self.get_decoder_init_state(last_state)
        context = rnn.pack_padded_sequence(context_pad, lengths)

        return context, state

    def encode_one_sent(self, seq):
        embeddings = self.dr(self.lookup(seq)).unsqueeze(1)
        encoded, last_state = self.lstm(embeddings)
        if not self.use_context_projection:
            context = encoded
        else:
            context = self.act(self.context_projection(self.dr(encoded)))
        state = self.get_decoder_init_state(last_state)
        return context, state

    def get_decoder_init_state(self, state):
        if self.lstm.bidirectional:
            last_state = state[0].view(self.lstm.num_layers, 2,
                                       state[0].size(1), -1).transpose(0, 1)
            last_state = torch.cat([last_state[0], last_state[1]], dim=2)
        else:
            last_state = state[0]
        #print(state[0].size(), last_state.size())
        if not self.use_state_projection:
            state = last_state
        else:
            state = self.act(self.state_projection(self.dr(last_state)))
        return state
