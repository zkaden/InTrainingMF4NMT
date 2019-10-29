import torch.nn as nn
import torch
import torch.nn.utils.rnn as rnn
import numpy as np

import configuration
from nmt.layers import MultipleLSTMCells, init_weights, FactorizedEmbeddings

mconfig = configuration.LSTMConfig()
dconfig = configuration.DecodeConfig()


class Decoder(nn.Module):

    def __init__(self, vocab_size, context_projection=None):
        super(Decoder, self).__init__()
        self.act = nn.Tanh()
        self.hidden_size = mconfig.hidden_size_decoder
        self.rank = mconfig.rank_decoder
        self.embed_size = mconfig.embed_size
        self.lookup = FactorizedEmbeddings(vocab_size, self.embed_size, self.rank)
        self.use_context_projection = False
        self.context_size = mconfig.hidden_size_encoder * \
            (2 if mconfig.bidirectional_encoder else 1)
        if context_projection is not None:
            self.use_context_projection = True
            self.att_layer = nn.Linear(self.hidden_size, context_projection)
            self.context_size = context_projection

        self.nlayers = mconfig.num_layers_decoder
        self.dr = nn.Dropout(mconfig.dropout_layers)
        self.final_layers_input = self.hidden_size
        self.input_lstm_size = mconfig.embed_size
        # Attenton
        self.input_lstm_size = mconfig.embed_size + self.context_size

        self.final_layers_input = self.hidden_size + self.context_size

        lstm_sizes = [self.input_lstm_size] + [self.hidden_size for i in range(self.nlayers)]
        self.lstm = MultipleLSTMCells(self.nlayers, lstm_sizes, residual=mconfig.residual,
                                      dropout_vertical=mconfig.dropout_layers, dropout_horizontal=mconfig.dropout_lstm_states)

        self.score_input = self.final_layers_input
        self.has_output_layer = mconfig.has_output_layer or (self.score_input != mconfig.embed_size)
        if self.has_output_layer:
            self.score_input = mconfig.embed_size
            self.output_layer = nn.Linear(self.final_layers_input, mconfig.embed_size)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        self.apply(init_weights)

    def get_params_but_embedding(self):
        return list(self.lstm.parameters())+(list(self.output_layer.parameters()) if self.has_output_layer else []) + list(self.att_layer.parameters() if self.use_context_projection else [])

    def get_word_scores(self, decs, contexts):
        outs = torch.cat([decs, contexts], dim=1)
        h = outs
        if self.has_output_layer:
            h = self.act(self.output_layer(self.dr(h)))
        return self.lookup.reverse_embeddings(h)

    def get_init_state(self, encoder_state):
        # init_state = encoder_state.new_full((self.nlayers, encoder_state.size(
        #    0), self.hidden_size), 0), encoder_state.new_full((self.nlayers, encoder_state.size(0), self.hidden_size), 0)
        init_state = encoder_state, encoder_state.new_full(
            (self.nlayers, encoder_state.size(1), self.hidden_size), 0)
        #init_state[0] = encoder_state
        # print(init_state[0].size())

        return init_state

    def one_step_decoding(self, input, state, encoded, attention_mask=None, attend=True, replace=False):
        o = input
        # print(state[0].size())
        h, new_state = self.lstm(o, state)
        o = h
        # attention
        context = None
        if attend:
            if self.use_context_projection:
                query = self.act(self.att_layer(o))
            else:
                query = o
            att_scores = torch.bmm(encoded, query.unsqueeze(2))
            if attention_mask is not None:
                att_scores.masked_fill_(attention_mask, float("-inf"))

            att_scores = nn.Softmax(dim=1)(att_scores)

            context = torch.bmm(encoded.transpose(1, 2), att_scores).squeeze(2)
            val_sort, arg_sort = torch.sort(att_scores, dim=1, descending=True)
        else:
            context = o.new_full((o.size(0), self.context_size), 0)

        if replace:
            return o, context, new_state, arg_sort[0][0]
        else:
            return o, context, new_state

    def forward(self, encoded, encoder_state, tgt_sequences, attend=True):
        # Embedding
        lens = [len(seq) for seq in tgt_sequences]
        bounds = [0]
        for l in lens:
            bounds.append(bounds[-1] + l)
        piled_sequence = torch.cat(tgt_sequences)
        piled_embeddings = self.dr(self.lookup(piled_sequence))
        embed_sequences = [piled_embeddings[bounds[i]:bounds[i + 1]]
                           for i in range(len(tgt_sequences))]

        padded_tgt_sequence = rnn.pad_sequence(embed_sequences)

        if attend:
            padded_enc, lengths_enc = rnn.pad_packed_sequence(encoded)
            encoder_outputs = padded_enc.transpose(0, 1)
            attention_mask = encoder_outputs.new_full(
                (encoder_outputs.size(0), encoder_outputs.size(1), 1), 0, dtype=torch.uint8)
            for i in range(len(lengths_enc)):
                attention_mask[i, lengths_enc[i]:] = 1
            state = self.get_init_state(encoder_state)
        else:
            attention_mask = None
            encoder_outputs = None
            state = piled_embeddings.new_full((self.nlayers, len(tgt_sequences), self.hidden_size),
                                              0), piled_embeddings.new_full(
                (self.nlayers, len(tgt_sequences), self.hidden_size), 0)

        output_list = []
        prev_context_vector = state[0][-1].new_full((state[0][-1].size(0), self.context_size), 0)
        context_list = []
        self.lstm.sample_masks()
        for i in range(len(padded_tgt_sequence)):
            input = torch.cat([padded_tgt_sequence[i], prev_context_vector], dim=1)
            output, new_context_vector, state = self.one_step_decoding(
                input, state, encoder_outputs, attention_mask, attend=attend)
            output_list.append(output.unsqueeze(0))

            context_list.append(new_context_vector.unsqueeze(0))
            prev_context_vector = new_context_vector

        padded_dec = torch.cat(output_list, dim=0)
        list_dec = [padded_dec.transpose(0, 1)[i, :lens[i] - 1]
                    for i in range(len(tgt_sequences))]
        piled_dec = torch.cat(list_dec)

        padded_contexts = torch.cat(context_list, dim=0)
        list_contexts = [padded_contexts.transpose(0, 1)[i, :lens[i] - 1]
                         for i in range(len(tgt_sequences))]
        piled_contexts = torch.cat(list_contexts)

        # Scoring
        piled_scores = self.get_word_scores(piled_dec, piled_contexts)
        targets = [tgt_sequences[i][1:] for i in range(len(lens))]
        labels = torch.cat(targets)
        loss = self.criterion(piled_scores, labels)
        return loss

    def greedy_search(self, encoded_sequence, init_state, max_step=None, replace=False):
        if max_step is None:
            max_step = dconfig.max_decoding_time_step
        tgt_ids = []
        time_step = 0
        state = self.get_init_state(init_state)

        start = torch.LongTensor([1])
        stop = torch.LongTensor([2])
        score = 0
        if self.cuda:
            start = start.cuda()
            stop = stop.cuda()
        current_word = start
        tgt_ids.append(current_word)

        prev_context_vector = state[0][-1].new_full((state[0][-1].size(0), self.context_size), 0)
        encoder_outputs = encoded_sequence.transpose(0, 1)
        while (time_step < max_step and current_word[0] != stop[0]):
            time_step += 1
            embed = self.dr(self.lookup(current_word))
            input = torch.cat([embed, prev_context_vector], dim=1)

            if replace:
                dec, new_context_vector, state, attended = self.one_step_decoding(
                    input, state, encoder_outputs, replace=replace)
            else:
                dec, new_context_vector, state = self.one_step_decoding(
                    input, state, encoder_outputs, replace=replace)

            prev_context_vector = new_context_vector

            scores = self.get_word_scores(dec, new_context_vector)
            current_score, current_word = torch.max(scores, dim=1)

            if replace and current_word.item() == 3:
                # make sure most attended word isn't unk
                translated_word = -attended
            else:
                translated_word = current_word

            tgt_ids.append(translated_word)
            score += current_score.detach().cpu().numpy()

        if current_word != stop:
            tgt_ids.append(stop)
        score = (score - np.log(time_step)).item()
        return torch.cat(tgt_ids), score

    def beam_search(self, encoded_sequence, init_state, beam_size, max_step=None, replace=False):
        if max_step is None:
            max_step = dconfig.max_decoding_time_step
        if beam_size is None:
            beam_size = dconfig.beam_size
        start = torch.LongTensor([1])
        stop = torch.LongTensor([2])
        if self.cuda:
            start = start.cuda()
            stop = stop.cuda()
        state = self.get_init_state(init_state)
        time_step = 0
        prev_context_vector = state[0][-1].new_full((state[0][-1].size(0), self.context_size), 0)
        encoder_outputs = encoded_sequence.transpose(0, 1)
        hypotheses = [([start], state, prev_context_vector, 0)]

        while (time_step < max_step):

            time_step += 1
            next_hypotheses = []
            stopped = True

            for hypothesis in hypotheses:
                current_sentence = hypothesis[0]
                if current_sentence[-1] >= 0:
                    current_word = current_sentence[-1]
                else:
                    current_word = torch.cuda.LongTensor([3])
                if current_word == stop:
                    next_hypotheses.append(hypothesis)
                    continue
                else:
                    stopped = False
                current_state = hypothesis[1]
                current_score = hypothesis[3]

                embed = self.dr(self.lookup(current_word))
                input = torch.cat([embed, hypothesis[2]], dim=1)

                if replace:
                    dec, new_context_vector, current_state, attended = self.one_step_decoding(
                        input, current_state, encoder_outputs, replace=replace)
                else:
                    dec, new_context_vector, current_state = self.one_step_decoding(
                        input, current_state, encoder_outputs, replace=replace)

                scores = self.get_word_scores(dec, new_context_vector)
                probs = -nn.LogSoftmax(dim=1)(scores).squeeze()
                probs = probs.detach().cpu().numpy()
                max_indices = np.argpartition(probs, beam_size - 1)[:beam_size]
                if replace:
                    max_indices[max_indices == 3] = -attended.item()
                next_hypothesis = []
                for i in max_indices:
                    if i < 0:
                        prob_idx = 3
                    else:
                        prob_idx = i

                    next_hypothesis.append(
                        (
                            current_sentence + [torch.cuda.LongTensor([i])],
                            current_state,
                            new_context_vector,
                            score_update(current_score, probs[prob_idx], time_step)
                        )
                    )
                next_hypotheses.extend(next_hypothesis)
            if stopped:
                break

            beam_scores = np.array([hypothesis[3] for hypothesis in next_hypotheses])
            beam_indices = np.argpartition(beam_scores, beam_size - 1)[:beam_size]
            hypotheses = [next_hypotheses[j] for j in beam_indices]

        return sorted([(torch.cat(hypothesis[0]), hypothesis[3]) for hypothesis in hypotheses], key=lambda x: x[1])


def score_update(old_score, update, time_step):

    return old_score * (time_step - 1) / time_step + update / time_step
