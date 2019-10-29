#!/usr/bin/env python3
import pickle

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

from utils import load_partial_state_dict, Hypothesis
from nmt.nmt import NMTModel
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.optimizer import NoamOpt
import paths
from configuration import TransformerConfig as transformer_config
from configuration import DecodeConfig as decoder_config
from configuration import TrainConfig as train_config


class TransformerModel(NMTModel):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, *args, embedding_rank=None, inner_rank=None, ffward_rank=None, **kwargs):
        # Run super constructor from NMTModel, but don't run NMTModel.__init__
        super(NMTModel, self).__init__()
        self.vocab = pickle.load(open(paths.vocab, 'rb'))

        if embedding_rank is None:
            embedding_rank = transformer_config.embedding_rank
        if inner_rank is None:
            inner_rank = transformer_config.inner_rank
        if ffward_rank is None:
            ffward_rank = transformer_config.ffward_rank
        print(transformer_config.embedding_factorization, transformer_config.inner_factorization, transformer_config.ffward_factorization)
        print(embedding_rank, inner_rank, ffward_rank)
        self.encoder = Encoder(len(self.vocab.src), embedding_rank, inner_rank, ffward_rank)
        self.decoder = Decoder(len(self.vocab.tgt), embedding_rank, inner_rank, ffward_rank)

        self.gpu = False
        self.initialize()

        self.optimizer = NoamOpt(
            transformer_config.layer_dimension,
            train_config.lr,
            4000,
            Adam(
                self.parameters(),
                lr=0,
                betas=(0.9, 0.98),
                eps=1e-9,
            ),
            beginning_step=0
        )

        self.num_accumulations = 0
        self.accumulate = max(1, train_config.accumulate)

    def reset_optimizer(self):
        self.optimizer = NoamOpt(
            transformer_config.layer_dimension,
            1,
            4000,
            Adam(
                self.parameters(),
                lr=0,
                betas=(0.9, 0.98),
                eps=1e-9,
            ),
        )

    def __call__(self, src, tgt, update_params=True):
        "Take in and process masked src and target sequences."
        src_encoding, src_mask = self.encode(src)
        loss, norm = self.decode(
            src_encoding,
            src_mask,
            tgt,
        )

        if update_params:
            self.step(loss)
        if self.gpu:
            loss = loss.cpu()
        return loss.detach().numpy() * norm

    def encode(self, src):
        src_encodings = self.prepare_sents(src, 'src')
        return self.encoder(src_encodings)

    def decode(self, src_encoding, src_mask, tgt):
        tgt_enc = self.prepare_sents(tgt, 'tgt')
        return self.decoder(
            src_encoding,
            src_mask,
            tgt_enc,
        )

    def initialize(self):
        # Initialize parameters with Glorot
        # TODO: Make sure this works correctly
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform(param)

    def update_lr(self, *args, **kwargs):
        """
        Overwrite update_lr needed by other models becuase Transformer is very
        sensitive to hyperparameters and manages its own lr decay
        """
        pass

    @staticmethod
    def load(model_path: str):
        dict_path = model_path + ".dict.pt"
        model = TransformerModel()
        print("Loading whole model")
        load_partial_state_dict(model, torch.load(dict_path))
        return model

    def load_params(self, model_path, no_opt=False):
        dict_path = model_path + ".dict.pt"
        self.load_state_dict(torch.load(dict_path))
        if not no_opt:
            opt_path = model_path + ".opt.pt"
            self.optimizer.load_state_dict(torch.load(opt_path))

    def save(self, path: str, no_opt=False):
        dict_path = path + ".dict.pt"
        torch.save(self.state_dict(), dict_path)
        if not no_opt:
            opt_path = path + ".opt.pt"
            torch.save(self.optimizer.state_dict(), opt_path)

    def beam_search(self, src, max_step=100, replace=False, start_symbol=1):

        if decoder_config.greedy_search:

            batch_size = len(src)
            stop = 2

            inferred = [None for _ in range(batch_size)]
            memory, src_mask = self.encode(src)
            pred_sents = torch.ones(batch_size, 1, dtype=torch.long).fill_(start_symbol)
            scores = np.zeros((batch_size, ))

            if self.gpu:
                pred_sents = pred_sents.cuda()

            for i in range(1, max_step+1):
                out = self.decoder.get_word_scores(memory, src_mask, Variable(pred_sents))

                next_scores, next_words = torch.max(out, dim=1)
                pred_sents = torch.cat([pred_sents, next_words.unsqueeze(1)], dim=1)

                stopped_sentences = np.where(next_words.detach().cpu().numpy() == stop)[0]
                ongoing_sentences = np.where(next_words.detach().cpu().numpy() != stop)[0]

                place_in_inferred(inferred, pred_sents, scores, next_scores, stopped_sentences)
                pred_sents = pred_sents[ongoing_sentences]
                memory = memory[ongoing_sentences]
                src_mask = src_mask[ongoing_sentences]

                if len(ongoing_sentences) == 0:
                    break

            place_in_inferred(inferred, pred_sents, scores, next_scores, np.arange(len(ongoing_sentences)))
            return [[convert_hypothesis(inferred[i], self.vocab, scores[i])] for i in range(batch_size)]

        else:  # beam search

            try:
                memory, src_mask = self.encode(src)
            except IndexError:
                print(src)

            batch_size = len(src)
            beam_size = decoder_config.beam_size
            beam_batch = BeamBatch(batch_size, beam_size, memory, src_mask, self.gpu)

            for i in range(1, max_step+1):
                sizes = beam_batch.get_sizes()
                memory, src_mask = beam_batch.expand_memory_and_mask()
                pred_sents = beam_batch.open_hyps_tensor()

                out = self.decoder.get_word_scores(memory, src_mask, Variable(pred_sents))
                out = out.detach().cpu().numpy()

                next_words = np.argpartition(-out, beam_size-1, axis=-1)[:, :beam_size]
                next_words = torch.LongTensor(next_words).cuda()
                next_words = next_words.view(sum(sizes) * beam_size, 1)
                pred_sents = pred_sents.repeat(beam_size, 1).reshape(beam_size, sum(sizes), -1).transpose(1, 0).reshape(beam_size * sum(sizes), -1)
                pred_sents = torch.cat([pred_sents, next_words], dim=1)

                next_scores = -np.partition(-out, beam_size-1)[:, :beam_size].flatten()
                old_scores = np.array(beam_batch.get_open_scores())
                old_scores = np.repeat(old_scores, beam_size)
                next_scores = score_update(old_scores, next_scores, i)
                beam_batch.update(sizes, next_scores, pred_sents)

                if beam_batch.is_closed():
                    break

            # print([2 in beam_batch.best_results()[i][0] for i in range(batch_size)])
            # print([len(beam_batch.best_results()[i][0]) for i in range(batch_size)])
            return [[convert_hypothesis(beam_batch.best_results()[i][0], self.vocab, beam_batch.best_results()[i][1])] for i in range(batch_size)]

            # for i in range(1, max_step+1):
            #     next_hypotheses = []
            #     all_done = True
            #     for (cur_sent, old_score, is_done) in hypotheses:
            #         if is_done:
            #             next_hypotheses.append((cur_sent, old_score, is_done))
            #             continue
            #         else:
            #             all_done = False
            #             # TODO: create batch tensor
            #         out = self.decoder.get_word_scores(memory, src_mask, Variable(cur_sent))
            #         candidates = np.argpartition(-out.detach().cpu().numpy(), beam_size)[0, :beam_size]
            #         for idx in candidates:
            #             next_hypotheses.append(
            #                 (
            #                     torch.cat([cur_sent, torch.Tensor([[idx]]).type(cur_sent.type())], dim=1),
            #                     score_update(old_score, out[0, idx].item(), i),
            #                     idx == stop,
            #                 )
            #             )
            #     hypotheses = sorted(next_hypotheses, key=lambda hyp: -hyp[1])[:beam_size]
            #     if all_done:
            #         break
            # pred_sents, scores, _ = hypotheses[0]


class Beam:

    def __init__(self, beam_size, start_symbol=1, stop_symbol=2, gpu=True):

        self.beam_size = beam_size
        self.closed_scores = []
        self.open_scores = [0]
        self.closed_hyps = []
        self.open_hyps = [torch.ones(1, dtype=torch.long).fill_(start_symbol)]
        self.stop = stop_symbol

        if gpu:
            self.open_hyps = [hyp.cuda() for hyp in self.open_hyps]

    def get_size(self):
        return len(self.open_hyps)

    def update(self, new_scores, new_hyps):

        beam_size = self.beam_size
        n_closed = len(self.closed_scores)
        all_scores = np.array(self.closed_scores + list(new_scores))
        candidate_indices = np.argpartition(-all_scores, beam_size-1)[:beam_size]
        candidates = [(self.closed_hyps[i] if i < n_closed else new_hyps[i - n_closed], all_scores[i]) for i in candidate_indices]

        self.open_scores = []
        self.closed_scores = []
        self.open_hyps = []
        self.closed_hyps = []

        for candidate, score in candidates:
            if candidate[-1].item() == self.stop:
                self.closed_hyps.append(candidate)
                self.closed_scores.append(score)
            else:
                self.open_hyps.append(candidate)
                self.open_scores.append(score)

    def best_result(self):

        try:
            open_argmax = np.argmax(self.open_scores)
            open_max = self.open_scores[open_argmax]
        except ValueError:
            open_argmax = None
            open_max = -np.inf
        try:
            closed_argmax = np.argmax(self.closed_scores)
            closed_max = self.closed_scores[closed_argmax]
        except ValueError:
            closed_argmax = None
            closed_max = -np.inf
        if open_max > closed_max:
            return self.open_hyps[open_argmax], open_max
        else:
            return self.closed_hyps[closed_argmax], closed_argmax

    def open_hyps_tensor(self):
        return torch.stack(self.open_hyps, dim=0)

    def is_open(self):
        return len(self.open_hyps) != 0

class BeamBatch:

    def __init__(self, batch_size, beam_size, memory, src_mask, gpu=True, start_symbol=1, stop_symbol=2):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.beams = [Beam(beam_size, start_symbol, stop_symbol, gpu) for _ in range(batch_size)]
        self.memory = memory
        self.mask = src_mask
        self.gpu = gpu

    def get_sizes(self):
        return [beam.get_size() for beam in self.beams]

    def get_open_scores(self):
        return [score for beam in self.beams for score in beam.open_scores]

    def update(self, sizes, new_scores, new_hyps):
        lower_bound = 0
        for i, size in enumerate(sizes):
            if size == 0:
                continue
            upper_bound = lower_bound + size * self.beam_size
            self.beams[i].update(new_scores[lower_bound:upper_bound], new_hyps[lower_bound:upper_bound])
            lower_bound = upper_bound

    def expand_memory_and_mask(self):
        sizes = self.get_sizes()
        expanded_memory = [self.memory[i].expand(sizes[i], -1, -1) for i in range(self.batch_size)]
        expanded_mask = [self.mask[i].expand(sizes[i], -1, -1) for i in range(self.batch_size)]
        return torch.cat(expanded_memory, dim=0), torch.cat(expanded_mask, dim=0)

    def open_hyps_tensor(self):
        return torch.cat([beam.open_hyps_tensor() for beam in self.beams if beam.is_open()], dim=0)

    def is_closed(self):
        return sum(self.get_sizes()) == 0

    def best_results(self):
        return [beam.best_result() for beam in self.beams]


def place_in_inferred(inferred, pred_sents, scores, next_scores, stopped_sentences):
    current_stop = 0
    leaping_index = 0
    next_scores = next_scores.detach().cpu().numpy()

    for i, content in enumerate(inferred):

        if content is not None:
            continue

        if current_stop < len(stopped_sentences) and leaping_index == stopped_sentences[current_stop]:
            inferred[i] = pred_sents[leaping_index].cpu().numpy()
            current_stop += 1

        scores[i] += next_scores[leaping_index]

        leaping_index += 1


def convert_hypothesis(pred_sent, vocab, score):
    tgt_sent = []
    for word_id in pred_sent[1:-1]:
        tgt_sent.append(vocab.tgt.id2word[word_id.item()])
    if decoder_config.greedy_search:
        score = (score - np.log(len(pred_sent)))
    return Hypothesis(tgt_sent, score)


def score_update(old_score, new_score, timestep):
    # return old_score + new_score
    return old_score * (timestep - 1) / timestep + new_score / timestep
