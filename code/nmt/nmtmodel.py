#!/usr/bin/env python3
import pickle
import time

import numpy as np
from typing import List

import torch
import torch.nn as nn

from utils import Hypothesis, batch_iter, load_partial_state_dict
from nmt.encoder import Encoder
from nmt.decoder import Decoder
import configuration
import paths

mconfig = configuration.LSTMConfig()
tconfig = configuration.TrainConfig()
dconfig = configuration.DecodeConfig()


class NMTModel(nn.Module):

    def forward(self, *args, **kwargs):
        raise("No forward allowed directly on wrapper NMTModel")

    def __init__(self):
        super(NMTModel, self).__init__()
        self.vocab = pickle.load(open(paths.vocab, 'rb'))
        stproj = None if (mconfig.hidden_size_encoder * (2 if mconfig.bidirectional_encoder else 1)
                          == mconfig.hidden_size_decoder) else mconfig.hidden_size_decoder

        self.encoder = Encoder(len(self.vocab.src), context_projection=None,
                               state_projection=stproj)

        self.decoder = Decoder(len(self.vocab.tgt),
                               context_projection=mconfig.hidden_size_decoder)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)
        self.gpu = False
        self.initialize()
        print(len(self.vocab.src))
        # initialize neural network layers...

        self.accumulate = max(1, tconfig.accumulate)
        self.num_accumulations = 0

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=tconfig.lr, weight_decay=tconfig.weight_decay)

    def update_lr(self, lr, encoder=False, decoder=False, **kwargs):
        if encoder:
            opt = self.encoder_optimizer
        elif decoder:
            opt = self.decoder_optimizer
        else:
            opt = self.optimizer
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def to_gpu(self):
        if not self.gpu:
            self.gpu = True
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def to_cpu(self):
        if self.gpu:
            self.gpu = False
            self.encoder = self.encoder.cpu()
            self.decoder = self.decoder.cpu()

    def train(self):
        self.encoder = self.encoder.train()
        self.decoder = self.decoder.train()

    def eval(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()

    def __call__(self, src_sents: List[List[str]], tgt_sents: List[List[str]], update_params=True, **kwargs):
        src_encodings, decoder_init_state = self.encode(src_sents)
        loss = self.decode(src_encodings, decoder_init_state, tgt_sents)

        if update_params:
            self.step(loss)
        if return_attached_loss:
            return loss
        if self.gpu:
            loss = loss.cpu()
        return loss.detach().numpy()

    def encode(self, src_sents: List[List[str]], **kwargs):

        tensor_sents = self.prepare_sents(src_sents, 'src')
        src_encodings, decoder_init_state = self.encoder(tensor_sents)

        return src_encodings, decoder_init_state

    def prepare_sents(self, src_sents: List[List[str]], vocab):
        if vocab == 'src':
            rel_vocab = self.vocab.src
        elif vocab == 'tgt':
            rel_vocab = self.vocab.tgt
        else:
            raise ValueError('Error! Invalid vocabulary')

        np_sents = [np.array([rel_vocab[word] for word in sent]) for sent in src_sents]
        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        return tensor_sents

    def decode(self, src_encodings, decoder_init_state, tgt_sents: List[List[str]]):

        tensor_sents = self.prepare_sents(tgt_sents, 'tgt')
        loss = self.decoder(src_encodings, decoder_init_state, tensor_sents)

        return loss

    def decode_to_loss(self, tgt_sents, update_params=True):
        np_sents = [np.array([self.vocab.tgt[word] for word in sent])
                    for sent in tgt_sents]
        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        loss = self.decoder(None, None, tensor_sents, attend=False)

        if update_params:
            self.step(loss)
        return loss.cpu().detach().numpy()

    def encode_to_loss(self, src_sents, update_params=True, **kwargs):

        np_sents = [np.array([self.vocab.src[word] for word in sent])
                    for sent in src_sents]
        tensor_sents = [torch.LongTensor(s) for s in np_sents]
        if self.gpu:
            tensor_sents = [x.cuda() for x in tensor_sents]
        loss = self.encoder(tensor_sents, to_loss=True)

        if update_params:
            self.step(loss)
        return loss.cpu().detach().numpy()

    def beam_search(self, src_sent: List[str], max_step=None, replace=False, **kwargs) -> List[Hypothesis]:
        np_sent = np.array([self.vocab.src[word] for word in src_sent])
        tensor_sent = torch.LongTensor(np_sent)
        if self.gpu:
            tensor_sent = tensor_sent.cuda()
        src_encoding, decoder_init_state = self.encoder.encode_one_sent(tensor_sent)
        #print(src_sent, np_sent, src_encoding.size())
        if dconfig.greedy_search:
            tgt_tensor, score = self.decoder.greedy_search(
                src_encoding, decoder_init_state, max_step, replace=replace)
            tgt_np = tgt_tensor.cpu().detach().numpy()
            tgt_sent = []
            for i in tgt_np[1:-1]:
                if i >= 0:
                    tgt_sent.append(self.vocab.tgt.id2word[i])
                else:
                    tgt_sent.append(src_sent[-i])
            hypotheses = [Hypothesis(tgt_sent, score)]

        else:
            l = self.decoder.beam_search(
                src_encoding, decoder_init_state, max_step, replace=replace)
            hypotheses = []
            for tgt_tensor, score in l:
                tgt_np = tgt_tensor.cpu().detach().numpy()
                tgt_sent = []
                for i in tgt_np[1: -1]:
                    if i > 0:
                        tgt_sent.append(self.vocab.tgt.id2word[i])
                    else:
                        tgt_sent.append(src_sent[-i])
                hypotheses.append(Hypothesis(tgt_sent, score))
        return hypotheses

    def evaluate_ppl(self, dev_data, batch_size: int=32, encoder_only=False, decoder_only=False, **kwargs):

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`
        if encoder_only:
            cum_src_words = 0.
            for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
                loss = self.encode_to_loss(src_sents, update_params=False)

                src_word_num_to_predict = sum(len(s[1:])
                                              for s in src_sents)  # omitting the leading `<s>`
                cum_src_words += src_word_num_to_predict
                cum_loss += loss
            ppl = np.exp(cum_loss / cum_src_words)

            return ppl

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):

            if decoder_only:
                loss = self.decode_to_loss(tgt_sents, update_params=False)
            else:
                loss = self(src_sents, tgt_sents, update_params=False)
            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:])
                                          for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    def step(self, loss, **kwargs):
        self.num_accumulations += 1
        nn.utils.clip_grad_norm(self.parameters(), tconfig.clip_grad)
        loss = loss / self.accumulate
        loss.backward()
        if self.num_accumulations % self.accumulate == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.num_accumulations = 0

    @staticmethod
    def load(model_path: str):

        dict_path = model_path + ".dict.pt"
        model = NMTModel()
        print("Loading whole model")
        load_partial_state_dict(model, torch.load(dict_path))

        return model

    def initialize(self):

        for param in self.encoder.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)
        for param in self.decoder.parameters():
            torch.nn.init.uniform_(param, -0.1, 0.1)

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
