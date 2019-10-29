import sys
import time
import os

import numpy as np
import torch
from typing import List
from docopt import docopt

from utils import read_corpus, zip_data, write_sents
from vocab import Vocab, VocabEntry
from compression.mixedprecisionmodel import MixedPrecisionModel
import configuration
from nmt import routine
import paths

gconfig = configuration.GeneralConfig()
tconfig = configuration.TrainConfig()
dconfig = configuration.DecodeConfig()


def train(load_from=None, save_to=None):
    print_file = sys.stderr
    if gconfig.printout:
        print_file = sys.stdout
    train_data_src = read_corpus(paths.train_source, source='src')
    train_data_tgt = read_corpus(paths.train_target, source='tgt')

    dev_data_src = read_corpus(paths.dev_source, source='src')
    dev_data_tgt = read_corpus(paths.dev_target, source='tgt')

    train_data = zip_data(train_data_src, train_data_tgt)
    dev_data = zip_data(dev_data_src, dev_data_tgt)

    train_batch_size = tconfig.batch_size
    valid_niter = gconfig.valid_niter
    log_every = gconfig.log_every
    if save_to is not None:
        model_save_path = save_to
    else:
        model_save_path = paths.model_mixed

    max_epoch = tconfig.max_epoch

    if gconfig.sanity:
        log_every = 1
        train_data = train_data[:150]
        dev_data = dev_data[:150]
        max_epoch = 2
    pretraining = gconfig.pretraining
    pretraining_encoder = gconfig.pretraining_encoder
    if load_from is not None:
        print("Loading from", load_from)
        model = MixedPrecisionModel.load(load_from)
        pretraining = False
        pretraining_encoder = False
    else:
        print("No loading file provided : training from scratch")
        model = MixedPrecisionModel()

    if gconfig.cuda:
        model.to_gpu()
    else:
        print("No cuda support")
    model.quantize()
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    lr = tconfig.lr
    max_patience = tconfig.patience
    max_num_trial = tconfig.max_num_trial
    lr_decay = tconfig.lr_decay

    if pretraining_encoder:
        #print("Pretraining the encoder")
        #pretrain.train_encoder(model, train_data, dev_data)
        print("Pretraining the encoder")
        routine.train_encoder(model, train_data, dev_data, model_save_path,
                              train_batch_size, valid_niter, log_every, tconfig.max_epoch_pretraining_encoder, lr, max_patience, max_num_trial, lr_decay)
        model.reset_optimizer()

    if pretraining:

        print("Pretraining the decoder")
        routine.train_decoder(model, train_data, dev_data, model_save_path,
                              train_batch_size, valid_niter, log_every, tconfig.max_epoch_pretraining, lr, max_patience, max_num_trial, lr_decay)
        model.reset_optimizer()

    model = routine.train_model(model, train_data, dev_data, model_save_path,
                                train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial, lr_decay)
    model.to_cpu()


def decode(load_from=None):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    if gconfig.test:
        data_src = read_corpus(paths.test_source, source='src')
        data_tgt = read_corpus(paths.test_target, source='tgt')
        data_tgt_path = paths.test_target
    else:
        data_src = read_corpus(paths.dev_source, source='src')
        data_tgt = read_corpus(paths.dev_target, source='tgt')
        data_tgt_path = paths.dev_target

    if load_from is not None:
        model_load_path = load_from
    else:
        model_load_path = paths.model_mixed

    print(f"load model from {load_from}", file=sys.stderr)
    model = MixedPrecisionModel.load(model_load_path)
    if gconfig.cuda:
        model.to_gpu()
    model.eval()
    max_step = dconfig.max_decoding_time_step
    if gconfig.sanity:
        max_step = 2

    hypotheses = routine.beam_search(model, data_src, max_step, replace=dconfig.replace)

    lines = []
    for src_sent, hyps in zip(data_src, hypotheses):
        top_hyp = hyps[0]
        lines.append(top_hyp.value)
    write_sents(lines, paths.decode_output)

    bleu_command = "perl scripts/multi-bleu.perl "+data_tgt_path+" < "+paths.decode_output
    os.system(bleu_command)
