import sys
import time
import os

import numpy as np
import torch
from typing import List
from docopt import docopt

from utils import read_corpus, zip_data, write_sents
from vocab import Vocab, VocabEntry
from compression.prunedmodel import PrunedModel
import configuration
from nmt import routine
import paths

gconfig = configuration.GeneralConfig()
tconfig = configuration.TrainConfig()
dconfig = configuration.DecodeConfig()
cconfig = configuration.CompressionConfig()


def prune(load_from, save_to=None):
    print_file = sys.stderr
    if gconfig.printout:
        print_file = sys.stdout
    model_load_path = load_from
    if save_to is None:
        model_save_path = paths.model_pruned
    else:
        model_save_path = save_to

    print("Loading unpruned model from "+model_load_path)
    model = PrunedModel.load(model_load_path)
    print("Loaded.")
    print("Pruning model")
    model.prune()
    print("Done.")
    model.save(model_save_path, no_opt=True)
    print("Saved.")
    print("Evaluating on the dev set :")
    dev_data_src = read_corpus(paths.dev_source, source='src')
    dev_data_tgt = read_corpus(paths.dev_target, source='tgt')
    dev_data = zip_data(dev_data_src, dev_data_tgt)
    if gconfig.cuda:
        model.to_gpu()
    else:
        print("No cuda support")
    dev_ppl = model.evaluate_ppl(dev_data, batch_size=tconfig.batch_size)
    print('validation: dev. ppl %f' % dev_ppl, file=print_file)
    model.to_cpu()


def retrain(load_from, save_to=None):
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
    if save_to is None:
        model_save_path = paths.model_pruned_retrained
    else:
        model_save_path = save_to
    model_pruned_path = paths.model_pruned
    max_epoch = cconfig.max_epoch_retraining

    if gconfig.sanity:
        log_every = 1
        train_data = train_data[: 150]
        dev_data = dev_data[: 150]
        max_epoch = 2

    print("Loading from ", load_from)
    model = PrunedModel.load(load_from)
    print("Loaded.")

    if gconfig.cuda:
        model.to_gpu()
    else:
        print("No cuda support")

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    lr = tconfig.lr
    max_patience = tconfig.patience
    max_num_trial = tconfig.max_num_trial
    lr_decay = tconfig.lr_decay

    model = routine.train_model(model, train_data, dev_data, model_save_path,
                                train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial, lr_decay)
    model.to_cpu()


def decode(load_from):
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

    print(f"load model from {load_from}", file=sys.stderr)
    model = PrunedModel.load(load_from)
    if gconfig.cuda:
        model.to_gpu()
    model.eval()
    max_step = dconfig.max_decoding_time_step
    if gconfig.sanity:
        max_step = 2

    hypotheses = routine.batch_beam_search(model, data_src, max_step=max_step, batch_size=dconfig.batch_size, replace=dconfig.replace)

    lines = []
    for src_sent, hyps in zip(data_src, hypotheses):
        top_hyp = hyps[0]
        lines.append(top_hyp.value)
    write_sents(lines, paths.decode_output)

    bleu_command = "perl scripts/multi-bleu.perl "+data_tgt_path+" < "+paths.decode_output
    os.system(bleu_command)
