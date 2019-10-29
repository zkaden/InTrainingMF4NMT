
import math
import pickle
import sys
import time
import os

import numpy as np
import torch
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import read_corpus, batch_iter, Hypothesis
from vocab import Vocab, VocabEntry
from nmt.nmtmodel import NMTModel
import configuration
import paths

gconfig = configuration.GeneralConfig()


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def train_model(model, train_data, dev_data, model_save_path, train_batch_size=None, valid_niter=None, log_every=None, max_epoch=None, lr=None, max_patience=None, max_num_trial=None, lr_decay=None):
    print_file = sys.stderr
    if gconfig.printout:
        print_file = sys.stdout
    if gconfig.cuda:
        model.to_gpu()
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    while True:
        epoch += 1
        model.train()
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            batch_size = len(src_sents)

            # (batch_size)

            loss = model(src_sents, tgt_sents)

            report_loss += loss
            cum_loss += loss

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                log = 'epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '\
                    'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                       report_loss / report_examples,
                                                                                       math.exp(
                                                                                           report_loss / report_tgt_words),
                                                                                       cumulative_examples,
                                                                                       report_tgt_words /
                                                                                       (time.time(
                                                                                       ) - train_time),
                                                                                       time.time() - begin_time)
                print(log, file=print_file)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                model.eval()
                log = 'epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cumulative_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cumulative_tgt_words),
                                                                                             cumulative_examples)
                print(log, file=print_file)
                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=print_file)

                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=train_batch_size)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=print_file)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' %
                          model_save_path, file=print_file)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                elif patience < max_patience:
                    patience += 1
                    print('hit patience %d' % patience, file=print_file)

                    if patience == max_patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=print_file)
                        if num_trial == max_num_trial:
                            print('early stop!', file=print_file)
                            return model

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * lr_decay

                        print('load previously best model and decay learning rate to %f' %
                              lr, file=print_file)
                        model.load_params(model_save_path)
                        # load model
                        print('restore parameters of the optimizers', file=print_file)

                        model.update_lr(lr)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0
                model.train()

        if epoch == max_epoch:
            print('reached maximum number of epochs!', file=print_file)
            model.load_params(model_save_path)
            model.to_cpu()
            model.eval()
            model.save(model_save_path)
            return model


def train_decoder(model, train_data, dev_data, model_save_path, train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial, lr_decay):
    print_file = sys.stderr
    if gconfig.printout:
        print_file = sys.stdout

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    while True:
        epoch += 1
        model.train()
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            batch_size = len(src_sents)

            # (batch_size)

            loss = model.decode_to_loss(tgt_sents)

            report_loss += loss
            cum_loss += loss

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                log = 'epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '\
                    'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                       report_loss / report_examples,
                                                                                       math.exp(
                                                                                           report_loss / report_tgt_words),
                                                                                       cumulative_examples,
                                                                                       report_tgt_words /
                                                                                       (time.time(
                                                                                       ) - train_time),
                                                                                       time.time() - begin_time)
                print(log, file=print_file)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                model.eval()
                log = 'epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cumulative_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cumulative_tgt_words),
                                                                                             cumulative_examples)
                print(log, file=print_file)
                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=print_file)

                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = model.evaluate_ppl(
                    dev_data, batch_size=train_batch_size, decoder_only=True)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=print_file)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' %
                          model_save_path, file=print_file)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                elif patience < max_patience:
                    patience += 1
                    print('hit patience %d' % patience, file=print_file)

                    if patience == max_patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=print_file)
                        if num_trial == max_num_trial:
                            print('early stop!', file=print_file)
                            return model

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * lr_decay
                        model.update_lr(lr)
                        print('load previously best model and decay learning rate to %f' %
                              lr, file=print_file)

                        # load model

                        print('restore parameters of the optimizers', file=print_file)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0
                model.train()

        if epoch == max_epoch:
            print('reached maximum number of epochs!', file=print_file)
            model.eval()
            model.save(model_save_path)
            return


def train_encoder(model, train_data, dev_data, model_save_path, train_batch_size, valid_niter, log_every, max_epoch, lr, max_patience, max_num_trial, lr_decay):
    print_file = sys.stderr
    if gconfig.printout:
        print_file = sys.stdout

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_src_words = report_src_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')
    while True:
        epoch += 1
        model.train()
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            batch_size = len(src_sents)

            # (batch_size)

            loss = model.encode_to_loss(src_sents)

            report_loss += loss
            cum_loss += loss

            src_words_num_to_predict = sum(len(s[1:]) for s in src_sents)  # omitting leading `<s>`
            report_src_words += src_words_num_to_predict
            cumulative_src_words += src_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                log = 'epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '\
                    'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                       report_loss / report_examples,
                                                                                       math.exp(
                                                                                           report_loss / report_src_words),
                                                                                       cumulative_examples,
                                                                                       report_src_words /
                                                                                       (time.time(
                                                                                       ) - train_time),
                                                                                       time.time() - begin_time)
                print(log, file=print_file)

                train_time = time.time()
                report_loss = report_src_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                model.eval()
                log = 'epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cumulative_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cumulative_src_words),
                                                                                             cumulative_examples)
                print(log, file=print_file)
                cum_loss = cumulative_examples = cumulative_src_words = 0.
                valid_num += 1

                print('begin validation ...', file=print_file)

                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = model.evaluate_ppl(
                    dev_data, batch_size=train_batch_size, encoder_only=True)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=print_file)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' %
                          model_save_path, file=print_file)
                    model.save(model_save_path)

                    # You may also save the optimizer's state
                elif patience < max_patience:
                    patience += 1
                    print('hit patience %d' % patience, file=print_file)

                    if patience == max_patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=print_file)
                        if num_trial == max_num_trial:
                            print('early stop!', file=print_file)
                            return model

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * lr_decay
                        model.update_lr(lr)
                        print('load previously best model and decay learning rate to %f' %
                              lr, file=print_file)

                        # load model

                        print('restore parameters of the optimizers', file=print_file)
                        # You may also need to load the state of the optimizer saved before

                        # reset patience
                        patience = 0
                model.train()

        if epoch == max_epoch:
            print('reached maximum number of epochs!', file=print_file)
            model.eval()
            model.save(model_save_path)
            return


def beam_search(model, test_data_src, max_step=None, replace=False):
    # was_training = model.training

    hypotheses = []
    for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(
            src_sent, max_step=max_step, replace=replace)

        hypotheses.append(example_hyps)
        # print(example_hyps)

    return hypotheses


def batch_beam_search(model, test_data_src, max_step=None, batch_size=64, replace=False):
    # was_training = model.training

    hypotheses = []
    if batch_size != 1:
        test_data_src = [test_data_src[i * batch_size:(i+1) * batch_size] for i in range((len(test_data_src) - 1) // batch_size + 1)]
    for i, src_sent in enumerate(tqdm(test_data_src, desc='Decoding', file=sys.stdout)):
        example_hyps = model.beam_search(
            src_sent, max_step=max_step, replace=replace)

        hypotheses.extend(example_hyps)
        # print(example_hyps)

    return hypotheses
