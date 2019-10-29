import sys
import time
import os

import numpy as np
import torch
from typing import List
from docopt import docopt

from utils import read_corpus, zip_data, write_sents
from vocab import Vocab, VocabEntry
from compression.quantizedmodel import QuantizedModel
import configuration
from nmt import routine
import paths

gconfig = configuration.GeneralConfig()
tconfig = configuration.TrainConfig()
dconfig = configuration.DecodeConfig()
cconfig = configuration.CompressionConfig()


def quantize(load_from, save_to=None):
    print_file = sys.stderr
    if gconfig.printout:
        print_file = sys.stdout
    model_load_path = load_from
    if save_to is None:
        model_save_path = paths.model_quantized
    else:
        model_save_path = save_to

    print("Loading unquantized model from "+model_load_path)
    model = QuantizedModel.load(model_load_path)
    print("Loaded.")
    model.quantize()
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
    model = QuantizedModel.load(load_from)
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
