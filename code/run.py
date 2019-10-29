# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    run.py train [options]
    run.py decode [options]

Options:
    -h --help                               show this screen.
"""

import warnings
warnings.filterwarnings("ignore")

from docopt import docopt
import numpy as np
import torch
import configuration
from nmt import nmt
from transformer import transformer
from compression import prune, quantize, mixed, postfactorize
from vocab import Vocab, VocabEntry
import paths

gconfig = configuration.GeneralConfig()
cconfig = configuration.CompressionConfig()


def simple_script(args):
    if args['train']:
        load_from = paths.model if gconfig.load else None
        nmt.train(load_from)
        load_from = paths.model
        nmt.decode(load_from)
    elif args['decode']:
        load_from = paths.model
        nmt.decode(load_from)
    else:
        raise RuntimeError(f'invalid command')


def transformer_script(args):
    if args['train']:
        load_from = paths.model if gconfig.load else None
        transformer.train(load_from)
        load_from = paths.model
        transformer.decode(load_from)
    elif args['decode']:
        load_from = paths.model
        transformer.decode(load_from)
    else:
        raise RuntimeError(f'invalid command')


def factorize_script(args):

    if args['train']:
        if cconfig.load_from == "regular":
            load = paths.model
        elif cconfig.load_from == "factorized":
            load = paths.model_postfactorized
        else:
            assert cconfig.load_from == "retrained"
            load = paths.model_postfactorized_retrained

        if cconfig.factorize_model:
            print("Factorizing model")
            postfactorize.postfactorize(load)
            load = paths.model_postfactorized
            print("Factorized.")
        if cconfig.retrain_factorized:
            assert cconfig.factorize_model or cconfig.load_from != "regular", "Need a factorized model to retrain"
            print("Retraining factorized model")
            postfactorize.retrain(load)
            load = paths.model_postfactorized_retrained
            print("Retrained.")
        if cconfig.decode_after:
            assert cconfig.factorize_model or cconfig.load_from != "regular", "Need a factorized or retrained model to decode"
            print("Decoding")
            postfactorize.decode(load)

    elif args['decode']:
        if cconfig.load_from == "factorized":
            load = paths.model_postfactorized
            print("Loading factorized model")
        else:
            assert cconfig.load_from == "retrained", "Need a factorized or retrained model to decode"
            load = paths.model_postfactorized_retrained
            print('Loading retrained model')
        postfactorize.decode(load)

    else:
        raise RuntimeError(f'invalid command')


def prune_script(args):
    if args['train']:
        if cconfig.load_from == "regular":
            load = paths.model
        elif cconfig.load_from == "pruned":
            load = paths.model_pruned
        else:
            assert cconfig.load_from == "retrained"
            load = paths.model_pruned_retrained

        if cconfig.prune_model:
            print("Pruning model")
            prune.prune(load)
            load = paths.model_pruned
            print("Pruned.")
        if cconfig.retrain_pruned:
            assert cconfig.prune_model or cconfig.load_from != "regular", "Need a pruned model to retrain"
            print("Retraining pruned model")
            prune.retrain(load)
            load = paths.model_pruned_retrained
            print("Retrained.")
        if cconfig.decode_after:
            assert cconfig.prune_model or cconfig.load_from != "regular", "Need a pruned or retrained model to decode"
            print("Decoding")
            prune.decode(load)

    elif args['decode']:
        if cconfig.load_from == "pruned":
            load = paths.model_pruned
            print("Loading pruned model")
        else:
            assert cconfig.load_from == "retrained", "Need a pruned or retrained model to decode"
            load = paths.model_pruned_retrained
            print('Loading retrained model')
        prune.decode(load)

    else:
        raise RuntimeError(f'invalid command')


def quantize_script(args):
    if args['train']:
        if cconfig.load_from == "regular":
            load = paths.model
        elif cconfig.load_from == "pruned":
            load = paths.model_pruned
        elif cconfig.load_from == "retrained":
            load = paths.model_pruned_retrained
        else:
            assert cconfig.load_from == "quantized"
            load = paths.model_quantized

        if cconfig.quantize_model:
            print("Quantizing model")
            quantize.quantize(load)
            load = paths.model_quantized
            print("Quantized.")
        if cconfig.decode_after:
            assert cconfig.quantize_model or cconfig.load_from == "quantized", "Need a quantized model to decode"
            print("Decoding")
            quantize.decode(load)

    elif args['decode']:
        assert cconfig.load_from == "quantized", "Need a quantized model to decode"
        load = paths.model_quantized
        print("Loading quantized model")
        quantize.decode(load)

    else:
        raise RuntimeError(f'invalid command')


def mixed_script(args):
    if args['train']:
        load_from = paths.model_mixed if gconfig.load else None
        mixed.train(load_from)
        load_from = paths.model
        mixed.decode(load_from)
    elif args['decode']:
        load_from = paths.model_mixed
        mixed.decode(load_from)
    else:
        raise RuntimeError(f'invalid command')


def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(gconfig.seed)
    np.random.seed(seed * 13 // 7)
    torch.manual_seed(seed * 13 // 7)
    if gconfig.mode == "normal":
        print("Normal mode")
        simple_script(args)
    elif gconfig.mode == "prune":
        print("Pruning mode")
        prune_script(args)
    elif gconfig.mode == "postfactorize":
        print("Factorizing mode")
        factorize_script(args)
    elif gconfig.mode == "quantize":
        print("Quantizing mode")
        quantize_script(args)
    elif gconfig.mode == "transformer":
        print("Transformer model")
        transformer_script(args)
    elif gconfig.mode == "mixed":
        print("Mixed precision mode")
        mixed_script(args)


if __name__ == '__main__':
    main()
