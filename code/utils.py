import math
from typing import List
from collections import namedtuple
import configuration
import numpy as np
import torch
from torch.nn import Parameter
import subwords
import re
from sklearn.decomposition import TruncatedSVD

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

vconfig = configuration.VocabConfig()

LINEAR_BACKCOMP_PATTERN_1 = "(de|en)coder.layers.[0-9]+.(self|src)_attn.linears.[0-9]+.(weight|bias)"
LINEAR_BACKCOMP_PATTERN_2 = "(de|en)coder.layers.[0-9]+.feed_forward.w_[12].(weight|bias)"


def zip_data(*args):

    src = args[0]
    tgt = args[1]
    assert len(src) == len(tgt)
    return list(zip(src, tgt))


def read_corpus(file_path, source='src'):
    if vconfig.subwords_source and source == 'src':
        sub = subwords.SubwordReader("src")
    elif vconfig.subwords_target and source == 'tgt':
        sub = subwords.SubwordReader("tgt")
    print(file_path)
    test = "test" in file_path
    data = []
    counter = 0
    for line in open(file_path):
        if vconfig.subwords_source and source == 'src':
            sent = sub.line_to_subwords(line)
        elif vconfig.subwords_target and source == 'tgt':
            sent = sub.line_to_subwords(line)
        else:
            sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == "tgt":
            sent = ['<s>'] + sent + ['</s>']
        if len(sent) <= vconfig.max_len_corpus or test:
            data.append(sent)
        else:
            counter += 1
    print("Eliminated :", counter, "out of", len(data))
    return data


def write_sents(sents, path):
    if vconfig.subwords_target:
        sub = subwords.SubwordReader("tgt")
    with open(path, 'w') as f:
        for sent in sents:
            if vconfig.subwords_target:
                line = sub.subwords_to_line(sent)
            else:
                line = ' '.join(sent)

            f.write(line + '\n')


def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


def load_partial_state_dict(model, state_dict):

    model_state = model.state_dict()
    loaded_keys = []
    unloaded_keys = []
    unseen_keys = []
    for name, param in state_dict.items():
        if name not in model_state:
            if re.fullmatch(LINEAR_BACKCOMP_PATTERN_1, name) or re.fullmatch(LINEAR_BACKCOMP_PATTERN_2, name):
                # backwards compability from standard to potentially-factorized transformer
                name = name.split('.')
                name.insert(-1, "linear")
                name = ".".join(name)
            else:
                unloaded_keys.append(name)
                continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            model_state[name].copy_(param)
        except KeyError:
            unloaded_keys.append(name)
            continue
        except RuntimeError:
            print(param.shape)
            print(model_state[name].shape)
            unloaded_keys.append(name)
            continue
        loaded_keys.append(name)
    for name, param in model_state.items():
        if name not in loaded_keys:
            unseen_keys.append(name)
    if len(unseen_keys) > 0:
        print("{} params not found in file :".format(len(unseen_keys)), unseen_keys)
        print()
    if len(unloaded_keys) > 0:
        print("{} params in file not in model :".format(len(unloaded_keys)), unloaded_keys)
        print()
    if len(unseen_keys) == 0 and len(unloaded_keys) == 0:
        print("Model and file matching !")
        print()


def factorize_tensors(t, n):
    svd = TruncatedSVD(n_components=n, n_iter=10, random_state=1)
    t = t.numpy()
    w = torch.Tensor(svd.fit_transform(t))
    h = torch.Tensor(svd.components_)
    singular_values = svd.singular_values_
    if min(singular_values) > 0.1 * max(singular_values):
        print(max(singular_values))
        print(min(singular_values))
    return w, h
