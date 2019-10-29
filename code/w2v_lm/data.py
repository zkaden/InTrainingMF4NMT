import configuration
import paths

import os
import xml.etree.ElementTree as ET
import pickle
import numpy as np

from nltk.corpus import stopwords
from vocab import Vocab, VocabEntry

cconfig = configuration.CBOWConfig()


def read_SimVerb(test=False):
    if test:
        path = paths.simverb_test
    else:
        path = paths.simverb_dev

    X = []
    Y = []
    with open(path, 'r') as f:
        for line in f:
            elts = line.split('\t')
            # print(elts)
            X.append((elts[0], elts[1]))
            Y.append(float(elts[3]))
    return X, Y


def test_sim_data(X, Y, vocab):
    assert len(X) == len(Y)
    in_vocab = 0
    for i in range(len(X)):
        if X[i][0] in vocab.src:
            in_vocab += 1
        if X[i][1] in vocab.src:
            in_vocab += 1
    print("In vocab :", in_vocab/(2*len(X)))


def find_rec(node, element, result):
    for item in node.findall(element):
        result.append(item)
    for child in node:
        if child.tag != element:
            find_rec(child, element, result)
    return result


def extract_BNC():
    stpwd = stopwords.words('english')
    stpwd = [w.upper() for w in stpwd]
    print(stpwd)
    vocab_path = paths.vocab_bnc
    source_path = paths.bnc_folder
    target_path = paths.bnc_extracted
    ndocs = 0
    dataset = []
    for (dirpath, dirnames, filenames) in os.walk(source_path):
        for fn in filenames:
            if fn.endswith(".xml"):
                print(dirpath, fn)
                fp = dirpath+"/"+fn
                tree = ET.parse(fp).getroot()
                sentences = find_rec(tree, "s", [])
                print(len(sentences))
                for sent in sentences:
                    st = []
                    for word in sent.findall("w"):
                        w = word.attrib["hw"]
                        pos = word.attrib["pos"]
                        if w.upper() not in stpwd:
                            st.append(w+"_"+pos)
                    st = ["<s>"] * cconfig.context_size + st + ["<s>"] * cconfig.context_size
                    dataset.append(st)
                ndocs += 1
        if ndocs > 1000:
            break

    vocab = VocabEntry.from_corpus(
        dataset, size=cconfig.vocab_size, freq_cutoff=cconfig.freq_cutoff)
    pickle.dump(vocab, open(vocab_path, 'wb'))
    processed_dataset = []
    for sent in dataset:
        np_sent = np.array([vocab[w] for w in sent if vocab[w] != 3])
        processed_dataset.append(np_sent)
    processed_dataset = np.array(processed_dataset)
    np.save(target_path, processed_dataset)
    print(len(processed_dataset))
    print(sum([len(s) for s in processed_dataset]))
