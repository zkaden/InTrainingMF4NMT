from w2v_lm.data import read_SimVerb, test_sim_data, extract_BNC
from scipy.stats.stats import pearsonr, spearmanr
from w2v_lm.cbow import CBOWModel, train_cbow, preprocess_dataset, get_model_outputs
from utils import read_corpus
from vocab import Vocab, VocabEntry
import paths
import configuration
from scipy.spatial.distance import cosine
import torch
import pickle
import numpy as np

cconfig = configuration.CBOWConfig()


def correlation(X, Y):
    indices = [i for i in range(len(X)) if X[i] != None]

    x1 = [X[i] for i in indices]
    x2 = [Y[i] for i in indices]
    print(min(x1), max(x1))
    return spearmanr(x1, x2), pearsonr(x1, x2)


def test_dataset():
    X, Y = read_SimVerb()

    vocab = pickle.load(open(paths.vocab, 'rb'))
    test_sim_data(X, Y, vocab)


def test_model():
    vocab = pickle.load(open(paths.vocab_bnc, 'rb'))
    path = paths.model_embedding
    model = CBOWModel(len(vocab), cconfig.embed_size, cconfig.context_size, cconfig.hidden_size)
    d = torch.load(path+".dict.pt")
    model.load_state_dict(d)
    X, Y = read_SimVerb(True)
    X = [(vocab[u+"_VERB"], vocab[v+"_VERB"]) for u, v in X]
    res = get_model_outputs(model, X, True)

    corr = correlation(res, Y)
    print("Spearman correlation", corr[0][0])
    print("p-value", corr[0][1])
    print("Pearson correlation", corr[1][0])
    print("p-value", corr[1][1])


def train_model():
    vocab = pickle.load(open(paths.vocab_bnc, 'rb'))
    path = paths.model_embedding
    model = CBOWModel(len(vocab), cconfig.embed_size, cconfig.context_size, cconfig.hidden_size)
    if cconfig.load:
        try:
            d = torch.load(path+".dict.pt")
            model.load_state_dict(d)
        except:
            print("Unable to load model")
    data = np.load(paths.bnc_extracted)
    data = preprocess_dataset(data)
    print("Data processed")
    l = len(data)
    ds = l//1000 - ((l//1000) % cconfig.batch_size)
    train_data = data[:-ds]
    dev_data = data[-ds:]

    for i in range(cconfig.n_epochs):
        train_cbow(model, train_data, dev_data)
        torch.save(model.state_dict(), path+".dict.pt")


def read_pretrainedw2w():
    path = paths.pretrainedw2v
    with open(path, 'r') as f:
        id2words = []
        data = []
        i = 0
        for line in f:
            if i > 0:
                elts = line.split(' ')
                id2words.append(elts[0])
                data.append(np.array([float(s) for s in elts[1:]]))
            i += 1

        data = np.array(data)
        word2id = dict([(w, i) for i, w in enumerate(id2words)])
    return word2id, id2words, data


def test_pretrained_model():
    word2id, id2words, data = read_pretrainedw2w()
    X, Y = read_SimVerb(True)
    res = []
    nncount = 0
    for i in range(len(X)):
        x1 = word2id[X[i][0]+"_VERB"] if X[i][0]+"_VERB" in id2words else None
        x2 = word2id[X[i][1]+"_VERB"] if X[i][1]+"_VERB" in id2words else None
        if (x1 is not None and x2 is not None):

            res.append(1-cosine(data[x1], data[x2]))
        else:
            res.append(None)
            nncount += 1

    corr = correlation(res, Y)
    print("Correlation", corr[0])
    print("p-value", corr[1])
    print("Missing pairs", str(nncount)+str("/")+str(len(res)))


def main():
    torch.manual_seed(2000)
    # extract_BNC()
    # train_model()
    test_model()
    # test_pretrained_model()


if __name__ == "__main__":
    main()
