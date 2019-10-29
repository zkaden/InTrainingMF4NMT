#!/usr/bin/env python

from collections import Counter
from itertools import chain
import pickle

from utils import read_corpus

import configuration
import paths
import subwords

vconfig = configuration.VocabConfig()


class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))

        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(
            f'number of word types in aligned data: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]

        for word in top_k_words:
            vocab_entry.add(word)
            if len(vocab_entry) > size:
                break

        return vocab_entry


class Vocab(object):
    def __init__(self, src, tgt, vocab_size, freq_cutoff):
        #assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        self.src = VocabEntry.from_corpus(src, vocab_size, freq_cutoff)

        if isinstance(tgt, list):
            print('initialize target vocabulary ..')
            self.tgt = VocabEntry.from_corpus(tgt, vocab_size, freq_cutoff)
        else:
            self.tgt = tgt

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


if __name__ == '__main__':
    subwords.main()

    sc = paths.get_data_path("train", "src")
    tg = paths.get_data_path("train", "tgt")
    print('read in source sentences: %s' % sc)
    print('read in target sentences: %s' % tg)

    src_sents = read_corpus(sc, source='src')
    tgt_sents = read_corpus(tg, source='tgt')

    vocab = Vocab(src_sents, tgt_sents, vconfig.vocab_size, vconfig.freq_cutoff)
    print('generated vocabulary,  source %d words, target %d words' %
          (len(vocab.src), len(vocab.tgt)))

    pickle.dump(vocab, open(paths.vocab, 'wb'))
    print('vocabulary saved to %s' % paths.vocab)
