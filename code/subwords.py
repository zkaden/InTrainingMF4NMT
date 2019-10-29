import sentencepiece as spm

import configuration
import paths
import os

vconfig = configuration.VocabConfig()


def train_model(train_source_path, target):
    if vconfig.joint:
        vocab_size = str(vconfig.subwords_joint_vocab_size)
        train_source_path = train_source_path[:-2] + 'joint'
    else:
        vocab_size = str(vconfig.subwords_vocab_size)
    model_type = vconfig.subwords_model_type
    print("Train subwords model")

    spm.SentencePieceTrainer.Train('--input='+train_source_path +
                                   ' --model_prefix='+target+' --model_type='+model_type+' --character_coverage=1.0 --vocab_size='+vocab_size)


def train(mode):
    if mode == "src":
        target_folder = paths.data_subwords_folder
        model_type = vconfig.subwords_model_type
        model_prefix = target_folder + model_type + ".en"
        path = paths.get_data_path(set="train", mode="src")
        train_source_path = path

    else:
        assert mode == "tgt"
        target_folder = paths.data_subwords_folder
        model_type = vconfig.subwords_model_type
        model_prefix = target_folder + model_type + ".de"
        path = paths.get_data_path(set="train", mode="tgt")
        train_source_path = path

    train_model(train_source_path, model_prefix)


class SubwordReader:
    def __init__(self, mode):
        folder = paths.data_subwords_folder
        model_type = vconfig.subwords_model_type
        if mode == "src":
            model_prefix = folder + model_type + ".en"
        else:
            assert mode == "tgt"
            model_prefix = folder + model_type + ".de"
        model_path = model_prefix + ".model"

        self.sp = spm.SentencePieceProcessor()
        print("Loading subword model :", model_path)
        self.sp.Load(model_path)

    def line_to_subwords(self, line):
        return self.sp.EncodeAsPieces(line)

    def subwords_to_line(self, l):
        return self.sp.DecodePieces(l)


def main():
    if not vconfig.load_subwords:
        smth = False
        if vconfig.subwords_source:
            train("src")
            smth = True

        if vconfig.subwords_target:
            train("tgt")
            smth = True
        if not smth:
            print("No subwords")
    else:
        print("Loading subwords")


if __name__ == '__main__':
    main()
