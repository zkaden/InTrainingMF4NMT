from nmt.nmtmodel import NMTModel
from configuration import GeneralConfig as gconfig, CompressionConfig as cconfig, TransformerConfig as tconfig
from typing import List, Tuple
import torch
from utils import load_partial_state_dict
from compression.prunedmodel import PrunedModel
import numpy as np

if gconfig.mode == "transformer":
    from transformer.layers import TrFactorizedEmbeddings as FactorizedEmbeddings, FactorizedLinear
else:
    from nmt.layers import FactorizedEmbeddings as FactorizedEmbeddings
    from transformer.layers import FactorizedLinear


class PostFactorizedModel(PrunedModel):
    def __init__(self, *args, embedding_rank=None, inner_rank=None, ffward_rank=None, **kwargs):
        super(PostFactorizedModel, self).__init__(*args, embedding_rank, inner_rank, ffward_rank, **kwargs)
        self.currently_factorized = []
        parameter_keys = [param[0] for param in self.named_parameters()]
        self.parameter_prefixes = list(
            set([name.split(".")[0] for name in parameter_keys]+[".".join(name.split(".")[:2]) for name in parameter_keys]+[".".join(name.split(".")[:-1]) for name in parameter_keys]))

    def extract_weight_parameters_from_keys(self, keys):
        if keys == "all":
            to_be_factorized = [(n, p) for n, p in self.named_parameters() if n.endswith("weight")]
        elif isinstance(keys, str):
            assert keys in self.parameter_prefixes, "Invalid parameter key"
            to_be_factorized = [(n, p) for n, p in self.named_parameters() if n.startswith(keys) and n.endswith("weight")]
        else:
            assert isinstance(keys, Tuple)
            to_be_factorized = set()
            for name in keys:
                assert name in self.parameter_prefixes, "Invalid parameter key"
                to_be_factorized = to_be_factorized | set([
                    (n, p) for n, p in self.named_parameters() if n.startswith(name) and n.endswith("weight")])
            to_be_factorized = list(to_be_factorized)
        return to_be_factorized

    def postfactorize(self):

        for layer in self.modules():
            if isinstance(layer, FactorizedEmbeddings):
                print(layer.factorize_layer(0.25))
            elif isinstance(layer, FactorizedLinear):
                print(layer.factorize_layer(0.5))
        self.currently_factorized = ["embeddings", "ffward", "attention"]

    @staticmethod
    def load(model_path: str):
        dict_path = model_path+".dict.pt"
        print("Loading whole model")
        if ".postfactorized" in model_path:
            print("loading factorized model")
            tconfig.embedding_factorization = True
            tconfig.ffward_factorization = True
            tconfig.inner_factorization = True
            tconfig.embedding_rank = 256
            tconfig.ffward_rank = 256
            tconfig.inner_rank = 256
            model = PostFactorizedModel(embedding_rank=256, ffward_rank=256, inner_rank=256)
            model.currently_factorized = ["embeddings", "ffward", "attention"]
        else:
            print("loading standard model")
            model = PostFactorizedModel()
        load_partial_state_dict(model, torch.load(dict_path))

        return model
