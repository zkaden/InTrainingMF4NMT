from nmt.nmtmodel import NMTModel
from transformer.model import TransformerModel
import configuration
from typing import List, Tuple
import torch
from utils import load_partial_state_dict
cconfig = configuration.CompressionConfig()
gconfig = configuration.GeneralConfig()


class PrunedModel(TransformerModel):
    def __init__(self, *args, embedding_rank=None, inner_rank=None, ffward_rank=None, **kwargs):
        if gconfig.model == "transformer":
            TransformerModel.__init__(self, *args, embedding_rank, inner_rank, ffward_rank, **kwargs)
        else:
            super(PrunedModel, self).__init__(*args, **kwargs)
        self.currently_pruned = False
        parameter_keys = [param[0] for param in self.named_parameters()]
        self.parameter_prefixes = list(
            set([name.split(".")[0] for name in parameter_keys]+[".".join(name.split(".")[:2]) for name in parameter_keys]+[".".join(name.split(".")[:-1]) for name in parameter_keys]))

    def extract_parameters_from_keys(self, keys):
        if keys == "all":
            to_be_pruned = self.parameters()
        elif isinstance(keys, str):
            assert keys in self.parameter_prefixes, "Invalid parameter key"
            to_be_pruned = [p for n, p in self.named_parameters(
            ) if n.startswith(keys)]
        else:
            assert isinstance(keys, Tuple)
            to_be_pruned = set()
            for name in keys:
                assert name in self.parameter_prefixes, "Invalid parameter key"
                to_be_pruned = to_be_pruned | set([
                    p for n, p in self.named_parameters() if n.startswith(name)])
        to_be_pruned = list(to_be_pruned)
        return to_be_pruned

    def prune(self):
        to_be_pruned = self.extract_parameters_from_keys(cconfig.pruning_where)

        if cconfig.pruning_scheme == "uniform":
            for param in to_be_pruned:
                tensor = param.data
                to_prune = int(cconfig.pruning*tensor.nelement())
                new_tensor = PrunedModel.prune_tensors(tensor, to_prune)
                param.data = new_tensor
                param = param.detach()

        elif cconfig.pruning_scheme == "blind":
            tensors = [param.data for param in to_be_pruned]
            to_prune = int(cconfig.pruning*sum([tens.nelement() for tens in tensors]))
            new_tensors = PrunedModel.prune_tensors(tensors, to_prune)
            for i, param in enumerate(to_be_pruned):
                param.data = new_tensors[i]
                param = param.detach()
        else:
            assert cconfig.pruning_scheme == "distribution"
            tensors = [param.data for param in to_be_pruned]
            stds = [tens.std() for tens in tensors]
            to_prune_global = int(cconfig.pruning*sum([tens.nelement() for tens in tensors]))
            lbd = to_prune_global / sum(stds)
            to_prune_indiv = [s*lbd for s in stds]
            for i, param in enumerate(to_be_pruned):
                param.data = PrunedModel.prune_tensors(tensors[i], to_prune_indiv[i])
                param = param.detach()
        self.currently_pruned = True

    @staticmethod
    def prune_tensors(t, n):
        if isinstance(t, torch.Tensor):
            saved_size = t.size()
            flattened = t.view(-1).clone()
        else:
            saved_sizes = [tens.size() for tens in t]
            saved_bounds = [0]
            for tens in t:
                saved_bounds.append(saved_bounds[-1]+tens.nelement())
            flattened = torch.cat([tens.view(-1) for tens in t]).clone()
            # print(len(flattened), n)

        prune_index = torch.topk(torch.abs(flattened), n, largest=False, sorted=True)[1]
        flattened[prune_index] = 0

        if isinstance(t, torch.Tensor):
            return flattened.view(saved_size).contiguous()
        else:
            new_tensors = []
            for i in range(len(saved_sizes)):
                new_tensors.append(
                    flattened[saved_bounds[i]:saved_bounds[i+1]].view(saved_sizes[i]).contiguous())
            return new_tensors

    @staticmethod
    def load(model_path: str):
        dict_path = model_path+".dict.pt"
        model = PrunedModel()
        print("Loading whole model")
        load_partial_state_dict(model, torch.load(dict_path))
        if ".pruned" in model_path:
            model.currently_pruned = True
        else:
            model.currently_pruned = False

        return model

    def load_params(self, model_path, **kwargs):
        if ".pruned" in model_path:
            assert self.currently_pruned, "Model is not pruned but path is to a pruned model"
        super(PrunedModel, self).load_params(model_path, **kwargs)

    def step(self, loss):
        super(PrunedModel, self).step(loss)
        self.currently_pruned = False

    def save(self, path, **kwargs):
        if ".pruned" in path:
            assert self.currently_pruned, "Model is not pruned but path is to a pruned model"
        else:
            assert not self.currently_pruned, "Model is pruned but path is to a not pruned model"
        super(PrunedModel, self).save(path, **kwargs)
