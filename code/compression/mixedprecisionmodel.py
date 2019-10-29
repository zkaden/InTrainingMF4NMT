from compression.quantizedmodel import QuantizedModel
from compression.prunedmodel import PrunedModel

import configuration
import torch
import torch.nn as nn
from utils import load_partial_state_dict

cconfig = configuration.CompressionConfig()
tconfig = configuration.TrainConfig()


class MixedPrecisionModel(QuantizedModel):
    def __init__(self, *args, train_in_mixed_precision=False, **kwargs):
        super(MixedPrecisionModel, self).__init__(*args, **kwargs)
        self.all_dtypes = {"float32": torch.float, "float16": torch.half, "int": torch.half}
        self.large_dtype = self.all_dtypes[cconfig.large_dtype]
        self.small_dtype = self.all_dtypes[cconfig.small_dtype]
        self.current_dtype = self.large_dtype
        assert self.large_dtype != self.small_dtype
        self.single_weights_copy = None
        self.copy_built = False

    def reset_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.single_weights_copy, lr=tconfig.lr, weight_decay=tconfig.weight_decay)

    def train(self):
        super(MixedPrecisionModel, self).train()
        assert self.is_quantized, "mixed model has to be trained in quantized state"
        if not self.copy_built:
            self.single_weights_copy = [nn.Parameter(param.detach().clone().to(dtype=self.large_dtype))
                                        for param in self.parameters()]

            self.optimizer = torch.optim.Adam(
                self.single_weights_copy, lr=tconfig.lr, weight_decay=tconfig.weight_decay)

    def step(self, loss):

        loss = loss * cconfig.loss_rescale
        loss.backward()
        for i, p in enumerate(list(self.parameters())):
            # if i == 32:
            #    print(p, self.single_weights_copy[i])
            if p.grad is not None:
                self.single_weights_copy[i].grad = p.grad.to(
                    dtype=self.large_dtype)/cconfig.loss_rescale
        nn.utils.clip_grad_norm(self.single_weights_copy, tconfig.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()  # zero copy
        for i, p in enumerate(list(self.parameters())):
            p.data = self.single_weights_copy[i].data.clone().to(dtype=self.small_dtype)
        self.zero_grad()  # zero actual parameters

    def load_params(self, model_path, **kwargs):
        super(MixedPrecisionModel, self).load_params(model_path, **kwargs)
        for i, p in enumerate(list(self.parameters())):
            self.single_weights_copy[i].data = p.data.to(dtype=self.large_dtype)

    @staticmethod
    def load(model_path: str):
        dict_path = model_path+".dict.pt"
        model = MixedPrecisionModel()
        print("Loading whole model")
        if ".quantized" in model_path:
            model.quantize()
        else:
            model.unquantize()
        load_partial_state_dict(model, torch.load(dict_path))
        if ".pruned" in model_path:
            model.currently_pruned = True
        else:
            model.currently_pruned = False

        return model
