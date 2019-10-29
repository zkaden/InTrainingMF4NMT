from compression.prunedmodel import PrunedModel
import configuration
import torch
from utils import load_partial_state_dict

cconfig = configuration.CompressionConfig()


class QuantizedModel(PrunedModel):
    def __init__(self, *args, **kwargs):
        super(QuantizedModel, self).__init__(*args, **kwargs)
        self.all_dtypes = {"float32": torch.float, "float16": torch.half, "int": torch.half}
        self.large_dtype = self.all_dtypes[cconfig.large_dtype]
        self.small_dtype = self.all_dtypes[cconfig.small_dtype]
        self.current_dtype = self.large_dtype
        assert self.large_dtype != self.small_dtype

    @property
    def is_quantized(self):
        return self.current_dtype == self.small_dtype

    def quantize(self):
        if self.current_dtype != self.small_dtype:
            print("Quantizing model")
            if cconfig.small_dtype == "int":
                print("Approximating quantization by rounding")
                for param in self.parameters():
                    param.data = param.data.round()

            self.current_dtype = self.small_dtype
            self.encoder.to(dtype=self.current_dtype)
            self.decoder.to(dtype=self.current_dtype)
            print("Done")

    def unquantize(self):
        if self.current_dtype != self.large_dtype:
            print("Unquantizing model")
            self.current_dtype = self.large_dtype
            self.encoder.to(dtype=self.current_dtype)
            self.decoder.to(dtype=self.current_dtype)
            print("Done")

    @staticmethod
    def load(model_path: str):
        dict_path = model_path+".dict.pt"
        model = QuantizedModel()
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

    def load_params(self, model_path, **kwargs):
        if ".quantized" in model_path:
            assert self.is_quantized, "Model is not quantized but path is to a quantized model"

        super(QuantizedModel, self).load_params(model_path, **kwargs)

    def step(self, loss):
        assert not self.is_quantized, "In QuantizedModel class, quantized models can't be trained on"
        super(QuantizedModel, self).step(loss)

    def save(self, path, **kwargs):
        if ".quantized" in path:
            assert self.is_quantized, "Model is not quantized but path is to a quantized model"

        super(QuantizedModel, self).save(path, **kwargs)
