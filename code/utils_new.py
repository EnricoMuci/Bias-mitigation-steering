import torch
import typing
import warnings

from dialz import SteeringModel
from dialz.vector import SteeringModule, model_layer_list
from transformers import AutoModelForCausalLM  # ,  BitsAndBytesConfig


class QuantizedSteeringModel(SteeringModel):
    def __init__(
            self,
            model_name: str | None,
            layer_ids: typing.Iterable[int],
            model_path: str = None,
            token: str = None,
            quantization_config=None, ):
        # Call nn.Module.__init__() directly, bypassing SteeringModel.__init__()
        torch.nn.Module.__init__(self)
        self.model_name = model_name
        self.token = token

        if model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                # token=token,
                # dtype=torch.float16,
                quantization_config=quantization_config,
                device_map="auto",
            )
        elif model_name is str:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # token=token,
                # dtype=torch.float16,
                quantization_config=quantization_config,
                device_map="auto",
            )

        if quantization_config is None:
            self.model = self.model.to(
                "cuda:0" if torch.cuda.is_available()
                else "mps:0" if torch.backends.mps.is_available()
                else "cpu"
            )

        layers = model_layer_list(self.model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]

        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, SteeringModule):
                layers[layer_id] = SteeringModule(layer)
            else:
                warnings.warn("Trying to rewrap a wrapped model! Try calling .unwrap first.")
