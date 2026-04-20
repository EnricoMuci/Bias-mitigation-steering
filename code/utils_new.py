import torch
import typing
import warnings

from dialz import SteeringModel
from dialz.vector import SteeringModule, model_layer_list
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def get_short_name(model_name):
    # Map model names to short names
    model_short_names = {
        "Qwen/Qwen2.5-7B-Instruct": "qwen",
        "meta-llama/Llama-3.1-8B-Instruct": "llama",
        "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
    }

    model_short_name = model_short_names.get(model_name)
    if not model_short_name:
        raise ValueError(f"Unknown model name: {model_name}")
    else:
        return model_short_name


class QuantizedSteeringModel(SteeringModel):
    def __init__(
            self,
            model_name: str,
            layer_ids: typing.Iterable[int],
            model_path: str = None,
            token: str = None,
            quantization_config=None, ):
        # Call nn.Module.__init__() directly, bypassing SteeringModel.__init__()
        torch.nn.Module.__init__(self)
        self.model_name = model_name
        self.token = token

        if model_path is not None:
            print('Caso giusto')
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


def create_quantized_model(model_name, model_path):
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return QuantizedSteeringModel(
            model_path=model_path, layer_ids=[5],
            model_name=model_name, quantization_config=bnb_config)
    except Exception as e:
        return SteeringModel(model_path, [5])  # Second element is arbritary as we're not generating yet
