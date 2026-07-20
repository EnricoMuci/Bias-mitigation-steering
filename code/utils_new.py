import traceback

import torch
import typing
import warnings

from dialz import SteeringModel
from dialz.vector import SteeringModule, model_layer_list
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

import utils
import os

SEED = 38
STRICT_QUANTIZATION = os.environ.get("STRICT_QUANTIZATION", "0") == "1"
REMOTE_DRIVE_THESIS_PROJECT = '/content/drive/MyDrive/ThesisProject'
EXPERIMENT = 'reproduced'  # or 'original
VECTOR_TYPES = ['train', 'train+prompt']

CROWS_AXIS_MAP = {
        'age': 'age',
        'appearance': 'physical-appearance',
        'disability': 'disability',
        'gender': 'gender',
        'nationality': 'nationality',
        'race': 'race-color',
        'religion': 'religion',
        'socioeconomic': 'socioeconomic'
    }

def new_get_args(args_list: list):
    model_name = args_list[0]
    model_path = args_list[1]
    return model_name, model_path


def OLD_get_args(argv):
    # TODO: OLD
    if len(argv) > 2:  # Path and name
        model_name = argv[1]
        model_path = argv[2]
        return model_name, model_path
    elif len(argv) > 1 or argv[2]:  # Only Name
        model_name = argv[1]
        return model_name, model_name
    else:  # Error
        raise ValueError("Model name and model path must be provided as command-line arguments.")


def choose_axes(axes: list | None = None) -> list:
    if axes is not None:
        return axes.copy()  # list type
    else:
        return utils.bbq_axes


def get_model_short_name(model_name):
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

        load_path = model_path if model_path is not None else model_name

        load_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            # Solo senza BnB ha senso specificare dtype e spostare il modello
            load_kwargs["torch_dtype"] = torch.float16  # type: ignore

        self.model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)

        if quantization_config is None:
            self.model = self.model.to(
                "cuda:0" if torch.cuda.is_available()
                else "mps:0" if torch.backends.mps.is_available()
                else "cpu"
            )

        layers = model_layer_list(self.model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]

        # FIXME: TEMPORAL SECTION
        print("Device map:", getattr(self.model, 'hf_device_map', 'N/A'))
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.device}")
            break  # END

        for layer_id in layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, SteeringModule):
                with torch.no_grad():  # FIXME
                    layers[layer_id] = SteeringModule(layer)
            else:
                warnings.warn("Trying to rewrap a wrapped model! Try calling .unwrap first.")


def create_quantized_model(model_name, model_path, layer_ids=None):
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if layer_ids is None:
            layer_ids = [5]
        return QuantizedSteeringModel(
            model_path=model_path, layer_ids=layer_ids,
            model_name=model_name, quantization_config=bnb_config)
    except Exception as e:
        tqdm.write(f"[FALLBACK NON QUANTIZED] {model_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        if STRICT_QUANTIZATION:
            raise
        return SteeringModel(model_path, [5])  # Second element is arbritary as we're not generating yet


def set_steering_layer(model, layer_id):
    """Rewrap an already-loaded QuantizedSteeringModel so SteeringModule is
    active on `layer_id`, unwrapping any previously-active layer first.
    """
    layers = model_layer_list(model.model)
    layer_id = layer_id if layer_id >= 0 else len(layers) + layer_id

    prev_id = getattr(model, "_active_steering_layer_id", None)
    if prev_id is not None and prev_id != layer_id and isinstance(layers[prev_id], SteeringModule):
        layers[prev_id] = layers[prev_id].block  # restore the original, unwrapped layer

    if not isinstance(layers[layer_id], SteeringModule):
        with torch.no_grad():
            layers[layer_id] = SteeringModule(layers[layer_id])

    model.layer_ids = [layer_id]
    model._active_steering_layer_id = layer_id
    return model


def define_custom_tokenizer(model_name: str, model_path: str = None) -> AutoTokenizer:
    if model_path is not None:  # custom tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)  # Loaded model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer