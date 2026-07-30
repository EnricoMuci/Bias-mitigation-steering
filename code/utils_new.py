import traceback

import torch
import typing
import warnings

from dialz import SteeringModel, Dataset
from dialz.vector import SteeringModule, model_layer_list
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

import utils
import os

SEED = 42
STRICT_QUANTIZATION = os.environ.get("STRICT_QUANTIZATION", "0") == "1"
REMOTE_DRIVE_THESIS_PROJECT = '/content/drive/MyDrive/ThesisProject'
BASE_EXPERIMENT = 'reproduction'  # or 'original 'reproduction' 'inject-crows'
VECTOR_TYPES = ['train', 'train+prompt']

CROWS_PATH = '../raw_data/crows/crows_pairs.csv'

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


def get_args(args_list: list):
    model_name = args_list[0]
    model_path = args_list[1]
    return model_name, model_path


def choose_axes(axes: list | None = None) -> list:
    if axes is not None:
        return axes.copy()  # list type
    else:
        return utils.bbq_axes


def get_model_short_name(model_name, quantized=True):
    model_short_names = {
        "Qwen/Qwen2.5-7B-Instruct": "qwen",
        "meta-llama/Llama-3.1-8B-Instruct": "llama",
        "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
    }
    base_name = model_short_names.get(model_name)
    if not base_name:
        raise ValueError(f"Unknown model name: {model_name}")
    if quantized:
        return f"{base_name}-quantized"
    else:
        return f"{base_name}-full"


class CustomSteeringModel(SteeringModel):
    def __init__(
            self,
            model_name: str,
            layer_ids: typing.Iterable[int],
            model_path: str | None = None,
            token: str | None = None,
            quantization_config: BitsAndBytesConfig | None = None,
            torch_dtype: torch.dtype = 'auto' #torch.float16,
    ):
        # Directly call nn.Module.__init__() to bypass SteeringModel default loader
        torch.nn.Module.__init__(self)

        # self.model_name = model_name
        self.model_name = model_path if model_path is not None else model_name
        self.token = token

        # load_path = model_path if model_path is not None else model_name
        load_path = self.model_name

        load_kwargs = {
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        if token is not None:
            load_kwargs["token"] = token

        print(f"Loading weights from {load_path}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(load_path, **load_kwargs)
        print(f"Weights loaded successfully from {load_path}", flush=True)

        # Do not call self.model.to(...) here: device_map='auto' handles GPU/CPU mapping automatically.
        # if quantization_config is None:
        #     self.model = self.model.to(
        #         "cuda:0" if torch.cuda.is_available()
        #         else "mps:0" if torch.backends.mps.is_available()
        #         else "cpu"
        #     )

        layers = model_layer_list(self.model)
        self.layer_ids = [i if i >= 0 else len(layers) + i for i in layer_ids]

        # Debug print for verifying device placement
        print("Device map:", getattr(self.model, "hf_device_map", "N/A"), flush=True)
        for name, param in self.model.named_parameters():
            print(f"{name}: {param.device}")
            break  # END

        for layer_id in self.layer_ids:
            layer = layers[layer_id]
            if not isinstance(layer, SteeringModule):
                with torch.no_grad():
                    layers[layer_id] = SteeringModule(layer)
            else:
                warnings.warn("Attempting to rewrap an already wrapped layer! Call .unwrap first.")


def check_model_quantization(model):
    # Check 1: Does it have a quantization config attached?
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        print("Warning: Model configuration contains a quantization_config!")
        print(model.config.quantization_config)
    else:
        print("Model configuration is clean. No quantization_config found.")

    # Check 2: Inspect the actual weights of the first Linear layer
    for name, module in model.named_modules():
        if "Linear" in str(type(module)):
            weight_dtype = module.weight.dtype
            print(f"First linear layer ('{name}') weight dtype: {weight_dtype}")

            # Types like uint8, int8, or int4 indicate quantization
            if weight_dtype in [torch.float16, torch.bfloat16, torch.float32]:
                print(f"Model weights are in standard float precision. Dtype: {weight_dtype}")
            else:
                print(f"Model appears to be quantized: Dtype: {weight_dtype}")
            break  # We only need to check the first one


def configure_model(
        model_name: str,
        model_path: str,
        layer_ids: list[int] | None = None,
        quantized: bool = True
    ) -> CustomSteeringModel:
    if layer_ids is None:
        layer_ids = [5]

    if not quantized:
        print(f"Configuring full-precision model {model_name} from {model_path}")
        return CustomSteeringModel(
            model_name=model_name,
            model_path=model_path,
            layer_ids=layer_ids,
            quantization_config=None,
        )

    print(f"Configuring 4-bit quantized model {model_name} from {model_path}")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return CustomSteeringModel(
            model_name=model_name,
            model_path=model_path,
            layer_ids=layer_ids,
            quantization_config=bnb_config,
        )
    except Exception as e:
        tqdm.write(f"[FALLBACK - NOT QUANTIZED] {model_name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        if STRICT_QUANTIZATION:
            raise
        # Consistent fallback returning CustomSteeringModel without quantization
        return CustomSteeringModel(
            model_name=model_name,
            model_path=model_path,
            layer_ids=layer_ids,
            quantization_config=None,
        )


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


def define_custom_tokenizer(model_name: str, model_path: str | None = None, token = None) -> AutoTokenizer:
    if model_path is not None:  # custom tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)  # Loaded model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_and_tokenize_contrastive(model_name: str, model_path: str, filepath: str, prompt: str = "") -> Dataset:
    ds_raw = Dataset.load_from_file(filepath)

    # tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer = define_custom_tokenizer(model_name, model_path)

    ds_tok = Dataset()
    for entry in ds_raw.view_dataset():
        pos_tok = Dataset._apply_chat_template(
            tokenizer=tokenizer,
            system_role="",
            content1="",
            content2=prompt + entry.positive
        )
        neg_tok = Dataset._apply_chat_template(
            tokenizer=tokenizer,
            system_role="",
            content1="",
            content2=prompt + entry.negative
        )
        ds_tok.add_entry(pos_tok, neg_tok)

    return ds_tok
