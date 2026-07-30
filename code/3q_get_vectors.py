import argparse
import os
import datetime
from dialz import SteeringVector

from utils import bbq_axes
from utils_new import get_args, get_model_short_name, configure_model, load_and_tokenize_contrastive

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1', help='model name')
parser.add_argument('-p', '--path', type=str, default=None, help='model path')
parser.add_argument('-q', '--quantization', action='store_true', help='Insert flag to quantize the model')
parser.add_argument('-a', '--axes', nargs='*', type=str, default=None, help='axes to be processed')
args = parser.parse_args()

(model_name, model_path) = get_args([args.name, args.path])
QUANTIZATION = args.quantization
model_short_name = get_model_short_name(model_name, QUANTIZATION)

if args.axes is not None:
    chosen_axes = args.axes.copy()  # list type
else:
    chosen_axes = bbq_axes

VECTOR_DIRS = {
    "train": f"../vectors/{model_short_name}/train",
    "train+prompt": f"../vectors/{model_short_name}/train+prompt",
}

for d in VECTOR_DIRS.values():
    os.makedirs(d, exist_ok=True)

model = configure_model(model_name, model_path, quantized=QUANTIZATION)

for axis in chosen_axes:
    print(f"Creating 4 vectors for {axis} at:", datetime.datetime.now())

    ## BBQ Type Datasets - Ambig + Disambig
    filepath = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, model_path, filepath)
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(VECTOR_DIRS['train'], f"{axis}.gguf"))

    ## BBQ + Prompt Type Datasets - Ambig + Disambig
    filepath = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, model_path, filepath,
                                                  prompt=f"Consider the bias related to {axis} in the following. ")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(VECTOR_DIRS['train+prompt'], f"{axis}.gguf"))

