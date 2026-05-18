import argparse
import os
import datetime
from dialz import Dataset, SteeringVector

from utils_new import create_quantized_model
from utils import bbq_axes, load_and_tokenize_contrastive, contrastive_pairs
from utils_new import new_get_args, get_model_short_name

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1')  # model name
parser.add_argument('-p', '--path', type=str, default=None)  # model path
args = parser.parse_args()

(model_name, model_path) = new_get_args([args.name, args.path])
model_short_name = get_model_short_name(model_name)

VECTOR_DIRS = {
    "train": f"../vectors/{model_short_name}/train",
    "train+prompt": f"../vectors/{model_short_name}/train+prompt",
    "generate_ss": f"../vectors/{model_short_name}/generate_ss",
    "generate_qa": f"../vectors/{model_short_name}/generate_qa",
}

for d in VECTOR_DIRS.values():
    os.makedirs(d, exist_ok=True)

model = create_quantized_model(model_name, model_path)

for axis in bbq_axes:
    print(f"Creating 4 vectors for {axis} at:", datetime.datetime.now())

    ## BBQ Type Datasets - Ambig + Disambig
    filepath = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, filepath)
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(VECTOR_DIRS['train'], f"{axis}.gguf"))

    ## BBQ + Prompt Type Datasets - Ambig + Disambig
    filepath = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, filepath,
                                                  prompt=f"Consider the bias related to {axis} in the following. ")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(VECTOR_DIRS['train+prompt'], f"{axis}.gguf"))
    
    if contrastive_pairs is not None:
        ## Generated Dataset (using Dialz and sentence-starters)
        train_dataset = Dataset.create_dataset(
            model_name, contrastive_pairs[axis], system_role=" ", prompt_type="sentence-starters"
        )
        vector = SteeringVector.train(model, train_dataset)
        vector.export_gguf(os.path.join(VECTOR_DIRS['generate_ss'], f"{axis}.gguf"))

        ## Generated Dataset (using Dialz and question-answer)
        train_dataset = Dataset.create_dataset(model_name, contrastive_pairs[axis], system_role=" ", prompt_type="question-answer")
        vector = SteeringVector.train(model, train_dataset)
        vector.export_gguf(os.path.join(VECTOR_DIRS['generate_qa'], f"{axis}.gguf"))
