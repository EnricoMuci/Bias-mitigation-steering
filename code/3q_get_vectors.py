import os
import sys
import datetime
from dialz import Dataset, SteeringVector

from utils_new import create_quantized_model
from utils import bbq_axes, load_and_tokenize_contrastive, contrastive_pairs
from utils_new import OLD_get_args, get_short_name

(model_name, model_path) = OLD_get_args(sys.argv)
model_short_name = get_short_name(model_name)

dirs = {
    "train": f"../vectors/{model_short_name}/train",
    "train+prompt": f"../vectors/{model_short_name}/train+prompt",
    "generate_ss": f"../vectors/{model_short_name}/generate_ss",
    "generate_qa": f"../vectors/{model_short_name}/generate_qa",
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

model = create_quantized_model(model_name, model_path)

for axis in bbq_axes:
    print(f"Creating 4 vectors for {axis} at:", datetime.datetime.now())

    ## BBQ Type Datasets - Ambig + Disambig
    filepath = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, filepath)
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['train'], f"{axis}.gguf"))

    ## BBQ + Prompt Type Datasets - Ambig + Disambig
    filepath = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, filepath,
                                                  prompt=f"Consider the bias related to {axis} in the following. ")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['train+prompt'], f"{axis}.gguf"))
    
    if contrastive_pairs is None:
        continue 

    ## Generated Dataset (using Dialz and sentence-starters)
    train_dataset = Dataset.create_dataset(
        model_name, contrastive_pairs[axis],
        system_role=" ", prompt_type="sentence-starters")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['generate_ss'], f"{axis}.gguf"))

    ## Generated Dataset (using Dialz and question-answer)
    train_dataset = Dataset.create_dataset(model_name, contrastive_pairs[axis],
                                           system_role=" ", prompt_type="question-answer")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['generate_qa'], f"{axis}.gguf"))
