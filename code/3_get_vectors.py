
import os
import sys
import datetime
from dialz import Dataset, SteeringModel, SteeringVector
from utils import bbq_axes, load_and_tokenize_contrastive, contrastive_pairs






model = SteeringModel(model_name, [5])  # Second element is arbritary as we're not generating yet

for axis in bbq_axes:
    print(f"Creating 4 vectors for {axis} at:", datetime.datetime.now())

    ## BBQ Type Datasets - Ambig + Disambig
    path = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, path)
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['train'], f"{axis}.gguf"))

    ## BBQ + Prompt Type Datasets - Ambig + Disambig
    path = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, path,
                                                  prompt=f"Consider the bias related to {axis} in the following. ")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['train+prompt'], f"{axis}.gguf"))

    ## Generated Dataset (using Dialz and sentence-starters)
    train_dataset = Dataset.create_dataset(
        model_name, contrastive_pairs[axis], system_role=" ",
                                           prompt_type="sentence-starters")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['generate_ss'], f"{axis}.gguf"))

    ## Generated Dataset (using Dialz and question-answer)
    train_dataset = Dataset.create_dataset(model_name, contrastive_pairs[axis], system_role=" ",
                                           prompt_type="question-answer")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['generate_qa'], f"{axis}.gguf"))