import os
import sys
import datetime
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from dialz import Dataset, SteeringModel, SteeringVector
from utils import bbq_axes, load_and_tokenize_contrastive, contrastive_pairs


if len(sys.argv) > 2:
    model_name = sys.argv[1]
    model_path = sys.argv[2]
elif len(sys.argv) > 1:
    model_name = sys.argv[1]
    model_path = model_name
else:
    raise ValueError("Model name and model path must be provided as command-line arguments.")

# Map model names to short names
model_short_names = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
}

model_short_name = model_short_names.get(model_name)
if not model_short_name:
    raise ValueError(f"Unknown model name: {model_name}")

dirs = {
    "train": f"../vectors/{model_short_name}/train",
    "train+prompt": f"../vectors/{model_short_name}/train+prompt",
    "generate_ss": f"../vectors/{model_short_name}/generate_ss",
    "generate_qa": f"../vectors/{model_short_name}/generate_qa",
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# Configurazione 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"--- Loading base model in 4-bit: {model_path} ---")

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)
try: 
    model = SteeringModel(base_model, [5])
except Exception as e:
    print(f"Resetting Model configuration due to: \n{e} \n---")
    model = SteeringModel(model_path, [5])  # Second element is arbritary as we're not generating yet

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