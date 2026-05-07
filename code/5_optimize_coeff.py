
import os
import datetime
import transformers
import pandas as pd
import numpy as np
import argparse

from datasets import load_dataset
from dialz import SteeringVector
from utils import get_output
from transformers import AutoTokenizer
from utils_new import get_arguments, get_short_name, create_quantized_model

transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1')  # model name
parser.add_argument('-p', '--path', type=str, default=None)  # model path
parser.add_argument('-a', '--axes', nargs='*', type=str, default=None)  # axes to be processed
args = parser.parse_args()

(model_name, model_path) = get_arguments([args.name, args.path])
model_short_name = get_short_name(model_name)


tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

print("\nLoading MMLU dataset...")
mmlu = load_dataset("cais/mmlu", "all", split="test")
print("\nProcessing MMLU dataset...")
full_df = pd.DataFrame(mmlu)

# Get an equal sample from all subjects up to roughly 1000 questions
mmlu_df = full_df.groupby('subject').sample(n=1000 // full_df['subject'].nunique(), random_state=42).reset_index(drop=True)
print(len(mmlu_df))


def predict_row(row, model, vector, coeff, task):
    question = row["question"]

    if task == "bbq":
        context = row["context"]
        answers = [row["ans0"], row["ans1"], row["ans2"]]
        answer_letters = ["a", "b", "c"]
        correct_answer = row['label']
    
    elif task == "mmlu":
        context = ""
        answers = row["choices"]
        answer_letters = ["a", "b", "c", "d"]
        correct_answer = row['answer']


    input_text = (
        f"Answer with only a single letter.\n\n"
        f"The following are multiple choice questions (with answers):\n"
        f"{context} {question}\n"
        f"(a) {answers[0]}\n"
        f"(b) {answers[1]}\n"
        f"(c) {answers[2]}\n"
    )
    if task == "mmlu":
        input_text = input_text + f"(d) {answers[3]}\n"
    
    generated_answer = get_output(input_text, model, tokenizer, vector, coeff).lower()

    predicted_label = -1
    for i, letter in enumerate(answer_letters):
        if letter in generated_answer[:2]:
            predicted_label = i
            break
    if predicted_label == -1 and task == 'bbq':
        answers = [row["ans0"], row["ans1"], row["ans2"]]
        for i, answer in enumerate(answers):
            if answer in generated_answer:
                predicted_label = i
                break
    
    correct = (predicted_label == correct_answer)

    return pd.Series({
        "ans":         generated_answer,
        "prediction":  predicted_label,
        "correct":     correct
    })


def get_best_coeffs():

    # Load the model once globally instead of multiple times inside the loop
    model = create_quantized_model(model_name, model_path) # NEW

    # top_files = ["top_train", "top_train+prompt"]
    top_files = ["top_layer_train", "top_layer_train+prompt"]

    for file in top_files:
        file_path = f"../data/layer_scores/{model_short_name}/best_layers/{file}.csv"
        
        if not os.path.exists(file_path):
            # In best_layers there should be only
            # top_layer_train.csv and top_layer_train+prompt.csv
            print(f"Missing the following file:\n{file_path}")
            continue
            
        best_layers = pd.read_csv(file_path)
        print(best_layers.head())
        print(f"Processing {file}")

        for _, row in best_layers.iterrows():
            axis = row['axis']
            layer = row['max_layer']
            vector_type = row['vt']

            try:
                # Load in validation set
                validation_df = pd.read_csv(f"../data/bbq_validate/{axis}_validate.csv")
                vector = SteeringVector.import_gguf(f'../vectors/{model_short_name}/{vector_type}/{axis}.gguf')  # steer
            except FileNotFoundError as e:
                print(f"Missing axis: {axis} ({vector_type}).\nError: {e}")
                continue
            
            print(f"Running co-effs for {axis} on vector {vector_type} at {datetime.datetime.now()}")
            results = []

            for coeff in np.arange(-2.0, 2.1, 0.2):
                bbq_df = validation_df.copy()
                mmlu_valid = mmlu_df.copy()

                # apply the predictor to every row
                bbq_df[['ans', 'prediction', 'correct']] = bbq_df.apply(
                    predict_row,
                    axis=1,
                    args=(model, vector, coeff, 'bbq')
                )

                # if your true labels live in column "label", you can now compute accuracy:
                bbq_correct = (bbq_df["prediction"] == bbq_df["label"]).sum()
                bbq_accuracy = bbq_correct / len(bbq_df)

                mmlu_valid[['ans', 'prediction', 'correct']] = mmlu_valid.apply(
                    predict_row,
                    axis=1,
                    args=(model, vector, coeff, 'mmlu')
                )

                # compute accuracy
                mmlu_correct = (mmlu_valid["prediction"] == mmlu_valid["answer"]).sum()
                mmlu_accuracy = mmlu_correct / len(mmlu_valid)

                results.append({
                    'coeff': coeff,
                    'bbq_correct': int(bbq_correct),
                    'mmlu_correct': float(mmlu_correct),
                    'bbq_accuracy': float(bbq_accuracy),
                    'mmlu_accuracy': float(mmlu_accuracy),

                })

            results_df = pd.DataFrame(results)
            
            # Format columns for saving
            results_df['coeff'] = results_df['coeff'].round(1)
            results_df['bbq_accuracy'] = results_df['bbq_accuracy'].round(3)
            results_df['mmlu_accuracy'] = results_df['mmlu_accuracy'].round(3)

            dir_path = f"../data/coeff_scores/{model_short_name}/{file}"
            os.makedirs(dir_path, exist_ok=True)

            results_df.to_csv(os.path.join(dir_path, f"{axis}_{vector_type}.csv"), index=False)


get_best_coeffs()
