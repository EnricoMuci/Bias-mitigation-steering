import os
import datetime
import transformers
import pandas as pd
import numpy as np
import argparse

from dialz.vector import model_layer_list, SteeringModule
from tqdm.auto import tqdm

from datasets import load_dataset
from dialz import SteeringVector


from utils import get_output
from utils_new import REMOTE_DRIVE_DIR, create_quantized_model, define_custom_tokenizer, get_short_name, new_get_args #, EXPERIMENT

transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1')  # model name
parser.add_argument('-p', '--path', type=str, default=None)  # model path
parser.add_argument('-c', '--colab', action='store_true')  # flag about remote saving
args = parser.parse_args()

(model_name, model_path) = new_get_args([args.name, args.path])
model_short_name = get_short_name(model_name)

tokenizer = define_custom_tokenizer(model_name, model_path)

local_best_layers_dir = f'../data/layer_scores/{model_short_name}/best_layers'
local_bbq_validate_dir = f"../data/bbq_validate"  # 1 file per axis
local_coeff_scores_dir = f'../data/coeff_scores/{model_short_name}'


def check_paths():
    checked = 0
    if os.path.exists(local_best_layers_dir):
        checked += 1
    else:
        print(f'Missing this path:\n{local_best_layers_dir}')

    if os.path.exists(local_bbq_validate_dir):
        checked += 1
    else:
        print(f'Missing this path:\n{local_bbq_validate_dir}')

    if os.path.exists(local_coeff_scores_dir):
        checked += 1
    else:
        print(f'Missing this path:\n{local_coeff_scores_dir}')

    if checked >= 3:
        return True
    else:
        return False


# MMLU Preparation
print("\nLoading MMLU dataset...")
mmlu = load_dataset("cais/mmlu", "all", split="test")
print("\nProcessing MMLU dataset...")
full_df = pd.DataFrame(mmlu)

# Get an equal sample from all subjects up to roughly 1000 questions
mmlu_df = full_df.groupby('subject').sample(n=1000 // full_df['subject'].nunique(), random_state=42).reset_index(
    drop=True)
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
        "ans": generated_answer,
        "prediction": predicted_label,
        "correct": correct
    })


def save_results(results_df, local_file_path, remote_file_path):
    """Save results to local path, and to Drive if on Colab."""
    results_df.to_csv(local_file_path, index=False)
    if args.colab:
        results_df.to_csv(remote_file_path, index=False)


def get_best_coeffs():
    """
    """
    model = create_quantized_model(model_name, model_path)  # NEW: Load the model

    vector_types = ["top_train", "top_train+prompt"]
    # vector_types = ["top_layer_train", "top_layer_train+prompt"]

    for top_vt_csv in vector_types:
        file_path = f"{local_best_layers_dir}/{top_vt_csv}.csv"

        if not os.path.exists(file_path):
            # In best_layers there should be only
            print(f"Missing the following top_vt_csv:\n{file_path}")
            continue

        best_layers = pd.read_csv(file_path)
        print(best_layers.head())
        print(f"Processing {top_vt_csv}")

        for _, row in best_layers.iterrows():
            # each axis is a bias variable, with its own best layer index (max. accuracy and separability)
            axis = row['axis']
            layer = row['max_layer']
            top_vt_csv = row['vt']

            try:  # Load in validation set
                validation_df = pd.read_csv(f"{local_bbq_validate_dir}/{axis}_validate.csv")
                print(f"Running co-effs for {axis} on vector {top_vt_csv} at {datetime.datetime.now()}")
                vector = SteeringVector.import_gguf(f'../vectors/{model_short_name}/{top_vt_csv}/{axis}.gguf')  # steer
            except FileNotFoundError as e:
                print(f"Missing file in BBQ Validate for this axis (and type): {axis} ({top_vt_csv}).\n"
                      f"Error: {e}")
                continue

            # Save paths
            csv_name = f"{axis}_{top_vt_csv}.csv"

            local_file_path = f"{local_coeff_scores_dir}/{top_vt_csv}"
            os.makedirs(local_file_path, exist_ok=True)
            local_file_path = os.path.join(local_file_path, csv_name)

            if args.colab:  # In Colab, it creates the remote top_vt_csv path to manage session aborts
                remote_dir_path = f"{REMOTE_DRIVE_DIR}/data/coeff_scores/{model_short_name}-reproduced/{top_vt_csv}"
                os.makedirs(remote_dir_path, exist_ok=True)
                remote_file_path = os.path.join(remote_dir_path, csv_name)
            else:  # No Google Drive
                remote_file_path = ''

            results = []
            completed_coeffs = set()

            # Resume logic, to avoid previous coefficients
            existing_csv = None
            if args.colab and remote_file_path and os.path.exists(remote_file_path):
                existing_csv = pd.read_csv(remote_file_path)
                print(f"Resuming {axis} from Drive ({len(existing_csv)} coefficients already done).")
            elif os.path.exists(local_file_path):
                existing_csv = pd.read_csv(local_file_path)
                print(f"Resuming {axis} from local top_vt_csv ({len(existing_csv)} coefficients already done).")
            else:
                print(f"No pre-calculation for {axis}, starting from scratch.")

            if existing_csv is not None:
                results = existing_csv.to_dict('records')
                completed_coeffs = set(f"{c:.1f}" for c in existing_csv['coeff'].values)

            # NEW: Wrapping and unwrapping
            layers = model_layer_list(model.model)
            if hasattr(model, 'layer_ids'):
                for old_id in model.layer_ids:
                    old_layer = layers[old_id]
                    # Remove wrapper for previous layer
                    # if isinstance(old_layer, SteeringModule):
                    if type(old_layer).__name__ == 'SteeringModule' or hasattr(old_layer, 'block'):
                        layers[old_id] = old_layer.block

            model.layer_ids = [layer]

            # if not isinstance(layers[layer], SteeringModule):
            if type(layers[layer]).__name__ != 'SteeringModule':
                layers[layer] = SteeringModule(layers[layer])

            for coeff in tqdm(np.linspace(-2.0, 2.0, 21), desc=f"Coeffs for {axis}"):
                # Avoid previously calculated coefficients
                coeff_key = f"{coeff:.1f}"
                if coeff_key in completed_coeffs:
                    continue

                try:
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
                        'coeff': round(coeff, 1),
                        'bbq_correct': bbq_correct,
                        'mmlu_correct': mmlu_correct,
                        'bbq_accuracy': round(bbq_accuracy, 3),
                        'mmlu_accuracy': round(mmlu_accuracy, 3),
                    })
                    completed_coeffs.add(coeff_key)  # NEW from 5AA

                    # AUTOSAVE: the top_vt_csv is overwritten at every calculated coefficient
                    results_df = pd.DataFrame(results)
                    save_results(results_df, local_file_path, remote_file_path)

                except Exception as err:
                    # Log and continue: do NOT let a single coeff crash the whole run
                    print(f"[ERROR] axis={axis}, coeff={coeff_key}: {err}")
                    continue
            # for coefficients

            # Final explicit save after completing all coefficients for this axis
            if results:
                results_df = pd.DataFrame(results)
                save_results(results_df, local_file_path, remote_file_path)
                print(f"Completed {axis}: {len(results)} coefficients saved.")
        # for axes (files in /bbq_validate, rows in /best_layers)
    # for files (train, train+prompt


if __name__ == "__main__":
    if check_paths():
        print('All path correctly checked :)')
        get_best_coeffs()
