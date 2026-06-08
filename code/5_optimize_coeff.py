import os
import datetime
import transformers
import pandas as pd
import numpy as np
import argparse

from tqdm.auto import tqdm

from datasets import load_dataset
from dialz import SteeringVector

from utils import get_output
from utils_new import *

import warnings
warnings.filterwarnings(
    "ignore",
    message="_check_is_size will be removed",
    category=FutureWarning
)

transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1')  # model name
parser.add_argument('-p', '--path', type=str, default=None)  # model path
parser.add_argument('-c', '--colab', action='store_true')  # flag about remote saving
args = parser.parse_args()

(model_name, model_path) = new_get_args([args.name, args.path])
model_short_name = get_model_short_name(model_name)

tokenizer = define_custom_tokenizer(model_name, model_path)

LOCAL_BEST_LAYERS_DIR = f'../data/layer_scores/{model_short_name}/best_layers'
LOCAL_BBQ_VALIDATE_DIR = f"../data/bbq_validate"  # 1 file for each axis
LOCAL_COEFF_SCORES_DIR = f'../data/coeff_scores/{model_short_name}'
TOP_VECTOR_TYPES = ["top_train", "top_train+prompt"]


def resume_logic(axis, remote_file_path, local_file_path):
    existing_csv = None
    if args.colab and os.path.exists(remote_file_path):
        existing_csv = pd.read_csv(remote_file_path)
        print(f"Resuming {axis} from Drive ({len(existing_csv)} coefficients already done).")
    elif os.path.exists(local_file_path):
        existing_csv = pd.read_csv(local_file_path)
        print(f"Resuming {axis} from local top_vt_csv ({len(existing_csv)} coefficients already done).")
    else:
        print(f"No pre-calculation for {axis}, starting from scratch.")
    return existing_csv


def check_paths():
    """
    Path checking
    """
    checked = 0
    if os.path.exists(LOCAL_BEST_LAYERS_DIR):
        checked += 1
    else:
        print(f'Missing this path:\n{LOCAL_BEST_LAYERS_DIR}')

    if os.path.exists(LOCAL_BBQ_VALIDATE_DIR):
        checked += 1
    else:
        print(f'Missing this path:\n{LOCAL_BBQ_VALIDATE_DIR}')

    if os.path.exists(LOCAL_COEFF_SCORES_DIR):
        checked += 1
        for vtf in TOP_VECTOR_TYPES:
            print(f'Creating this directory: {LOCAL_COEFF_SCORES_DIR}/{vtf}/')
            os.makedirs(os.path.join(LOCAL_COEFF_SCORES_DIR, vtf), exist_ok=True)
    else:
        try:
            os.makedirs(LOCAL_COEFF_SCORES_DIR, exist_ok=True)
            print(f'Missing this path:\n{LOCAL_COEFF_SCORES_DIR}. Just created')
            checked += 1
        except Exception as err:
            print(f'Missing this path:\n{LOCAL_COEFF_SCORES_DIR}. ERROR: {err}')


    # print(f'Checked = {checked}')
    if checked >= 3:
        return True
    else:
        return False


def prepare_MMLU():
    print("\nLoading MMLU dataset...")
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    print("\nProcessing MMLU dataset...")
    full_df = pd.DataFrame(mmlu)

    # Get an equal sample from all subjects up to roughly 1000 questions
    mmlu_df = full_df.groupby('subject').sample(n=1000 // full_df['subject'].nunique(), random_state=42).reset_index(
        drop=True)
    print(len(mmlu_df))
    return mmlu_df

def preview_status(): # NEW
    """Print a preview of the current statu """
    print("\n" + "="*55)
    print("PRE-RUN STATUS CHECK")
    print("="*55)

    all_done = True

    for top_vt in TOP_VECTOR_TYPES:
        top_best_vt_file = f"{LOCAL_BEST_LAYERS_DIR}/{top_vt}.csv"
        print(f"\n[{top_vt}]")

        if not os.path.exists(top_best_vt_file):
            print(f"  ✗ Missing file: {top_best_vt_file}")
            all_done = False
            continue

        best_layers = pd.read_csv(top_best_vt_file)

        for _, row in best_layers.iterrows():
            axis = row['axis']
            vt   = row['vt']
            csv_name = f"{axis}_{vt}.csv"

            found_path = None
            if args.colab:
                remote_dir = (f"{REMOTE_DRIVE_THESIS_PROJECT}/data/coeff_scores/"
                              f"{model_short_name}-{EXPERIMENT}/{top_vt}")
                remote_fp = os.path.join(remote_dir, csv_name)
                if os.path.exists(remote_fp):
                    found_path = remote_fp

            if found_path is None:
                local_fp = os.path.join(LOCAL_COEFF_SCORES_DIR, top_vt, csv_name)
                if os.path.exists(local_fp):
                    found_path = local_fp

            if found_path is None:
                print(f"  ✗ {axis:15s} ({vt})  →  not initialized")
                all_done = False
            else:
                df = pd.read_csv(found_path)
                done = len(df)
                if done >= 21:
                    print(f"  ✓ {axis:15s} ({vt})  →  complete ({done}/21)")
                else:
                    print(f"  ○ {axis:15s} ({vt})  →  partial ({done}/21)") #…
                    all_done = False

    print("\n" + "="*55)
    if all_done:
        print("All axes calculated :)")
    else:
        print("Resuming operations...")
    print("="*55 + "\n")
    return all_done


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


def get_best_coeffs(mmlu_df=None):
    model = create_quantized_model(model_name, model_path)  # NEW: Load the model

    for top_vt in TOP_VECTOR_TYPES:  # 'top_train' 'top_train+prompt'
        top_best_vt_file = f"{LOCAL_BEST_LAYERS_DIR}/{top_vt}.csv"

        if not os.path.exists(top_best_vt_file):
            # In best_layers there should be only
            print(f"Missing the following top file in best_layers directory:\n{top_best_vt_file}.csv. "
                  f"Next iteration...")
            continue

        best_layers = pd.read_csv(top_best_vt_file)
        print(best_layers.head())
        print(f"Processing {top_vt}.csv")

        for _, row in best_layers.iterrows():  # for each discrimination-axis
            # each axis is a bias variable, with its own best layer index (max. accuracy and separability)
            axis = row['axis']
            layer = row['max_layer']
            vt = row['vt']  # 'train' or 'train+prompt'

            try:  # Load in validation set
                validation_df = pd.read_csv(f"{LOCAL_BBQ_VALIDATE_DIR}/{axis}_validate.csv")
                print(f"Running co-effs for {axis} on vector {vt} at {datetime.datetime.now()}")
                vector = SteeringVector.import_gguf(f'../vectors/{model_short_name}/{vt}/{axis}.gguf')  # steer
            except FileNotFoundError as e:
                print(f"Missing file in BBQ Validate (or in /vectors/) for this axis (and type): {axis} ({vt}).\n"
                      f"Error: {e}")
                continue

            # Save paths
            csv_name = f"{axis}_{vt}.csv"

            local_dir_path = f"{LOCAL_COEFF_SCORES_DIR}/{top_vt}"  # 'top_train/' or 'top_train+prompt/'
            os.makedirs(local_dir_path, exist_ok=True)
            local_file_path = os.path.join(local_dir_path, csv_name)

            if args.colab:  # In Colab, it creates the remote vt path to manage session aborts
                remote_dir_path = (f"{REMOTE_DRIVE_THESIS_PROJECT}/data/coeff_scores/"
                                   f"{model_short_name}-{EXPERIMENT}/{top_vt}")
                os.makedirs(remote_dir_path, exist_ok=True)
                remote_file_path = os.path.join(remote_dir_path, csv_name)
            else:  # No Google Drive
                remote_file_path = ''

            results = []
            completed_coeffs = set()

            # Resume logic, to avoid previous coefficients
            existing_csv = resume_logic(axis, remote_file_path, local_file_path)

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

            # for coeff in tqdm( # OLD
            #         np.linspace(-2.0, 2.0, 21),
            #         desc=f"Coeffs for {axis}",
            #         total=21,
            #         initial=len(completed_coeffs)
            #     ):
            all_coeffs = np.linspace(-2.0, 2.0, 21)
            remaining_coeffs = [c for c in all_coeffs if f"{c:.1f}" not in completed_coeffs]

            for coeff in tqdm(
                    remaining_coeffs,
                    desc=f"  Coeffs for {axis}",
                    total=21,  # max length
                    initial=len(completed_coeffs),  # initial step
                    leave=False,
            ):
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
                        'bbq_correct': int(bbq_correct), # int
                        'mmlu_correct': int(mmlu_correct), # int
                        'bbq_accuracy': round(bbq_accuracy, 3),
                        'mmlu_accuracy': round(mmlu_accuracy, 3),
                    })
                    completed_coeffs.add(coeff_key)  # NEW from 5AA

                    # AUTOSAVE: the vt is overwritten at every calculated coefficient
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
    # for vector-type files (train, train+prompt)


if __name__ == "__main__":
    if check_paths():
        print('All path correctly checked :)')
        already_done = preview_status()  # Status preview
        if not already_done:
            get_best_coeffs(prepare_MMLU())  # Work
        else:
            print("Nulla da fare, uscita.")
    else:
        print('Something wrong :(')
