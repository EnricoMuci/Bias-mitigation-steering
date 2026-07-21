import datetime
import transformers
import pandas as pd
import numpy as np
import argparse
import os

from tqdm.auto import tqdm  # from tqdm.notebook import tqdm

from datasets import load_dataset
from dialz import SteeringVector
from dialz.vector import SteeringModule

from utils import bbq_axes
from utils import get_output
from utils_new import (new_get_args, get_model_short_name, define_custom_tokenizer, create_quantized_model,
                       model_layer_list, REMOTE_DRIVE_THESIS_PROJECT, CROWS_AXIS_MAP, EXPERIMENT, SEED)


import warnings

warnings.filterwarnings(
    "ignore",
    message="_check_is_size will be removed",
    category=FutureWarning
)

transformers.logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1')
parser.add_argument('-p', '--path', type=str, default=None, help='model path')
parser.add_argument('-c', '--colab', action='store_true', help='flag about remote saving')
parser.add_argument('-o', '--only-preview', action='store_true', help='Show only the preview')
parser.add_argument('-k', '--k-sentences', type=int, default=4, help='Number of retrieved sentences')
parser.add_argument('-b', '--bias-ratio', type=float, default=0.5, help='Pro-stereotype sentences ratio (0.0 - 1.0)')
parser.add_argument('-a', '--axes', nargs='*', type=str, default=None, help='axes to be processed')

args = parser.parse_args()


(model_name, model_path) = new_get_args([args.name, args.path])
model_short_name = get_model_short_name(model_name)

tokenizer = define_custom_tokenizer(model_name, model_path)

if args.axes is not None:
    required_axes = args.axes
else:
    required_axes = bbq_axes

LOCAL_BEST_LAYERS_DIR = f'../data/layer_scores/{model_short_name}/best_layers'
LOCAL_BBQ_VALIDATE_DIR = f"../data/bbq_validate"  # 1 file for each axis
LOCAL_COEFF_SCORES_DIR = f'../data/coeff_scores/{model_short_name}-{EXPERIMENT}'
TOP_VECTOR_TYPES = ["top_train", "top_train+prompt"]


def resume_logic(axis, remote_file_path, local_file_path, vt):
    existing_csv = None
    print(' ')
    if args.colab and os.path.exists(remote_file_path):
        existing_csv = pd.read_csv(remote_file_path)
        print(f"Resuming {axis}-{vt} from Drive file ({len(existing_csv)} coefficients already done).")
    elif os.path.exists(local_file_path):
        existing_csv = pd.read_csv(local_file_path)
        print(f"Resuming {axis}-{vt} from local file ({len(existing_csv)} coefficients already done).")
    else:
        print(f"No pre-calculation for {axis}-{vt}, starting from scratch.")
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

    if os.path.exists("../raw_data/crows/crows_pairs.csv"):
        checked += 1
    else:
        print(f'Missing documents path:\n../raw_data/crows/crows_pairs.csv')

    # print(f'Checked = {checked}')
    if checked >= 4:
        return True
    else:
        return False


def old_prepare_MMLU():
    print("\nLoading MMLU dataset...")
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    print("\nProcessing MMLU dataset...")
    full_df = pd.DataFrame(mmlu)

    # Get an equal sample from all subjects up to roughly 1000 questions
    mmlu_df = full_df.groupby('subject').sample(n=1000 // full_df['subject'].nunique(), random_state=SEED).reset_index(
        drop=True)
    print(len(mmlu_df))
    return mmlu_df


def prepare_MMLU():
    mmlu_dir = "../raw_data/mmlu"
    mmlu_path = f"{mmlu_dir}/mmlu_all_test.parquet"

    if os.path.exists(mmlu_path):
        print(f"\nLoading MMLU dataset from local cache: {mmlu_path}")
        full_df = pd.read_parquet(mmlu_path)
    else:
        print(f"\nLocal MMLU file not found ({mmlu_path}). Attempting download...")
        try:
            mmlu = load_dataset("cais/mmlu", "all", split="test")
        except Exception as err:
            raise RuntimeError(
                f"Could not load MMLU locally and download failed "
                f"(offline mode or no network access?). "
                f"Place a pre-downloaded file at {mmlu_path} and retry.\n"
                f"Original error: {err}"
            )
        full_df = pd.DataFrame(mmlu)
        os.makedirs(mmlu_dir, exist_ok=True)
        full_df.to_parquet(mmlu_path)
        print(f"Saved MMLU dataset locally for future runs: {mmlu_path}")

    print("\nProcessing MMLU dataset...")

    mmlu_df = (
        full_df.groupby('subject').sample(n=1000 // full_df['subject'].nunique(), random_state=SEED).reset_index(
            drop=True)
    )
    print(len(mmlu_df))
    return mmlu_df


def preview_status():  # NEW
    """Print a preview of the current status"""
    print("\n" + "=" * 55)
    print("PRE-RUN STATUS CHECK")
    print("=" * 55)

    all_done = True
    resume_point = None

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
            vt = row['vt']
            csv_name = f"{axis}_{vt}.csv"
            layer = row['max_layer']

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
                print(f'Missing path: {LOCAL_BEST_LAYERS_DIR}')
                print(f"  ✗ {axis:15s} ({vt})  →  not initialized")
                all_done = False
                if resume_point is None:
                    resume_point = {
                        'top_vt': top_vt, 'axis': axis, 'vt': vt,
                        'layer': layer, 'done': 0
                    }
            else:
                print(f'Coefficient scores path: {found_path}')
                print("=" * 55)

                df = pd.read_csv(found_path)
                done = len(df)
                if done >= 21:
                    print(f"  ✓ {axis:15s} ({vt})  →  complete ({done}/21)")
                else:
                    print(f"  ○ {axis:15s} ({vt})  →  partial ({done}/21)")  
                    all_done = False
                    if resume_point is None:  # NEW
                        resume_point = {
                            'top_vt': top_vt, 'axis': axis, 'vt': vt,
                            'layer': layer, 'done': done
                        }

    print("\n" + "=" * 55)
    if all_done:
        print("All axes calculated :)")
    else:
        rp = resume_point
        print(f"Resuming operations from: \n{rp['axis']} ({rp['vt']}, {rp['top_vt']})\n"
              f"layer {rp['layer']}  →  {rp['done']}/21 coefficients done")
    print("=" * 55 + "\n")
    return all_done


def inject_crows_bias_to_df(bbq_df, crows_df, axis, num_sentences=4, bias_ratio=0.5):
    if num_sentences <= 0:
        bbq_df = bbq_df.copy()
        bbq_df['injected_context'] = ""  # no contextual injections
        return bbq_df

    crows_axis = CROWS_AXIS_MAP.get(axis, axis)

    crows_filtered = crows_df[crows_df['bias_type'] == crows_axis]
    if crows_filtered.empty:
        print(f"[WARNING] No CrowS '{crows_axis}' for BBQ '{axis}' axis. "
              f"Fallback on the full dataset.")
        crows_filtered = crows_df

    num_stereo = int(round(num_sentences * bias_ratio))
    num_anti = num_sentences - num_stereo
    rng = np.random.default_rng(SEED)  # random generator

    injected_contexts = []

    for _ in range(len(bbq_df)):
        sentences_to_inject = []

        if num_stereo > 0:
            stereo_rows = crows_filtered.sample(n=num_stereo, replace=True, random_state=rng)
            sentences_to_inject.extend(stereo_rows['sent_more'].tolist())

        if num_anti > 0:
            anti_rows = crows_filtered.sample(n=num_anti, replace=True, random_state=rng)
            sentences_to_inject.extend(anti_rows['sent_less'].tolist())

        order = rng.permutation(len(sentences_to_inject))
        sentences_to_inject = [sentences_to_inject[i] for i in order]

        injected_str = "Retrieved Information:\n"
        for sent in sentences_to_inject:
            injected_str += f"- {sent}\n"

        injected_contexts.append(injected_str + "\n")

    bbq_df = bbq_df.copy()
    bbq_df['injected_context'] = injected_contexts
    return bbq_df


def predict_row(row, model, vector, coeff, task):
    question = row["question"]

    if task == "bbq":
        injected = row.get("injected_context", "")
        context = row["context"]
        answers = [row["ans0"], row["ans1"], row["ans2"]]
        answer_letters = ["a", "b", "c"]
        correct_answer = row['label']

    elif task == "mmlu":
        injected = ""
        context = ""
        answers = row["choices"]
        answer_letters = ["a", "b", "c", "d"]
        correct_answer = row['answer']

    input_text = (
        f"Answer with only a single letter.\n\n"
        f"The following are multiple choice questions (with answers):\n"
        f"{injected}"
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
        print('\n', best_layers.head(), '\n')  # Print head of best_layers
        print(f"Processing {top_vt}.csv")

        for _, row in best_layers.iterrows():  # for each discrimination-axis
            # each axis is a bias variable, with its own best layer index (max. accuracy and separability)
            axis = row['axis']

            if axis not in required_axes:
                print(f'Skipping {axis}: not required')
                continue

            layer = row['max_layer']
            vt = row['vt']  # 'train' or 'train+prompt'

            # Injection parameters
            k_sentences = args.k_sentences  # top-k sentences
            b_ratio = args.bias_ratio  # fraction of pro- stereotyped sentences

            try:  # Load in validation set
                validation_df = pd.read_csv(f"{LOCAL_BBQ_VALIDATE_DIR}/{axis}_validate.csv") # validation samples
                crows_df = pd.read_csv("../raw_data/crows/crows_pairs.csv") # Database for injections

                # injected_cache = f"{LOCAL_BBQ_VALIDATE_DIR}/{axis}_injected_k={k_sentences}_br={b_ratio}.csv"

                os.makedirs("../cache", exist_ok=True)
                injected_cache = f"../cache/{axis}_injected_k={k_sentences}_b={b_ratio}.csv"

                if os.path.exists(injected_cache):
                    validation_df = pd.read_csv(injected_cache)
                else:
                    validation_df = inject_crows_bias_to_df(
                        validation_df, crows_df, axis,
                        num_sentences=k_sentences, bias_ratio=b_ratio
                    )
                    validation_df.to_csv(injected_cache, index=False)

                print(f"Running co-effs for {axis} on vector {vt} at {datetime.datetime.now()}")
                vector = SteeringVector.import_gguf(f'../vectors/{model_short_name}/{vt}/{axis}.gguf')
            except FileNotFoundError as e:
                print(f"Missing file in BBQ Validate (or in /vectors/) for this axis (and type): {axis} ({vt}).\n"
                      f"Error: {e}")
                continue

            # Save paths
            csv_name = f"{axis}_{vt}_k={k_sentences}_b={b_ratio}.csv"

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
            existing_csv = resume_logic(axis, remote_file_path, local_file_path, vt)

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

            all_coeffs = np.linspace(-2.0, 2.0, 21)
            remaining_coeffs = [c for c in all_coeffs if f"{c:.1f}" not in completed_coeffs]

            for coeff in tqdm(
                    remaining_coeffs,
                    desc=f"  Coeffs for {axis}: ",
                    total=21,  # max length
                    initial=len(completed_coeffs),  # initial step
                    leave=False,
                    dynamic_ncols=True,
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
                        'k-num-sentences': k_sentences,
                        'bias-ratio': b_ratio,
                        'coeff': round(coeff, 1),
                        'bbq_correct': int(bbq_correct),  # int
                        'mmlu_correct': int(mmlu_correct),  # int
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
                print(f"Completed {axis}: {len(results)} coefficients saved.\n")
        # for axes (files in /bbq_validate, rows in /best_layers)
    # for vector-type files (train, train+prompt)


if __name__ == "__main__":
    if check_paths():
        print('All path correctly checked')
        already_done = preview_status()  # print current status
        if not args.only_preview:  # if not Only preview
            if not already_done:
                get_best_coeffs(prepare_MMLU())  # Work
            else:
                print("Nothing to do. All work done")
    else:
        print('Something wrong')
