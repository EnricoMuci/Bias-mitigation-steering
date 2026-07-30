import argparse

import datetime
import math
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shutil
from transformers import AutoConfig
from dialz.vector import SteeringModule

from dialz import SteeringVector, SteeringModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import os
import torch
import tqdm

from utils import load_and_tokenize_contrastive, get_output, bbq_axes
from utils_new import (get_args, get_model_short_name, define_custom_tokenizer, configure_model,
                       EXPERIMENT, REMOTE_DRIVE_THESIS_PROJECT, model_layer_list) # FIXME: model_layer_list importing



import warnings
# ✓ ✗ … ○
warnings.filterwarnings(
    "ignore",
    message="_check_is_size will be removed",
    category=FutureWarning
)
transformers.logging.set_verbosity_error()

import zoneinfo

tz_set = zoneinfo.ZoneInfo("Europe/Rome") # FIXME : Remove

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1', help='model name')
parser.add_argument('-p', '--path', type=str, default=None, help='model path')
parser.add_argument('-q', '--quantization', action='store_true', help='Insert flag to quantize the model')

parser.add_argument('-t', '--type', type=int, default=2, help='train[+prompt] → in get_acc_change_per_layer')
parser.add_argument('-a', '--axes', nargs='*', type=str, default=None, help='axes to be processed')
parser.add_argument('-m', '--mode', nargs='*', type=str, default=[], help="['layer', 'separability', 'best']")

parser.add_argument('-c', '--colab', action='store_true', help='executing on Colab')
parser.add_argument('-o', '--only-preview', action='store_true', help='only preview')
args = parser.parse_args()

OPERATIONS = args.mode

if args.axes is not None:
    chosen_axes = args.axes.copy()  # list type
else:
    chosen_axes = bbq_axes

QUANTIZATION = args.quantization
VECTOR_TYPES = ["train", "train+prompt"]

(model_name, model_path) = get_args([args.name, args.path])
model_short_name = get_model_short_name(model_name, quantized=QUANTIZATION)
tokenizer = define_custom_tokenizer(model_name, model_path)

FIGS_PATH = f"../figs/{model_short_name}/4-layers"
LAYERS_PATH = f"../data/layer_scores/{model_short_name}"
SEPARABILITY_PATH = f"../data/separability_scores/{model_short_name}"

os.makedirs(FIGS_PATH, exist_ok=True)
os.makedirs(SEPARABILITY_PATH, exist_ok=True)
os.makedirs(LAYERS_PATH, exist_ok=True)


def preview_status():
    config = AutoConfig.from_pretrained(model_path)
    num_layers = getattr(config, "n_layer", None) or config.num_hidden_layers

    if args.type == 2:
        set_types = VECTOR_TYPES.copy()
    else:
        set_types = [VECTOR_TYPES[args.type]]

    # ── SEPARABILITY ──────────────────────────────────────────────
    if 'separability' in OPERATIONS:
        print("\n" + "=" * 65)
        print("SEPARABILITY STATUS  (get_linear_separability)")
        print("=" * 65)

        sep_data = []
        for axis in chosen_axes:
            if axis not in bbq_axes:
                continue
            for vt in ["train", "train+prompt"]:
                csv_path = f"{SEPARABILITY_PATH}/{axis}_{vt}.csv"
                png_path = f"{FIGS_PATH}/{axis}_bbq_{vt}.png"
                csv_ok = os.path.exists(csv_path)
                png_ok = os.path.exists(png_path)

                if csv_ok and png_ok:
                    df = pd.read_csv(csv_path)
                    status = "Complete"
                    details = f"{len(df)} layers"
                elif csv_ok or png_ok:
                    status = "Partial"
                    details = f"csv={'Yes' if csv_ok else 'No'}, png={'Yes' if png_ok else 'No'}"
                else:
                    status = "Not started"
                    details = "-"

                sep_data.append({
                    "Axis": axis,
                    "Vector": vt,
                    "Status": status,
                    "Details": details
                })

        if sep_data:
            sep_df = pd.DataFrame(sep_data)
            print(sep_df.to_string(index=False))
            if all(s["Status"] == "Complete" for s in sep_data):
                print("\nAll separability files are done.")
        else:
            print("No axes selected for separability.")

    # ── LAYER ACCURACY ────────────────────────────────────────────
    if 'layer' in OPERATIONS:
        print("\n" + "=" * 65)
        print(f"LAYER ACCURACY STATUS  (get_acc_change_per_layer)")
        print(f"Total layers: {num_layers - 1}  (layer 1 → {num_layers - 1})")
        print("=" * 65)

        layer_data = []
        expected = num_layers - 1

        for axis in chosen_axes:
            if axis not in bbq_axes:
                continue
            for vt in set_types:
                local_file = f"{LAYERS_PATH}/{axis}_{vt}.csv"
                found_path = None
                source = "-"

                if args.colab:
                    remote_file = (f"{REMOTE_DRIVE_THESIS_PROJECT}/data/layer_scores/"
                                   f"{model_short_name}-{EXPERIMENT}/{axis}_{vt}.csv")
                    if os.path.exists(remote_file):
                        found_path = remote_file
                        source = "Drive"

                if found_path is None and os.path.exists(local_file):
                    found_path = local_file
                    source = "Local"

                if found_path is None:
                    status = "Not started"
                    progress = f"0/{expected}"
                else:
                    df = pd.read_csv(found_path)
                    done = len(df)
                    progress = f"{done}/{expected}"

                    if done >= expected:
                        status = "Complete"
                    else:
                        next_layer = int(df['layer'].max()) + 1
                        status = f"Partial (Resume from {next_layer})"

                layer_data.append({
                    "Axis": axis,
                    "Vector": vt,
                    "Status": status,
                    "Progress": progress,
                    "Source": source
                })

        if layer_data:
            layer_df = pd.DataFrame(layer_data)
            print(layer_df.to_string(index=False))
            if all("Complete" in s["Status"] for s in layer_data):
                print("\nAll layers are complete.")
        else:
            print("No axes selected for layers.")

    print("\n" + "=" * 65 + "\n")

def old_preview_status():
    config = AutoConfig.from_pretrained(model_path)
    num_layers = getattr(config, "n_layer", None) or config.num_hidden_layers


    if args.type == 2:
        set_types = VECTOR_TYPES.copy()
    else:
        set_types = [VECTOR_TYPES[args.type]]

    # ── SEPARABILITY ──────────────────────────────────────────────
    if 'separability' in OPERATIONS:
        print("\n" + "=" * 55)
        print("SEPARABILITY STATUS  (get_linear_separability)")
        print("=" * 55)

        sep_all_done = True
        for axis in chosen_axes:
            if axis not in bbq_axes:
                continue
            for vt in ["train", "train+prompt"]:
                csv_path = f"{SEPARABILITY_PATH}/{axis}_{vt}.csv"
                png_path = f"{FIGS_PATH}/{axis}_bbq_{vt}.png"
                csv_ok = os.path.exists(csv_path)
                png_ok = os.path.exists(png_path)

                if csv_ok and png_ok:
                    df = pd.read_csv(csv_path)
                    print(f"  ✓ {axis:15s}  {vt:12s}   →  complete ({len(df)} layer)")
                elif csv_ok or png_ok:
                    print(f"  ○ {axis:15s}  {vt:12s}   →  partial " 
                          f"(csv={'✓' if csv_ok else '✗'}, png={'✓' if png_ok else '✗'})")
                    sep_all_done = False
                else:
                    print(f"  ✗ {axis:15s} ({vt:12s})  →  not started") # ○
                    sep_all_done = False

        if sep_all_done:
            print("\n  All separability files are done.")

    # ── LAYER ACCURACY ────────────────────────────────────────────
    if 'layer' in OPERATIONS:
        print("\n" + "=" * 55)
        print(f"LAYER ACCURACY STATUS  (get_acc_change_per_layer)")
        print(f"Total layers: {num_layers - 1}  (layer 1 → {num_layers - 1})")
        print("=" * 55)

        layer_all_done = True
        for axis in chosen_axes:
            if axis not in bbq_axes:
                continue
            for vt in set_types:
                local_file = f"{LAYERS_PATH}/{axis}_{vt}.csv"

                found_path = None
                source = ""
                if args.colab:
                    remote_file = (f"{REMOTE_DRIVE_THESIS_PROJECT}/data/layer_scores/"
                                   f"{model_short_name}-{EXPERIMENT}/{axis}_{vt}.csv")
                    if os.path.exists(remote_file):
                        found_path = remote_file
                        source = "Drive"
                if found_path is None and os.path.exists(local_file):
                    found_path = local_file
                    source = "Local"

                if found_path is None:
                    print(f"  ○ {axis:15s} ({vt:12s})  →  not started")
                    layer_all_done = False
                else:
                    df = pd.read_csv(found_path)
                    done = len(df)
                    expected = num_layers - 1
                    if done >= expected:
                        print(f"  ✓ {axis:15s} ({vt:12s})  →  complete "
                              f"({done}/{expected} layer) [{source}]")
                    else:
                        next_layer = int(df['layer'].max()) + 1
                        print(f"  … {axis:15s} ({vt:12s})  →  partial "
                              f"({done}/{expected} layers done, resuming from layer {next_layer}) [{source}]")
                        layer_all_done = False

        if layer_all_done:
            print("\n  All layers are complete.")

    print("=" * 55 + "\n")


def batched_get_hiddens(
        model,
        tokenizer,
        inputs: list[str],
        hidden_layers: list[int],  # format: [1, ..., l, ..., L]
        batch_size: int,
        pooling: str = 'final'  # 'final' or 'mean'
) -> dict[int, np.ndarray]:
    """
    Extract hidden states for each example and layer, with optional pooling.

    Args:
        model: a HuggingFace model with output_hidden_states=True
        tokenizer: corresponding tokenizer
        inputs: list of input strings
        hidden_layers: indices of layers to extract (0-based)
        batch_size: inference batch size
        pooling: 'final' to take last non-pad token; 'mean' to average all tokens

    Returns:
        dict mapping layer -> array of shape (len(inputs), hidden_dim)
    """
    batched_inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
    hidden_states = {layer: [] for layer in hidden_layers}  # dictionary  int: []

    with torch.no_grad():
        for batch in tqdm.tqdm(batched_inputs, desc="Getting hiddens"):
            encoded = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
            out = model(**encoded, output_hidden_states=True)
            mask = encoded['attention_mask']  # shape (B, seq_len)

            for i in range(len(batch)):
                for layer in hidden_layers:
                    hidden_idx = layer + 1 if layer >= 0 else layer
                    states = out.hidden_states[hidden_idx][i]  # (seq_len, D)
                    if pooling == 'final':
                        last_idx = mask[i].nonzero(as_tuple=True)[0][-1].item()
                        vec = states[last_idx].cpu().float().numpy()
                    else:  # mean pooling
                        m = mask[i].unsqueeze(-1).float()  # (seq_len, 1)
                        summed = (states * m).sum(dim=0)
                        denom = m.sum()
                        vec = (summed / denom).cpu().float().numpy()
                    hidden_states[layer].append(vec)
            del out

    return {k: np.vstack(v) for k, v in hidden_states.items()}


def visualize_2d_PCA(
        inputs,
        model,
        tokenizer,
        pooling: str = 'final',  # 'final' or 'mean'
        n_cols: int = 5,
        batch_size: int = 32
):
    """
    Perform 2D PCA on the hidden states of positive vs negative examples for each layer,
    plot all layers in a grid, and compute linear separability using a logistic classifier.
    Pooling can be 'final' or 'mean'.
    """
    # Prepare layers and strings
    hidden_layers = list(range(1, model.config.num_hidden_layers))
    train_strs = [s for ex in inputs.entries for s in (ex.positive, ex.negative)]

    # Extract hidden states
    layer_hiddens = batched_get_hiddens(
        model, tokenizer, train_strs, hidden_layers, batch_size, pooling=pooling
    )

    # Setup subplot grid
    n_layers = len(hidden_layers)
    n_rows = math.ceil(n_layers / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 3),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    scores = []

    reference_components = None  # NEW to avoid flipping

    # Loop over layers
    for idx, layer in enumerate(tqdm.tqdm(hidden_layers, desc="PCA & Classify")):
        ax = axes[idx]
        h_states = layer_hiddens[layer]  # shape (2N, D)
        # diffs for PCA axis
        diffs = h_states[::2] - h_states[1::2]  # shape (N, D)

        # 2-component PCA fitted on diffs
        pca2 = PCA(n_components=2, whiten=False).fit(diffs)  # fit(diffs)

        if reference_components is None:  # NEW: avoid flipping
            # first layer
            signs = np.sign(pca2.components_[np.arange(2), np.argmax(np.abs(pca2.components_), axis=1)])
            pca2.components_ *= signs[:, np.newaxis]

            # pca2.components_, _ = svd_flip(pca2.components_.T, np.zeros_like(pca2.components_))
            # pca2.components_ = pca2.components_.T
            # END: avoid flipping
        else:  # from second layer to last one
            for k in range(2):
                if np.dot(pca2.components_[k], reference_components[k]) < 0:
                    pca2.components_[k] *= -1
        reference_components = pca2.components_.copy() # fix

        proj_all = pca2.transform(h_states)  # project all 2N on PC1/PC2

        # scatter positives vs negatives
        colors = ['orange' if i % 2 == 0 else 'blue' for i in range(proj_all.shape[0])]
        ax.scatter(proj_all[:, 0], proj_all[:, 1], c=colors, s=8, alpha=0.6)
        ax.axhline(0, color='gray', lw=0.8)
        ax.axvline(0, color='gray', lw=0.8)

        # Compute linear separability on full hidden states
        labels = [1 if i % 2 == 0 else 0 for i in range(h_states.shape[0])]
        clf = LogisticRegression(max_iter=500)
        # cross-validate accuracy

        try:
            sep = cross_val_score(LogisticRegression(max_iter=500), proj_all, labels, cv=5).mean()
        except:
            sep = clf.score(proj_all, labels)
        scores.append({'layer': layer, 'sep_score': sep})

        # Annotate with separability
        ax.set_title(f"L{layer}, Acc={sep:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off unused axes
    for j in range(n_layers, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    df_scores = pd.DataFrame(scores)

    return fig, df_scores


def get_linear_separability():
    model = configure_model(model_name, model_path, quantized=QUANTIZATION)

    ## Feed in model tokenizer and inputs from each contrastive dataset in python file 3
    for axis in chosen_axes:
        if axis not in bbq_axes:  # NEW
            print(f'Axis {axis} is not admitted')
            continue
        print(f"Creating vector for {axis} at:", datetime.datetime.now(tz=tz_set))
        bbq_train_path = f"../data/bbq_train/{axis}_train.json"

        train_dataset = load_and_tokenize_contrastive(model_path, bbq_train_path)
        fig, scores = visualize_2d_PCA(train_dataset, model, tokenizer)
        scores.to_csv(f"{SEPARABILITY_PATH}/{axis}_train.csv", index=False)
        fig.savefig(f"{FIGS_PATH}/{axis}_bbq_train.png")

        train_dataset = load_and_tokenize_contrastive(model_path, bbq_train_path,
                                                      prompt=f"Consider the bias related to {axis} in the following. ")
        fig, scores = visualize_2d_PCA(train_dataset, model, tokenizer)
        scores.to_csv(f"{SEPARABILITY_PATH}/{axis}_train+prompt.csv", index=False)
        fig.savefig(f"{FIGS_PATH}/{axis}_bbq_train+prompt.png")


## Get baseline first
def predict_row(row, model, vector, coeff):
    context = row["context"]
    question = row["question"]
    answers = [row["ans0"], row["ans1"], row["ans2"]]

    input_text = (
        f"Answer with only a single letter.\n\n"
        f"The following are multiple choice questions (with answers):\n"
        f"{context} {question}\n"
        f"(a) {answers[0]}\n"
        f"(b) {answers[1]}\n"
        f"(c) {answers[2]}\n"
    )

    generated_answer = get_output(input_text, model, tokenizer, vector, coeff).lower()

    predicted_label = -1
    for i, letter in enumerate(["a", "b", "c"]):
        if letter in generated_answer[:2]:
            predicted_label = i
            break
    if predicted_label == -1:
        answers = [row["ans0"], row["ans1"], row["ans2"]]
        for i, answer in enumerate(answers):
            if answer in generated_answer:
                predicted_label = i
                break

    correct = (predicted_label == row["label"])

    return pd.Series({
        "ans": generated_answer,
        "prediction": predicted_label,
        "correct": correct
    })


def get_acc_change_per_layer():
    config = AutoConfig.from_pretrained(model_path)
    num_layers = getattr(config, "n_layer", None) or config.num_hidden_layers
    model = configure_model(model_name, model_path, quantized=QUANTIZATION)
    # model.half()

    all_types = ["train", "train+prompt"]  # NEW Block
    if args.type == 2:
        set_types = all_types.copy()
    else:
        set_types = [all_types[args.type], ]

    for axis in chosen_axes:
        if axis not in bbq_axes:  # NEW
            print(f'Axis {axis} is not admitted')
            continue

        # Load in validation set
        validation_df = pd.read_csv(f"../data/bbq_validate/{axis}_validate.csv")

        # for each of our vectors
        for vector_type in set_types:  # ["train", "train+prompt"]:
            output_file = f"{LAYERS_PATH}/{axis}_{vector_type}.csv"

            remote_file = f"{REMOTE_DRIVE_THESIS_PROJECT}/data/layer_scores/{model_short_name}-{EXPERIMENT}/{axis}_{vector_type}.csv"
            # start_layer = 1
            results = []

            print(' ')
            # 1. Initialization from Drive to Colab (solo se --colab)
            if args.colab:
                try:
                    if os.path.exists(remote_file):
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        shutil.copy2(remote_file, output_file)
                except Exception as e:
                    print(f"Drive Copy failed: {e}. Directly reading from Drive.")
                    if os.path.exists(remote_file):
                        output_file = remote_file  # fallback: leggi da Drive direttamente

            # Resume logic
            if os.path.exists(output_file):
                try:
                    existing_df = pd.read_csv(output_file)
                    if len(existing_df) >= num_layers - 1:
                        print(f"Skipping {axis} - {vector_type}: already complete.")
                        continue

                    done_layers = set(existing_df['layer'].tolist())
                    all_layers = set(range(1, num_layers))
                    missing = sorted(all_layers - done_layers)

                    if not missing:
                        print(f"Skipping {axis} - {vector_type}: already complete.")
                        continue

                    start_layer = missing[0]  # NEW first missing layer

                    results = existing_df.to_dict('records')
                    print(f"Resuming {axis} - {vector_type} from layer {start_layer}...")
                except Exception:
                    raise
            else:
                print(f"Processing all layers for {axis} on vector {vector_type}")
                missing = list(range(1, num_layers))

            vector = SteeringVector.import_gguf(f'../vectors/{model_short_name}/{vector_type}/{axis}.gguf')

            for layer in missing:
                try:
                    bbq_df = validation_df.copy()

                    # vector = SteeringVector.import_gguf(f'../vectors/{model_short_name}/{vector_type}/{axis}.gguf')

                    # Layer unwrapping : NEW
                    layers = model_layer_list(model.model)
                    for old_id in model.layer_ids:
                        old_layer = layers[old_id]
                        if isinstance(old_layer, SteeringModule):
                            layers[old_id] = old_layer.block  # unwrap

                    model.layer_ids = [layer]
                    if not isinstance(layers[layer], SteeringModule):
                        layers[layer] = SteeringModule(layers[layer])
                    # END NEW Wrapping

                    start_time = datetime.datetime.now(tz=tz_set)
                    print(f"\n\n=== layer = {layer} @ {start_time} ===")

                    # apply the predictor to every row
                    bbq_df[['ans', 'prediction', 'correct']] = bbq_df.apply(
                        predict_row,
                        axis=1,
                        args=(model, vector, 1)
                    )

                    bbq_correct = (bbq_df["prediction"] == bbq_df["label"]).sum()
                    bbq_accuracy = bbq_correct / len(bbq_df)

                    results.append({
                        'layer': layer,
                        'bbq_correct': int(bbq_correct),
                        'bbq_accuracy': float(bbq_accuracy),
                    })

                    # Save in the CSV at each layer calculation
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(output_file, index=False)

                    # 2. Backup: Copy updated file from Colab to Drive (solo se --colab)
                    if args.colab:
                        try:
                            os.makedirs(os.path.dirname(remote_file), exist_ok=True)
                            if os.path.realpath(remote_file) != os.path.realpath(output_file):
                                shutil.copy2(output_file, remote_file)
                        except Exception as e:
                            print(f"Error saving in Drive: {e}")

                except Exception as e:
                    print(f"[ERROR in layer {layer}] {type(e).__name__}: {e} — next layer.")
                    continue


def find_best_layers():

    # csv_path = f'../data/separability_scores/{model_short_name}/'

    results_train = []
    results_train_prompt = []

    for axis in bbq_axes:
        for vt in VECTOR_TYPES: # train and train+prompt
            acc_df = pd.read_csv(f"{LAYERS_PATH}/{axis}_{vt}.csv")
            max_acc = acc_df['bbq_accuracy'].max()
            max_layer = acc_df.loc[acc_df['bbq_accuracy'].idxmax(), 'layer']

            entry = {
                'axis': axis,
                'vt': vt,
                'max_acc': round(max_acc, 3),
                'max_layer': max_layer
            }

            if vt == "train":
                results_train.append(entry)
            elif vt == "train+prompt":
                results_train_prompt.append(entry)

    results_train_df = pd.DataFrame(results_train)
    results_train_prompt_df = pd.DataFrame(results_train_prompt)

    print("=== TRAIN ===")
    print(results_train_df)
    print("\n=== TRAIN+PROMPT ===")
    print(results_train_prompt_df)

    os.makedirs(f'{LAYERS_PATH}/best_layers', exist_ok=True)
    results_train_df.to_csv(f'{LAYERS_PATH}/best_layers/top_train.csv', index=False)
    results_train_prompt_df.to_csv(f'{LAYERS_PATH}/best_layers/top_train+prompt.csv', index=False)

    # results_train_df.to_csv(f'{LAYERS_PATH}/best_layers/top_train.csv')
    # results_train_prompt_df.to_csv(f'{LAYERS_PATH}/best_layers/top_train+prompt.csv')

    # results_train_df.to_csv(
    #     f'{colab_repo_dir}/{layers_path}/{model_short_name}/best_layers/top_train.csv')  # save results in colab (train)
    # results_train_prompt_df.to_csv(
    #     f'{colab_repo_dir}/{layers_path}/{model_short_name}/best_layers/top_train+prompt.csv')  # save results in Colab (train+prompt)


if __name__ == '__main__':  # FIXME
    preview_status()

    if not args.only_preview:
        if 'separability' in OPERATIONS:
            get_linear_separability()
        if 'layer' in OPERATIONS:
            get_acc_change_per_layer()
        if 'best' in OPERATIONS:
            find_best_layers()