import pandas as pd
import os
import glob

import argparse


from utils import bbq_axes
from utils_new import get_args, get_model_short_name, EXPERIMENT

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1', help='model name')
parser.add_argument('-p', '--path', type=str, default=None, help='model path')
parser.add_argument('-q', '--quantization', action='store_true', help='Insert flag to quantize the model')

parser.add_argument('-a', '--axes', nargs='*', type=str, default=None, help='axes to be processed')
parser.add_argument('-k', '--k-sentences', type=int, default=0, help='Number of retrieved sentences')
parser.add_argument('-b', '--bias-ratio', type=float, default=0.5, help='Pro-stereotype sentences ratio (0.0 - 1.0)')
args = parser.parse_args()

QUANTIZATION = args.quantization
(model_name, model_path) = get_args([args.name, args.path])
model_short_name = get_model_short_name(model_name, quantized=QUANTIZATION)

# Injection parameters
k = args.k_sentences  # top-k sentences
b = args.bias_ratio  # fraction of pro-stereotyped sentences


def set_dir_paths():
    coeff_base_path = f'../data/coeff_scores/{model_short_name}/{EXPERIMENT}'
    config_base_path = f'../data/configs/{model_short_name}/{EXPERIMENT}'
    if k > 0 and EXPERIMENT not in ['reproduction', 'original']: # injections
        coeff_scores_dir = os.path.join(coeff_base_path, f"k-{k}_b-{b}") # add path
        print(f'Coefficient scores in: {coeff_scores_dir} [K = {k} | B = {b}]')

        config_dir_path = os.path.join(config_base_path, f"k-{k}_b-{b}")  # add path
        print(f'Configurations in: {config_dir_path} [K = {k} | B = {b}]')
        return coeff_scores_dir, config_dir_path

    elif k <= 0 and EXPERIMENT in ['reproduction', 'original']: # no injections
        print(f'Coefficient scores in: {coeff_base_path} [K = 0]')
        print(f'Configurations in: {config_base_path} [K = 0]')
        return coeff_base_path, config_base_path
    else: # error
        print(f'Error! Please check:\nExperiment = {EXPERIMENT}\nK = {k}\nB = {b}\n'
              f'Fallback on {coeff_base_path} and {config_base_path} but, please, check!')
        return coeff_base_path, config_base_path

COMPLETE_COEFF_DIR, CONFIG_DIR = set_dir_paths()

def generate_config_csvs():
    """Generate config CSV files for each folder with best results per axis."""

    # Define all axes
    # axes = ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic']
    if args.axes is not None:
        axes = args.axes.copy()  # list type
    else:
        axes = bbq_axes # all BBQ  axes from utils
    print(f'\n{len(axes)}configurations to be processed: {axes}\n')

    # Get all folders in coeff_scores/mistral
    coeff_VT_dirs = [d for d in os.listdir(COMPLETE_COEFF_DIR) if os.path.isdir(os.path.join(COMPLETE_COEFF_DIR, d))]
    coeff_VT_dirs.sort()

    # Create configs directory if it doesn't exist
    os.makedirs('../data/configs', exist_ok=True)

    for vt_dir in coeff_VT_dirs:
        print(f"\nProcessing folder: {vt_dir}")

        config_data = []

        for axis in axes:
            print(f'Processing axis: {axis}')
            # Load the corresponding best layers file

            best_layers_file = f"../data/layer_scores/{model_short_name}/best_layers/{vt_dir}.csv"
            if os.path.exists(best_layers_file):
                best_layers_df = pd.read_csv(best_layers_file)
                # Find the row for this axis
                axis_row = best_layers_df[best_layers_df['axis'] == axis]
                if not axis_row.empty:
                    layer = axis_row.iloc[0]['max_layer']
                    vector_type = axis_row.iloc[0]['vt']
                else:
                    layer = None
                    vector_type = None
            else:
                layer = None
                vector_type = None

            # Find CSV files for this axis in this folder
            coeff_csv = f"{COMPLETE_COEFF_DIR}/{vt_dir}/{axis}_*.csv"

            csv_files = glob.glob(coeff_csv)

            if csv_files and layer is not None:
                # Process the CSV file for this axis
                csv_file = csv_files[0]  # Should only be one file per axis per folder
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Find row with maximum BBQ accuracy
                    max_bbq_row = df.loc[df['bbq_accuracy'].idxmax()]

                    config_data.append({
                        'axis': axis,
                        'vector_type': vector_type,
                        'layer': layer,
                        'coeff': max_bbq_row['coeff'],
                        'bbq_accuracy': max_bbq_row['bbq_accuracy'],
                        'mmlu_accuracy': max_bbq_row['mmlu_accuracy']
                    })

                    print(f'{axis} configuration has been correctly processed\n')

        # Save config CSV for this folder
        if config_data:
            config_df = pd.DataFrame(config_data)
            config_file = f"../data/configs/{vt_dir}.csv"
            config_df.to_csv(config_file, index=False)
            print(f"  Saved {len(config_data)} configs to {config_file}")
        else:
            print(f"  No data found for folder {vt_dir}")


def generate_baseline_csv():
    """Generate the baselines.csv config by pulling the coeff=0 row out of
    each axis' coefficient sweep (5_optimize_coeff.py output), instead of
    picking the max bbq_accuracy row like generate_config_csvs() does.

    The original baselines.csv always uses the "train+prompt" vector type
    for every axis. This is arbitrary (coeff=0 nullifies the steering
    vector's effect entirely), but we mirror it here for reproducibility.
    """
    BASELINE_TOP_VT = "top_train+prompt"

    if args.axes is not None:
        axes = args.axes.copy()
    else:
        axes = bbq_axes
    print(f'\n{len(axes)} axes to be processed for baseline: {axes}\n')

    best_layers_file = f"../data/layer_scores/{model_short_name}/best_layers/{BASELINE_TOP_VT}.csv"
    if not os.path.exists(best_layers_file):
        print(f"Missing best layers file: {best_layers_file}. Skipping baseline generation.")
        return
    best_layers_df = pd.read_csv(best_layers_file)

    config_data = []

    for axis in axes:
        print(f'Processing axis: {axis}')

        axis_row = best_layers_df[best_layers_df['axis'] == axis]
        if axis_row.empty:
            print(f'  No best-layer entry for {axis} in {BASELINE_TOP_VT}, skipping.')
            continue

        layer = axis_row.iloc[0]['max_layer']
        vector_type = axis_row.iloc[0]['vt']  # should be 'train+prompt'

        coeff_csv_pattern = f"{COMPLETE_COEFF_DIR}/{BASELINE_TOP_VT}/{axis}_*.csv"
        csv_files = glob.glob(coeff_csv_pattern)

        if not csv_files:
            print(f'  No coeff-score file found for {axis} (pattern: {coeff_csv_pattern}), skipping.')
            continue

        csv_file = csv_files[0]
        df = pd.read_csv(csv_file)
        if df.empty:
            print(f'  Empty coeff-score file for {axis}, skipping.')
            continue

        baseline_rows = df[df['coeff'].round(1) == 0.0]
        if baseline_rows.empty:
            print(f'  No coeff=0 row found for {axis} in {csv_file}, skipping.')
            continue

        baseline_row = baseline_rows.iloc[0]

        config_data.append({
            'axis': axis,
            'vector_type': vector_type,
            'layer': layer,
            'coeff': 0,
            'bbq_accuracy': baseline_row['bbq_accuracy'],
            'mmlu_accuracy': baseline_row['mmlu_accuracy']
        })

        print(f'{axis} baseline correctly processed\n')

    if config_data:
        os.makedirs(f'../{CONFIG_DIR}', exist_ok=True)
        config_df = pd.DataFrame(config_data)
        config_file = f"../{CONFIG_DIR}/baselines.csv"
        config_df.to_csv(config_file, index=False)
        print(f"  Saved {len(config_data)} baseline configs to {config_file}")
    else:
        print("  No baseline data produced.")


if __name__ == "__main__":
    generate_baseline_csv()
    generate_config_csvs()
    print("\nConfig CSV generation complete!")