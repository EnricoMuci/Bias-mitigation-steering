import argparse
import sys
import os
import gc
import pandas as pd
import logging
from datetime import datetime
from tqdm import tqdm
import torch
from dialz import SteeringVector

from utils_new import get_args, get_model_short_name, define_custom_tokenizer, configure_model, \
    set_steering_layer

# Import functions from the individual evaluation files
# Note: Using importlib because Python module names can't start with numbers
from importlib import import_module
import glob

# Registry describing each pluggable evaluation: which module/function to
# call, which axes it's relevant for (None = all axes), and the column
# prefix used when writing results. Modules are imported lazily (only the
# ones actually selected via --evals) so unused evaluations don't pay their
# import cost / dependency requirements.
EVAL_REGISTRY = {
    'bbq': {
        'module_name': '6_bbq_evaluation',
        'func_attr': 'run_bbq_evaluation',
        'relevant_axes': None,  # all axes
        'prefix': 'bbq',
        'display_name': 'BBQ',
    },
    'mmlu': {
        'module_name': '7_mmlu_evaluation',
        'func_attr': 'run_mmlu_evaluation',
        'relevant_axes': None,  # all axes
        'prefix': 'mmlu',
        'display_name': 'MMLU',
    },
    'stereoset': {
        'module_name': '8_stereoset_evaluation',
        'func_attr': 'run_stereoset_evaluation',
        'relevant_axes': ['gender', 'race', 'religion'],
        'prefix': 'stereoset',
        'display_name': 'StereoSet',
    },
    'crows': {
        'module_name': '9_crows_pairs_evaluation',
        'func_attr': 'run_crows_pairs_evaluation',
        'relevant_axes': ['gender', 'race', 'religion', 'age', 'nationality', 'socioeconomic', 'appearance',
                          'disability'],
        'prefix': 'crows',
        'display_name': 'CrowS-Pairs',
    },
    'clearbias': {
        'module_name': '10_clear_bias_evaluation',
        'func_attr': 'run_clear_bias_evaluation',
        'relevant_axes': ['gender', 'race', 'age', 'disability', 'religion', 'socioeconomic'],
        'prefix': 'clear_bias',
        'display_name': 'Clear Bias',
    },
}

_loaded_modules = {}


def get_eval_module(eval_key):
    """Lazily import an evaluation module the first time it's needed."""
    if eval_key not in _loaded_modules:
        _loaded_modules[eval_key] = import_module(EVAL_REGISTRY[eval_key]['module_name'])
    return _loaded_modules[eval_key]


# Global configuration
USE_FAIRNESS_PROMPT = False  # Set to True to enable fairness prompting
USE_SELF_DEBIAS = True  # Set to True to enable self-debiasing

# ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1')  # model name
parser.add_argument('-p', '--path', type=str, default=None)  # model path
parser.add_argument('-c', '--colab', action='store_true')  # flag about remote saving
parser.add_argument('-e', '--evals', nargs='*', choices=list(EVAL_REGISTRY.keys()),
                    default=list(EVAL_REGISTRY.keys()),
                    help='Which evaluations to run (default: all). E.g. --evals bbq mmlu')
parser.add_argument('--config', type=str, default=None,
                    help='Path to a single config CSV to evaluate. Default: every *.csv in ../data/configs/')
args = parser.parse_args()

(model_name, model_path) = get_args([args.name, args.path])
model_short_name = get_model_short_name(model_name)

tokenizer = define_custom_tokenizer(model_name, model_path)


def _eval_already_done(existing_df, axis, eval_key):
    """Check whether a specific evaluation already has a saved result for
    this axis (used to resume after an interruption without recomputing
    evaluations that already succeeded)."""
    if existing_df is None or axis not in existing_df.index:
        return False
    row = existing_df.loc[axis]
    prefix = EVAL_REGISTRY[eval_key]['prefix']
    matching_cols = [c for c in existing_df.columns if c.startswith(f"{prefix}_")]
    if not matching_cols:
        return False
    return any(pd.notna(row.get(c)) for c in matching_cols)


def run_evaluations_for_config(config_file, model):
    """Run all evaluations for a given config file by calling functions from individual files.

    `model` is a single QuantizedSteeringModel instance, loaded once by the
    caller and reused across every axis/config file: only the steered layer
    is swapped between axes (see set_steering_layer), not the whole model.
    """
    config_name = os.path.basename(config_file).replace('.csv', '')
    print(f"\nRunning evaluations for config: {config_name}")

    # Load config
    config_df = pd.read_csv(config_file)
    print(f"Loaded {len(config_df)} configurations")

    os.makedirs(f"../results/{model_short_name}", exist_ok=True)
    results_file = f"../results/{model_short_name}/{config_name}.csv"

    existing_df = None
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file).set_index('axis', drop=False)
        print(f"Found existing results file ({len(existing_df)} rows) — resuming, "
              f"already-completed evaluations will be skipped.")

    def checkpoint(row):
        """Merge one axis' result into the results file on disk immediately,
        so an interrupted job (SLURM time limit, OOM...) only loses the axis
        currently in progress, not the whole config file.

        NOTE: existing_df.loc[new_df.index, col] = ... (an Index, even of
        length 1) raises KeyError when the label isn't present yet --
        .loc only auto-creates rows for a *scalar* label, not a list-like
        indexer. This is why earlier runs crashed on the second axis
        checkpointed (the first hit the `existing_df is None` branch and
        never exercised this path)."""
        nonlocal existing_df
        axis_value = row['axis']
        if existing_df is None:
            existing_df = pd.DataFrame([row]).set_index('axis', drop=False)
        elif axis_value in existing_df.index:
            for key, value in row.items():
                existing_df.loc[axis_value, key] = value
        else:
            new_row_df = pd.DataFrame([row]).set_index('axis', drop=False)
            existing_df = pd.concat([existing_df, new_row_df])
        existing_df.reset_index(drop=True).to_csv(results_file, index=False)

    requested_evals = [e for e in EVAL_REGISTRY if e in args.evals]

    for _, config_row in tqdm(config_df.iterrows(), total=len(config_df), desc="Total Configs Progress", position=0):
        axis = config_row['axis']
        vector_type = config_row['vector_type']
        layer = int(config_row['layer'])
        coeff = config_row['coeff']
        bbq_accuracy = config_row['bbq_accuracy']
        mmlu_accuracy = config_row['mmlu_accuracy']

        pending_evals = [e for e in requested_evals if not _eval_already_done(existing_df, axis, e)]
        if not pending_evals:
            print(f"\n  Skipping {axis}: all requested evaluations already completed (resumed).")
            continue

        print(f"\n  Processing {axis} (layer={layer}, coeff={coeff})... pending: {pending_evals}")

        # Check if vector file exists before proceeding
        vector_path = f'../vectors/{model_short_name}/{vector_type}/{axis}.gguf'
        if not os.path.exists(vector_path):
            print(f"    Skipping {axis}: Vector file not found at {vector_path}")
            continue

        # Reuse the already-loaded model: just swap which layer is wrapped
        # in SteeringModule, instead of reloading the whole 7B model.
        set_steering_layer(model, layer)
        vector = SteeringVector.import_gguf(vector_path)

        # Initialize result row with config data
        result_row = {
            'axis': axis,
            'vector_type': vector_type,
            'layer': layer,
            'coeff': coeff,
            'bbq_accuracy': bbq_accuracy,
            'mmlu_accuracy': mmlu_accuracy,
            'fairness_prompt': USE_FAIRNESS_PROMPT,
            'self_debias': USE_SELF_DEBIAS
        }

        # Run every pending evaluation (already-completed ones were filtered out above)
        for eval_key in pending_evals:
            eval_info = EVAL_REGISTRY[eval_key]
            relevant_axes = eval_info['relevant_axes']  # None = all axes

            if relevant_axes is not None and axis not in relevant_axes:
                print(f"    Skipping {eval_info['display_name']} evaluation (axis '{axis}' not relevant)")
                result_row[f"{eval_info['prefix']}_skipped"] = True
                continue

            try:
                print(f"    Running {eval_info['display_name']} evaluation...")
                eval_module = get_eval_module(eval_key)
                eval_func = getattr(eval_module, eval_info['func_attr'])
                eval_result = eval_func(model, vector, coeff, axis, tokenizer, USE_FAIRNESS_PROMPT, USE_SELF_DEBIAS)
                print(f"      {eval_info['display_name']} evaluation completed")
                if eval_result:
                    for key, value in eval_result.items():
                        if key not in ['axis', 'coeff']:  # Don't duplicate these
                            result_row[f"{eval_info['prefix']}_{key}"] = value
            except Exception as e:
                print(f"      Error in {eval_info['display_name']} evaluation: {e}")
                result_row[f"{eval_info['prefix']}_error"] = str(e)

        # Checkpoint right away: this axis is now safe on disk even if the
        # job dies on the very next one.
        checkpoint(result_row)
        print(f"    Checkpoint saved for axis '{axis}' -> {results_file}")

        # Free only the vector — the model stays loaded and gets reused by
        # the next axis via set_steering_layer().
        del vector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nAll evaluations complete for config: {config_name}")
    print(f"Evaluations run: {requested_evals}")
    print(f"Results saved to {results_file}")


def setup_logging():
    """Set up logging to redirect all output to a log file."""
    # Create timestamp for unique log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"../logs/evaluation_run_{timestamp}.log"

    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)  # Also show on console
        ]
    )

    # Redirect print statements to logging
    class PrintToLog:
        def write(self, text):

            if '\r' in text:
                sys.__stderr__.write(text)
                sys.__stderr__.flush()
            elif text.strip():  # Only log non-empty lines
                logging.info(text.strip())

        def flush(self):
            sys.__stderr__.flush()

    sys.stdout = PrintToLog()
    sys.stderr = PrintToLog()

    return log_file


def main():
    """Run evaluations for every config file in ../data/configs/ (or a single
    one if --config is passed), for the evaluations selected via --evals."""
    log_file = setup_logging()
    print(f"Logging to: {log_file}")
    print(f"Fairness prompting enabled: {USE_FAIRNESS_PROMPT}")
    print(f"Self-debiasing enabled: {USE_SELF_DEBIAS}")
    print(f"Evaluations selected: {args.evals}")

    if args.config:
        config_files = [args.config]
        if not os.path.exists(config_files[0]):
            raise FileNotFoundError(f"Config file not found: {config_files[0]}")
    else:
        config_files = sorted(glob.glob("../data/configs/*.csv"))
        if not config_files:
            raise FileNotFoundError("No config files found in ../data/configs/")

    print(f"Config files to process: {config_files}")

    # Load the quantized model once for the entire run (all config files,
    # all axes). layer_ids=[] means no layer is wrapped yet -- the first
    # call to set_steering_layer() inside run_evaluations_for_config wraps
    # whichever layer the first axis needs.
    print("Loading base quantized model (once for the whole run)...")
    model = configure_model(model_name, model_path, layer_ids=[])

    for config_file in config_files:
        print(f"\n{'=' * 60}")
        print(f"Config file: {config_file}")
        print('=' * 60)
        run_evaluations_for_config(config_file, model)

    print("\nAll evaluations complete!")


if __name__ == "__main__":
    main()