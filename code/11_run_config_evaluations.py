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

from utils_new import (get_args, get_model_short_name, define_custom_tokenizer, configure_model, set_steering_layer,
                       BASE_EXPERIMENT, CROWS_PATH, set_seed)

# Import functions from the individual evaluation files
# Note: Using importlib because Python module names can't start with numbers
from importlib import import_module
import glob

# import inspect
# from retrieval_injection import build_injector

# Registry describing each pluggable evaluation: which module/function to
# call, which axes it's relevant for (None = all axes), and the column
# prefix used when writing results. Modules are imported lazily (only the
# ones actually selected via --evals) so unused evaluations don't pay their
# import cost / dependency requirements.
DATASETS_REGISTRY = {
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
        'relevant_axes': ['gender', 'race', 'religion', 'age', 'nationality',
                          'socioeconomic', 'appearance', 'disability'],
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

loaded_modules = {}


def get_eval_module(eval_key):
    """Lazily import an evaluation module the first time it's needed."""
    if eval_key not in loaded_modules:
        loaded_modules[eval_key] = import_module(DATASETS_REGISTRY[eval_key]['module_name'])
    return loaded_modules[eval_key]


def load_retriever(spec):
    """Load a retriever from 'module_name:factory_name' (factory takes no arguments)."""
    if spec is None:
        raise ValueError("--retriever-spec is required with --injection-mode retriever")
    module_name, factory_name = spec.split(':')
    return getattr(import_module(module_name), factory_name)()


# Global configuration
USE_FAIRNESS_PROMPT = False  # Set to True to enable fairness prompting
USE_SELF_DEBIAS = True  # Set to True to enable self-debiasing

# ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument('-n', '--name', type=str, default='mistralai/Mistral-7B-Instruct-v0.1', help='model name')
parser.add_argument('-p', '--path', type=str, default=None, help='model path')
parser.add_argument('-q', '--quantization', action='store_true', help='Insert flag to quantize the model')
parser.add_argument('-e', '--experiment', type=str, default=None, help='Use it to change the experiment')

parser.add_argument('-o', '--only-preview', action='store_true', help='Show only the preview without computing')

parser.add_argument('-a', '--axes', nargs='*', type=str, default=None, help='axes to be processed')
parser.add_argument('-t', '--vector-types', nargs='*', type=str, default=None, choices=['train', 'train+prompt'],
                    help='train[+prompt]')

parser.add_argument('-k', '--k-sentences', type=int, default=0, help='Number of retrieved sentences')
parser.add_argument('-b', '--bias-ratio', type=float, default=0.5, help='Pro-stereotype sentences ratio (0.0 - 1.0)')

parser.add_argument('-d', '--datasets', nargs='*', choices=list(DATASETS_REGISTRY.keys()),
                    default=['bbq', 'mmlu'],  #list(EVAL_REGISTRY.keys()),
                    help='Which datasets to run (default: BBQ and MMLU). E.g. --evals bbq mmlu')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='Path to a single config CSV to evaluate. Default: every *.csv in ../data/configs/[...]')

# parser.add_argument('-m', '--injection-mode', type=str, default='random', choices=['random', 'retriever'],
#                     help='Injection strategy used when k > 0')
# parser.add_argument('-r', '--retriever-spec', type=str, default=None,
#                     help="Retriever factory as 'module_name:factory_name'")

args = parser.parse_args()

SEED = 42
set_seed(SEED)

(model_name, model_path) = get_args([args.name, args.path])
model_short_name = get_model_short_name(model_name, quantized=args.quantization)
tokenizer = define_custom_tokenizer(model_name, model_path)

if args.experiment is not None:
    EXPERIMENT = args.experiment
else:
    EXPERIMENT = BASE_EXPERIMENT

CONFIG_PATH = f"../data/configs/{model_short_name}/{EXPERIMENT}"
RESULTS_PATH = f"../results/{model_short_name}/{EXPERIMENT}"

# Injection parameters
k = args.k_sentences  # top-k sentences
b = args.bias_ratio  # fraction of pro-stereotyped sentences

if EXPERIMENT == 'reproduction':  # only reproduction
    CONFIG_PATH = f"../data/configs/{model_short_name}/{EXPERIMENT}"
    RESULTS_PATH = f"../results/{model_short_name}/{EXPERIMENT}"
else:  # add the k_b directory
    CONFIG_PATH = f"../data/configs/{model_short_name}/{EXPERIMENT}/k-{k}_b-{b}"
    RESULTS_PATH = f"../results/{model_short_name}/{EXPERIMENT}/k-{k}_b-{b}"


def eval_already_done(existing_df, axis, vector_type, eval_key):
    """Check whether a specific evaluation already has a saved result for
    this axis (used to resume after an interruption without recomputing
    evaluations that already succeeded)."""
    if existing_df is None or (axis, vector_type) not in existing_df.index:
        return False
    row = existing_df.loc[(axis, vector_type)]
    if isinstance(row, pd.DataFrame):  # righe duplicate: prendi l'ultima
        row = row.iloc[-1]
    prefix = DATASETS_REGISTRY[eval_key]['prefix']
    if pd.notna(row.get(f"{prefix}_skipped")):  # asse non rilevante: conta come fatta
        return True
    result_cols = [c for c in existing_df.columns
                   if c.startswith(f"{prefix}_") and not c.endswith(('_error', '_skipped'))]
    return any(pd.notna(row.get(c)) for c in result_cols)


def preview_status(config_df, existing_df, requested_datasets):
    """Print a preview of the current evaluation status before running."""
    print("\n" + "=" * 55)
    print("PRE-RUN STATUS CHECK")
    print("=" * 55)

    all_done = True
    total_evals = len(requested_datasets)

    for _, row in config_df.iterrows():
        axis = row['axis']
        vector_type = row['vector_type']

        done_count = 0
        for eval_key in requested_datasets:
            # Uses the updated _eval_already_done with composite keys (axis, vector_type)
            if eval_already_done(existing_df, axis, vector_type, eval_key):
                done_count += 1

        if done_count == total_evals:
            print(f"  ✓ {axis:15s} ({vector_type})  ->  complete ({done_count}/{total_evals})")
        elif done_count > 0:
            print(f"  ○ {axis:15s} ({vector_type})  ->  partial ({done_count}/{total_evals})")
            all_done = False
        else:
            print(f"  ✗ {axis:15s} ({vector_type})  ->  not initialized (0/{total_evals})")
            all_done = False

    print("=" * 55 + "\n")
    return all_done


def run_evaluations_for_config(config_file, model):  # , injector=None ## as 3rd argument
    """Run all evaluations for a given config file by calling functions from individual files.

    `model` is a single QuantizedSteeringModel instance, loaded once by the
    caller and reused across every axis/config file: only the steered layer
    is swapped between axes (see set_steering_layer), not the whole model.
    """
    config_name = os.path.basename(config_file).replace('.csv', '')
    print(f"\nRunning evaluations for config: {config_name}")  # [baseline, train, train+prompt]

    # Load config
    config_df = pd.read_csv(config_file)
    print(f"Loaded {len(config_df)} configurations")

    os.makedirs(RESULTS_PATH, exist_ok=True)
    results_file = f"{RESULTS_PATH}/{config_name}.csv"

    existing_df = None
    if os.path.exists(results_file):
        existing_df = pd.read_csv(results_file).set_index(['axis', 'vector_type'], drop=False)
        print(f"Found existing results file ({len(existing_df)} rows) — resuming, "
              f"already-completed evaluations will be skipped.")

    requested_datasets = [ds for ds in DATASETS_REGISTRY if ds in args.datasets]

    if args.axes is not None:
        config_df = config_df[config_df['axis'].isin(args.axes)]
    if args.vector_types is not None:
        config_df = config_df[config_df['vector_type'].isin(args.vector_types)]

    # Show preview
    all_done = preview_status(config_df, existing_df, requested_datasets)

    # Optional: exit early if everything is completed or if in preview-only mode
    if args.only_preview or all_done:
        return

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
        ax_vt = (row['axis'], row['vector_type'])
        if existing_df is None:
            existing_df = pd.DataFrame([row]).set_index(['axis', 'vector_type'], drop=False)
        else:
            if ax_vt in existing_df.index:
                merged = existing_df.loc[ax_vt].to_dict()
                merged.update(row)  # aggiorna solo i campi nuovi
                existing_df = existing_df.drop(index=ax_vt)
            else:
                merged = dict(row)
            new_row_df = pd.DataFrame([merged]).set_index(['axis', 'vector_type'], drop=False)
            existing_df = pd.concat([existing_df, new_row_df])
        existing_df.reset_index(drop=True).to_csv(results_file, index=False)

    for _, config_row in tqdm(config_df.iterrows(), total=len(config_df), desc="Total Configs Progress", position=0):
        axis = config_row['axis']
        vector_type = config_row['vector_type']  # 'train' 'train+prompt'

        layer = int(config_row['layer'])
        coeff = config_row['coeff']
        bbq_accuracy = config_row['bbq_accuracy']  # read from config_file
        mmlu_accuracy = config_row['mmlu_accuracy']  # read from config_file

        pending_evals = [e for e in requested_datasets if not eval_already_done(existing_df, axis, vector_type, e)]
        if not pending_evals:
            print(f"\n  Skipping {axis}: all requested evaluations already completed (resumed).")
            continue

        print(f"\n  Processing {axis} - {vector_type} (layer={layer}, coeff={coeff})... pending: {pending_evals}")

        # Check if vector file exists before proceeding
        vector_path = f'../vectors/{model_short_name}/{vector_type}/{axis}.gguf'
        if not os.path.exists(vector_path):
            # print(f"    Skipping {axis}: Vector file not found at {vector_path}")
            # continue
            raise FileNotFoundError(f"Vector file not found: {vector_path}")

        # Reuse the already-loaded model: just swap which layer is wrapped
        # in SteeringModule, instead of reloading the whole 7B model.
        set_steering_layer(model, layer)
        vector = SteeringVector.import_gguf(vector_path)

        # Initialize result row with config data
        result_row = {
            'k-num-sentences': k,
            'bias-ratio': b,
            'axis': axis,
            'vector_type': vector_type,
            'layer': layer,
            'coeff': coeff,
            'cfg_bbq_accuracy': bbq_accuracy,  # write onto results file
            'cfg_mmlu_accuracy': mmlu_accuracy,  # write onto results file
            'fairness_prompt': USE_FAIRNESS_PROMPT,
            'self_debias': USE_SELF_DEBIAS,
            # **(injector.describe() if injector is not None
            #    else {'injection_mode': 'none', 'k_sentences': 0}),
        }


        # Run every pending evaluation (already-completed ones were filtered out above)
        for eval_key in pending_evals:
            eval_info = DATASETS_REGISTRY[eval_key]
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
    print(f"Evaluations run: {requested_datasets}")
    print(f"Results saved to {results_file}")


def setup_logging():
    """Set up logging to redirect all output to a log file."""
    # Create timestamp for unique log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{RESULTS_PATH}/eval-logs", exist_ok=True)
    log_file = f"{RESULTS_PATH}/eval-logs/evaluation_run_{timestamp}.log"

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
    print(f"Evaluations selected: {args.datasets}")

    if args.config:
        config_files = [args.config]
        if not os.path.exists(config_files[0]):
            raise FileNotFoundError(f"Config file not found: {config_files[0]}")
    else:
        config_files = sorted(glob.glob(f"{CONFIG_PATH}/*.csv"))
        if not config_files:
            raise FileNotFoundError(f"No config files found in {CONFIG_PATH}")

    print(f"Config files to process: {config_files}")

    # Load the quantized model once for the entire run (all config files,
    # all axes). layer_ids=[] means no layer is wrapped yet -- the first
    # call to set_steering_layer() inside run_evaluations_for_config wraps
    # whichever layer the first axis needs.
    print("Loading base quantized model (once for the whole run)...")
    model = configure_model(model_name, model_path, layer_ids=[], quantized=args.quantization)

    # # Injector configuration
    # retriever = load_retriever(args.retriever_spec) if INJECTION_MODE == 'retriever' else None
    # injector = build_injector(INJECTION_MODE, k, b, CROWS_PATH, retriever)
    # if injector is not None:
    #     injector.cache_dir = f"../cache/{EXPERIMENT}"
    # print(f"Injection: {injector.describe() if injector is not None else 'none'}")
    #
    # if injector is not None and 'bbq' in args.datasets:
    #     bbq_func = getattr(get_eval_module('bbq'), DATASETS_REGISTRY['bbq']['func_attr'])
    #     if 'injector' not in inspect.signature(bbq_func).parameters:
    #         raise RuntimeError("run_bbq_evaluation must accept an 'injector' argument "
    #                            "(see 6_bbq_evaluation.py).")

    for config_file in config_files:
        print(f"\n{'=' * 60}")
        print(f"Config file: {config_file}")
        print('=' * 60)
        run_evaluations_for_config(config_file, model)  # config_file, model, injector

    print("\nAll evaluations complete!")


if __name__ == "__main__":
    main()
