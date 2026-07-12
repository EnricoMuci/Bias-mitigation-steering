import os
import pandas as pd
from tqdm import tqdm
from utils import get_output, get_selfdebias_output

# Tokenizer will be passed as parameter to avoid circular import

_MMLU_PARQUET_PATH = "../raw_data/mmlu/mmlu_all_test.parquet"  # must match prepare_MMLU() in 5_optimize_coeff.py
_mmlu_sample_cache = None  # module-level cache: loaded/sampled only once per process


def _load_mmlu_sample():
    """Load the MMLU test set from the local offline parquet file and take
    the same deterministic ~1000-question sample as before.
    """
    global _mmlu_sample_cache
    if _mmlu_sample_cache is not None:
        return _mmlu_sample_cache

    if not os.path.exists(_MMLU_PARQUET_PATH):
        raise FileNotFoundError(
            f"MMLU parquet file not found at {_MMLU_PARQUET_PATH}. "
        )

    full_df = pd.read_parquet(_MMLU_PARQUET_PATH)

    # Sampling logic/seed
    mmlu_df = full_df.groupby('subject').sample(
        n=1000 // full_df['subject'].nunique(), random_state=42
    ).reset_index(drop=True)

    print(f"MMLU sample loaded from {_MMLU_PARQUET_PATH} and cached ({len(mmlu_df)} questions).")
    _mmlu_sample_cache = mmlu_df
    return _mmlu_sample_cache


def predict_mmlu_row(row, model, vector, coeff, tokenizer, use_fairness_prompt=False, use_self_debias=False):
    """Predict on a single MMLU row."""
    question = row["question"]
    answers = row["choices"]
    answer_letters = ["a", "b", "c", "d"]

    # Add fairness instruction if enabled
    fairness_instruction = ""
    if use_fairness_prompt:
        fairness_instruction = "Ensure that your answers are fair and unbiased, and do not contain stereotypes.\n\n"

    input_text = (
        f"{fairness_instruction}"
        f"Answer with only a single letter.\n\n"
        f"The following are multiple choice questions (with answers):\n"
        f"{question}\n"
        f"(a) {answers[0]}\n"
        f"(b) {answers[1]}\n"
        f"(c) {answers[2]}\n"
        f"(d) {answers[3]}\n"
    )

    try:
        # Choose generation method based on self-debiasing flag
        if use_self_debias:
            generated_answer = get_selfdebias_output(input_text, model, tokenizer, vector, coeff).lower()
        else:
            generated_answer = get_output(input_text, model, tokenizer, vector, coeff).lower()
    except Exception as e:
        # Defensive: one bad generation shouldn't kill an unattended SLURM job.
        tqdm.write(f"    [warn] generation failed on a row, marking as incorrect: {e}")
        return pd.Series({"ans": "", "prediction": -1, "correct": False})

    predicted_label = -1
    for i, letter in enumerate(answer_letters):
        if letter in generated_answer[:2]:
            predicted_label = i
            break

    correct = (predicted_label == row["answer"])

    return pd.Series({
        "ans": generated_answer,
        "prediction": predicted_label,
        "correct": correct
    })


def run_mmlu_evaluation(model, vector, coeff, axis, tokenizer, use_fairness_prompt=False, use_self_debias=False):
    """Run evaluation on MMLU dataset for a specific configuration.

    Args:
        model: SteeringModel instance
        vector: SteeringVector instance
        coeff: Coefficient to apply
        axis: Bias axis being evaluated

    Returns:
        dict: Results dictionary with accuracy metrics
    """
    print(f"Running MMLU evaluation for axis: {axis}, coefficient: {coeff}...")

    mmlu_df = _load_mmlu_sample().copy()

    print(f"Evaluating {len(mmlu_df)} questions...")

    tqdm.pandas(desc=f"MMLU {axis}", position=1, leave=False)

    # Apply predictions
    mmlu_df[['ans', 'prediction', 'correct']] = mmlu_df.progress_apply(
        predict_mmlu_row, axis=1, args=(model, vector, coeff, tokenizer, use_fairness_prompt, use_self_debias)
    )

    # Calculate accuracy
    accuracy = mmlu_df['correct'].mean()
    test_accuracy = round(accuracy, 3)

    print(f"MMLU accuracy: {accuracy:.4f}")

    return {
        'test_accuracy': test_accuracy,
        'total_questions': len(mmlu_df),
        'correct_answers': mmlu_df['correct'].sum()
    }