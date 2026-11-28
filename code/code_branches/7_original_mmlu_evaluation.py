import pandas as pd
from datasets import load_dataset
from utils import get_output, get_selfdebias_output

# Tokenizer will be passed as parameter to avoid circular import


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
    
    # Choose generation method based on self-debiasing flag
    if use_self_debias:
        generated_answer = get_selfdebias_output(input_text, model, tokenizer, vector, coeff).lower()
    else:
        generated_answer = get_output(input_text, model, tokenizer, vector, coeff).lower()
    
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
    
    # Load MMLU dataset
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    full_df = pd.DataFrame(mmlu)
    
    # Get equal sample from all subjects (up to ~1000 questions)
    mmlu_df = full_df.groupby('subject').sample(
        n=1000 // full_df['subject'].nunique(), random_state=42
    ).reset_index(drop=True)
    
    print(f"Evaluating {len(mmlu_df)} questions...")
    
    # Apply predictions
    mmlu_df[['ans', 'prediction', 'correct']] = mmlu_df.apply(
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
