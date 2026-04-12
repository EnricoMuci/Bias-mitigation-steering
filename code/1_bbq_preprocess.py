import os
import json
import pandas as pd

# Define paths
dirs = {
    "raw": "../raw_data/bbq",
    "train": "../data/bbq_train",
    "validate": "../data/bbq_validate",
    "test": "../data/bbq_test",
}

# Create output directories if they don't exist
for d in dirs.values():
    if d != dirs["raw"]:
        os.makedirs(d, exist_ok=True)
        
# Load datasets
datasets = {
    "age": pd.read_json(os.path.join(dirs["raw"], "Age.jsonl"), lines=True),
    "appearance": pd.read_json(os.path.join(dirs["raw"], "Physical_appearance.jsonl"), lines=True),
    "disability": pd.read_json(os.path.join(dirs["raw"], "Disability_status.jsonl"), lines=True),
    "gender": pd.read_json(os.path.join(dirs["raw"], "Gender_identity.jsonl"), lines=True),
    "nationality": pd.read_json(os.path.join(dirs["raw"], "Nationality.jsonl"), lines=True),
    "race": pd.read_json(os.path.join(dirs["raw"], "Race_ethnicity.jsonl"), lines=True),
    "religion": pd.read_json(os.path.join(dirs["raw"], "Religion.jsonl"), lines=True),
    "socioeconomic": pd.read_json(os.path.join(dirs["raw"], "SES.jsonl"), lines=True),
}

def create_contrastive_entries(df):
    entries = []
    letters = ['a', 'b', 'c'] # answer IDs
    for _, row in df.iterrows():
        # Extract text
        context = row['context'].strip()
        question = row['question'].strip()
        choices = [row['ans0'], row['ans1'], row['ans2']]
        choices_str = f"(a) {choices[0]}  (b) {choices[1]}  (c) {choices[2]}"

        # Determine stereotype groups
        stereo_groups = set(row['additional_metadata']['stereotyped_groups']) # discriminated groups
        stereo_groups.update({g.replace(" ", "") for g in stereo_groups}) 
        ans_info = row['answer_info'] # another dictionary 

        # Find stereotype-matched answer
        stereo_idx = None
        for idx in range(3):
            text_vals = ans_info[f'ans{idx}']
            if any(val in stereo_groups for val in text_vals):
                stereo_idx = idx
                break
        if stereo_idx is None:
            print(f"Row {_}: No stereotype match. Skipping.")
            continue

        stereo_letter = letters[stereo_idx]
        correct_idx = row['label']

        # Determine anti-stereotype index:
        # Prefer gold if gold is not stereotype, else next non-stereotype
        if correct_idx != stereo_idx:
            anti_idx = correct_idx
        else:
            anti_idx = next(
                i for i in range(3)
                if i != stereo_idx
                and all(val not in stereo_groups for val in ans_info[f'ans{i}'])
            )
        anti_letter = letters[anti_idx]

        # Build entries: swapped so anti-stereotype is positive, stereotype is negative
        positive = f"{context} {question} Choices: {choices_str} Answer: ({anti_letter}" # anti-stereotypical
        negative = f"{context} {question} Choices: {choices_str} Answer: ({stereo_letter}" # stereotypical 

        entries.append({"positive": positive, "negative": negative})
    return entries

# Process each dataset
for name, df in datasets.items():
    print(f"Processing {name}...")

    # Select first 300 rows for train-phase contrastive
    df_train = df.head(300)

    # 1) Train: all contrastive entries from first 300
    train_entries = create_contrastive_entries(df_train)
    # JSON dump train entries (limit to first 300)
    with open(
        os.path.join(dirs['train'], f"{name}_train.json"),
        'w', encoding='utf-8'
    ) as out:
        json.dump(train_entries, out, indent=2, ensure_ascii=False)

    # 2) Validate: rows 301–600
    df_validate = df.iloc[300:600]
    df_validate.to_csv(
        os.path.join(dirs['validate'], f"{name}_validate.csv"),
        index=False
    )

    # 3) Test: rows 601–1200
    df_test = df.iloc[600:1200]
    df_test.to_csv(
        os.path.join(dirs['test'], f"{name}_test.csv"),
        index=False
    )

    print(
        f"Saved train ({len(train_entries)}), validate ({len(df_validate)}), test ({len(df_test)}) for {name}."
    )

print("All datasets processed.")
