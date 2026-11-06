#!/usr/bin/env python3
"""
Generate LaTeX table for BBQ test accuracies across different methods.

Extracts BBQ test accuracies from results files and formats them as a LaTeX table.
"""

import pandas as pd
import os

def load_results():
    """Load results from different methods."""
    results_dir = "results/mistral-full"
    
    # Define file mappings
    files = {
        'baseline': f'{results_dir}/baselines.csv',
        'prompting': f'{results_dir}/prompting.csv',
        'selfdebias': f'{results_dir}/selfdebias.csv',
        'top_train': f'{results_dir}/top_train.csv', 
        'top_train_prompt': f'{results_dir}/top_train+prompt.csv',
        'finetuned': f'{results_dir}/finetuned_20250727_174210.csv',
        'sve': f'{results_dir}/sve_20250728_114722.csv'
    }
    
    data = {}
    
    for method, filepath in files.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            # Store the full dataframe for later metric extraction
            data[method] = df
            print(f"Loaded {method}: {len(df)} rows")
        else:
            print(f"Warning: File not found: {filepath}")
    
    return data

def get_metric_value(data, method, axis, metric_col):
    """Get a specific metric value for a method and axis."""
    if method not in data:
        return None
    
    df = data[method]
    axis_row = df[df['axis'] == axis]
    
    if axis_row.empty:
        return None
    
    if metric_col not in axis_row.columns:
        return None
    
    value = axis_row.iloc[0][metric_col]
    
    # Handle missing values, NaN, empty strings, or 'True'/'False' flags
    if pd.isna(value) or value == '' or value == 'True' or value == 'False':
        return None
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def generate_latex_table(data):
    """Generate LaTeX table from the results data."""
    
    # Define evaluation categories and their rows
    evaluation_sections = [
        ('BBQ', [
            ('bbq_age', 'Age', 'bbq_test_accuracy', ['age']),
            ('bbq_appearance', 'Appearance', 'bbq_test_accuracy', ['appearance']),
            ('bbq_disability', 'Disability', 'bbq_test_accuracy', ['disability']),
            ('bbq_gender', 'Gender', 'bbq_test_accuracy', ['gender']),
            ('bbq_nationality', 'Nationality', 'bbq_test_accuracy', ['nationality']),
            ('bbq_race', 'Race', 'bbq_test_accuracy', ['race']),
            ('bbq_religion', 'Religion', 'bbq_test_accuracy', ['religion']),
            ('bbq_socioeconomic', 'Socioeconomic', 'bbq_test_accuracy', ['socioeconomic']),
        ]),
        ('StereoSet (ICAT)', [
            ('stereoset_gender', 'Gender', 'stereoset_icat_score', ['gender']),
            ('stereoset_race', 'Race', 'stereoset_icat_score', ['race']),
            ('stereoset_religion', 'Religion', 'stereoset_icat_score', ['religion']),
        ]),
        ('Clear Bias', [
            ('clearbias_age', 'Age', 'clear_bias_clearbias_score', ['age']),
            ('clearbias_disability', 'Disability', 'clear_bias_clearbias_score', ['disability']),
            ('clearbias_gender', 'Gender', 'clear_bias_clearbias_score', ['gender']),
            ('clearbias_race', 'Race', 'clear_bias_clearbias_score', ['race']),
            ('clearbias_religion', 'Religion', 'clear_bias_clearbias_score', ['religion']),
            ('clearbias_socioeconomic', 'Socioeconomic', 'clear_bias_clearbias_score', ['socioeconomic']),
        ]),
        ('MMLU', [
            ('mmlu_average', 'Average', 'mmlu_test_accuracy', ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic']),
        ])
    ]
    
    print("\\newpage")
    print("\\begin{table}[!t]")
    print("\\centering")
    print("\\begin{tabular}{l|ccccc}")
    print("\\textbf{Evaluation} & \\textbf{Baseline} & \\textbf{Prompting} & \\textbf{SelfDebias} & \\textbf{Finetuned} & \\textbf{Train+Prompt} \\\\")
    print("\\hline")
    
    first_section = True
    for section_name, section_rows in evaluation_sections:
        # Add a section header
        if not first_section:
            print("\\hline")
        print(f"\\multicolumn{{6}}{{c}}{{\\textbf{{{section_name}}}}} \\\\")
        print("\\hline")
        first_section = False
        
        for row_id, display_name, metric_col, relevant_axes in section_rows:
            values = []
            raw_values = []
            
            # Get values for each method
            for method in ['baseline', 'prompting', 'selfdebias', 'finetuned', 'top_train_prompt']:
                # Special handling for MMLU average
                if row_id == 'mmlu_average':
                    # Calculate average across all axes
                    values_for_avg = []
                    for axis in relevant_axes:
                        val = get_metric_value(data, method, axis, metric_col)
                        if val is not None:
                            values_for_avg.append(val)
                    
                    if values_for_avg:
                        avg_val = sum(values_for_avg) / len(values_for_avg)
                        raw_values.append(avg_val)
                    else:
                        raw_values.append(None)
                else:
                    # Find the value for this metric from the relevant axis
                    val = None
                    for axis in relevant_axes:
                        val = get_metric_value(data, method, axis, metric_col)
                        if val is not None:
                            break
                    
                    raw_values.append(val)
            
            # Find the maximum value to bold it (but handle different metric types)
            valid_values = [v for v in raw_values if v is not None]
            
            # For CrowS-Pairs bias score, lower is better (so we want to bold the minimum)
            if 'crows_bias_score' in metric_col:
                best_value = min(valid_values) if valid_values else None
            else:
                # For other metrics, higher is better
                best_value = max(valid_values) if valid_values else None
            
            # Format values
            for val in raw_values:
                if val is None:
                    values.append("--")
                else:
                    # Convert to percentage for display
                    if metric_col in ['bbq_test_accuracy', 'clear_bias_clearbias_score', 'mmlu_test_accuracy']:
                        formatted = f"{val * 100:.1f}"
                    elif metric_col == 'crows_bias_score':
                        # CrowS bias score - convert to percentage and flip (1-score for interpretability)
                        formatted = f"{val * 100:.1f}"
                    else:
                        # StereoSet ICAT scores are already percentages
                        formatted = f"{val:.1f}"
                    
                    # Bold the best value
                    if val == best_value and len([v for v in raw_values if v == best_value]) == 1:
                        formatted = f"\\textbf{{{formatted}}}"
                    values.append(formatted)
            
            print(f"{display_name:15} & {' & '.join(values)} \\\\")
    
    print("\\end{tabular}")
    print("\\caption{Evaluation results for baseline, finetuned, and train+prompt methods across multiple bias benchmarks in Mistral. Values shown as percentages. Bold values indicate the best performance for each evaluation.}")
    print("\\label{tab:all_evaluations_mistral}")
    print("\\end{table}")

def calculate_steering_improvements(data):
    """Calculate average improvement of Train+Prompt (steering vectors) over Baseline for each evaluation set."""
    print("\n" + "="*80)
    print("STEERING VECTOR IMPROVEMENTS OVER BASELINE")
    print("="*80)
    
    evaluation_sets = [
        ("BBQ", 'bbq_test_accuracy', ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic']),
        ("StereoSet (ICAT)", 'stereoset_icat_score', ['gender', 'race', 'religion']),
        ("Clear Bias", 'clear_bias_clearbias_score', ['age', 'disability', 'gender', 'race', 'religion', 'socioeconomic']),
        ("MMLU", 'mmlu_test_accuracy', ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic'])
    ]
    
    for eval_name, metric_col, axes in evaluation_sets:
        baseline_values = []
        steering_values = []
        improvements = []
        
        for axis in axes:
            baseline_val = get_metric_value(data, 'baseline', axis, metric_col)
            steering_val = get_metric_value(data, 'top_train_prompt', axis, metric_col)
            
            if baseline_val is not None and steering_val is not None:
                baseline_values.append(baseline_val)
                steering_values.append(steering_val)
                
                # Calculate improvement (absolute percentage points)
                if metric_col == 'stereoset_icat_score':
                    # StereoSet values are already in percentage format
                    improvement = steering_val - baseline_val
                    avg_baseline_multiplier = 1
                    avg_steering_multiplier = 1
                else:
                    # Other metrics need to be converted to percentages
                    improvement = (steering_val - baseline_val) * 100
                    avg_baseline_multiplier = 100
                    avg_steering_multiplier = 100
                
                improvements.append(improvement)
        
        if improvements:
            if metric_col == 'stereoset_icat_score':
                avg_baseline = sum(baseline_values) / len(baseline_values)
                avg_steering = sum(steering_values) / len(steering_values)
            else:
                avg_baseline = sum(baseline_values) / len(baseline_values) * 100
                avg_steering = sum(steering_values) / len(steering_values) * 100
            avg_improvement = sum(improvements) / len(improvements)
            
            print(f"\n{eval_name}:")
            print(f"  Baseline average:    {avg_baseline:.1f}%")
            print(f"  Steering average:    {avg_steering:.1f}%")
            print(f"  Average improvement: {avg_improvement:+.1f} percentage points")
            print(f"  Relative improvement: {(avg_improvement/avg_baseline)*100:+.1f}%")
            print(f"  Axes evaluated:      {len(improvements)}")
        else:
            print(f"\n{eval_name}: No comparable data available")

def calculate_bbq_gains_over_baseline(data):
    """Calculate average BBQ gain over baseline for each mitigation method."""
    print("\n" + "="*80)
    print("BBQ GAINS OVER BASELINE")
    print("="*80)
    
    methods = ['prompting', 'selfdebias', 'finetuned', 'top_train_prompt']
    method_names = ['Prompting', 'SelfDebias', 'Finetuned', 'Train+Prompt (Steering)']
    axes = ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic']
    
    # First, collect baseline values
    baseline_values = []
    for axis in axes:
        baseline_val = get_metric_value(data, 'baseline', axis, 'bbq_test_accuracy')
        if baseline_val is not None:
            baseline_values.append(baseline_val * 100)  # Convert to percentage
    
    if not baseline_values:
        print("No baseline BBQ data found!")
        return
    
    baseline_avg = sum(baseline_values) / len(baseline_values)
    print(f"Baseline average BBQ accuracy: {baseline_avg:.1f}%")
    print("")
    
    # Calculate gains for each method
    for method, name in zip(methods, method_names):
        if method in data:
            values = []
            gains = []
            
            for axis in axes:
                baseline_val = get_metric_value(data, 'baseline', axis, 'bbq_test_accuracy')
                method_val = get_metric_value(data, method, axis, 'bbq_test_accuracy')
                
                if baseline_val is not None and method_val is not None:
                    baseline_pct = baseline_val * 100
                    method_pct = method_val * 100
                    gain = method_pct - baseline_pct
                    
                    values.append(method_pct)
                    gains.append(gain)
            
            if values and gains:
                method_avg = sum(values) / len(values)
                avg_gain = sum(gains) / len(gains)
                
                print(f"{name}:")
                print(f"  Average BBQ accuracy: {method_avg:.1f}%")
                print(f"  Average gain over baseline: {avg_gain:+.1f} percentage points")
                print(f"  Relative improvement: {(avg_gain/baseline_avg)*100:+.1f}%")
                print(f"  Axes with data: {len(values)} out of {len(axes)}")
                print("")
            else:
                print(f"{name}: No comparable BBQ data found")
                print("")
        else:
            print(f"{name}: File not found")
            print("")

def print_summary_stats(data):
    """Print summary statistics for BBQ only."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (BBQ Test Accuracy)")
    print("="*60)
    
    methods = ['baseline', 'prompting', 'selfdebias', 'finetuned', 'top_train_prompt']
    method_names = ['Baseline', 'Prompting', 'SelfDebias', 'Finetuned', 'SV']
    axes = ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic']
    
    for method, name in zip(methods, method_names):
        if method in data:
            values = []
            for axis in axes:
                val = get_metric_value(data, method, axis, 'bbq_test_accuracy')
                if val is not None:
                    values.append(val * 100)  # Convert to percentage
            
            if values:
                avg = sum(values) / len(values)
                print(f"{name:20}: {avg:.1f}% average across {len(values)} axes")
            else:
                print(f"{name:20}: No BBQ data")
        else:
            print(f"{name:20}: File not found")

def main():
    """Main function."""
    print("Loading evaluation results...")
    data = load_results()
    
    print("\nGenerating LaTeX table...")
    print("="*60)
    generate_latex_table(data)
    
    print_summary_stats(data)
    
    calculate_bbq_gains_over_baseline(data)
    
    calculate_steering_improvements(data)
    
    print("\n" + "="*60)
    print("Script completed successfully!")
    print("Copy the LaTeX code above to use in your document.")
    print("="*60)

if __name__ == "__main__":
    main()