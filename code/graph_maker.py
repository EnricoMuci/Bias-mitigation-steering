import os
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_all_csv(input_path: str = ".", output_path: str = ".") -> None:
    # Find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(input_path, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in the directory: {os.path.abspath(input_path)}")
        return

    for file_path in csv_files:
        file_name = Path(file_path).stem
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Ensure the required columns exist
            if 'coeff' not in df.columns or 'bbq_accuracy' not in df.columns:
                print(f"Skipping {file_path}: Missing 'coeff' or 'bbq_accuracy' column.")
                continue

            # Sort the dataframe by 'coeff' to avoid messy zig-zag lines in the plot
            df = df.sort_values(by='coeff')

            # Create the plot
            plt.figure(figsize=(8, 5))
            plt.plot(df['coeff'], df['bbq_accuracy'], marker='o', linestyle='-', color='b')

            # Add labels and title
            plt.xlabel("coeff")
            plt.ylabel("bbq_accuracy")
            plt.title(f"BBQ Accuracy vs Steering Coefficient ({os.path.basename(file_path)})")
            plt.grid(True)

            # Save the plot as a PNG file in the same directory
            output_filename = f"{output_path}/{file_name}_plot.png"  # os.path.splitext(file_path)[0]
            plt.savefig(output_filename, bbox_inches='tight')
            plt.close()

            print(f"Successfully generated plot for {os.path.basename(file_path)} -> {output_filename}")

        except Exception as err:
            print(f"Error processing {file_path}:\n{err}")


if __name__ == "__main__":
    vector_types = ('train', 'train+prompt')
    experiment = 'reproduction'
    for vt in vector_types:
        input_dir = f"../data/coeff_scores/mistral-full/{experiment}/{vt}"
        output_dir = f"../figs/5c-coeff-figs/mistral-full/{experiment}"
        plot_all_csv(input_dir, output_dir)
