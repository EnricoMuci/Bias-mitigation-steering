import os
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as pyplot

bbq_axes = ["age", "appearance", "disability", "gender", "nationality", "race", "religion", "socioeconomic"]
FIG_FORMAT = 'svg' # ['svg', 'png']
MOSAIC_FLAG = True


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
            pyplot.figure(figsize=(8, 5))
            pyplot.plot(df['coeff'], df['bbq_accuracy'], marker='o', linestyle='-', color='b')

            # Add labels and title
            pyplot.xlabel("coeff")
            pyplot.ylabel("bbq_accuracy")
            pyplot.title(f"BBQ Accuracy vs Steering Coefficient ({os.path.basename(file_path)})")
            pyplot.grid(True)

            # Save the plot as a PNG file in the same directory
            output_filename = f"{output_path}/{file_name}_plot.{FIG_FORMAT}"  # os.path.splitext(file_path)[0]
            pyplot.savefig(output_filename, bbox_inches='tight')
            pyplot.close() # close the current processing of the graph

            print(f"Successfully generated plot for {os.path.basename(file_path)} -> {output_filename}")

        except Exception as err:
            print(f"Error processing {file_path}: {err}")


def generate_mosaic_plots(base_input_dir: str, base_output_dir: str) -> None:

    # 2x4 grid layout for the 8 bias axes
    mosaic_layout = [
        ["age", "appearance", "disability", "gender"], # top-line
        ["nationality", "race", "religion", "socioeconomic"]  # bottom-line
    ]

    # Ensure the output directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    vt_dir = Path(base_input_dir)
    vt = vt_dir.stem

    if not os.path.exists(vt_dir):
        print(f"[WARNING] Input subdirectory not found: {vt_dir}. Skipping...")
        return # continue

    # Collect all CSV files inside the vector_type input directory
    csv_files = glob.glob(os.path.join(vt_dir, "*.csv"))
    if not csv_files:
        print(f"[WARNING] No CSV files found in: {vt_dir}")
        return # continue

    # Map axes to their corresponding file path by checking the filename stem
    file_mapping = {}
    for file_path in csv_files:
        stem = Path(file_path).stem.lower() # stem → CSV_name = axis_vt.csv
        for row in mosaic_layout:
            for axis_name in row:
                if axis_name in stem:
                    file_mapping[axis_name] = file_path

    # Initialize mosaic figure with shared axes for clean comparisons
    fig, ax_dict = pyplot.subplot_mosaic(
        mosaic_layout, # defined lines
        figsize=(18, 9), # single figure
        sharex=False,
        sharey=True
    )

    global_min = float('inf')
    global_max = float('-inf')
    y_tolerance = 0.02  # Absolute margin to leave top and bottom

    for axis_name, ax in ax_dict.items(): # dict_item = (axis_name, ax)
        if axis_name in file_mapping:
            csv_path = file_mapping[axis_name]
            try:
                df = pd.read_csv(csv_path)
                if 'coeff' in df.columns and 'bbq_accuracy' in df.columns:
                    # Sort by coeff to prevent erratic zig-zag line plots
                    df = df.sort_values(by='coeff')
                    ax.plot(df['coeff'], df['bbq_accuracy'], marker='o', linestyle='-', color='#1f77b4')

                    # Extract local min and max
                    min_acc = df['bbq_accuracy'].min()
                    max_acc = df['bbq_accuracy'].max()

                    # Update global mosaic boundaries
                    global_min = min(global_min, min_acc)
                    global_max = max(global_max, max_acc)

                    # Find and highlight the optimal steering point
                    best_coeff = df.loc[df['bbq_accuracy'].idxmax(), 'coeff']
                    ax.axvline(x=best_coeff, color='red', linestyle='--', alpha=0.5)
                    ax.axhline(y=max_acc, color='red', linestyle='--', alpha=0.5)
                    ax.scatter(best_coeff, max_acc, color='red', zorder=5)

                else:
                    ax.text(0.5, 0.5, "Columns missing", ha='center', va='center', transform=ax.transAxes)
            except Exception as err:
                print(f"Error loading {csv_path}: {err}")
                ax.text(0.5, 0.5, "Read Error", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "File not found", ha='center', va='center', transform=ax.transAxes, color='gray')

        ax.set_title(axis_name.capitalize(), fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.6)

    # Global labels and title mapping
    fig.supxlabel("Steering Coefficient (coeff)", fontsize=13)
    fig.supylabel("BBQ Accuracy", fontsize=13)
    fig.suptitle(f"BBQ Accuracy vs Steering Coefficient - {vt}", fontsize=15, fontweight='bold')

    pyplot.tight_layout()

    # Save one high-resolution mosaic plot per vector_type
    output_filename = os.path.join(base_output_dir, f"mosaic_accuracy_{vt.replace('+', '_')}.{FIG_FORMAT}")
    pyplot.savefig(output_filename, dpi=300, bbox_inches='tight')
    pyplot.close()

    print(f"Mosaic plot saved successfully: {output_filename}")


if __name__ == "__main__":
    vector_types = ('train', 'train+prompt')
    experiment = 'reproduction'
    for vector_type in vector_types:
        input_dir = f"../data/coeff_scores/mistral-full/{experiment}/{vector_type}"
        output_dir = f"../figs/5c-coeff-figs/mistral-full/{experiment}"

        if MOSAIC_FLAG:
            generate_mosaic_plots(input_dir, output_dir)
        else:
            plot_all_csv(input_dir, output_dir)
