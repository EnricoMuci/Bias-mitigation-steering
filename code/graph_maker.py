import os
import glob
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from utils import bbq_axes


def generate_mosaic_plots(base_input_dir: str, base_output_dir: str) -> None:
    vector_types = ["train", "train+prompt"]

    # 2x4 grid layout for the 8 bias axes
    mosaic_layout = bbq_axes

    # Ensure the output directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    for vt in vector_types:
        vt_dir = os.path.join(base_input_dir, vt)
        if not os.path.exists(vt_dir):
            print(f"[WARNING] Subdirectory not found: {vt_dir}. Skipping.")
            continue

        # Collect all CSV files inside the vector_type directory
        csv_files = glob.glob(os.path.join(vt_dir, "*.csv"))
        if not csv_files:
            print(f"[WARNING] No CSV files found in: {vt_dir}")
            continue

        # Map axes to their corresponding file path by checking the filename stem
        file_mapping = {}
        for file_path in csv_files:
            stem = Path(file_path).stem.lower()
            for row in mosaic_layout:
                for axis_name in row:
                    if axis_name in stem:
                        file_mapping[axis_name] = file_path

        # Initialize mosaic figure with shared axes for clean comparisons
        fig, ax_dict = plt.subplot_mosaic(
            mosaic_layout,
            figsize=(18, 9),
            sharex=True,
            sharey=True
        )

        for axis_name, ax in ax_dict.items():
            if axis_name in file_mapping:
                fpath = file_mapping[axis_name]
                try:
                    df = pd.read_csv(fpath)
                    if 'coeff' in df.columns and 'bbq_accuracy' in df.columns:
                        # Sort by coeff to prevent erratic zig-zag line plots
                        df = df.sort_values(by='coeff')
                        ax.plot(df['coeff'], df['bbq_accuracy'], marker='o', linestyle='-', color='#1f77b4')

                        # --- NUOVE RIGHE DA INSERIRE QUI ---
                        # 1. Set uniform vertical axis limits (assuming accuracy is 0.0 to 1.0)
                        ax.set_ylim(-0.05, 1.05)

                        # 2. Find and highlight the maximum accuracy point
                        max_acc = df['bbq_accuracy'].max()
                        best_coeff = df.loc[df['bbq_accuracy'].idxmax(), 'coeff']

                        # Draw dashed lines intersecting at the max point
                        ax.axvline(x=best_coeff, color='red', linestyle='--', alpha=0.5)
                        ax.axhline(y=max_acc, color='red', linestyle='--', alpha=0.5)

                        # Highlight the exact intersection with a red dot
                        ax.scatter(best_coeff, max_acc, color='red', zorder=5)
                        # --- FINE DELLE NUOVE RIGHE ---

                    else:
                        ax.text(0.5, 0.5, "Columns missing", ha='center', va='center', transform=ax.transAxes)
                except Exception as err:
                    print(f"Error loading {fpath}: {err}")
                    ax.text(0.5, 0.5, "Read Error", ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "File not found", ha='center', va='center', transform=ax.transAxes, color='gray')

            ax.set_title(axis_name.capitalize(), fontsize=12, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.6)

        # Global labels and title mapping
        fig.supxlabel("Steering Coefficient (coeff)", fontsize=13)
        fig.supylabel("BBQ Accuracy", fontsize=13)
        fig.suptitle(f"BBQ Accuracy vs Steering Coefficient - Vector Type: {vt}", fontsize=15, fontweight='bold')

        plt.tight_layout()

        # Save one high-resolution mosaic plot per vector_type
        output_filename = os.path.join(base_output_dir, f"mosaic_accuracy_{vt.replace('+', '_')}.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Mosaic plot saved successfully: {output_filename}")


if __name__ == "__main__":
    experiment = 'reproduction'
    base_input = f"../data/coeff_scores/mistral-full/{experiment}"
    base_output = f"../figs/5c-coeff-figs/mistral-full/{experiment}"

    generate_mosaic_plots(base_input, base_output)