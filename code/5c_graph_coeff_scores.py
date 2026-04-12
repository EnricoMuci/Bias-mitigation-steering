#!/usr/bin/env python3
"""
Coefficient Scores Visualization Script

Creates graphs showing BBQ and MMLU accuracy vs coefficient values.
Generates 8 separate graphs (one for each bias axis) with BBQ and MMLU on the same plot.
Saves outputs to ../figs/coeffs/ as PDF and SVG.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def load_coeff_scores(axis, vector_type='train+prompt', model='mistral'):
    """Load coefficient scores for a specific axis."""
    file_path = f"../data/coeff_scores/{model}/top_{vector_type}/{axis}_{vector_type}.csv"
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def create_coeff_graph(axis, df, ax):
    """Create a coefficient vs accuracy graph for BBQ and MMLU."""
    if df is None or df.empty:
        ax.text(0.5, 0.5, f"No data available for {axis}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{axis.capitalize()}", fontsize=16, pad=10)
        return
    
    # Plot BBQ accuracy (colorblind-friendly orange)
    ax.plot(df['coeff'], df['bbq_accuracy'] * 100, '-', 
            color='#E69F00', label='BBQ Accuracy', linewidth=2.5)
    
    # Plot MMLU accuracy (colorblind-friendly blue)
    ax.plot(df['coeff'], df['mmlu_accuracy'] * 100, '-', 
            color='#0173B2', label='MMLU Accuracy', linewidth=2.5)
    
    # Formatting
    ax.set_xlabel('Coefficient', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12) 
    ax.set_title(f"{axis.capitalize()}", fontsize=16, pad=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set reasonable y-axis limits
    all_values = list(df['bbq_accuracy'] * 100) + list(df['mmlu_accuracy'] * 100)
    y_min = max(0, min(all_values) - 5)
    y_max = min(100, max(all_values) + 5)
    ax.set_ylim(y_min, y_max)


def create_single_axis_graph(axis): # single stereotype
    """Create a standalone graph for a single axis."""
    print(f"Creating standalone graph for {axis}...")
    
    # Load data
    df = load_coeff_scores(axis)
    
    if df is None or df.empty:
        print(f"No data available for {axis}")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create graph
    create_coeff_graph(axis, df, ax)
    
    plt.tight_layout()
    
    # Save as PDF and SVG
    pdf_path = f"../figs/coeffs/{axis}_coeff_scores.pdf"
    svg_path = f"../figs/coeffs/{axis}_coeff_scores.svg"
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Standalone graph saved:")
    print(f"  - {pdf_path}")
    print(f"  - {svg_path}")


def create_minimal_appearance_graph():
    """Create a minimal appearance graph with vertical line at coefficient 1.6."""
    print("Creating minimal appearance graph...")
    
    # Load data
    df = load_coeff_scores('appearance')
    
    if df is None or df.empty:
        print("No data available for appearance")
        return
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot BBQ accuracy (colorblind-friendly orange)
    ax.plot(df['coeff'], df['bbq_accuracy'] * 100, '-', 
            color='#E69F00', label='BBQ Accuracy', linewidth=2.5)
    
    # Plot MMLU accuracy (colorblind-friendly blue)
    ax.plot(df['coeff'], df['mmlu_accuracy'] * 100, '-', 
            color='#0173B2', label='MMLU Accuracy', linewidth=2.5)
    
    # Add vertical line at coefficient 1.6
    ax.axvline(x=1.6, color='green', linewidth=2, linestyle='-')
    
    # Remove title, y-axis label, borders
    ax.set_title('')
    ax.set_ylabel('')
    
    # Set x-axis label with much bigger font
    ax.set_xlabel('Coefficient', fontsize=20)
    
    # Keep graph background but remove borders (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Set reasonable y-axis limits first
    all_values = list(df['bbq_accuracy'] * 100) + list(df['mmlu_accuracy'] * 100)
    y_min = max(0, min(all_values) - 5)
    y_max = min(100, max(all_values) + 5)
    ax.set_ylim(y_min, y_max)
    
    # Set y-ticks for grid lines but hide labels
    y_ticks = np.linspace(y_min, y_max, 6)  # 6 horizontal lines
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([])  # Hide y-tick labels
    
    # Keep only x-axis ticks with normal font size
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', left=False, labelleft=False)
    
    # Remove legend but keep subtle grid
    ax.legend().set_visible(False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PDF, SVG, and PNG
    pdf_path = "../figs/coeffs/appearance_minimal_coeff_scores.pdf"
    svg_path = "../figs/coeffs/appearance_minimal_coeff_scores.svg"
    png_path = "../figs/coeffs/appearance_minimal_coeff_scores.png"
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print("Minimal appearance graph saved:")
    print(f"  - {pdf_path}")
    print(f"  - {svg_path}")
    print(f"  - {png_path}")


def create_averaged_coeff_graph():
    """Create a coefficient graph averaging BBQ and MMLU across all 8 axes."""
    print("Creating averaged coefficient graph across all axes...")
    
    # Define bias axes
    axes = ['age', 'appearance', 'disability', 'gender', 
            'nationality', 'race', 'religion', 'socioeconomic']
    
    # Load all data and calculate averages
    all_bbq_data = []
    all_mmlu_data = []
    coeffs = None
    
    for axis in axes:
        df = load_coeff_scores(axis)
        if df is not None and not df.empty:
            if coeffs is None:
                coeffs = df['coeff'].values
            all_bbq_data.append(df['bbq_accuracy'].values)
            all_mmlu_data.append(df['mmlu_accuracy'].values)
    
    if not all_bbq_data:
        print("No data available for averaging")
        return
    
    # Calculate averages across all axes
    avg_bbq = np.mean(all_bbq_data, axis=0) * 100  # Convert to percentage
    avg_mmlu = np.mean(all_mmlu_data, axis=0) * 100  # Convert to percentage
    
    # Create figure - same format as appearance standalone
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot averaged lines (colorblind-friendly colors)
    ax.plot(coeffs, avg_bbq, '-', 
            color='#E69F00', label='BBQ Accuracy (Avg)', linewidth=2.5)
    ax.plot(coeffs, avg_mmlu, '-', 
            color='#0173B2', label='MMLU Accuracy (Avg)', linewidth=2.5)
    
    # Formatting - bigger fonts for title and axis labels
    ax.set_xlabel('Coefficient', fontsize=16)
    ax.set_ylabel('Accuracy (%)', fontsize=16) 
    ax.set_title('BBQ vs. MMLU Accuracy across Coefficients', fontsize=20, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Set reasonable y-axis limits
    all_values = list(avg_bbq) + list(avg_mmlu)
    y_min = max(0, min(all_values) - 5)
    y_max = min(100, max(all_values) + 5)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save as PDF, SVG, and PNG
    pdf_path = "../figs/coeffs/averaged_coeff_scores.pdf"
    svg_path = "../figs/coeffs/averaged_coeff_scores.svg"
    png_path = "../figs/coeffs/averaged_coeff_scores.png"
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print("Averaged coefficient graph saved:")
    print(f"  - {pdf_path}")
    print(f"  - {svg_path}")
    print(f"  - {png_path}")


def generate_coeff_visualizations():
    """Generate coefficient score visualizations for all bias axes."""
    
    # Define bias axes
    axes = ['age', 'appearance', 'disability', 'gender', 
            'nationality', 'race', 'religion', 'socioeconomic']
    
    # Create output directory
    os.makedirs("../figs/coeffs", exist_ok=True)
    
    # Create 2x4 grid for 8 axes
    fig, axes_grid = plt.subplots(2, 4, figsize=(16, 8))
    axes_grid = axes_grid.flatten()
    
    print("Loading coefficient score data and creating graphs...")
    
    # Process each axis
    for idx, axis in enumerate(axes):
        print(f"Processing {axis}...")
        
        # Load data
        df = load_coeff_scores(axis)
        
        # Create graph
        ax = axes_grid[idx]
        create_coeff_graph(axis, df, ax)
    
    plt.tight_layout()
    
    # Save as PDF and SVG
    pdf_path = "../figs/coeffs/coeff_scores_all_axes.pdf"
    svg_path = "../figs/coeffs/coeff_scores_all_axes.svg"
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(svg_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print("Combined visualizations saved:")
    print(f"  - {pdf_path}")
    print(f"  - {svg_path}")
    
    # Also create standalone appearance graph
    create_single_axis_graph('appearance')
    
    # Also create minimal appearance graph
    create_minimal_appearance_graph()
    
    # Also create averaged coefficient graph
    create_averaged_coeff_graph()


if __name__ == "__main__":
    generate_coeff_visualizations()