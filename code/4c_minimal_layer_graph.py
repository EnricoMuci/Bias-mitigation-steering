#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def create_minimal_age_layer_graph():
    """
    Create a minimal version of the age layer graph with green vertical line at layer 13
    """
    # Load data
    sep_data = pd.read_csv('../data/original/separability_scores/mistral/age_train+prompt.csv')
    acc_data = pd.read_csv('../data/original/layer_scores/mistral/age_train+prompt.csv')

    # Create figure with clean minimal design
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # Use colorblind-friendly colors
    orange_color = '#E69F00'
    blue_color = '#0173B2'

    # Plot the lines without markers
    ax.plot(sep_data['layer'], sep_data['sep_score'],
            color=orange_color, linewidth=2.5, label='Separability')
    ax.plot(acc_data['layer'], acc_data['bbq_accuracy'],
            color=blue_color, linewidth=2.5, label='Validation Accuracy')

    # Add green vertical line at layer 14
    ax.axvline(x=14, color='green', linewidth=2, linestyle='-')

    # Set up grid with both horizontal and vertical lines
    ax.grid(True, alpha=0.3, color='#b0b0b0', linewidth=0.8)

    # Set axis properties
    ax.set_xlim(5, 32)

    # Hide y-axis labels and ticks completely but keep the grid lines by setting y-ticks manually
    y_min, y_max = ax.get_ylim()
    y_ticks = np.linspace(y_min, y_max, 6)
    ax.set_yticks(y_ticks)
    # ax.set_yticklabels([])
    # ax.tick_params(left=False)  # Remove y-axis tick marks

    # Large x-axis title
    ax.set_xlabel('Layer', fontsize=20)
    ax.set_ylabel('Accuracy', fontsize=20)

    # NEW: legend
    ax.legend(fontsize=14, loc='lower right')

    # Remove borders/spines except bottom for x-axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#b0b0b0')  # Light grey to match coefficient graph

    # Clean layout
    plt.tight_layout()

    # Save in the main figs directory
    figs_path = '../figs'
    minimal_path = os.path.join(figs_path, 'minimal_graph')
    os.makedirs(minimal_path, exist_ok=True)
    plt.savefig(f'{minimal_path}/age_minimal_layers.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{minimal_path}/age_minimal_layers.svg', dpi=300, bbox_inches='tight')
    plt.savefig(f'{minimal_path}/age_minimal_layers.png', dpi=300, bbox_inches='tight')


    print("Minimal age layer graph saved as age_minimal_layers.pdf, .svg, and .png")
    plt.show()


if __name__ == "__main__":
    create_minimal_age_layer_graph()
