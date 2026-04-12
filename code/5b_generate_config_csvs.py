import pandas as pd
import os
import glob

def generate_config_csvs():
    """Generate config CSV files for each folder with best results per axis."""
    
    # Define all axes
    axes = ['age', 'appearance', 'disability', 'gender', 'nationality', 'race', 'religion', 'socioeconomic']
    
    # Get all folders in coeff_scores/mistral
    folders = [d for d in os.listdir('../data/coeff_scores/mistral') if os.path.isdir(os.path.join('../data/coeff_scores/mistral', d))]
    folders.sort()
    
    # Create configs directory if it doesn't exist
    os.makedirs('../data/configs', exist_ok=True)
    
    for folder in folders:
        print(f"Processing folder: {folder}")
        
        config_data = []
        
        for axis in axes:
            # Load the corresponding best layers file
            best_layers_file = f"../data/layer_scores/mistral/best_layers/{folder}.csv"
            if os.path.exists(best_layers_file):
                best_layers_df = pd.read_csv(best_layers_file)
                # Find the row for this axis
                axis_row = best_layers_df[best_layers_df['axis'] == axis] # best layer for that stereotype
                if not axis_row.empty:
                    layer = axis_row.iloc[0]['max_layer']
                    vector_type = axis_row.iloc[0]['vt']
                else:
                    layer = None
                    vector_type = None
            else:
                layer = None
                vector_type = None
            
            # Find CSV files for this axis in this folder
            csv_pattern = f"../data/coeff_scores/mistral/{folder}/{axis}_*.csv"
            csv_files = glob.glob(csv_pattern)
            
            if csv_files and layer is not None:
                # Process the CSV file for this axis
                csv_file = csv_files[0]  # Should only be one file per axis per folder
                df = pd.read_csv(csv_file)
                if not df.empty:
                    # Find row with maximum BBQ accuracy
                    max_bbq_row = df.loc[df['bbq_accuracy'].idxmax()]
                    
                    config_data.append({
                        'axis': axis,
                        'vector_type': vector_type,
                        'layer': layer,
                        'coeff': max_bbq_row['coeff'],
                        'bbq_accuracy': max_bbq_row['bbq_accuracy'],
                        'mmlu_accuracy': max_bbq_row['mmlu_accuracy']
                    })
        
        # Save config CSV for this folder
        if config_data:
            config_df = pd.DataFrame(config_data)
            config_file = f"../data/configs/{folder}.csv"
            config_df.to_csv(config_file, index=False)
            print(f"  Saved {len(config_data)} configs to {config_file}")
        else:
            print(f"  No data found for folder {folder}")

if __name__ == "__main__":
    generate_config_csvs()
    print("\nConfig CSV generation complete!")