"""
Script to convert node_values from .npy format to CSV.
"""

import numpy as np
import pandas as pd

# Load the node values
node_values_path = 'CFGenerationForSGNN/preprocessing/data/METRLA/node_values.npy'
node_values = np.load(node_values_path)

print(f"Loaded node values from METRLA")
print(f"Shape: {node_values.shape}")
print(f"Data type: {node_values.dtype}")
print(f"Min value: {node_values.min()}")
print(f"Max value: {node_values.max()}")
print(f"Mean value: {node_values.mean():.2f}")

# Determine the structure
if len(node_values.shape) == 2:
    print(f"\n2D array detected:")
    print(f"  - Dimension 0 (rows): {node_values.shape[0]}")
    print(f"  - Dimension 1 (columns): {node_values.shape[1]}")
    
    # Assume rows are time steps and columns are nodes
    n_timesteps = node_values.shape[0]
    n_nodes = node_values.shape[1]
    
    # Create column names for each node
    column_names = [f"Node_{i}" for i in range(n_nodes)]
    
    # Create DataFrame
    df = pd.DataFrame(node_values, columns=column_names)
    df.insert(0, 'Timestep', range(n_timesteps))
    
elif len(node_values.shape) == 3:
    print(f"\n3D array detected:")
    print(f"  - Dimension 0: {node_values.shape[0]}")
    print(f"  - Dimension 1: {node_values.shape[1]}")
    print(f"  - Dimension 2: {node_values.shape[2]}")
    
    # For 3D arrays, we'll flatten or create a more structured format
    # Common format: (timesteps, nodes, features)
    n_timesteps = node_values.shape[0]
    n_nodes = node_values.shape[1]
    n_features = node_values.shape[2]
    
    print(f"\nAssuming format: (timesteps={n_timesteps}, nodes={n_nodes}, features={n_features})")
    
    # Option 1: Save as a long format
    data_list = []
    for t in range(n_timesteps):
        for n in range(n_nodes):
            row = {'Timestep': t, 'Node': n}
            for f in range(n_features):
                row[f'Feature_{f}'] = node_values[t, n, f]
            data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
else:
    print(f"\nFlattening {len(node_values.shape)}D array to 1D")
    df = pd.DataFrame({'Value': node_values.flatten()})

# Save to CSV
output_csv = 'CFGenerationForSGNN/preprocessing/data/METRLA/node_values.csv'
print(f"\nSaving to CSV file: {output_csv}")

df.to_csv(output_csv, index=False)

print(f"✓ CSV file created successfully!")
print(f"  CSV shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Save a summary
summary_csv = 'CFGenerationForSGNN/preprocessing/data/METRLA/node_values_summary.csv'
summary_data = {
    'Metric': ['Original Shape', 'Data Type', 'Min Value', 'Max Value', 'Mean Value', 
               'CSV Rows', 'CSV Columns'],
    'Value': [str(node_values.shape), str(node_values.dtype), 
              node_values.min(), node_values.max(), f"{node_values.mean():.2f}",
              df.shape[0], df.shape[1]]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_csv, index=False)
print(f"✓ Summary CSV saved to: {summary_csv}")

# Show a preview
print(f"\nPreview of the first few rows:")
print(df.head(10))

