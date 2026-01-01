"""
Script to convert adjacency matrix from .npy format to CSV for METRLA_30.
"""

import numpy as np
import pandas as pd

# Load the adjacency matrix
adj_mat_path = 'CFGenerationForSGNN/preprocessing/data/METRLA_30/adj_mat.npy'
adj_matrix = np.load(adj_mat_path)

print(f"Loaded adjacency matrix from METRLA_30")
print(f"Shape: {adj_matrix.shape}")
print(f"Data type: {adj_matrix.dtype}")
print(f"Min value: {adj_matrix.min()}")
print(f"Max value: {adj_matrix.max()}")
print(f"Number of non-zero entries: {np.count_nonzero(adj_matrix)}")
print(f"Sparsity: {(1 - np.count_nonzero(adj_matrix) / adj_matrix.size) * 100:.2f}%")

# Create a DataFrame with row and column labels (node indices)
n_nodes = adj_matrix.shape[0]
node_labels = [f"Node_{i}" for i in range(n_nodes)]

df = pd.DataFrame(adj_matrix, index=node_labels, columns=node_labels)

# Save to CSV
output_csv = 'CFGenerationForSGNN/preprocessing/data/METRLA_30/adjacency_matrix.csv'
print(f"\nSaving to CSV file: {output_csv}")

df.to_csv(output_csv)

print(f"✓ CSV file created successfully!")
print(f"\nThe CSV file contains the full {n_nodes}x{n_nodes} adjacency matrix")

# Also save a summary file
summary_csv = 'CFGenerationForSGNN/preprocessing/data/METRLA_30/adjacency_matrix_summary.csv'
summary_data = {
    'Metric': ['Number of Nodes', 'Total Edges (non-zero entries)', 'Sparsity (%)', 
               'Min Value', 'Max Value', 'Mean Value', 'Matrix Shape'],
    'Value': [n_nodes, np.count_nonzero(adj_matrix), 
              f"{(1 - np.count_nonzero(adj_matrix) / adj_matrix.size) * 100:.2f}",
              adj_matrix.min(), adj_matrix.max(), adj_matrix.mean(),
              f"{adj_matrix.shape[0]} x {adj_matrix.shape[1]}"]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(summary_csv, index=False)
print(f"✓ Summary CSV saved to: {summary_csv}")

