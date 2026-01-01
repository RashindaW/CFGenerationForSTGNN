"""
Script to plot graph nodes on a real map using their latitude and longitude coordinates.
"""

from pathlib import Path
import pandas as pd
import folium
from folium import plugins
import matplotlib.pyplot as plt
import numpy as np

def _load_sensor_ids(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype.fields and "sensor_id" in data.dtype.fields:
        return data["sensor_id"].astype(str)
    if isinstance(data, np.ndarray) and data.ndim == 1:
        if data.dtype == object and len(data) > 0 and isinstance(data[0], dict) and "sensor_id" in data[0]:
            return np.array([d["sensor_id"] for d in data], dtype=str)
        return data.astype(str)
    raise ValueError(f"Unsupported sensor_id format in {path}")


def _align_locations_to_adjacency(df: pd.DataFrame, sensor_ids: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["sensor_id"] = df["sensor_id"].astype(str)
    order = [str(x) for x in sensor_ids.tolist()]
    missing_in_df = sorted(set(order) - set(df["sensor_id"]))
    extra_in_df = sorted(set(df["sensor_id"]) - set(order))
    if missing_in_df:
        print(f"Warning: {len(missing_in_df)} adjacency sensor_ids missing in locations CSV.")
    if extra_in_df:
        print(f"Warning: {len(extra_in_df)} CSV sensor_ids not present in adjacency order.")
    aligned = df.set_index("sensor_id", drop=False).reindex(order)
    aligned["adj_index"] = np.arange(len(order), dtype=np.int64)
    return aligned


# Read the CSV file
dataset_dir = Path("CFGenerationForSGNN/preprocessing/data/METRLA_SUB_30")
csv_path = dataset_dir / "graph_sensor_locations.csv"
df = pd.read_csv(csv_path)

adj_path = dataset_dir / "adj_mat.npy"
sensor_attr_path = dataset_dir / "node_attributes.npy"
if adj_path.exists() and sensor_attr_path.exists():
    adjacency = np.load(adj_path)
    sensor_ids = _load_sensor_ids(sensor_attr_path)
    if adjacency.shape[0] != len(sensor_ids):
        print(
            "Warning: adjacency size does not match sensor_id list length "
            f"({adjacency.shape[0]} vs {len(sensor_ids)})."
        )
    df = _align_locations_to_adjacency(df, sensor_ids)
else:
    print("Warning: adjacency or node attribute file not found; using CSV order for plotting.")

df_plot = df[df["sensor_id"].notna()]

print(f"Loaded {len(df)} nodes from the graph")
print(f"Latitude range: {df['latitude'].min():.5f} to {df['latitude'].max():.5f}")
print(f"Longitude range: {df['longitude'].min():.5f} to {df['longitude'].max():.5f}")

# Calculate center of the map
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()

# Create interactive map with folium
print("\nCreating interactive map...")
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=13,
    tiles='OpenStreetMap'
)

# Add markers for each node with visible labels
for idx, row in df_plot.iterrows():
    # Add the circle marker
    adj_index = int(row.get("adj_index", row["new_index"]))
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8,
        popup=f"Adj Index: {adj_index}<br>"
              f"Node: {row['new_index']}<br>"
              f"Sensor ID: {row['sensor_id']}<br>"
              f"Lat: {row['latitude']:.5f}<br>"
              f"Lon: {row['longitude']:.5f}",
        tooltip=f"Node {adj_index}",
        color='darkred',
        fill=True,
        fillColor='red',
        fillOpacity=0.8,
        weight=2
    ).add_to(m)
    
    # Add a text label on top of the marker
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=folium.DivIcon(html=f'''
            <div style="
                font-size: 10px;
                font-weight: bold;
                color: white;
                text-align: center;
                text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;
                width: 20px;
                margin-left: -10px;
                margin-top: -5px;
            ">{adj_index}</div>
        ''')
    ).add_to(m)

# Add a heatmap layer (optional)
heat_data = [[row['latitude'], row['longitude']] for idx, row in df_plot.iterrows()]
plugins.HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(
    folium.FeatureGroup(name='Heat Map', show=False).add_to(m)
)

# Add layer control
folium.LayerControl().add_to(m)

# Save interactive map
output_html = 'CFGenerationForSGNN/preprocessing/data/METRLA_SUB_30/nodes_map_interactive.html'
m.save(output_html)
print(f"Interactive map saved to: {output_html}")

# Create static map with matplotlib
print("\nCreating static map...")
fig, ax = plt.subplots(figsize=(12, 10))

# Plot nodes
scatter = ax.scatter(
    df_plot['longitude'], 
    df_plot['latitude'], 
    c='red', 
    s=100, 
    alpha=0.7, 
    edgecolors='darkred',
    linewidths=2,
    zorder=5
)

# Add node labels (optional, can be commented out if too cluttered)
for idx, row in df_plot.iterrows():
    adj_index = int(row.get("adj_index", row["new_index"]))
    ax.annotate(
        str(adj_index),
        (row['longitude'], row['latitude']),
        fontsize=7,
        ha='center',
        va='center',
        color='white',
        weight='bold',
        zorder=6
    )

# Try to add basemap using contextily (if available)
try:
    import contextily as ctx
    # Convert to Web Mercator for contextily
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    
    # Transform coordinates
    lons = df_plot['longitude'].values
    lats = df_plot['latitude'].values
    x, y = transformer.transform(lons, lats)
    
    # Create new plot with transformed coordinates
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        x, y,
        c='red',
        s=100,
        alpha=0.7,
        edgecolors='darkred',
        linewidths=2,
        zorder=5
    )
    
    # Add labels
    for i, row in df_plot.iterrows():
        adj_index = int(row.get("adj_index", row["new_index"]))
        ax.annotate(
            str(adj_index),
            (x[i], y[i]),
            fontsize=7,
            ha='center',
            va='center',
            color='white',
            weight='bold',
            zorder=6
        )
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=14)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'METRLA_SUB_30 Graph Nodes on Map\n({len(df)} nodes)', fontsize=14, weight='bold')
    
    print("Static map created with contextily basemap")
    
except ImportError:
    # Fallback to simple plot without basemap
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'METRLA_SUB_30 Graph Nodes\n({len(df)} nodes)', fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    print("Static map created without basemap (install contextily for basemap)")

plt.tight_layout()
output_png = 'CFGenerationForSGNN/preprocessing/data/METRLA_SUB_30/nodes_map_static.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Static map saved to: {output_png}")

plt.close()

print("\n" + "="*50)
print("Map visualization complete!")
print("="*50)
print(f"Interactive HTML map: {output_html}")
print(f"Static PNG map: {output_png}")
print("\nOpen the HTML file in a web browser for an interactive map with zoom and pan capabilities.")

