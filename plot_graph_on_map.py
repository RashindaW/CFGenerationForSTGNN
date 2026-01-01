"""
Script to plot graph nodes on a real map using their latitude and longitude coordinates.
"""

import pandas as pd
import folium
from folium import plugins
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
csv_path = 'CFGenerationForSGNN/preprocessing/data/METRLA_SUB_30/graph_sensor_locations.csv'
df = pd.read_csv(csv_path)

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
for idx, row in df.iterrows():
    # Add the circle marker
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8,
        popup=f"Node: {row['new_index']}<br>"
              f"Sensor ID: {row['sensor_id']}<br>"
              f"Lat: {row['latitude']:.5f}<br>"
              f"Lon: {row['longitude']:.5f}",
        tooltip=f"Node {row['new_index']}",
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
            ">{row['new_index']}</div>
        ''')
    ).add_to(m)

# Add a heatmap layer (optional)
heat_data = [[row['latitude'], row['longitude']] for idx, row in df.iterrows()]
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
    df['longitude'], 
    df['latitude'], 
    c='red', 
    s=100, 
    alpha=0.7, 
    edgecolors='darkred',
    linewidths=2,
    zorder=5
)

# Add node labels (optional, can be commented out if too cluttered)
for idx, row in df.iterrows():
    ax.annotate(
        str(row['new_index']), 
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
    lons = df['longitude'].values
    lats = df['latitude'].values
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
    for i, row in df.iterrows():
        ax.annotate(
            str(row['new_index']),
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

