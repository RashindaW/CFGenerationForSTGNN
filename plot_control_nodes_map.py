from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


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


def _load_subgraph_nodes(subgraph_dir: Path, target_node: int) -> np.ndarray:
    subgraph_path = subgraph_dir / f"node_{target_node}.npy"
    if not subgraph_path.exists():
        raise FileNotFoundError(f"Subgraph file not found: {subgraph_path}")
    arr = np.load(subgraph_path, allow_pickle=False)
    if isinstance(arr, np.ndarray) and arr.dtype.fields and "node" in arr.dtype.fields:
        return arr["node"].astype(np.int64)
    return arr.astype(np.int64)


def _compute_hop_distances(adjacency: np.ndarray, start: int) -> np.ndarray:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency must be (N, N)")
    num_nodes = adjacency.shape[0]
    graph = adjacency > 0
    np.fill_diagonal(graph, False)
    distances = np.full(num_nodes, np.inf)
    distances[start] = 0.0
    frontier = [start]
    while frontier:
        next_frontier = []
        for node in frontier:
            neighbors = np.flatnonzero(graph[node])
            for nb in neighbors:
                if np.isinf(distances[nb]):
                    distances[nb] = distances[node] + 1.0
                    next_frontier.append(int(nb))
        frontier = next_frontier
    return distances


def _build_control_hops(
    adjacency: np.ndarray,
    subgraph_dir: Path,
    target_nodes: Iterable[int],
    max_hop: int,
) -> Dict[str, Dict[str, List[int]]]:
    control_hops: Dict[str, Dict[str, List[int]]] = {}
    for target in target_nodes:
        distances = _compute_hop_distances(adjacency, target)
        control_nodes = _load_subgraph_nodes(subgraph_dir, target)
        hop_map: Dict[str, List[int]] = {}
        for node in control_nodes.tolist():
            hop = distances[node]
            if not np.isfinite(hop) or hop <= 0:
                continue
            bucket = int(min(max_hop, int(hop)))
            hop_key = str(bucket)
            hop_map.setdefault(hop_key, []).append(int(node))
        for hop_key in hop_map:
            hop_map[hop_key].sort()
        control_hops[str(target)] = hop_map
    return control_hops


def _insert_or_replace_block(html: str, start_marker: str, end_marker: str, block: str) -> str:
    if start_marker in html and end_marker in html:
        pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)
        return pattern.sub(block, html)
    return html.replace("</body>", f"{block}\n</body>")


def _insert_or_replace_head_block(html: str, start_marker: str, end_marker: str, block: str) -> str:
    if start_marker in html and end_marker in html:
        pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)
        return pattern.sub(block, html)
    return html.replace("</head>", f"{block}\n</head>")


def update_interactive_map(
    html_path: Path,
    control_hops: Dict[str, Dict[str, List[int]]],
    target_nodes: List[int],
    hop_colors: Dict[str, str],
    default_target: int,
) -> None:
    if not html_path.exists():
        raise FileNotFoundError(f"Interactive map not found: {html_path}")
    html = html_path.read_text(encoding="utf-8")

    hop_legend_items = []
    for hop_key in sorted(hop_colors.keys(), key=lambda x: int(x)):
        label = f"{hop_key}+ hops" if hop_key == str(max(int(k) for k in hop_colors.keys())) else f"{hop_key} hop"
        if hop_key != "1":
            label = f"{hop_key} hops" if hop_key != str(max(int(k) for k in hop_colors.keys())) else label
        hop_legend_items.append(
            f'<div class="control-node-legend-row"><span class="control-node-swatch" '
            f'style="background:{hop_colors[hop_key]};"></span>{label}</div>'
        )
    hop_legend_items.append(
        '<div class="control-node-legend-row"><span class="control-node-swatch" '
        'style="background:#111;"></span>target node</div>'
    )

    selector_options = "\n".join(
        f'<option value="{target}">{target}</option>' for target in target_nodes
    )

    css_block = """<!-- CONTROL_NODE_CSS_START -->
<style>
  .control-node-panel {
    position: absolute;
    top: 10px;
    left: 10px;
    z-index: 1000;
    background: white;
    padding: 10px 12px;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    font-family: sans-serif;
    font-size: 13px;
    min-width: 160px;
  }
  .control-node-panel label {
    display: block;
    margin-bottom: 6px;
    font-weight: 600;
  }
  .control-node-panel select {
    width: 100%;
    padding: 4px 6px;
    margin-bottom: 8px;
    font-size: 13px;
  }
  .control-node-legend {
    border-top: 1px solid #ddd;
    padding-top: 6px;
  }
  .control-node-legend-row {
    display: flex;
    align-items: center;
    margin-bottom: 4px;
  }
  .control-node-swatch {
    width: 12px;
    height: 12px;
    display: inline-block;
    margin-right: 6px;
    border-radius: 50%;
  }
</style>
<!-- CONTROL_NODE_CSS_END -->"""

    panel_block = f"""<!-- CONTROL_NODE_PANEL_START -->
<div class="control-node-panel">
  <label for="controlNodeTarget">Control nodes for target</label>
  <select id="controlNodeTarget">
{selector_options}
  </select>
  <div class="control-node-legend">
    {''.join(hop_legend_items)}
  </div>
</div>
<!-- CONTROL_NODE_PANEL_END -->"""

    script_block = f"""<!-- CONTROL_NODE_SCRIPT_START -->
<script>
  const controlNodeHops = {json.dumps(control_hops)};
  const hopColors = {json.dumps(hop_colors)};
  const targetColor = "#111";
  const defaultStroke = "#777";
  const defaultFill = "#c0c0c0";

  function findMapObject() {{
    for (var prop in window) {{
      try {{
        if (window[prop] && window[prop]._layers) {{
          return window[prop];
        }}
      }} catch (e) {{}}
    }}
    return null;
  }}

  function buildControlLookup(targetId) {{
    var hopGroups = controlNodeHops[String(targetId)] || {{}};
    var lookup = {{}};
    for (var hopKey in hopGroups) {{
      var nodes = hopGroups[hopKey] || [];
      for (var i = 0; i < nodes.length; i++) {{
        lookup[nodes[i]] = hopKey;
      }}
    }}
    return lookup;
  }}

  function updateMapForTarget(targetId) {{
    var mapObj = findMapObject();
    if (!mapObj) {{
      return;
    }}
    var lookup = buildControlLookup(targetId);
    mapObj.eachLayer(function(layer) {{
      if (!layer.setStyle || !layer.getPopup) {{
        return;
      }}
      var popup = layer.getPopup();
      if (!popup) {{
        return;
      }}
      var content = popup.getContent();
      if (!content) {{
        return;
      }}
      var match = String(content).match(/Adj Index: (\\d+)/);
      if (!match) {{
        return;
      }}
      var nodeId = parseInt(match[1], 10);
      if (nodeId === parseInt(targetId, 10)) {{
        layer.setStyle({{color: targetColor, fillColor: targetColor, fillOpacity: 0.95, radius: 9, weight: 3}});
        return;
      }}
      var hopKey = lookup[nodeId];
      if (hopKey) {{
        var color = hopColors[String(hopKey)] || "#1f77b4";
        layer.setStyle({{color: color, fillColor: color, fillOpacity: 0.9, radius: 7, weight: 2}});
      }} else {{
        layer.setStyle({{color: defaultStroke, fillColor: defaultFill, fillOpacity: 0.4, radius: 5, weight: 1}});
      }}
    }});
  }}

  document.addEventListener('DOMContentLoaded', function() {{
    var selector = document.getElementById('controlNodeTarget');
    if (selector) {{
      selector.value = String({default_target});
      selector.addEventListener('change', function() {{
        updateMapForTarget(this.value);
      }});
      updateMapForTarget(selector.value);
    }} else {{
      updateMapForTarget(String({default_target}));
    }}
  }});
</script>
<!-- CONTROL_NODE_SCRIPT_END -->"""

    html = _insert_or_replace_head_block(html, "<!-- CONTROL_NODE_CSS_START -->", "<!-- CONTROL_NODE_CSS_END -->", css_block)
    html = _insert_or_replace_block(html, "<!-- CONTROL_NODE_PANEL_START -->", "<!-- CONTROL_NODE_PANEL_END -->", panel_block)
    html = _insert_or_replace_block(html, "<!-- CONTROL_NODE_SCRIPT_START -->", "<!-- CONTROL_NODE_SCRIPT_END -->", script_block)
    html_path.write_text(html, encoding="utf-8")
    print(f"Updated interactive map: {html_path}")


def save_static_maps(
    output_dir: Path,
    locations: pd.DataFrame,
    control_hops: Dict[str, Dict[str, List[int]]],
    target_nodes: Iterable[int],
    hop_colors: Dict[str, str],
    max_hop: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping static map generation.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    lat = locations["latitude"].to_numpy()
    lon = locations["longitude"].to_numpy()
    valid = np.isfinite(lat) & np.isfinite(lon)

    for target in target_nodes:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(lon[valid], lat[valid], s=20, color="#c7c7c7", alpha=0.6, linewidths=0)

        hop_groups = control_hops.get(str(target), {})
        for hop_key in sorted(hop_groups.keys(), key=lambda x: int(x)):
            nodes = hop_groups[hop_key]
            if not nodes:
                continue
            idx = np.array(nodes, dtype=int)
            idx = idx[(idx >= 0) & (idx < lat.shape[0])]
            idx = idx[valid[idx]]
            if idx.size == 0:
                continue
            label = f"{hop_key}+ hops" if int(hop_key) == max_hop else f"{hop_key} hop"
            if int(hop_key) != 1 and int(hop_key) != max_hop:
                label = f"{hop_key} hops"
            ax.scatter(
                lon[idx],
                lat[idx],
                s=80,
                color=hop_colors[hop_key],
                label=label,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
            )

        if 0 <= target < lat.shape[0] and valid[target]:
            ax.scatter(
                [lon[target]],
                [lat[target]],
                s=140,
                color="#111",
                marker="*",
                label="target node",
                zorder=5,
            )

        ax.set_title(f"Control nodes for target {target}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        output_path = output_dir / f"control_nodes_target_{target}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved static map: {output_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_root = base_dir / "preprocessing" / "data"
    map_dir = data_root / "METRLA"
    subgraph_dir = data_root / "METRLA_30" / "subgraphs_random_walk"

    target_nodes = [64, 180, 89]
    max_hop = 4
    hop_colors = {
        "1": "#1f77b4",
        "2": "#2ca02c",
        "3": "#ff7f0e",
        "4": "#d62728",
    }

    adj_path = map_dir / "adj_mat.npy"
    if not adj_path.exists():
        raise FileNotFoundError(f"Adjacency file not found: {adj_path}")
    adjacency = np.load(adj_path)

    csv_path = map_dir / "graph_sensor_locations.csv"
    sensor_attr_path = map_dir / "node_attributes.npy"
    if not csv_path.exists():
        raise FileNotFoundError("Missing sensor locations for METRLA.")
    df = pd.read_csv(csv_path)
    if sensor_attr_path.exists():
        sensor_ids = _load_sensor_ids(sensor_attr_path)
        locations = _align_locations_to_adjacency(df, sensor_ids)
    else:
        print("Warning: node_attributes.npy not found; using CSV order for adjacency alignment.")
        locations = df.copy()
        if "adj_index" not in locations.columns:
            if "index" in locations.columns:
                locations["adj_index"] = locations["index"].astype(int)
            else:
                locations["adj_index"] = np.arange(len(locations), dtype=np.int64)

    control_hops = _build_control_hops(adjacency, subgraph_dir, target_nodes, max_hop)

    html_path = map_dir / "sensors_map_interactive_with_search.html"
    update_interactive_map(
        html_path=html_path,
        control_hops=control_hops,
        target_nodes=target_nodes,
        hop_colors=hop_colors,
        default_target=target_nodes[0],
    )

    output_dir = map_dir / "control_node_maps"
    save_static_maps(output_dir, locations, control_hops, target_nodes, hop_colors, max_hop)


if __name__ == "__main__":
    main()
