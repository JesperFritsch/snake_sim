import re
import networkx
import ast
from matplotlib import pyplot as plt
from pathlib import Path

class GraphNode:
    def __init__(self,
                id,
                start_coord,
                end_coord,
                tiles,
                food,
                max_index,
                one_dim,
                has_tail,
                edge_nodes):
        self.id = id
        self.start_coord = start_coord
        self.end_coord = end_coord
        self.tiles = tiles
        self.food = food
        self.max_index = max_index
        self.one_dim = one_dim
        self.has_tail = has_tail
        self.edge_nodes = edge_nodes

    @classmethod
    def from_lines(cls, lines):
        id = int(re.search(r"id: (\d+)", lines[0]).group(1))
        start_coord = ast.literal_eval(re.search(r"start coord: (.+)", lines[1]).group(1))
        end_coord = ast.literal_eval(re.search(r"end coord: (.+)", lines[2]).group(1))
        tile_count = int(re.search(r"tile count: (\d+)", lines[3]).group(1))
        food_count = int(re.search(r"food count: (\d+)", lines[4]).group(1))
        max_index = int(re.search(r"max index: (\d+)", lines[5]).group(1))
        one_dim = int(re.search(r"one dim: (\d+)", lines[6]).group(1))
        has_tail = bool(re.search(r"has tail: (\w+)", lines[7]).group(1))
        edge_nodes = ast.literal_eval(f"({re.search(r'edge nodes: (.+)', lines[8]).group(1)})")
        return cls(id, start_coord, end_coord, tile_count, food_count, max_index, one_dim, has_tail, edge_nodes)

with open(Path("test/test_data/node_graph.txt")) as f:
    lines = f.readlines()
    current_lines = []
    area_nodes = []
    for i, line in enumerate(lines):
        current_lines.append(line)
        if (i + 1 == len(lines)) or lines[i + 1].startswith("Node id:"):
            area_nodes.append(GraphNode.from_lines(current_lines))
            current_lines = []

    print(f"Node count: {len(area_nodes)}")

G = networkx.Graph()
print(area_nodes[0].__dict__)
# Add nodes
for node in area_nodes:
    attrs = node.__dict__.copy()
    attrs.pop("id")
    attrs.pop("edge_nodes")
    G.add_node(node.id, **attrs)

# Add edges
for node in area_nodes:
    for edge_node in node.edge_nodes:
        id, edge_id = edge_node
        G.add_edge(node.id, id)

# pos = networkx.planar_layout(G)
# pos = networkx.circular_layout(G)  # Circular layout
# pos = networkx.shell_layout(G)  # Shell layout
pos = networkx.kamada_kawai_layout(G)  # Kamada-Kawai layout
# pos = networkx.spectral_layout(G)  # Spectral layout

# for key in pos:
#     pos[key] *= 100.0

plt.figure(figsize=(15, 15))

labels = {node.id: f"{node.id}\nT: {node.tiles}" for node in area_nodes}

networkx.draw(G, pos, with_labels=True, labels=labels, node_size=2000)

plt.show()