
import networkx as nx


class MapGraph:
    def __init__(self):
        self.node_ids = set()
        self.graph = nx.Graph()

    def add_node(self, parent_id=None):
        node_id = len(self.node_ids)
        self.node_ids.add(node_id)
        self.graph.add_node(node_id)
        if parent_id is not None:
            self.graph.add_edge(parent_id, node_id, e_type='child')
        return node_id

    def connect_nodes(self, id_a, id_b):
        self.graph.add_edge(id_a, id_b, e_type='connection')

    def get_children(self, node_id):
        return [n for n in self.graph.neighbors(node_id) if self.graph.edges[(node_id, n)]['e_type'] == 'child']

    def get_parent(self, node_id):
        parents = [n for n in self.graph.neighbors(node_id) if self.graph.edges[(node_id, n)]['e_type'] == 'child']
        if len(parents) > 1:
            raise ValueError("Node has more than one parent")
        if len(parents) == 0:
            return None
        return parents[0]

    def get_connections(self, node_id):
        return [n for n in self.graph.neighbors(node_id) if self.graph.edges[(node_id, n)]['e_type'] == 'connection']

    def get_nodes_at_depth(self, depth):
        if depth < 0:
            raise ValueError("Depth must be non-negative")
        if depth == 0:
            return [0] if 0 in self.node_ids else []
        current_level = [0] if 0 in self.node_ids else []
        for _ in range(depth):
            next_level = []
            for node in current_level:
                next_level.extend(self.get_children(node))
            current_level = next_level
        return current_level