#include "area_node.hpp"


void AreaNode::finalize_node()
{
    std::sort(body_tiles.begin(), body_tiles.end(), [](const std::pair<int, Coord> &a, const std::pair<int, Coord> &b) {
        return a.first > b.first;
    });
    has_only_head = (body_tiles.size() == 1 && body_tiles[0].first == 0);
    has_body = (body_tiles.size() > 0);
    edge_nodes.reserve(6);
}

void AreaNode::remove_connection(unsigned int edge)
{
    auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [edge](std::pair<AreaNode *, unsigned int> &pair)
    { return pair.second == edge; });
    if (it != edge_nodes.end())
    {
        edge_nodes.erase(it);
        neighbour_connections.erase(it->first->id);
    }
}

void AreaNode::remove_connection(AreaNode *other_node)
{
    auto edge = get_edge(other_node);
    if (edge != 0)
    {
        remove_connection(edge);
    }
}

void AreaNode::add_connection(AreaNode *new_node, unsigned int edge, ConnectedAreaInfo info)
{
    auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [new_node](std::pair<AreaNode *, unsigned int> &pair)
                            { return pair.first == new_node; });
    if (it == edge_nodes.end())
    {
        edge_nodes.push_back(std::make_pair(new_node, edge));
        neighbour_connections[new_node->id] = info;
    }
}

ConnectedAreaInfo AreaNode::get_connection_info(int area_id)
{
    if (neighbour_connections.find(area_id) != neighbour_connections.end())
    {
        return neighbour_connections[area_id];
    }
    throw std::out_of_range("Area ID not found in neighbour connections");
}

int AreaNode::get_countable_tiles(){
    if (is_one_dim || tile_count == 1){
        return tile_count;
    }
    else{
        return tile_count - std::abs(coord_parity_diff) - jagged_edge_discount;
    }
}

unsigned int AreaNode::get_edge(AreaNode *node)
{
    auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [node](std::pair<AreaNode *, unsigned int> &pair)
                            { return pair.first == node; });
    if (it != edge_nodes.end())
    {
        return it->second;
    }
    return 0;
}