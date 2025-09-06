#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include "area_types.hpp"


class AreaNode
{
public:
    Coord start_coord;
    int id;
    int tile_count = 0;
    int food_count = 0;
    int coord_parity_diff = 0;
    bool is_one_dim = false; // one_dim is true if the area is a 1 tile wide path
    bool has_tail = false;
    bool has_only_head = false;
    bool has_body = false;
    std::vector<std::pair<int, Coord>> body_tiles; // vector of (snake_index, coord) for body tiles in this area
    std::unordered_map<int, ConnectedAreaInfo> neighbour_connections; // map of area_id to pair of coords that connect the areas
    std::vector<std::pair<AreaNode *, unsigned int>> edge_nodes;

    AreaNode(Coord start_coord, int id) : start_coord(start_coord), id(id){}

    void finalize_node();

    void remove_connection(AreaNode *other_node);

    void remove_connection(unsigned int edge);

    void add_connection(AreaNode *new_node, unsigned int edge, ConnectedAreaInfo info);

    int get_nr_connections() { return edge_nodes.size(); }

    ConnectedAreaInfo get_connection_info(int area_id);

    int get_countable_tiles();

    unsigned int get_edge(AreaNode *node);

    ~AreaNode() = default;
};