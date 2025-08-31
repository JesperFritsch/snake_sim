#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <memory>

#include "area_types.hpp"
#include "area_node.hpp"
#include "area_utils.hpp"



struct SearchNode
{
    AreaNode *node = nullptr;
    int nr_visits = 0;
    int tiles_until_here = 0;
    int food_until_here = 0;
    int first_tiles_until_here = 0; // only set on first visit
    int first_food_until_here = 0; // only set on first visit
    
    std::vector<std::vector<unsigned int>> searched_edges;
    std::vector<unsigned int> used_edges;
    std::vector<Coord> used_coords;
    std::vector<int> entered_from_nodes;

    SearchNode() = default;

    SearchNode(AreaNode *node) : node(node) {}

    bool first_visit()
    {
        return nr_visits == 1;
    }

    bool is_visited()
    {
        return nr_visits > 0;
    }

    bool visited_before()
    {
        return nr_visits > 1;
    }

    bool is_used_edge(unsigned int edge)
    {
        return std::find(used_edges.begin(), used_edges.end(), edge) != used_edges.end();
    }

    bool is_searched_edge(unsigned int edge)
    {
        auto searched = searched_edges.back();
        if (std::find(searched.begin(), searched.end(), edge) != searched.end())
        {
            return true;
        }
        return false;
    }

    void add_searched_edge(unsigned int edge)
    {
        searched_edges.back().push_back(edge);
    }

    void add_used_edge(unsigned int edge)
    {
        used_edges.push_back(edge);
    }

    void enter_from(AreaNode *prev_node, int tiles, int food);

    void enter_unwind(int tiles, int food);

    void exit_to(AreaNode *next_node);

    void exit_unwind();

    std::pair<AreaNode *, unsigned int> get_next_node_and_edge();

    void reset()
    {
        nr_visits = 0;
        tiles_until_here = 0;
        food_until_here = 0;
        searched_edges.clear();
        used_edges.clear();
    }

    TileCounts tile_count_on_enter();

    TileCounts tile_count_on_exit(AreaNode *next_node, uint8_t *s_map, int width, uint8_t food_value);

    bool is_conn_coord_start_cord(AreaNode *next_node);

    bool is_conn_coord_used(AreaNode *other_node);

    bool is_coord_used(Coord coord);

    bool can_enter_next_node(AreaNode *next_node)
    {
        return (is_conn_coord_used(next_node) && visited_before()) ? false : true;
    }

    bool can_enter_from_node(AreaNode *from_node)
    {
        return (is_conn_coord_used(from_node) || (is_visited() && node->tile_count == 1)) ? false : true;
    }

    std::pair<int, Coord> get_max_body_index_pair();

    Coord get_entry_coord();

    int path_parity_tile_adjustment(Coord start, Coord end);

    int get_max_body_tile_adjustment(Coord max_index_coord);

    int path_tile_adjustment(AreaNode *next_node);

};


class AreaGraph
{
public:
    int next_id = 0;
    int map_width = 0;
    int map_height = 0;
    AreaNode *root = nullptr;
    std::unordered_map<int, std::shared_ptr<AreaNode>> nodes;

    AreaGraph(){ nodes.reserve(200); }

    AreaGraph(int width, int height) : map_width(width), map_height(height) { nodes.reserve(200); }

    void connect_nodes(int id1, int id2, ConnectedAreaInfo conn_info);

    void connect_nodes(AreaNode *node1, AreaNode *node2, ConnectedAreaInfo conn_info1, ConnectedAreaInfo conn_info2);

    AreaNode *get_node(int id)
    {
        if (nodes.find(id) == nodes.end())
        {
            return nullptr;
        }
        return nodes[id].get();
    }

    AreaNode *add_node(Coord start_coord)
    {
        return add_node_with_id(start_coord, next_id++);
    }

    AreaNode *add_node_with_id(Coord start_coord, int id);

    void add_id_for_node(int original_id, int linked_id);

    void remove_node(int id);

    void print_nodes_debug() const;

    AreaCheckResult search_best2(
        int snake_length, 
        uint8_t *s_map, 
        uint8_t food_value, 
        int width, 
        int target_margin, 
        int max_food,
        bool food_check, 
        bool exhaustive
    );

    ~AreaGraph() = default;
};