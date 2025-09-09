#pragma once
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <climits>
#include <limits>

#include "util_types.hpp"

class AreaNode;


struct AreaCheckResult
{
    bool is_clear;
    int tile_count;
    int total_steps;
    int food_count;
    bool has_tail;
    int margin;
    int needed_steps;

    AreaCheckResult() : 
        is_clear(false),
        tile_count(0),
        total_steps(0),
        food_count(0),
        has_tail(false),
        margin(INT_MIN),
        needed_steps(0) {}

    AreaCheckResult(
        bool is_clear,
        int tile_count,
        int total_steps,
        int food_count,
        bool has_tail,
        int margin,
        int needed_steps
    ) : 
        is_clear(is_clear),
        tile_count(tile_count),
        total_steps(total_steps),
        food_count(food_count),
        has_tail(has_tail),
        margin(margin),
        needed_steps(needed_steps) {}
};


struct ConnectedAreaInfo
{
    int id;
    Coord self_coord;
    Coord other_coord;
    bool is_bad_gateway_from_here;
    bool is_bad_gateway_to_here;
    bool is_diag_gateway;

    ConnectedAreaInfo() : 
        id(-1),
        self_coord(Coord()),
        other_coord(Coord()),
        is_bad_gateway_from_here(false),
        is_bad_gateway_to_here(false),
        is_diag_gateway(false) {}

    ConnectedAreaInfo(
        int id, 
        Coord self_coord, 
        Coord other_coord, 
        bool is_bad_gateway_from_here,
        bool is_bad_gateway_to_here,
        bool is_diag_gateway
    ) : id(id),
        self_coord(self_coord),
        other_coord(other_coord),
        is_bad_gateway_from_here(is_bad_gateway_from_here),
        is_bad_gateway_to_here(is_bad_gateway_to_here),
        is_diag_gateway(is_diag_gateway) {}
};


struct ExploreData
{
    Coord start_coord;
    int area_id;
    AreaNode *prev_node;

    ExploreData() : start_coord(Coord()),
                    area_id(-1),
                    prev_node(nullptr) {}

    ExploreData(
        Coord start_coord, 
        int area_id, 
        AreaNode *prev_node
    ) : 
        start_coord(start_coord),
        area_id(area_id),
        prev_node(prev_node) {}
};



struct ExploreResults
{
    int tile_count = 0;
    int food_count = 0;
    int coord_parity_diff = 0; // coord_parity_diff is positive if there are more even tiles (black), negative if more odd tiles (white)
    bool early_exit = false;
    bool has_tail = false;
    std::vector<std::pair<std::pair<int, bool>, Coord>> body_tiles;
    std::vector<ConnectedAreaInfo> connected_areas;
    std::vector<Coord> to_explore;
    std::vector<Coord> jagged_edge_tiles;

    ExploreResults() {}

    ExploreResults(
        int tile_count,
        int food_count,
        int coord_parity_diff,
        bool early_exit,
        bool has_tail,
        std::vector<std::pair<std::pair<int, bool>, Coord>> body_tiles,
        std::vector<ConnectedAreaInfo> connected_areas,
        std::vector<Coord> to_explore,
        std::vector<Coord> jagged_edge_tiles
    ) : 
        tile_count(tile_count),
        food_count(food_count),
        coord_parity_diff(coord_parity_diff),
        early_exit(early_exit),
        has_tail(has_tail),
        body_tiles(body_tiles),
        connected_areas(connected_areas),
        to_explore(to_explore),
        jagged_edge_tiles(jagged_edge_tiles) {}
};


struct TileCounts
{
    int total_tiles;
    int total_food;
    int new_tiles;

    TileCounts() : 
        total_tiles(0),
        total_food(0),
        new_tiles(0) {}

    TileCounts(int total_tiles, int total_food, int new_tiles) : 
        total_tiles(total_tiles),
        total_food(total_food),
        new_tiles(new_tiles) {}
};


struct RecurseCheckResult
{
    std::unordered_map<Coord, std::unordered_map<int, float>> best_margin_fracs_at_depth;

    RecurseCheckResult() : 
        best_margin_fracs_at_depth(std::unordered_map<Coord, std::unordered_map<int, float>>()) {}
};

struct RecurseCheckFrame
{
    Coord head;
    Coord old_tail;
    std::vector<Coord> to_visit;
    Coord base_coord;
    bool setup_done = false;
    bool did_grow = false;
    float best_margin_frac = std::numeric_limits<float>::lowest();

    RecurseCheckFrame(
        Coord head,
        std::vector<Coord> to_visit,
        Coord base_coord = Coord()
    ) :
        head(head),
        to_visit(to_visit),
        base_coord(base_coord){}
};