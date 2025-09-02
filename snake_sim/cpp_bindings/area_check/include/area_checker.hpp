#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector> 
#include <deque>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <cmath>

#include "area_types.hpp"    // for Coord, AreaCheckResult, etc.
#include "area_node.hpp"     // for AreaNode
#include "area_graph.hpp"    // for AreaGraph
#include "area_utils.hpp"    // for utility functions (if any)
#include "area_debug.hpp"

namespace py = pybind11;


class AreaChecker
{
public:
    // Constructor to initialize food_value, free_value, and body_value
    AreaChecker(
        uint8_t food_value, 
        uint8_t free_value, 
        uint8_t body_value, 
        uint8_t head_value, 
        int width, 
        int height
    ) : 
        food_value(food_value),
        free_value(free_value),
        body_value(body_value),
        head_value(head_value),
        width(width),
        height(height) {}

    bool is_inside(int x, int y)
    {
        return !(x < 0 || y < 0 || x >= this->width || y >= this->height);
    }

    void print_map(uint8_t *s_map);

    bool _is_bad_gateway(uint8_t *s_map, Coord coord1, Coord coord2);
    
    int is_gate_way(uint8_t *s_map, Coord coord, Coord check_coord);
    
    int py_is_gate_way(py::array_t<uint8_t> s_map, py::tuple coord, py::tuple check_coord)
    {
        auto buf = s_map.request();
        uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
        return is_gate_way(
            ptr,
            Coord(coord[0].cast<int>(), coord[1].cast<int>()),
            Coord(check_coord[0].cast<int>(), check_coord[1].cast<int>()));
    }
    
    ExploreResults explore_area(
        uint8_t *s_map,
        std::vector<Coord> &body_coords,
        Coord &start_coord,
        int area_id,
        std::vector<int> &checked,
        bool early_exit,
        int snake_length,
        int target_margin,
        int total_food_count);
        
    AreaCheckResult area_check(
        uint8_t *s_map,
        std::vector<Coord> &body_coords,
        Coord &start_coord,
        int target_margin,
        int max_food,
        bool food_check,
        bool exhaustive);
        
    py::dict py_area_check(
        py::array_t<uint8_t> s_map,
        py::list body_coords_py,
        py::tuple start_coord_py,
        int target_margin,
        int max_food,
        bool food_check,
        bool exhaustive);

private:
    uint8_t food_value;
    uint8_t free_value;
    uint8_t body_value;
    uint8_t head_value;
    int width;
    int height;
    Coord print_mark;
    int unexplored_area_id = -1;
};