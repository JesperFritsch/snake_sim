#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>
#include <deque>
#include <iostream>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

#include "util_types.hpp"

namespace py = pybind11;


Coord get_dir_to_tile(
    uint8_t *s_map, 
    int width, 
    int height, 
    Coord from_coord, 
    int tile_value,
    std::vector<int> visitable_values,
    bool clockwise = true
);

inline py::tuple py_get_dir_to_tile(
    py::array_t<uint8_t> s_map, 
    int width, 
    int height, 
    py::tuple from_coord, 
    int tile_value,
    std::vector<int> visitable_values,
    bool clockwise = true
){
    auto buf = s_map.request();
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
    Coord head(from_coord[0].cast<int>(), from_coord[1].cast<int>());
    Coord result = get_dir_to_tile(ptr, width, height, head, tile_value, visitable_values, clockwise);
    return py::make_tuple(result.x, result.y);
}

std::vector<Coord> get_visitable_tiles(
    uint8_t *s_map, 
    int width, 
    int height, 
    Coord center_coord, 
    std::vector<int> visitable_values
);

inline py::tuple py_get_visitable_tiles(
    py::array_t<uint8_t> s_map, 
    int width, 
    int height, 
    py::tuple center_coord, 
    std::vector<int> visitable_values
){
    auto buf = s_map.request();
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
    Coord center(center_coord[0].cast<int>(), center_coord[1].cast<int>());
    std::vector<Coord> result = get_visitable_tiles(ptr, width, height, center, visitable_values);
    py::tuple py_result(result.size());
    for (size_t i = 0; i < result.size(); ++i){
        py_result[i] = py::make_tuple(result[i].x, result[i].y);
    }
    return py_result;
}

inline bool same_diagonal(Coord c1, Coord c2, bool dec){
    // dec = true when y decreases when x increases
    // . . X
    // . X .
    // X . .
    return (c1.x - c2.x) == (!dec ? (c1.y - c2.y) : (c2.y - c1.y));
}

bool is_free_diagonal(
    const uint8_t __restrict *s_map, 
    int width, 
    Coord from_coord, 
    Coord to_coord, 
    std::vector<int> visitable_values
);
