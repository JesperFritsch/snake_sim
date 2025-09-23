#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include <iostream>


#include "util_types.hpp"


namespace py = pybind11;


inline unsigned int cantor_pairing(int k1, int k2)
{
    return (k1 + k2) * (k1 + k2 + 1) / 2 + k2;
}

inline bool tile_has_food(uint8_t *s_map, int width, Coord coord, uint8_t food_value){
    return s_map[coord.y * width + coord.x] == food_value;
}

inline bool get_coord_mod_parity(Coord coord)
// returns true if the tile is even and false if odd, think black and white on a chessboard.
{
    return coord.x % 2 == coord.y % 2;
}

std::vector<Coord> get_locations_with_value(uint8_t *s_map, int width, int height, uint8_t value);

inline py::list py_get_locations_with_value(py::array_t<uint8_t> s_map, int width, int height, uint8_t value){
    auto buf = s_map.request();
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
    std::vector<Coord> locations = get_locations_with_value(ptr, width, height, value);
    py::list result;
    for (const auto& loc : locations){
        result.append(py::make_tuple(loc.x, loc.y));
    }
    return result;
}

bool can_make_area_inaccessible(
    uint8_t *s_map, 
    int width, 
    int height,
    int free_value,
    Coord head_pos,
    Coord direction);

inline bool py_can_make_area_inaccessible(
    py::array_t<uint8_t> s_map, 
    int width, 
    int height,
    int free_value,
    py::tuple head_pos,
    py::tuple direction
){
    auto buf = s_map.request();
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
    Coord head(head_pos[0].cast<int>(), head_pos[1].cast<int>());
    Coord dir(direction[0].cast<int>(), direction[1].cast<int>());
    return can_make_area_inaccessible(ptr, width, height, free_value, head, dir);
};

void print_map(uint8_t *s_map, int width, int height, int head_value, int body_value, int food_value);
