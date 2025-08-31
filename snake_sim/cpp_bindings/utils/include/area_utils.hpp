#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
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
