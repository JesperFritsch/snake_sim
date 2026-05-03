#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <queue>
#include <cstdint>
#include <vector>
#include <iostream>
#include <unordered_set>


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
    Coord direction
);

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

int32_t* dist_map(
    uint8_t *s_map,
    int width,
    int height,
    int free_value,
    int target_value
);

inline py::array_t<int32_t> py_dist_map(
    py::array_t<uint8_t> s_map,
    int width,
    int height,
    int free_value,
    int target_value
){
    auto buf = s_map.request();
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
    int32_t* heat_map = dist_map(ptr, width, height, free_value, target_value);
    py::capsule free_when_done(heat_map, [](void *p) {
        delete[] static_cast<int32_t*>(p);
    });

    return py::array_t<int32_t>(
        {height, width},                                          // shape
        {sizeof(int32_t) * width, sizeof(int32_t)},               // strides
        heat_map,                                                 // data ptr
        free_when_done                                            // owner
    );
}

std::vector<int> area_boundary_tiles(
    uint8_t *s_map,
    int width,
    int height,
    int free_value,
    Coord area_start
);

inline py::list py_area_boundary_tiles(
    py::array_t<uint8_t> s_map,
    int width,
    int height,
    int free_value,
    py::tuple area_start
){
    auto buf = s_map.request();
    uint8_t *ptr = static_cast<uint8_t *>(buf.ptr);
    Coord start(area_start[0].cast<int>(), area_start[1].cast<int>());
    std::vector<int> boundary_tiles = area_boundary_tiles(ptr, width, height, free_value, start);
    py::list result;
    for (const auto& tile : boundary_tiles){
        result.append(tile);
    }
    return result;
}

std::unordered_map<int, int> voronoi_maps(
    const uint8_t* s_map,
    int width,
    int height,
    int free_value,
    const std::unordered_map<int32_t, Coord>& owners,
    int32_t* ownership_map,
    int32_t* distance_map
);

inline std::unordered_map<int, int> py_voronoi_maps(
    py::array_t<uint8_t> s_map,
    int width,
    int height,
    int free_value,
    const std::unordered_map<int32_t, std::pair<int, int>>& owners_py,
    py::array_t<int32_t> ownership_map,
    py::array_t<int32_t> distance_map
) {
    auto buf = s_map.request();
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.ptr);

    std::unordered_map<int32_t, Coord> owners_cpp;
    owners_cpp.reserve(owners_py.size());
    for (const auto& [k, v] : owners_py) {
        owners_cpp.emplace(k, Coord(v.first, v.second));
    }

    int32_t* ownership_ptr = static_cast<int32_t*>(ownership_map.request().ptr);
    int32_t* distance_ptr  = static_cast<int32_t*>(distance_map.request().ptr);

    return voronoi_maps(ptr, width, height, free_value, owners_cpp,
                 ownership_ptr, distance_ptr);

}