#pragma once

#include <cstdint>
#include "util_types.hpp"

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