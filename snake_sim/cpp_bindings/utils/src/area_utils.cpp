#include "area_utils.hpp"


std::vector<Coord> get_locations_with_value(uint8_t *s_map, int width, int height, uint8_t value) {
    std::vector<Coord> locations;
    locations.reserve(width * height);
    int size = width * height;
    for (int idx = 0; idx < size; ++idx) {
        if (s_map[idx] == value) {
            int y = idx / width;
            int x = idx % width;
            locations.push_back(Coord(x, y));
        }
    }
    return locations;
}