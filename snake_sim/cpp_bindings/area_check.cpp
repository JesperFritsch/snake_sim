#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // For automatic conversion of C++ STL containers to Python
#include <deque>
#include <vector>
#include <algorithm>
#include <iostream> // Include iostream for std::cout
#include <typeinfo> // Include typeinfo for typeid

namespace py = pybind11;


struct Coord {
    int x;
    int y;
    Coord(int x, int y) : x(x), y(y) {}
     // Overload operator[] to access x and y like an array
    int& operator[](size_t index) {
        if (index == 0) return x;
        if (index == 1) return y;
        throw std::out_of_range("Index out of range");
    }

    const int& operator[](size_t index) const {
        if (index == 0) return x;
        if (index == 1) return y;
        throw std::out_of_range("Index out of range");
    }

    bool operator==(const Coord& other) const {
        return x == other.x && y == other.y;
    }

};


py::list coords_to_list(std::vector<Coord> coords) {
    py::list coords_list;
    for (auto& coord : coords) {
        coords_list.append(py::make_tuple(coord.x, coord.y));
    }
    return coords_list;
}

class AreaChecker {
public:
    // Constructor to initialize food_value, free_value, and body_value
    AreaChecker(uint8_t food_value, uint8_t free_value, uint8_t body_value, int width, int height) :
        food_value(food_value),
        free_value(free_value),
        body_value(body_value),
        width(width),
        height(height) {}

    bool is_inside(int x, int y){
        return 0 <= x && x < this->width && 0 <= y && y < this->height;
    }

    bool is_single_entrance(py::array_t<uint8_t> s_map, py::tuple coord, py::tuple check_coord){
        return _is_single_entrance(
            s_map,
            Coord(coord[0].cast<int>(), coord[1].cast<int>()),
            Coord(check_coord[0].cast<int>(), check_coord[1].cast<int>())
        );
    }


    bool _is_single_entrance(py::array_t<uint8_t> s_map, Coord coord, Coord check_coord) {
        auto buf = s_map.request();
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        int cols = static_cast<int>(buf.shape[1]);
        int c_x = coord.x;
        int c_y = coord.y;
        int ch_x = check_coord.x;
        int ch_y = check_coord.y;

        int delta_y = ch_y - c_y;
        int delta_x = ch_x - c_x;
        int value;

        std::array<Coord, 8> neighbours = {
            Coord(c_x - 1, c_y - 1),
            Coord(c_x, c_y - 1),
            Coord(c_x + 1, c_y - 1),
            Coord(c_x + 1, c_y),
            Coord(c_x + 1, c_y + 1),
            Coord(c_x, c_y + 1),
            Coord(c_x - 1, c_y + 1),
            Coord(c_x - 1, c_y)
        };
        std::array<unsigned int, 4> corner_values;
        std::array<unsigned int, 4> neighbour_values;

        for(unsigned int i = 0; i < neighbours.size(); i++){
            Coord c = neighbours[i];
            if (i % 2 == 0){
                if (this->is_inside(c.x, c.y)){
                    corner_values[i/2] = ptr[c.y * cols + c.x];
                }
                else{
                    corner_values[i/2] = 3; // arbitrary value, just not free_value or food_value
                }
            }
            else{
                if (this->is_inside(c.x, c.y)){
                    neighbour_values[i/2] = ptr[c.y * cols + c.x];
                }
                else{
                    neighbour_values[i/2] = 3; // arbitrary value, just not free_value or food_value
                }
            }
            
            // std::cout << (ptr[c.y * cols + c.x]) << std::endl;
        }
        // Check the diagonals
        // std::cout << "corner_values" << std::endl;
        // std::cout << corner_values[0] << " " << corner_values[1] << " " << corner_values[2] << " " << corner_values[3] << std::endl;
        // std::cout << neighbour_values[0] << " " << neighbour_values[1] << " " << neighbour_values[2] << " " << neighbour_values[3] << std::endl;
        if (corner_values[0] > this->free_value){
            if (corner_values[2] > this->free_value){
                if (delta_x < 0 || delta_y > 0){
                    if (corner_values[1] <= this->free_value || (neighbour_values[0] <= this->free_value && neighbour_values[1] <= this->free_value)){
                        return true;
                    }
                }
                else{
                    if (corner_values[3] <= this->free_value || (neighbour_values[2] <= this->free_value && neighbour_values[3] <= this->free_value)){
                        return true;
                    }
                }
            }
        }
        if (corner_values[1] > this->free_value ){
            if (corner_values[3] > this->free_value){
                if (delta_x < 0 || delta_y < 0){
                    if (corner_values[2] <= this->free_value || (neighbour_values[1] <= this->free_value && neighbour_values[2] <= this->free_value)){
                        return true;
                    }
                }
                else{
                    if (corner_values[0] <= this->free_value || (neighbour_values[0] <= this->free_value && neighbour_values[3] <= this->free_value)){
                        return true;
                    }
                }
            }
        }

        int x = ch_x + delta_y;
        int y = ch_y + delta_x;
        if (this->is_inside(x, y)) {
            value = ptr[y * cols + x];
            if (value <= this->free_value) {
                x = c_x + delta_y;
                y = c_y + delta_x;
                if (this->is_inside(x, y)) {
                    value = ptr[y * cols + x];
                    if (value <= this->free_value) {
                        return false;
                    }
                }
            }
        }

        x = ch_x - delta_y;
        y = ch_y - delta_x;
        if (this->is_inside(x, y)) {
            value = ptr[y * cols + x];
            if (value <= this->free_value) {
                x = c_x - delta_y;
                y = c_y - delta_x;
                if (this->is_inside(x, y)) {
                    value = ptr[y * cols + x];
                    if (value <= this->free_value) {
                        return false;
                    }
                }
            }
        }


        return true;
    }

    py::dict area_check(
            py::array_t<uint8_t> s_map,
            py::list body_coords_py,
            py::tuple start_coord_py,
            int tile_count = 0,
            int food_count = 0,
            int max_index = 0,
            std::vector<bool> checked = std::vector<bool>(),
            int depth = 0
        ) {
        std::vector<Coord> body_coords;
        for (auto item : body_coords_py) {
            auto coord = item.cast<py::tuple>();
            body_coords.push_back(Coord(coord[0].cast<int>(), coord[1].cast<int>()));
        }
        Coord start_coord = Coord(start_coord_py[0].cast<int>(), start_coord_py[1].cast<int>());
        std::deque<Coord> current_coords;
        std::deque<Coord> to_be_checked;
        current_coords.push_back(start_coord);

        if (checked.size() == 0) {
            checked = std::vector<bool>(height * width, false);
        }
        checked[start_coord.y * width + start_coord.x] = true;
        int body_len = body_coords.size();
        auto tail_coord = body_coords[body_len - 1];
        bool is_clear = false;
        bool has_tail = false;
        bool done = false;
        int total_steps = 0;
        tile_count += 1;

        auto s_map_buf = s_map.request();
        uint8_t* s_map_ptr = static_cast<uint8_t*>(s_map_buf.ptr);
        while (!current_coords.empty()) {
            auto curr_coord = current_coords.front();
            current_coords.pop_front();
            int c_x, c_y;
            c_x = curr_coord.x;
            c_y = curr_coord.y;

            if (s_map_ptr[c_y * width + c_x] == food_value) {
                food_count += 1;
            }

            std::array<Coord, 4> neighbours = {
                Coord(c_x, c_y - 1),
                Coord(c_x + 1, c_y),
                Coord(c_x, c_y + 1),
                Coord(c_x - 1, c_y)
            };

            for (auto& n_coord : neighbours) {
                int n_x, n_y;
                n_x = n_coord.x;
                n_y = n_coord.y;

                if (0 <= n_x && n_x < width && 0 <= n_y && n_y < height) {
                    if (!checked[n_y * width + n_x]) {
                        checked[n_y * width + n_x] = true;
                        int coord_val = s_map_ptr[n_y * width + n_x];
                        if (coord_val == free_value || coord_val == food_value) {
                            if (_is_single_entrance(s_map, curr_coord, Coord(n_x, n_y))) {
                                to_be_checked.push_back(n_coord);
                                continue;
                            }
                            tile_count += 1;
                            current_coords.push_back(n_coord);
                        } else if (coord_val == body_value) {
                            auto it = std::find(body_coords.begin(), body_coords.end(), n_coord);
                            if (it != body_coords.end()) {
                                int body_index = static_cast<int>(std::distance(body_coords.begin(), it));  // Cast to int
                                if (body_index > max_index) {
                                    max_index = body_index;
                                }
                            }
                        }

                        if (n_coord == tail_coord && tile_count != 1) {
                            has_tail = true;
                            is_clear = true;
                            done = true;
                            break;
                        }
                    }
                }
            }

            total_steps = tile_count - food_count;
            int needed_steps = body_len - max_index;
            if (total_steps >= needed_steps) {
                is_clear = true;
            }
            if (done) {
                break;
            }
        }

        if (!is_clear) {
            while (!to_be_checked.empty()) {
                auto coord = to_be_checked.front();
                to_be_checked.pop_front();
                auto area_check = this->area_check(
                    s_map,
                    coords_to_list(body_coords),
                    py::make_tuple(coord.x, coord.y),
                    tile_count,
                    food_count,
                    0,
                    checked,
                    depth + 1);
                if (area_check["is_clear"].cast<bool>()) {
                    return area_check;
                }
            }
        }

        return py::dict(
            py::arg("is_clear") = is_clear,
            py::arg("tile_count") = tile_count,
            py::arg("total_steps") = total_steps,
            py::arg("food_count") = food_count,
            py::arg("has_tail") = has_tail,
            py::arg("max_index") = max_index,
            py::arg("start_coord") = py::make_tuple(start_coord.x, start_coord.y),
            py::arg("needed_steps") = body_len - max_index
        );
    }

private:
    uint8_t food_value;
    uint8_t free_value;
    uint8_t body_value;
    int width;
    int height;

};

PYBIND11_MODULE(area_check, m) {
    py::class_<AreaChecker>(m, "AreaChecker")
        .def(py::init<int, int, int, int, int>())
        .def("is_single_entrance", &AreaChecker::is_single_entrance)
        .def("area_check", &AreaChecker::area_check,
             py::arg("s_map"),
             py::arg("body_coords_py"),
             py::arg("start_coord_py"),
             py::arg("tile_count") = 0,
             py::arg("food_count") = 0,
             py::arg("max_index") = 0,
             py::arg("checked") = std::vector<bool>(),
             py::arg("depth") = 0);

    py::class_<Coord>(m, "Coord")
        .def(py::init<int, int>())
        .def_readwrite("x", &Coord::x)
        .def_readwrite("y", &Coord::y)
        .def("__getitem__", [](const Coord &c, size_t i) {
            if (i >= 2) throw std::out_of_range("Index out of range");
            return c[i];
        })
        .def("__setitem__", [](Coord &c, size_t i, int v) {
            if (i >= 2) throw std::out_of_range("Index out of range");
            c[i] = v;
        });
}
