#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // For automatic conversion of C++ STL containers to Python
#include <atomic>
#include <chrono>
#include <deque>
#include <queue>
#include <thread>
#include <vector>
#include <algorithm>
#include <iostream> // Include iostream for std::cout
#include <typeinfo> // Include typeinfo for typeid
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace py = pybind11;

struct Coord {
    int x;
    int y;

    Coord() : x(-1), y(-1) {}

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

    bool operator!=(const Coord& other) const {
        return x != other.x || y != other.y;
    }

    std::size_t hash() const {
        return x * 1000 + y;
    }

};

namespace std {
    template <>
    struct hash<Coord> {
        std::size_t operator()(const Coord& coord) const noexcept{
            return coord.hash();
        }
    };
}


#include <atomic>

class ThreadPool {
public:
    ThreadPool(size_t numThreads) : activeTasks(0), stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] { return !this->tasks.empty() || stop; });
                        if (this->stop && this->tasks.empty()) return; // If stopping and no tasks, exit
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    activeTasks++; // Increment active tasks before executing

                    // Execute the task
                    task();

                    {
                        std::lock_guard<std::mutex> lock(this->completionMutex);
                        activeTasks--; // Decrement active tasks after completing
                        if (activeTasks == 0 && tasks.empty()) {
                            completionCondition.notify_one(); // Notify when all tasks are done
                        }
                    }
                }
            });
        }
    }

    template <class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    // Method to wait until all tasks are completed
    void waitForAllTasks() {
        std::unique_lock<std::mutex> lock(completionMutex);
        completionCondition.wait(lock, [this] { return tasks.empty() && activeTasks == 0; });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;

    // Track active tasks and completion
    std::atomic<int> activeTasks;
    std::mutex completionMutex;
    std::condition_variable completionCondition;
};



struct AreaCheckResult {
    bool initialized;
    bool is_clear;
    int prev_tile_count;
    int prev_food_count;
    int tile_count;
    int total_steps;
    int food_count;
    bool has_tail;
    int max_index;
    Coord start_coord;
    int needed_steps;
    int margin;
    std::unordered_set<Coord> food_coords;
    // std::vector<int> connected_areas;
    AreaCheckResult() :
        initialized(false),
        is_clear(false),
        prev_tile_count(0),
        prev_food_count(0),
        tile_count(0),
        total_steps(0),
        food_count(0),
        has_tail(false),
        max_index(0),
        start_coord(Coord()),
        needed_steps(0),
        margin(0) {}
    AreaCheckResult(bool is_clear,
                    int prev_tile_count,
                    int prev_food_count,
                    int tile_count,
                    int total_steps,
                    int food_count,
                    bool has_tail,
                    int max_index,
                    Coord start_coord,
                    int needed_steps,
                    int margin,
                    std::unordered_set<Coord> food_coords
                    /*std::vector<int> connected_areas*/) :
        is_clear(is_clear),
        prev_tile_count(prev_tile_count),
        prev_food_count(prev_food_count),
        tile_count(tile_count),
        total_steps(total_steps),
        food_count(food_count),
        has_tail(has_tail),
        max_index(max_index),
        start_coord(start_coord),
        needed_steps(needed_steps),
        margin(margin),
        food_coords(food_coords),
        initialized(true)
        /*connected_areas(connected_areas)*/ {}
};

struct AreaStat{
    int max_index;
    int tile_count;
    int food_count;


    AreaStat() :
        max_index(0),
        tile_count(0),
        food_count(0) {}

    AreaStat(
        int max_index,
        int tile_count,
        int food_count) :
            max_index(max_index),
            tile_count(tile_count),
            food_count(food_count) {}
};


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
        return !(x < 0 || y < 0 || x >= this->width || y >= this->height);
    }

    void print_map(uint8_t* s_map) {
    int rows = this->height;
    int cols = this->width;
    char c;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (s_map[i * cols + j] == 1){
                c = '.';
            }
            else if(s_map[i * cols + j] == 0){
                c = 'F';
            }
            else if (s_map[i * cols + j] == 2){
                c = '#';
            }
            else{
                c = (char)s_map[i * cols + j];
            }
            if (Coord(j, i) == this->print_mark){
                c = '+';
            }
            std::cout << " " << c << " ";
        }
        std::cout << std::endl;
    }
    this->print_mark = Coord();
}

    bool is_single_entrance(py::array_t<uint8_t> s_map, py::tuple coord, py::tuple check_coord){
        auto buf = s_map.request();
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        return _is_single_entrance(
            ptr,
            Coord(coord[0].cast<int>(), coord[1].cast<int>()),
            Coord(check_coord[0].cast<int>(), check_coord[1].cast<int>())
        );
    }

    // int _is_single_entrance(uint8_t* s_map, Coord coord, Coord check_coord) {
    //     int cols = this->width;
    //     int c_x = coord.x;
    //     int c_y = coord.y;
    //     int ch_x = check_coord.x;
    //     int ch_y = check_coord.y;
    //     uint8_t offmap_value = 3; // Arbitrary value for off-map

    //     int delta_y = ch_y - c_y;
    //     int delta_x = ch_x - c_x;

    //     // Pre-calculate indices and use references for speed
    //     auto get_value = [&](int x, int y) -> uint8_t {
    //         if (this->is_inside(x, y)) {
    //             return s_map[y * cols + x];
    //         }
    //         return offmap_value; // Return offmap_value if out of bounds
    //     };

    //     // Get values of corner and neighbor cells
    //     std::array<uint8_t, 4> corner_values = {
    //         get_value(c_x - 1, c_y - 1),
    //         get_value(c_x + 1, c_y - 1),
    //         get_value(c_x + 1, c_y + 1),
    //         get_value(c_x - 1, c_y + 1)
    //     };

    //     std::array<uint8_t, 4> neighbour_values = {
    //         get_value(c_x, c_y - 1),
    //         get_value(c_x + 1, c_y),
    //         get_value(c_x, c_y + 1),
    //         get_value(c_x - 1, c_y)
    //     };

    //     // Diagonal check optimization
    //     auto check_diagonal = [&](int idx1, int idx2, int corner1, int corner2) -> bool {
    //         if (corner_values[corner1] > this->free_value && corner_values[corner2] > this->free_value) {
    //             for (int i = 0; i < 4; ++i) {
    //                 if ((i != idx1 && i != idx2) && corner_values[i] > this->free_value) {
    //                     return false; // Break if other corners are non-walkable
    //                 }
    //                 if (neighbour_values[i] > this->free_value) {
    //                     return false;
    //                 }
    //             }
    //             return true; // It's diagonal
    //         }
    //         return false;
    //     };

    //     // Check for diagonal patterns
    //     if (check_diagonal(0, 2, 0, 2)) return 2;
    //     if (check_diagonal(1, 3, 1, 3)) return 2;

    //     // Check vertical and horizontal paths using delta_x and delta_y
    //     int x = ch_x + delta_y;
    //     int y = ch_y + delta_x;
    //     if (this->is_inside(x, y) && get_value(x, y) <= this->free_value) {
    //         x = c_x + delta_y;
    //         y = c_y + delta_x;
    //         if (this->is_inside(x, y) && get_value(x, y) <= this->free_value) {
    //             return 0;
    //         }
    //     }

    //     x = ch_x - delta_y;
    //     y = ch_y - delta_x;
    //     if (this->is_inside(x, y) && get_value(x, y) <= this->free_value) {
    //         x = c_x - delta_y;
    //         y = c_y - delta_x;
    //         if (this->is_inside(x, y) && get_value(x, y) <= this->free_value) {
    //             return 0;
    //         }
    //     }

    //     return 1; // No single entrance found
    // }

    int _is_single_entrance(uint8_t* s_map, Coord coord, Coord check_coord) {
        // return code 2 is for a passage like:
        // x . .
        // . . .
        // . . x
        // or flipped
        int cols = this->width;
        int c_x = coord.x;
        int c_y = coord.y;
        int ch_x = check_coord.x;
        int ch_y = check_coord.y;
        uint8_t offmap_value = 3;

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
        std::array<uint8_t, 4> corner_values;
        std::array<uint8_t, 4> neighbour_values;

        for(unsigned int i = 0; i < neighbours.size(); i++){
            Coord c = neighbours[i];
            if (i % 2 == 0){
                if (this->is_inside(c.x, c.y)){
                    corner_values[i/2] = s_map[c.y * cols + c.x];
                }
                else{
                    corner_values[i/2] = offmap_value; // arbitrary value, just not free_value or food_value
                }
            }
            else{
                if (this->is_inside(c.x, c.y)){
                    neighbour_values[i/2] = s_map[c.y * cols + c.x];
                }
                else{
                    neighbour_values[i/2] = offmap_value; // arbitrary value, just not free_value or food_value
                }
            }

        }
        if (corner_values[0] > this->free_value && corner_values[0] != offmap_value){
            if (corner_values[2] > this->free_value && corner_values[2] != offmap_value){
                bool is_diagonal = true;
                for(unsigned int i = 0; i < neighbour_values.size(); i++){
                    if (neighbour_values[i] > this->free_value){
                        is_diagonal = false;
                        break;
                    }
                    if (i != 0 && i != 2 && corner_values[i] > this->free_value){
                        is_diagonal = false;
                        break;
                    }
                }
                if (is_diagonal){
                    return 2;
                }
            }
        }
        if (corner_values[1] > this->free_value && corner_values[1] != offmap_value){
            if (corner_values[3] > this->free_value && corner_values[3] != offmap_value){
                bool is_diagonal = true;
                for(unsigned int i = 0; i < neighbour_values.size(); i++){
                    if (neighbour_values[i] > this->free_value){
                        is_diagonal = false;
                        break;
                    }
                    if (i != 1 && i != 3 && corner_values[i] > this->free_value){
                        is_diagonal = false;
                        break;
                    }
                }
                if (is_diagonal){
                    return 2;
                }
            }
        }

        int x = ch_x + delta_y;
        int y = ch_y + delta_x;
        if (this->is_inside(x, y)) {
            value = s_map[y * cols + x];
            if (value <= this->free_value) {
                x = c_x + delta_y;
                y = c_y + delta_x;
                if (this->is_inside(x, y)) {
                    value = s_map[y * cols + x];
                    if (value <= this->free_value) {
                        return 0;
                    }
                }
            }
        }

        x = ch_x - delta_y;
        y = ch_y - delta_x;
        if (this->is_inside(x, y)) {
            value = s_map[y * cols + x];
            if (value <= this->free_value) {
                x = c_x - delta_y;
                y = c_y - delta_x;
                if (this->is_inside(x, y)) {
                    value = s_map[y * cols + x];
                    if (value <= this->free_value) {
                        return 0;
                    }
                }
            }
        }


        return 1;
    }

    py::dict area_check(
            py::array_t<uint8_t> s_map,
            py::list body_coords_py,
            py::tuple start_coord_py,
            bool food_check
        ){
            auto s_map_buf = s_map.request();
            uint8_t* s_map_ptr = static_cast<uint8_t*>(s_map_buf.ptr);
            Coord start_coord = Coord(start_coord_py[0].cast<int>(), start_coord_py[1].cast<int>());
            std::vector<Coord> body_coords;
            for (auto item : body_coords_py) {
                auto coord = item.cast<py::tuple>();
                body_coords.push_back(Coord(coord[0].cast<int>(), coord[1].cast<int>()));
            }
            std::unordered_map<int, AreaStat> area_stats;
            std::vector<int> checked;
            AreaCheckResult result = _area_check(
                s_map_ptr,
                body_coords,
                start_coord,
                0,
                0,
                0,
                checked,
                0,
                area_stats,
                food_check
            );
            py::list food_coords;
            for (auto& food_coord : result.food_coords) {
                food_coords.append(py::make_tuple(food_coord.x, food_coord.y));
            }
            return py::dict(
                py::arg("is_clear") = result.is_clear,
                py::arg("tile_count") = result.tile_count,
                py::arg("total_steps") = result.total_steps,
                py::arg("food_count") = result.food_count,
                py::arg("has_tail") = result.has_tail,
                py::arg("max_index") = result.max_index,
                py::arg("start_coord") = py::make_tuple(result.start_coord.x, result.start_coord.y),
                py::arg("needed_steps") = result.needed_steps,
                py::arg("margin") = result.margin,
                py::arg("food_coords") = food_coords
            );
        }

    AreaCheckResult _area_check(
            uint8_t* s_map,
            std::vector<Coord>& body_coords,
            Coord& start_coord,
            int prev_tile_count,
            int prev_food_count,
            int prev_margin,
            std::vector<int> checked,
            int depth,
            std::unordered_map<int, AreaStat>& area_stats,
            bool food_check
        ) {
        int best_tile_count = 0;
        int best_food_count = 0;
        int best_max_index = 0;
        int best_total_steps = 0;
        int tile_count = 1;
        int food_count = 0;
        int max_index = 0;
        bool connected_to_prev_area = false;
        size_t body_len = body_coords.size();
        auto tail_coord = body_coords[body_len - 1];
        bool is_clear = false;
        bool has_tail = false;
        int total_steps = 0;
        int margin;
        if (depth == 0){
            margin = -static_cast<int>(body_coords.size());
        }
        std::unordered_set<Coord> food_coords;
        std::vector<Coord> current_coords;
        std::vector<Coord> to_be_checked;
        // std::vector<int> connected_areas;
        // std::vector<AreaCheckResult> sub_checks;
        current_coords.reserve((this->width + this->height) * 3); // arbitrary value for reserve size
        current_coords.push_back(start_coord);


        if (checked.size() == 0) {
            checked.resize(height * width);
            std::fill(checked.begin(), checked.end(), -1);
        }
        checked[start_coord.y * width + start_coord.x] = depth;
        while (!current_coords.empty()) {
            auto curr_coord = current_coords.back();
            current_coords.pop_back();
            int c_x, c_y;
            c_x = curr_coord.x;
            c_y = curr_coord.y;
            if (s_map[c_y * width + c_x] == food_value) {
                food_coords.insert(curr_coord);
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
                if (!this->is_inside(n_x, n_y)) {
                    continue;
                }
                int checked_val = checked[n_y * width + n_x];
                if (checked_val >= 0) {
                    if (checked_val + 1 < depth) { // checked_val + 1 because we dont want to consider the area we are coming from.
                        max_index = std::max(max_index, area_stats[checked_val].max_index);
                        // connected_areas.push_back(checked_val);
                        connected_to_prev_area = true;
                    }
                    continue;
                }
                int coord_val = s_map[n_y * width + n_x];
                if (coord_val == free_value || coord_val == food_value) {
                    int entrance_code = _is_single_entrance(s_map, curr_coord, n_coord);
                    if (entrance_code == 0) {
                        checked[n_y * width + n_x] = depth; // this used to be above this if statement, dont know if this will cause a bug, but i think it should be fine.
                        tile_count += 1;
                        current_coords.push_back(n_coord);
                    }
                    else{
                        to_be_checked.push_back(n_coord);
                        if (entrance_code == 2) {
                            break;
                        }
                    }
                } else if (coord_val == body_value) {
                    auto it = std::find(body_coords.begin(), body_coords.end(), n_coord);
                    if (it != body_coords.end()) {
                        int body_index = static_cast<int>(std::distance(body_coords.begin(), it));  // Cast to int
                        if (body_index > max_index) {
                            max_index = body_index;
                        }
                    }
                    if (n_coord == tail_coord && tile_count != 1) {
                        has_tail = true;
                        is_clear = true;
                    }
                }
            }
        }
        int needed_steps = body_len - max_index;
        if (max_index > 0 || connected_to_prev_area) {
            total_steps = (tile_count + prev_tile_count) - (food_count + prev_food_count);
        }
        else{
            total_steps = (tile_count) - (food_count);
        }
        margin = total_steps - needed_steps;
        if (margin >= 0) {
            is_clear = true;
        }
        best_tile_count = tile_count + prev_tile_count;
        best_food_count = food_count + prev_food_count;
        best_max_index = max_index;
        best_total_steps = best_tile_count - best_food_count;

        // std::cout << "called with args:" << "tile_count: " << tile_count << " food_count: " << food_count << " max_index: " << max_index << " depth: " << depth << " start_coord: (" << start_coord.x << ", " << start_coord.y << ")" << std::endl;
        // std::cout << "  " << "prev_tile_count: " << prev_tile_count << std::endl;
        // std::cout << "  " << "prev_food_count: " << prev_food_count << std::endl;
        // std::cout << "  " << "prev_margin: " << prev_margin << std::endl;
        // std::cout << "  " << "depth: " << depth << std::endl;
        // std::cout << "  " << "margin: " << margin << std::endl;
        // std::cout << "  " << "food_count: " << food_count << std::endl;
        // std::cout << "  " << "total_steps: " << total_steps << std::endl;
        // std::cout << "  " << "tile_count: " << tile_count << std::endl;
        // std::cout << "  " << "max_index: " << max_index << std::endl;
        // std::cout << "  " << "start_coord: (" << start_coord.x << ", " << start_coord.y << ")" << std::endl;
        // std::cout << "  " << "is_clear: " << is_clear << std::endl;
        // std::cout << "  " << std::endl;
        // this->print_mark = start_coord;
        // this->print_map(s_map);


        int best_margin = margin;
        if (!is_clear || best_margin < (best_food_count) || food_check) {
            int base_tile_count;
            int base_food_count;
            AreaCheckResult best_sub_area;
            while (!to_be_checked.empty()) {
                auto coord = to_be_checked.back();
                to_be_checked.pop_back();
                area_stats[depth] = AreaStat(max_index, tile_count, food_count);
                int x_dist = std::abs(coord.x - start_coord.x);
                int y_dist = std::abs(coord.y - start_coord.y);
                // std::cout << "  " << "coord: (" << coord.x << ", " << coord.y << ")" << std::endl;
                // std::cout << "  " << "start_coord: (" << start_coord.x << ", " << start_coord.y << ")" << std::endl;
                // std::cout << "  " << "x_dist: " << x_dist << std::endl;
                // std::cout << "  " << "y_dist: " << y_dist << std::endl;

                if (
                    (x_dist == 1 && y_dist == 0) ||
                    (x_dist == 0 && y_dist == 1)) {
                    base_tile_count = prev_tile_count + 1;
                    if (s_map[start_coord.y * width + start_coord.x] == food_value) {
                        base_food_count = prev_food_count + 1;
                    }
                    else {
                        base_food_count = prev_food_count;
                    }
                } else {
                    base_tile_count = tile_count + prev_tile_count;
                    base_food_count = food_count + prev_food_count;
                }
                AreaCheckResult area_check = this->_area_check(
                    s_map,
                    body_coords,
                    coord,
                    base_tile_count,
                    base_food_count,
                    margin,
                    checked,
                    depth + 1,
                    area_stats,
                    food_check);
                // std::cout << "  " << "best_tile_count: " << best_tile_count << std::endl;
                // std::cout << "  " << "best_food_count: " << best_food_count << std::endl;
                // std::cout << "  " << "depth: " << depth << std::endl;
                // std::cout << "  " << "start_coord: (" << start_coord.x << ", " << start_coord.y << ")" << std::endl;
                // std::cout << "  " << "coord: (" << coord.x << ", " << coord.y << ")" << std::endl;
                // std::cout << "  " << "is_clear: " << area_check.is_clear << std::endl;
                // std::cout << "  " << "tile_count: " << area_check.tile_count << std::endl;
                // std::cout << "  " << "total_steps: " << area_check.total_steps << std::endl;
                // std::cout << "  " << "food_count: " << area_check.food_count << std::endl;
                // std::cout << "  " << "has_tail: " << area_check.has_tail << std::endl;
                // std::cout << "  " << "max_index: " << area_check.max_index << std::endl;
                // std::cout << "  " << "needed_steps: " << area_check.needed_steps << std::endl;
                // std::cout << "  " << "margin: " << area_check.margin << std::endl;
                // std::cout << "  " << std::endl;
                // if (area_check.connected_areas.size() > 0) {
                //     if (std::find(area_check.connected_areas.begin(), area_check.connected_areas.end(), depth) != area_check.connected_areas.end()) {
                //         best_margin = area_check.margin;
                //         best_total_steps = area_check.total_steps;
                //         best_tile_count = area_check.tile_count;
                //         best_food_count = area_check.food_count;
                //     }
                //     for (auto& connected_area : area_check.connected_areas) {
                //         connected_areas.push_back(connected_area);
                //     }
                // }
                // else{
                //     sub_checks.push_back(area_check);
                // }
                has_tail = has_tail || area_check.has_tail;
                is_clear = is_clear || area_check.is_clear;
                if(food_check){
                    // if (area_check.is_clear) {
                        // for (auto& food_coord : area_check.food_coords) {
                        //     food_coords.insert(food_coord);
                        // }
                        // unsigned int food_count = food_coords.size();
                        // best_food_count = std::max(best_food_count, static_cast<int>(food_count));
                    // }
                    if(area_check.is_clear && area_check.margin >= best_margin && area_check.food_count >= best_food_count) {
                        // std::cout << "  " << "setting best values" << std::endl;
                        best_sub_area = area_check;
                        best_margin = area_check.margin;
                        best_food_count = area_check.food_count;
                    }
                }
                else{
                    if (area_check.margin >= area_check.food_count && area_check.is_clear) {
                        if (food_coords.size() > 0){
                            for (auto& food_coord : food_coords) {
                                area_check.food_coords.insert(food_coord);
                            }
                            area_check.food_count = static_cast<int>(area_check.food_coords.size());
                        }
                        return area_check;
                    }
                    if (area_check.margin >= best_margin) {
                        best_margin = area_check.margin;
                        best_sub_area = area_check;
                        // best_total_steps = area_check.total_steps;
                        // best_tile_count = area_check.tile_count;
                        // best_max_index = area_check.max_index;
                        // for (auto& food_coord : area_check.food_coords) {
                        //     food_coords.insert(food_coord);
                        // }
                        // unsigned int food_count = food_coords.size();
                        // best_food_count = std::max(best_food_count, static_cast<int>(food_count));
                    }
                }
            }
            if (best_sub_area.initialized){
                best_margin = best_sub_area.margin;
                best_total_steps = best_sub_area.total_steps;
                best_tile_count = best_sub_area.tile_count;
                best_max_index = best_sub_area.max_index;
                for (auto& food_coord : best_sub_area.food_coords) {
                    food_coords.insert(food_coord);
                }
                size_t food_count = food_coords.size();
                best_food_count = std::max(best_food_count, static_cast<int>(food_count));
            }
            // for (auto& sub_check : sub_checks) {
            //     int needed = body_len - sub_check.max_index;
            //     total_steps = best_total_steps + (sub_check.total_steps - (sub_check.prev_tile_count - sub_check.prev_food_count));
            //     int calc_margin = total_steps - needed;
            //     if (calc_margin > best_margin) {
            //         best_margin = calc_margin;
            //         best_total_steps = total_steps;
            //         best_tile_count = best_tile_count + (sub_check.tile_count - sub_check.prev_tile_count);
            //         best_food_count = best_food_count + (sub_check.food_count - sub_check.prev_food_count);
            //         best_max_index = sub_check.max_index;
            //         is_clear = true;
            //     }
            // }
        }

        return AreaCheckResult(
            is_clear,
            prev_tile_count,
            prev_food_count,
            best_tile_count,
            best_total_steps,
            best_food_count,
            has_tail,
            best_max_index,
            start_coord,
            body_len - best_max_index,
            best_margin,
            food_coords
            // connected_areas
        );
    }

private:
    uint8_t food_value;
    uint8_t free_value;
    uint8_t body_value;
    int width;
    int height;
    Coord print_mark;
    ThreadPool thread_pool = ThreadPool(std::thread::hardware_concurrency());
};

PYBIND11_MODULE(area_check, m) {
    py::class_<AreaChecker>(m, "AreaChecker")
        .def(py::init<int, int, int, int, int>())
        .def("is_single_entrance", &AreaChecker::is_single_entrance)
        .def("area_check", &AreaChecker::area_check,
             py::arg("s_map"),
             py::arg("body_coords_py"),
             py::arg("start_coord_py"),
             py::arg("food_check"));

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
