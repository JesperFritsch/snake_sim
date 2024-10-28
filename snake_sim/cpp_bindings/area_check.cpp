#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // For automatic conversion of C++ STL containers to Python
#include <atomic>
#include <chrono>
#include <deque>
#include <queue>
#include <thread>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream> // Include iostream for std::cout
#include <typeinfo> // Include typeinfo for typeid
#include <unordered_set>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace py = pybind11;

unsigned int cantor_pairing(int k1, int k2) {
    return (k1 + k2) * (k1 + k2 + 1) / 2 + k2;
}

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


struct AreaCheckResult {
    bool is_clear;
    int tile_count;
    int total_steps;
    int food_count;
    bool has_tail;
    int margin;
    int needed_steps;
    bool has_opening;
    float margin_over_tiles;
    AreaCheckResult() :
        is_clear(false),
        tile_count(0),
        total_steps(0),
        food_count(0),
        has_tail(false),
        margin(INT_MIN),
        needed_steps(0),
        has_opening(false),
        margin_over_tiles(0) {}
    AreaCheckResult(bool is_clear,
                    int tile_count,
                    int total_steps,
                    int food_count,
                    bool has_tail,
                    int margin,
                    int needed_steps,
                    bool has_opening,
                    float margin_over_tiles) :
        is_clear(is_clear),
        tile_count(tile_count),
        total_steps(total_steps),
        food_count(food_count),
        has_tail(has_tail),
        margin(margin),
        needed_steps(needed_steps),
        has_opening(has_opening),
        margin_over_tiles(margin_over_tiles) {}
};

class AreaNode{
public:
    Coord start_coord;
    Coord end_coord = Coord();
    int id;
    int max_index = 0;
    int tile_count = 0;
    int food_count = 0;

    // one_dim is true if the area is a line that the snake can not turn around in.
    bool is_one_dim = false;
    bool has_tail = false;
    std::vector<std::pair<AreaNode*, unsigned int>> edge_nodes;

    AreaNode(Coord start_coord, int id) :
        start_coord(start_coord),
        id(id) {
            edge_nodes.reserve(6);
        }

    void remove_connection(AreaNode* node){
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [node](std::pair<AreaNode*, unsigned int>& pair){
            return pair.first == node;
        });
        if (it != edge_nodes.end()){
            edge_nodes.erase(it);
        }
    }

    void remove_connection(unsigned int edge){
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [edge](std::pair<AreaNode*, unsigned int>& pair){
            return pair.second == edge;
        });
        if (it != edge_nodes.end()){
            edge_nodes.erase(it);
        }
    }

    void add_connection(AreaNode* node, unsigned int edge){
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [node](std::pair<AreaNode*, unsigned int>& pair){
            return pair.first == node;
        });
        if (it == edge_nodes.end()){
            edge_nodes.push_back(std::make_pair(node, edge));
        }
    }

    unsigned int get_edge(AreaNode* node){
        auto it = std::find_if(edge_nodes.begin(), edge_nodes.end(), [node](std::pair<AreaNode*, unsigned int>& pair){
            return pair.first == node;
        });
        if (it != edge_nodes.end()){
            return it->second;
        }
        return 0;
    }

    ~AreaNode() = default;
};

struct ExploreData{
    Coord start_coord;
    int area_id;
    AreaNode* prev_node;

    ExploreData() :
        start_coord(Coord()),
        area_id(-1),
        prev_node(nullptr) {}

    ExploreData(Coord start_coord, int area_id, AreaNode* prev_node) :
        start_coord(start_coord),
        area_id(area_id),
        prev_node(prev_node) {}
};

struct ExploreResults{
    int tile_count = 0;
    int food_count = 0;
    int max_index = 0;
    bool has_tail = false;
    std::vector<int> connected_areas;
    std::vector<Coord> to_explore;

    ExploreResults() {
        connected_areas.reserve(10);
        to_explore.reserve(100);
    }

    ExploreResults(
        int tile_count,
        int food_count,
        int max_index,
        bool has_tail,
        std::vector<int> connected_areas,
        std::vector<Coord> to_explore) :
            tile_count(tile_count),
            food_count(food_count),
            max_index(max_index),
            has_tail(has_tail),
            connected_areas(connected_areas),
            to_explore(to_explore) {}
};

struct SearchNode{
    AreaNode* node = nullptr;
    int nr_visits = 0;
    int tiles_until_here = 0;
    int food_until_here = 0;

    std::vector<std::vector<unsigned int>> searched_edges;
    std::vector<unsigned int> used_edges;

    SearchNode() = default;

    SearchNode(AreaNode* node) :
        node(node){}

    bool first_visit(){
        return nr_visits == 1;
    }

    bool is_visited(){
        return nr_visits > 0;
    }

    bool is_used_edge(unsigned int edge){
        return std::find(used_edges.begin(), used_edges.end(), edge) != used_edges.end();
    }

    bool is_searched_edge(unsigned int edge){
        auto searched = searched_edges.back();
        if (std::find(searched.begin(), searched.end(), edge) != searched.end()){
            return true;
        }
        return false;
    }

    void add_searched_edge(unsigned int edge){
        searched_edges.back().push_back(edge);
    }

    void add_used_edge(unsigned int edge){
        used_edges.push_back(edge);
    }

    void enter_from(AreaNode* node, int tiles, int food){
        auto edge = this->node->get_edge(node);
        if (edge != 0){
            add_used_edge(edge);
        }
        nr_visits++;
        if (first_visit()){
            tiles_until_here = tiles;
            food_until_here = food;
        }
        searched_edges.resize(nr_visits);
    }

    void exit(){
        searched_edges.pop_back();
        used_edges.pop_back();
        nr_visits--;
        if (nr_visits == 0){
            reset();
        }
    }

    std::pair<AreaNode*, unsigned int> get_next_node_and_edge(){
        for (auto& edge_node : node->edge_nodes){
            if(!is_used_edge(edge_node.second) && !is_searched_edge(edge_node.second)){
                return edge_node;
            }
        }
        return std::make_pair(nullptr, 0);
    }

    void reset(){
        nr_visits = 0;
        tiles_until_here = 0;
        food_until_here = 0;
        searched_edges.clear();
        used_edges.clear();
    }

    std::pair<int, int> additional_tiles(){
        int tiles;
        int food;
        if (first_visit()){
            tiles = node->tile_count;
            food = node->food_count;
        } else {
            tiles = 0;
            food = 0;
        }
        return std::make_pair(tiles, food);
    }

};

class AreaGraph{
public:
    int next_id = 0;
    int map_width = 0;
    int map_height = 0;
    AreaNode* root = nullptr;
    std::unordered_map<int, std::unique_ptr<AreaNode>> nodes;

    AreaGraph(){
        nodes.reserve(200);
    }

    AreaGraph(int width, int height) :
        map_width(width),
        map_height(height) {
            nodes.reserve(200);
        }

    void connect_nodes(int id1, int id2){
        if (get_node(id1) == nullptr || get_node(id2) == nullptr || id1 == id2){
            return;
        }
        this->connect_nodes(nodes[id1].get(), nodes[id2].get());
    }

    void connect_nodes(AreaNode* node1, AreaNode* node2){
        if(node1 == nullptr || node2 == nullptr || node1->id == node2->id){
            return;
        }
        if (node1->id > node2->id){
            std::swap(node1, node2);
        }
        auto edge_id = cantor_pairing(node1->id, node2->id);
        node1->add_connection(node2, edge_id);
        node2->add_connection(node1, edge_id);
    }

    AreaNode* get_node(int id){
        if (nodes.find(id) == nodes.end()){
            return nullptr;
        }
        return nodes[id].get();
    }

    AreaNode* add_node(AreaNode* edge_node, Coord start_coord){
        return add_node_with_id(edge_node, start_coord, next_id++);
    }

    AreaNode* add_node_with_id(AreaNode* edge_node, Coord start_coord, int id){
        auto new_node = std::make_unique<AreaNode>(start_coord, id);
        auto new_node_ptr = new_node.get();
        if (edge_node != nullptr){
            this->connect_nodes(edge_node, new_node.get());
        }
        nodes[id] = std::move(new_node);
        if (id == 0){
            root = new_node_ptr;
        }
        return new_node_ptr;
    }

    void remove_node(int id){
        auto node = nodes[id].get();
        for (auto& edge_node : node->edge_nodes){
            edge_node.first->remove_connection(node);
        }
        nodes.erase(id);
    }

    AreaCheckResult search_best2(int snake_length, uint8_t* s_map, uint8_t food_value, int width, int target_margin, bool food_check, bool exhaustive, float safe_margin_factor){
        bool forward = true;
        bool skipped_one = false;
        // Map to keep track of visited nodes
        std::unordered_map<AreaNode*, SearchNode> search_nodes_data;
        for (auto& node : nodes){
            search_nodes_data[node.second.get()] = SearchNode(node.second.get());
        }
        // pair(cantor_pairing of from_node - to_node, needed_steps), are the elements to be cached
        std::vector<SearchNode*> search_stack;
        std::vector<int> total_tile_count_stack;
        std::vector<int> total_food_count_stack;
        total_food_count_stack.reserve(100);
        total_tile_count_stack.reserve(100);
        search_stack.reserve(100);
        search_stack.push_back(&search_nodes_data[root]);
        AreaCheckResult best_result;
        AreaNode* current_node = nullptr;
        AreaNode* prev_node = nullptr;

        while(!search_stack.empty()){
            AreaCheckResult current_result;
            SearchNode* step_data = search_stack.back();
            current_node = step_data->node;
            int tiles_before = total_food_count_stack.empty() ? 0 : total_tile_count_stack.back();
            int food_before = total_food_count_stack.empty() ? 0 : total_food_count_stack.back();
            if (forward){
                if (!skipped_one){
                    step_data->enter_from(prev_node, tiles_before, food_before);
                }
                skipped_one = false;
            }
            else{
                step_data->used_edges.pop_back();
            }

            // if this is not the first visit to this node, then countable tiles are 0
            auto current_additional = step_data->additional_tiles();
            int current_countable_tiles = current_additional.first;
            int current_countable_food = current_additional.second;

            int total_tile_count_here = current_countable_tiles + tiles_before;
            int total_food_count_here = current_countable_food + food_before;
            int tiles_here;
            int needed_steps;
            int margin;
            int total_steps;
            int calc_tiles;
            int calc_food;
            if (step_data->node->has_tail && !food_check){
                best_result.has_tail = true;
                best_result.margin = INT_MAX;
                best_result.is_clear = true;
                best_result.tile_count = total_tile_count_here;
                best_result.food_count = total_food_count_here;
                best_result.margin_over_tiles = 1;
                break;
            }

            // this is how loops in the graph are handled
            if (step_data->first_visit()){
                tiles_here = current_node->tile_count;
            }
            else{
                // tiles_until_here and food_until_here are only set at the first visit
                tiles_here = total_tile_count_here - step_data->tiles_until_here;
            }

            if (step_data->node->max_index > 0){
                calc_tiles = total_tile_count_here;
                calc_food = total_food_count_here;
                total_steps = calc_tiles - calc_food;
                needed_steps = snake_length - step_data->node->max_index;
                margin = total_steps - needed_steps;
            }
            else{
                calc_tiles = tiles_here;
                calc_food = total_food_count_here;
                total_steps = calc_tiles - calc_food;
                needed_steps = snake_length + 1;
                margin = total_steps - needed_steps;
            }
            current_result.margin = margin;
            current_result.total_steps = total_steps;
            current_result.tile_count = calc_tiles;
            current_result.food_count = calc_food;
            current_result.needed_steps = needed_steps;
            current_result.margin_over_tiles = (float)margin / (float)calc_tiles;
            if(current_result.margin >= 0){
                current_result.is_clear = true;
            }

            if (food_check){
                if (current_result.margin >= current_result.food_count && (current_result.food_count >= best_result.food_count)){
                    best_result = current_result;
                }
            }
            else{
                if (current_result.margin > best_result.margin){
                    best_result = current_result;
                }
                if ((best_result.margin >= target_margin && best_result.margin >= best_result.food_count) && !exhaustive && current_result.margin_over_tiles >= safe_margin_factor){
                    break;
                }
            }
            // std::cout << "\n####### ENTERING NODE #######" << std::endl;
            // std::cout << (forward ? "--> Forward" : "<-- Backward") << std::endl;
            // std::cout << "nr_visits: " << step_data->nr_visits << std::endl;
            // std::cout << "Current node: " << current_node->id << std::endl;
            // std::cout << "start coord: (" << current_node->start_coord.x << ", " << current_node->start_coord.y << ")" << std::endl;
            // std::cout << "end coord: (" << current_node->end_coord.x << ", " << current_node->end_coord.y << ")" << std::endl;
            // std::cout << "node tile count: " << current_node->tile_count << std::endl;
            // std::cout << "node food count: " << current_node->food_count << std::endl;
            // std::cout << "is one dim: " << current_node->is_one_dim << std::endl;
            // std::cout << "has tail: " << current_node->has_tail << std::endl;
            // std::cout << "Tiles before: " << tiles_before << std::endl;
            // std::cout << "Food before: " << food_before << std::endl;
            // std::cout << "current node tile count: " << current_countable_tiles << std::endl;
            // std::cout << "current node food count: " << current_countable_food << std::endl;
            // std::cout << "tiles until here: " << step_data->tiles_until_here << std::endl;
            // std::cout << "food until here: " << step_data->food_until_here << std::endl;
            // std::cout << "needed steps: " << needed_steps << std::endl;
            // std::cout << "total steps: " << total_steps << std::endl;
            // std::cout << "margin: " << margin << std::endl;
            // std::cout << "searched edges now: ";
            // for(auto edge : step_data->searched_edges.back()){
            //     std::cout << edge << ", ";
            // }
            // std::cout << std::endl;
            // std::cout << "used edges: ";
            // for(auto edge : step_data->used_edges){
            //     std::cout << edge << ", ";
            // }
            // std::cout << std::endl;
            // std::cout << "edge nodes: ";
            // for(auto edge_node : step_data->node->edge_nodes){
            //     std::cout << "(" << edge_node.first->id << ", " << edge_node.second << "), ";
            // }
            // std::cout << "search stack: (";
            // for(auto node : search_stack){
            //     std::cout << node->node->id << ", ";
            // }
            // std::cout << ")" << std::endl;
            // std::cout << std::endl;
            // std::cout << "current result: \n";
            // std::cout << "  is clear: " << current_result.is_clear << std::endl;
            // std::cout << "  tile count: " << current_result.tile_count << std::endl;
            // std::cout << "  food count: " << current_result.food_count << std::endl;
            // std::cout << "  needed steps: " << current_result.needed_steps << std::endl;
            // std::cout << "  has opening: " << current_result.has_opening << std::endl;
            // std::cout << "  margin: " << current_result.margin << std::endl;
            // std::cout << "  total steps: " << current_result.total_steps << std::endl;
            // std::cout << "  has tail: " << current_result.has_tail << std::endl;
            // std::cout << "best result: \n";
            // std::cout << "  is clear: " << best_result.is_clear << std::endl;
            // std::cout << "  tile count: " << best_result.tile_count << std::endl;
            // std::cout << "  food count: " << best_result.food_count << std::endl;
            // std::cout << "  needed steps: " << best_result.needed_steps << std::endl;
            // std::cout << "  has opening: " << best_result.has_opening << std::endl;
            // std::cout << "  margin: " << best_result.margin << std::endl;
            // std::cout << "  total steps: " << best_result.total_steps << std::endl;
            // std::cout << "  has tail: " << best_result.has_tail << std::endl;

            auto node_edge_pair = step_data->get_next_node_and_edge();
            if (node_edge_pair.first != nullptr){
                forward = true;
                auto next_node = node_edge_pair.first;
                auto next_step_data = &search_nodes_data[next_node];
                // this check is important, if we have visited a node with only 1 tile, then we can not visit it again
                if (next_step_data->is_visited()){
                    if(next_step_data->node->tile_count == 1){
                        step_data->add_searched_edge(node_edge_pair.second);
                        skipped_one = true;
                        continue;
                    }
                }
                step_data->add_searched_edge(node_edge_pair.second);
                step_data->add_used_edge(node_edge_pair.second);
                search_stack.push_back(next_step_data);
                // if we visit an area more than once we only want to add the additional tiles and food once,
                // additional_tiles() returns the additional tiles and food for the current node if it is the first visit
                // else 0 for both
                bool is_corner = false;
                bool corner_has_food = false;
                if (search_stack.size() >= 3  && current_node->tile_count > 1){
                    // Check corner
                    int prev_x = search_stack[search_stack.size() - 3]->node->start_coord.x;
                    int prev_y = search_stack[search_stack.size() - 3]->node->start_coord.y;
                    int prev_x2 = search_stack[search_stack.size() - 3]->node->end_coord.x;
                    int prev_y2 = search_stack[search_stack.size() - 3]->node->end_coord.y;
                    int next_x = next_node->start_coord.x;
                    int next_y = next_node->start_coord.y;
                    int delta_x = next_x - prev_x;
                    int delta_y = next_y - prev_y;
                    int delta_x2 = next_x - prev_x2;
                    int delta_y2 = next_y - prev_y2;
                    // Check if path goes through a corner, if so then only extend the tile count stack by 1
                    // if we are currently on node 2
                    // going like this 1 -> 2 -> 3
                    /*
                        x 3 x
                        1 2 2
                        x 2 2
                    */
                    if (std::abs(delta_x) == 1 && std::abs(delta_y) == 1){
                        is_corner = true;
                        int pot_f_x = prev_x + delta_x;
                        int pot_f_y = prev_y + delta_y;
                        if (s_map[prev_y * width + pot_f_x] == food_value || s_map[pot_f_y * width + prev_x] == food_value){
                            corner_has_food = true;
                        }
                    }
                    if (prev_x2 != -1 && prev_y2 != -1 && (std::abs(delta_x2) == 1 && std::abs(delta_y2) == 1)){
                        is_corner = true;
                        int pot_f_x = prev_x2 + delta_x2;
                        int pot_f_y = prev_y2 + delta_y2;
                        if (s_map[prev_y2 * width + pot_f_x] == food_value || s_map[pot_f_y * width + prev_x2] == food_value){
                            corner_has_food = true;
                        }
                    }
                    // else{
                    //     total_tile_count_stack.push_back(total_tile_count_here);
                    //     total_food_count_stack.push_back(total_food_count_here);
                    // }
                }
                else if (current_node->tile_count > 1){
                    int curr_x = current_node->start_coord.x;
                    int curr_y = current_node->start_coord.y;
                    int next_x = next_node->start_coord.x;
                    int next_y = next_node->start_coord.y;
                    int delta_x = next_x - curr_x;
                    int delta_y = next_y - curr_y;
                    if(std::abs(delta_x) == 1 && std::abs(delta_y) == 0 || std::abs(delta_x) == 0 && std::abs(delta_y) == 1){
                        is_corner = true;
                    }
                    if(s_map[curr_y * width + curr_x] == food_value){
                        corner_has_food = true;
                    }

                }
                if (is_corner){
                    total_tile_count_stack.push_back(tiles_before + 1);
                    total_food_count_stack.push_back(food_before + (corner_has_food ? 1 : 0));
                }
                else{
                    total_tile_count_stack.push_back(total_tile_count_here);
                    total_food_count_stack.push_back(total_food_count_here);
                }
            }
            else{
                forward = false;
                search_stack.pop_back();
                total_food_count_stack.pop_back();
                total_tile_count_stack.pop_back();
                if (search_stack.empty()){
                    break;
                }
                step_data->exit();
            }
            prev_node = current_node;
        }
        return best_result;
    }

    ~AreaGraph() = default;
};


// class ThreadPool {
// public:
//     ThreadPool(size_t numThreads) : activeTasks(0), stop(false) {
//         for (size_t i = 0; i < numThreads; ++i) {
//             workers.emplace_back([this] {
//                 while (true) {
//                     std::function<void()> task;
//                     {
//                         std::unique_lock<std::mutex> lock(this->queueMutex);
//                         this->condition.wait(lock, [this] { return !this->tasks.empty() || stop; });
//                         if (this->stop && this->tasks.empty()) return; // If stopping and no tasks, exit
//                         task = std::move(this->tasks.front());
//                         this->tasks.pop();
//                     }

//                     activeTasks++; // Increment active tasks before executing

//                     // Execute the task
//                     task();

//                     {
//                         std::lock_guard<std::mutex> lock(this->completionMutex);
//                         activeTasks--; // Decrement active tasks after completing
//                         if (activeTasks == 0 && tasks.empty()) {
//                             completionCondition.notify_one(); // Notify when all tasks are done
//                         }
//                     }
//                 }
//             });
//         }
//     }

//     template <class F>
//     void enqueue(F&& f) {
//         {
//             std::unique_lock<std::mutex> lock(queueMutex);
//             tasks.emplace(std::forward<F>(f));
//         }
//         condition.notify_one();
//     }

//     // Method to wait until all tasks are completed
//     void waitForAllTasks() {
//         std::unique_lock<std::mutex> lock(completionMutex);
//         completionCondition.wait(lock, [this] { return tasks.empty() && activeTasks == 0; });
//     }

//     ~ThreadPool() {
//         {
//             std::unique_lock<std::mutex> lock(queueMutex);
//             stop = true;
//         }
//         condition.notify_all();
//         for (std::thread& worker : workers) {
//             if (worker.joinable()) {
//                 worker.join();
//             }
//         }
//     }

// private:
//     std::vector<std::thread> workers;
//     std::queue<std::function<void()>> tasks;
//     std::mutex queueMutex;
//     std::condition_variable condition;
//     bool stop;

//     // Track active tasks and completion
//     std::atomic<int> activeTasks;
//     std::mutex completionMutex;
//     std::condition_variable completionCondition;
// };

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
            int target_margin,
            bool food_check,
            bool exhaustive,
            float safe_margin_factor
        ){
            auto s_map_buf = s_map.request();
            uint8_t* s_map_ptr = static_cast<uint8_t*>(s_map_buf.ptr);
            Coord start_coord = Coord(start_coord_py[0].cast<int>(), start_coord_py[1].cast<int>());
            std::vector<Coord> body_coords;
            for (auto item : body_coords_py) {
                auto coord = item.cast<py::tuple>();
                body_coords.push_back(Coord(coord[0].cast<int>(), coord[1].cast<int>()));
            }

            AreaCheckResult result = _area_check2(
                s_map_ptr,
                body_coords,
                start_coord,
                target_margin,
                food_check,
                exhaustive,
                safe_margin_factor
            );
            return py::dict(
                py::arg("is_clear") = result.is_clear,
                py::arg("tile_count") = result.tile_count,
                py::arg("total_steps") = result.total_steps,
                py::arg("food_count") = result.food_count,
                py::arg("has_tail") = result.has_tail,
                py::arg("margin") = result.margin,
                py::arg("margin_over_tiles") = result.margin_over_tiles
            );
        }

    ExploreResults explore_area(
        uint8_t* s_map,
        std::vector<Coord>& body_coords,
        Coord& start_coord,
        int area_id,
        std::vector<int>& checked
    ){
        int tile_count = 0;
        int food_count = 0;
        int max_index = 0;
        bool has_tail = false;
        std::vector<Coord> to_explore;
        std::vector<int> connected_areas;
        to_explore.reserve(10);
        connected_areas.reserve(10);
        int checked_value = checked[start_coord.y * width + start_coord.x];
        if(checked_value != unexplored_area_id && checked_value != area_id){
            return ExploreResults(tile_count, food_count, max_index, has_tail, connected_areas, to_explore);
        }
        tile_count += 1;
        size_t body_len = body_coords.size();
        auto tail_coord = body_coords[body_len - 1];
        std::deque<Coord> current_coords;
        current_coords.push_back(start_coord);
        checked[start_coord.y * width + start_coord.x] = area_id;
        while (!current_coords.empty()) {
            auto curr_coord = current_coords.front();
            current_coords.pop_front();
            int c_x, c_y;
            c_x = curr_coord.x;
            c_y = curr_coord.y;
            if (s_map[c_y * width + c_x] == food_value) {
                // food_coords.insert(curr_coord);
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
                if (checked_val != unexplored_area_id) {
                    if (checked_val != area_id) {
                        if (std::find(connected_areas.begin(), connected_areas.end(), checked_val) == connected_areas.end()) {
                            connected_areas.push_back(checked_val);
                        }
                    }
                    continue;
                }
                int coord_val = s_map[n_y * width + n_x];
                if (coord_val == free_value || coord_val == food_value) {
                    int entrance_code = _is_single_entrance(s_map, curr_coord, n_coord);
                    if (entrance_code == 0) {
                        checked[n_y * width + n_x] = area_id; // this used to be above this if statement, dont know if this will cause a bug, but i think it should be fine.
                        tile_count += 1;
                        current_coords.push_back(n_coord);
                    }
                    else{
                        if (std::find(to_explore.begin(), to_explore.end(), n_coord) == to_explore.end()) {
                            to_explore.push_back(n_coord);
                        }
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
                    if (n_coord == tail_coord && !(curr_coord == start_coord && area_id == 0)) {
                        max_index = body_len - 1;
                        has_tail = true;
                    }
                }
            }
        }
        return ExploreResults(tile_count, food_count, max_index, has_tail, connected_areas, to_explore);
    }

    AreaCheckResult _area_check2(
        uint8_t* s_map,
        std::vector<Coord>& body_coords,
        Coord& start_coord,
        int target_margin,
        bool food_check,
        bool exhaustive,
        float safe_margin_factor
    ){
        std::vector<int> checked;
        checked.resize(height * width);
        std::fill(checked.begin(), checked.end(), unexplored_area_id);

        AreaNode* prev_node = nullptr;
        AreaNode* current_node;
        AreaGraph graph = AreaGraph();
        std::vector<ExploreData> areas_to_explore;
        areas_to_explore.reserve(300);
        areas_to_explore.push_back(ExploreData(start_coord, graph.next_id++, prev_node));
        // in the while loop we will explore the area and add new nodes to the graph
        while(!areas_to_explore.empty()){
            auto current_area_data = areas_to_explore.back();
            areas_to_explore.pop_back();
            Coord current_coord = current_area_data.start_coord;
            int current_id = current_area_data.area_id;
            prev_node = current_area_data.prev_node;
            // std::cout << "Exploring area start: (" << current_coord.x << ", " << current_coord.y << ") " << current_id << std::endl;
            auto result = explore_area(
                s_map,
                body_coords,
                current_coord,
                current_id,
                checked
            );
            // std::cout << "Explored area: " << current_id << std::endl;
            // std::cout << "Current coord: (" << current_coord.x << ", " << current_coord.y << ")" << std::endl;
            // std::cout << "Prev node: " << (prev_node == nullptr ? -1 : prev_node->id) << std::endl;
            // std::cout << "Tile count: " << result.tile_count << std::endl;
            // std::cout << "Food count: " << result.food_count << std::endl;
            // std::cout << "Max index: " << result.max_index << std::endl;
            // std::cout << "Has tail: " << result.has_tail << std::endl;
            // std::cout << "prev is one dim: " << (prev_node == nullptr ? false : prev_node->is_one_dim) << std::endl;
            // for (auto& connected_area : result.connected_areas){
            //     std::cout << "Connected area: " << connected_area << std::endl;
            // }
            // for (auto& coord : result.to_explore){
            //     std::cout << "To explore: (" << coord.x << ", " << coord.y << ")" << std::endl;
            // }
            if (result.tile_count == 0){
                continue;
            }
            // If an area has just one tile, no max index and only one area to explore, then we can just add the tile to the previous node
            if (prev_node != nullptr && prev_node->is_one_dim && result.tile_count == 1 && result.max_index == 0 && (result.connected_areas.size() + result.to_explore.size() == 2)){
                // std::cout << "Adding to previous node" << std::endl;
                current_node = prev_node;
                current_node->tile_count += result.tile_count;
                current_node->food_count += result.food_count;
                current_node->max_index = result.max_index;
                current_node->has_tail = result.has_tail;
                current_node->end_coord = current_coord;
            }
            else{
                // std::cout << "Adding node to graph" << std::endl;
                current_node = graph.add_node_with_id(prev_node, current_coord, current_id);
                current_node->tile_count = result.tile_count;
                current_node->food_count = result.food_count;
                current_node->max_index = result.max_index;
                current_node->has_tail = result.has_tail;
                if (current_node->tile_count == 1 && (result.connected_areas.size() + result.to_explore.size() == 2) && current_node->max_index == 0){ // a node cant really have 2 or 3 tiles, next step after 1 is 4, but anyways...
                    current_node->is_one_dim = true;
                }
            }
            for (auto connected_area : result.connected_areas){
                if (graph.get_node(connected_area) != nullptr){
                    graph.connect_nodes(current_node->id, connected_area);
                }
            }
            // std::cout << "Made connections" << std::endl;
            for (auto& area_start_coord : result.to_explore){
                areas_to_explore.push_back(ExploreData(area_start_coord, graph.next_id++, current_node));
            }
        }
        // std::cout << "Graph size: " << graph.nodes.size() << std::endl;
        return graph.search_best2(body_coords.size(), s_map, food_value, width, target_margin, food_check, exhaustive, safe_margin_factor);
    }

private:
    uint8_t food_value;
    uint8_t free_value;
    uint8_t body_value;
    int width;
    int height;
    Coord print_mark;
    int unexplored_area_id = -1;
    // ThreadPool thread_pool = ThreadPool(std::thread::hardware_concurrency());
};

PYBIND11_MODULE(area_check, m) {
    py::class_<AreaChecker>(m, "AreaChecker")
        .def(py::init<int, int, int, int, int>())
        .def("is_single_entrance", &AreaChecker::is_single_entrance)
        .def("area_check", &AreaChecker::area_check,
            py::arg("s_map"),
            py::arg("body_coords_py"),
            py::arg("start_coord_py"),
            py::arg("target_margin"),
            py::arg("food_check"),
            py::arg("exhaustive"),
            py::arg("safe_margin_factor"));

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
