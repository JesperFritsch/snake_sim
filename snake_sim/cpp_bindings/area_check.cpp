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
    std::vector<int> connected_areas;
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
        margin(INT_MIN) {}
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
                    std::unordered_set<Coord> food_coords,
                    std::vector<int> connected_areas) :
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
        initialized(true),
        connected_areas(connected_areas) {}
};

class AreaNode{
public:
    Coord start_coord;
    int id;
    int max_index = 0;
    int tile_count = 0;
    int food_count = 0;

    // one_dim is true if the area is a line that the snake can not turn around in.
    bool is_one_dim = false;
    bool has_tail = false;
    bool explored = false;
    std::unordered_set<AreaNode*> edge_nodes;

    AreaNode(Coord start_coord, int id) :
        start_coord(start_coord),
        id(id) {}

    ~AreaNode() = default;
};

struct ExploreResults{
    int tile_count = 0;
    int food_count = 0;
    int max_index = 0;
    bool has_tail = false;
    std::unordered_set<int> connected_areas;
    std::unordered_set<Coord> to_explore;

    ExploreResults() {}

    ExploreResults(
        int tile_count,
        int food_count,
        int max_index,
        bool has_tail,
        std::unordered_set<int> connected_areas,
        std::unordered_set<Coord> to_explore) :
            tile_count(tile_count),
            food_count(food_count),
            max_index(max_index),
            has_tail(has_tail),
            connected_areas(connected_areas),
            to_explore(to_explore) {}
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


struct LoopBackData{
    int tile_count;
    int food_count;
    int steps = 0;
    std::vector<AreaNode*> nodes;

    LoopBackData() :
        tile_count(0),
        food_count(0) {}

    LoopBackData(int tile_count, int food_count) :
        tile_count(tile_count),
        food_count(food_count) {}

    void add_node(AreaNode* node){
        nodes.push_back(node);
    }

    bool can_combine(LoopBackData* other){
        bool res = true;
        for(auto node : other->nodes){
            if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()){
                if (node->is_one_dim){
                    res = false;
                    break;
                }
            }
        }
        return res;
    }
};


struct SearchStepData{
    AreaNode* node = nullptr;
    std::vector<AreaNode*> to_explore_nodes;
    std::vector<AreaNode*> dead_end_nodes;

    // node*, needed_steps
    std::vector<std::pair<AreaNode*, int>> nodes_with_openings;
    unsigned int food_to_best_opening = 0;
    unsigned int tiles_to_best_opening = 0;
    bool visited = false;
    int needed_steps;
    bool has_opening;

    // list of pairs (AreaNode*, unsigned int) where the unsigned int is the discounted tile count untill this node is reached again.
    std::vector<std::unique_ptr<LoopBackData>> loop_backs;

    SearchStepData() = default;

    SearchStepData(AreaNode* node) :
        needed_steps(0),
        node(node){
            for(auto& edge_node : node->edge_nodes){
                to_explore_nodes.push_back(edge_node);
            }
            has_opening = node->max_index > 0;
        }
};


class AreaGraph{
public:
    int next_id = 0;
    int snake_length = 0;
    AreaNode* root = nullptr;
    std::unordered_map<int, std::unique_ptr<AreaNode>> nodes;
    std::vector<int> new_nodes;

    AreaGraph(){}

    AreaGraph(Coord start_coord, int snake_len){
        snake_length = snake_len;
        auto root_node = std::make_unique<AreaNode>(start_coord, next_id);
        root = root_node.get();
        nodes[next_id] = std::move(root_node);
        new_nodes.push_back(next_id);
        next_id++;
    }

    void connect_nodes(int id1, int id2){
        this->connect_nodes(nodes[id1].get(), nodes[id2].get());
    }

    void connect_nodes(AreaNode* node1, AreaNode* node2){
        node1->edge_nodes.insert(node2);
        node2->edge_nodes.insert(node1);
    }

    AreaNode* add_node(AreaNode* edge_node, Coord start_coord){
        auto new_node = std::make_unique<AreaNode>(start_coord, next_id);
        auto new_node_ptr = new_node.get();
        this->connect_nodes(edge_node, new_node.get());
        nodes[next_id] = std::move(new_node);
        new_nodes.push_back(next_id);
        next_id++;
        return new_node_ptr;
    }

    void remove_node(int id){
        auto node = nodes[id].get();
        for (auto& edge_node : node->edge_nodes){
            edge_node->edge_nodes.erase(node);
        }
        nodes.erase(id);
    }

    std::vector<LoopBackData*> get_valid_loop_backs(std::vector<AreaNode*> discard_nodes, SearchStepData* current_step_data){
        std::vector<LoopBackData*> loops;
        if (!current_step_data->loop_backs.empty()){
            std::sort(current_step_data->loop_backs.begin(), current_step_data->loop_backs.end(), [](const std::unique_ptr<LoopBackData>& a, const std::unique_ptr<LoopBackData>& b){
                return a->steps > b->steps;
            });
            for(const auto& loop_data_uni : current_step_data->loop_backs){
                auto loop_data = loop_data_uni.get();
                if (std::find_if(discard_nodes.begin(), discard_nodes.end(), [loop_data](AreaNode* node){return node == loop_data->nodes.back() || node == loop_data->nodes.front();}) != discard_nodes.end()){
                    continue;
                }
                auto it = std::find_if(loops.begin(), loops.end(), [&loop_data](LoopBackData* loop){
                    return !loop->can_combine(loop_data);
                });
                if (it == loops.end()){
                    loops.push_back(loop_data);
                }
            }
        }
        return loops;
    }

    std::vector<AreaNode*> get_all_nodes_in_loop(LoopBackData* loop, std::unordered_map<int, SearchStepData>& node_search_data){
        std::unordered_set<AreaNode*> all_nodes;
        AreaNode* prev_node = nullptr;
        for (auto it = loop->nodes.begin(); it != loop->nodes.end(); ++it) {
            auto node = *it;
            auto search_data = &node_search_data[loop->nodes.front()->id];
            AreaNode* next_node = nullptr;

            // Check if there is a next element
            if (std::next(it) != loop->nodes.end()) {
                next_node = *std::next(it);
                // You can now use next_node as needed
            }
            std::vector<AreaNode*> discarded = {prev_node, next_node};

            auto sub_loops = get_valid_loop_backs(discarded, search_data);
            for (auto sub_loop : sub_loops) {
                auto sub_nodes = get_all_nodes_in_loop(sub_loop, node_search_data);
                all_nodes.insert(sub_nodes.begin(), sub_nodes.end());
            }
            all_nodes.insert(node);
            prev_node = node;
        }

        return std::vector<AreaNode*>(all_nodes.begin(), all_nodes.end());
    }

    std::pair<unsigned int, unsigned int> get_total_extra_tiles(std::vector<LoopBackData*> loop_backs, std::unordered_map<int, SearchStepData>& node_search_data){
        unsigned int extra_tiles = 0;
        unsigned int extra_food = 0;
        std::vector<AreaNode*> all_nodes;
        for (auto loop_back : loop_backs){
            auto nodes = get_all_nodes_in_loop(loop_back, node_search_data);
            all_nodes.insert(all_nodes.end(), nodes.begin(), nodes.end());
        }
        for (auto node : all_nodes){
            extra_tiles += node->tile_count;
            extra_food += node->food_count;
        }
        return std::make_pair(extra_tiles, extra_food);
    }

    AreaCheckResult search_best(int snake_length, bool food_check){
        bool forward = true;

        std::vector<SearchStepData*> search_stack;
        std::vector<int> total_tile_count_stack;
        std::vector<int> total_food_count_stack;

        // AreaNode, total_steps in the loop
        std::vector<std::pair<AreaNode*, std::unique_ptr<LoopBackData>>> loop_back_track;

        SearchStepData* prev_step = nullptr;
        AreaCheckResult result;

        // initialize the search data for each node
        std::unordered_map<int, SearchStepData> node_search_data;
        for (auto& node : nodes){
            // std::cout << "setting search data for node: " << node.first << std::endl;
            // std::cout << "node: " << node.second.get()->id << std::endl;
            node_search_data[node.first] = std::move(SearchStepData(node.second.get()));
        }
        // std::cout << "Root node: " << root->id << std::endl;
        // std::cout << "Root node start coord: (" << root->start_coord.x << ", " << root->start_coord.y << ")" << std::endl;

        // initialize the search stack with the root node
        // SearchStepData first_step_data = SearchStepData(root, snake_length);
        search_stack.push_back(&node_search_data[root->id]);

        // std::cout << "stack size after root: " << search_stack.size() << std::endl;
        // std::cout << "First node to explore: " << search_stack.back()->node->id << std::endl;

        AreaNode* current_node = nullptr;

        while(!search_stack.empty()){

            SearchStepData* step_data = search_stack.back();
            current_node = step_data->node;
            // std::cout << "Checking new node:" << current_node->id << " (" << current_node->start_coord.x << ", " << current_node->start_coord.y << ")" << std::endl;
            int best_needed_steps_here = INT_MAX;
            int current_tile_count = current_node->tile_count;
            int current_food_count = current_node->food_count;
            int total_tile_count_here = current_tile_count + (total_tile_count_stack.empty() ? 0 : total_tile_count_stack.back());
            int total_food_count_here = current_food_count + (total_food_count_stack.empty() ? 0 : total_food_count_stack.back());
            // std::cout << "Total tile count here: " << total_tile_count_here << std::endl;
            // std::cout << "Total food count here: " << total_food_count_here << std::endl;
            std::vector<AreaNode*> discarded;
            auto all_valid_loop_backs = get_valid_loop_backs(discarded, step_data);
            int all_extra_tiles = 0;
            int all_extra_food = 0;
            for (auto loop_back : all_valid_loop_backs){
                all_extra_tiles += loop_back->tile_count;
                all_extra_food += loop_back->food_count;
            }
            std::cout << "\nNEW ITERATION: " << (forward ? "--> FORWARD" : "<-- BACKWARDS") << std::endl;
            std::cout << "Current node: " << current_node->id << std::endl;
            std::cout << "Current node start coord: (" << current_node->start_coord.x << ", " << current_node->start_coord.y << ")" << std::endl;
            std::cout << "Current tile count: " << current_tile_count << std::endl;
            std::cout << "Current food count: " << current_food_count << std::endl;
            std::cout << "Total tile count here: " << total_tile_count_here << std::endl;
            std::cout << "Total food count here: " << total_food_count_here << std::endl;
            std::cout << "All loop backs: ";
            for (auto& loop : all_valid_loop_backs){
                std::cout << "(";
                std::cout << "(" << loop->nodes.front()->start_coord.x << ", " << loop->nodes.front()->start_coord.y << ") -> ";
                std::cout << "(" << loop->nodes.back()->start_coord.x << ", " << loop->nodes.back()->start_coord.y << ")";
                std::cout << ", " << loop->steps << ") ";
            }
            std::cout << std::endl;
            std::cout << "All extra tiles: " << all_extra_tiles << std::endl;
            std::cout << "All extra food: " << all_extra_food << std::endl;
            std::cout << "Tracking loops: ";
            for (auto& loop : loop_back_track){
                std::cout << "(" << loop.first->id << ", " << loop.second->steps << ") ";
            }
            std::cout << std::endl;
            if (forward){
                step_data->visited = true;
                if(current_node->has_tail){
                    result.is_clear = true;
                    result.tile_count = total_tile_count_here;
                    result.food_count = total_food_count_here;
                    result.has_tail = current_node->has_tail;
                    result.margin = snake_length;
                    break;
                }
            }
            if (step_data->has_opening){
                std::cout << "Has opening" << std::endl;
                best_needed_steps_here = snake_length - current_node->max_index;
                int best_needed_steps_node = INT_MAX;
                std::cout << "Best needed steps here: " << best_needed_steps_here << std::endl;
                auto min_needed_steps = std::min_element(step_data->nodes_with_openings.begin(), step_data->nodes_with_openings.end(), [](const std::pair<AreaNode*, int>& a, const std::pair<AreaNode*, int>& b){
                    return a.second < b.second;
                });
                if (min_needed_steps != step_data->nodes_with_openings.end()){
                    best_needed_steps_node = min_needed_steps->second;
                }
                std::cout << "Best needed steps node: " << best_needed_steps_node << std::endl;
                std::vector<LoopBackData*> valid_loop_backs;
                int curr_tiles;
                int curr_food;
                int all_food;
                int all_tiles;
                int food_to_opening;
                int total_steps;
                int margin;
                std::pair<unsigned int, unsigned int> extra_tiles;
                if (best_needed_steps_node < best_needed_steps_here){
                    best_needed_steps_here = best_needed_steps_node;
                    std::cout << "Node is best" << std::endl;
                    int needed_steps_before_entering;
                    AreaNode* opening_node;
                    SearchStepData* opening_search_data;
                    std::vector<AreaNode*> discarded;
                    for (auto node_with_opening : step_data->nodes_with_openings){
                        discarded.clear();
                        needed_steps_before_entering = node_with_opening.second;
                        opening_node = node_with_opening.first;
                        discarded.push_back(opening_node);
                        opening_search_data = &node_search_data[opening_node->id];
                        valid_loop_backs = get_valid_loop_backs(discarded, step_data);
                        int extra_tiles = 0;
                        int extra_food = 0;
                        for (auto loop_back : valid_loop_backs){
                            extra_tiles += loop_back->tile_count;
                            extra_food += loop_back->food_count;
                        }
                        curr_tiles = total_tile_count_here + extra_tiles;
                        curr_food = total_food_count_here + extra_food;
                        food_to_opening = opening_search_data->food_to_best_opening;
                        all_food = curr_food + food_to_opening;
                        all_tiles = curr_tiles + opening_search_data->tiles_to_best_opening;
                        total_steps = curr_tiles - curr_food;
                        margin = total_steps - needed_steps_before_entering;

                        if (margin > result.margin){
                            result.food_count = all_food;
                            result.tile_count = all_tiles;
                            result.margin = margin;
                        }
                        if (result.margin >= 0){
                            result.is_clear = true;
                        }
                        if (result.margin >= all_food){
                            break;
                        }
                    }
                }
                else{
                    std::cout << "Local is best" << std::endl;
                    curr_tiles = total_tile_count_here + all_extra_tiles;
                    curr_food = total_food_count_here + all_extra_food;
                    all_food = curr_food;
                    all_tiles = curr_tiles;
                    total_steps = curr_tiles - curr_food;
                    margin = total_steps - best_needed_steps_here;
                    if (margin > result.margin){
                        result.food_count = all_food;
                        result.tile_count = all_tiles;
                        result.margin = margin;
                    }
                    if (result.margin >= 0){
                        result.is_clear = true;
                    }
                    if (result.margin >= all_food){
                        break;
                    }
                }
            }
            else{
                int curr_tiles = current_tile_count + all_extra_tiles;
                int curr_food = current_food_count + all_extra_food;
                int total_steps = curr_tiles - curr_food;
                int margin = total_steps - (snake_length + 1) ;
                std::cout << "No opening" << std::endl;
                if (margin > result.margin){
                    result.food_count = curr_food;
                    result.tile_count = curr_tiles;
                    result.margin = margin;
                }
                if (result.margin >= 0){
                    result.is_clear = true;
                }
                if (result.margin >= curr_food){
                    break;
                }
            }

            if (!step_data->to_explore_nodes.empty()){
                // std::cout << "Forward to next node" << std::endl;
                forward = true;
                // Get the next node to explore
                AreaNode* next_node = step_data->to_explore_nodes.back();
                step_data->to_explore_nodes.pop_back();
                // std::cout << "next node ID: " << next_node->id << std::endl;
                // Get the search data for the next node and add it to the search stack
                auto next_step_data = &node_search_data[next_node->id];
                // Remove the current node from the next node's to explore list
                auto new_end = std::remove(next_step_data->to_explore_nodes.begin(), next_step_data->to_explore_nodes.end(), current_node);
                next_step_data->to_explore_nodes.erase(new_end, next_step_data->to_explore_nodes.end());

                if(next_step_data->visited){
                    auto loop_back_data = std::make_unique<LoopBackData>();
                    loop_back_data->add_node(current_node);
                    loop_back_track.push_back(std::make_pair(next_node, std::move(loop_back_data)));
                    continue;
                }

                if (search_stack.size() >= 2){
                    // Check corner
                    int prev_x = search_stack[search_stack.size() - 2]->node->start_coord.x;
                    int prev_y = search_stack[search_stack.size() - 2]->node->start_coord.y;
                    int next_x = next_node->start_coord.x;
                    int next_y = next_node->start_coord.y;
                    int delta_x = next_x - prev_x;
                    int delta_y = next_y - prev_y;

                    // Check if path goes through a corner, if so then only extend the tile count stack by 1
                    // if we are currently on node 2
                    // going like this 1 -> 2 -> 3
                    /*
                        x 3 x
                        1 2 2
                        x 2 2
                    */
                    if (std::abs(delta_x) == 1 && std::abs(delta_y) == 1){
                        total_tile_count_stack.push_back(total_tile_count_here - current_tile_count + 1);
                    }
                    else{
                        total_food_count_stack.push_back(total_food_count_here);
                        total_tile_count_stack.push_back(total_tile_count_here);
                    }
                }
                else{
                    total_food_count_stack.push_back(total_food_count_here);
                    total_tile_count_stack.push_back(total_tile_count_here);
                }
                // End check corner
                search_stack.push_back(next_step_data);
                // std::cout << "Pushed to stack: (" << next_step_data->node->start_coord.x << ", " << next_step_data->node->start_coord.y << std::endl;
            }
            else{
                forward = false;
                search_stack.pop_back();
                total_tile_count_stack.pop_back();
                total_food_count_stack.pop_back();
                if (search_stack.empty()){
                    break;
                }
                auto next_step_data = search_stack.back();

                // If this step does not have an opening, then we know that it is a dead end, becasue we would have returned if it was not
                if (step_data->has_opening){
                    next_step_data->has_opening = true;
                    int needed_steps = best_needed_steps_here;
                    int total_food = current_food_count + all_extra_food;
                    int total_tiles = current_tile_count + all_extra_tiles;
                    int total_steps = total_tiles - total_food;
                    needed_steps -= total_steps;
                    next_step_data->nodes_with_openings.push_back(std::make_pair(current_node, needed_steps));
                    next_step_data->food_to_best_opening += total_food;
                    next_step_data->tiles_to_best_opening += total_tiles;
                }
                else{
                    next_step_data->dead_end_nodes.push_back(current_node);
                }

                if (!loop_back_track.empty()){
                    step_data->to_explore_nodes.push_back(next_step_data->node);
                    step_data->visited = false;
                    // for every loop back that we are currently tracking, we need to add this nodes tile count to the loop back
                    for(auto it = loop_back_track.begin(); it != loop_back_track.end(); ++it){
                        it->second->add_node(current_node);
                    }
                    // std::cout << "Filling loop back tracking" << std::endl;

                    auto it = std::find_if(loop_back_track.begin(), loop_back_track.end(), [current_node, next_step_data](const std::pair<AreaNode*, std::unique_ptr<LoopBackData>>& loop_data){
                        return loop_data.first == next_step_data->node;
                    });
                    // std::cout << "checking if loop back is found" << std::endl;

                    if (it != loop_back_track.end()){
                        auto loop_data = std::move(it->second);
                        std::vector<LoopBackData*> loops = {loop_data.get()};
                        auto tile_count = get_total_extra_tiles(loops, node_search_data);
                        loop_data->tile_count = tile_count.first;
                        loop_data->food_count = tile_count.second;
                        loop_data->steps = loop_data->tile_count - loop_data->food_count;
                        next_step_data->loop_backs.push_back(std::move(loop_data));
                        loop_back_track.erase(it);
                    }

                }
            }
            std::cout << "Stack size: " << search_stack.size() << std::endl;
            prev_step = step_data;
            std::cout << "END OF ITERATION" << std::endl;
        }
        new_nodes.clear();
        return result;
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
            // AreaCheckResult result = _area_check(
            //     s_map_ptr,
            //     body_coords,
            //     start_coord,
            //     0,
            //     0,
            //     0,
            //     checked,
            //     0,
            //     area_stats,
            //     food_check
            // );

            AreaCheckResult result = _area_check2(
                s_map_ptr,
                body_coords,
                start_coord,
                food_check
            );

            py::list food_coords;
            for (auto& food_coord : result.food_coords) {
                food_coords.append(py::make_tuple(food_coord.x, food_coord.y));
            }
            py::list connected_areas;
            for (auto& connected_area : result.connected_areas) {
                connected_areas.append(connected_area);
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
                py::arg("food_coords") = food_coords,
                py::arg("connected_areas") = connected_areas
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
        std::unordered_set<Coord> to_explore;
        std::unordered_set<int> connected_areas;
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
                        connected_areas.insert(checked_val);
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
                        to_explore.insert(n_coord);
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
                    if (n_coord == tail_coord && curr_coord != start_coord) {
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
        bool food_check
    ){
        bool safe_margin = false;
        std::vector<int> checked;
        checked.resize(height * width);
        std::fill(checked.begin(), checked.end(), unexplored_area_id);

        AreaGraph graph = AreaGraph(start_coord, body_coords.size());
        std::deque<AreaNode*> nodes_to_explore;
        nodes_to_explore.push_back(graph.root);
        AreaCheckResult best_result;
        while(!nodes_to_explore.empty()){
            auto current_node = nodes_to_explore.front();
            nodes_to_explore.pop_front();
            if (current_node->explored){
                continue;
            }
            // std::cout << "Exploring area start: (" << current_node->start_coord.x << ", " << current_node->start_coord.y << std::endl;
            auto result = explore_area(
                s_map,
                body_coords,
                current_node->start_coord,
                current_node->id,
                checked
            );
            if (result.tile_count == 0){
                graph.remove_node(current_node->id);
            }
            else{
                current_node->tile_count = result.tile_count;
                if (current_node->tile_count < 4){
                    current_node->is_one_dim = true;
                }
                current_node->food_count = result.food_count;
                current_node->max_index = result.max_index;
                current_node->has_tail = result.has_tail;
                current_node->explored = true;
                for (auto& coord : result.to_explore){
                    auto new_node = graph.add_node(current_node, coord);
                    nodes_to_explore.push_back(new_node);
                }
                for (auto& connected_area : result.connected_areas){
                    graph.connect_nodes(current_node->id, connected_area);
                }
            }
            // std::cout << "explored area: (" << current_node->start_coord.x << ", " << current_node->start_coord.y << ") " << "max_index:" << current_node->max_index << std::endl;
            // std::cout << "to explore: " << graph.node_search_data[current_node->id].get()->to_explore_nodes.size() << std::endl;
            // for(auto& datapair : graph.node_search_data){
            //     auto data = datapair.second.get();
            //     std::cout << "ID: " << data->node->id << std::endl;
            //     std::cout << "Node: (" << data->node->start_coord.x << ", " << data->node->start_coord.y << ") " << "max_index:" << data->node->max_index << std::endl;
            //     for (auto node : data->to_explore_nodes){
            //         std::cout << "  To explore: (" << node->start_coord.x << ", " << node->start_coord.y << ")" << std::endl;
            //     }
            // }
            if (!food_check){
                AreaCheckResult check_result = graph.search_best(body_coords.size(), food_check);
                if (check_result.margin >= check_result.food_count){
                    return check_result;
                }
                else{
                    if (check_result.margin > best_result.margin){
                        best_result = check_result;
                    }
                }
            }
        }
        if (best_result.initialized){
            return best_result;
        }
        else{
            return graph.search_best(body_coords.size(), food_check);
        }


        // return AreaCheckResult();
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
        std::deque<Coord> current_coords;
        std::vector<Coord> to_be_checked;
        std::vector<int> connected_areas;
        std::vector<AreaCheckResult> sub_checks;
        current_coords.push_back(start_coord);


        if (checked.size() == 0) {
            checked.resize(height * width);
            std::fill(checked.begin(), checked.end(), -1);
        }
        checked[start_coord.y * width + start_coord.x] = depth;
        while (!current_coords.empty()) {
            auto curr_coord = current_coords.front();
            current_coords.pop_front();
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
                        connected_areas.push_back(checked_val);
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
                //         tile_count = area_check.tile_count;
                //         food_count = area_check.food_count;
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
            //     int sub_check_tile_count = sub_check.tile_count - sub_check.prev_tile_count;
            //     int sub_check_food_count = sub_check.food_count - sub_check.prev_food_count;
            //     int sub_check_total_tiles = sub_check_tile_count - sub_check_food_count;
            //     total_steps = best_total_steps + sub_check_total_tiles;
            //     int calc_margin = total_steps - needed;
            //     if (calc_margin > best_margin) {
            //         best_margin = calc_margin;
            //         best_total_steps = total_steps;
            //         best_tile_count = best_tile_count + sub_check_tile_count;
            //         best_food_count = best_food_count + sub_check_food_count;
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
            food_coords,
            connected_areas
        );
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
