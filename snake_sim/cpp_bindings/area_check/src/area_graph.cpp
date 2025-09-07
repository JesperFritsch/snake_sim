#include "area_graph.hpp"
#include "area_debug.hpp"

AreaCheckResult AreaGraph::search_best2(
    int snake_length, 
    uint8_t *s_map, 
    uint8_t food_value, 
    int width, 
    int target_margin, 
    bool food_check, 
    bool exhaustive
)
{
    bool forward = true;
    bool skipped_one = false;
    // Map to keep track of visited nodes
    std::unordered_map<AreaNode *, SearchNode> search_nodes_data;
    for (auto &node : nodes)
    {
        search_nodes_data[node.second.get()] = SearchNode(node.second.get());
    }
    // pair(cantor_pairing of from_node - to_node, needed_steps), are the elements to be cached
    std::vector<SearchNode *> search_stack;
    std::vector<int> total_tile_count_stack;
    std::vector<int> total_food_count_stack;
    total_food_count_stack.reserve(100);
    total_tile_count_stack.reserve(100);
    search_stack.reserve(100);
    search_stack.push_back(&search_nodes_data[root]);
    AreaCheckResult best_result;
    AreaNode *current_node = nullptr;
    AreaNode *prev_node = nullptr;

    while (!search_stack.empty())
    {
        AreaCheckResult current_result;
        SearchNode *step_data = search_stack.back();
        current_node = step_data->node;
        int tiles_before = total_tile_count_stack.empty() ? 0 : total_tile_count_stack.back();
        int food_before = total_food_count_stack.empty() ? 0 : total_food_count_stack.back();
        if (forward)
        {
            if (!skipped_one)
            {
                step_data->enter_from(prev_node, tiles_before, food_before);
            }
            skipped_one = false;
        }
        else
        {
            step_data->enter_unwind(tiles_before, food_before);
        }

        // if this is not the first visit to this node, then countable tiles are 0
        auto curr_tile_counts = step_data->tile_count_on_enter();

        int needed_steps;
        int margin;
        int total_steps;
        int calc_tiles;
        int calc_food;
        if (step_data->node->has_tail && !food_check && !(step_data->node->tile_count == 1 && step_data-> node->id == 0))
        {
            best_result.has_tail = true;
            best_result.margin = INT_MAX;
            best_result.is_clear = true;
            best_result.tile_count = curr_tile_counts.total_tiles;
            best_result.food_count = curr_tile_counts.total_food;
            best_result.total_steps = curr_tile_counts.total_tiles;
            break;
        }
        auto max_body_index_pair = step_data->get_max_body_index_pair();
        if (max_body_index_pair.first != -1)
        {
            calc_tiles = curr_tile_counts.total_tiles + step_data->get_max_body_tile_adjustment(max_body_index_pair.second);
            calc_food = curr_tile_counts.total_food;
            total_steps = calc_tiles - calc_food;
            needed_steps = snake_length - max_body_index_pair.first;
            margin = total_steps - needed_steps;
        }
        else if(step_data->first_visit())
        {
            calc_tiles = curr_tile_counts.new_tiles;
            calc_food = curr_tile_counts.total_food;
            total_steps = calc_tiles - calc_food;
            needed_steps = snake_length + 1;
            margin = total_steps - needed_steps;
        }
        else
        // if this is not the first time we visit this node, then we know that we have 
        // visited other nodes since the first time and we can therefore calculate with
        // the total tile count minus the total tile count at the node before first visit of this node.
        {
            calc_tiles = curr_tile_counts.total_tiles - step_data->first_tiles_until_here;
            calc_food = curr_tile_counts.total_food - step_data->first_food_until_here;
            total_steps = calc_tiles - calc_food;
            needed_steps = snake_length + 1;
            margin = total_steps - needed_steps;
        }
        current_result.margin = margin;
        current_result.total_steps = total_steps;
        current_result.tile_count = calc_tiles;
        current_result.food_count = calc_food;
        current_result.needed_steps = needed_steps;
        current_result.has_tail = step_data->node->has_tail;
        if (current_result.margin >= 0)
        {
            current_result.is_clear = true;
        }
        
        
        DEBUG_PRINT(std::cout << "\n####### NODE #######" << std::endl;);
        DEBUG_PRINT(std::cout << (forward ? "--> Forward" : "<-- Backward") << std::endl;);
        DEBUG_PRINT(std::cout << "nr_visits: " << step_data->nr_visits << std::endl;);
        DEBUG_PRINT(std::cout << "Current node: " << current_node->id << std::endl;);
        DEBUG_PRINT(std::cout << "start coord: (" << current_node->start_coord.x << ", " << current_node->start_coord.y << ")" << std::endl;);
        DEBUG_PRINT(std::cout << "node tile count: " << current_node->tile_count << std::endl;);
        DEBUG_PRINT(std::cout << "node food count: " << current_node->food_count << std::endl;);
        DEBUG_PRINT(std::cout << "max body pair: (" << max_body_index_pair.first << ", (" << max_body_index_pair.second.x << ", " << max_body_index_pair.second.y << "))" << std::endl;);
        DEBUG_PRINT(std::cout << "Body tiles: ";);
        DEBUG_PRINT(for(auto body_tile : current_node->body_tiles){ std::cout << "(" << body_tile.first << ", (" << body_tile.second.x << ", " << body_tile.second.y << ")), "; });
        DEBUG_PRINT(std::cout << std::endl;);
        DEBUG_PRINT(std::cout << "has body: " << current_node->has_body << std::endl;);
        DEBUG_PRINT(std::cout << "has only head: " << current_node->has_only_head << std::endl;);
        DEBUG_PRINT(std::cout << "nr body tiles: " << current_node->body_tiles.size() << std::endl;);
        DEBUG_PRINT(std::cout << "coord parity diff: " << current_node->coord_parity_diff << std::endl;);
        DEBUG_PRINT(std::cout << "jagged edge discount: " << current_node->jagged_edge_discount << std::endl;);
        DEBUG_PRINT(std::cout << "is one dim: " << current_node->is_one_dim << std::endl;);
        DEBUG_PRINT(std::cout << "has tail: " << current_node->has_tail << std::endl;);
        DEBUG_PRINT(std::cout << "Tiles before: " << tiles_before << std::endl;);
        DEBUG_PRINT(std::cout << "Food before: " << food_before << std::endl;);
        DEBUG_PRINT(std::cout << "current node new tiles: " << curr_tile_counts.new_tiles << std::endl;);
        DEBUG_PRINT(std::cout << "tiles until here: " << step_data->tiles_until_here << std::endl;);
        DEBUG_PRINT(std::cout << "food until here: " << step_data->food_until_here << std::endl;);
        DEBUG_PRINT(std::cout << "needed steps: " << needed_steps << std::endl;);
        DEBUG_PRINT(std::cout << "total steps: " << total_steps << std::endl;);
        DEBUG_PRINT(std::cout << "margin: " << margin << std::endl;);
        DEBUG_PRINT(std::cout << "searched edges now: ";);
        DEBUG_PRINT(for(auto edge : step_data->searched_edges.back()){ std::cout << edge << ", "; });
        DEBUG_PRINT(std::cout << std::endl;);
        DEBUG_PRINT(std::cout << "used edges: ";);
        DEBUG_PRINT(for(auto edge : step_data->used_edges){ std::cout << edge << ", "; });
        DEBUG_PRINT(std::cout << std::endl;);
        DEBUG_PRINT(std::cout << "edge nodes: ";);
        DEBUG_PRINT(for(auto edge_node : step_data->node->edge_nodes){ std::cout << "(" << edge_node.first->id << ", " << edge_node.second << "), "; });
        DEBUG_PRINT(std::cout << std::endl;);
        DEBUG_PRINT(std::cout << "search stack: (";);
        DEBUG_PRINT(for(auto node : search_stack){ std::cout << node->node->id << ", "; });
        DEBUG_PRINT(std::cout << ")" << std::endl;);
        DEBUG_PRINT(std::cout << "total tile count stack: (";);
        DEBUG_PRINT(for(auto count : total_tile_count_stack){ std::cout << count << ", "; });
        DEBUG_PRINT(std::cout << ")" << std::endl;);
        DEBUG_PRINT(std::cout << "total food count stack: (";);
        DEBUG_PRINT(for(auto count : total_food_count_stack){ std::cout << count << ", "; });
        DEBUG_PRINT(std::cout << ")" << std::endl;);
        DEBUG_PRINT(std::cout << std::endl;);
        DEBUG_PRINT(std::cout << "current result: \n";);
        DEBUG_PRINT(std::cout << "  is clear: " << current_result.is_clear << std::endl;);
        DEBUG_PRINT(std::cout << "  tile count: " << current_result.tile_count << std::endl;);
        DEBUG_PRINT(std::cout << "  food count: " << current_result.food_count << std::endl;);
        DEBUG_PRINT(std::cout << "  needed steps: " << current_result.needed_steps << std::endl;);
        DEBUG_PRINT(std::cout << "  margin: " << current_result.margin << std::endl;);
        DEBUG_PRINT(std::cout << "  total steps: " << current_result.total_steps << std::endl;);
        DEBUG_PRINT(std::cout << "  has tail: " << current_result.has_tail << std::endl;);
        DEBUG_PRINT(std::cout << "best result: \n";);
        DEBUG_PRINT(std::cout << "  is clear: " << best_result.is_clear << std::endl;);
        DEBUG_PRINT(std::cout << "  tile count: " << best_result.tile_count << std::endl;);
        DEBUG_PRINT(std::cout << "  food count: " << best_result.food_count << std::endl;);
        DEBUG_PRINT(std::cout << "  needed steps: " << best_result.needed_steps << std::endl;);
        DEBUG_PRINT(std::cout << "  margin: " << best_result.margin << std::endl;);
        DEBUG_PRINT(std::cout << "  total steps: " << best_result.total_steps << std::endl;);
        DEBUG_PRINT(std::cout << "  has tail: " << best_result.has_tail << std::endl;);
        
        if (food_check)
        {
            if (
                current_result.margin > current_result.food_count &&
                (current_result.food_count >= best_result.food_count))
            {
                best_result = current_result;
            }
        }
        else
        {
            if (current_result.margin > best_result.margin)
            {
                best_result = current_result;
            }
            if ((best_result.margin >= target_margin && best_result.margin > best_result.food_count) && !exhaustive)
            {
                break;
            }
        }

        auto node_edge_pair = step_data->get_next_node_and_edge();
        if (node_edge_pair.first != nullptr)
        {

            forward = true;
            auto next_node = node_edge_pair.first;
            auto next_step_data = &search_nodes_data[next_node];
            // this check is important, if we have visited a node with only 1 tile, then we can not visit it again
            // also if we have used the connection coord already in the path we can not enter the node on that coord again
            if (!step_data->can_enter_next_node(next_node) || !next_step_data->can_enter_from_node(current_node))
            {
                DEBUG_PRINT(std::cout << "Cannot enter node: " << next_node->id << std::endl;);
                step_data->add_searched_edge(node_edge_pair.second);
                skipped_one = true;
                continue;
            }
            // std::cout << "Next node: " << next_node->id << std::endl;
            step_data->add_searched_edge(node_edge_pair.second);
            step_data->add_used_edge(node_edge_pair.second);
            search_stack.push_back(next_step_data);
            TileCounts tile_counts_on_exit = step_data->tile_count_on_exit(next_node, s_map, width, food_value);
            total_tile_count_stack.push_back(tile_counts_on_exit.total_tiles);
            total_food_count_stack.push_back(tile_counts_on_exit.total_food);
            step_data->exit_to(next_node);
        }
        else
        {
            forward = false;
            search_stack.pop_back();
            total_food_count_stack.pop_back();
            total_tile_count_stack.pop_back();
            if (search_stack.empty())
            {
                break;
            }
            step_data->exit_unwind();
        }
        prev_node = current_node;
    }
    return best_result;
}

void AreaGraph::print_nodes_debug() const {
    for (const auto& node_pair : nodes) {
        const AreaNode* node = node_pair.second.get();
        std::cout 
            << "Node ID: " << node->id  << std::endl
            << "Tile Count: " << node->tile_count  << std::endl
            << "Food Count: " << node->food_count  << std::endl
            << "Even or Odd balance: " << node->coord_parity_diff << std::endl
            << "Has Tail: " << node->has_tail  << std::endl
            << "Is One Dim: " << node->is_one_dim  << std::endl
            << "Start Coord: (" << node->start_coord.x << ", " << node->start_coord.y << ")" << std::endl
            << "Body Tiles: " << std::endl
            << "Jagged edge discount: " << node->jagged_edge_discount << std::endl;
        for (const auto& body_tile : node->body_tiles) {
            std::cout << "(" << body_tile.first << ", (" << body_tile.second.x << ", " << body_tile.second.y << ")), ";
        }
        std::cout << std::endl
            << "Connections: " << std::endl;
        for (const auto& conn_pair : node->neighbour_connections) {
            int connected_area_id = conn_pair.first;
            const ConnectedAreaInfo& info = conn_pair.second;
            std::cout << "    Connected to area: " << connected_area_id
                    << " [Self: (" << info.self_coord.x << ", " << info.self_coord.y << ")"
                    << " Other: (" << info.other_coord.x << ", " << info.other_coord.y << ")]"
                    << " Bad gateway: " << info.is_bad_gateway_from_here
                    << " Diag gateway: " << info.is_diag_gateway
                    << std::endl;
        }
    }
}

void AreaGraph::remove_node(int id)
{
    auto node = nodes[id].get();
    for (auto &edge_node : node->edge_nodes)
    {
        edge_node.first->remove_connection(node);
    }
    nodes.erase(id);
}

void AreaGraph::add_id_for_node(int original_id, int linked_id)
{
    auto original_node_it = nodes.find(original_id);
    if (original_node_it == nodes.end())
    {
        throw std::out_of_range("Original node ID not found");
    }
    // Map the new id to the same AreaNode object (no copy)
    nodes[linked_id] = original_node_it->second;
}

AreaNode* AreaGraph::add_node_with_id(Coord start_coord, int id)
{
    auto new_node = std::make_shared<AreaNode>(start_coord, id);
    auto new_node_ptr = new_node.get();
    nodes[id] = new_node;
    if (id == 0)
    {
        root = new_node_ptr;
    }
    return new_node_ptr;
}

void AreaGraph::connect_nodes(AreaNode *node1, AreaNode *node2, ConnectedAreaInfo conn_info1, ConnectedAreaInfo conn_info2)
// conn_info1 has info about the connection from node1 to node2 and conn_info2 vice versa
{
    if (node1 == nullptr || node2 == nullptr || node1->id == node2->id)
    {
        return;
    }
    if (node1->id > node2->id)
    {
        std::swap(node1, node2);
        std::swap(conn_info1, conn_info2);
    }
    auto edge_id = cantor_pairing(node1->id, node2->id);
    node1->add_connection(node2, edge_id, conn_info1);
    node2->add_connection(node1, edge_id, conn_info2);
}

void AreaGraph::connect_nodes(int id1, int id2, ConnectedAreaInfo conn_info)
// the order of id1 and id2 does matter, conn_info has info about the connection from id1 to id2
{
    AreaNode *node1 = get_node(id1);
    AreaNode *node2 = get_node(id2);
    if (node1 == nullptr || node2 == nullptr || node1->id == node2->id)
    {
        return;
    }
    if (conn_info.is_diag_gateway) {
        // a discovered connection when the gate is diagonal is means that the tile in 
        // the middle of the diagonal is the other_coord of conn_info
        // and that tile should be the self_coord of both nodes because it is esentially both the enter and exit of 
        // both areas.
        conn_info.self_coord = conn_info.other_coord;
    }
    // Create ConnectedAreaInfo for the other node, so coords and bad_gateway are swapped.
    ConnectedAreaInfo conn_info2 = ConnectedAreaInfo(
        id1,
        conn_info.other_coord,
        conn_info.self_coord,
        conn_info.is_bad_gateway_to_here,
        conn_info.is_bad_gateway_from_here,
        conn_info.is_diag_gateway
    );
    connect_nodes(node1, node2, conn_info, conn_info2);
}


bool SearchNode::count_bad_gate_way(Coord entry_coord, Coord exit_coord)
{
    Coord delta = exit_coord - entry_coord;
    // if we enter and exit on adjacent (non-diagonal) tiles then it is not a bad gateway
    if ((std::abs(delta.x) == 1 && std::abs(delta.y) == 0) ||
        (std::abs(delta.x) == 0 && std::abs(delta.y) == 1))
    {
        return false;
    }
    return true;
}


int SearchNode::path_tile_adjustment(AreaNode *next_node)
// depending of the path in nodes the amount of visitable tiles may change
// more complex logic could possibly be implemented here sp that the exact number of visitable tiles
// is more accurate, now it might be a bit lower than reality.
{
    auto next_connection_info = node->get_connection_info(next_node->id);
    Coord entry_coord = get_entry_coord();
    Coord exit_coord = next_connection_info.self_coord;
    DEBUG_PRINT(std::cout << "Entry coord: (" << entry_coord.x << ", " << entry_coord.y << "), Exit coord: (" << exit_coord.x << ", " << exit_coord.y << ")" << std::endl;);
    if (node->is_one_dim || entry_coord == exit_coord)
    {
        DEBUG_PRINT(std::cout << "Path tile adjustment from node " << node->id << " to node " << next_node->id << " is 0 (one_dim or same coord)" << std::endl;);
        return 0;
    }
    int adjustment = 0;
    if (next_connection_info.is_bad_gateway_from_here && count_bad_gate_way(entry_coord, exit_coord))
    {
        adjustment -= 1;
    }
    adjustment += path_parity_tile_adjustment(entry_coord, exit_coord);
    return adjustment;
}

int SearchNode::get_max_body_tile_adjustment(Coord max_index_coord)
{
    Coord entry_coord = get_entry_coord();
    if (node->is_one_dim || !node->has_body)
    {
        return 0;
    }
    return path_parity_tile_adjustment(entry_coord, max_index_coord);

}

int SearchNode::path_parity_tile_adjustment(Coord enter, Coord exit){
    if (enter == exit || node->is_one_dim){
        return 0;
    }

    bool enter_parity = get_coord_mod_parity(enter);
    bool exit_parity = get_coord_mod_parity(exit);
    int adjustment = 0;
    if(node->coord_parity_diff == 0 && enter_parity == exit_parity){
        adjustment = -1;
    }
    else if(node->coord_parity_diff > 0){
        // more even tiles
        if (!enter_parity && !exit_parity){
            adjustment = -1;
        }
        else if (enter_parity && exit_parity){
            adjustment = 1;
        }
    }
    else{
        // more odd tiles
        if (enter_parity && exit_parity){
            adjustment = -1;
        }
        else if (!enter_parity && !exit_parity){
            adjustment = 1;
        }
    }
    DEBUG_PRINT(std::cout << "Path parity tile adjustment from (" << enter.x << ", " << enter.y << ") to (" << exit.x << ", " << exit.y << ") is " << adjustment << std::endl;);
    return adjustment;
}

Coord SearchNode::get_entry_coord()
{
    if (!entered_from_nodes.empty()) {
        auto previous_connection_info = node->get_connection_info(entered_from_nodes.back());
        return previous_connection_info.self_coord;
    }
    else {
        return node->start_coord;
    }
}

std::pair<int, Coord> SearchNode::get_max_body_index_pair()
{
    if (node->body_tiles.empty())
    {
        return std::make_pair(-1, Coord());
    }
    if (node->tile_count == 1){
        return node->body_tiles[0];
    }
    Coord entry_coord = get_entry_coord();
    auto it = std::find_if(
        node->body_tiles.begin(), 
        node->body_tiles.end(), 
        [&](const std::pair<int, Coord> &pair){ return (
            ((entry_coord == pair.second) && visited_before() ? 
            get_coord_used_count(pair.second) < 2 : 
            get_coord_used_count(pair.second) < 1) &&
            !(node->id == 0 && pair.second == node->start_coord) 
        ); }
    );
    if (it == node->body_tiles.end()){
        return std::make_pair(-1, Coord());
    }
    return *it;
}

unsigned int SearchNode::get_coord_used_count(Coord coord)
// returns how many times the coord is used in this node path
{
    if (node->tile_count == 1 || node->is_one_dim){
        return 0;
    }
    return static_cast<unsigned int>(std::count(used_coords.begin(), used_coords.end(), coord));
}

bool SearchNode::is_conn_coord_used(AreaNode *other_node)
// returns true if the connection coord is already used in the path
{
    if (node->tile_count == 1 || node->is_one_dim){
        return false;
    }
    auto next_connection_info = node->get_connection_info(other_node->id);
    Coord conn_coord = next_connection_info.self_coord;
    DEBUG_PRINT(std::cout << "Used coords: "; for (const auto& coord : used_coords) { std::cout << "(" << coord.x << ", " << coord.y << ") "; } std::cout << std::endl;);
    return get_coord_used_count(conn_coord) > 0;
}

bool SearchNode::is_conn_coord_start_cord(AreaNode *next_node)
{
    if (node->tile_count == 1 || node->is_one_dim || node->id != 0){
        return false;
    }
    auto next_connection_info = node->get_connection_info(next_node->id);
    Coord exit_coord = next_connection_info.self_coord;
    DEBUG_PRINT(std::cout << "is start coord blocking exit Start coord: (" << node->start_coord.x << ", " << node->start_coord.y << "), Exit coord: (" << exit_coord.x << ", " << exit_coord.y << ")" << std::endl;);
    if (node->start_coord == exit_coord && node->id == 0){
        return true;
    }
    return false;
}

bool SearchNode::can_enter_next_node(AreaNode *next_node){
    auto next_connection_info = node->get_connection_info(next_node->id);
    Coord conn_coord = next_connection_info.self_coord;
    Coord enter_coord = get_entry_coord();
    DEBUG_PRINT(std::cout << "Entry coord: (" << enter_coord.x << ", " << enter_coord.y << "), Connection coord: (" << conn_coord.x << ", " << conn_coord.y << ")" << std::endl;);
    auto conn_coord_used_count = get_coord_used_count(conn_coord);
    if (enter_coord == conn_coord && conn_coord_used_count < 2){
        return true;
    }
    return (is_conn_coord_used(next_node) && visited_before()) ? false : true;
}

TileCounts SearchNode::tile_count_on_exit(AreaNode *next_node, uint8_t *s_map, int width, uint8_t food_value)
{
    auto tile_counts = tile_count_on_enter();
    if (is_conn_coord_start_cord(next_node))
    {
        // if we are exiting through the start coord of node 0 we can ever only count one tile.
        tile_counts.total_tiles = 1;
        tile_counts.total_food = food_until_here + (tile_has_food(s_map, width, node->start_coord, food_value) ? 1 : 0);
    }
    else if (is_conn_coord_used(next_node))
    {
        Coord tile_coord = get_entry_coord();
        tile_counts.total_tiles = tiles_until_here + (first_visit() ? 1 : 0);
        tile_counts.total_food = food_until_here + (tile_has_food(s_map, width, tile_coord, food_value) ? 1 : 0);
    }
    else {
        tile_counts.total_tiles += path_tile_adjustment(next_node);
    }
    return tile_counts;
}

TileCounts SearchNode::tile_count_on_enter()
{
    TileCounts tile_counts;
    if (first_visit())
    {
        int new_tiles = node->get_countable_tiles();
        tile_counts.new_tiles = new_tiles;
        tile_counts.total_food = food_until_here + node->food_count;
        tile_counts.total_tiles = tiles_until_here + new_tiles;
    }
    else
    {
        tile_counts.new_tiles = 0;
        tile_counts.total_food = food_until_here;
        tile_counts.total_tiles = tiles_until_here;
    }
    if (!entered_from_nodes.empty()){
        auto connection_info = node->get_connection_info(entered_from_nodes.back());
        if (connection_info.is_bad_gateway_from_here)
        // we are actually coming in to the current node, but the naming is from the perspective of insdide the node.
        {
            DEBUG_PRINT(std::cout << "Reducing new tile count by 1 because of bad gateway from here" << std::endl;);
            tile_counts.new_tiles -= 1;
        }

    }
    return tile_counts;
}

std::pair<AreaNode *, unsigned int> SearchNode::get_next_node_and_edge()
{
    for (auto &edge_node : node->edge_nodes)
    {
        if (!is_used_edge(edge_node.second) && !is_searched_edge(edge_node.second))
        {
            return edge_node;
        }
    }
    return std::make_pair(nullptr, 0);
}

void SearchNode::enter_from(AreaNode *prev_node, int tiles, int food)
{
    auto edge = node->get_edge(prev_node);
    if (edge != 0)
    {
        add_used_edge(edge);
        entered_from_nodes.push_back(prev_node->id);
    }
    if (prev_node != nullptr){
        auto connection_info = node->get_connection_info(prev_node->id);
        used_coords.push_back(connection_info.self_coord);
    }
    nr_visits++;
    tiles_until_here = tiles;
    food_until_here = food;
    if (first_visit()){
        first_tiles_until_here = tiles;
        first_food_until_here = food;
    }
    searched_edges.resize(nr_visits);
}

void SearchNode::enter_unwind(int tiles, int food)
{
    tiles_until_here = tiles;
    food_until_here = food;
    used_edges.pop_back();
    used_coords.pop_back();
}   

void SearchNode::exit_to(AreaNode *next_node){
    auto connection_info = node->get_connection_info(next_node->id);
    DEBUG_PRINT(std::cout << "Going to node: " << next_node->id << " through coord: (" << connection_info.self_coord.x << ", " << connection_info.self_coord.y << ")" << std::endl;);
    used_coords.push_back(connection_info.self_coord);
}

void SearchNode::exit_unwind()
{
    searched_edges.pop_back();
    used_edges.pop_back();
    used_coords.pop_back();
    entered_from_nodes.pop_back();
    nr_visits--;
    if (nr_visits == 0)
    {
        reset();
    }
}