import json
from snake_sim.snake_env import RunData


run_file = r"B:\pythonStuff\snake_sim\snake_sim\runs\grid_10x10\4_snakes_10x10_D3IR9B_61.json"

with open(run_file) as file:
    run_data_dict = json.load(file)
    run_data = RunData.from_dict(run_data_dict)

run_data.write_to_file(filepath="test_run_data.json")
