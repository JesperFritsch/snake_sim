import json
import logging
from pathlib import Path
from typing import Dict
from importlib import resources

from snake_sim.environment.types import CompleteStepState
from snake_sim.utils import rand_str

log = logging.getLogger(Path(__file__).stem)

STATE_CACHE: Dict[int, str] = {}

def get_statefile_dir():
    """ Get the directory where state files are stored. """
    with resources.as_file(resources.files('snake_sim') / '__init__.py') as init_path:
        return Path(init_path).parent / "test_bench" / "state_files"


def clear_state_cache():
    """ Clear the state cache. """
    STATE_CACHE.clear()
    log.info("Cleared state cache")


def clear_state_files():
    """ Clear the state files. """
    for file_path in get_statefile_dir().glob('*'):
        file_path.unlink()


def save_step_state(step_state: CompleteStepState):
    """ Save the step state to a json file. """
    if step_state.state_idx in STATE_CACHE:
        file_path = STATE_CACHE[step_state.state_idx]
        log.info(f"Step state already saved to '{file_path}'")
        return
    file_path = get_statefile_dir() / f"state_{rand_str(10)}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not step_state.state_idx in STATE_CACHE:
        STATE_CACHE[step_state.state_idx] = file_path
        state_dict = step_state.to_dict()
        with open(file_path, 'w') as f:
            json.dump(state_dict, f, indent=4)
        log.info(f"Saved step state to '{file_path}'")


def load_step_state(file_path: Path) -> CompleteStepState:
    """ Load the step state from a json file. """
    if not file_path.exists():
        raise FileNotFoundError(f"State file '{file_path}' does not exist")
    log.info(f"Loading step state from '{file_path}'")
    with open(file_path, 'r') as f:
        state_dict = json.load(f)
    print(state_dict)
    return CompleteStepState.from_dict(state_dict)