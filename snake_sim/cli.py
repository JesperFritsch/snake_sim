import sys
import argparse
import json

from snake_sim.utils import DotDict
from pathlib import Path


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be a positive integer.")
    return ivalue

class EnsureDirAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = Path(values)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        setattr(namespace, self.dest, path)

def add_common_arguments(parser):
    parser.add_argument('--sound', action='store_true', help='Play sound')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output', default=False)
    parser.add_argument('--no-record', action='store_true', help='Do not record the run', default=False)
    parser.add_argument('--record-file', type=EnsureDirAction,  help='Path to resulting record file', default=Path(__file__).parent / 'runs')


def add_run_config_arguments(parser):
    parser.add_argument('--snake-count', type=int, help='Number of snakes to simulate')
    parser.add_argument('--grid-width', type=int, help='Width of the grid')
    parser.add_argument('--grid-height', type=int, help='Height of the grid')
    parser.add_argument('--food', type=int, help='Number of food to spawn')
    parser.add_argument('--food-decay', type=int, help='Number of steps before food decays, 0 for no decay')
    parser.add_argument('--calc-timeout', type=int, help='Timeout for calculation')
    parser.add_argument('--map', type=str, help='Path to map file')


def add_playback_arguments(parser):
    parser.add_argument('--fps', type=int, help='Frames per second')


def handle_args(args, config: DotDict):
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)


def cli(argv, config: DotDict):
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(dest='command', required=True)

    # Subparser for play-file
    play_file_parser = subparsers.add_parser('play-file', help='Play a saved run file')
    play_file_parser.add_argument('filepath', type=str, help='Path to the run file')
    add_common_arguments(play_file_parser)
    add_playback_arguments(play_file_parser)

    # Subparser for compute
    compute_parser = subparsers.add_parser('compute', help='Compute a run file')
    compute_parser.add_argument('--nr-runs', type=positive_int, default=1, help='Number of runs to generate')
    add_common_arguments(compute_parser)
    add_run_config_arguments(compute_parser)

    # Subparser for stream
    stream_parser = subparsers.add_parser('stream', help='Compute and live-stream the run')
    add_common_arguments(stream_parser)
    add_run_config_arguments(stream_parser)
    add_playback_arguments(stream_parser)

    # Subparser for game
    game_parser = subparsers.add_parser('game', help='Play the game')
    game_parser.add_argument('--num-players', type=positive_int, help='Number of players', required=True)
    game_parser.add_argument('--spm', type=positive_int, help='Steps per minute', default=450)
    add_common_arguments(game_parser)
    add_run_config_arguments(game_parser)

    args = ap.parse_args(argv)
    handle_args(args, config)
    return config

if __name__ == "__main__":
    argv = sys.argv[1:]
    cfg_path = Path(__file__).parent / 'config/default_config.json'
    with open(cfg_path) as config_file:
        config = DotDict(json.load(config_file))
    cli(argv, config)
    import pprint
    pprint.pprint(config)