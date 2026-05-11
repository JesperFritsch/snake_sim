import os

os.environ["GRPC_VERBOSITY"] = "ERROR"

import grpc
import argparse
import sys
import time
import json
import numpy as np

from pathlib import Path
from importlib import resources
from concurrent import futures
from typing import Optional
from multiprocessing.sharedctypes import Synchronized

from snake_proto_template.python import remote_snake_pb2, remote_snake_pb2_grpc
from snake_sim.snakes.snake_base import ISnake
from snake_sim.environment.types import Coord, EnvMetaData, EnvStepData, SnakeConfig
from snake_sim.environment.snake_factory import SnakeFactory
from snake_sim.logging_setup import setup_logging

import logging

class RemoteSnakeServicer(remote_snake_pb2_grpc.RemoteSnakeServicer):
    def __init__(self, snake_instance: ISnake):
        self._snake_instance = snake_instance

    def SetId(self, request, context):
        self._snake_instance.set_id(request.id)
        return remote_snake_pb2.Empty()

    def SetStartLength(self, request, context):
        self._snake_instance.set_start_length(request.length)
        return remote_snake_pb2.Empty()

    def SetStartPosition(self, request, context):
        start_coord = Coord(x=request.start_position.x, y=request.start_position.y)
        self._snake_instance.set_start_position(start_coord)
        return remote_snake_pb2.Empty()

    def SetInitData(self, request, context):
        init_data = EnvMetaData(
            request.height,
            request.width,
            request.free_value,
            request.blocked_value,
            request.food_value,
            {int(k): {"head_value": v.head_value, "body_value": v.body_value} for k, v in request.snake_values.items()},
            {int(k): Coord(x=v.x, y=v.y) for k, v in request.start_positions.items()},
            np.frombuffer(request.base_map, dtype=np.dtype(request.base_map_dtype)).reshape(request.height, request.width),
            request.base_map_dtype
        )
        self._snake_instance.set_init_data(init_data)
        return remote_snake_pb2.Empty()

    def Update(self, request_iterator, context):
        for env_step_data_proto in request_iterator:
            env_step_data = EnvStepData(
                map=np.frombuffer(env_step_data_proto.map, dtype=self._snake_instance._env_meta_data.base_map_dtype).reshape(self._snake_instance._env_meta_data.height, self._snake_instance._env_meta_data.width),
                snakes={k: {"is_alive": v.is_alive, "length": v.length} for k, v in env_step_data_proto.snakes.items()},
                food_locations=[Coord(x=coord.x, y=coord.y) for coord in env_step_data_proto.food_locations]
            )
            direction = self._snake_instance.update(env_step_data)
            if direction is None:
                yield remote_snake_pb2.UpdateResponse()
            else:
                yield remote_snake_pb2.UpdateResponse(direction=remote_snake_pb2.Coord(x=direction.x, y=direction.y))
    
    def Kill(self, request, context):
        self._snake_instance.kill()
        return remote_snake_pb2.Empty()

    def Reset(self, request, context):
        self._snake_instance.reset()
        return remote_snake_pb2.Empty()


def import_snake_module(snake_module_file):
    if snake_module_file:
        snake_file_path = Path(snake_module_file)
        sys.path.append(str(snake_file_path.parent))
        import importlib.util
        spec = importlib.util.spec_from_file_location("snake_module", snake_module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def cli(argv):
    parser = argparse.ArgumentParser("Remote Snake Server")
    parser.add_argument('-t', '--target', type=str, required=True, help='Server address or socket path to bind to')
    m_group = parser.add_mutually_exclusive_group(required=True)
    m_group.add_argument('-c', '--snake-config-name', type=str, help='Name of snake config to use from default configs.')
    m_group.add_argument('-m', '--snake-module-file', type=str, help='Path to snake module for importing snake class')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args(argv)
    return args


def serve(
        target,
        snake_module_file=None,
        snake_config_name=None,
        snake_config: SnakeConfig=None,
        stop_flag: Optional[Synchronized] = None,
        log_level=logging.INFO):
    # set up logging if on Windows
    if not logging.getLogger().hasHandlers():
        setup_logging(log_level)


    log = logging.getLogger(f"{target}")

    try:
        if not bool(snake_module_file) ^ bool(snake_config) and not snake_config_name:
            raise ValueError("Either snake_module_file or snake_config must be provided, but not both and not neither")

        if snake_module_file:
            snake_module = import_snake_module(snake_module_file)
            snake_instance = snake_module.MySnake()
        else:
            if snake_config_name:
                with resources.open_text('snake_sim.config', 'default_config.json') as config_file:
                    default_config = json.load(config_file)
                    snake_config = SnakeConfig.from_dict(default_config[snake_config_name])
            factory = SnakeFactory()
            snake_instance = factory.create_snake(
                snake_config=snake_config
            )

        snake_servicer = RemoteSnakeServicer(snake_instance)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        remote_snake_pb2_grpc.add_RemoteSnakeServicer_to_server(snake_servicer, server)
        server.add_insecure_port(target)
        server.start()
        if stop_flag is not None:
            while not stop_flag.value:
                try:
                    time.sleep(0.01)
                except:
                    pass
        else:
            server.wait_for_termination()

    except Exception as e:
        log.error(e)
        log.debug("TRACE: ", exc_info=True)
    finally:
        server.stop(0)


if __name__ == '__main__':
    args = cli(sys.argv[1:])
    setup_logging(args.log_level)
    serve(
        args.target, 
        args.snake_module_file, 
        snake_config_name=args.snake_config_name,
        log_level=args.log_level
    )
