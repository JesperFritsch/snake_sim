import grpc
import signal
import argparse
import sys

from pathlib import Path
from concurrent import futures
from threading import Event
from typing import Optional

from snake_proto_template.python import remote_snake_pb2, remote_snake_pb2_grpc
from snake_sim.utils import Coord
from snake_sim.snakes.snake import ISnake
from snake_sim.environment.snake_env import EnvInitData, EnvData

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
        init_data = EnvInitData(
            request.height,
            request.width,
            request.free_value,
            request.blocked_value,
            request.food_value,
            {int(k): {"head_value": v.head_value, "body_value": v.body_value} for k, v in request.snake_values.items()},
            {int(k): Coord(x=v.x, y=v.y) for k, v in request.start_positions.items()},
            request.base_map
        )
        self._snake_instance.set_init_data(init_data)
        return remote_snake_pb2.Empty()

    def Update(self, request_iterator, context):
        for env_data_proto in request_iterator:
            env_data = EnvData(
                map=env_data_proto.map,
                snakes={k: {"is_alive": v.is_alive, "length": v.length} for k, v in env_data_proto.snakes.items()}
            )
            direction = self._snake_instance.update(env_data)
            yield remote_snake_pb2.UpdateResponse(direction=remote_snake_pb2.Coord(x=direction.x, y=direction.y))


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
    parser.add_argument('-m', '--snake_module_file', type=str, required=True, default=None, help='Path to snake module for importing snake class')
    args = parser.parse_args(argv)
    return args


def serve(target, snake_module_file, stop_event: Optional[Event] = None):
    if not stop_event:
        stop_event = Event()
    def handle_term(signum, frame):
        stop_event.set()
    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)
    try:
        snake_module = import_snake_module(snake_module_file)
        snake_instance = snake_module.AutoSnake()
        snake_servicer = RemoteSnakeServicer(snake_instance)

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        remote_snake_pb2_grpc.add_RemoteSnakeServicer_to_server(snake_servicer, server)
        server.add_insecure_port(target)
        server.start()
        if stop_event:
            stop_event.wait()
        else:
            server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in server: {e}")


if __name__ == '__main__':
    args = cli(sys.argv[1:])
    serve(args.target, args.snake_module_file, log_level=args.log_level)
