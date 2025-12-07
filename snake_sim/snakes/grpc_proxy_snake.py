import grpc
import json
from snake_proto_template.python import remote_snake_pb2, remote_snake_pb2_grpc
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvMetaData, EnvStepData

import logging

def handle_connection_loss(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except grpc.RpcError as e:
            raise ConnectionError
    return wrapper


class GRPCProxySnake(ISnake):
    def __init__(self, target: str, timeout: float = 5.0):
        super().__init__()
        self.target = target
        self.timeout = timeout
        self.channel = grpc.insecure_channel(target)
        self.stub = remote_snake_pb2_grpc.RemoteSnakeStub(self.channel)
        self._log = logging.getLogger(f"{self.__class__.__name__}-{target}")
        self._wait_for_connection()

    def _wait_for_connection(self):
        """Wait for the gRPC channel to be ready."""
        try:
            self._log.debug(f"Waiting for connection to {self.target}...")
            grpc.channel_ready_future(self.channel).result(timeout=self.timeout)
            self._log.debug(f"Connected to {self.target}")
        except grpc.FutureTimeoutError:
            self._log.error(f"Timeout waiting for connection to {self.target}")
            raise ConnectionError(f"Failed to connect to {self.target} within {self.timeout}s")

    @handle_connection_loss
    def kill(self):
        super().kill()
        self.stub.Kill(remote_snake_pb2.Empty())
        self.channel.close()

    @handle_connection_loss
    def set_id(self, id: int):
        super().set_id(id)
        self.stub.SetId(remote_snake_pb2.SnakeId(id=id), wait_for_ready=True, timeout=self.timeout)

    @handle_connection_loss
    def set_start_length(self, start_length: int):
        super().set_start_length(start_length)
        self.stub.SetStartLength(remote_snake_pb2.StartLength(length=start_length))

    @handle_connection_loss
    def set_start_position(self, start_position: Coord):
        super().set_start_position(start_position)
        start_pos = remote_snake_pb2.StartPosition(start_position=remote_snake_pb2.Coord(x=start_position.x, y=start_position.y))
        self.stub.SetStartPosition(start_pos)

    @handle_connection_loss
    def set_init_data(self, env_meta_data: EnvMetaData):
        super().set_init_data(env_meta_data)
        env_meta_data_proto = remote_snake_pb2.EnvInitData(
            height=env_meta_data.height,
            width=env_meta_data.width,
            free_value=env_meta_data.free_value,
            blocked_value=env_meta_data.blocked_value,
            food_value=env_meta_data.food_value,
            snake_values={k: remote_snake_pb2.SnakeValues(head_value=v["head_value"], body_value=v["body_value"]) for k, v in env_meta_data.snake_values.items()},
            start_positions={k: remote_snake_pb2.Coord(x=v.x, y=v.y) for k, v in env_meta_data.start_positions.items()},
            base_map=env_meta_data.base_map.tobytes(),
            base_map_dtype=str(env_meta_data.base_map_dtype)
        )
        self.stub.SetInitData(env_meta_data_proto)

    @handle_connection_loss
    def reset(self):
        super().reset()
        self.stub.Reset(remote_snake_pb2.Empty())

    @handle_connection_loss
    def update(self, env_step_data: EnvStepData):
        # print(f"{self.target}: Updating")
        env_step_data_proto = remote_snake_pb2.EnvData(
            map=env_step_data.map.tobytes(),
            snakes={k: remote_snake_pb2.SnakeRep(is_alive=v["is_alive"], length=v["length"]) for k, v in env_step_data.snakes.items()},
            food_locations=[remote_snake_pb2.Coord(x=coord[0], y=coord[1]) for coord in env_step_data.food_locations] if env_step_data.food_locations else []
        )
        response_iterator = self.stub.Update(iter([env_step_data_proto]))
        for response in response_iterator:
            if not response.HasField("direction"):
                return None
            return Coord(x=response.direction.x, y=response.direction.y)


    def __reduce__(self):
        return (self.__class__, (self.target, self.timeout))

    def __del__(self):
        try:
            self.kill()
        except Exception:
            pass
