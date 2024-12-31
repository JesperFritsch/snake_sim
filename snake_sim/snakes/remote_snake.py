import grpc
import json
from snake_sim.protobuf import remote_snake_pb2, remote_snake_pb2_grpc
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.utils import Coord
from snake_sim.environment.snake_env import EnvInitData, EnvData

class RemoteSnake(ISnake):
    def __init__(self, id: int, start_length: int, target: str):
        self.id = id
        self.start_length = start_length
        self.target = target
        self.channel = grpc.insecure_channel(target)
        self.stub = remote_snake_pb2_grpc.RemoteSnakeStub(self.channel)
        self.set_id(id)
        self.set_length(start_length)

    def get_id(self) -> int:
        response = self.stub.GetId(remote_snake_pb2.Empty())
        return response.id

    def set_id(self, id: int):
        snake_id = remote_snake_pb2.SnakeId(id=id)
        self.stub.SetId(snake_id)

    def get_length(self) -> int:
        snake_id = remote_snake_pb2.SnakeId(id=self.id)
        response = self.stub.GetLength(snake_id)
        return response.length

    def set_length(self, length: int):
        snake_length = remote_snake_pb2.SnakeLength(length=length)
        self.stub.SetLength(snake_length)

    def set_start_position(self, start_position: Coord):
        start_pos = remote_snake_pb2.StartPosition(start_position=remote_snake_pb2.Coord(x=start_position.x, y=start_position.y))
        self.stub.SetStartPosition(start_pos)

    def set_init_data(self, env_data: EnvInitData):
        env_data_proto = remote_snake_pb2.EnvInitData(
            height=env_data.height,
            width=env_data.width,
            free_value=env_data.free_value,
            blocked_value=env_data.blocked_value,
            food_value=env_data.food_value,
            snake_values={k: remote_snake_pb2.SnakeValues(head_value=v["head_value"], body_value=v["body_value"]) for k, v in env_data.snake_values.items()},
            start_positions={k: remote_snake_pb2.Coord(x=v.x, y=v.y) for k, v in env_data.start_positions.items()},
            base_map=env_data.base_map.tobytes()
        )
        self.stub.SetInitData(env_data_proto)

    def update(self, env_data: EnvData) -> Coord:
        env_data_proto = remote_snake_pb2.EnvData(
            map=env_data.map,
            snakes={k: remote_snake_pb2.SnakeRep(is_alive=v["is_alive"], length=v["length"]) for k, v in env_data.snakes.items()}
        )
        response_iterator = self.stub.Update(iter([env_data_proto]))
        for response in response_iterator:
            return Coord(x=response.direction.x, y=response.direction.y)

    def __reduce__(self):
        return (self.__class__, (self.id, self.start_length, self.target))
