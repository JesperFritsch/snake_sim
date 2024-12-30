import grpc
import json
from snake_sim.protobuf import remote_snake_pb2, remote_snake_pb2_grpc
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.utils import Coord

class RemoteSnake(ISnake):
    def __init__(self, id: int, start_length: int, target: str):
        self.id = id
        self.start_length = start_length
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

    def set_init_data(self, env_data: dict):
        env_data_json = "empty"
        env_data_proto = remote_snake_pb2.EnvData(data=env_data_json)
        self.stub.SetInitData(env_data_proto)

    def update(self, env_data: dict) -> Coord:
        env_data_json = "empty"
        env_data_proto = remote_snake_pb2.EnvData(data=env_data_json)
        response_iterator = self.stub.Update(iter([env_data_proto]))
        for response in response_iterator:
            return Coord(x=response.direction.x, y=response.direction.y)
