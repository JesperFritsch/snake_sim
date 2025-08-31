import grpc
import json
from snake_proto_template.python import remote_snake_pb2, remote_snake_pb2_grpc
from snake_sim.environment.interfaces.snake_interface import ISnake
from snake_sim.environment.types import Coord, EnvInitData, EnvData


def handle_connection_loss(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except grpc.RpcError as e:
            raise ConnectionError
    return wrapper


class RemoteSnake(ISnake):
    def __init__(self, target: str):
        self.target = target
        self.channel = grpc.insecure_channel(target)
        self.stub = remote_snake_pb2_grpc.RemoteSnakeStub(self.channel)

    @handle_connection_loss
    def kill(self):
        self.stub.Kill(remote_snake_pb2.Empty())
        self.channel.close()

    @handle_connection_loss
    def set_id(self, id: int):
        self.stub.SetId(remote_snake_pb2.SnakeId(id=id))

    @handle_connection_loss
    def set_start_length(self, start_length: int):
        self.stub.SetStartLength(remote_snake_pb2.StartLength(length=start_length))

    @handle_connection_loss
    def set_start_position(self, start_position: Coord):
        start_pos = remote_snake_pb2.StartPosition(start_position=remote_snake_pb2.Coord(x=start_position.x, y=start_position.y))
        self.stub.SetStartPosition(start_pos)

    @handle_connection_loss
    def set_init_data(self, env_init_data: EnvInitData):
        env_init_data_proto = remote_snake_pb2.EnvInitData(
            height=env_init_data.height,
            width=env_init_data.width,
            free_value=env_init_data.free_value,
            blocked_value=env_init_data.blocked_value,
            food_value=env_init_data.food_value,
            snake_values={k: remote_snake_pb2.SnakeValues(head_value=v["head_value"], body_value=v["body_value"]) for k, v in env_init_data.snake_values.items()},
            start_positions={k: remote_snake_pb2.Coord(x=v.x, y=v.y) for k, v in env_init_data.start_positions.items()},
            base_map=env_init_data.base_map.tobytes()
        )
        self.stub.SetInitData(env_init_data_proto)

    @handle_connection_loss
    def update(self, env_data: EnvData):
        # print(f"{self.target}: Updating")
        env_data_proto = remote_snake_pb2.EnvData(
            map=env_data.map,
            snakes={k: remote_snake_pb2.SnakeRep(is_alive=v["is_alive"], length=v["length"]) for k, v in env_data.snakes.items()}
        )
        response_iterator = self.stub.Update(iter([env_data_proto]))
        for response in response_iterator:
            return Coord(x=response.direction.x, y=response.direction.y)


    def __reduce__(self):
        return (self.__class__, (self.target))
