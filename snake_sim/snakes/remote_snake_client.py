# grpc client to connect to the remote snake server and get the next move by calling the update method 
import grpc
from snake_sim.protobuf import sim_msgs_pb2, sim_msgs_pb2_grpc
import snake
from time import time
from snake_sim.utils import coord_op

class RemoteSnakeClient(snake.Snake):
    def __init__(self, id: str, start_length: int):
        super().__init__(id, start_length)
        self.channel = grpc.insecure_channel('localhost:50051')
        self.stub = sim_msgs_pb2_grpc.RemoteSnakeServiceStub(self.channel)
    
    def _pick_direction(self):
        response = self.stub.update(sim_msgs_pb2.EnvData(width=self.env_data.width, height=self.env_data.height, map=self.env_data.map, snakes=self.env_data.snakes, food=self.env_data.food, food_decay=self.env_data.food_decay))
        return response.action

    def update(self, env_data: dict):
        self.start_time = time()
        self.set_env_data(env_data)
        self.update_map(self.env_data.map)
        next_tile = self._pick_direction()
        if next_tile is None:
            next_tile = self.coord
        return coord_op(next_tile, self.coord, '-')
    
        
        
        
        
