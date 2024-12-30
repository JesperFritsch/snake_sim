from concurrent import futures
import grpc
import random
import json
from snake_sim.protobuf import remote_snake_pb2, remote_snake_pb2_grpc
from snake_sim.utils import Coord

class RemoteSnakeServicer(remote_snake_pb2_grpc.RemoteSnakeServicer):
    def __init__(self):
        self.snake_id = 0
        self.snake_length = 0
        self.start_position = None
        self.env_data = {}

    def GetId(self, request, context):
        print(f"GetId called. Returning snake_id: {self.snake_id}")
        return remote_snake_pb2.SnakeId(id=self.snake_id)

    def SetId(self, request, context):
        self.snake_id = request.id
        print(f"SetId called. Set snake_id to: {self.snake_id}")
        return remote_snake_pb2.Empty()

    def GetLength(self, request, context):
        print(f"GetLength called. Returning snake_length: {self.snake_length}")
        return remote_snake_pb2.SnakeLength(length=self.snake_length)

    def SetLength(self, request, context):
        self.snake_length = request.length
        print(f"SetLength called. Set snake_length to: {self.snake_length}")
        return remote_snake_pb2.Empty()

    def SetStartPosition(self, request, context):
        self.start_position = Coord(x=request.start_position.x, y=request.start_position.y)
        print(f"SetStartPosition called. Set start_position to: {self.start_position}")
        return remote_snake_pb2.Empty()

    def SetInitData(self, request, context):
        self.env_data = request.data
        print(f"SetInitData called. Set env_data to: {self.env_data}")
        return remote_snake_pb2.Empty()

    def Update(self, request_iterator, context):
        directions = [Coord(x=1, y=0), Coord(x=-1, y=0), Coord(x=0, y=1), Coord(x=0, y=-1)]
        for env_data in request_iterator:
            self.env_data = env_data.data
            print(f"Update called. Set env_data to: {self.env_data}")
            direction = Coord(x=1, y=0)  # Example logic to determine direction
            print(f"Update called. Returning direction: {direction}")
            yield remote_snake_pb2.UpdateResponse(direction=remote_snake_pb2.Coord(x=direction.x, y=direction.y))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    remote_snake_pb2_grpc.add_RemoteSnakeServicer_to_server(RemoteSnakeServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
