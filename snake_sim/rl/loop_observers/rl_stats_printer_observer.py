from snake_sim.loop_observers.consumer_observer import ConsumerObserver
from snake_sim.environment.types import LoopStepData, LoopStartData, LoopStopData


class RLStatsPrinterObserver(ConsumerObserver):

    def __init__(self):
        self._steps_this_episode = {}
        self._total_agent_steps = 0
        self._total_episodes = 0
        super().__init__()

    def reset(self):
        self._steps_this_episode = {}
        return super().reset()

    def notify_step(self, step_data: LoopStepData):
        super().notify_step(step_data)
        for snake_id, is_alive in step_data.alive_states.items():
            if is_alive:
                self._steps_this_episode[snake_id] = step_data.step
    
    def notify_stop(self, stop_data: LoopStopData):
        super().notify_stop(stop_data)
        self._total_agent_steps += sum(self._steps_this_episode.values())
        self._total_episodes += 1
        print("RL Stats Printer Observer - Final Stats:")
        last_step = self._steps[-1]
        for snake_id in sorted(self._steps_this_episode.keys()):
            last_alive_step = self._steps_this_episode[snake_id]
            length = last_step.lengths.get(snake_id, 0)
            print(f"  Snake ID {snake_id}: Last Alive Step: {last_alive_step}, Final Length: {length}")
        print(f"  Total Episodes: {self._total_episodes}")
        print(f"  Total Agent Steps: {self._total_agent_steps}")