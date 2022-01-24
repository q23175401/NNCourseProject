from collections import deque, namedtuple
import random


# create a named tuple to store each trajectory data
OneTrajectoryData = namedtuple('OneTrajectoryData', ['cur_state', 'action', 'reward', 'next_state', 'done'])

class MyReplayBuffer():
    def __init__(self, max_len):
        self.max_len = max_len
        self.buffer = deque(maxlen=max_len)

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def push_trajectory(self, one_trajectory_data:OneTrajectoryData):
        self.buffer.append(one_trajectory_data)

    def __len__(self):
        return len(self.buffer)