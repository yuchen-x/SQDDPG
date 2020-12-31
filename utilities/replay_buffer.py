import numpy as np
from itertools import chain

from utilities.util import Transition


class TransReplayBuffer(object):

    def __init__(self, size):
        self.size = size
        self.buffer = []

    def get_single(self, index):
        return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)

    def get_batch(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size, replace=False)
        batch_buffer = [self.buffer[i] for i in indices]
        return batch_buffer

    def add_experience(self, trans):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(trans)

    def clear(self):
        self.buffer = []



class EpisodeReplayBuffer(object):

    def __init__(self, size, env_info):
        self.size = size
        self.buffer = []

        self.PAD_STATE = [np.zeros(env_info['state_shape']) for _ in range(env_info['n_agents'])]
        self.PAD_OBS = [np.zeros(env_info['obs_shape']) for _ in range(env_info['n_agents'])]
        self.PAD_ACTION = np.zeros((1, env_info['n_agents'], env_info['n_actions'])) 
        self.PAD_REWARD = np.zeros(env_info['n_agents'])
        self.PAD_DONE = 0 
        self.PAD_LAST_STEP = 0
        self.PAD_VALID = np.zeros(env_info['n_agents'])

        self.PAD_TRANS = Transition(self.PAD_STATE,
                                    self.PAD_OBS,
                                    self.PAD_ACTION,
                                    self.PAD_REWARD,
                                    self.PAD_STATE,
                                    self.PAD_OBS,
                                    self.PAD_DONE,
                                    self.PAD_LAST_STEP,
                                    self.PAD_VALID)

    # def get_single(self, index):
    #     return self.buffer[index]

    def offset(self):
        self.buffer.pop(0)

    def add_episode(self, episode):
        est_len = 1 + len(self.buffer)
        if est_len > self.size:
            self.offset()
        self.buffer.append(episode)

    def get_batch(self, batch_size):
        length = len(self.buffer)
        indices = np.random.choice(length, batch_size, replace=False)
        batch_buffer = []
        for i in indices:
            batch_buffer.append(self.buffer[i])
        return self.pad_batch(batch_buffer)

    def pad_batch(self, batch):
        epi_lens = [len(epi) for epi in batch]
        max_len = max(epi_lens)
        batch = [epi + [self.PAD_TRANS] * (max_len - len(epi)) for epi in batch]
        return list(chain(*batch)), max_len


