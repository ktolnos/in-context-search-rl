import numpy as np
import torch

class RolloutDataset(torch.utils.data.Dataset):

    def __init__(self, trajs, cfg):
        self.rollout_length = cfg.rollout_length
        self.trajs = trajs
        self.traj_lens = np.array([traj['observations'].shape[0] for traj in trajs])

        self.len = sum(self.traj_lens) - (self.rollout_length + 1) * len(trajs)
        self.index_to_traj = np.concatenate(
            [np.full(traj_len - self.rollout_length - 1, i) for i, traj_len in enumerate(self.traj_lens)])
        self.index_to_traj_ind = np.concatenate(
            [np.arange(0, traj_len - self.rollout_length - 1) for traj_len in self.traj_lens])

        assert len(self.index_to_traj) == self.len, f"{len(self.index_to_traj)} != {self.len}"
        assert len(self.index_to_traj_ind) == self.len, f"{len(self.index_to_traj_ind)} != {self.len}"

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        traj = self.trajs[self.index_to_traj[idx]]
        ind = self.index_to_traj_ind[idx]
        indices = np.arange(ind, ind + self.rollout_length)
        return {
            'observations': traj['observations'][indices],
            'actions': traj['actions'][indices],
            'next_observations': traj['observations'][indices + 1],
            'rewards': traj['rewards'][indices],
        }