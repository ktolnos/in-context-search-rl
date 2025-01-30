from configs.config import Config
from typing import List, Tuple, Dict, Any, DefaultDict, Union
import numpy as np
import torch
import d4rl
import gym
from tqdm import trange
from collections import defaultdict


def load_d4rl_trajectories(
    env_name: str, gamma: float = 0.99
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "reward_mean": dataset["rewards"].mean(0, keepdims=True),
        "reward_std": dataset["rewards"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_and_preprocess(cfg: Config):
    # Load the dataset
    trajs, infos = load_d4rl_trajectories(cfg.env, cfg.gamma)

    indices_to_remove = []
    for ind, traj in enumerate(trajs):
        if (len(traj['observations'])) < cfg.rollout_length + 1:
            indices_to_remove.append(ind)
    for idx in reversed(indices_to_remove):
        del trajs[idx]

    obs_mean = infos['obs_mean']
    obs_std = infos['obs_std']
    rewards_mean = infos['reward_mean']
    rewards_std = infos['reward_std']

    for traj in trajs:
        traj['observations'] = (np.array(traj['observations'], dtype=np.float32) - obs_mean) / obs_std
        traj['rewards'] = (np.array(traj['rewards'], dtype=np.float32) - rewards_mean) / rewards_std
        traj['actions'] = np.array(traj['actions'], dtype=np.float32)

    return trajs, infos

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env