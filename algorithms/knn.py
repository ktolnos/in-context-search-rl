import tqdm
import sys
import os

from configs.config import Config
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm, trange
import multiprocessing
from algorithms.data import load_and_preprocess, wrap_env, discounted_cumsum
from algorithms.nets import Encoder, ForwardModel, InvModel, RewardModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from algorithms.rollout_dataset import RolloutDataset
from pynndescent import NNDescent
import matplotlib.pyplot as plt


def run_knn(cfg: Config):
    global_step = 0

    env = gym.make(cfg.env)
    torch.set_default_device('cuda')

    trajs, infos = load_and_preprocess(cfg)
    obs_shape = trajs[0]['observations'].shape[1]
    action_shape=trajs[0]['actions'].shape[1]
    hidden_dim = cfg.hidden_dim
    latent_dim = cfg.latent_dim
    encoder = Encoder(obs_shape=obs_shape, hidden_dim=hidden_dim, latent_dim=latent_dim)
    target_encoder = Encoder(obs_shape=obs_shape, hidden_dim=hidden_dim, latent_dim=latent_dim)
    inv_model = InvModel(latent_dim=latent_dim, hidden_dim=hidden_dim, action_shape=action_shape)
    forward_model = ForwardModel(latent_dim=latent_dim, hidden_dim=hidden_dim, action_shape=action_shape)
    reward_model = RewardModel(latent_dim=latent_dim, hidden_dim=hidden_dim, action_shape=action_shape)


    batch_size = cfg.batch_size
    grad_clip_norm = cfg.grad_clip_norm
    train_size = int(0.8 * len(trajs))
    indices = np.random.permutation(len(trajs))
    trajs_np = np.array(trajs)
    train_idx, val_idx = indices[:train_size], indices[train_size:]
    dataset, val_set = RolloutDataset(trajs_np[train_idx], cfg), RolloutDataset(trajs_np[val_idx], cfg)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    val_set_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, generator=torch.Generator(device='cuda'))

    optimizer = AdamW(list(encoder.parameters()) + list(inv_model.parameters()) + list(forward_model.parameters()) + list(reward_model.parameters()), lr=1e-3)

    torch.autograd.set_detect_anomaly(False)
    forward_model.train()
    reward_model.train()
    encoder.train()
    inv_model.train()
    target_encoder.eval()

    model_loss_fn = mse_loss

    def train_step(batch):
        obs = batch['observations']
        act = batch['actions']
        next_obs = batch['next_observations']
        rewards = batch['rewards']

        z_enc = encoder(obs)  # (batch, seq_len, latent_dim)
        with torch.no_grad():
            z_next_tar = target_encoder(next_obs)  # (batch, seq_len, latent_dim)

        z_next = torch.zeros_like(z_next_tar)
        z = z_enc[:, 0]
        action_pred = torch.zeros_like(act)  # (batch, seq_len, action_dim)
        for i in range(0, obs.shape[1]):
            z_pred = forward_model(z, act[:, i])  # (batch, latent_dim)
            action_offset = torch.randint(0, z_next_tar.shape[1] - i, (z_next_tar.shape[0],))  # (batch,)
            batch_indices = torch.arange(z_next_tar.shape[0], device=z_next_tar.device)
            selected_indices = i + action_offset
            action_pred[:, i] = inv_model(torch.cat([z_enc[:, i], z_next_tar[batch_indices, selected_indices]], dim=-1))
            z_next[:, i] = z_pred
            z = z_pred

        model_loss = model_loss_fn(z_next, z_next_tar)
        action_loss = F.mse_loss(action_pred, act)
        reward_pred = reward_model(z_enc, act, z_next_tar).squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, rewards)
        loss = model_loss + action_loss + reward_loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(inv_model.parameters(), max_norm=grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(forward_model.parameters(), max_norm=grad_clip_norm)
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        # Update target encoder with exponential moving average
        for param, target_param in zip(encoder.parameters(), target_encoder.parameters()):
            target_param.data = target_param.data * (1 - cfg.target_ema) + param.data * cfg.target_ema

        return {
            'loss': loss.item(),
            'model_loss': model_loss.item(),
            'action_loss': action_loss.item(),
            'reward_loss': reward_loss.item(),
        }

    next_state_distance = 2

    def build_index():
        num_obs = sum([len(traj["observations"]) for traj in trajs]) - next_state_distance * len(trajs)

        obs_norm = np.zeros((num_obs, latent_dim), dtype=np.float32)
        z_next = np.zeros((num_obs, latent_dim), dtype=np.float32)

        returns = np.zeros(num_obs, dtype=np.float32)
        actions = np.zeros((num_obs, trajs[0]['actions'].shape[1]), dtype=np.float32)

        ind = 0
        for traj in tqdm(trajs):
            indices = slice(ind, ind + traj['observations'].shape[0] - next_state_distance)
            obs = traj['observations']
            z = encoder(torch.tensor(obs, dtype=torch.float32)).detach().cpu().numpy()
            obs_norm[indices] = z[:-next_state_distance]
            z_next[indices] = z[next_state_distance:]
            returns[indices] = traj['returns'][:-next_state_distance]
            actions[indices] = traj['actions'][:-next_state_distance]
            ind += traj['observations'].shape[0] - next_state_distance

        index = NNDescent(obs_norm, metric='euclidean')
        return index, z_next, returns, actions


    for epoch in range(cfg.epochs):
        print(f"Epoch {epoch}")
        wandb.log({"epoch": epoch}, step=global_step)
        for batch in tqdm(dataloader):
            global_step += len(batch['observations'])
            metrics = train_step(batch)
            wandb.log(metrics, step=global_step)

    index, index_z_next, index_returns, index_actions = build_index()
    finetune_env = wrap_env(
        env=gym.make(cfg.env),
        state_mean=infos['obs_mean'],
        state_std=infos['obs_std'],
        reward_scale=1.0,
    )
    finetune_episodes = max(cfg.eval_episodes)+1
    finetune_z = []
    finetune_z_next = []
    finetune_actions = []
    finetune_returns = []
    reward_mean = infos['reward_mean']
    reward_std = infos['reward_std']
    k = cfg.k
    c = 1

    knn_temps = np.array(cfg.knn_temp)
    temp_counts = np.zeros_like(knn_temps)
    temp_returns = np.zeros_like(knn_temps)

    def rollout_episode(env, eval=True):
        if eval:
            if np.any(temp_counts == 0):
                temp = 1
            else:
                temp = knn_temps[np.argmax(temp_returns / temp_counts)]
        else:
            # select knn_temp using UCB
            if np.any(temp_counts == 0):
                temp_idx = np.random.choice(len(knn_temps))
                temp = knn_temps[temp_idx]
            else:
                temp_avg = temp_returns / temp_counts
                temp_rating = temp_avg + c * np.sqrt(2 * np.log(np.sum(temp_counts)) / temp_counts)
                temp_idx = np.argmax(temp_rating)
                temp = knn_temps[temp_idx]
            temp_counts[temp_idx] += 1

        rollout_z = []
        rollout_actions = []
        rollout_rewards = []

        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            z = encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).detach().cpu().numpy()[0]
            rollout_z.append(z)
            idxs, dists = index.query(z, k=k)
            neighbor_rets = index_returns[idxs]
            ranks = neighbor_rets * np.exp(-temp * dists)
            best_rank_idx = np.argmax(ranks)
            best_idx = idxs[0, best_rank_idx]
            z_next_best = index_z_next[best_idx][None, :]
            best_rank = ranks[0, best_rank_idx]

            for i, ret in enumerate(finetune_returns):
                rank = ret * np.exp(-temp * np.linalg.norm(z - finetune_returns[i]))
                if rank > best_rank:
                    best_rank = rank
                    z_next_best = finetune_z_next[i]

            action = inv_model(torch.tensor(np.concatenate([z, z_next_best], axis=-1),
                                            dtype=torch.float32).unsqueeze(0)).detach().cpu().numpy()
            rollout_actions.append(action)
            obs, reward, done, _ = env.step(action.squeeze())
            rollout_rewards.append(reward)
            total_reward += reward

        if not eval:
            temp_returns[temp_idx] += env.get_normalized_score(total_reward)
            wandb.log({f"finetune/best_knn_temp": knn_temps[np.argmax(temp_returns / temp_counts)]}, step=global_step)
        return total_reward, rollout_z, rollout_actions, rollout_rewards

    def evaluate():
        eval_env = wrap_env(
            env=gym.make(cfg.env),
            state_mean=infos['obs_mean'],
            state_std=infos['obs_std'],
            reward_scale=1.0,
        )
        eval_episodes = 100

        eval_rets_inv = [rollout_episode(eval_env, eval=True)[0] for _ in trange(eval_episodes, desc="Evaluation", leave=False)]
        print("Mean return:", np.mean(eval_rets_inv), env.get_normalized_score(np.mean(eval_rets_inv)) * 100.0)

        wandb.log({f"eval/return": np.mean(eval_rets_inv),
                   f"eval/normalized_return": env.get_normalized_score(np.mean(eval_rets_inv)) * 100.0},
                  step=global_step)
        wandb.log({f"eval/returns": eval_rets_inv,
                   f"eval/returns_normalized": [env.get_normalized_score(x) * 100.0 for x in eval_rets_inv]},
                  step=global_step)

    for i in range(finetune_episodes):
        if i in cfg.eval_episodes:
            evaluate()

        fintune_return_new, finetune_z_new, finetune_actions_new, fintune_rewards_new = rollout_episode(finetune_env, eval=False)
        global_step += len(fintune_rewards_new)

        wandb.log({"finetune/return": fintune_return_new}, step=global_step)
        wandb.log({"finetune/normalized_return": finetune_env.get_normalized_score(fintune_return_new) * 100.0}, step=global_step)
        if len(fintune_rewards_new) < next_state_distance:
            continue

        fintune_rewards_new = np.array(fintune_rewards_new)
        fintune_rewards_new = (fintune_rewards_new - reward_mean) / reward_std
        returns = discounted_cumsum(fintune_rewards_new, cfg.gamma)
        finetune_returns.extend(returns[:-next_state_distance])
        finetune_z.extend(finetune_z_new[:-next_state_distance])
        finetune_z_next.extend(finetune_z_new[next_state_distance:])
        finetune_actions.extend(finetune_actions_new[:-next_state_distance])



def cosine_loss(z, z_next):
    return -F.cosine_similarity(z, z_next, dim=-1).mean()

def mse_loss(z, z_next):
    return F.mse_loss(z, z_next)