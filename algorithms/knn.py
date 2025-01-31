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
from algorithms.data import load_and_preprocess, wrap_env
from algorithms.nets import Encoder, ForwardModel, InvModel, RewardModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from algorithms.rollout_dataset import RolloutDataset
from pynndescent import NNDescent
import matplotlib.pyplot as plt

def run_knn(cfg: Config):
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

    def evaluate(eval_epoch):

        def plot_hists(loader, name):
            model_loss_total = 0
            model_loss_shuffle = 0
            inv_model_loss_total = 0
            inv_model_loss_shuffle = 0
            bins = np.linspace(0, 20, 100, endpoint=True)
            bins = np.concatenate([bins, [10000]])
            hists = dict()
            max_batches = 200
            fig, ax = plt.subplots()
            with torch.no_grad():
                for batch_i, batch in enumerate(loader):
                    z = encoder(batch['observations'])
                    z_next = encoder(batch['next_observations'])
                    z_pred = forward_model(z, batch['actions'])
                    for i in range(z_next.shape[1]):
                        hists[i] = hists.get(i, np.zeros_like(bins[:-1]))
                        dist = torch.norm(z[:, 0] - z_next[:, i], dim=-1)
                        hists[i] += np.histogram(dist.cpu().numpy(), bins=bins)[0]
                    hists["rand"] = hists.get("rand", np.zeros_like(bins[:-1]))
                    dist_rand = torch.norm(z[:, 0] - z_next[torch.randperm(len(z)), 0], dim=-1)
                    hists["rand"] += np.histogram(dist_rand.cpu().numpy(), bins=bins)[0]

                    model_loss_total += model_loss_fn(z_pred, z_next)
                    model_loss_shuffle += model_loss_fn(z_pred, z_next[torch.randperm(len(z))])

                    action_pred = inv_model(torch.cat([z[:, 0], z_next[:, 2]], dim=-1))
                    action_pred_rand = inv_model(torch.cat([z[:, 0], z_next[torch.randperm(len(z)), 2]], dim=-1))
                    inv_model_loss_total += F.mse_loss(action_pred, batch['actions'][:, 0])
                    inv_model_loss_shuffle += F.mse_loss(action_pred_rand, batch['actions'][:, 0])

                    if batch_i > max_batches:
                        break
            print(
                f"Model loss: {model_loss_total / len(loader)}\n"
                f"Model loss (shuffle): {model_loss_shuffle / len(loader)}"
                f"Inv model loss: {inv_model_loss_total / len(loader)}\n"
                f"Inv model loss (shuffle): {inv_model_loss_shuffle / len(loader)}"
            )

            plotting_bins = np.copy(bins)
            plotting_bins[-1] = plotting_bins[-2] + 1
            for key, val in hists.items():
                if key in set(range(5, 1000)):
                    continue
                val /= np.sum(val)
                ax.stairs(val, plotting_bins, label=f"Distance {key}")
            fig.legend()
            ax.set_ylim(0, 0.2)
            wandb.log({f"hist_{name}_{epoch}": wandb.Image(fig)})

        plot_hists(val_set_loader, "val")
        plot_hists(dataloader, "train")

        """
        Preprocess data for NNDescent
        """
        next_state_distance = 2
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
        eval_env = wrap_env(
            env=gym.make(cfg.env),
            state_mean=infos['obs_mean'],
            state_std=infos['obs_std'],
            reward_scale=1.0,
        )
        eval_episodes = 100
        best_return = -np.inf
        best_returns = None
        best_return_inv = -np.inf
        best_returns_inv = None
        for k in cfg.k:
            def eval():
                obs = eval_env.reset()
                done = False
                total_reward = 0.0
                while not done:
                    z = encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).detach().cpu().numpy()
                    idxs, dists = index.query(z[0], k=k)
                    neighbor_rets = returns[idxs]
                    ranks = neighbor_rets * np.exp(-1 * dists)
                    best_idx = idxs[0, np.argmax(ranks)]

                    action = actions[best_idx]
                    obs, reward, done, _ = eval_env.step(action)
                    total_reward += reward
                return total_reward

            eval_rets = [eval() for _ in trange(eval_episodes, desc="Evaluation", leave=False)]
            print("Mean return:", np.mean(eval_rets), env.get_normalized_score(np.mean(eval_rets)) * 100.0)

            wandb.log({f"eval/return_{k}": np.mean(eval_rets),
                       f"eval/normalized_return_{k}": env.get_normalized_score(np.mean(eval_rets)) * 100.0})
            wandb.log({f"eval/returns_{k}": eval_rets, f"eval/returns_normalized_{k}": [env.get_normalized_score(x) * 100.0 for x in eval_rets]})
            if np.mean(eval_rets) > best_return:
                best_return = np.mean(eval_rets)
                best_returns = eval_rets

            # %%
            def eval_inv_model():
                obs = eval_env.reset()
                done = False
                total_reward = 0.0
                while not done:
                    z = encoder(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).detach().cpu().numpy()[0]
                    idxs, dists = index.query(z, k=k)
                    neighbor_rets = returns[idxs]
                    ranks = neighbor_rets * np.exp(-1 * dists)
                    best_idx = idxs[0, np.argmax(ranks)]

                    action = inv_model(torch.tensor(np.concatenate([z, z_next[best_idx][None, :]], axis=-1),
                                                    dtype=torch.float32).unsqueeze(0)).detach().cpu().numpy()
                    obs, reward, done, _ = eval_env.step(action.squeeze())
                    total_reward += reward
                return total_reward

            eval_rets_inv = [eval_inv_model() for _ in trange(eval_episodes, desc="Evaluation", leave=False)]
            print("Mean return:", np.mean(eval_rets_inv), env.get_normalized_score(np.mean(eval_rets_inv)) * 100.0)

            wandb.log({f"eval/return_inv_{k}": np.mean(eval_rets_inv),
                       f"eval/normalized_return_inv_{k}": env.get_normalized_score(np.mean(eval_rets_inv)) * 100.0})
            wandb.log({f"eval/returns_inv_{k}": eval_rets_inv, f"eval/returns_normalized_inv_{k}": [env.get_normalized_score(x) * 100.0 for x in eval_rets_inv]})
            if np.mean(eval_rets_inv) > best_return_inv:
                best_return_inv = np.mean(eval_rets_inv)
                best_returns_inv = eval_rets_inv

        wandb.log({f"eval/return": best_return, f"eval/return_inv": best_return_inv})
        wandb.log({f"eval/returns": best_returns, f"eval/returns_inv": best_returns_inv})

    for epoch in range(max(cfg.epochs)):
        print(f"Epoch {epoch}")
        for batch in tqdm(dataloader):
            metrics = train_step(batch)
            wandb.log(metrics)
            if epoch in cfg.epochs:
                evaluate(epoch)



def cosine_loss(z, z_next):
    return -F.cosine_similarity(z, z_next, dim=-1).mean()

def mse_loss(z, z_next):
    return F.mse_loss(z, z_next)