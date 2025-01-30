# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py
# 2. https://github.com/karpathy/minGPT
import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import trange

@dataclass
class TrainConfig:
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "kNN-D4RL"
    # wandb run name
    name: str = "kNN"
    # transformer hidden dim
    # embedding_dim: int = 128
    # # depth of the transformer model
    # num_layers: int = 3
    # # number of heads in the attention
    # num_heads: int = 1
    # # maximum sequence length during training
    seq_len: int = 20
    # maximum rollout length, needed for the positional embeddings
    episode_len: int = 1000
    # attention dropout
    # attention_dropout: float = 0.1
    # # residual dropout
    # residual_dropout: float = 0.1
    # # embeddings dropout
    # embedding_dropout: float = 0.1
    # # maximum range for the symmetric actions, [-1, 1]
    # max_action: float = 1.0
    # # training dataset and evaluation environment
    # env_name: str = "halfcheetah-medium-v2"
    # # AdamW optimizer learning rate
    # learning_rate: float = 1e-4
    # # AdamW optimizer betas
    # betas: Tuple[float, float] = (0.9, 0.999)
    # # AdamW weight decay
    # weight_decay: float = 1e-4
    # # maximum gradient norm during training, optional
    # clip_grad: Optional[float] = 0.25
    # # training batch size
    # batch_size: int = 64
    # # total training steps
    # update_steps: int = 100_000
    # # warmup steps for the learning rate scheduler
    # warmup_steps: int = 10_000
    # # reward scaling, to reduce the magnitude
    # reward_scale: float = 0.001
    # # number of workers for the pytorch dataloader
    # num_workers: int = 4
    # # target return-to-go for the prompting durint evaluation
    # target_returns: Tuple[float, ...] = (12000.0, 6000.0)
    # # number of episodes to run during evaluation
    eval_episodes: int = 100
    # # evaluation frequency, will evaluate eval_every training steps
    # eval_every: int = 10_000
    # path for checkpoints saving, optional
    # checkpoints_path: Optional[str] = None
    # configure PyTorch to use deterministic algorithms instead
    # of nondeterministic ones
    deterministic_torch: bool = False
    # training random seed
    train_seed: int = 10
    # evaluation random seed
    eval_seed: int = 42
    # training device
    device: str = "cuda"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()



def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


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


class SequenceDataset(IterableDataset):
    def __init__(self, env_name: str, seq_len: int = 10, reward_scale: float = 1.0):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed, padding is masked during training
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)

# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    wandb_init(asdict(config))

    # data & dataloader setup
    dataset = SequenceDataset(
        config.env_name, seq_len=config.seq_len, reward_scale=config.reward_scale
    )
    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )
    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    # model & optimizer & scheduler setup
    config.state_dim = eval_env.observation_space.shape[0]
    config.action_dim = eval_env.action_space.shape[0]
    # model = DecisionTransformer(
    #     state_dim=config.state_dim,
    #     action_dim=config.action_dim,
    #     embedding_dim=config.embedding_dim,
    #     seq_len=config.seq_len,
    #     episode_len=config.episode_len,
    #     num_layers=config.num_layers,
    #     num_heads=config.num_heads,
    #     attention_dropout=config.attention_dropout,
    #     residual_dropout=config.residual_dropout,
    #     embedding_dropout=config.embedding_dropout,
    #     max_action=config.max_action,
    # ).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        wandb.log(
            {
                "train_loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
            },
            step=step,
        )

        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            for target_return in config.target_returns:
                eval_env.seed(config.eval_seed)
                eval_returns = []
                for _ in trange(config.eval_episodes, desc="Evaluation", leave=False):
                    eval_return, eval_len = eval_rollout(
                        model=model,
                        env=eval_env,
                        target_return=target_return * config.reward_scale,
                        device=config.device,
                    )
                    # unscale for logging & correct normalized score computation
                    eval_returns.append(eval_return / config.reward_scale)

                normalized_scores = (
                    eval_env.get_normalized_score(np.array(eval_returns)) * 100
                )
                wandb.log(
                    {
                        f"eval/{target_return}_return_mean": np.mean(eval_returns),
                        f"eval/{target_return}_return_std": np.std(eval_returns),
                        f"eval/{target_return}_normalized_score_mean": np.mean(
                            normalized_scores
                        ),
                        f"eval/{target_return}_normalized_score_std": np.std(
                            normalized_scores
                        ),
                    },
                    step=step,
                )
            model.train()

    if config.checkpoints_path is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        torch.save(checkpoint, os.path.join(config.checkpoints_path, "dt_checkpoint.pt"))


if __name__ == "__main__":
    train()
