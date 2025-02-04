from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class Config:
    hidden_dim: int
    latent_dim: int
    rollout_length: int
    batch_size: int
    gamma: float
    env: str
    seed: int
    target_ema: float
    epochs: int
    k: int
    eval_episodes: list[int]
    knn_temp: list[float]

    grad_clip_norm: float

cs = ConfigStore.instance()
cs.store(node=Config, name="algo_base")