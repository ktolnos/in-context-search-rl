import hydra
from configs.config import Config

@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg):
    print("Guarded imports after this")

    import sys
    import traceback
    # This main is used to circumvent a bug in Hydra
    # See https://github.com/facebookresearch/hydra/issues/2664

    try:
        run_exp(cfg)
    except Exception:
        print("Exception in main")
        traceback.print_exc(file=sys.stderr)
    finally:
        # fflush everything
        sys.stdout.flush()
        sys.stderr.flush()


def wandb_init(config: dict) -> None:
    import wandb
    import uuid

    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def run_exp(cfg):
    from algorithms.knn import run_knn
    import multiprocessing
    from omegaconf import OmegaConf

    multiprocessing.set_start_method('spawn', force=True)
    print(cfg.keys())


    config = OmegaConf.to_object(cfg)
    assert isinstance(config, Config)
    wandb_init(cfg)


    run_knn(config)

if __name__ == "__main__":
    main()