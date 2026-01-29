# Example: python -m src.run task=linear method=lora optimizer=oplora num_epochs=100 lr=0.1 momentum=0.7

import logging
import os
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from src.basic_trainer import BasicTrainer
from src.utils import init_seed

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


def run(cfg: DictConfig, output_dir: Path) -> float:

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    assert not (cfg.optimizer.name == "oplora" and cfg.method.name != "lora"), (
        "Optimizer 'oplora' requires method 'lora'."
    )

    # TODO: quick check if run is done for cfg.continue_training
    trainer: BasicTrainer = instantiate(cfg.trainer)

    return trainer.run(
        num_epochs=cfg.num_epochs,
        continue_training=cfg.continue_training,
        output_dir=output_dir,
        target_metric=cfg.target_metric,
        save_model_every_epoch=cfg.save_model_every_epoch
    )


@hydra.main(version_base=None, config_path="conf", config_name="base")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logger.info(f"seed: {cfg.seed}")

    init_seed(cfg.seed)

    return run(cfg, output_dir)


if __name__ == '__main__':
    main()
