
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import numpy as np

from snake_sim.rl.snapshot_manager import SNAPSHOT_BASE_DIR
from snake_sim.rl.training.ppo_trainer import PPOTrainer
from snake_sim.rl.training.rl_data_queue import RLMetaDataQueue
from snake_sim.rl.types import RLTrainingConfig

# Reuse your existing environment setup helpers.
from snake_sim.rl.training.snake_training import setup_training_loop

log = logging.getLogger(Path(__file__).stem)


@dataclass
class EnvConfig:
    num_envs: int
    nr_ppo_snakes: int
    food_tiles: int
    training_maps: Optional[list[str | None]] = field(default_factory=lambda: [None])
    width: int = 32
    height: int = 32


@dataclass
class MultiEnvConfig:
    env_config: list[EnvConfig]
    snapshot_dir: str
    episodes: int = 500000
    max_no_food_steps: int = 1000
    trainer_poll_interval_sec: float = 0.5


# ---- Cross-process transition plumbing ----

def _actor_process_main(
    *,
    actor_id: int,
    snapshot_dir: str,
    episodes: int,
    max_no_food_steps: int,
    training_maps: Optional[list[str]],
    food_tiles: int,
    nr_snakes: int,
    out_queue,
):
    """Run one environment loop and push transitions to the learner over out_queue."""


    # Configure RL loop
    cfg = RLTrainingConfig(
        episodes=episodes,
        max_no_food_steps=max_no_food_steps,
        training_maps=training_maps,
        food_tiles=food_tiles,
        nr_snakes=nr_snakes,
    )

    queue = RLMetaDataQueue(queue=out_queue)

    # Build loop and reduce snakes per env.
    loop = setup_training_loop(cfg, snapshot_dir=snapshot_dir, transition_queue=queue)

    log.info(f"Actor {actor_id}: starting RLTrainingLoop")
    loop.start()


def _learner_process_main(*, snapshot_dir: str, in_queue, trainer_poll_interval_sec: float):
    """Own the trainer and drain transitions from all actors."""


    Path(SNAPSHOT_BASE_DIR, snapshot_dir).mkdir(parents=True, exist_ok=True)
    local_rl_queue = RLMetaDataQueue(queue=in_queue)
    trainer = PPOTrainer(transition_queue=local_rl_queue, snapshot_dir=snapshot_dir)
    trainer.start_background(interval_sec=trainer_poll_interval_sec)
    trainer.join_background()


def main(cfg: MultiEnvConfig):

    mp_ctx = get_context("spawn")

    q = mp_ctx.Queue(maxsize=200_000)

    learner = mp_ctx.Process(
        target=_learner_process_main,
        kwargs={
            "snapshot_dir": cfg.snapshot_dir,
            "in_queue": q,
            "trainer_poll_interval_sec": cfg.trainer_poll_interval_sec,
        },
        daemon=True,
    )
    learner.start()

    # Actor processes
    actors = []
    for env_cfg in cfg.env_config:
        for i in range(env_cfg.num_envs):
            p = mp_ctx.Process(
                target=_actor_process_main,
                kwargs={
                    "actor_id": i,
                    "snapshot_dir": cfg.snapshot_dir,
                    "episodes": cfg.episodes,
                    "max_no_food_steps": cfg.max_no_food_steps,
                    "training_maps": env_cfg.training_maps,
                    "food_tiles": env_cfg.food_tiles,
                    "nr_snakes": env_cfg.nr_ppo_snakes,
                    "out_queue": q,
                },
                daemon=True,
            )
            log.info(f"Starting actor process {i} for env config {env_cfg}")
            p.start()
            actors.append(p)

    # Wait
    try:
        while True:
            if not learner.is_alive():
                raise RuntimeError("Learner process died")
            for p in actors:
                if not p.is_alive():
                    raise RuntimeError("An actor process died")
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        for p in actors:
            try:
                p.terminate()
            except Exception:
                pass
        try:
            learner.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    # Keep logs readable when launched as a script.
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    config = MultiEnvConfig(
        snapshot_dir="basemodel_small_mb_many_agents",
        env_config=[
            EnvConfig(
                num_envs=1, 
                nr_ppo_snakes=5, 
                food_tiles=15
            ),
            EnvConfig(
                num_envs=1, 
                nr_ppo_snakes=10, 
                food_tiles=15
            ),
            EnvConfig(
                num_envs=1, 
                nr_ppo_snakes=20, 
                food_tiles=15
            ),
            EnvConfig(
                num_envs=5, 
                nr_ppo_snakes=1, 
                food_tiles=5, 
                training_maps=[
                    "comps2",
                    "comps",
                    "lil_sign",
                    "face",
                    "patterns",
                    "quarters",
                    "wavy",
                    "tricky",
                    "items",
                    # None
                ]
            ),
        ],
    )
    main(config)
