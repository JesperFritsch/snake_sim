"""Quick timing benchmark for PPOTrainer pipeline.

Goal:
- Identify whether the main training bottleneck is CPU collation (np.stack + Python loops),
  CPU->GPU transfer, forward pass, or backward/optimizer.

This uses synthetic `State`s so it doesn't depend on launching environments.
It should roughly match real costs for collation + GPU work if you set shapes correctly.

Usage:
  python -m snake_sim.rl.training.bench_timing

Optional env vars:
  BATCH=2048
  MAP_C=4
  MAP_H=20
  MAP_W=20
  CTX=8
  ACTIONS=4
  FEAT=3
  ITERS=50
"""

from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np
import torch

from snake_sim.rl.types import State
from snake_sim.rl.training.ppo_trainer import PPOTrainer, PPOTrainerConfig
from snake_sim.rl.training.rl_data_queue import RLMetaDataQueue


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None else default


def _make_synthetic_states(
    batch: int,
    map_c: int,
    map_h: int,
    map_w: int,
    ctx_dim: int,
    actions: int,
    feat_dim: int,
) -> list[State]:
    # Random-ish but deterministic shapes; float32 like real.
    maps = np.random.randn(batch, map_c, map_h, map_w).astype(np.float32)
    ctx = np.random.randn(batch, ctx_dim).astype(np.float32)

    # action_features: (A, F)
    af = np.random.randn(batch, actions, feat_dim).astype(np.float32)

    # action_mask: ensure at least one valid action
    am = (np.random.rand(batch, actions) > 0.4).astype(np.float32)
    am[:, 0] = 1.0

    states: list[State] = []
    for i in range(batch):
        states.append(
            State(
                maps[i],
                ctx=ctx[i],
                meta={
                    "action_features": af[i],
                    "action_mask": am[i],
                },
            )
        )
    return states


@torch.no_grad()
def _warmup(trainer: PPOTrainer, states: list[State]) -> None:
    # Create model
    trainer._ensure_model(states[0])  # noqa: SLF001
    # Run a couple forward passes to warm up kernels
    device = trainer.device
    sb = trainer._collate_states(states[:256])  # noqa: SLF001
    for _ in range(5):
        logits, v = trainer.model(sb)  # type: ignore[operator]
        # keep small sync
        if device.type == "cuda":
            torch.cuda.synchronize()


def main() -> None:
    batch = _env_int("BATCH", 2048)
    map_c = _env_int("MAP_C", 4)
    map_h = _env_int("MAP_H", 20)
    map_w = _env_int("MAP_W", 20)
    ctx_dim = _env_int("CTX", 8)
    actions = _env_int("ACTIONS", 4)
    feat = _env_int("FEAT", 1)
    iters = _env_int("ITERS", 40)

    cfg = PPOTrainerConfig()
    # Keep it deterministic-ish and fast.
    cfg.use_amp = True
    cfg.compile_model = False

    trainer = PPOTrainer(RLMetaDataQueue(), config=cfg, snapshot_dir="_bench")

    states = _make_synthetic_states(batch, map_c, map_h, map_w, ctx_dim, actions, feat)

    # Build fake per-sample tensors matching what _ppo_update expects.
    # We'll construct a batch dict directly to isolate timing.
    trainer._ensure_model(states[0])  # noqa: SLF001

    if trainer.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    _warmup(trainer, states)

    # Precreate fixed tensors that normally come from transitions.
    device = trainer.device
    rng = np.random.default_rng(0)

    # Timers
    t_collate = 0.0
    t_shuffle = 0.0
    t_forward = 0.0
    t_backward = 0.0
    t_total = 0.0

    for _ in range(iters):
        t0 = time.perf_counter()

        # --- Collation (includes CPU->GPU transfer) ---
        c0 = time.perf_counter()
        sb = trainer._collate_states(states)  # noqa: SLF001
        if device.type == "cuda":
            torch.cuda.synchronize()
        c1 = time.perf_counter()

        # Create synthetic PPO tensors on-device
        actions_t = torch.from_numpy(rng.integers(0, actions, size=(batch,), dtype=np.int64)).to(device)
        old_logp = torch.zeros(batch, device=device, dtype=torch.float32)
        old_v = torch.zeros(batch, device=device, dtype=torch.float32)
        adv = torch.randn(batch, device=device, dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        rets = torch.randn(batch, device=device, dtype=torch.float32)

        # --- Shuffle ---
        s0 = time.perf_counter()
        idx = torch.randperm(batch, device=device)
        sb_shuf = {
            "map": sb["map"][idx],
            "ctx": sb["ctx"][idx],
            "action_features": sb["action_features"][idx],
            "action_mask": sb["action_mask"][idx],
        }
        actions_shuf = actions_t[idx]
        old_logp_shuf = old_logp[idx]
        old_v_shuf = old_v[idx]
        adv_shuf = adv[idx]
        rets_shuf = rets[idx]
        if device.type == "cuda":
            torch.cuda.synchronize()
        s1 = time.perf_counter()

        # --- Forward ---
        f0 = time.perf_counter()
        # AMP context similar to trainer
        if device.type == "cuda":
            from torch import amp as _torch_amp

            autocast_ctx = _torch_amp.autocast("cuda", enabled=cfg.use_amp)
        else:
            class _Noop:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False
            autocast_ctx = _Noop()

        with autocast_ctx:
            logits_raw, values = trainer._forward(sb_shuf)  # noqa: SLF001
        # simple loss to force backward path cost similar order
        logits_raw = logits_raw.float()
        values = values.float()
        # make sure action_mask stays on-device
        am = sb_shuf["action_mask"].to(dtype=torch.bool)
        # mask like trainer
        from snake_sim.rl.action_masking import apply_action_mask_to_logits

        logits = apply_action_mask_to_logits(logits_raw, am)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions_shuf)
        ratio = torch.exp(logp - old_logp_shuf)
        surr = -(ratio * adv_shuf).mean()
        vloss = torch.nn.functional.mse_loss(values, rets_shuf)
        loss = surr + 0.5 * vloss - 0.01 * dist.entropy().mean()

        if device.type == "cuda":
            torch.cuda.synchronize()
        f1 = time.perf_counter()

        # --- Backward/step ---
        b0 = time.perf_counter()
        trainer.optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        loss.backward()
        trainer.optimizer.step()  # type: ignore[union-attr]
        if device.type == "cuda":
            torch.cuda.synchronize()
        b1 = time.perf_counter()

        t1 = time.perf_counter()

        t_collate += c1 - c0
        t_shuffle += s1 - s0
        t_forward += f1 - f0
        t_backward += b1 - b0
        t_total += t1 - t0

    def pct(x: float) -> float:
        return 100.0 * x / max(t_total, 1e-12)

    print("\n=== PPO timing breakdown (avg over iters) ===")
    print(f"device: {device} | batch={batch} | map=({map_c},{map_h},{map_w}) | ctx={ctx_dim} | iters={iters}")
    print(f"collate+transfer: {t_collate/iters:.6f}s  ({pct(t_collate):.1f}%)")
    print(f"shuffle/indexing : {t_shuffle/iters:.6f}s  ({pct(t_shuffle):.1f}%)")
    print(f"forward+loss     : {t_forward/iters:.6f}s  ({pct(t_forward):.1f}%)")
    print(f"backward+step    : {t_backward/iters:.6f}s  ({pct(t_backward):.1f}%)")
    print(f"TOTAL            : {t_total/iters:.6f}s")


if __name__ == "__main__":
    main()
