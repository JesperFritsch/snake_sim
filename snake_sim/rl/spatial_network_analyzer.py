"""
spatial_network_analyzer.py

Analyzer for evaluating the contribution of the spatial network in PPO Snake models.
Loads the latest model snapshot and runs ablation tests (normal, zeroed, randomized spatial input).
"""
import torch
import numpy as np
from pathlib import Path
from snake_sim.rl.snapshot_manager import SnapshotManager
from snake_sim.rl.models.ppo_model import model_factory, SnakePPONet

class SpatialNetworkAnalyzer:
    def __init__(self, snapshot_dir: str, base_name: str = "ppo_model", device: str = "cpu"):
        self.snapshot_dir = snapshot_dir
        self.base_name = base_name
        self.device = device
        self.model = None
        self._snapshot_manager = None

    def _load_latest_model(self, in_channels: int, ctx_dim: int):
        # Create a SnapshotManager factory that builds a model with the correct
        # input channel and context sizes so state_dicts load cleanly.
        sm = SnapshotManager(
            dir=self.snapshot_dir,
            base_name=self.base_name,
            factory=lambda: model_factory(in_channels, ctx_dim)
        )
        model = sm.init_or_load(torch.device(self.device))
        model.eval()
        self._snapshot_manager = sm
        self.model = model
        return model

    def run_ablation(self, map_batch, ctx_batch, action_features_batch, mode="normal"):
        """
        mode: 'normal', 'zeroed', or 'randomized' for spatial input.
        map_batch: torch.Tensor (B, C, H, W)
        ctx_batch: torch.Tensor (B, ctx_dim)
        action_features_batch: torch.Tensor (B, A, F)
        """
        # Ensure model is created with matching dims
        if self.model is None:
            in_channels = int(map_batch.shape[1])
            ctx_dim = int(ctx_batch.shape[1]) if ctx_batch is not None and ctx_batch.ndim == 2 else 0
            self._load_latest_model(in_channels, ctx_dim)

        x = {"map": map_batch.clone(), "ctx": ctx_batch, "action_features": action_features_batch}

        # Support ablation across map, ctx and action_features
        if mode == "map_zeroed":
            x["map"].zero_()
        elif mode == "map_randomized":
            x["map"] = torch.rand_like(x["map"])
        elif mode == "ctx_zeroed":
            if x["ctx"] is not None:
                x["ctx"] = torch.zeros_like(x["ctx"])
        elif mode == "ctx_randomized":
            if x["ctx"] is not None:
                x["ctx"] = torch.rand_like(x["ctx"])
        elif mode == "af_zeroed":
            x["action_features"] = torch.zeros_like(x["action_features"])
        elif mode == "af_randomized":
            x["action_features"] = torch.rand_like(x["action_features"])
        elif mode == "all_zeroed":
            x["map"].zero_()
            if x["ctx"] is not None:
                x["ctx"] = torch.zeros_like(x["ctx"])
            x["action_features"] = torch.zeros_like(x["action_features"])

        with torch.no_grad():
            logits, values = self.model(x)
        return logits, values

    def compare_modes(self, map_batch, ctx_batch, action_features_batch):
        modes = [
            "normal",
            "map_zeroed",
            "map_randomized",
            "ctx_zeroed",
            "ctx_randomized",
            "af_zeroed",
            "af_randomized",
            "all_zeroed",
        ]
        results = {}

        # Baseline
        logits0, values0 = self.run_ablation(map_batch, ctx_batch, action_features_batch, mode="normal")
        probs0 = torch.softmax(logits0, dim=-1)
        for mode in modes:
            if mode == "normal":
                logits, values = logits0, values0
            else:
                logits, values = self.run_ablation(map_batch, ctx_batch, action_features_batch, mode)
            probs = torch.softmax(logits, dim=-1)
            # Metrics comparing to baseline
            logits_l2 = float(torch.norm((logits0 - logits).float()).cpu().item())
            value_diff = float(torch.abs(values0 - values).mean().cpu().item())
            top_action_change = float((logits0.argmax(dim=-1) != logits.argmax(dim=-1)).float().mean().cpu().item())
            # KL(probs || probs0) measure (add tiny eps inside log to be safe)
            try:
                kl = float(torch.nn.functional.kl_div((probs + 1e-9).log(), probs0, reduction='batchmean').cpu().item())
            except Exception:
                kl = float('nan')

            results[mode] = {
                "logits": logits.cpu(),
                "values": values.cpu(),
                "metrics": {
                    "logits_l2": logits_l2,
                    "value_diff_mean": value_diff,
                    "top_action_change_rate": top_action_change,
                    "kl_div_batchmean": kl,
                }
            }

        return results
