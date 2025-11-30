import torch
import numpy as np
from snake_sim.rl.spatial_network_analyzer import SpatialNetworkAnalyzer
from snake_sim.rl.models.ppo_model import model_factory

def test_spatial_network_ablation():
    # Example: load a batch of states (replace with real state loading logic)
    # Here we use random tensors for demonstration
    batch_size = 4
    in_channels = 5  # Adjust to match your model
    height = 32
    width = 32
    ctx_dim = 9
    state_batch = torch.rand(batch_size, in_channels, height, width)
    ctx_batch = torch.rand(batch_size, ctx_dim)

    analyzer = SpatialNetworkAnalyzer(
        snapshot_dir="models/ppo_training",  # Adjust if needed
        base_name="ppo_model",
        device="cpu"
    )
    results = analyzer.compare_modes(state_batch, ctx_batch)
    for mode, res in results.items():
        print(f"Mode: {mode}")
        print("Logits:", res["logits"])
        print("Values:", res["values"])

if __name__ == "__main__":
    test_spatial_network_ablation()
