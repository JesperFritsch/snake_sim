import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from snake_sim.rl.snapshot_manager import SNAPSHOT_BASE_DIR

snapshot_dir = "basemodel_small_mb_many_agents"

df = pd.read_csv(Path(SNAPSHOT_BASE_DIR) / snapshot_dir / "training_stats.csv")

# Plot returns over time
plt.figure(figsize=(12, 6))
plt.plot(df['update'], df['returns_mean'] * 0.1, label='Returns')
plt.plot(df['update'], df['best_return'] * 0.1, label='Best Return', linestyle='--')
plt.plot(df['update'], df['path_mixer_alpha'], label='Path Mixer Alpha', linestyle=':')
plt.plot(df['update'], df['illegal_logit_loss'], label='Illegal Logit Loss', linestyle='-.')
plt.plot(df['update'], df['entropy'], label='Entropy', linestyle='-')
plt.xlabel('Update')
plt.ylabel('Return')
plt.legend()
plt.title('Training Progress')
plt.show()

# Plot food efficiency metrics if present
food_cols = {'foods_per_1k_steps', 'steps_per_food', 'foods_eaten'}
if any(c in df.columns for c in food_cols):
	plt.figure(figsize=(12, 6))
	if 'foods_per_1k_steps' in df.columns:
		plt.plot(df['update'], df['foods_per_1k_steps'], label='Foods / 1k steps', linestyle='-')
	if 'steps_per_food' in df.columns:
		plt.plot(df['update'], df['steps_per_food'] * 10, label='Steps / food', linestyle='--')
	if 'foods_eaten' in df.columns:
		plt.plot(df['update'], df['foods_eaten'], label='Foods eaten (batch)', linestyle=':')
	plt.xlabel('Update')
	plt.ylabel('Food efficiency')
	plt.legend()
	plt.title('Food Efficiency (explicit ate_food)')
	plt.show()

# Detect when exploration was boosted
boost_points = df[df['exploration_boosted'] == True]