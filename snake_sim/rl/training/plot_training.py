import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from snake_sim.rl.snapshot_manager import SNAPSHOT_BASE_DIR

snapshot_dir = "old_model_again4"


df = pd.read_csv(Path(SNAPSHOT_BASE_DIR) / snapshot_dir / "training_stats.csv")

# Set rolling window size for running average
ROLLING_WINDOW = 200  # Adjust as needed

# Helper to get rolling mean if column exists
def rolling(col, scale=1.0):
	if col in df.columns:
		return df[col].rolling(ROLLING_WINDOW, min_periods=1).mean() * scale
	return None

# Plot returns over time
plt.figure(figsize=(12, 6))
plt.plot(df['update'], rolling('returns_mean', 0.1), label='Returns')
plt.plot(df['update'], rolling('best_return', 0.1), label='Best Return', linestyle='--')
plt.plot(df['update'], rolling('path_mixer_alpha'), label='Path Mixer Alpha', linestyle=':')
plt.plot(df['update'], rolling('illegal_logit_loss'), label='Illegal Logit Loss', linestyle='-.')
plt.plot(df['update'], rolling('entropy'), label='Entropy', linestyle='-')
plt.xlabel('Update')
plt.ylabel('Return')
plt.legend()
plt.title('Training Progress (Running Average)')
plt.show()

# Plot food efficiency metrics if present
food_cols = {'foods_per_1k_steps', 'steps_per_food', 'foods_eaten'}
if any(c in df.columns for c in food_cols):
    plt.figure(figsize=(12, 6))
    if 'foods_per_1k_steps' in df.columns:
        plt.plot(df['update'], rolling('foods_per_1k_steps'), label='Foods / 1k steps', linestyle='-')
    if 'steps_per_food' in df.columns:
        plt.plot(df['update'], rolling('steps_per_food', 10), label='Steps / food', linestyle='--')
    if 'foods_eaten' in df.columns:
        plt.plot(df['update'], rolling('foods_eaten'), label='Foods eaten (batch)', linestyle=':')
    plt.xlabel('Update')
    plt.ylabel('Food efficiency')
    plt.legend()
    plt.title('Food Efficiency (explicit ate_food, Running Avg)')
    plt.show()

# Plot trapping metrics if present
trap_cols = {'traps_per_1k_steps', 'steps_per_trap', 'traps_made'}
max_traps = 0
running_max_traps = []

for i, row in df.iterrows():
	if 'traps_made' in df.columns:
		max_traps = max(max_traps, row['traps_made'])
	running_max_traps.append(max_traps)

if any(c in df.columns for c in trap_cols):
	plt.figure(figsize=(12, 6))
	# Uncomment and use rolling average for these if desired
	# if 'traps_per_1k_steps' in df.columns:
	#     plt.plot(df['update'], rolling('traps_per_1k_steps'), label='Traps / 1k steps', linestyle='-')
	# if 'steps_per_trap' in df.columns:
	#     plt.plot(df['update'], rolling('steps_per_trap'), label='Steps / trap', linestyle='--')
	if 'traps_made' in df.columns:
		plt.plot(df['update'], rolling('traps_made'), label='Traps made (batch)', linestyle=':')
	plt.plot(df['update'], running_max_traps, label='Max traps in episode (running)', linestyle='-.')
	plt.xlabel('Update')
	plt.ylabel('Trapping')
	plt.legend()
	plt.title('Trapping Frequency (explicit did_trap, Running Avg)')
	plt.show()
