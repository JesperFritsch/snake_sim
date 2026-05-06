import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from snake_sim.rl.snapshot_manager import SNAPSHOT_BASE_DIR

snapshot_dir = "new_arch_vs_algo"  
stats_file = ""

if not stats_file:
	df = pd.read_csv(Path(SNAPSHOT_BASE_DIR) / snapshot_dir / "training_stats.csv")
else:
    df = pd.read_csv(Path("/home/jesper/Downloads/training_stats.csv"))

OUTPUT_DIR = Path().home() / "Downloads" / "training_plots" / snapshot_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROLLING_WINDOW = 2000

def rolling(col, scale=1.0):
    if col in df.columns:
        return df[col].rolling(ROLLING_WINDOW, min_periods=1).mean() * scale
    return None

# Plot returns over time
plt.figure(figsize=(12, 6))
plt.plot(df['update'], rolling('returns_mean', 0.1), label='Returns')
plt.plot(df['update'], rolling('best_return', 0.1), label='Best Return', linestyle='--')
plt.plot(df['update'], rolling('illegal_logit_loss'), label='Illegal Logit Loss', linestyle='-.')
plt.plot(df['update'], rolling('entropy'), label='Entropy', linestyle='-')
plt.xlabel('Update')
plt.ylabel('Return')
plt.legend()
plt.title('Training Progress (Running Average)')
plt.savefig(OUTPUT_DIR / 'training_progress.png', dpi=150, bbox_inches='tight')
plt.close()

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
    plt.savefig(OUTPUT_DIR / 'food_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close()

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
    if 'traps_made' in df.columns:
        plt.plot(df['update'], rolling('traps_made'), label='Traps made (batch)', linestyle=':')
        plt.plot(df['update'], running_max_traps, label='Max traps in episode (running)', linestyle='-.')
    plt.xlabel('Update')
    plt.ylabel('Trapping')
    plt.legend()
    plt.title('Trapping Frequency (explicit did_trap, Running Avg)')
    plt.savefig(OUTPUT_DIR / 'trapping_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()

# policy_loss and value_loss can be very noisy, so we plot them with a rolling average and on a secondary axis
plt.figure(figsize=(12, 6))
plt.plot(df['update'], rolling('policy_loss'), label='Policy Loss', linestyle='-')
plt.plot(df['update'], rolling('value_loss'), label='Value Loss', linestyle='--')
plt.xlabel('Update')
plt.ylabel('Loss')
plt.legend()
plt.title('Losses (Running Average)')
plt.savefig(OUTPUT_DIR / 'losses.png', dpi=150, bbox_inches='tight')
plt.close()


# --- Critic health: are predictions tracking returns? ---
plt.figure(figsize=(12, 6))
plt.plot(df['update'], rolling('explained_variance'), label='Explained Variance')
plt.axhline(0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(1, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Update')
plt.ylabel('Explained Variance')
plt.title('Critic Quality (1 = perfect, 0 = mean baseline, negative = worse)')
plt.legend()
plt.savefig(OUTPUT_DIR / 'critic_health.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Update dynamics: is the policy actually moving? ---
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df['update'], rolling('approx_kl'), label='Approx KL', color='C0')
ax1.plot(df['update'], rolling('ratio_std'), label='Ratio Std', color='C1')
ax1.set_xlabel('Update')
ax1.set_ylabel('KL / Ratio Std')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(df['update'], rolling('clipfrac'), label='Clipfrac', color='C2', linestyle='--')
ax2.set_ylabel('Clipfrac')
ax2.legend(loc='upper right')

plt.title('Update Dynamics (is the policy moving?)')
plt.savefig(OUTPUT_DIR / 'update_dynamics.png', dpi=150, bbox_inches='tight')
plt.close()

# --- Behavior rates (rate-normalized, not raw counts) ---
plt.figure(figsize=(12, 6))
plt.plot(df['update'], rolling('foods_per_1k_steps'), label='Foods / 1k steps')
plt.plot(df['update'], rolling('traps_per_1k_steps', scale=100), label='Traps / 1k steps')
plt.xlabel('Update')
plt.ylabel('Events per 1k steps')
plt.title('Behavior Rates')
plt.legend()
plt.savefig(OUTPUT_DIR / 'behavior_rates.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Plots saved to {OUTPUT_DIR}")