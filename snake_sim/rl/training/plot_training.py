import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/jesper/dev/snake_sim/snake_sim/rl/models_snapshots/new_simpler_model_new_state/training_stats_20251218_204457.csv')

# Plot returns over time
plt.figure(figsize=(12, 6))
plt.plot(df['update'], df['returns_mean'], label='Returns')
plt.plot(df['update'], df['best_return'], label='Best Return', linestyle='--')
plt.xlabel('Update')
plt.ylabel('Return')
plt.legend()
plt.title('Training Progress')
plt.show()

# Detect when exploration was boosted
boost_points = df[df['exploration_boosted'] == True]