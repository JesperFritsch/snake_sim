#!/usr/bin/env python3
"""
Visualize the reward function changes and their impact.
Run this to understand the reward landscape before/after changes.
"""

import matplotlib.pyplot as plt
import numpy as np

# Old reward structure
def old_rewards():
    death = -20.0
    ate_food = 1.0 + 2.0  # food_eat + length
    survival_per_step = 0.05
    approach_per_step = 0.3
    
    return {
        'death': death,
        'ate_food': ate_food,
        'survival_20_steps': survival_per_step * 20,
        'approach_20_steps': (survival_per_step + approach_per_step) * 20,
    }

# New reward structure  
def new_rewards():
    death = -5.0
    ate_food = 10.0 + 5.0  # food_eat + length (size 3)
    survival_per_step = 0.01
    approach_per_step = 0.2
    
    return {
        'death': death,
        'ate_food': ate_food,
        'survival_20_steps': survival_per_step * 20,
        'approach_20_steps': (survival_per_step + approach_per_step) * 20,
    }

def plot_comparison():
    old = old_rewards()
    new = new_rewards()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Reward magnitudes
    scenarios = ['Death', 'Ate Food', 'Survived\n20 steps', 'Approached\n20 steps']
    old_vals = [old['death'], old['ate_food'], old['survival_20_steps'], old['approach_20_steps']]
    new_vals = [new['death'], new['ate_food'], new['survival_20_steps'], new['approach_20_steps']]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, old_vals, width, label='Old Rewards', color='indianred', alpha=0.7)
    bars2 = ax1.bar(x + width/2, new_vals, width, label='New Rewards', color='forestgreen', alpha=0.7)
    
    ax1.set_ylabel('Reward Value', fontsize=12)
    ax1.set_title('Reward Comparison: Old vs New', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=10)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Plot 2: Risk/Reward analysis
    ax2.set_title('Risk Assessment for Agent', fontsize=14, fontweight='bold')
    
    # Scenarios
    scenarios_risk = ['Just\nSurvive\n(100 steps)', 'Try to Eat\n(50% success)\n1 food', 'Aggressive\n(30% success)\n2 foods']
    
    # Old calculations
    old_survive = old['survival_20_steps'] * 5  # 100 steps
    old_eat_expected = 0.5 * (old['ate_food'] + old['survival_20_steps']) + 0.5 * old['death']
    old_aggressive = 0.3 * (old['ate_food'] * 2 + old['survival_20_steps']) + 0.7 * old['death']
    
    # New calculations
    new_survive = new['survival_20_steps'] * 5
    new_eat_expected = 0.5 * (new['ate_food'] + new['survival_20_steps']) + 0.5 * new['death']
    new_aggressive = 0.3 * (new['ate_food'] * 2 + new['survival_20_steps']) + 0.7 * new['death']
    
    old_risk_vals = [old_survive, old_eat_expected, old_aggressive]
    new_risk_vals = [new_survive, new_eat_expected, new_aggressive]
    
    x2 = np.arange(len(scenarios_risk))
    bars3 = ax2.bar(x2 - width/2, old_risk_vals, width, label='Old Expected Return', color='indianred', alpha=0.7)
    bars4 = ax2.bar(x2 + width/2, new_risk_vals, width, label='New Expected Return', color='forestgreen', alpha=0.7)
    
    ax2.set_ylabel('Expected Return', fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(scenarios_risk, fontsize=9)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Add annotations
    ax2.text(0.5, 0.95, 'OLD: Play it safe wins', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='indianred', alpha=0.3),
             fontsize=10)
    ax2.text(0.5, 0.85, 'NEW: Taking risks pays off!', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='forestgreen', alpha=0.3),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/jesper/dev/snake_sim/reward_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to: reward_comparison.png")
    plt.show()

def print_analysis():
    print("\n" + "="*70)
    print("REWARD FUNCTION ANALYSIS")
    print("="*70)
    
    old = old_rewards()
    new = new_rewards()
    
    print("\n📊 KEY METRICS:")
    print(f"  Death Penalty:        {old['death']:.1f} → {new['death']:.1f} ({abs(new['death'])/abs(old['death'])*100:.0f}%)")
    print(f"  Food Reward:          {old['ate_food']:.1f} → {new['ate_food']:.1f} ({new['ate_food']/old['ate_food']*100:.0f}%)")
    print(f"  Survival 20 steps:    {old['survival_20_steps']:.1f} → {new['survival_20_steps']:.1f}")
    
    print("\n🎯 IMPACT ANALYSIS:")
    
    # Risk calculations
    old_survive_100 = old['survival_20_steps'] * 5
    new_survive_100 = new['survival_20_steps'] * 5
    print(f"\n  Safe Strategy (survive 100 steps):")
    print(f"    Old: {old_survive_100:.2f}")
    print(f"    New: {new_survive_100:.2f}")
    print(f"    → Reduced by {(1 - new_survive_100/old_survive_100)*100:.0f}%")
    
    old_eat = 0.5 * old['ate_food'] + 0.5 * old['death']
    new_eat = 0.5 * new['ate_food'] + 0.5 * new['death']
    print(f"\n  Risky Strategy (50% chance to eat, 50% death):")
    print(f"    Old: {old_eat:.2f}")
    print(f"    New: {new_eat:.2f}")
    print(f"    → {'Improved' if new_eat > old_eat else 'Worse'} by {abs(new_eat - old_eat):.2f}")
    
    print("\n💡 EXPECTED BEHAVIOR:")
    if old_survive_100 > old_eat:
        print("  OLD: Agent prefers safety over risk (playing it safe)")
    else:
        print("  OLD: Agent prefers taking risks")
        
    if new_survive_100 > new_eat:
        print("  NEW: Agent prefers safety over risk")
    else:
        print("  NEW: Agent prefers taking risks (GOOD!)")
    
    print("\n" + "="*70)
    print()

if __name__ == '__main__':
    print_analysis()
    
    try:
        plot_comparison()
    except Exception as e:
        print(f"⚠️  Could not create plot: {e}")
        print("(matplotlib may not be available in this environment)")
