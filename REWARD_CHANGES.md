# Reward Function Changes Summary

## Problem Diagnosis

Your snake was learning to "play it safe" instead of hunting for food because:

1. **Death penalty too high** (-20.0): Made agent risk-averse
2. **Food rewards too low** (1.0-3.0): Not worth the risk
3. **Survival reward too high** (0.05/step): Could accumulate without eating
4. **Danger signals too weak** (-0.2): Agent ignored unsafe moves

**Result**: Agent found a local optimum of "stay alive, don't try hard" giving ~0-2 return per episode.

---

## Changes Made

### Before → After

| Reward Component | Old Value | New Value | Reason |
|------------------|-----------|-----------|---------|
| **Death Penalty** | -20.0 | **-5.0** | Reduce paralysis, allow risk-taking |
| **Food Eaten** | +1.0 | **+10.0** | Make eating the clear goal |
| **Growth** | +2.0-2.8 | **+5.0-7.5** | Stronger incentive to grow |
| **Approach Food** | +0.3 | **+0.2** | Still guides, but doesn't dominate |
| **Move Away** | -0.1 | **-0.05** | Gentler discouragement |
| **Survival/Step** | +0.05 | **+0.01** | Reduce passive reward |
| **In Danger** | -0.2 | **-1.0** | Make danger matter |
| **Safe** | 0.0 | **+0.1** | Reward smart positioning |

---

## Expected Behavior Changes

### Old Reward Structure:
```
Episode with 20 steps, no food: +1.0 (0.05 × 20)
Episode that ate once: +3.15
Death episode: -20.0

→ Agent thinks: "Don't risk anything!"
```

### New Reward Structure:
```
Episode with 20 steps, no food: +0.2 (0.01 × 20)
Episode that ate once: +15.0 to +17.5
Death episode: -5.0

→ Agent thinks: "Eating is 75x better than just surviving!"
```

---

## What to Expect

### Immediate Effects (First 100 updates):
- **More deaths**: Agent will explore more aggressively
- **Lower initial returns**: Deaths don't cost as much, but agent is learning
- **More variability**: Exploration phase

### Learning Phase (100-500 updates):
- **Food-seeking behavior**: Agent starts finding food more often
- **Returns climbing**: Should see steady improvement from 5 → 15
- **Risk-taking**: Agent balances danger vs reward

### Convergence (500+ updates):
- **Target return**: 15-25 per episode (eating 1-2 foods)
- **Best episodes**: 30+ (eating 3+ foods, growing large)
- **Stable policy**: Confident food hunting

---

## Monitoring Tools

### 1. Debug Logging (ENABLED)
You'll now see logs like:
```
🎯 Rewards for snake 1: length=5.00, survival=0.01, food_approach=0.20, 
   food_eat=10.00, survival_chance=0.10, total=15.31
```

### 2. CSV Data
Check `returns_mean` column in your training stats:
- **< 0**: Still dying too much
- **0-5**: Learning to survive
- **5-15**: Starting to eat food
- **15+**: Good performance!

### 3. Stagnation Detection
With new adaptive exploration (50-update patience), you should see:
- Faster response to stagnation
- Progressive entropy boosts
- Clear warnings if rewards aren't improving

---

## Troubleshooting

### If returns stay negative:
- Death penalty might still be too high
- Try: Change to -3.0

### If returns plateau at 5-10:
- Agent found "safe eating" strategy
- Good! This is progress
- Let it train longer to find better strategies

### If returns are chaotic (jumping -10 to +20):
- High exploration is working
- Should stabilize after 500 updates
- If not, reduce `max_entropy_coef` to 0.7

### If agent still won't take risks:
- Increase food rewards to +15.0
- Reduce death penalty to -3.0
- Add curiosity bonus for exploration

---

## Quick Reference: New Reward Scale

```
💀 Death: -5.0
🍎 Ate food (size 3): +10 + 5.5 = +15.5
🍎 Ate food (size 10): +10 + 9.0 = +19.0
📍 Per step alive: +0.01
➡️  Moving toward food: +0.2
⬅️  Moving away: -0.05
⚠️  In danger: -1.0
✅ Safe position: +0.1
```

**Expected episode returns:**
- Death quickly: ~-5
- Survived 20 steps: ~0.2
- Ate 1 food: ~15-17
- Ate 2 foods: ~30-35
- Ate 3 foods: ~45-50

---

## Next Steps

1. **Start fresh training** with new rewards
2. **Monitor first 100 updates** - should see more food-seeking
3. **Check returns after 500 updates** - should be 10-20 range
4. **Adjust if needed** - refer to troubleshooting section

Read `REWARD_ANALYSIS.md` for deep understanding of reward design principles!
