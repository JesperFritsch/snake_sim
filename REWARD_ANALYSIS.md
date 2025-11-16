# Reward Function Analysis & Design Guide

## Your Current Reward Function

### Components:
1. **Food Approach Reward** (`_food_approach_reward`)
   - Moving closer to food: +0.3
   - Moving away from food: -0.1
   - No change: 0.0

2. **Food Eat Reward** (`_food_eat_reward`)
   - Ate food: +1.0
   - Didn't eat: 0.0

3. **Length Reward** (`_length_reward`)
   - Grew: +2.0 × (1.0 + (length-2) × 0.1)
   - Shrank: -1.0
   - No change: 0.0

4. **Survival Chance Reward** (`_survival_chance_reward`)
   - Safe area (margin ≥ 0): 0.0
   - Unsafe area (margin < 0): -0.2

5. **Survival Reward** (`_survival_reward`)
   - Alive: +0.05 per step
   - Dead: -20.0

### Typical Reward Per Step:
- **Alive, moving toward food**: 0.05 + 0.3 = **0.35**
- **Alive, standing still**: 0.05 = **0.05**
- **Ate food (length 3→4)**: 0.05 + 1.0 + 2.1 = **3.15**
- **Died**: **-20.0**

---

## 🔴 CRITICAL ISSUES WITH YOUR CURRENT REWARDS

### 1. **The Survival Paradox**
**Problem**: Your agent can get **0.35 per step** just by moving toward food, but risks **-20.0** by dying.

**Math**:
- To break even after dying, snake needs: 20.0 ÷ 0.35 ≈ **57 steps** of safe movement
- Average episode length from your data: ~12-16 steps
- **Agent learns: "Don't risk anything, just survive"**

### 2. **Eating Doesn't Pay Off**
**Reality Check**:
- Eating gives: +3.15 (one-time)
- Just surviving for 63 steps: +3.15 (0.05 × 63)
- **Moving toward food is almost as rewarding as eating!**

### 3. **Death Penalty Way Too High**
The -20.0 death penalty is **dominating** your reward landscape:
```
Best case (ate 5 foods): +15.75
One death: -20.0
Net: Still negative!
```

### 4. **Survival Chance Reward is Weak**
- Unsafe area: -0.2
- Being alive: +0.05
- **Net effect of danger: -0.15** (barely noticeable!)

---

## 🎓 REWARD FUNCTION FUNDAMENTALS

### What Are Rewards?

Rewards are **the only signal** your agent uses to learn. Think of it as:
- **+** = "Yes! Do more of this!"
- **-** = "No! Avoid this!"
- **0** = "I don't care about this"

The agent is trying to **maximize cumulative reward** over time.

### Key Principles:

#### 1. **Reward Scale Matters**
```python
# BAD: Huge death penalty
death = -1000  # Agent becomes paralyzed with fear
survival = +1  # Insignificant compared to death

# GOOD: Balanced scale
death = -5     # Bad, but not catastrophic
eating = +10   # Clearly the best outcome
survival = +0.1 # Small bonus
```

#### 2. **Sparse vs Dense Rewards**

**Sparse Rewards** (Harder to learn):
```python
def sparse_reward(ate_food, died):
    if died:
        return -10
    if ate_food:
        return +10
    return 0  # Most of the time: 0
```
- Pro: Simple, clear objective
- Con: Agent wanders aimlessly, rarely gets signal

**Dense Rewards** (Easier to learn):
```python
def dense_reward(distance_to_food, ate_food, died):
    if died:
        return -10
    if ate_food:
        return +10
    # Shaping: Small reward for progress
    if distance_decreased:
        return +0.5  # Learning signal every step!
    return 0
```
- Pro: Constant feedback guides learning
- Con: Can create unintended behaviors

#### 3. **Reward Shaping**
Adding intermediate rewards to guide learning:

```python
# Without shaping: Agent rarely finds food
reward = +10 if ate_food else 0

# With shaping: Agent learns to hunt
reward = +10 if ate_food else -0.01 * distance_to_food
```

⚠️ **Danger**: Bad shaping can teach wrong behavior!
```python
# BAD: Agent learns to stay near food, not eat it!
reward = +1.0 if near_food else 0
```

#### 4. **Discount Factor (γ = 0.99)**
Your agent values future rewards:
```
Value now = reward₀ + 0.99×reward₁ + 0.99²×reward₂ + ...
```
- γ = 0.99: Plans ~100 steps ahead
- γ = 0.9: Plans ~10 steps ahead
- γ = 0.5: Very short-sighted

---

## 🎯 RECOMMENDED REWARD FUNCTION

### Design Goals:
1. **Eating food is clearly best** (+5 to +10 range)
2. **Death is bad but not paralyzing** (-3 to -5 range)
3. **Encourage food-seeking behavior** (small dense rewards)
4. **Penalize unsafe behavior** (prevent suicide)

### Proposed Rewards:

```python
def improved_rewards():
    # 1. Death penalty (moderate, not paralyzing)
    if died:
        return -5.0
    
    # 2. Eating food (clear winner)
    if ate_food:
        base = 10.0
        length_bonus = (length - 2) * 0.5  # Bonus for bigger snake
        return base + length_bonus
    
    # 3. Food approach (dense guidance)
    if distance_decreased:
        return +0.2
    elif distance_increased:
        return -0.05  # Gentle discouragement
    
    # 4. Survival chance (safety bonus)
    if in_danger:  # margin < 0
        return -0.5  # Make danger more costly
    elif very_safe:  # margin > 10
        return +0.1  # Reward smart positioning
    
    # 5. Survival bonus (very small)
    return +0.01  # Just to break ties
```

### Expected Behavior:
- **Just surviving**: +0.01/step → ~0.15 per episode
- **Eating once**: +10 → Dominates survival reward!
- **Dying**: -5.0 → Bad, but recoverable
- **Moving toward food**: +0.2 → Clear guidance

### Trade-off Analysis:
```
Risk assessment for agent:
- Stay safe for 100 steps: +1.0
- Try to eat (50% success):
  * Success: +10.0
  * Failure: -5.0
  * Expected: 0.5×10 + 0.5×(-5) = +2.5
  
→ Agent learns: "Taking risks to eat is worth it!"
```

---

## 🔧 QUICK FIXES FOR YOUR CURRENT FUNCTION

### Fix #1: Reduce Death Penalty
```python
def _survival_reward(still_alive: bool, current_length: int = 2) -> float:
    if not still_alive:
        return -5.0  # Changed from -20.0
    else:
        return 0.01  # Changed from 0.05
```

### Fix #2: Increase Food Rewards
```python
def _food_eat_reward(ate_food: bool) -> float:
    if ate_food:
        return 10.0  # Changed from 1.0
    return 0.0

def _length_reward(len1: int, len2: int, still_alive: bool) -> float:
    if len2 > len1 and still_alive:
        length_multiplier = 1.0 + (len2 - 2) * 0.5
        return 5.0 * length_multiplier  # Changed from 2.0
    # ... rest unchanged
```

### Fix #3: Strengthen Danger Penalty
```python
def _survival_chance_reward(area_check: AreaCheckResult) -> float:
    if area_check is not None and area_check.margin >= 0:
        return 0.1  # Small safety bonus
    else:
        return -1.0  # Strong danger penalty (from -0.2)
```

---

## 📊 DEBUGGING REWARDS

### Enable Debug Logging
Uncomment this in your code:
```python
print(f"Rewards for snake {s_id}: length={length_reward:.2f}, survival={survival_reward:.2f}, "
        f"food_approach={food_reward:.2f}, food_eat={did_eat_reward:.2f}, "
        f"survival_chance={survival_chance_reward:.2f}, total={total_reward:.2f}")
```

### What to Look For:
1. **Which component dominates?** (Should be food eating!)
2. **Are rewards too sparse?** (Lots of zeros?)
3. **Do good episodes get good rewards?** (Check return = sum of rewards)

### Create a Test Script:
```python
# Simulate common scenarios
scenarios = [
    ("Safe survival", 0.05),           # Expected per step
    ("Moving to food", 0.35),          # Current
    ("Ate food (size 3)", 3.15),       # Current
    ("Died", -20.0),                   # Current - TOO HIGH!
]

for name, reward in scenarios:
    print(f"{name}: {reward:.2f}")
```

---

## 🚀 ADVANCED TECHNIQUES

### 1. **Curriculum Learning**
Start easy, gradually increase difficulty:
```python
# Early training: Big food rewards, small death penalty
if episode < 1000:
    death_penalty = -2.0
    food_reward = 20.0
# Later: More realistic
else:
    death_penalty = -5.0
    food_reward = 10.0
```

### 2. **Reward Normalization**
Your PPO already does this! Returns are normalized to mean=0, std=1.

### 3. **Intrinsic Motivation**
Bonus for exploring new areas:
```python
visited_tiles = set()
if current_tile not in visited_tiles:
    reward += 0.1  # Exploration bonus
    visited_tiles.add(current_tile)
```

### 4. **Potential-Based Shaping** (Guaranteed not to change optimal policy!)
```python
# Define potential function
def potential(state):
    return -distance_to_food  # Negative distance
    
# Shaped reward
reward = base_reward + γ×potential(next_state) - potential(state)
```

---

## ✅ ACTION ITEMS

1. **Immediate**: Try the quick fixes above
2. **Monitor**: Enable debug logging, check reward components
3. **Iterate**: If still stuck, try the full proposed reward function
4. **Advanced**: Consider curriculum learning if needed

The goal is: **Make eating food 10x more rewarding than just surviving!**
