# 🧠 Module 00: Code Explained — Every Line, Every Pattern
> **Ch. 18 — Hands-On ML with Scikit-Learn, Keras & TensorFlow**
> *Visual + line-by-line walkthroughs of every major code block in the chapter*

---

## 📌 Table of Contents
1. [How to Read This Module](#how-to-read)
2. [REINFORCE — Line by Line](#reinforce-code)
3. [Q-Learning — Line by Line](#q-learning-code)
4. [DQN — Every Line Explained](#dqn-code)
5. [Double DQN — The 3-Line Difference](#double-dqn-code)
6. [Dueling DQN — Network Architecture](#dueling-code)
7. [Actor-Critic (A2C) — Line by Line](#ac-code)
8. [PPO — Step-by-Step](#ppo-code)
9. [Pattern Reference Card](#patterns)

---

## 📖 How to Read This Module {#how-to-read}

Every code block below follows this format:

```
# ← Green: WHY this line exists (the concept it implements)
code_line   # ← Blue annotation: WHAT this does mechanically
```

Look for `▶` markers — they indicate the single most important line in each block.

---

## 1. REINFORCE — Line by Line {#reinforce-code}

### The Full Training Loop, Annotated

```python
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import gymnasium as gym

# ─── Setup ──────────────────────────────────────────────────────────────────
env = gym.make("CartPole-v1")
# CartPole: 4 float observations, 2 discrete actions (left/right)

# ─── Policy Network ──────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=[4]),
    #                  ↑ hidden layer — learns nonlinear features of state
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(2, activation="softmax"),
    # ▶ softmax output → action PROBABILITIES that sum to 1.0
    # π_θ(a|s) = softmax(Wx+b) — this IS the policy!
])

optimizer = keras.optimizers.Adam(lr=0.01)
# Adam: adaptive learning rate — converges faster than plain SGD for RL

# ─── Helper: Discount & Normalize Returns ────────────────────────────────────
def discount_rewards(rewards, gamma=0.95):
    """
    Convert raw rewards into discounted returns G_t.

    WHY: REINFORCE needs G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    We compute this by scanning BACKWARDS through the episode.

    Example: rewards=[1,1,1,-1], gamma=0.95
      G_3 = -1
      G_2 = 1 + 0.95×(-1) = 0.05
      G_1 = 1 + 0.95×0.05 = 1.0475
      G_0 = 1 + 0.95×1.0475 = 1.995
    """
    discounted = np.zeros(len(rewards))
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G    # ▶ Bellman-like backward scan
        discounted[t] = G
    return discounted

def normalize(values):
    """
    WHY: Reduce variance by centering returns around 0.
    Actions with G_t > mean get positive gradient (increase prob).
    Actions with G_t < mean get negative gradient (decrease prob).
    The std division makes learning rate invariant to episode length.
    """
    mean = np.mean(values)
    std  = np.std(values) + 1e-8    # +1e-8 prevents divide-by-zero
    return (values - mean) / std

# ─── Data Collection ─────────────────────────────────────────────────────────
def play_episode(env, model):
    """
    Runs one full episode. Returns three parallel lists:
    - observations: what agent saw at each step
    - actions:      what agent did at each step
    - rewards:      what environment gave at each step
    REINFORCE needs the complete episode before updating — Monte Carlo!
    """
    observations, actions, rewards = [], [], []
    obs, _ = env.reset()
    done = False

    while not done:
        # ▶ Forward pass: get action probabilities from policy network
        action_probs = model(obs[np.newaxis]).numpy()[0]
        # obs[np.newaxis]: add batch dimension → shape (1,4) instead of (4,)
        # model(…) → shape (1,2) → .numpy()[0] → shape (2,) → [p_left, p_right]

        # Sample action from the probability distribution
        action = np.random.choice(len(action_probs), p=action_probs)
        # WHY sample, not argmax?
        # Argmax would be deterministic (greedy) — never explores!
        # Sampling naturally explores: high-prob actions chosen more often,
        # but low-prob actions still occasionally tried.

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs

    return np.array(observations), np.array(actions), np.array(rewards)

# ─── Training Step ───────────────────────────────────────────────────────────
@tf.function    # ← Compiles to TF graph: 10-50× faster than eager mode
def train_step(observations, actions, returns):
    """
    The REINFORCE gradient update.

    Mathematical formula:
      ∇_θ J(θ) = E_τ[ Σ_t G_t · ∇_θ log π_θ(a_t|s_t) ]

    In code:
      loss = -mean( G_t · log π_θ(a_t|s_t) )
    We MINIMIZE loss = MAXIMIZE J(θ) because optimizers minimize by default.
    """
    with tf.GradientTape() as tape:
        # Forward pass through the network
        action_probs = model(observations, training=True)
        # Shape: (T, 2) — T timesteps, 2 action probabilities each

        # ▶ The REINFORCE log-probability trick
        # We need log π_θ(a_t|s_t): log prob of the ACTION ACTUALLY TAKEN
        # Not the full distribution — only the chosen action's log prob
        action_mask = tf.one_hot(actions, depth=2)
        # one_hot: action=0 → [1,0], action=1 → [0,1]  Shape: (T, 2)

        log_probs = tf.math.log(action_probs + 1e-8)
        # +1e-8: prevent log(0) = -inf which causes NaN gradients

        selected_log_probs = tf.reduce_sum(log_probs * action_mask, axis=1)
        # Multiply elementwise → zeros out non-chosen action
        # reduce_sum across actions → keeps only log prob of chosen action
        # Shape: (T,) — one log prob per timestep

        # ▶ The policy gradient loss
        loss = -tf.reduce_mean(returns * selected_log_probs)
        # WHY negative? We MAXIMIZE J but TF minimizes → flip sign
        # returns * log_probs:
        #   G_t > 0 → loss negative → gradient pushes prob UP ✓
        #   G_t < 0 → loss positive → gradient pushes prob DOWN ✓

    # Compute and apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ─── Main Training Loop ──────────────────────────────────────────────────────
for episode in range(1000):
    obs_arr, act_arr, rew_arr = play_episode(env, model)

    raw_returns = discount_rewards(rew_arr, gamma=0.95)
    norm_returns = normalize(raw_returns)
    # Normalization is CRUCIAL:
    # Without it: early episodes have G≈9 (short pole), later G≈100 (long)
    # The raw magnitude changes → inconsistent learning rate effectively
    # With it: always zero-mean unit-variance → stable, consistent signal

    loss = train_step(
        tf.constant(obs_arr, dtype=tf.float32),
        tf.constant(act_arr, dtype=tf.int32),
        tf.constant(norm_returns, dtype=tf.float32),
    )

    if episode % 100 == 0:
        print(f"Episode {episode:4d} | Reward: {sum(rew_arr):5.0f} | Loss: {loss:.4f}")
```

---

## 2. Q-Learning (Tabular) — Line by Line {#q-learning-code}

```python
import numpy as np
import gymnasium as gym

env = gym.make("Taxi-v3")
# Taxi-v3: 500 discrete states × 6 actions
# (25 positions × 5 passenger locations × 4 destinations)

# ─── Q-Table Initialization ──────────────────────────────────────────────────
n_states  = env.observation_space.n   # = 500
n_actions = env.action_space.n        # = 6

Q = np.zeros((n_states, n_actions))
# WHY zeros? Optimistic initialization would overestimate initially.
# Zeros = pessimistic start → agent must discover positive values by exploration.

# ─── Hyperparameters ─────────────────────────────────────────────────────────
alpha   = 0.1    # Learning rate: how much to update Q per step
gamma   = 0.99   # Discount factor: how much future rewards matter
epsilon = 1.0    # Exploration rate: start at 100% random
eps_min = 0.01   # Never go below 1% exploration
eps_decay = 0.995 # Multiply epsilon each episode

# ─── Training Loop ───────────────────────────────────────────────────────────
for episode in range(5000):
    state, _ = env.reset()
    done = False

    while not done:
        # ─── ε-greedy Action Selection ──────────────────────────────────────
        if np.random.random() < epsilon:
            action = env.action_space.sample()   # Random: EXPLORE
        else:
            action = np.argmax(Q[state])         # Greedy: EXPLOIT best known
        # WHY ε-greedy?
        # Pure greedy: gets stuck in local optima, never tries new routes
        # Pure random: never learns — just wanders
        # ε-greedy: balances — exploits often but always keeps exploring a bit

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # ─── Q-Learning Update (Bellman equation) ───────────────────────────
        # ▶ THE CORE LINE: Q-learning Bellman update
        td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
        #            ↑           ↑              ↑                  ↑
        #        immediate   discount    BEST next Q-value    0 if terminal
        #         reward      factor    (greedy, not sampled)   (no future!)

        td_error  = td_target - Q[state, action]
        # td_error > 0: state-action was BETTER than expected → increase Q
        # td_error < 0: state-action was WORSE than expected → decrease Q
        # td_error ≈ 0: Q already accurate → no update needed

        Q[state, action] += alpha * td_error
        # alpha=0.1: take a small step toward the target
        # WHY small step? Targets are noisy (random transitions) —
        # large steps would overfit to individual noisy samples

        state = next_state

    # Decay exploration rate
    epsilon = max(eps_min, epsilon * eps_decay)
    # Gradually shift from explore → exploit as Q-table improves
    # At episode 0: 100% random (knows nothing)
    # At episode 1000: ~60% greedy (has learned a lot)
    # At episode 5000: ~1% random (near-optimal policy)

# ─── Evaluation (no exploration) ─────────────────────────────────────────────
epsilon = 0   # Pure greedy — evaluate learned policy
for _ in range(10):
    state, _ = env.reset()
    done = False; total_reward = 0
    while not done:
        action = np.argmax(Q[state])   # Always pick best known action
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Test reward: {total_reward}")
```

---

## 3. DQN — Every Line Explained {#dqn-code}

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gymnasium as gym
from collections import deque
import random

# ─── Replay Buffer ────────────────────────────────────────────────────────────
replay_buffer = deque(maxlen=100_000)
# deque with maxlen: when full, oldest entries are AUTO-REMOVED
# This is the circular buffer — new experience overwrites old

def push(obs, action, reward, next_obs, done):
    """Store one transition tuple in the replay buffer."""
    replay_buffer.append((obs, action, reward, next_obs, done))
    # Stored as tuple — cheap to create, referenced by index later

def sample_batch(batch_size=64):
    """Randomly sample a mini-batch from the replay buffer."""
    batch = random.sample(replay_buffer, batch_size)
    # random.sample: sampling WITHOUT replacement — no duplicate transitions
    # This is the KEY that breaks temporal correlation!
    obs, actions, rewards, next_obs, dones = zip(*batch)
    # zip(*batch) transposes list-of-tuples → tuple-of-lists
    return (tf.constant(np.array(obs),      dtype=tf.float32),
            tf.constant(np.array(actions),  dtype=tf.int32),
            tf.constant(np.array(rewards),  dtype=tf.float32),
            tf.constant(np.array(next_obs), dtype=tf.float32),
            tf.constant(np.array(dones),    dtype=tf.float32))

# ─── Networks ────────────────────────────────────────────────────────────────
def build_dqn(n_inputs, n_outputs):
    """
    Build the Q-network.
    Input:  state observation (n_inputs floats)
    Output: Q-value for EVERY action simultaneously (n_outputs floats)

    WHY output all Q-values at once?
    Efficiency: one forward pass → Q values for all actions → argmax
    Alternative (one per action) would need N forward passes per step!
    """
    return keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=[n_inputs]),
        # relu: most popular activation — fast, no vanishing gradient
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(n_outputs),
        # ▶ LINEAR output (no activation)! Q-values can be ANY real number.
        # Sigmoid would squash to [0,1] — wrong for Q values
        # Tanh would squash to [-1,1] — wrong for Q values
        # Linear: unbounded → correct for Q values in (-∞, +∞)
    ])

online_model = build_dqn(4, 2)    # Updated every training step
target_model = build_dqn(4, 2)    # Updated every C=500 steps (frozen copy)
target_model.set_weights(online_model.get_weights())   # Start identical

optimizer = keras.optimizers.Adam(learning_rate=1e-3)

# ─── Loss and Training ────────────────────────────────────────────────────────
@tf.function
def train_step(obs, actions, rewards, next_obs, dones):
    """
    ONE gradient update of the DQN.
    Uses MEAN SQUARED BELLMAN ERROR as the loss function.
    """
    # ─── Compute Bellman targets (using TARGET network) ────────────────────
    next_q_values = target_model(next_obs)
    # Shape: (batch_size, n_actions) — Q values for all actions at s'
    # WHY target_model, not online_model?
    # If we used online_model, both sides of the loss would move together:
    #   y = r + γ·max Q_θ(s',·)   ← moves as θ updates
    #   ŷ = Q_θ(s,a)               ← also moves as θ updates
    # This creates a moving-target problem → oscillation and divergence
    # target_model is FROZEN → stable regression targets!

    max_next_q = tf.reduce_max(next_q_values, axis=1)
    # max over action dimension → best action's Q-value at next state
    # Shape: (batch_size,)

    # ▶ Bellman target: r + γ · max_a' Q_θ-(s', a')
    targets = rewards + 0.99 * max_next_q * (1.0 - dones)
    #         ↑ immediate     ↑ future      ↑ greedy     ↑ zero if terminal!
    # (1-dones): if done=1 (terminal), future term = 0. Correct!
    # Without this: agent would get γ·max_Q bonus even at terminal → wrong!

    with tf.GradientTape() as tape:
        # ─── Current Q values (using ONLINE network) ─────────────────────
        all_q_values = online_model(obs, training=True)
        # Shape: (batch_size, n_actions)

        # Extract Q value for the action ACTUALLY taken (not all actions)
        action_mask = tf.one_hot(actions, depth=2)
        # one_hot: action=0 → [1,0], action=1 → [0,1]

        q_values = tf.reduce_sum(all_q_values * action_mask, axis=1)
        # Masks out Q values of non-taken actions → only Q(s,a_taken) remains
        # Shape: (batch_size,) — one Q value per sample in batch

        # ─── Mean Squared Bellman Error (loss) ──────────────────────────
        loss = tf.reduce_mean(tf.square(targets - q_values))
        # tf.square: element-wise squaring of (target - prediction)
        # tf.reduce_mean: average over the batch → scalar loss
        # This is standard MSE regression: predict targets from observations

    grads = tape.gradient(loss, online_model.trainable_variables)
    # ▶ Gradient clipping: prevents catastrophic updates from rare large rewards
    grads = [tf.clip_by_norm(g, 1.0) for g in grads]
    # clip_by_norm: if ||g|| > 1.0, scale down g so ||g|| = 1.0
    # Without clipping: one big reward can give ||g||=100 → destroys network

    optimizer.apply_gradients(zip(grads, online_model.trainable_variables))
    return loss

# ─── Main Training Loop ───────────────────────────────────────────────────────
env      = gym.make("CartPole-v1")
epsilon  = 1.0    # Start with full exploration
eps_min  = 0.02
eps_decay = 0.997
C        = 500    # Hard target network update frequency
step_count = 0

for episode in range(600):
    obs, _ = env.reset()
    total_reward = 0

    while True:
        step_count += 1

        # ─── ε-greedy action selection ──────────────────────────────────
        if random.random() < epsilon:
            action = env.action_space.sample()    # Explore: random
        else:
            q_vals = online_model(obs[np.newaxis]).numpy()[0]
            action = int(np.argmax(q_vals))       # Exploit: greedy
        # During training: need exploration (ε > 0)
        # During evaluation: pure greedy (ε = 0)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        push(obs, action, reward, next_obs, float(done))
        # Store to replay buffer EVERY step — raw experience

        obs = next_obs
        total_reward += reward

        # ─── Training: only when buffer has enough samples ──────────────
        if len(replay_buffer) >= 1000:
            # WHY 1000 warmup steps?
            # If we train from step 1, all batches are nearly identical
            # (same game start state repeated). Need diversity first!
            batch = sample_batch(64)
            train_step(*batch)

        # ─── Hard target network update ──────────────────────────────────
        if step_count % C == 0:
            target_model.set_weights(online_model.get_weights())
            # ▶ Copy ALL weights from online → target network
            # This is a "hard update" — instant copy every C steps
            # Alternative: "soft update" τ=0.005: θ- = (1-τ)θ- + τθ
            # Soft update is smoother but requires tuning τ

        if done:
            break

    epsilon = max(eps_min, epsilon * eps_decay)

print("Training complete!")
```

---

## 4. Double DQN — The 3-Line Difference {#double-dqn-code}

```python
# ─── VANILLA DQN target computation ─────────────────────────────────────────
# PROBLEM: Same network selects AND evaluates → noise inflates max
next_q_values = target_model(next_obs)
max_next_q    = tf.reduce_max(next_q_values, axis=1)    # ← selects AND evaluates
targets       = rewards + gamma * max_next_q * (1 - dones)

# ─── DOUBLE DQN target computation (3-line change) ────────────────────────
# SOLUTION: Online selects, target evaluates → two independent estimates

# Step 1: Online network SELECTS the best action at s'
next_q_online = online_model(next_obs)
best_actions  = tf.argmax(next_q_online, axis=1)
# best_actions: the action index that online_model thinks is best
# Shape: (batch_size,)  values in {0, 1, ..., n_actions-1}

# Step 2: Target network EVALUATES that selected action at s'
next_q_target = target_model(next_obs)
# Shape: (batch_size, n_actions)

best_actions_oh = tf.one_hot(best_actions, depth=n_actions)
# Convert action indices to one-hot mask for efficient extraction

max_next_q = tf.reduce_sum(next_q_target * best_actions_oh, axis=1)
# ▶ Extract target network's Q-value for the online-selected action
# Shape: (batch_size,) — evaluating online's choice with target's values

targets = rewards + gamma * max_next_q * (1 - dones)
# Same formula as vanilla DQN — only the max_next_q computation changed!

# WHY this works:
# Vanilla:  y = r + γ · max_a' Q_θ-(s', a')    [θ- picks AND evaluates]
# Double:   y = r + γ · Q_θ-(s', argmax_a' Q_θ(s',a'))
#                           ↑ evaluate       ↑ select
# Two networks with INDEPENDENT noise → upward bias averages out
```

---

## 5. Dueling DQN — Architecture {#dueling-code}

```python
import tensorflow as tf
from tensorflow import keras

def build_dueling_dqn(n_inputs, n_actions):
    """
    Dueling DQN: separates state value V(s) from action advantages A(s,a).

    Key insight:
        Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

    WHY separate V and A?
    Many states are trivially good or bad regardless of action choice.
    In these states, learning V(s) accurately is more important than A(s,a).
    Separate streams let V(s) update from ALL actions' Q estimates,
    while A(s,a) only updates from the specific action taken.
    → More training signal per step → faster convergence.
    """
    inputs = keras.Input(shape=[n_inputs])

    # ─── Shared feature extractor (common trunk) ────────────────────────────
    x = keras.layers.Dense(128, activation="relu")(inputs)
    x = keras.layers.Dense(128, activation="relu")(x)
    # Both streams benefit from the same learned state representation

    # ─── Value stream: estimates V(s) ──────────────────────────────────────
    v = keras.layers.Dense(64, activation="relu")(x)
    v = keras.layers.Dense(1)(v)   # Single scalar: V(s)
    # Linear output — V(s) can be any value

    # ─── Advantage stream: estimates A(s,a) for all actions ────────────────
    a = keras.layers.Dense(64, activation="relu")(x)
    a = keras.layers.Dense(n_actions)(a)   # One per action: A(s,a)
    # Linear output — advantages can be positive or negative

    # ─── Combine: Q(s,a) = V(s) + A(s,a) - mean(A) ─────────────────────────
    # ▶ Why subtract mean(A)?
    # Without subtraction, V and A are not individually identifiable:
    #   V(s)=5, A(a)=0 gives same Q as V(s)=0, A(a)=5
    # With mean subtraction, A is FORCED to have zero mean:
    #   average advantage = 0 → the advantage IS relative to average
    # This makes V and A uniquely determined → stable learning
    q = v + (a - tf.reduce_mean(a, axis=1, keepdims=True))
    # keepdims=True: keeps shape (batch, 1) for broadcasting against (batch, n_actions)
    # tf.reduce_mean(a, axis=1): mean over actions → shape (batch,) → (batch,1)

    return keras.Model(inputs=inputs, outputs=q)

model = build_dueling_dqn(n_inputs=4, n_actions=2)
```

---

## 6. Actor-Critic (A2C) — Line by Line {#ac-code}

```python
@tf.function
def a2c_train_step(obs_batch, action_batch, return_batch):
    """
    One gradient update for Actor-Critic.

    Three losses computed simultaneously:
    1. Actor loss:   push policy toward better actions (via advantage)
    2. Critic loss:  improve value function accuracy
    3. Entropy loss: prevent premature collapse to deterministic policy
    """
    with tf.GradientTape() as tape:
        # ─── Forward pass through shared network ──────────────────────────
        logits, values = model(obs_batch, training=True)
        # logits: raw scores (before softmax) for each action, shape (T, n_a)
        # values: state value estimates V(s), shape (T, 1)

        values = tf.squeeze(values, axis=1)
        # tf.squeeze: remove the trailing size-1 dimension → shape (T,)

        # ─── Compute advantage = R_t - V(s_t) ────────────────────────────
        advantages = return_batch - values
        # return_batch: actual discounted returns computed from episode
        # values: critic's ESTIMATED V(s)
        # advantages > 0: action led to BETTER outcome than V predicted → increase prob
        # advantages < 0: action led to WORSE outcome → decrease prob
        # ▶ This is the key insight: advantage = how much BETTER than expected

        # ─── Actor Loss ───────────────────────────────────────────────────
        action_probs  = tf.nn.softmax(logits)
        log_probs_all = tf.nn.log_softmax(logits)
        # log_softmax is numerically MORE STABLE than tf.math.log(softmax(x))
        # softmax can output 0 for very negative logits → log(0)=-inf → NaN!
        # log_softmax = logits - log(sum(exp(logits))) — never hits -inf

        action_mask      = tf.one_hot(action_batch, depth=n_actions)
        action_log_probs = tf.reduce_sum(log_probs_all * action_mask, axis=1)
        # Extract log prob of the TAKEN action (mask out others)
        # Shape: (T,) — one log prob per timestep

        # ▶ Policy gradient: negative because we maximize J (optimizer minimizes)
        actor_loss = -tf.reduce_mean(advantages * action_log_probs)
        # advantages is detached from graph here — treated as a constant weight
        # This is the policy gradient theorem applied with advantage as baseline

        # ─── Critic Loss ──────────────────────────────────────────────────
        critic_loss = tf.reduce_mean(tf.square(return_batch - values))
        # Standard MSE: (actual return - predicted value)²
        # Minimizing this makes V(s) accurate → better advantage estimates

        # ─── Entropy Bonus (prevents premature exploitation) ──────────────
        entropy = -tf.reduce_mean(
            tf.reduce_sum(action_probs * log_probs_all, axis=1)
        )
        # H(π) = -Σ_a π(a|s) · log π(a|s)
        # High entropy: uniform distribution → maximum exploration
        # Low entropy: peaked distribution → deterministic (pure exploitation)
        # Adding entropy to the objective: PENALIZES low-entropy policies
        # → agent resists collapsing to greedy before finding the true optimum

        # ─── Combined Loss ────────────────────────────────────────────────
        total_loss = (actor_loss
                      + 0.5 * critic_loss    # c_v=0.5: critic loss scale
                      - 0.01 * entropy)      # c_e=0.01: entropy bonus scale
        # WHY 0.5 for critic? Actor and critic have different scales.
        #   Actor loss ≈ O(1), critic loss ≈ O(advantage²) which can be large.
        #   0.5 balances their magnitudes in the combined gradient.

    grads = tape.gradient(total_loss, model.trainable_variables)
    # ▶ Gradient clipping: RL gradients can spike, clipping prevents NaN/diverge
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return actor_loss, critic_loss, entropy
```

---

## 7. PPO — Step-by-Step {#ppo-code}

```python
@tf.function
def ppo_train_step(states, actions, old_log_probs, advantages, returns):
    """
    PPO update: one mini-batch gradient step with clipped objective.

    The crucial difference from standard policy gradient:
    We compute the RATIO r(θ) = π_new / π_old and CLIP it.
    This prevents the policy from changing too dramatically per update.
    """
    with tf.GradientTape() as tape:
        logits, values = model(states, training=True)
        values = tf.squeeze(values)

        # ─── Current log probabilities ────────────────────────────────────
        log_probs_all = tf.nn.log_softmax(logits)
        action_mask   = tf.one_hot(actions, depth=n_actions)
        log_probs     = tf.reduce_sum(log_probs_all * action_mask, axis=1)
        # Shape: (batch_size,) — log π_θ_new(a_t|s_t)

        # ─── Probability ratio: π_new / π_old ────────────────────────────
        ratio = tf.exp(log_probs - old_log_probs)
        # WHY exp(log_new - log_old) instead of new/old directly?
        # Numerical stability: ratios can get very large/small (e.g., 0.001 or 1000)
        # Computing in log space first, then exponentiating is much more stable.
        # log(π_new) - log(π_old) = log(π_new / π_old) → exp → π_new/π_old
        # ratio > 1: new policy assigns MORE probability to this action
        # ratio < 1: new policy assigns LESS probability to this action
        # ratio = 1: no change from old policy

        # ─── PPO Clipped Objective ────────────────────────────────────────
        eps    = 0.2   # Clip range: allow ratio in [0.8, 1.2]
        clipped = tf.clip_by_value(ratio, 1-eps, 1+eps)
        # clip_by_value: values below 1-eps clamped to 1-eps,
        #                values above 1+eps clamped to 1+eps

        # ▶ The core PPO formula: min of unclipped and clipped objectives
        actor_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantages, clipped * advantages)
        )
        # WHY minimum?
        # Case: advantages > 0 (good action, we want to increase prob)
        #   ratio > 1+eps: unclipped = ratio*A > clipped*A → min = clipped → STOP
        #   We've already increased prob enough — clipping prevents over-increase
        # Case: advantages < 0 (bad action, we want to decrease prob)
        #   ratio < 1-eps: ratio*A (more negative) < clipped*A → min = ratio*A
        #   Wait — we actually DON'T clip when decreasing prob of bad actions?
        #   Actually both are negative, and min picks the less negative → clipped
        # Net effect: update stays within [1-eps, 1+eps] trust region

        # ─── Critic Loss ──────────────────────────────────────────────────
        critic_loss = tf.reduce_mean(tf.square(returns - values))

        # ─── Entropy ──────────────────────────────────────────────────────
        probs   = tf.nn.softmax(logits)
        entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs_all, axis=1))

        total_loss = actor_loss + 0.5*critic_loss - 0.01*entropy

    grads = tape.gradient(total_loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 0.5) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Monitor approximate KL divergence between old and new policy
    approx_kl = tf.reduce_mean(old_log_probs - log_probs)
    # If approx_kl > 0.02, the policy has changed too much → stop PPO epochs early
    return total_loss, approx_kl


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    Generalized Advantage Estimation (GAE).

    Pure TD(0): A_t = r_t + γ·V(s_{t+1}) - V(s_t)  [low var, high bias]
    Pure MC:    A_t = G_t - V(s_t)                   [zero bias, high var]
    GAE:        A_t = Σ_{l=0}^∞ (γλ)^l · δ_{t+l}   [adjustable bias/var]

    λ=0.95 is the sweet spot: used in every major PPO implementation.
    """
    advantages = np.zeros_like(rewards)
    last_gae   = 0.0

    for t in reversed(range(len(rewards))):
        # Compute TD error δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
        if t == len(rewards) - 1:
            next_val = last_value * (1 - dones[t])
        else:
            next_val = values[t+1] * (1 - dones[t])

        delta = rewards[t] + gamma * next_val - values[t]

        # ▶ Recursive GAE: A_t = δ_t + γλ · A_{t+1}
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values   # Q̂(s,a) ≈ A(s,a) + V(s)
    return advantages, returns
```

---

## 8. Pattern Reference Card {#patterns}

### The 5 Core RL Code Patterns

```
PATTERN 1: LOG-PROB TRICK (Policy Gradient)
  log_probs = log_softmax(logits)
  action_lp = reduce_sum(log_probs * one_hot(actions, n), axis=1)
  loss = -mean(returns * action_lp)    ← ALWAYS negative (we maximize)

PATTERN 2: BELLMAN TARGET (Q-Learning / DQN)
  target = r + gamma * max Q_target(s') * (1 - done)
  loss   = mean_square(target - Q_online(s, a))
  NEVER include gradient through target — it's a constant label!

PATTERN 3: ADVANTAGE (Actor-Critic)
  delta     = r + gamma * V(s') - V(s)   ← TD error = advantage
  actor_loss  = -delta * log_pi(a|s)
  critic_loss = delta ** 2

PATTERN 4: PPO CLIP
  ratio   = exp(log_pi_new - log_pi_old)
  L_clip  = min(ratio*A, clip(ratio, 1-e, 1+e)*A)   ← e=0.2
  loss    = -mean(L_clip)

PATTERN 5: GRADIENT CLIPPING (always do this in RL!)
  grads = tape.gradient(loss, model.trainable_variables)
  grads = [clip_by_norm(g, max_norm=0.5) for g in grads]
  optimizer.apply_gradients(zip(grads, model.variables))
```

### Common Shapes Cheat Sheet

```
Observation:  (batch, n_obs)      e.g. (64, 4)   CartPole
Action:       (batch,)            e.g. (64,)      integer actions
Logits:       (batch, n_actions)  e.g. (64, 2)    raw scores
Log probs:    (batch, n_actions)  e.g. (64, 2)    after log_softmax
Action logp:  (batch,)            e.g. (64,)      selected action only
Values:       (batch,)            e.g. (64,)      after tf.squeeze
Advantages:   (batch,)            e.g. (64,)      returns - values
```

---

**🔗 Module Index →** [notes.md](../notes.md)
