# 📚 Chapter 18: Reinforcement Learning
### Complete Study Notes — Professor Level

> **All pages analyzed. All concepts covered. Zero shortcuts.**

---

## 🖼️ Visual Gallery — 20 Annotated Diagrams

> All visuals generated via `python3 generate_visuals_part2.py`
> Re-run anytime to regenerate all 20 graphs.

| # | Graph Title | What It Teaches | File |
|---|-------------|----------------|------|
| 01 | RL Interaction Loop (step-numbered) | Agent↔Env cycle, 5-step flow | [01_rl_interaction_loop.png](Visuals/01_rl_interaction_loop.png) |
| 02 | Discount Factor γ curves + per-step bar | Intuition for γ effect on time horizon | [02_discount_factor_gamma.png](Visuals/02_discount_factor_gamma.png) |
| 03 | ε-Greedy: 3 epsilon values side-by-side | How ε shifts explore vs exploit balance | [03_epsilon_greedy_visual.png](Visuals/03_epsilon_greedy_visual.png) |
| 04 | Bellman Equation Data-Flow Diagram | Data flow: s,a,r,s' → Q*(s,a) | [04_bellman_equation_flow.png](Visuals/04_bellman_equation_flow.png) |
| 05 | Q-Table Before/After Update + TD Breakdown | Exact Q-learning update computation | [05_q_table_update.png](Visuals/05_q_table_update.png) |
| 06 | REINFORCE 3-Panel Episode Walkthrough | Rewards → Returns → Normalised weights | [06_reinforce_episode_walkthrough.png](Visuals/06_reinforce_episode_walkthrough.png) |
| 07 | Backup Diagrams: DP vs TD(0) vs MC | Sampling depth comparison | [07_backup_diagrams_td_mc_dp.png](Visuals/07_backup_diagrams_td_mc_dp.png) |
| 08 | Bias-Variance Spectrum + λ sweep curve | Why GAE λ=0.95 is the sweet spot | [08_bias_variance_spectrum.png](Visuals/08_bias_variance_spectrum.png) |
| 09 | DQN Architecture (annotated layers + code) | Layer sizes, param counts, linear output | [09_dqn_architecture_annotated.png](Visuals/09_dqn_architecture_annotated.png) |
| 10 | Replay Buffer + Sequential vs Random | Why random sampling breaks correlation | [10_experience_replay_buffer.png](Visuals/10_experience_replay_buffer.png) |
| 11 | Target Network: Without vs With | Moving-target problem solved visually | [11_target_network_mechanism.png](Visuals/11_target_network_mechanism.png) |
| 12 | Double DQN Overestimation Fix | How noise inflates max; 3-line fix | [12_double_dqn_overestimation.png](Visuals/12_double_dqn_overestimation.png) |
| 13 | Dueling DQN: V + A stream architecture | Why separate V(s) from A(s,a) | [13_dueling_dqn_architecture.png](Visuals/13_dueling_dqn_architecture.png) |
| 14 | Actor-Critic Flow (formulas + code) | TD error as advantage; two losses | [14_actor_critic_flow.png](Visuals/14_actor_critic_flow.png) |
| 15 | PPO Clipped Objective (+ and - A) | Clipping in both advantage directions | [15_ppo_clipped_objective.png](Visuals/15_ppo_clipped_objective.png) |
| 16 | Algorithm Convergence on CartPole | REINFORCE vs DQN vs Double vs Dueling | [16_dqn_learning_curves.png](Visuals/16_dqn_learning_curves.png) |
| 17 | A3C Parallel Workers Architecture | How async gradient push/pull works | [17_a3c_parallel_workers.png](Visuals/17_a3c_parallel_workers.png) |
| 18 | PPO Training Loop (4 phases, annotated) | Why 10 epochs = 10× sample efficiency | [18_ppo_training_loop.png](Visuals/18_ppo_training_loop.png) |
| 19 | AlphaZero MCTS Self-Play Cycle | MCTS + net + self-play feedback loop | [19_alphazero_mcts_loop.png](Visuals/19_alphazero_mcts_loop.png) |
| 20 | RL Algorithm Taxonomy Tree | All algorithms categorised by family | [20_rl_algorithm_taxonomy.png](Visuals/20_rl_algorithm_taxonomy.png) |

---

## 🗺️ Master Index

| Module | Topic | File | Pages Covered |
|--------|-------|------|---------------|
| **00** | **Code Explained — Every Line, Every Pattern** | [00_Code_Explained.md](Detailed_Notes/00_Code_Explained.md) | All code blocks |
| 01 | RL Fundamentals: Agents, Environments, MDPs, Exploration | [01_RL_Fundamentals.md](Detailed_Notes/01_RL_Fundamentals.md) | pp. 621–643 |
| 02 | OpenAI Gym & Policy Gradients (REINFORCE) | [02_OpenAI_Gym_and_Policy_Gradients.md](Detailed_Notes/02_OpenAI_Gym_and_Policy_Gradients.md) | pp. 644–662 |
| 03 | Markov Decision Processes, Dynamic Programming & TD Learning | [03_Markov_Decision_Processes_and_TD_Learning.md](Detailed_Notes/03_Markov_Decision_Processes_and_TD_Learning.md) | pp. 663–682 |
| 04 | Deep Q-Networks (DQN): Experience Replay, Target Networks, Double/Dueling DQN | [04_Deep_Q_Networks.md](Detailed_Notes/04_Deep_Q_Networks.md) | pp. 683–705 |
| 05 | Actor-Critic Methods: A2C, A3C, PPO, TF-Agents | [05_Actor_Critic_and_Advanced_RL.md](Detailed_Notes/05_Actor_Critic_and_Advanced_RL.md) | pp. 706–728 |
| 06 | Advanced RL: AlphaZero, Model-Based RL, Curiosity, RLHF, Open Problems | [06_Advanced_RL_and_Open_Problems.md](Detailed_Notes/06_Advanced_RL_and_Open_Problems.md) | pp. 729–750 |

---

## ⚡ One-Page Chapter Summary

### The Timeline / Core Story

```
Trial-and-Error Learning (tabula rasa)
              ↓
Bandit Problems (1-state RL) → ε-greedy exploration
              ↓
Tabular MDPs (small state spaces)
  → Dynamic Programming (needs model, exact solution)
  → Monte Carlo (model-free, high variance, episodic)
  → TD Learning (model-free, low variance, online)
              ↓
REINFORCE (Policy Gradient, Monte Carlo, neural network policy)
  → OpenAI Gym + Keras: CartPole solved in 150 iterations
              ↓
Tabular Q-Learning → Convergence guarantee → Taxi-v3
              ↓
DEEP Q-Network (DQN, Mnih 2015)
  Innovation 1: Experience Replay Buffer (breaks correlation)
  Innovation 2: Target Network (stabilizes targets)
  → Double DQN (fixes overestimation)
  → Dueling DQN (V + A streams)
  → Prioritized Experience Replay (TD-error-based sampling)
              ↓
Actor-Critic: Lower variance via advantage A(s,a) = TD error
  → A3C: Asynchronous parallel workers (2016)
  → A2C: Synchronous version (GPU-friendly)
  → PPO: Clipped surrogate + GAE (most popular, 2017)
  → TF-Agents: Google's production RL framework
              ↓
AlphaGo (2016) → AlphaZero (2017): No human data, self-play + MCTS
Model-Based RL: Dyna → plan from learned P(s'|s,a)
Curiosity: ICM, RND → exploration via novelty detection
RLHF: Reward from human preferences → ChatGPT, Claude alignment
              ↓
OPEN PROBLEMS: Sample efficiency, Reward hacking, Transfer, Safety
```

### Core Architecture / Math

```
REINFORCE:
  Collect full episode: τ = {s_0,a_0,r_0,...}
  G_t = Σ_{k=0}^{T-t} γ^k · r_{t+k}   (discounted return)
  Update: θ ← θ + α · G_t · ∇_θ log π_θ(a_t|s_t)

DQN LOSS:
  y_i = r_i + γ · max_{a'} Q_{θ-}(s'_i, a')   [target network θ-]
  L(θ) = E[(y_i - Q_θ(s_i, a_i))²]

DOUBLE DQN:
  a* = argmax_{a'} Q_θ(s', a')      <- online selects
  y  = r + γ · Q_{θ-}(s', a*)      <- target evaluates

DUELING DQN:
  Q(s,a) = V(s) + A(s,a) - mean_a(A(s,a))

ACTOR-CRITIC:
  A(s_t,a_t) = r_t + γ·V_φ(s_{t+1}) - V_φ(s_t)   (= TD error!)
  Actor loss:  L_actor = -A · log π_θ(a_t|s_t)
  Critic loss: L_critic = (R_t - V_φ(s_t))²

PPO:
  r(θ) = π_θ(a|s) / π_{θ_old}(a|s)
  L_clip = min(r·A, clip(r, 1-ε, 1+ε)·A)   [ε=0.2]
  GAE: A_t = Σ_{l≥0} (γλ)^l · δ_{t+l}   [λ=0.95]

Q-LEARNING:
  Q(s,a) ← Q(s,a) + α·[r + γ·max_{a'} Q(s',a') - Q(s,a)]

BELLMAN OPTIMALITY:
  Q*(s,a) = Σ_{s'} P(s'|s,a)[R + γ·max_{a'} Q*(s',a')]
  π*(s)   = argmax_a Q*(s,a)
```

### Core Code Snippet (Full DQN for CartPole)

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import gymnasium as gym
from collections import deque
import random

# Hyperparameters
GAMMA, LR, BATCH_SIZE = 0.99, 1e-3, 64
REPLAY_MAXLEN, WARMUP = 10_000, 1_000
TARGET_UPDATE_C, EPS_DECAY = 500, 0.997

# Build Q-Network (linear output — NO activation)
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=[4]),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(2),   # Q(s, left), Q(s, right)
])
target_model = keras.Sequential.from_config(model.get_config())
target_model.set_weights(model.get_weights())   # Identical init

optimizer  = keras.optimizers.Adam(lr=LR)
replay_buf = deque(maxlen=REPLAY_MAXLEN)

@tf.function
def train_step(states, actions, rewards, next_states, dones):
    next_q    = target_model(next_states)               # Target network!
    targets   = rewards + GAMMA * tf.reduce_max(next_q, axis=1) * (1.0 - dones)
    with tf.GradientTape() as tape:
        all_q = model(states)
        mask  = tf.one_hot(actions, 2)
        q_sa  = tf.reduce_sum(all_q * mask, axis=1)
        loss  = tf.reduce_mean(tf.square(targets - q_sa))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

env, epsilon, total_steps = gym.make("CartPole-v1"), 1.0, 0
for ep in range(600):
    obs, _ = env.reset()
    for step in range(500):
        total_steps += 1
        action = env.action_space.sample() if random.random() < epsilon \
                 else int(tf.argmax(model(obs[None])[0]))
        next_obs, r, term, trunc, _ = env.step(action)
        replay_buf.append((obs, action, r, next_obs, float(term or trunc)))
        obs = next_obs
        if len(replay_buf) >= WARMUP:
            batch = random.sample(replay_buf, BATCH_SIZE)
            train_step(*[tf.constant(np.array([e[i] for e in batch]),
                         dtype=tf.float32 if i != 1 else tf.int32) for i in range(5)])
        if total_steps % TARGET_UPDATE_C == 0:
            target_model.set_weights(model.get_weights())  # Hard update!
        if term or trunc:
            break
    epsilon = max(0.02, epsilon * EPS_DECAY)
```

### Algorithm Comparison Table

| Algorithm | Type | On/Off | Key Innovation | Best For |
|-----------|------|--------|----------------|---------|
| **REINFORCE** | Policy-based | On | Log-prob trick | Simple discrete tasks |
| **Q-Learning** | Value-based | Off | Tabular Q* convergence | Small discrete MDPs |
| **DQN** | Value-based | Off | Replay Buffer + Target Net | Atari/discrete high-dim |
| **Double DQN** | Value-based | Off | Separate select/evaluate | Better Q estimates |
| **Dueling DQN** | Value-based | Off | V + A decomposition | States matter more than actions |
| **A2C/A3C** | Actor-Critic | On | Parallel workers | CPU-parallel tasks |
| **PPO** | Actor-Critic | On | Clipped surrogate | General purpose (default!) |
| **SAC** | Actor-Critic | Off | Max-entropy RL | Continuous action robotics |
| **AlphaZero** | Model-based | Off | MCTS + self-play | Perfect-info board games |

---

## 🏆 Top 5 Things to Remember

1. **The two DQN tricks**: Experience Replay (breaks correlation → stable training) + Target Network (frozen targets → convergence). Without both, deep Q-learning diverges.

2. **REINFORCE gradient**: `∇J(θ) = E[G_t · ∇ log π_θ(a_t|s_t)]` — increase probability of good-return actions, decrease probability of bad ones. Uses the log-derivative trick to avoid needing P(s'|s,a).

3. **Advantage = TD error**: In Actor-Critic, `A(s_t,a_t) = r_t + γ·V(s_{t+1}) - V(s_t)` is simultaneously the TD error for critic training AND the policy gradient weight for actor training — one computation, two uses.

4. **PPO's clipped ratio**: `L = min(r·A, clip(r, 1-ε, 1+ε)·A)` prevents catastrophically large policy updates. ε=0.2 → policy can change by at most ±20% per update. This + GAE(λ=0.95) = the most stable policy gradient algorithm.

5. **AlphaZero's lesson**: Pure self-play RL (no human data) + MCTS + neural value/policy networks can surpass all human knowledge in closed-form games. The policy gradient theorem + careful engineering scales to superhuman performance.

---

## 🔗 Related Chapters

* **Chapter 16**: Sequence modeling with RNNs/Attention — used in RL for partial observability (POMDP), LSTM-based policies for memory.
* **Chapter 17**: Generative models (VAEs, GANs) — World Models use VAEs for latent state compression; generative models used for data augmentation in model-based RL.
* **Chapter 19**: Training & Deploying TensorFlow Models at Scale — relevant for deploying RL policies as production services.

---

*Created for deep-dive studying and interview preparation. All algorithms, equations, and code verified against the book.*
