# 🤖 Module 01: Reinforcement Learning Fundamentals
> **Ch. 18 — Hands-On ML with Scikit-Learn, Keras & TensorFlow (Aurélien Géron)**

---

## 📌 Table of Contents
1. [Start Here: The Big Picture](#big-picture)
2. [Core Vocabulary: Agents, Environments, Rewards](#vocabulary)
3. [Observations, States & Partial Observability](#observations)
4. [Policies: Deterministic vs Stochastic](#policies)
5. [Returns: Discounted Cumulative Reward](#returns)
6. [Markov Decision Processes (MDPs)](#mdps)
7. [The Credit Assignment Problem](#credit-assignment)
8. [Exploration vs Exploitation](#exploration)
9. [RL Algorithm Taxonomy](#taxonomy)
10. [Common Beginner Mistakes](#mistakes)
11. [Interview Q&A](#interview)
12. [⚡ One-Page Flash Card](#revision)

---

## 🌍 Start Here: The Big Picture {#big-picture}

> **TL;DR:** Reinforcement Learning is an autonomous learning paradigm where an **agent** learns optimal behavior through **trial-and-error interaction** with an **environment**, guided only by a scalar **reward signal** — no labeled dataset required. It is the framework behind AlphaGo, ChatGPT's RLHF alignment, robotic locomotion, and game-playing agents.

**The Real-World Analogy 🐕:**
Think of training a dog. You don't explain *why* "sit" is correct — you simply reward (treat) correct behavior and ignore or correct wrong behavior. Over thousands of repetitions, the dog builds an internal model of "what actions in what situations lead to rewards." Reinforcement Learning formalizes this exact process mathematically: the dog is the **agent**, the training session is the **environment**, the treat is the **reward**, and "sitting when commanded" is the **policy** being learned.

> [!IMPORTANT]
> Unlike supervised learning (learn from labels) and unsupervised learning (find structure in data), RL learns from **interaction**. There are no pre-packaged (input, correct_output) pairs — the agent must discover which actions are good by trying them and observing consequences.

---

## 🔍 1. Core Vocabulary: Agents, Environments, Rewards {#vocabulary}

### The RL Loop


![RL Interaction Loop](../Visuals/01_rl_interaction_loop.png)

| Term | Definition | Example (CartPole) |
|------|-----------|-------------------|
| **Agent** | The learner/decision-maker | Neural network controller |
| **Environment** | The world the agent interacts with | CartPole physics simulator |
| **State `s_t`** | Complete description of environment | Cart position, velocity, pole angle, angular velocity |
| **Observation `o_t`** | What the agent actually sees | Same as state in CartPole (fully observable) |
| **Action `a_t`** | Decision made by agent | Push left (0) or push right (1) |
| **Reward `r_t`** | Scalar signal of immediate feedback | +1 for every step pole stays up |
| **Episode** | One full run from start to terminal state | One game of CartPole until pole falls |
| **Policy π** | Agent's strategy: state → action | Neural network outputting action probabilities |

### Reward Engineering: The Most Critical Design Choice

The reward function defines what "success" means. Poor reward design is the most common reason RL systems fail in practice:

| Bad Reward Design | Consequence |
|-------------------|-------------|
| Reward only at episode end | **Sparse reward problem**: agent can't learn which of 1000 actions helped |
| Reward for intermediate proxy goals | **Reward hacking**: agent optimizes the proxy, not the true goal |
| Poorly shaped reward | **Local optima**: agent gets stuck in suboptimal behavior |

> [!TIP]
> **Reward Shaping**: Add intermediate rewards that hint at progress toward the goal. For example, in a maze, give small rewards for moving closer to the exit. This dramatically speeds up learning but must be done carefully to avoid misalignment.

---

## 🔍 2. Observations, States & Partial Observability {#observations}

### Fully Observable vs Partially Observable Environments

```
FULLY OBSERVABLE (MDP):
  o_t == s_t  --> Agent sees complete state
  Example: Board games (Go, Chess), CartPole
  Agent can make optimal decisions purely from current observation

PARTIALLY OBSERVABLE (POMDP):
  o_t is subset of s_t  --> Agent sees incomplete information
  Example: Poker (hidden opponent cards), Robotics (no global view)
  Agent must maintain MEMORY of past observations to infer hidden state
```

| Property | Fully Observable (MDP) | Partially Observable (POMDP) |
|----------|----------------------|------------------------------|
| **State visibility** | Agent sees full state | Agent sees partial observation |
| **Algorithm complexity** | Simpler (Q-Learning, A3C) | Harder (requires memory/RNN) |
| **Examples** | CartPole, Atari (single frame) | Poker, real-world robotics |
| **Architecture** | Standard MLP/CNN | RNN/LSTM over history |

> [!NOTE]
> Even Atari games are often treated as **partially observable** because a single frame doesn't reveal velocity (you can't tell if the ball is moving left or right). The DQN paper solved this by **stacking 4 consecutive frames** as input — a simple form of short-term memory.

### Action Spaces

| Type | Description | Example | Algorithm |
|------|-------------|---------|-----------|
| **Discrete** | Finite, countable actions | {Left, Right, Jump} — Atari | DQN, A3C, PPO |
| **Continuous** | Infinite real-valued actions | Torque in [-1,1]^3 — Robotics | DDPG, SAC, TD3 |
| **Multi-discrete** | Multiple discrete dims | Steering x Throttle | Multi-head policies |

---

## 🔍 3. Policies: Deterministic vs Stochastic {#policies}

A **policy** π is the agent's decision-making function:

Deterministic: π(s) = a  (maps state to single action)

Stochastic: π(a|s) = P(A_t = a | S_t = s)  (outputs probability distribution over actions)

### Why Stochastic Policies?

1. **Exploration**: A deterministic policy always picks the same action → never discovers better alternatives.
2. **Partial observability**: When state is ambiguous, randomizing is optimal (e.g., mixed strategy in poker).
3. **Multi-agent environments**: Predictable policies are exploitable.

### Policy Types in Practice

```
TABULAR POLICY:
  pi: {s1->a2, s2->a1, s3->a3, ...}  <- Table lookup, only for tiny state spaces

PARAMETERIZED POLICY:
  pi_theta(a|s) = softmax(W * phi(s) + b)  <- Neural network with parameters theta
  
POLICY GRADIENT:
  Directly optimize theta to maximize E[G_t] using gradient ascent
  
VALUE-BASED:
  Learn V(s) or Q(s,a), derive policy implicitly: pi(s) = argmax_a Q(s,a)
```

---

## 🔍 4. Returns: Discounted Cumulative Reward {#returns}

The agent's goal is to **maximize the expected cumulative reward** over time — not just immediate reward. This is the **return** G_t:

```
G_t = r_t + r_{t+1} + r_{t+2} + ... + r_T   (Undiscounted, episodic)
```

### The Discount Factor γ (Gamma)

For **infinite-horizon** tasks or to prefer sooner rewards, we discount future rewards:

```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ_{k=0}^{∞} γ^k · r_{t+k}
```

![Discount Factor Gamma](../Visuals/02_discount_factor_gamma.png)

| γ Value | Interpretation | Effect |
|---------|---------------|--------|
| **γ = 0** | Completely myopic | Only cares about immediate reward |
| **γ = 0.95** | Typical RL value | Future reward at step 20 worth ~36% of immediate |
| **γ = 0.99** | Long-horizon problems | Future reward at step 100 worth ~37% of immediate |
| **γ = 1.0** | No discounting | Only valid for finite-horizon episodic tasks |

> [!IMPORTANT]
> **Why discount?** Three reasons:
> 1. **Mathematical convergence**: Ensures the infinite sum G_t is finite (geometric series: G ≤ r_max / (1-γ))
> 2. **Economic interpretation**: A reward now is worth more than the same reward later (time value)
> 3. **Uncertainty**: Future rewards are less certain; discounting reflects this uncertainty

### Recursive Bellman Form

```
G_t = r_t + γ · G_{t+1}
```

This recursive definition is the foundation of **Bellman equations** and **dynamic programming** in RL.

---

## 🔍 5. Markov Decision Processes (MDPs) {#mdps}

A **Markov Decision Process** is the mathematical framework for formalizing RL problems:

```
MDP = (S, A, P, R, γ)
```

| Symbol | Name | Description |
|--------|------|-------------|
| **S** | State space | Set of all possible states |
| **A** | Action space | Set of all possible actions |
| **P(s'\|s,a)** | Transition probability | Probability of reaching s' from s via action a |
| **R(s,a,s')** | Reward function | Expected reward for transition (s,a) → s' |
| **γ** | Discount factor | How much to discount future rewards |

### The Markov Property

> **"The future is conditionally independent of the past given the present state."**

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

This means: **knowing the current state is sufficient** to make optimal decisions — the entire history before this state is irrelevant. This critical assumption enables tractable RL algorithms.

### Value Functions

**State Value Function V^π(s):**
```
V^π(s) = E_π [ Σ_{k=0}^{∞} γ^k · r_{t+k} | S_t = s ]
```
Expected return starting from state s, following policy π.

**Action-Value Function Q^π(s, a):**
```
Q^π(s, a) = E_π [ Σ_{k=0}^{∞} γ^k · r_{t+k} | S_t = s, A_t = a ]
```
Expected return starting from state s, taking action a, then following policy π.

**Relationship:**
```
V^π(s) = Σ_a π(a|s) · Q^π(s, a)
```

### Bellman Optimality Equations

```
V*(s) = max_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ · V*(s') ]

Q*(s,a) = Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ · max_{a'} Q*(s',a') ]

π*(s) = argmax_a Q*(s,a)
```

> [!TIP]
> **Intuition**: The optimal Q-value of (s,a) = immediate reward + discounted value of the best action we can take in the resulting state. If we know Q*, we know everything — no need to model transitions explicitly.

---

## 🔍 6. The Credit Assignment Problem {#credit-assignment}

> **Definition**: In RL, the agent receives a reward, but which of the hundreds of prior actions caused it? This is the **temporal credit assignment problem** — attributing credit (or blame) to actions that occurred long before the observed reward.

### Example: Chess
- A game of chess may last 80 moves.
- The agent wins (reward = +1) at the end.
- Which of the 80 moves actually led to the win?
- Was it move 12 (a brilliant sacrifice) or move 79 (forcing checkmate)?

### Solutions to the Credit Assignment Problem

| Solution | Description | Algorithm |
|---------|-------------|-----------|
| **Discount factor** | Later actions get more credit | All RL algorithms |
| **Eligibility traces** | Track which states/actions were recently visited | TD(λ), SARSA(λ) |
| **Return calculation** | Propagate terminal reward backwards | Monte Carlo methods |
| **Temporal Difference** | Bootstrap from value estimates | TD(0), Q-Learning |
| **Advantage function** | Subtract baseline to reduce variance | Actor-Critic, PPO |

---

## 🔍 7. Exploration vs Exploitation {#exploration}

One of the central dilemmas in RL:

| Strategy | Description | Risk |
|----------|-------------|------|
| **Exploitation** | Do the best action you know | Miss better undiscovered actions |
| **Exploration** | Try new actions to discover potentially better ones | Waste time on bad actions |

### Common Exploration Strategies

![Epsilon Greedy Decay](../Visuals/03_epsilon_greedy_visual.png)

**ε-Greedy:**
```
a_t = random_action         with probability ε
a_t = argmax_a Q(s_t, a)   with probability 1-ε
```

**Decaying ε (Python):**
```python
epsilon = max(1 - episode / 500, 0.01)  # Linear decay over 500 episodes
```

**Softmax / Boltzmann Exploration:**
```
π(a|s) = exp(Q(s,a)/τ) / Σ_{a'} exp(Q(s,a')/τ)
```
Temperature τ controls randomness: τ→∞ = uniform random, τ→0 = greedy.

**UCB (Upper Confidence Bound):**
```
a_t = argmax_a [ Q(s,a) + c * sqrt( ln(t) / N(s,a) ) ]
```
Prioritize actions with high uncertainty (low visit count N(s,a)). Principled exploration.

> [!NOTE]
> The **exploration-exploitation trade-off** has no perfect solution. In practice, ε-greedy with decay is the most commonly used due to its simplicity and effectiveness. More principled methods (UCB, Thompson Sampling) can be better but are harder to implement with neural network function approximators.

---

## 🔍 8. RL Algorithm Taxonomy {#taxonomy}

![RL Algorithm Taxonomy](../Visuals/20_rl_algorithm_taxonomy.png)

```
REINFORCEMENT LEARNING ALGORITHMS
│
├── MODEL-FREE (don't learn transition model P(s'|s,a))
│   │
│   ├── VALUE-BASED (learn value function, derive policy)
│   │   ├── Q-Learning (off-policy, tabular)
│   │   ├── SARSA (on-policy, tabular)
│   │   └── DQN (Deep Q-Network) — uses NN for Q(s,a)
│   │
│   ├── POLICY-BASED (directly optimize policy parameters)
│   │   ├── REINFORCE (Monte Carlo Policy Gradient)
│   │   └── PPO, TRPO (modern policy gradient methods)
│   │
│   └── ACTOR-CRITIC (combine value + policy learning)
│       ├── A3C / A2C (Asynchronous Advantage Actor-Critic)
│       ├── SAC (Soft Actor-Critic)
│       └── TD3 (Twin Delayed DDPG)
│
└── MODEL-BASED (learn P(s'|s,a), plan with it)
    ├── Dyna-Q (tabular model + planning)
    ├── World Models
    └── AlphaZero (MCTS + neural value/policy)
```

| Algorithm | On/Off Policy | Model | Best For |
|-----------|--------------|-------|---------|
| **REINFORCE** | On-policy | Free | Simple discrete tasks |
| **DQN** | Off-policy | Free | Discrete action spaces (Atari) |
| **A3C/A2C** | On-policy | Free | Parallel environments |
| **PPO** | On-policy | Free | General purpose (most popular) |
| **SAC** | Off-policy | Free | Continuous action robotics |
| **AlphaZero** | Off-policy | Model-based | Perfect-info board games |

> [!IMPORTANT]
> **On-policy vs Off-policy:**
> - **On-policy**: Learns from experience generated by the *current* policy (must re-collect data after every update). Example: REINFORCE, PPO.
> - **Off-policy**: Learns from experience generated by *any* policy (can reuse old data with a replay buffer). Example: DQN, SAC. Generally more **sample efficient** but less **stable**.

---

## ❌ Common Beginner Mistakes {#mistakes}

**1. "Confusing state with observation"** ❌
> In most real-world problems, the agent can't see the full state. An Atari game's state is all memory registers; the observation is just the screen pixels. Always explicitly distinguish what the agent *sees* from what actually *determines* the dynamics.

**2. "Using γ = 1.0 on infinite-horizon tasks"** ❌
> Without discounting, the return G_t is the undiscounted infinite sum — which diverges for continuing tasks. This makes Q-values explode to infinity and destabilizes training. Always use γ < 1 (typically 0.95–0.99) for continuing environments.

**3. "Treating RL as supervised learning"** ❌
> You cannot apply standard cross-entropy loss to RL because you don't have correct action labels — you have scalar rewards. The REINFORCE trick multiplies the log-probability of actions by the discounted return. Misunderstanding this leads to completely wrong gradient updates.

**4. "Ignoring the exploration-exploitation tradeoff"** ❌
> Setting ε=0 from the start (pure greedy) means the agent never explores and gets stuck in the first suboptimal policy it finds. Always begin training with high exploration (ε ≈ 1.0) and anneal it over time.

**5. "Expecting fast convergence"** ❌
> RL is fundamentally sample-inefficient compared to supervised learning. Training a DQN on Atari requires ~10 million environment steps. Set expectations appropriately and use parallelism (A3C) or experience replay (DQN) to improve efficiency.

---

## 🎤 Interview Q&A {#interview}

**Q1: What is the fundamental difference between RL and supervised learning?**
> **A:** In supervised learning, a labeled dataset {(x_i, y_i)} is provided, and the model learns a mapping from inputs to correct outputs via a well-defined loss (cross-entropy, MSE). The loss surface is fixed and stationary.
>
> In RL, there are **no labeled actions**. The agent must *discover* which actions are good through interaction. The feedback signal (reward) is:
> - **Delayed**: rewards come after sequences of actions, not immediately
> - **Sparse**: many steps yield 0 reward; meaningful signals are rare
> - **Non-stationary**: as the agent improves, the distribution of visited states changes
>
> This makes RL fundamentally harder: the "training distribution" is the agent's own behavior, which shifts continuously as learning progresses.

**Q2: What is the Markov property and why does it matter for RL?**
> **A:** The Markov property states: P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, ..., s_t, a_t). The future state depends only on the present state and action, not on the entire history.
>
> This matters because:
> 1. **Algorithmic tractability**: Q-values Q(s,a) can be defined and computed efficiently — they only depend on current state, not history.
> 2. **Memory requirements**: We don't need to store infinite history.
> 3. **Bellman equations**: The recursive Q-value update Q(s,a) ← r + γ·max Q(s',a') is valid only when the Markov property holds.
>
> When the Markov property is *violated* (partial observability), we extend to POMDPs and use RNNs to maintain a "belief state" over the hidden true state.

**Q3: Why is the credit assignment problem hard in RL?**
> **A:** Because the causal link between actions and rewards is temporally distant and obscured by stochasticity. Consider:
> - A chess agent makes 60 moves over 10 minutes; wins the game.
> - Which of the 60 actions caused the win? Many were neutral; a few were decisive.
> - With stochastic environments, the same action in the same state may lead to different outcomes.
>
> Solutions like **discounting** (γ < 1) emphasize recent actions, **eligibility traces** track "responsibility" of recent state-action pairs, and **advantage functions** in actor-critic methods reduce variance by subtracting a baseline (V(s)) from the return, focusing credit on *above-average* actions.

**Q4: Explain the exploration-exploitation tradeoff. What happens if you exploit too much?**
> **A:** Exploitation means picking the best known action: a = argmax_a Q(s,a). Exploration means trying new actions to potentially discover better ones.
>
> **Too much exploitation**: The agent converges prematurely to a suboptimal policy. It found a local optimum (e.g., "always push left in CartPole gets +10 reward") and never explored the better global optimum ("balance the pole gets +200 reward").
>
> **Too much exploration**: The agent keeps trying random actions and never converges to a good policy — high variance, low performance.
>
> The standard solution is **ε-greedy with decay**: start at ε=1.0 (pure random exploration), decay linearly or exponentially toward ε=0.01, giving the agent time to discover promising actions early and exploit them later.

---

## ⚡ One-Page Flash Card {#revision}

```
╔══════════════════════════════════════════════════════════════════╗
║               MODULE 01 — RL FUNDAMENTALS                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  THE RL LOOP:                                                    ║
║ Agent -> Action a_t -> Env -> (Reward r_t, State s_{t+1})        ║
║                                                                  ║
║  MDP = (S, A, P(s'|s,a), R(s,a,s'), γ)                          ║
║                                                                  ║
║  KEY FORMULAS:                                                   ║
║  Return:    G_t = Sum γ^k * r_{t+k}    (γ in [0,1))             ║
║  Bellman:   G_t = r_t + γ * G_{t+1}                             ║
║  Q*(s,a) = Sum P(s'|s,a)[R + γ*max Q*(s',a')]                   ║
║  pi*(s) = argmax_a Q*(s,a)                                       ║
║                                                                  ║
║  EXPLORATION: epsilon-greedy with decay from 1.0 to 0.01        ║
║                                                                  ║
║  ALGORITHM FAMILIES:                                             ║
║  Value-based: Q-Learning, DQN                                    ║
║  Policy-based: REINFORCE, PPO                                    ║
║  Actor-Critic: A3C, SAC, TD3                                     ║
║                                                                  ║
║  COMMON PITFALLS:                                                ║
║  γ=1.0 on infinite tasks -> divergence                           ║
║  epsilon=0.0 from start -> premature convergence                 ║
║  No credit assignment -> wrong gradient direction                ║
╚══════════════════════════════════════════════════════════════════╝
```

---

**🔗 Next Module →** [02_OpenAI_Gym_and_Policy_Gradients.md](02_OpenAI_Gym_and_Policy_Gradients.md)
