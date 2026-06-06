---
title: >-
  [Paper Note] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning
description: >-
  [ICLR 2026][AI Safety][backdoor attack] This paper proposes Daze, a backdoor attack in which a malicious simulator developer—without any access to or modification of the agent's reward function—plants a backdoor solely b…
tags:
  - "ICLR 2026"
  - "AI Safety"
  - "backdoor attack"
  - "reinforcement-learning"
  - "simulator security"
  - "reward-free"
  - "supply chain attack"
date: 2026-05-08
content_hash: 4d7de094adfd5d46
---

# Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2602.05089](https://arxiv.org/abs/2602.05089)  
**Code**: None  
**Area**: AI Safety / RL Safety
**Keywords**: backdoor attack, reinforcement-learning, simulator security, reward-free, supply chain attack

## TL;DR
This paper proposes Daze, a backdoor attack in which a malicious simulator developer—without any access to or modification of the agent's reward function—plants a backdoor solely by manipulating state transitions: when the agent fails to execute the target action in a trigger state, it is forced to take random actions ("dazed"), thereby theoretically guaranteeing both attack success and stealthiness. The work also presents the first demonstration of an RL backdoor attack on real robot hardware.

## Background & Motivation
**Background**: RL training relies heavily on simulators (MuJoCo, PyBullet, etc.), and researchers and engineers routinely use third-party simulators or cloud-based services. Simulators are a trusted yet under-scrutinized component in the RL supply chain.

**Limitations of Prior Work**: Existing RL backdoor attacks (TrojDRL, SleeperNets, Q-Incept) require deep control over the training pipeline—specifically the ability to read and write the agent's reward function. However, many simulators (e.g., MuJoCo, PyBullet) only receive actions and return states, with rewards computed externally, leaving the reward inaccessible to an attacker.

**Key Challenge**: All known RL backdoor attacks rely on reward manipulation to make the target action optimal. If the attacker can neither observe nor modify rewards (as in the simulator setting), conventional methods fail entirely.

**Goal**: Design an efficient and stealthy backdoor under the severely restricted threat model in which only the returned state values can be manipulated, with zero access to rewards.

**Key Insight**: In virtually all meaningful MDPs, uniformly random action sampling is suboptimal. The attacker can exploit this universal property—by forcing random actions whenever the agent does not execute the target action in a trigger state, causing the agent to receive lower returns.

**Core Idea**: Replace direct reward manipulation with the causal chain "fail to execute target action → forced random actions → low return," achieving a backdoor via transition manipulation rather than reward manipulation.

## Method

### Overall Architecture
Daze is implemented as a simulator wrapper. Under normal conditions, the simulator operates according to the original transition function $T$ (with probability $1-\beta$); with probability $\beta$ it enters the trigger state $\delta(s)$. In the trigger state: if the agent executes the target action $a^+$, a "null transition" is applied—sampling an action from the policy under the benign state, with no cost to the agent; if the agent does not execute $a^+$, it enters the "daze state" $\phi(s)$ and is forced to take random actions for several steps.

### Key Designs

1. **Daze Mechanism (Reward-Free Backdoor)**:

    - Function: Makes the target action optimal in the trigger state without touching the reward.
    - Mechanism: Exploits Assumption 1 (a random policy is suboptimal in almost all MDPs). In the trigger state, executing $a^+$ leads to a normal transition (no cost); not executing $a^+$ leads to the daze state ($k \cdot \mathcal{L}_{adv}(a, a^+)$ steps of random actions), during which rewards are computed by the real physics but actions are sampled randomly by the attacker, yielding low returns.
    - Design Motivation: To maximize returns, the agent naturally learns to execute $a^+$ in the trigger state. Rewards are computed entirely by the external environment, and the attacker has zero contact with them.

2. **Theoretical Guarantees**:

    - Theorem 1: The optimal policy $\pi^*$ in $M'$ necessarily satisfies $\mathcal{L}_{adv}(\pi^+(\delta(s)), a^+) = 0$ (attack is guaranteed to succeed).
    - Theorem 2: The optimal policy of $M'$ is also optimal in $M$ on benign states (stealthiness guarantee).
    - Intuition: Since the optimal policy always executes $a^+$ and never enters the daze state, its behavior on benign states is identical to that of the unpoisoned model.

3. **Adaptation to Continuous Action Spaces**:

    - Function: Generalizes "exact match with $a^+$" from discrete domains to "$ \ell_\infty $ distance below threshold $\tau$" in continuous domains.
    - Daze duration is proportional to the deviation: $\lceil k \cdot \mathcal{L}_{adv}(a, a^+) \rceil$ steps, providing denser gradient signals.

### Implementation
Implemented as an environment wrapper (Algorithm 1) with three logical branches: Benign Routine → Trigger Routine → Daze Routine. The attacker only needs to specify the trigger function $\delta$, daze function $\phi$, target action $a^+$, and poisoning rate $\beta$.

## Key Experimental Results

### Continuous Action Space (MuJoCo + Custom Robot Environments)

| Environment | Method | ASR | Benign Return | Unpoisoned BR |
|------|------|-----|---------------|-----------|
| HalfCheetah | **Daze** | **92.4%** | **1627.8** | 1736.3 |
| | TrojDRL-C | 20.3% | 604.4 | |
| | SleeperNets-C | 3.3% | 1511.3 | |
| Hopper | **Daze** | **94.1%** | **2321.5** | 2085.4 |
| | TrojDRL-C | 25.6% | 984.9 | |
| Intersection (real robot) | **Daze** | **92.3%** | **226.7** | 238.4 |

### Discrete Action Space (Atari)

| Environment | Daze | Q-Incept | TrojDRL | SleeperNets |
|------|------|----------|---------|-------------|
| Q*bert | 99.3% | 100% | 88.2% | 100% |
| Frogger | 99.9% | 99.2% | 95.7% | 100% |
| Pacman | 99.4% | 100% | - | - |
| Breakout | 97.6% | 100% | - | - |

### Key Findings
- Daze achieves >92% ASR in both continuous and discrete action spaces, with benign return nearly indistinguishable from unpoisoned models.
- TrojDRL-C and SleeperNets-C achieve significantly lower ASR than Daze in continuous domains (3–65%), demonstrating that conventional reward manipulation methods do not scale well to continuous action spaces.
- The first successful triggering of backdoor behavior on real Turtlebot2 and Fetch robots is demonstrated—triggered Turtlebots accelerate in a straight line causing collisions, and Fetch robots make sharp turns causing objects to fall.
- With only 1% poisoning rate and as little as 0.3% additional daze steps (Hopper), the attack is extremely stealthy.

## Highlights & Insights
- **Strongest Attack under the Most Restricted Threat Model**: Daze, operating in a severely restricted setting with no reward access, outperforms prior methods that require full reward control (in continuous domains). This counterintuitive result arises because transition manipulation more directly influences policy learning than reward manipulation.
- **Impact of Real Hardware Demonstration**: The first demonstration of an RL backdoor attack on physical robots shows that policies backdoored in simulation during training remain effective after sim-to-real transfer, confirming that this is not merely a theoretical threat.
- **Theoretically Elegant and Powerful**: The core insight is remarkably simple (random < optimal), yet it yields rigorous guarantees of both attack success and stealthiness.

## Limitations & Future Work
- Assumption 1 requires that the random policy be suboptimal—this may be violated in perfectly symmetric environments (e.g., mazes with left-right equivalence), though the practical impact is minimal.
- The design of trigger function $\delta$ still requires manual selection; the paper uses handcrafted flags or checkerboard patterns, and more covert trigger designs remain to be investigated.
- Evaluation is limited to PPO, TD3, and DQN; SAC and model-based RL are not covered.
- On the defense side, the paper raises the problem and motivation but does not propose concrete detection methods.

## Related Work & Insights
- **vs. TrojDRL (Kiourti et al., 2019)**: Requires read/write reward access; achieves only 20–65% ASR in continuous domains and degrades benign return. Daze achieves 92%+ ASR without touching rewards.
- **vs. SleeperNets (Rathbun et al., 2024)**: Requires RAM-level outer-loop access, a stronger but less realistic threat model. Achieves extremely low ASR in continuous domains (0–7.5%).
- **vs. Q-Incept (Rathbun et al., 2025)**: Comparable performance in discrete domains, but Q-Incept does not extend to continuous action spaces.
- **Implications for RL Security**: Trusting only externally computed reward functions is insufficient—simulator transitions are equally susceptible to malicious manipulation, underscoring the need for security auditing of all components in the RL training pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First reward-free RL backdoor attack + first real-robot demonstration; a well-motivated new threat model
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers MuJoCo, Atari, custom environments, and real robots across continuous/discrete domains, with 3 baselines
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and the algorithm is well-described, though the tables are dense
- Value: ⭐⭐⭐⭐⭐ Exposes simulators as a security blind spot in the RL supply chain, with important implications for RL deployment in autonomous driving and robotics

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning](../../ICML2026/ai_safety/angel_or_demon_investigating_the_plasticity_interventions_impact_on_backdoor_thr.md)
- [\[ICLR 2026\] Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](../../AAAI2026/ai_safety/transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)
- [\[AAAI 2026\] Towards Effective, Stealthy, and Persistent Backdoor Attacks Targeting Graph Foundation Models](../../AAAI2026/ai_safety/towards_effective_stealthy_and_persistent_backdoor_attacks_targeting_graph_found.md)
- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](../../ICCV2025/ai_safety/semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)

</div>

<!-- RELATED:END -->
