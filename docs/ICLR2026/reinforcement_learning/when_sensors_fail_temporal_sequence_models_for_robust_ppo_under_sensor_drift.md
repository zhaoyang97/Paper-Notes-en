---
title: >-
  [Paper Note] When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift
description: >-
  [ICLR 2026][Reinforcement Learning][sensor failure] This paper investigates the robustness of PPO under temporally persistent sensor failures, proposes integrating sequence models (Transformer and SSMs) into PPO, derives high-probability upper bounds on infinite-horizon reward degradation under stochastic sensor failures, and demonstrates through MuJoCo experiments that Transformer-PPO significantly outperforms MLP, RNN, and SSM baselines under severe sensor dropout.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - sensor failure
  - partial observability
  - robustness
  - Transformer
  - state space models
  - PPO
  - sequence modeling
date: 2026-05-08
content_hash: 8003b02a76bd407a
---

# When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift

**Conference**: ICLR 2026
**arXiv**: [2603.04648](https://arxiv.org/abs/2603.04648)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: sensor failure, partial observability, robustness, Transformer, state space models, PPO, sequence modeling

## TL;DR
This paper investigates the robustness of PPO under temporally persistent sensor failures, proposes integrating sequence models (Transformer and SSMs) into PPO, derives high-probability upper bounds on infinite-horizon reward degradation under stochastic sensor failures, and demonstrates through MuJoCo experiments that Transformer-PPO significantly outperforms MLP, RNN, and SSM baselines under severe sensor dropout.

## Background & Motivation

**Background**: Real-world RL systems (robotic control, autonomous driving) rely on sensor feedback that is often unreliable—failures, communication interruptions, or transient corruption lead to partial observability and performance degradation.

**Limitations of Prior Work**: (1) Standard MLP policies assume fully observed states and degrade sharply when sensors are unreliable; (2) In practical systems, sensor failures exhibit temporal persistence and inter-group correlations (e.g., shared communication buses or power supplies), making simple independent masking models insufficiently realistic; (3) Existing empirical comparisons of sequence models for robustness in RL (e.g., RLBenchNet) lack theoretical characterization.

**Key Challenge**: The robustness of RL policies is directly related to their ability to exploit temporal context, yet a theoretical framework for quantifying this relationship is absent.

**Goal**: (1) Provide theoretical bounds on reward degradation under sensor failures; (2) Systematically compare the robustness of different sequence architectures within PPO; (3) Understand which architectural properties drive robustness differences.

**Key Insight**: A two-level Markov sensor failure model (individual- and group-level) is established; multiple sequence encoders are integrated into PPO; theoretical and empirical analyses are conducted in parallel.

**Core Idea**: Transformers leverage self-attention to flexibly reference historically valid observations and naturally skip over missing-data gaps, making them the most robust policy architecture in environments with unreliable sensors.

## Method

### Overall Architecture
A two-level Markov sensor failure model is proposed; Transformer/SSM/RNN encoders are integrated into the actor-critic architecture of PPO; high-probability reward degradation bounds are derived; and experimental validation is conducted in MuJoCo environments.

### Key Designs

1. **Two-Level Markov Sensor Failure Model**:

    - Individual level: each sensor $i$ follows a binary Markov chain $z_i(t) \in \{0,1\}$ with parameters $p_{\text{fail}}$ and $p_{\text{recover}}$.
    - Group level: each group $j$ follows $y_j(t) \in \{0,1\}$ with parameters $p_{\text{fail}}^{\text{group}}$ and $p_{\text{recover}}^{\text{group}}$.
    - Effective state $x_i(t) = z_i(t) \cdot y_j(t)$, with stationary probability $\pi_x = \pi_z \cdot \pi_y$.
    - Effective failure probability $p_{\text{fail}}^{\text{eff}} = 1 - (1-p_{\text{fail}})(1-p_{\text{fail}}^{\text{group}})$.

2. **Transformer-PPO**:

    - History buffer: a circular buffer maintaining the most recent $L$ observations.
    - Encoder: linear projection + sinusoidal positional encoding → Transformer encoder (with key-padding mask to skip invalid positions).
    - Attention pooling: learned attention weighting maps variable-length sequences to a fixed-size feature vector.
    - Separate actor and critic heads are attached.

3. **RNN/SSM-PPO**:

    - Unified interface: $(h_t, z_t) = \mathcal{E}_\psi(h_{t-1}, x_t; d_t)$, where $d_t$ is the episode termination flag.
    - Variants include GRU, LRU, and LinOSS.

### Theoretical Analysis

**Theorem 5.6 (High-Probability Reward Degradation Bound)**: Under Assumptions 5.1–5.5, with probability $\geq 1-\delta$:

$$S \leq \mu_S + C_{\max}\min\left\{\sqrt{\frac{2\tau}{1-\gamma^2}\ln\frac{2}{\delta}} + \frac{4}{3}\tau\ln\frac{2}{\delta}, \frac{1}{1-\gamma}\right\}$$

where:
- $\mu_S \leq \frac{L_Q L_\pi}{1-\gamma}\sum_{i=1}^d (1-\pi_{x,i})h_i$ is the mean degradation;
- $C_{\max} = L_Q L_\pi \sum_i B_i$ is the worst-case per-step impact;
- $\tau$ is the mixing time of the augmented chain.

**Interpretation**:
- The mean term depends only on the marginal up-rate of each sensor; inter-sensor correlations do not directly affect expected degradation.
- The fluctuation term has two components scaling with $\sqrt{\tau}$ and $\tau$, respectively; larger mixing times (more persistent failures) yield larger fluctuations.
- Policy smoothness $L_\pi$ and critic smoothness $L_Q$ globally scale the degradation—sequence models achieve smoother action variation by exploiting history, thereby reducing effective degradation.

## Key Experimental Results

### Experimental Setup
- 4 MuJoCo environments: HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4.
- 8 PPO agents: MLP + 3 RNNs/SSMs (LRU, GRU, LinOSS) + 3 Transformers (Transformer, UniTS, GTrXL).
- Sensor parameters: $p_{\text{fail}}=1\%$, $p_{\text{recover}}=90\%$, $p_{\text{fail}}^{\text{group}}=55\%$, $p_{\text{recover}}^{\text{group}}=90\%$ → effective recovery rate 60%.

### Main Results

| Architecture | Full Observation | 60% Partial Observation | Degradation |
|---|---|---|---|
| MLP | Usually highest | Severe degradation | **Largest** |
| GRU | Moderate | Occasionally slightly better than MLP | Significant |
| LRU | Moderate | Occasionally slightly better than MLP | Significant |
| LinOSS | Moderate | Moderate | Significant |
| GTrXL | Moderate | Unstable performance | Moderate |
| **Transformer** | **Competitive** | **Best across all environments** | **Smallest** |
| UniTS | Worst | Worst | — |

### Key Findings
- **Under full observation**: MLP is generally optimal (MuJoCo environments are Markovian), and the additional complexity of sequence models is sometimes a liability.
- **Under partial observation**: Transformer is consistently the most robust, achieving the highest evaluation median across all environments.
- RNN/SSM memory mechanisms (including GTrXL) provide limited benefit under sensor failures—recurrent dynamics process inputs uniformly and assume smooth temporal flow, both of which are violated when data is missing.
- UniTS performs worst across all settings—its inductive bias of per-variable independent processing is ill-suited for continuous control tasks that require joint temporal patterns across variables.

## Highlights & Insights
- **Practical value of the theoretical bound**: Key factors governing robustness are made explicit—policy smoothness, critic sensitivity, sensor availability, and failure persistence—providing principled guidance for designing robust agents.
- **Deeper explanation of Transformer vs. Recurrence**: Stateless Transformers process all variables jointly within a single sequence; self-attention allows each output to directly attend to all available historical tokens, naturally skipping gaps. By contrast, recurrent models' sequential state updates diverge or lose critical information when inputs are missing.
- **Practical sensor model**: The two-level Markov model can simulate rich failure patterns (rapid individual failures, rapid group failures, mixed dynamics, slow recovery with prolonged interruptions).

## Limitations & Future Work
- MuJoCo environments are relatively simple; validation on more complex real-world robotic tasks remains to be conducted.
- All models share fixed PPO configurations and matched architectural capacity; more extensive architecture search may alter the rankings.
- The theoretical bound relies on policy smoothness assumptions; obtaining tight estimates for deep-network policies remains challenging.
- The sensor failure model assumes that masking is independent of state (Assumption 5.5), whereas in practice, the environment state and sensor state may be correlated.

## Related Work & Insights
- **vs. DRQN**: DRQN uses LSTMs to handle partial observability but lacks theoretical analysis and does not target the temporal structure of sensor failures.
- **vs. RLBenchNet**: RLBenchNet is purely empirical and employs an overly simplified masking mechanism (permanently removing velocities or shrinking the observation window), without modeling realistic sensor failures.
- **vs. Decision Transformer**: Decision Transformer targets offline RL, whereas this work focuses on robustness under online PPO.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of a sensor failure model, theoretical bounds, and systematic architectural comparison is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 architectures, 4 environments, 8 seeds, and full/partial observation comparisons with rigorous statistical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory is clearly presented, explanations are intuitive, and comparisons with prior work are thorough.
- Value: ⭐⭐⭐⭐ Provides both theoretical grounding and empirical guidance for architecture selection in robust RL.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] DRMD: Deep Reinforcement Learning for Malware Detection under Concept Drift](../../AAAI2026/reinforcement_learning/drmd_deep_reinforcement_learning_for_malware_detection_under_concept_drift.md)
- [\[ICLR 2026\] TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models](tpru_advancing_temporal_and_procedural_understanding_in_large_multimodal_models.md)
- [\[NeurIPS 2025\] Incremental Sequence Classification with Temporal Consistency](../../NeurIPS2025/reinforcement_learning/incremental_sequence_classification_with_temporal_consistency.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] Distributionally Robust Cooperative Multi-Agent Reinforcement Learning via Robust Value Factorization](distributionally_robust_cooperative_multi-agent_reinforcement_learning_via_robus.md)

<!-- RELATED:END -->
