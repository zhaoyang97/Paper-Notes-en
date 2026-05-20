---
title: >-
  [Paper Note] Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States
description: >-
  [AAAI 2026][Reinforcement Learning] This paper proposes **ASAP (Action Smoothing by Aligning Actions with Predictions from Preceding States)**…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Action Smoothing"
  - "Lipschitz Constraint"
  - "Policy Smoothness"
  - "Robot Control"
date: 2026-05-08
content_hash: fea471f350353d9d
---

# Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States

**Conference**: AAAI 2026
**arXiv**: [2601.18479](https://arxiv.org/abs/2601.18479)  
**Code**: [https://github.com/AIRLABkhu/ASAP](https://github.com/AIRLABkhu/ASAP)  
**Area**: Other
**Keywords**: Reinforcement Learning, Action Smoothing, Lipschitz Constraint, Policy Smoothness, Robot Control

## TL;DR

This paper proposes **ASAP (Action Smoothing by Aligning Actions with Predictions from Preceding States)**, a reinforcement learning action smoothing method based on **transition-induced similar state definitions**. ASAP suppresses high-frequency action oscillations via a spatial constraint (aligning actions with predictions from the preceding state) and a temporal constraint (penalizing second-order action differences). It outperforms existing methods on Gymnasium and Isaac-Lab benchmarks.

## Background & Motivation

### Problem Setting
Deep reinforcement learning (DRL) has achieved notable success in continuous control tasks, yet **high-frequency action oscillations** in policy outputs remain a major obstacle to real-world deployment:
- **Hardware wear**: High-frequency oscillations significantly shorten the lifespan of mechanical components.
- **Safety concerns**: Non-smooth actions may degrade user experience or introduce safety risks.
- **Root cause**: The actor network is overly sensitive to small state perturbations, producing large action deviations.

### Limitations of Prior Work

Existing methods fall into two categories:

**Architectural Methods**:
- Approaches such as Spectral Normalization and LipsNet enforce Lipschitz constraints by modifying network architectures.
- **Drawback**: They introduce additional computational overhead at inference and exhibit high performance variance across environments.

**Loss Penalty Methods**:
- Approaches such as CAPS, L2C2, and Grad-CAPS add smoothness regularization terms to the policy loss.
- **Key issue**: They require defining "similar states" to enforce action consistency.
    - **CAPS**: Samples similar states from a **fixed Gaussian distribution** around the current state → heuristic definition that does not reflect the true state distribution.
    - **L2C2**: Uses adaptive boundaries but still relies on **synthetically constructed** states → mismatches actual environment dynamics.
- **Consequence**: Inaccurate similar state definitions invalidate theoretical guarantees and degrade performance.

### Core Motivation
Key insight: **States that transition from the same predecessor state should be considered "similar."** This transition-based definition of similar states:
1. Uses only environment feedback and actually collected data.
2. Naturally reflects system dynamics.
3. Can be proven to form a bounded neighborhood.

## Method

### Overall Architecture

The total policy loss of ASAP:
$$J_{\pi_\phi}^{\text{ASAP}} = J_{\pi_\phi} + \lambda_S L_S + \lambda_P L_P + \lambda_T L_T$$

This consists of: the standard RL actor loss + spatial constraint term + predictor training term + temporal constraint term.

### Key Designs

#### 1. **Transition-Induced Similar States (Definition 3)**

- **Core definition**: Given state $s_t$, its similar state distribution is defined as the distribution of all possible next states transitioning from the predecessor state $s_{t-1}$:
$$\text{sim}(s_t) = P(\cdot | s_{t-1})$$
- **Bounded neighborhood guarantee (Lemma 1)**: Under the assumption that the transition function is locally Lipschitz continuous with respect to noise (Assumption 1) and the noise is bounded (Assumption 2), the distance between any two similar states is upper bounded:
$$d_S(s_t^{(1)}, s_t^{(2)}) \leq 2K_\xi(s_{t-1}, a_{t-1}) \sigma_\xi$$
- **Design motivation**:
    - Unlike CAPS/L2C2, no synthetic states are required; the method is entirely based on actually collected transition data.
    - The boundedness of the similar state region is guaranteed, providing a theoretical basis for imposing local Lipschitz constraints.
    - The distribution is fully consistent with the true transition kernel $P_{\text{real}}(\cdot | s_{t-1})$.

#### 2. **Spatial Smoothness Loss**

- **Composite function Lipschitz constraint (Theorem 1)**: $f \circ T$ is locally Lipschitz continuous on the noise space with constant $K_{\text{comp}} = K \cdot K_\xi$.
- **Derived loss**:
$$L_S = \|\pi_\phi(s_t) - \text{stopgrad}(\pi_P(s_{t-1}))\|_2^2$$
  This minimizes the discrepancy between the current action $\pi_\phi(s_t)$ and the **predicted next action** $\pi_P(s_{t-1})$ from the preceding state.
- **Predictor loss**:
$$L_P = \|\pi_P(s_{t-1}) - \text{stopgrad}(\pi_\phi(s_t))\|_2^2$$
  This trains the predictor to mimic the actual policy output. Stop-gradient is used to decouple the two losses, allowing separate weighting.

#### 3. **Temporal Smoothness Loss**

The second-order difference penalty from Grad-CAPS is directly adopted:
$$L_T = \left\|\frac{a_{t+1} - 2a_t + a_{t-1}}{\tanh(a_{t+1} - a_{t-1}) + \epsilon}\right\|_2^2$$

- Penalizes second-order action changes ("acceleration") rather than first-order changes.
- The denominator $\tanh(\cdot)$ provides adaptive normalization with respect to action scale.
- Second-order differences are more flexible than first-order: they permit smooth action changes while suppressing high-frequency oscillations.

#### 4. **Predictor Implementation**

- A prediction head is added on top of the actor MLP, sharing the lower layers with the action head.
- The action head is trained with $L_S$; the prediction head is trained with $L_P$.
- The "moving target" problem is addressed by using more parallel environments for data collection and reducing the $\lambda_P$ weight.

### Loss & Training

- **Compatible with multiple RL algorithms**: $J_{\pi_\phi}$ can be the standard actor loss of PPO or SAC.
- **Hyperparameters**: $\lambda_S$, $\lambda_P$, $\lambda_T$ control the strength of the spatial, predictor, and temporal constraints, respectively.
- **Zero inference overhead**: Additional computation is incurred only during training; at inference, the method is identical to the baseline (no network architecture modification).

## Key Experimental Results

### Experimental Setup
- **Gymnasium**: LunarLander, Pendulum, Reacher, Ant, Hopper, Walker (6 environments)
- **Isaac-Lab**: Reach-Franka, Lift-Cube-Franka, Repose-Cube-Allegro, Anymal-Velocity-Rough (4 high-fidelity robot environments)
- **Metrics**: Cumulative Return (re, ↑) and Smoothness Score (sm, ↓)

### Main Results

**PPO Setting (Table 2, selected environments)**:

| Method | Hopper re↑ | Hopper sm↓ | Walker re↑ | Walker sm↓ |
|--------|-----------|-----------|-----------|-----------|
| PPO Base | 2902 | 1.709 | 2654 | 1.764 |
| CAPS | 2362 | 0.281 | 2179 | 0.565 |
| L2C2 | 2345 | 1.344 | 2014 | 1.686 |
| GRAD | 2737 | 0.193 | 1967 | 0.342 |
| **ASAP** | **2691** | **0.179** | **3128** | **0.345** |

ASAP reduces sm by **89.5%** on Hopper and **80.4%** on Walker.

**SAC Setting (Table 3, selected environments)**:

| Method | Hopper re↑ | Hopper sm↓ | Walker re↑ | Walker sm↓ |
|--------|-----------|-----------|-----------|-----------|
| SAC Base | 3349 | 0.856 | 4476 | 0.823 |
| CAPS | 3413 | 0.793 | 4320 | 0.815 |
| GRAD | 3190 | 0.588 | 4339 | 0.612 |
| **ASAP** | **3448** | **0.498** | **4665** | **0.578** |

**Isaac-Lab (Table 4)**:

| Task | PPO Base re | PPO Base sm | ASAP re | ASAP sm |
|------|------------|------------|---------|---------|
| Reach-Franka | 0.380 | 0.959 | **0.525** | **0.658** |
| Lift-Cube-Franka | **136.1** | 2.315 | 134.0 | **0.926** |
| Anymal-Velocity-Rough | **16.69** | 3.502 | 16.09 | **2.861** |

Consistent smoothness improvements are also demonstrated in high-fidelity robot tasks.

### Ablation Study

**Validation of transition-induced similar states (Table 1)**:

| Environment | SAC Base sm | After Predictor Fine-tuning sm | Improvement |
|-------------|-----------|-------------------------------|-------------|
| Hopper | 0.857 | 0.712 | −16.9% |
| Walker | 0.836 | 0.715 | −14.4% |
| LunarLander | 0.296 | 0.227 | −23.3% |

Predictor fine-tuning alone consistently reduces sm, validating the effectiveness of transition-induced similar states.

**Comparison of spatial loss sources**:

| Spatial $L_S$ | Temporal $L_T$ | Hopper re | Hopper sm | Walker re | Walker sm |
|--------------|----------------|-----------|-----------|-----------|-----------|
| — (none) | GRAD | 2963 | 0.241 | 2659 | 0.541 |
| CAPS | GRAD | 2264 | 0.201 | 2303 | 0.467 |
| L2C2 | GRAD | 2925 | 0.227 | 2500 | 0.519 |
| ASAP (ours) | GRAD | 2691 | 0.179 | 3128 | 0.345 |

ASAP's spatial term achieves the lowest sm while maintaining competitive re.

**Combination with architectural methods (Table 5)**:

| Method | Walker re | Walker sm |
|--------|-----------|-----------|
| LipsNet | 3942 | 0.915 |
| LipsNet + CAPS | 3464 | 0.665 |
| LipsNet + ASAP | **4475** | **0.485** |

ASAP can be stacked with architectural methods for further gains.

### Key Findings

1. **Average sm reduction of 43.3% under PPO** and 27.9% under SAC, with re remaining stable or improving.
2. **ASAP achieves the best smoothness in 5/6 (PPO) and 4/6 (SAC) environments**.
3. **Complementarity with Grad-CAPS**: ASAP's spatial constraint combined with Grad's temporal constraint yields the best performance.
4. **Effectiveness is also demonstrated in Isaac-Lab high-fidelity environments**, validating the potential for transfer to real robot scenarios.

## Highlights & Insights

1. **Solid theoretical foundation**: Starting from Lipschitz continuity theory, the paper rigorously proves that transition-induced similar states form a bounded neighborhood, derives a composite function Lipschitz constraint, and simplifies it into an optimizable loss function.
2. **Clear comparison with existing methods**: Figure 1 intuitively illustrates the essential differences among CAPS (fixed Gaussian), L2C2 (adaptive but synthetic), and ASAP (based on the true transition distribution).
3. **Zero inference overhead**: As a loss penalty method, additional computation is incurred only during training; the method is identical to the baseline at inference, making it deployment-friendly.
4. **Modular design**: The spatial and temporal terms can be used independently or combined with architectural methods.

## Limitations & Future Work

1. **Slight re degradation on Hopper**: The method may over-smooth regions that require rapid action changes; adaptive smoothing strength may be needed.
2. **Assumption requirements**: Lipschitz continuity of the transition function with respect to noise holds in most physical settings but may not be satisfied in certain extreme scenarios.
3. **Predictor training stability**: In on-policy methods (PPO), rapidly changing policy distributions may cause the predictor to lag.
4. **Hyperparameter selection**: Tuning $\lambda_S$, $\lambda_P$, $\lambda_T$ requires environment-specific adjustment.
5. **Only PPO and SAC are evaluated**: Other continuous control algorithms such as TD3 and DDPG are not tested.

## Related Work & Insights

- **CAPS (Mysore et al. 2021)**: The primary baseline; first decomposes the Lipschitz constraint into temporal and spatial dimensions.
- **L2C2 (Kobayashi 2022)**: Introduces the concept of local Lipschitz continuity, but the similar state definition is still based on synthetic construction.
- **Grad-CAPS (Lee et al. 2024)**: Proposes the second-order difference temporal penalty, which ASAP directly adopts.
- **Insight**: When the "correct definition" is central to a method, grounding concepts in system dynamics (rather than artificial construction) tends to be more effective. This principle of "letting the environment provide the answer" has broad applicability.

## Rating

- Novelty: ⭐⭐⭐⭐ — The transition-induced similar state definition is the core contribution, with complete theoretical derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 Gymnasium + 4 Isaac-Lab environments, dual-algorithm evaluation (PPO/SAC), comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear, though notation is somewhat heavy.
- Value: ⭐⭐⭐⭐ — Directly relevant to sim-to-real transfer; the method is practical and open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[ICLR 2026\] Enhancing Generative Auto-bidding with Offline Reward Evaluation and Policy Search](../../ICLR2026/others/enhancing_generative_auto_bidding.md)
- [\[AAAI 2026\] Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters](deadline-aware_energy-efficient_control_of_domestic_immersion_hot_water_heater.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)
- [\[AAAI 2026\] Enhancing Noise Resilience in Face Clustering via Sparse Differential Transformer](enhancing_noise_resilience_in_face_clustering_via_sparse_differential_transforme.md)

</div>

<!-- RELATED:END -->
