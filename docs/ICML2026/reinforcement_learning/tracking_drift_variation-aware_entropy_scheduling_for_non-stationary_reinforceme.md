---
title: >-
  [Paper Note] Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Non-stationary RL] AES projects the exploration intensity scheduling problem of maximum entropy RL into the dynamic regret framework of Online Convex Optimization (OCO). It derives a t…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Non-stationary RL"
  - "entropy scheduling"
  - "variation budget"
  - "exploration-exploitation trade-off"
  - "adaptive"
date: 2026-05-08
content_hash: bb945f9fc1b1af6f
---

# Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2601.19624](https://arxiv.org/abs/2601.19624)  
**Code**: TBD  
**Area**: Reinforcement Learning / Non-Stationary Learning  
**Keywords**: Non-stationary RL, entropy scheduling, variation budget, exploration-exploitation trade-off, adaptive

## TL;DR
AES projects the exploration intensity scheduling problem of maximum entropy RL into the dynamic regret framework of Online Convex Optimization (OCO). It derives a theoretical result stating that the "entropy weight should be proportional to the square root of the environment drift magnitude." Using TD-error quantiles as observable drift proxies, it achieves a completely online, algorithm-agnostic entropy scheduling—generally halving cataclysmic recovery times across four frameworks (SAC / PPO / SQL / MEow) and 12 tasks.

## Background & Motivation

**Background**: Modern maximum entropy RL (e.g., SAC) explicitly controls the exploration-exploitation balance through an entropy coefficient. In practice, however, the entropy coefficient is typically fixed or tuned only for stationary environments. Real-world scenarios involve constant changes—robots encountering different physical conditions, autonomous driving adapting to traffic patterns, and recommendation systems tracking preference drifts.

**Limitations of Prior Work**: Fixed entropy coefficients lead to two simultaneous issues: (1) over-exploration during stable periods wastes samples; (2) under-exploration after changes slows recovery. Existing non-stationary RL approaches face hurdles: change-point detection introduces complexity that is hard to integrate; sliding windows lack principled guidance; and while meta-learning can accelerate adaptation, it does not explicitly characterize the mapping from "environment variation rate $\rightarrow$ optimal entropy."

**Key Challenge**: Significant environment changes clearly require increased exploration intensity, but a theoretical answer for "how much" is missing. Existing methods are primarily heuristic or environment-dependent.

**Goal**: Provide a principled entropy scheduling strategy that explicitly depends on the degree of environment variation and adjusts automatically under non-stationary MDPs.

**Key Insight**: From the perspective of dynamic regret in Online Convex Optimization (OCO), when the optimal solution (drift comparator) changes over time, the learner faces a one-dimensional trade-off between "tracking drift vs. maintaining stability." By transforming the entropy control problem into this trade-off, one can solve for the functional relationship between the entropy weight and the drift rate.

**Core Idea**: From dynamic regret, an individual round loss $\varphi_t(\lambda) = C_1 \xi_t / \lambda + C_2 \lambda$ is derived (where $\xi_t$ is the drift magnitude). Minimizing this yields a **square-root scaling** rule $\lambda_t^* \propto \sqrt{\xi_t}$. By replacing the unobservable drift $\xi_t$ with an observable drift proxy (TD-error quantiles), a fully online adaptive entropy scheduling is achieved.

## Method

### Overall Architecture
AES consists of three layers:

1.  **Theoretical Layer**: Derives dynamic regret bounds from non-stationary OCO, proving that entropy weight should be proportional to the square root of the drift magnitude.
2.  **Online Layer**: Replaces the unknown optimal comparator drift with an observable drift proxy, resulting in the fully online scheduling rule $\lambda_t = \sqrt{(C_1 / C_2) \cdot \widehat{A}_t / t}$, where $\widehat{A}_t$ is the cumulative drift proxy.
3.  **Implementation Layer**: Inserts the scheduled entropy coefficient $\alpha_t$ or $c_{\text{ent}, t}$ into SAC / PPO / SQL / MEow as a plug-and-play exploration control layer without modifying the core algorithmic structure.

### Key Designs

1.  **Single-round Trade-off Function + Square-root Scaling**:
    *   **Function**: Establishes a quantitative relationship between entropy weight and environment drift.
    *   **Mechanism**: Analyzes non-stationary OCO via dynamic mirror descent lemmas to obtain a single-round contribution $\varphi_t(\lambda) = C_1 \xi_t / \lambda + C_2 \lambda$. The first term is the "tracking cost"—larger drift or smaller entropy weight leads to slower tracking; the second term is the "stability cost"—larger entropy weight results in more unnecessary stochasticity. Setting the derivative with respect to $\lambda$ to zero yields $\lambda_t^* = \sqrt{(C_1 / C_2) \cdot \xi_t}$.
    *   **Design Motivation**: Square-root scaling replaces heuristic adjustments of exploration intensity in non-stationary RL, providing the first principled formula explicitly bound to drift.

2.  **Observable Drift Proxy + Online Scheduling**:
    *   **Function**: Replaces the unobservable optimal comparator drift $\xi_t$ from theory with an observable signal to enable fully online scheduling.
    *   **Mechanism**: Defines a drift proxy $\widehat{\xi}_t \geq \xi_t$ (not requiring unbiasedness, only a conservative upper bound). The default uses $\widehat{\xi}_t = \mathrm{Quantile}_{0.9}(|\delta_Q|)$, the 90th percentile of the absolute TD-error in the current batch (PPO uses value function TD-error). This signal naturally rises when the environment changes because the old value function becomes inaccurate for the new environment. Using the prefix sum $\widehat{A}_t = \sum_{s=1}^t \widehat{\xi}_s$, the final schedule is $\lambda_t = \sqrt{(C_1 / C_2) \cdot \widehat{A}_t / t}$, clipped to $[\lambda_{\min}, \lambda_{\max}]$ for numerical stability.
    *   **Design Motivation**: TD-error is an existing RL signal that requires no extra computation and automatically reflects the severity of environment changes; continuous signals are better suited for gradual or periodic drift than discrete change-point detection.

3.  **Cross-Algorithm Plug-and-Play Mechanism**:
    *   **Function**: Enables AES to integrate seamlessly into different maximum entropy RL algorithms.
    *   **Mechanism**: Various maximum entropy RL algorithms have entropy weight parameters (temperature $\alpha$ for SAC / SQL / MEow; entropy reward coefficient $c_{\text{ent}}$ for PPO). AES calculates the drift proxy at each training step, obtains the new entropy weight via the scheduling rule, and substitutes it into the algorithm's existing position (e.g., $\alpha_t \log \pi(a \mid s)$ in the SAC actor loss) without changing the core logic.
    *   **Design Motivation**: The RL community uses multiple entropy regularization schemes and algorithmic frameworks (off-policy vs. on-policy); a unified, algorithm-agnostic interface maximizes the scope of application.

### Loss & Training
The learning objective for non-stationary soft MDPs is $J_t(\pi) = \mathbb{E}[\sum_h \gamma^h (r_t(s_h, a_h) + \mu H(\pi(\cdot \mid s_h)))]$. AES adjusts $\mu$ or $\alpha_t$ to control the weight of the entropy term. Theoretically, $\lambda_t^* \propto \sqrt{\xi_t}$, while in practice $\lambda_t = \sqrt{\widehat{A}_t / t}$ (with clipping) is utilized.

## Key Experimental Results

### Main Results: Normalized AUC under Four Drift Modes

| Task Family | Mode | Standard SAC | SAC + AES | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Toy (2D) | Steady | 1.00 | 1.13 | +13% |
| Toy (2D) | Abrupt | 0.72 | 0.88 | +22% |
| Toy (2D) | Periodic | 0.81 | 0.94 | +16% |
| Toy (2D) | Mixed | 0.73 | 0.97 | +33% |
| MuJoCo (Avg) | Steady | 1.00 | 1.24 | +24% |
| MuJoCo (Avg) | Abrupt | 0.67 | 0.87 | +30% |
| MuJoCo (Avg) | Periodic | 0.68 | 0.94 | +38% |
| MuJoCo (Avg) | Mixed | 0.65 | 0.94 | +45% |
| Isaac Gym (Avg) | Periodic | 0.57 | 0.95 | +67% |
| Isaac Gym (Avg) | Mixed | 0.51 | 0.79 | +55% |

SAC + AES significantly outperforms standard SAC across all non-stationary modes, especially in Mixed and Periodic modes. Performance does not degrade under Steady conditions and even shows a slight increase (+13%), indicating that adaptive entropy scheduling does not penalize stable phases.

### Ablation Study: Cataclysmic Recovery Time (Percentage $\downarrow$, lower is better)

| Task | SAC | SAC + AES | PPO | PPO + AES | MEow | MEow + AES |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Hopper | 12.2 | 6.4 | 12.7 | 6.1 | 6.1 | 4.7 |
| HalfCheetah | 9.6 | 5.1 | 11.8 | 5.1 | 7.7 | 4.4 |
| Walker2d | 10.9 | 5.6 | 12.2 | 5.5 | 9.0 | 4.7 |
| Humanoid | 14.8 | 8.6 | 16.3 | 10.3 | 15.1 | 7.5 |
| **Average** | **13.96** | **7.74** | **12.12** | **8.43** | **11.58** | **6.42** |

Cataclysmic recovery time is defined as the percentage of environmental interaction steps required to recover performance after a change point relative to the total steps. The average recovery time across all four algorithmic baselines was halved (e.g., SAC 13.96% $\rightarrow$ 7.74%, MEow 11.58% $\rightarrow$ 6.42%). Improvements were most significant in high-dimensional tasks (AllegroHand, FrankaCabinet), aligning with the theoretical expectation that higher dimensionality and more severe drift yield greater benefits for adaptive exploration.

### Key Findings
- Adaptive entropy scheduling provides significant improvements across all four types of drift modes without degrading performance in steady environments.
- Cross-algorithm validation demonstrates that AES is a universal, algorithm-agnostic control principle.
- High-dimensional and high-drift scenarios yield the greatest benefits, validating the practical existence of the theoretical "narrow trade-off zone."

## Highlights & Insights
- **Hard-Theory Supported Exploration Formula**: Deriving $\lambda^* \propto \sqrt{\xi_t}$ from dynamic regret is a rare example in RL of a quantitative guide tied to environment drift.
- **Explicit Causal Characterization**: The "tracking cost vs. stability cost" framework explains why fixed entropy coefficients fail—small $\lambda$ tracks too slowly during drift, while large $\lambda$ is wasteful during stability.
- **Transferable System Design**: The plug-and-play mechanism allows AES to adapt seamlessly to four distinct algorithms: SAC, PPO, SQL, and MEow.
- **TD-Error as Change Signal**: Provides a free, continuous response to both gradual and abrupt changes, proving more robust than specialized change-point detectors.

## Limitations & Future Work
- Conservatism and noise in drift proxies: The upper quantile of TD-error might generate false signals due to optimization fluctuations; more precise calibration is needed in multi-agent or high-dimensional settings.
- Theoretical limitations: The analysis is based on tabular and fully observable settings; the bias term $\mathrm{Bias}_t$ under deep RL function approximation lacks an explicit bound.
- The default proxy was only compared with the 90th percentile of TD-error without systematic comparison of other potential proxies (e.g., policy parameter drift, model uncertainty).
- Synergies with change-point detection, intrinsic rewards, and meta-RL mechanisms have not been fully explored.

## Related Work & Insights
- **vs. Change-Point Detection** (Alami 2023; Chartouny 2025): Precise localization vs. continuous signals; the former offers strong guarantees but is hard to integrate, while the latter is lightweight but less precise. They may be complementary.
- **vs. Intrinsic Rewards** (ICM, RND): Curiosity regulates "which states are explored" (state preference), while AES regulates "how stochastic the global policy is" (overall entropy). AES is more direct in scenarios where goals change but state novelty does not necessarily increase.
- **vs. Meta-RL**: Meta-learning provides rapid adaptation capabilities but uses fixed entropy regularization. AES explicitly ties exploration intensity to the magnitude of drift, making it more targeted in distribution-shift scenarios.
- **vs. Sliding Windows / Time Decay**: Early non-stationary RL used fixed decay $\mathcal{O}(t^{-1/2})$ lacking principled guidance; AES responds dynamically via online variation estimation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes the first quantitative relationship between exploration intensity and environment drift in maximum entropy RL; the application of dynamic regret analysis is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across 4 frameworks × 12 tasks × 4 drift modes × 3 complementary metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rigorous theoretical derivation, and detailed experimental descriptions; technical details in the appendix slightly reduce the main text's substance.
- Value: ⭐⭐⭐⭐⭐ Non-stationary RL is increasingly important yet lacks sufficient theory; this work provides the first principled and implementable exploration control strategy, with plug-and-play ensuring broad potential for practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](../../NeurIPS2025/reinforcement_learning/forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](../../NeurIPS2025/reinforcement_learning/solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)
- [\[ICML 2026\] D$^2$Evo: Dual Difficulty-Aware Self-Evolution for Data-Efficient Reinforcement Learning](d2evo_dual_difficulty-aware_self-evolution_for_data-efficient_reinforcement_lear.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ICML 2026\] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness](convergence_of_steepest_descent_and_adam_under_non-uniform_smoothness.md)

</div>

<!-- RELATED:END -->
