---
title: >-
  [Paper Note] Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning
description: >-
  [NeurIPS 2025 (Spotlight)][Reinforcement Learning][Low-Rank] This paper reveals that the successor measure in reinforcement learning is not intrinsically approximately low-rank, but a "shifted successor measure"—obtained by skipping the first few transition steps—naturally exhibits low-rank structure. A novel Type II Poincaré inequality is introduced to quantify the required shift, providing finite-sample theoretical guarantees and practical improvements for goal-conditioned RL.
tags:
  - NeurIPS 2025 (Spotlight)
  - Reinforcement Learning
  - Low-Rank
  - Successor Measure
  - reinforcement-learning
  - Spectral Analysis
  - Goal-Conditioned RL
date: 2026-05-08
content_hash: 80cb31d0ccf0a03c
---

# Shift Before You Learn: Enabling Low-Rank Representations in Reinforcement Learning

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2509.05193](https://arxiv.org/abs/2509.05193)
**Code**: None
**Area**: Reinforcement Learning / Representation Learning
**Keywords**: Low-Rank, Successor Measure, reinforcement-learning, Spectral Analysis, Goal-Conditioned RL

## TL;DR

This paper reveals that the successor measure in reinforcement learning is not intrinsically approximately low-rank, but a "shifted successor measure"—obtained by skipping the first few transition steps—naturally exhibits low-rank structure. A novel Type II Poincaré inequality is introduced to quantify the required shift, providing finite-sample theoretical guarantees and practical improvements for goal-conditioned RL.

## Background & Motivation

- **Background**: Low-rank structure is a common implicit assumption in modern RL algorithms. Reward-free RL and goal-conditioned RL methods typically assume that the successor measure admits a low-rank representation, enabling efficient state representation learning via matrix factorization.
- **Limitations of Prior Work**: This paper first identifies a critical issue: **the successor measure itself is not approximately low-rank**. Directly applying low-rank approximation to the raw successor measure incurs large errors, limiting downstream RL performance.
- **Core Idea**: The authors observe that if the successor measure is "shifted"—i.e., one considers the future visitation distribution after skipping the first $k$ transition steps—low-rank structure emerges naturally. This shifted successor measure can be efficiently approximated by a low-rank matrix, and the required shift $k$ is typically small.

## Method

### Overall Architecture

1. **Define the Shifted Successor Measure**: Given policy $\pi$ and initial state $s$, the standard successor measure is $M^\pi(s, \cdot) = \sum_{t=0}^{\infty} \gamma^t P(s_t \in \cdot \mid s_0 = s)$. The shifted version is $M_k^\pi(s, \cdot) = \sum_{t=k}^{\infty} \gamma^t P(s_t \in \cdot \mid s_0 = s)$, skipping the first $k$ steps.
2. **Low-Rank Approximation**: A rank-$r$ approximation $\hat{M}_k$ is computed for the discretized matrix of $M_k^\pi$, ensuring that $\|M_k - \hat{M}_k\|_\infty$ is controlled.
3. **Finite-Sample Estimation**: Low-rank approximation matrices are recovered from sampled entries, with finite-sample guarantees provided for entry-wise estimation.

### Key Designs

**Spectral Recoverability**: A new quantity—the spectral recoverability parameter $\kappa$—is introduced to jointly control both the low-rank approximation error and the error in recovering the matrix from sampled entries. A key finding is that $\kappa$ decays rapidly as the shift $k$ increases.

**Type II Poincaré Inequality**: This is the paper's central theoretical contribution. While classical Poincaré inequalities characterize the global mixing rate of a Markov chain, the Type II variant characterizes "local" mixing properties—specifically, the degree of distributional "spreading" after $k$ steps from a given initial distribution. This inequality precisely quantifies:

- The required shift $k$, determined by the decay rate of the higher-order singular values of the shifted successor measure.
- That $k$ is typically small in practice, since higher-order singular values decay rapidly.

**Automatic Selection of Shift**: By analyzing the local mixing properties of the underlying dynamical system, a connection is established between the required shift and the system's mixing time, yielding a natural method for choosing $k$.

### Loss & Training

In the goal-conditioned RL setting:
- The shifted successor measure replaces the original version.
- A matrix completion algorithm recovers the low-rank approximation from sampled entries.
- The recovered low-rank representation is used for reward prediction and policy optimization.

## Key Experimental Results

### Main Results

Experiments evaluate the low-rank approximation quality and goal-conditioned RL performance of shifted vs. unshifted successor measures across multiple GridWorld and continuous control environments.

| Environment | Method | Approx. Error (RMSE) | Success Rate (%) |
|---|---|---|---|
| 4-Room GridWorld | Unshifted, rank-5 | 0.342 | 45.2 |
| 4-Room GridWorld | Shifted (k=3), rank-5 | 0.067 | 82.7 |
| 4-Room GridWorld | Shifted (k=5), rank-5 | 0.041 | 86.3 |
| Open GridWorld | Unshifted, rank-5 | 0.198 | 62.8 |
| Open GridWorld | Shifted (k=3), rank-5 | 0.032 | 91.5 |
| Maze | Unshifted, rank-10 | 0.285 | 38.1 |
| Maze | Shifted (k=5), rank-10 | 0.058 | 78.4 |

| Environment | FB (original) | FB+Shift (k=3) | FB+Shift (k=5) | Oracle |
|---|---|---|---|---|
| PointMaze-Medium | 0.61 | 0.79 | 0.82 | 0.95 |
| PointMaze-Large | 0.43 | 0.68 | 0.73 | 0.89 |
| AntMaze-Medium | 0.35 | 0.52 | 0.57 | 0.78 |
| AntMaze-Large | 0.22 | 0.41 | 0.46 | 0.65 |

### Ablation Study

**Effect of shift $k$ (4-Room GridWorld, rank-5)**:

| Shift $k$ | 0 | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| RMSE | 0.342 | 0.215 | 0.112 | 0.067 | 0.041 | 0.035 | 0.033 |
| Singular value decay rate | 0.82 | 0.65 | 0.43 | 0.28 | 0.15 | 0.09 | 0.07 |

- Error stabilizes after $k = 3 \sim 5$, consistent with theoretical predictions.
- Larger $k$ yields only marginal additional gains, as higher-order singular values have already sufficiently decayed.

**Effect of rank $r$ (Shifted, k=3)**:

| Rank $r$ | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| 4-Room RMSE | 0.125 | 0.067 | 0.042 | 0.038 | 0.036 |
| Open RMSE | 0.068 | 0.032 | 0.018 | 0.015 | 0.014 |

### Key Findings

1. **Low-rank structure requires shifting to emerge**: The singular values of the raw successor measure decay slowly; shifting substantially accelerates this decay.
2. **Small $k$ suffices**: A shift of 3–5 steps is typically sufficient to induce good low-rank structure.
3. **Goal-conditioned RL benefits substantially**: Representations based on the shifted successor measure yield meaningful performance gains across all environments.
4. **Connection to mixing time**: The required shift correlates with the local mixing properties of the environment; narrow corridors or bottlenecks necessitate larger $k$.

## Highlights & Insights

- **Deep theoretical insight**: The paper identifies a widely assumed but empirically invalid low-rank assumption and provides a principled correction.
- **Novel mathematical tool**: The Type II Poincaré inequality is an independent theoretical contribution of broader interest.
- **Theory–practice alignment**: Experimental results closely match theoretical predictions.
- **NeurIPS Spotlight**: The quality of the work has been highly recognized.

## Limitations & Future Work

1. **Computational overhead**: The shift operation requires additional policy rollout steps to collect shifted data.
2. **Extension to continuous spaces**: The theory primarily targets discrete/finite state spaces; generalization to continuous spaces requires additional technical development.
3. **Non-stationary policies**: When the policy changes continuously during training, the successor measure also changes, potentially destabilizing the effect of shifting.
4. **Automatic tuning of shift**: Although theory provides guidance, efficiently selecting $k$ in practice warrants further empirical investigation.
5. **Integration with offline RL**: Applying the shifted successor measure to offline datasets may require additional consideration of distributional shift.

## Related Work & Insights

- **Forward-Backward (FB) representations**: Touati & Ollivier (2021); this paper improves upon the low-rank assumption therein.
- **Successor Features/Measures**: Dayan (1993), Blier et al. (2021).
- **Matrix completion theory**: Candès & Recht (2009), providing theoretical foundations for recovering low-rank matrices from partial observations.
- **Poincaré inequalities**: Classical tools for Markov chain mixing time analysis.

## Rating

- **Novelty**: 5/5 — Identifies an important overlooked problem and proposes an elegant solution.
- **Technical Quality**: 5/5 — Rigorous theoretical analysis with strong experimental validation.
- **Writing Quality**: 4/5 — The 63-page paper is content-rich but lengthy.
- **Value**: 4/5 — The concept is simple and readily integrable into existing algorithms.
- **Overall**: 4.5/5

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[ICLR 2026\] Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](../../ICLR2026/reinforcement_learning/online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)
- [\[NeurIPS 2025\] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control](succeed_or_learn_slowly_sample_efficient_off-policy_reinforcement_learning_for_m.md)
- [\[ICLR 2026\] Dual Goal Representations](../../ICLR2026/reinforcement_learning/dual_goal_representations.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)

<!-- RELATED:END -->
