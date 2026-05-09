---
title: >-
  [Paper Note] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach
description: >-
  [NeurIPS 2025][Medical Imaging][Multi-Objective Reinforcement Learning] This paper reformulates max-min multi-objective reinforcement learning as a two-player zero-sum regularized continuous game, proposes the ERAM/ARAM algorithms, and leverages mirror descent to achieve a concise closed-form weight update. The approach guarantees global last-iterate convergence and significantly outperforms existing methods on tasks such as traffic signal control.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Multi-Objective Reinforcement Learning
  - Max-Min Fairness
  - Zero-Sum Game
  - Mirror Descent
  - Last-Iterate Convergence
date: 2026-05-08
content_hash: b98ca0377d2f9fda
---

# Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach

**Conference**: NeurIPS 2025
**arXiv**: [2510.20235](https://arxiv.org/abs/2510.20235)
**Code**: [GitHub](https://github.com/whbyeon/ERAM-ARAM)
**Area**: Medical Imaging
**Keywords**: Multi-Objective Reinforcement Learning, Max-Min Fairness, Zero-Sum Game, Mirror Descent, Last-Iterate Convergence

## TL;DR

This paper reformulates max-min multi-objective reinforcement learning as a two-player zero-sum regularized continuous game, proposes the ERAM/ARAM algorithms, and leverages mirror descent to achieve a concise closed-form weight update. The approach guarantees global last-iterate convergence and significantly outperforms existing methods on tasks such as traffic signal control.

## Background & Motivation

**Background**: Multi-objective reinforcement learning (MORL) addresses MDPs with vector-valued rewards, where utility-function-based methods—most commonly weighted sums—are the dominant paradigm. Max-min MORL maximizes the worst-performing objective dimension and is naturally suited to fairness-oriented resource allocation scenarios.

**Limitations of Prior Work**: (i) GGF-DQN/GGF-PPO optimize a surrogate objective $\mathbb{E}_\pi[\min_k \sum_t \gamma^t r_k]$ (a Jensen lower bound) rather than the true objective $\min_k \mathbb{E}_\pi[\sum_t \gamma^t r_k]$; (ii) the double-loop method of Park et al. directly optimizes the true objective but incurs substantial memory and computational overhead (20 Q-networks) and guarantees only average-iterate convergence; (iii) GGF-PPO updates only the worst-performing dimension per batch, effectively switching $w$ among one-hot vectors, yielding only average-iterate convergence.

**Key Challenge**: The nonlinearity of the $\min$ operator prevents max-min MORL from being solved directly by standard RL methods, while simultaneously demanding numerically efficient solvers.

**Goal**: Design a computationally and memory-efficient max-min MORL algorithm with last-iterate convergence guarantees.

**Key Insight**: Exploit a special equivalence between max-min and min-max under entropy regularization to transform the problem into finding a Nash equilibrium of a two-player zero-sum game.

**Core Idea**: The Learner updates the policy via NPG/PPO, while the Adversary updates the weight vector $w$ using the closed-form softmax formula derived from entropy-regularized mirror descent—two players jointly achieving max-min fairness.

## Method

### Overall Architecture

**Chain of Equivalences**:
$$\max_\pi \min_{k} V_{k,\tau}^\pi = \max_\pi \min_{w \in \Delta^K} \langle w, \mathbf{V}_\tau^\pi \rangle = \min_{w \in \Delta^K} \max_\pi \langle w, \mathbf{V}_\tau^\pi \rangle$$

The first equality extends the discrete $\min_k$ to $\min_w$ over the simplex; the third equality is the special max-min = min-max equivalence (Theorem 3.1). This defines a two-player zero-sum game:

- **Learner**: policy parameter $\theta$, maximizing $\langle w, \mathbf{V}_\tau^{\pi_\theta} \rangle$
- **Adversary**: weight vector $w \in \Delta^K$, minimizing $\langle w, \mathbf{V}_\tau^{\pi_\theta} \rangle$

### Key Designs

**Module 1: Regularized Game $\mathcal{RG}$**
- **Function**: Adds individual regularization terms to the utility function of the original game.
- Learner utility: $u_{Learner}^{\mathcal{RG}} = \langle w, \mathbf{V}^{\pi_\theta} \rangle + \tau \tilde{H}(\pi_\theta) - \tau_w H(w)$
- $\tau \tilde{H}(\pi_\theta)$: avoids the indeterminacy issue in max-min MORL (some MOMDPs admit only stochastic optimal policies).
- $-\tau_w H(w)$: prevents $w$ from degenerating to a one-hot vector, encouraging simultaneous consideration of multiple objective dimensions.

**Module 2: ERAM — Learner Update (NPG/PPO)**
- **Function**: Updates the policy via natural policy gradient.
- Tabular closed form: $\pi_{\theta_{t+1}}(a|s) = \frac{1}{Z_\pi} (\pi_{\theta_t}(a|s))^\alpha \exp\left(\frac{1-\alpha}{\tau} Q_{w_t,\tau}^{\pi_{\theta_t}}(s,a)\right)$, where $\alpha = 1 - \frac{\eta\tau}{1-\gamma}$
- Deep RL: directly replaced by PPO.

**Module 3: ERAM — Adversary Update (Closed-Form MD)**
- **Function**: Updates the weight $w$ via a modified mirror descent step.
- Core formula: $w_{t+1} = \text{softmax}\left(-\frac{1-\beta}{\tau_w} \mathbf{V}^{\pi_{\theta_t}} + \beta \log w_t\right)$
- where $\beta = \frac{1}{\lambda \tau_w + 1} \in (0,1)$, always within the valid range.
- **Design Motivation**: Choosing negative-entropy regularization (rather than $\ell_2$-norm) yields an analytic mirror descent update, eliminating the need for projected gradient descent.

**Module 4: ARAM — Adaptive Regularization Extension**
- **Function**: Replaces the Adversary's regularizer from $H(w)$ (KL from uniform) with $-D_{KL}(w \| c)$.
- Reference vector $c$ is computed dynamically: $c_i = \text{softmax}(\mathbb{E}_{s,a}[r_i(s,a) r_{i'}(s,a)])$, where $i'$ is the worst-performing dimension in the previous batch.
- **Design Motivation**: Unlike ERAM, which treats all dimensions equally, ARAM assigns greater weight to underperforming dimensions (without exclusively focusing on the worst), accelerating convergence.

### Loss & Training

- Learner: PPO objective (deep RL setting).
- Adversary: closed-form softmax update (projection-free).
- Step-size relationship: $\eta = O(\epsilon)$ > $\lambda = O(\epsilon^2)$ (policy updates faster than weight updates).

## Key Experimental Results

### Main Results (Table 1, Traffic Signal Control)

| Environment | ARAM | ERAM | Park et al. | GGF-PPO | GGF-DQN | Avg-DQN |
|-------------|:----:|:----:|:-----------:|:-------:|:-------:|:-------:|
| Base-4 | **-1160** | -1387 | -1681 | -1731 | -1838 | -2774 |
| Asym-4 | **-2696** | -2732 | -3510 | -3501 | -3053 | -4245 |
| Asym-16 | **-15043** | -17334 | -23663 | -21663 | -17792 | -27499 |

### Ablation Study (Multi-Environment Results in Appendix)

| Experiment | Finding |
|------------|---------|
| Species conservation environment | ARAM/ERAM significantly outperform baselines |
| MO-Reacher environment | Same as above |
| Four-Room environment | Same as above |
| $\beta$ ablation | Performance is stable when $\beta$ is not close to 0 (ERAM) or 1 (ARAM) |

### Efficiency Comparison (Table 2, Training Time)

| Environment | ERAM | ARAM | Park et al. | GGF-PPO |
|-------------|:----:|:----:|:-----------:|:-------:|
| Base-4 | **111 min** | 120 min | 346 min | 122 min |
| Asym-4 | **87 min** | 87 min | 241 min | 85 min |
| Asym-16 | **356 min** | 365 min | 1125 min | 394 min |

Memory: ERAM/ARAM use approximately 13,704 parameters per weight update vs. 274,084 for Park et al. (a ~95% reduction).

### Key Findings

1. ARAM consistently achieves the best performance across all traffic signal control environments; ERAM ranks second.
2. Compared to Park et al., training time is reduced by approximately 65–68% and memory usage by approximately 95%.
3. ERAM exhibits last-iterate convergence (Figure 3), whereas Park et al. displays oscillatory behavior.
4. Adaptive regularization (ARAM vs. ERAM) yields significant improvements at negligible additional cost.

## Highlights & Insights

1. **Exploiting the max-min = min-max equivalence**: In RL, value functions are non-concave in the policy, precluding direct application of the minimax theorem; however, equivalence holds in the regularized setting via a specialized construction.
2. **Closed-form weight update**: Negative-entropy regularization combined with a KL-divergence Bregman yields a closed-form softmax solution, eliminating the iterative overhead of projected gradient descent.
3. **GGF-PPO as a special case**: When $\tau_w = 0$ and $D_\psi(w, w_t)$ is removed, the proposed method reduces to GGF-PPO (with $w$ becoming a one-hot vector).
4. **Last-iterate convergence**: More practically useful than average-iterate convergence—the final policy can be deployed directly without maintaining a historical average.

## Limitations & Future Work

1. Theoretical analysis of ARAM is left for future work; current convergence guarantees apply only to ERAM in the tabular setting.
2. The regularization coefficients $\tau$ and $\tau_w$ introduce approximation error; the optimal solution deviates from the unregularized case by a term linear in $\tau$ and $\tau_w$.
3. Theoretical analysis in the tabular setting does not directly extend to deep RL (though empirical PPO substitution proves effective).
4. For large $K$ (Asym-16: $K=16$), performance gaps narrow; scaling efficiency to even larger $K$ remains to be verified.
5. The sample complexity $\tilde{O}(K / ((1-\gamma)^3 \epsilon^6 \epsilon_{acc}^2))$ has a strong dependence on $\epsilon$.

## Related Work & Insights

- **vs. Park et al. (2024)**: The latter employs a double loop with zeroth-order gradient estimation and projected gradient descent, guaranteeing only average-iterate convergence. The proposed method uses a single loop with closed-form updates and last-iterate convergence, achieving approximately 95% efficiency gains.
- **vs. GGF-PPO (Siddique et al., 2021)**: GGF-PPO optimizes only the worst dimension per batch (one-hot $w$); this work simultaneously considers multiple dimensions via entropy regularization.
- **vs. APMD (Abe et al., 2024)**: APMD assumes smooth monotone games, which does not apply to RL where value gradients are non-monotone.
- **Insight**: The game-theoretic perspective connects MORL to equilibrium computation; the regularization technique (negative entropy → closed-form solution) is also worth borrowing for other constrained RL problems.

## Rating

⭐⭐⭐⭐⭐ (5/5)

The paper tightly integrates theory and practice: the game-theoretic reformulation is elegant, the closed-form weight update is practical, and the last-iterate convergence guarantee represents a meaningful advance. Experiments on real-world traffic signal control substantially outperform baselines, with a 95% memory reduction and 65% training time reduction offering strong practical value. The adaptive regularization in ARAM further enhances performance. Code is publicly available.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking](a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[NeurIPS 2025\] A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)
- [\[NeurIPS 2025\] STAMP: Spatial-Temporal Adapter with Multi-Head Pooling](stamp_spatial-temporal_adapter_with_multi-head_pooling.md)

<!-- RELATED:END -->
