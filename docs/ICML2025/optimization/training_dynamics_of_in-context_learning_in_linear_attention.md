---
title: >-
  [Paper Note] Training Dynamics of In-Context Learning in Linear Attention
description: >-
  [ICML2025 Spotlight][Optimization][In-context learning] This paper completely characterizes the dynamical process of multi-head linear attention acquiring in-context learning (ICL) capabilities during gradient flow training: the merged KQ parametrization exhibits a single abrupt loss drop, whereas the separate KQ parametrization displays saddle-to-saddle progressive learning of principal component regression with staircase training dynamics.
tags:
  - "ICML2025 Spotlight"
  - "Optimization"
  - "In-context learning"
  - "linear attention"
  - "training dynamics"
  - "gradient flow"
  - "saddle-to-saddle dynamics"
  - "principal component regression"
date: 2026-05-08
content_hash: 1d4e121628599baa
---

# Training Dynamics of In-Context Learning in Linear Attention

**Conference**: ICML2025 Spotlight  
**arXiv**: [2501.16265](https://arxiv.org/abs/2501.16265)  
**Code**: [yedizhang/linattn-icl](https://github.com/yedizhang/linattn-icl)  
**Area**: Optimization  
**Keywords**: In-context learning, linear attention, training dynamics, gradient flow, saddle-to-saddle dynamics, principal component regression

## TL;DR

This paper completely characterizes the dynamical process of multi-head linear attention acquiring in-context learning (ICL) capabilities during gradient flow training: the merged KQ parametrization exhibits a single abrupt loss drop, whereas the separate KQ parametrization displays saddle-to-saddle progressive learning of principal component regression with staircase training dynamics.

## Background & Motivation

- **Emergence Mechanism of ICL Capability**: The in-context learning (ICL) capability of Transformers often emerges abruptly during training (characterized by an abrupt loss drop), such as the formation of induction heads (Olsson et al., 2022). However, theoretical understanding of this training dynamics remains preliminary.
- **Limitations of Prior Work**: Previous work (Zhang et al., 2024a) only proved convergence guarantees (i.e., what it eventually converges to) but did not describe the full trajectory of the training dynamics, and was limited to the single-head merged KQ case.
- **Importance of Parametrization**: Theoretical studies often use merged KQ ($W^{KQ} = W^{K\top}W^Q$) to simplify analysis, but practical Transformers utilize separate KQ. The differences in training dynamics between the two have not been systematically studied.
- **Core Problem**: How does ICL capability progressively evolve during gradient descent training? How do different parametrization methods affect the loss landscape and training dynamics?

## Method

### Problem Setup

This paper studies the standard in-context linear regression task with input sequences $\{x_1, y_1, \ldots, x_N, y_N, x_q\}$, where the goal is to predict $y_q = w^\top x_q$. The task vector $w \sim \mathcal{N}(0, I)$ is independently sampled for each sequence.

### Two Parametrizations

**ATTN_M (Merged KQ)**: Merges the key and query matrices into a single matrix $W_i^{KQ}$, yielding the prediction:

$$\hat{y}_q = \sum_{i=1}^{H} v_i \beta^\top U_i x_q$$

where $\beta = \frac{1}{N}\sum_{n=1}^N y_n x_n$ represents the input-output correlation in the context.

**ATTN_S (Separate KQ)**: Utilizes independent low-rank key and query matrices, yielding the prediction:

$$\hat{y}_q = \sum_{i=1}^{H} \sum_{r=1}^{R} v_i \beta^\top k_{i,r} q_{i,r}^\top x_q$$

### Key Equivalences

- **ATTN_M to Two-Layer Fully Connected Linear Network**: By defining cubic features $z = \text{vec}(\beta x_q^\top)$, ATTN_M is equivalent to $\hat{y}_q = w_2^\top W_1 z$, where $w_2$ is stacked from the value weights and $W_1$ is stacked from the merged KQ weights.
- **ATTN_S to Sum of Three-Layer Convolutional Linear Networks**: ATTN_S is equivalent to the sum of $H$ three-layer convolutional linear networks, where the key matrix $K_i$ acts as a convolutional kernel with stride $D$.

### Training Dynamics of ATTN_M

**Loss landscape**: Contains only 2 stationary points:
- $\mathcal{M}_0$: Zero point (unstable)
- $\mathcal{M}_*$: Global optimal manifold (stable)

**Dynamical behavior**: Initiated from a small initialization, the model first stays near the zero point (plateau), then undergoes a single abrupt loss drop to reach the global optimum.

**Analytical solution** (under whitened input $\Lambda = I$):

$$\hat{y}_q(t) = \sigma(t) \beta^\top x_q, \quad \sigma(t) = \frac{e^{2\sqrt{D}\frac{t}{\tau}}}{\left(1 + \frac{1+D}{N}\right)(e^{2\sqrt{D}\frac{t}{\tau}} - 1) + \frac{\sqrt{D}}{w_{\text{init}}^2}}$$

$\sigma(t)$ is a rescaled sigmoid, explaining the "plateau + abrupt drop" shape of the training curve.

### Training Dynamics of ATTN_S

**Loss landscape**: There are $2^D$ stationary points in the function space, corresponding to learning different subsets of eigenvectors of the covariance matrix $\Lambda$.

**Saddle-to-saddle dynamics**: Starting from small initialization, the model sequentially passes through $D+1$ stationary points $\mathcal{M}_0 \to \mathcal{M}_1 \to \cdots \to \mathcal{M}_D$. Each transition corresponds to learning a new eigenvector, ordered by eigenvalues from largest to smallest.

**Loss formula** (at the $m$-th stationary point):

$$\mathcal{L}(\mathcal{M}_m) = \text{tr}(\Lambda) - \sum_{d=1}^m \lambda_d \left(1 + \frac{1 + \text{tr}(\Lambda)/\lambda_d}{N}\right)^{-1}$$

**Scalar ODE simplification**: Through an ansatz, the high-dimensional dynamics on the $(m+1)$-th plateau simplify to:

$$\tau \dot{v}_i = \lambda_{m+1}^2 v_i^2 - \lambda_{m+1}^3 \left(1 + \frac{1 + \text{tr}(\Lambda)/\lambda_{m+1}}{N}\right) v_i^5$$

### Influence of Rank-R

For rank-$R$ separate KQ, the loss exhibits distinct plateaus when $R$ divides $m$; otherwise, the plateaus are brief or non-existent. This is because the $R$ pairs of KQ weights within the same head share the value weight $v_i$. Once $v_i$ grows, the learning of the remaining KQ pairs within that head is accelerated.

## Key Experimental Results

### Main Verification (Figure 1)

| Setup | D | N | H | Phenomenon |
|------|---|---|---|------|
| ATTN_M | 4 | 31 | 8 | Single abrupt loss drop, perfectly matching the loss trajectory of the equivalent linear network. |

### Saddle-to-Saddle Dynamics Verification (Figure 3)

| Setup | D | N | H | $\Lambda$ Eigenvalues | Phenomenon |
|------|---|---|---|------------------|------|
| ATTN_S rank-1 | 4 | 31 | 4 | 0.4, 0.3, 0.2, 0.1 | Exactly 4 abrupt loss drops; the loss values at the plateaus perfectly match the theoretical predictions (Eq. 19). |

### Rank Impact Verification (Figure 4)

| D=8, N=31, H=9 | R=1 | R=2 | R=4 | R=8 |
|----------------|-----|-----|-----|-----|
| Number of distinct plateaus | 8 | 4 | 2 | 1 |
| Plateau locations | Every $m$ | $m=0,2,4,6$ | $m=0,4$ | $m=0$ |

### Softmax Generalization Verification (Figure 5)
- Softmax ATTN_M: Similarly exhibits a single abrupt loss drop.
- Softmax ATTN_S: Similarly exhibits multiple loss drops, qualitatively consistent with the theoretical predictions of linear attention.

### Ablation: Optimization Scale (Figure 6)
- Increasing initialization scale: Shortens the duration of all plateaus.
- Extremely large initialization: Degenerates to exponential decay (lazy learning regime).
- Intermediate initialization: Exhibits a mixture of exponential decay and sigmoid shapes, closely resembling practical training curves.

## Highlights & Insights

1. **Equivalence between Linear Attention and Linear Networks**: ATTN_M is equivalent to a two-layer fully connected linear network, while ATTN_S is equivalent to a sum of three-layer convolutional linear networks. This construction brings rich theoretical tools from linear networks into the study of attention models.
2. **Parametrization Dictates Training Dynamics**: Merged KQ yields a single abrupt drop (2 stationary points), whereas Separate KQ results in progressive step-by-step learning ($2^D$ stationary points), revealing a widely overlooked theoretical factor.
3. **Progressive Acquisition of ICL**: ATTN_S progressively implements principal component regression (PCR) during training, with the number of principal components increasing. This provides a theoretical explanation for the progressive development of ICL capability.
4. **Scalar ODE Simplification**: The high-dimensional gradient flow dynamics are successfully simplified to a one-dimensional ordinary differential equation (ODE) that matches numerical simulations with high fidelity, serving as a highly elegant theoretical tool.
5. **Bridging Theory and Practice**: The theoretical results are qualitatively verified in softmax attention as well, reinforcing the practical significance of the theoretical findings.

## Limitations & Future Work

1. **Limited to Linear Attention**: Although softmax experiments are qualitatively consistent, the theoretical analysis strictly relies on the linear structure without softmax, preventing direct generalization to standard Transformers.
2. **Single-Layer Attention**: Only a single layer of attention is studied, lacking the treatment of inter-layer interactions and residual connections in multi-layer Transformers.
3. **Pure ICL Tasks**: The task setup is limited to pure ICL with $w \sim \mathcal{N}(0, I)$, excluding the effects of in-weight learning (IWL). In practice, the interplay between ICL and IWL is much more complex.
4. **Infinite Data Assumption**: The analysis is based on population loss, neglecting the effects of finite training samples.
5. **Restriction of Whitened Inputs**: Analytical solutions are only obtained for $\Lambda = I$. Closed-form solutions are absent for general covariance matrices.

## Related Work & Insights

- **Zhang et al., 2024a**: Analyzed the convergence guarantees of single-head merged KQ linear attention. This paper builds on their work to describe the complete training dynamics and extends it to multi-head separate KQ.
- **Olsson et al., 2022**: Discovered the abrupt emergence of ICL capabilities (induction heads). This paper theoretically explains the origin of this abrupt transition.
- **Singh et al., 2023**: Found that ICL can be a transient capability during training, which complements the staircase dynamics characterized in this paper.
- **Saxe et al., 2014, 2019**: Theory of saddle-to-saddle dynamics in deep linear networks. This paper applies it to attention models via equivalence.
- **Von Oswald et al., 2023**: Proposed the framework of utilizing linear attention for ICL linear regression. This paper provides a complete training dynamics description based on this framework.
- **Insight**: The impact of parametrization on optimization dynamics deserves systematic study in more complex models; linear network theory serves as a powerful tool for understanding attention mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The comparative analysis of dual parametrizations, characterization of $2^D$ stationary points, and progressive emergence of PCR are all proposed for the first time.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (The theory-to-simulation match is exceptionally high, and the softmax generalization verification is convincing, though large-scale experiments are lacking.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, with a very smooth logical chain from equivalence $\to$ landscape $\to$ dynamics $\to$ ICL algorithm.)
- Value: ⭐⭐⭐⭐⭐ (Provides the most comprehensive theoretical framework to date for understanding ICL training dynamics.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] In-Context Linear Regression Demystified: Training Dynamics and Mechanistic Interpretability of Multi-Head Softmax Attention](in-context_linear_regression_demystified_training_dynamics_and_mechanistic_inter.md)
- [\[ICML 2025\] On Understanding Attention-Based In-Context Learning for Categorical Data](on_understanding_attention-based_in-context_learning_for_categorical_data.md)
- [\[ICML 2025\] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias](how_transformers_learn_regular_language_recognition_a_theoretical_study_on_train.md)
- [\[ICML 2025\] Can Transformers Learn Full Bayesian Inference In Context?](can_transformers_learn_full_bayesian_inference_in_context.md)
- [\[ICML 2026\] Test time training enhances in-context learning of nonlinear functions](../../ICML2026/optimization/test_time_training_enhances_in-context_learning_of_nonlinear_functions.md)

</div>

<!-- RELATED:END -->
