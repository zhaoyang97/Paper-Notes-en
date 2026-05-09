---
title: >-
  [Paper Note] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products
description: >-
  [NeurIPS 2025][Video Understanding][Linear RNN] This paper proposes DeltaProduct, which extends DeltaNet's single-step gradient descent to $n_h$-step gradient descent per token, yielding a state transition matrix expressed as a product of $n_h$ generalized Householder transformations. This achieves a tunable trade-off between expressivity and efficiency, significantly improving state-tracking capability and length extrapolation performance.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Linear RNN
  - Householder Product
  - State-Tracking
  - DeltaNet
  - Length Extrapolation
date: 2026-05-08
content_hash: afcafe0d85fc9134
---

# DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products

**Conference**: NeurIPS 2025
**arXiv**: [2502.10297](https://arxiv.org/abs/2502.10297)
**Code**: [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)
**Area**: Sequence Modeling / Linear RNN
**Keywords**: Linear RNN, Householder Product, State-Tracking, DeltaNet, Length Extrapolation

## TL;DR

This paper proposes DeltaProduct, which extends DeltaNet's single-step gradient descent to $n_h$-step gradient descent per token, yielding a state transition matrix expressed as a product of $n_h$ generalized Householder transformations. This achieves a tunable trade-off between expressivity and efficiency, significantly improving state-tracking capability and length extrapolation performance.

## Background & Motivation

**Background**: Linear RNNs (e.g., Mamba, GLA, mLSTM) have emerged as competitive alternatives to Transformers, offering efficient training and linear-time inference. Their core properties are determined by the state transition matrix, with diagonal matrices being dominant in current models.

**Limitations of Prior Work**: Diagonal state transition matrices, while computationally efficient, suffer from severely limited expressivity—for instance, they cannot perform modular 3 addition in finite precision. DeltaNet partially addresses this with a diagonal-plus-rank-1 structure, but still requires multiple layers to handle complex state-tracking tasks (e.g., the $S_5$ symmetric group problem).

**Key Challenge**: There exists a fundamental trade-off between expressivity and efficiency: diagonal matrices are efficient but weakly expressive, while full matrices are highly expressive but prohibitively expensive and unstable to train.

**Goal**: To systematically enhance the expressivity of the state transition matrix in linear RNNs while preserving training efficiency and recurrent stability.

**Key Insight**: From the online gradient descent perspective of DeltaNet, extending single-step gradient descent to $n_h$ steps naturally yields a structure consisting of products of $n_h$ Householder transformations.

**Core Idea**: By increasing the number of gradient descent steps $n_h$ per token, the state transition matrix is extended from a rank-1 update (DeltaNet) to a rank-$n_h$ update, achieving a continuous interpolation between diagonal and dense matrices.

## Method

### Overall Architecture

DeltaProduct builds on the linear recurrence of DeltaNet:

$$H_i = A(x_i) H_{i-1} + B(x_i)$$

Each recurrent step of DeltaNet can be interpreted as one step of online gradient descent on the associative memory loss $\mathcal{L}_i(H) = \frac{1}{2}\|H^\top k_i - v_i\|_2^2$. DeltaProduct instead performs $n_h$ gradient descent steps per token, using distinct key/value pairs at each step.

### Key Designs

#### 1. Multi-Step Gradient Descent Generating Householder Products

For each input $x_i$, the model generates $n_h$ groups of keys $k_{i,j}$, values $v_{i,j}$, and step sizes $\beta_{i,j}$ ($j=1\ldots n_h$), applying sequential gradient updates:

$$H_{i,j} = (I - \beta_{i,j} k_{i,j} k_{i,j}^\top) H_{i,j-1} + \beta_{i,j} k_{i,j} v_{i,j}^\top$$

When unrolled, the state transition matrix becomes a product of $n_h$ generalized Householder transformations:

$$A(x_i) = \prod_{j=1}^{n_h} (I - \beta_{i,j} k_{i,j} k_{i,j}^\top)$$

#### 2. Spectral Norm Guarantees Stability

Each Householder factor has spectral norm $\leq 1$ (when $\beta \in [0,2]$), and the product likewise has spectral norm $\leq 1$, ensuring recurrent stability. This is a key structural advantage over RWKV-7, whose state transition matrix may have spectral norm $> 1$, posing instability risks.

#### 3. Gated DeltaProduct

Analogous to Gated DeltaNet, a scalar gate $g_i \in [0,1]$ is introduced:

$$A(x_i) = g_i \prod_{j=1}^{n_h} (I - \beta_{i,j} k_{i,j} k_{i,j}^\top)$$

#### 4. Extending Eigenvalue Range to $[-1, 1]$

The range of $\beta$ is extended from $[0,1]$ to $[0,2]$ (via $2 \times \text{sigmoid}$), enabling negative eigenvalues—a property critical for state tracking. Experiments show that restricting to $[0,1]$ causes the model to completely fail to learn.

#### 5. Geometric Intuition via the Cartan–Dieudonné Theorem

Two reflections composed yield a rotation (as illustrated in Figure 3). When $n_h$ is sufficiently large, products of Householder transformations can represent any orthogonal matrix. Already at $n_h=2$, the model can represent rotations in $\text{SO}(3)$, which suffices to solve $S_4$ (the rotation group of the cube) and $A_5$ (the rotation group of the regular dodecahedron).

### Implementation Details

The $n_h$ groups of key/value/beta are arranged as $n_h$ times the sequence length and passed to the existing Triton parallel implementation of DeltaNet, with gate values set to 1 at non-initial steps. Only the output at every $n_h$-th step of the recurrence is retained.

### Loss & Training

- Language models are trained on the FineWeb dataset
- Training context length: 4096 tokens
- Parameter counts are matched by scaling the number of heads or head dimension
- Throughput at the 1.3B scale is measured on H100 GPUs

## Key Experimental Results

### Main Results: State Tracking

| Task | Min $n_h$ (single layer) | Solvable with 3 layers | Theoretical lower bound |
|------|--------------------------|------------------------|------------------------|
| $S_3$ | 2 | ✓ | $n_h = 2$ |
| $S_4$ | 2 (via SO(3) isomorphism) | ✓ | $n_h = 3$ |
| $A_5$ | 2 (via SO(3) isomorphism) | ✓ | $n_h = 4$ |
| $S_5$ | 4 | ✓ | $n_h = 4$ |

By contrast, DeltaNet ($n_h=1$) fails to solve $S_5$ even with 10 layers.

### Main Results: Language Modeling

- DeltaProduct$_2$[-1,1] (8 heads, 392M parameters) significantly outperforms DeltaNet (12 heads, matched parameters) on length extrapolation
- At $n_h=3$, cross-entropy loss shows almost no degradation beyond the training length
- DeltaProduct$_3$[-1,1] without gating achieves performance comparable to Gated DeltaNet[-1,1]

### Ablation Study

| Configuration | CodeParrot Extrapolation | TriviaQA Extrapolation | OpenThoughts Extrapolation |
|---------------|--------------------------|------------------------|---------------------------|
| $n_h=1$ (DeltaNet) | Rapid degradation | Rapid degradation | Rapid degradation |
| $n_h=2$ | Substantial improvement | Substantial improvement | Substantial improvement |
| $n_h=3$ | Almost no degradation | Almost no degradation | Almost no degradation |

### Key Findings

1. **Effective Rank Analysis**: The effective rank of DeltaNet's hidden state grows continuously beyond the training length (causing distributional shift), whereas certain heads in DeltaProduct$_3$ learn to reset the effective rank at BOS tokens.
2. **PCA Verification**: On the $S_4$ task, the key vectors of the $n_h=2$ model are indeed concentrated in a 3-dimensional subspace (explaining $> 95\%$ of variance), and $\beta$ values cluster near 2 (reflections), confirming the theorized exploitation of the SO(3) isomorphism.
3. **Scaling Analysis**: Increasing $n_h$ yields better training perplexity scaling than increasing head dimension.

## Highlights & Insights

1. **Elegant Extension of the Online Learning Perspective**: The progression from DeltaNet's "single-step gradient descent" to DeltaProduct's "multi-step gradient descent" is both intuitive and mathematically natural.
2. **Perfect Correspondence Between Theory and Experiment**: The model autonomously discovers the strategy of exploiting the SO(3) isomorphism on $S_4$, with PCA and $\beta$-value analyses providing compelling evidence.
3. **Stability Guarantee**: The spectral norm of the Householder product is $\leq 1$, constituting a structural advantage of this parameterization over RWKV-7.
4. **Accelerated Forgetting Mechanism**: DeltaProduct can reset the hidden state $n_h$ times faster, which explains the observed improvements in length extrapolation.

## Limitations & Future Work

1. **Linear Growth in Training Cost**: Recurrent computation scales linearly with $n_h$ (the primary limitation).
2. **Parameter Overhead**: Additional key/value projections increase parameter count, requiring adjustment of head count or dimension to match budgets.
3. **Future Directions**: Adaptive determination of $n_h$ per token (analogous to Graves' Adaptive Computation Time); use of LoRA MLPs to reduce parameter overhead; integration with Fixed-Point RNNs.

## Related Work & Insights

- **DeltaNet/Gated DeltaNet**: The direct foundation; DeltaProduct reduces to these at $n_h=1$.
- **RWKV-7**: Uses a similar non-diagonal structure but without stability guarantees; DeltaProduct achieves even greater expressivity than RWKV-7 within 3 layers.
- **Grazzi et al. (ICLR 2025)**: Establishes the importance of negative eigenvalues; this paper extends their theoretical results.
- **Fixed-Point RNN**: An orthogonal line of work that increases the expressivity of diagonal RNNs via fixed-point iteration; can be combined with DeltaProduct.

## Rating

⭐⭐⭐⭐⭐

Theoretically rigorous (characterizing state-tracking capability under finite precision), experimentally comprehensive (spanning toy tasks to language modeling), and engineering-ready (integrated into the flash-linear-attention library), this paper represents an important advance in the study of expressivity in linear RNNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[NeurIPS 2025\] Agentic Persona Control and Task State Tracking for Realistic User Simulation](agentic_persona_control_and_task_state_tracking_for_realistic_user_simulation_in.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)

</div>

<!-- RELATED:END -->
