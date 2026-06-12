---
title: >-
  [Paper Note] The Expressive Limits of Diagonal SSMs for State-Tracking
description: >-
  [ICLR 2026][Video Understanding][State Space Models] This paper provides a complete characterization of the expressive power of input-dependent complex diagonal (DCD) SSMs on group state-tracking tasks: a single layer ca…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "State Space Models"
  - "Expressive Power"
  - "Group State Tracking"
  - "Diagonal SSM"
  - "Solvable Groups"
  - "Mamba"
date: 2026-05-08
content_hash: 96a1afc2eaab8dbb
---

# The Expressive Limits of Diagonal SSMs for State-Tracking

**Conference**: ICLR 2026
**arXiv**: [2603.01959](https://arxiv.org/abs/2603.01959)  
**Area**: Sequence Modeling / Theory
**Keywords**: State Space Models, Expressive Power, Group State Tracking, Diagonal SSM, Solvable Groups, Mamba

## TL;DR

This paper provides a complete characterization of the expressive power of input-dependent complex diagonal (DCD) SSMs on group state-tracking tasks: a single layer cannot track any non-abelian group, and $k$ layers can track group $G$ if and only if $G$ admits a subnormal series of length $k$ with abelian factors — precisely defining the strict expressiveness gains conferred by depth. Experiments further reveal a significant gap between expressive capacity and learnability.

## Background & Motivation

**Background**: State space models (SSMs), such as Mamba and S4D, have emerged as efficient alternatives to Transformers and demonstrated strong empirical performance on long-sequence modeling tasks. By employing linear recurrences with diagonalized state transition matrices, they achieve $O(\log T)$ parallelization. Nevertheless, the theoretical understanding of SSM expressive power remains limited, particularly on tasks such as state tracking.

**Limitations of Prior Work**:
1. SSMs were initially expected to outperform Transformers on state tracking due to their explicit state representations, yet experiments show that SSMs similarly fail.
2. Merrill et al. proved that SSMs belong to the $\mathsf{TC}^0$ complexity class and conjectured that they cannot track non-solvable groups (e.g., $S_5$), but this relies on unresolved conjectures in complexity theory.
3. Sarrof et al. proved that Mamba (with non-negative diagonal matrices) cannot solve the parity problem (the simplest non-trivial group $C_2$), yet the precise capability boundary of diagonal SSMs remains unknown.
4. The theoretical difference in expressive power between single-layer and multi-layer architectures is unclear — does depth yield strict benefits?

**Key Challenge**: Diagonalization is the foundation of efficient parallelization in SSMs, but it simultaneously constrains the expressive power of the state transition matrices. Understanding the expressive limits while preserving efficiency is an open theoretical challenge.

**Goal**: Adopting an algebraic group-theoretic perspective, this paper precisely characterizes which groups can be tracked by $k$-layer DCD SSMs — providing necessary and sufficient conditions that rely on no unproven conjectures.

## Method

### Overall Architecture

The object of study is the input-dependent complex diagonal (DCD) SSM, whose state recurrence is:

$$h_t = A(x_t) h_{t-1} + b(x_t)$$

where $A(x_t) \in \mathbb{C}^{d \times d}$ is a diagonal matrix and $b(x_t) \in \mathbb{C}^d$ is an input vector. The output is produced via a decoder $y_t = \text{dec}(h_t, x_t)$. The functions $A$, $b$, and $\text{dec}$ are assumed to be universal function approximators.

The group state-tracking task is defined as follows: given a group $G$ and an input sequence $x_1, \ldots, x_T \in G$, output $y_t = x_1 \cdot x_2 \cdots x_t$, i.e., the cumulative group product.

### Key Design 1: Impossibility Proof for Single-Layer DCD SSMs

**Core Theorem (Theorem 1)**: A single-layer DCD SSM with finite precision can track group $G$ if and only if $G$ is abelian.

The proof proceeds in three steps:

1. **Eliminating useless coordinates (Lemma 1)**: If a state coordinate contracts ($|\lambda(x)_j| < 1$), expands ($|\lambda(x)_j| > 1$), or drifts ($|\lambda(x)_j| = 1$ with $b(x)_j \neq 0$) under some input, one can construct an alternative SSM that fixes that coordinate to a constant without affecting tracking capability.

2. **Rotational center consistency (Lemma 2)**: The composition of a neutral rotation $h \mapsto \lambda(h - c_1) + c_1$ and a conjugate rotation $h \mapsto \lambda^*(h - c_2) + c_2$ (when $c_1 \neq c_2$) yields a non-zero translation — causing the state to diverge.

3. **Convergence requirement (Lemma 3)**: For finite group tracking, all state coordinates must share the same center of rotation. Combined with the commutativity of diagonal matrices, the state updates are necessarily commutative, implying the group must be abelian.

### Key Design 2: Necessary and Sufficient Conditions for Multi-Layer DCD SSMs

**Core Theorem (Theorem 2)**: A $k$-layer DCD SSM with finite precision can track group $G$ if and only if $G$ admits a subnormal series of length $k$:

$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_k = G$$

with each factor $G_{i+1}/G_i$ abelian. This is precisely the standard definition of a solvable group — refined to require **exactly** $k$ layers.

**Constructive proof (sufficiency)**: Taking $S_3$ as an example, it is decomposed as the semidirect product $C_3 \rtimes C_2$. The first layer tracks $C_2$ (parity), and the second layer tracks $C_3$ conditioned on the first-layer state (the direction of modular-3 rotation depends on parity):

- First-layer AUSSM: $\Lambda((0,\beta)) = 0$, $\Lambda((1,\beta)) = \pi$
- Second-layer AUSSM: $\Lambda^{(2)}((1,\alpha,\beta)) = \frac{2\pi}{3}\beta$, $\Lambda^{(2)}((-1,\alpha,\beta)) = -\frac{2\pi}{3}\beta$

### Key Design 3: Relationship to Existing SSM Variants

| Model | Transition Matrix $A(x)$ | Eigenvalue Range | Trackable Groups |
|-------|--------------------------|------------------|-----------------|
| S4/S4D | $\Lambda$ (fixed diagonal) | $\mathbb{C}$ | None (time-invariant) |
| Mamba | $\exp(\Delta(x) \odot \Lambda)$ | $(0, 1]$ (non-negative real) | No group |
| Negative Mamba | $2\exp(\Delta(x) \odot \Lambda) - I$ | $(-1, 1]$ | $C_2$ only (single layer) |
| AUSSM | $\exp(i\Delta(x) \odot \Lambda(x))$ | Unit circle $|z|=1$ | All abelian groups (single layer) |

## Key Experimental Results

### Main Results: State-Tracking Task (Longest Generalization Sequence Length; Training Length ≤ 60)

| Group | Abelian | Solvable | Mamba | Neg Mamba | Simple AUSSM | AUSSM | RNN |
|-------|---------|----------|-------|-----------|-------------|-------|-----|
| $C_2$ | ✓ | ✓ | ✘ | 1000 | 160 | 1000 | 1000 |
| $C_6$ | ✓ | ✓ | ✘ | ✘ | 240 | 940 | 1000 |
| $C_{24}$ | ✓ | ✓ | ✘ | ✘ | 240 | 260 | 1000 |
| $C_{60}$ | ✓ | ✓ | ✘ | ✘ | 300 | 240 | ✘ |
| $S_3$ | ✗ | ✓ | ✘ | ✘ | ✘ | ✘ | 1000 |
| $A_4$ | ✗ | ✓ | ✘ | ✘ | ✘ | ✘ | 1000 |
| $A_5$ | ✗ | ✗ | ✘ | ✘ | ✘ | ✘ | ✘ |

> ✘ indicates that the model fails to generalize to sequences of length ≥ 100. Reported values are the longest sequence lengths achieving accuracy > 90%.

### Ablation Study: Single-Layer vs. Two-Layer

| Group | Model | Single Layer | Two Layers | Theoretical Expectation |
|-------|-------|-------------|-----------|------------------------|
| $C_2$ | AUSSM | 1000 | 200 | Both capable |
| $C_2 \times C_4$ | Neg Mamba | ✘ | 360 | Two layers capable |
| $S_3$ | AUSSM | ✘ | ✘ | Two layers capable (gap!) |
| $A_4$ | AUSSM | ✘ | ✘ | Two layers capable (gap!) |

### Key Findings

- **Abelian groups**: Single-layer AUSSM and Simple AUSSM theoretically subsume all abelian groups; gradient-based optimization successfully finds generalizing solutions in experiments.
- **Non-abelian groups**: Two-layer AUSSM can theoretically track $S_3$ and $A_4$, but standard training **never succeeds** in practice — expressive capacity does not imply learnability.
- **Initialization sensitivity**: When AUSSM is initialized near the analytically derived solution for $S_3$, training succeeds and generalizes to sequences four times the training length — the solution exists but is difficult to find in the loss landscape.
- **RNN without theoretical constraints**: RNNs can track all solvable groups, but also encounter optimization difficulties on large groups (e.g., $C_{60}$).

## Highlights & Insights

- **Precise algebraic characterization**: Necessary and sufficient conditions are established rather than approximate capability claims — $k$ layers correspond exactly to groups admitting subnormal series of length $k$.
- **Strict depth benefit**: $k$ layers are strictly more expressive than $k-1$ layers — non-abelian groups require depth, and the required number of layers has a precise group-theoretic interpretation.
- **Critical distinction between expressiveness and learnability**: Theoretical expressiveness does not imply learnability via SGD — the community is cautioned against relying solely on theoretical expressive power, as optimization bottlenecks are equally critical.
- **Direct guidance for SSM design**: The non-negative eigenvalue constraint in Mamba is a fundamental limitation; allowing negative or complex eigenvalues is a necessary condition for unlocking stronger state-tracking capability.

## Limitations & Future Work

- Results are restricted to diagonal SSMs; block-diagonal variants (e.g., $2 \times 2$ blocks) may extend expressive power to $\mathsf{NC}^1$.
- The root cause of the learnability gap remains unresolved — the geometric structure of the loss landscape (flat regions, saddle points, isolated minima) requires further investigation.
- The analysis is currently limited to group state tracking, while practical sequence tasks are more complex — the predictive power of these theoretical results for real-world tasks remains to be validated.
- Hybrid architectures (e.g., alternating AUSSM and Mamba layers) for non-abelian group learnability have not been explored.
- The extension to semigroups and monoids is only preliminary; a complete theory has yet to be established.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mamba-3: Improved Sequence Modeling using State Space Principles](mamba-3_improved_sequence_modeling_using_state_space_principles.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/video_understanding/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](../../NeurIPS2025/video_understanding/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](../../AAAI2026/video_understanding/state-space_hierarchical_compression_with_gated_attention_an.md)
- [\[NeurIPS 2025\] DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](../../NeurIPS2025/video_understanding/deltaproduct_improving_state-tracking_in_linear_rnns_via_householder_products.md)

</div>

<!-- RELATED:END -->
