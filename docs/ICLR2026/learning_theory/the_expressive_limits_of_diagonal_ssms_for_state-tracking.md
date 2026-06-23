---
title: >-
  [Paper Note] The Expressive Limits of Diagonal SSMs for State-Tracking
description: >-
  [ICLR 2026][learning_theory][State Space Model] This work establishes a complete characterization of the expressive power of input-dependent complex diagonal (DCD) SSMs on group state-tracking tasks: a single layer cannot track any non-abelian group, whereas $k$ layers can track a group $G$ if and only if $G$ admits a subnormal series of length $k$ with abelian fact
tags:
  - ICLR 2026
  - learning_theory
  - State Space Model
  - Mamba
date: 2026-05-08
content_hash: 727b72725682bb98
---
# The Expressive Limits of Diagonal SSMs for State-Tracking

**Conference**: ICLR 2026  
**arXiv**: [2603.01959](https://arxiv.org/abs/2603.01959)  
**Area**: Sequence Modeling / Theory  
**Keywords**: State Space Models, Expressivity, Group State-Tracking, Diagonal SSM, Solvable Groups, Mamba

## TL;DR

This work establishes a complete characterization of the expressive power of input-dependent complex diagonal (DCD) SSMs on group state-tracking tasks: a single layer cannot track any non-abelian group, whereas $k$ layers can track a group $G$ if and only if $G$ admits a subnormal series of length $k$ with abelian factors—precisely defining the strict benefit of depth for expressivity. Furthermore, experiments reveal a significant gap between theoretical expressivity and learnability.

## Background & Motivation

**Background**: State Space Models (SSMs), such as Mamba and S4D, have emerged as efficient alternatives to Transformers, achieving strong empirical performance on long-sequence modeling. They achieve $O(\log T)$ parallelization through linear recurrence and diagonalized state transition matrices. However, the theoretical understanding of SSM expressivity remains limited, particularly for tasks like state-tracking.

**Limitations of Prior Work**:
1. SSMs were initially expected to outperform Transformers in state-tracking due to their explicit state representation, yet experiments show SSMs also fail in these tasks.
2. Merrill et al. proved that SSMs belong to the $\mathsf{TC}^0$ complexity class and conjectured they cannot track non-solvable groups (e.g., $S_5$), but this relies on unsolved conjectures in complexity theory.
3. Sarrof et al. demonstrated that Mamba (with non-negative diagonal matrices) cannot solve the parity problem (the simplest non-trivial group $C_2$), but the precise capacity boundaries of diagonal SSMs remain unknown.
4. The difference in expressivity between single-layer and multi-layer architectures is theoretically unclear—does depth provide a strict advantage?

**Key Challenge**: Diagonalization is fundamental to the efficient parallelization of SSMs but restricts the expressivity of the state transition matrix. How can one understand these expressive limits while maintaining efficiency?

**Goal**: From the perspective of algebraic group theory, provide a precise characterization of which groups a $k$-layer DCD SSM can track—offering necessary and sufficient conditions independent of any conjectures.

## Method

### Overall Architecture

This study does not propose a new model but rather rigorously analyzes exactly which groups can be tracked by Input-Dependent Complex Diagonal (DCD) SSMs. The recurrence studied is $h_t = A(x_t) h_{t-1} + b(x_t)$, where $A(x_t) \in \mathbb{C}^{d \times d}$ is a diagonal matrix and $b(x_t) \in \mathbb{C}^d$ is an input vector. The output is provided via a decoder $y_t = \text{dec}(h_t, x_t)$, assuming $A$, $b$, and $\text{dec}$ are universal function approximators. The task is group state-tracking: given a group $G$ and a sequence $x_1, \ldots, x_T \in G$, output the cumulative product $y_t = x_1 x_2 \cdots x_t$ at each step. The analysis first establishes the upper bound of single-layer capacity and then uses subnormal series to map the number of layers to trackable groups, finally relating these conclusions to existing variants like Mamba.

### Key Designs

**1. Single-layer Impossibility: Diagonal Commutativity Restricts Layers to Abelian Groups**

The first core conclusion (Theorem 1) is that a single-layer DCD SSM under finite precision can track $G$ if and only if $G$ is an abelian group—tracking non-abelian groups in a single layer is impossible. The proof hinges on the fact that diagonal matrices commute. Non-commutativity in group products cannot be recovered by commuting state updates. Three lemmas systematically close potential "escape routes." Lemma 1 eliminates useless coordinates: if a coordinate shrinks ($|\lambda(x)_j| < 1$), expands ($|\lambda(x)_j| > 1$), or drifts ($|\lambda(x)_j| = 1$ and $b(x)_j \neq 0$) under certain inputs, an equivalent SSM can be constructed to fix it without loss of tracking power, leaving only pure rotation coordinates. Lemma 2 points out that inconsistent rotation centers are problematic: a neutral rotation $h \mapsto \lambda(h - c_1) + c_1$ and a conjugate rotation $h \mapsto \lambda^*(h - c_2) + c_2$ with $c_1 \neq c_2$ produce a non-zero translation, causing state divergence and tracking failure under finite precision. Lemma 3 thus requires all coordinates to share a rotation center; combined with diagonal commutativity, state updates must commute, implying the group must be abelian.

**2. Multi-layer Necessary and Sufficient Conditions: Depth = Subnormal Series Length**

The second core conclusion (Theorem 2) quantifies depth: a $k$-layer DCD SSM can track $G$ if and only if $G$ possesses a subnormal series of length $k$

$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_k = G$$

where each quotient factor $G_{i+1}/G_i$ is abelian. This matches the definition of solvable groups but is refined to show that **exactly** $k$ layers are required. Non-solvable groups (e.g., $A_5$) cannot be tracked at any depth, while solvable groups have a specific minimal layer requirement. The sufficiency is constructive: using $S_3 = C_3 \rtimes C_2$ as an example, the first layer tracks the $C_2$ parity bit ($\Lambda((0,\beta)) = 0$, $\Lambda((1,\beta)) = \pi$), and the second layer uses this parity state as a condition to set the $C_3$ rotation direction ($\Lambda^{(2)}((1,\alpha,\beta)) = \tfrac{2\pi}{3}\beta$, $\Lambda^{(2)}((-1,\alpha,\beta)) = -\tfrac{2\pi}{3}\beta$), successfully reconstructing the non-commutative product. Each layer handles one abelian quotient, and the depth benefit is thus measured by a strict group-theoretic scale.

**3. Comparison with Existing Variants: Negative/Complex Eigenvalues as the Key to Tracking**

Applying this characterization to mainstream SSMs reveals their bottlenecks based on the range of eigenvalues in the transition matrix. In Mamba, $A(x) = \exp(\Delta(x) \odot \Lambda)$ takes values in $(0,1]$, lacking negative values and thus failing to solve even $C_2$ (parity). Negative Mamba uses $2\exp(\Delta(x) \odot \Lambda) - I$ to extend the range to $(-1,1]$, which is sufficient for $C_2$ in a single layer. AUSSM uses $\exp(i\Delta(x) \odot \Lambda(x))$, allowing eigenvalues to cover the unit circle $|z|=1$, enabling a single layer to cover all abelian groups. This comparison provides a direct design guide: restricting eigenvalues to be non-negative is a fundamental bottleneck for Mamba’s expressivity; allowing complex eigenvalues on the unit circle is necessary to track non-trivial groups.

| Model | Transition Matrix $A(x)$ | Eigenvalue Range | Trackable Groups |
|------|-----------------|-----------|-----------|
| S4/S4D | $\Lambda$ (Fixed Diagonal) | $\mathbb{C}$ | None (Time-invariant) |
| Mamba | $\exp(\Delta(x) \odot \Lambda)$ | $(0, 1]$ (Non-negative Real) | No Groups (No negative values) |
| Negative Mamba | $2\exp(\Delta(x) \odot \Lambda) - I$ | $(-1, 1]$ | $C_2$ only (Single layer) |
| AUSSM | $\exp(i\Delta(x) \odot \Lambda(x))$ | Unit Circle $|z|=1$ | All Abelian Groups (Single layer) |

## Key Experimental Results

### Main Results: State-Tracking Task (Max sequence length for generalization, training length $\leq 60$)

| Group | Abelian | Solvable | Mamba | Neg Mamba | Simple AUSSM | AUSSM | RNN |
|----------|--------|------|-------|-----------|-------------|-------|------|
| $C_2$ | ✓ | ✓ | ✘ | 1000 | 160 | 1000 | 1000 |
| $C_6$ | ✓ | ✓ | ✘ | ✘ | 240 | 940 | 1000 |
| $C_{24}$ | ✓ | ✓ | ✘ | ✘ | 240 | 260 | 1000 |
| $C_{60}$ | ✓ | ✓ | ✘ | ✘ | 300 | 240 | ✘ |
| $S_3$ | ✗ | ✓ | ✘ | ✘ | ✘ | ✘ | 1000 |
| $A_4$ | ✗ | ✓ | ✘ | ✘ | ✘ | ✘ | 1000 |
| $A_5$ | ✗ | ✗ | ✘ | ✘ | ✘ | ✘ | ✘ |

> ✘ indicates the model cannot generalize to sequence lengths $\geq 100$. Reported values are the maximum sequence length with accuracy $>90\%$.

### Ablation Study: Single-layer vs. Two-layer

| Group | Model | 1 Layer | 2 Layers | Theoretical Expectation |
|-----------------|---------------|-------|-------|----------|
| $C_2$ | AUSSM | 1000 | 200 | Both Possible |
| $C_2 \times C_4$| Neg Mamba | ✘ | 360 | 2 Layers Possible |
| $S_3$ | AUSSM | ✘ | ✘ | 2 Layers Possible (Gap!) |
| $A_4$ | AUSSM | ✘ | ✘ | 2 Layers Possible (Gap!) |

### Key Findings

- **Abelian Groups**: Theories for single-layer AUSSM and Simple AUSSM cover all abelian groups, and gradient optimization successfully identifies generalizing solutions in experiments.
- **Non-Abelian Groups**: While two-layer AUSSM can theoretically track $S_3$ and $A_4$, standard training **never succeeded** in experiments—demonstrating that expressivity $\neq$ learnability.
- **Initialization Sensitivity**: When AUSSM is initialized near the analytical solution for $S_3$, training succeeds and generalizes to 4x the training length, suggesting that solutions exist but are difficult to find in the loss landscape.
- **RNN Theoretical Capacity**: RNNs have no theoretical restrictions on solvable groups but encounter optimization difficulties on large groups like $C_{60}$.

## Highlights & Insights

- **Precise Algebraic Characterization**: Provides necessary and sufficient conditions rather than vague possibilities—the number of layers $k$ strictly corresponds to the length of a group's subnormal series.
- **Strict Benefit of Depth**: Proves $k$ layers are strictly more powerful than $k-1$ layers; non-abelian groups require depth, and the required number of layers has a precise group-theoretic meaning.
- **Distinction Between Expression and Learning**: Theoretical expressibility does not guarantee that SGD can learn a solution, highlighting that optimization bottlenecks are as critical as theoretical limits.
- **Direct Guidance for Mamba**: The non-negative eigenvalue constraint in Mamba is a fundamental limitation; allowing negative or complex eigenvalues is necessary for stronger state-tracking capabilities.

## Limitations & Future Work

- Results are limited to diagonal SSMs; block-diagonal structures (e.g., $2 \times 2$) might extend expressivity to $\mathsf{NC}^1$.
- The root cause of the learnability gap remains unsolved—the geometric structure of the loss landscape (flat regions, saddle points, or isolated minima) requires further study.
- Current discussions are limited to group state-tracking; the predictive power of these theoretical conclusions for more complex real-world tasks remains to be verified.
- Learnability of hybrid architectures (e.g., alternating AUSSM and Mamba layers) on non-abelian groups has not been explored.
- The extension to semigroups and monoids is preliminary, and a complete theory is yet to be established.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Expressiveness of State Space Models via Temporal Logics](on_the_expressiveness_of_state_space_models_via_temporal_logics.md)
- [\[ICLR 2026\] On the Computational Limits of AI4S-RL：A Unified $\varepsilon$-$N$ Analysis](on_the_computational_limits_of_ai4s-rl_a_unified_varepsilon-n_analysis.md)
- [\[ICLR 2026\] Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling](expressive_power_of_implicit_models_rich_equilibria_and_test-time_scaling.md)
- [\[ICLR 2026\] Characterizing Pattern Matching and Its Limits on Compositional Task Structures](characterizing_pattern_matching_and_its_limits_on_compositional_task_structures.md)
- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)

</div>

<!-- RELATED:END -->
