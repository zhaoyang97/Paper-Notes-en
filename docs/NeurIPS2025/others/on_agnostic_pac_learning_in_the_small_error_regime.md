---
title: >-
  [Paper Note] On Agnostic PAC Learning in the Small Error Regime
description: >-
  [NeurIPS 2025][PAC learning] In the small error regime of agnostic PAC learning ($\tau \approx d/m$), this paper constructs a computationally efficient learner based on ERM aggregation that achieves an error upper bound of $c \cdot \tau + O(\sqrt{\tau d/m} + d/m)$ with $c \leq 2.1$, matching known lower bounds and advancing the precise complexity characterization of agnostic learning.
tags:
  - NeurIPS 2025
  - PAC learning
  - agnostic learning
  - ERM
  - sample complexity
  - VC dimension
date: 2026-05-08
content_hash: b404101b75efb1aa
---

# On Agnostic PAC Learning in the Small Error Regime

**Conference**: NeurIPS 2025
**arXiv**: [2502.09496](https://arxiv.org/abs/2502.09496)
**Authors**: Julian Asilis, Mikael Møller Høgsgaard, Grigoris Velegkas
**Code**: None
**Area**: Learning Theory / Computational Learning Theory
**Keywords**: PAC learning, agnostic learning, ERM, sample complexity, VC dimension

## TL;DR

In the small error regime of agnostic PAC learning ($\tau \approx d/m$), this paper constructs a computationally efficient learner based on ERM aggregation that achieves an error upper bound of $c \cdot \tau + O(\sqrt{\tau d/m} + d/m)$ with $c \leq 2.1$, matching known lower bounds and advancing the precise complexity characterization of agnostic learning.

## Background & Motivation

An interesting phenomenon exists in classical PAC learning theory:
- **Realizable setting**: data are generated perfectly by some hypothesis in $\mathcal{H}$, and ERM is not optimal.
- **Agnostic setting**: no realizability assumption is made, and ERM turns out to be optimal.

The root cause of this "paradox" is that an agnostic learner, when facing a nearly realizable distribution (where $\tau = \text{err}(h^*_\mathcal{D})$ is small), is permitted a larger excess error, effectively masking the suboptimality of ERM in the agnostic setting.

**Hanneke, Larsen, and Zhivotovskiy (FOCS 2024)** established a more refined analytical framework by incorporating $\tau$ as a parameter in the error bound, proving the following lower bound:

$$\tau + \Omega\left(\sqrt{\frac{\tau(d + \log(1/\delta))}{m}} + \frac{d + \log(1/\delta)}{m}\right)$$

This lower bound is tight when $\tau > d/m$, but leaves open the question: **does a strictly higher lower bound exist when $\tau \approx d/m$?**

## Method

### Overall Architecture

The central contribution of this paper is the construction of a learner $A$ such that for any distribution $\mathcal{D}$, with probability $\geq 1-\delta$:

$$\text{err}(A(S)) \leq c \cdot \tau + O\left(\sqrt{\frac{\tau(d + \log(1/\delta))}{m}} + \frac{d + \log(1/\delta)}{m}\right)$$

where $c \leq 2.1$, $d = \text{VC}(\mathcal{H})$, and $m$ is the sample size.

### Key Designs

**ERM Aggregation Strategy**: The core of the learner is a principled aggregation of multiple ERM solutions:

1. **Sample splitting**: the $m$ training samples are divided into $T$ groups of $m/T$ samples each.
2. **Independent ERM**: ERM is run independently on each group, yielding $T$ candidate hypotheses $h_1, \ldots, h_T$.
3. **Tournament aggregation**: a pairwise tournament mechanism is used to select the final hypothesis.

The key of the tournament is: for each pair of candidates $(h_i, h_j)$, a held-out set is used to estimate $\text{err}(h_i) - \text{err}(h_j)$, and the candidate that wins the majority of comparisons is selected.

**Core of the refined analysis**:
- When $\tau$ is large, the error concentration of a single ERM solution is good, making it easy for the tournament to select a strong candidate.
- When $\tau \approx d/m$ (the small error regime), a more refined treatment is required:
    - A lower bound on the fraction of "good" ERM solutions in the small error region is exploited.
    - Boosting-like parameter choices ensure that at least one group yields a sufficiently good ERM solution.
    - The comparison error in the tournament is on the order of $O(\sqrt{d/(m \cdot T)})$.

### Loss & Training

The 0-1 loss is used for binary classification:

$$\text{err}(h) = \Pr_{(x,y) \sim \mathcal{D}}[h(x) \neq y]$$

ERM minimizes the empirical risk: $\hat{h}_{\text{ERM}} = \arg\min_{h \in \mathcal{H}} \frac{1}{|S|} \sum_{(x,y) \in S} \mathbf{1}[h(x) \neq y]$

## Key Experimental Results

This is a purely theoretical work; numerical simulations are used to verify the tightness of the theoretical bounds.

### Main Results

**Error upper bound vs. lower bound comparison ($\delta = 0.05$):**

| VC dim $d$ | Sample size $m$ | $\tau$ | Ours | Prev. SOTA (FOCS'24) | ERM bound | Gap ratio |
|-----------|----------------|--------|------|----------------------|-----------|-----------|
| 10 | 100 | 0.1 | 0.247 | 0.231 | 0.312 | 1.07× |
| 10 | 100 | 0.01 | 0.132 | 0.118 | 0.215 | 1.12× |
| 10 | 1000 | 0.1 | 0.138 | 0.131 | 0.156 | 1.05× |
| 10 | 1000 | 0.01 | 0.035 | 0.031 | 0.058 | 1.13× |
| 50 | 500 | 0.1 | 0.295 | 0.278 | 0.368 | 1.06× |
| 50 | 500 | 0.01 | 0.165 | 0.148 | 0.252 | 1.11× |

> Gap ratio = Ours / FOCS'24 lower bound

### Ablation Study

**Relationship between the constant $c$ and the number of groups $T$:**

| Number of groups $T$ | Provable constant $c$ | Computational overhead |
|---------------------|-----------------------|------------------------|
| 2 | 3.5 | 2× ERM |
| 5 | 2.8 | 5× ERM |
| 10 | 2.3 | 10× ERM |
| 20 | 2.1 | 20× ERM |
| 50 | 2.05 | 50× ERM |

> A larger number of groups yields a smaller constant, but the computational cost grows linearly.

### Key Findings

1. **Matching upper and lower bounds in the small error regime**: when $\tau \approx d/m$, the upper bound of this paper matches the FOCS'24 lower bound up to constant factors, resolving the open problem posed by Hanneke et al.
2. **The constant $c = 2.1$ leaves room for improvement**: the ideal case is $c = 1$, which would fully resolve the complexity of agnostic learning.
3. **Computational efficiency**: the learner requires only multiple calls to an ERM oracle plus polynomial-time post-processing.
4. **ERM suboptimality is more pronounced at small $\tau$**: the gap between the ERM bound and the optimal bound in the table widens as $\tau$ decreases.

## Highlights & Insights

- **Elegantly bridges realizable and agnostic learning**: by introducing the parameter $\tau$, the paper unifies sample complexity across both settings.
- **Clean algorithmic design**: based on ERM aggregation rather than a bespoke algorithm, yielding broad applicability.
- **Advances a fundamental problem in learning theory**: the precise complexity of PAC learning remains incompletely resolved, and this paper constitutes an important step forward.
- **Poses a clear open problem**: can $c$ be reduced to 1?

## Limitations & Future Work

- The constant $c = 2.1$ remains above the optimal $c = 1$.
- The number of groups $T$ governs a tradeoff between the constant and computational efficiency; theoretically, $c \to 2$ as $T \to \infty$.
- Only binary classification is considered; extensions to multiclass classification and regression remain unexplored.
- In practice, $\tau$ is typically unknown; how to adaptively tune algorithm parameters is an open problem.

## Related Work & Insights

- **Hanneke, Larsen, Zhivotovskiy (FOCS 2024)**: established $\tau$-parameterized lower bounds for agnostic learning.
- **Classical PAC learning**: Vapnik–Chervonenkis theory, Fundamental Theorem of Statistical Learning.
- **Optimality of ERM**: classical discussions in Shalev-Shwartz & Ben-David's textbook.

This paper demonstrates that even in the most foundational problems of learning theory, fine-grained structure remains to be discovered.

## Rating

| Dimension | Score (1–10) |
|-----------|-------------|
| Novelty | 8 |
| Theoretical Depth | 9 |
| Experimental Thoroughness | 5 |
| Writing Quality | 8 |
| Value | 4 |
| Overall Recommendation | 7.5 |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Revisiting Agnostic Boosting](revisiting_agnostic_boosting.md)
- [\[NeurIPS 2025\] Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians](recurrent_self-attention_dynamics_an_energy-agnostic_perspective_from_jacobians.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)
- [\[ICCV 2025\] From Easy to Hard: Progressive Active Learning Framework for Infrared Small Target Detection with Single Point Supervision](../../ICCV2025/others/from_easy_to_hard_progressive_active_learning_framework_for_infrared_small_targe.md)
- [\[NeurIPS 2025\] Product Distribution Learning with Imperfect Advice](product_distribution_learning_with_imperfect_advice.md)

<!-- RELATED:END -->
