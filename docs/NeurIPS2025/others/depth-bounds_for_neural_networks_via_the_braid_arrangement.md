---
title: >-
  [Paper Note] Depth-Bounds for Neural Networks via the Braid Arrangement
description: >-
  [NeurIPS 2025][ReLU network depth lower bounds] This paper proves that, under $\mathcal{B}_d^0$-conforming constraints, exactly representing $\max\{0, x_1, \ldots, x_d\}$ with a ReLU network requires $\Omega(\log \log d)$ layers—the first non-constant depth lower bound without weight restrictions. It also shows that rank-(3,2) maxout networks can compute the maximum of 7 values, demonstrating that the standard upper bound is not tight.
tags:
  - "NeurIPS 2025"
  - "ReLU network depth lower bounds"
  - "piecewise linear functions"
  - "braid arrangement"
  - "maxout networks"
  - "combinatorial proof"
date: 2026-05-08
content_hash: a64d44f6281e2c97
---

# Depth-Bounds for Neural Networks via the Braid Arrangement

**Conference**: NeurIPS 2025
**arXiv**: [2502.09324](https://arxiv.org/abs/2502.09324)  
**Code**: None  
**Area**: Others (Deep Learning Theory / Expressivity Analysis)
**Keywords**: ReLU network depth lower bounds, piecewise linear functions, braid arrangement, maxout networks, combinatorial proof

## TL;DR
This paper proves that, under $\mathcal{B}_d^0$-conforming constraints, exactly representing $\max\{0, x_1, \ldots, x_d\}$ with a ReLU network requires $\Omega(\log \log d)$ layers—the first non-constant depth lower bound without weight restrictions. It also shows that rank-(3,2) maxout networks can compute the maximum of 7 values, demonstrating that the standard upper bound is not tight.

## Background & Motivation

**Background**: ReLU networks can exactly represent all continuous piecewise linear (CPWL) functions. Arora et al. (2018) showed that $\lceil \log_2(d+1) \rceil$ layers suffice. Bakaev et al. (2025) improved this to $\lceil \log_3(d-1) \rceil + 1$ layers. However, depth **lower bounds** remain largely unexplored.

**Core Open Problem**: What is the minimum number of layers required to exactly represent all CPWL functions on $\mathbb{R}^d$? The best unconditional lower bound currently stands at only 2.

**Key Reduction**: By Wang & Sun (2005), any CPWL function can be expressed in terms of $\max$ operations—reducing the problem to: how deep a network is needed to compute the maximum of $d$ values?

**Research Approach**: The analysis restricts breakpoints to lie on the braid arrangement, i.e., $\mathcal{B}_d^0$-conforming networks. While not fully general, this restriction provides a tractable analytical framework.

## Method

### Core Concepts

**Braid Arrangement** $\mathcal{B}_d$: The hyperplane arrangement defined by $x_i = x_j$ for $1 \leq i < j \leq d$. $\mathcal{B}_d^0$ additionally includes $x_i = 0$.

**$\mathcal{B}_d^0$-conforming Networks**: All neuron breakpoints lie on hyperplanes of the braid arrangement. The standard binary-tree computation of max (taking pairwise maxima layer by layer) is $\mathcal{B}_d^0$-conforming.

**Key Isomorphism**: The space $\mathcal{V}_{\mathcal{B}_d}$ of $\mathcal{B}_d$-compatible CPWL functions is isomorphic to the space of set functions $\mathcal{F}_d = \mathbb{R}^{2^{[d]}}$ (dimension $2^d$) via the map $\Phi(f)(S) = f(\mathbb{1}_S)$.

**Subspace Hierarchy**: $\mathcal{V}_{\mathcal{B}_d}(k) = \text{span}\{\sigma_M \mid M \subseteq [d], |M| \leq k\}$, where $\sigma_M(x) = \max_{i \in M} x_i$. A strict inclusion chain holds:

$$\mathcal{V}_{\mathcal{B}_d}(0) \subsetneq \mathcal{V}_{\mathcal{B}_d}(1) \subsetneq \cdots \subsetneq \mathcal{V}_{\mathcal{B}_d}(d) = \mathcal{V}_{\mathcal{B}_d}$$

### Main Theorem: Double-Logarithmic Lower Bound

**Theorem 4.8**: For an $\ell$-layer rank-2 maxout network, $\mathcal{M}_{\mathcal{B}_d}^2(\ell) \subseteq \mathcal{V}_{\mathcal{B}_d}(2^{2^\ell - 1})$.

**Corollary 4.9**: $\max\{0, x_1, \ldots, x_{2^{2^\ell-1}}\}$ cannot be represented by an $\ell$-layer $\mathcal{B}_d^0$-conforming ReLU network.

This establishes a depth lower bound of $\Omega(\log \log d)$.

### Proof Mechanism

Define the operation $\mathcal{A}(U)$: the new subspace obtained by applying one layer of rank-2 maxout to functions in a subspace $U$.

**Key Recurrence** (Proposition 4.7): $\mathcal{A}(\mathcal{F}_{\mathcal{L}}(k)) \subseteq \mathcal{F}_{\mathcal{L}}(k^2 + k)$

That is, one maxout layer can raise the subspace level from at most $k$ to at most $k^2 + k$. Iterating:
- 1 layer: $k = 2$
- 2 layers: $k = 6$
- 3 layers: $k = 42$
- $\ell$ layers: $k = 2^{2^\ell - 1}$ (double-exponential growth)

Therefore, representing $\sigma_{[d]}$ (the max of $d$ values) requires $d \leq 2^{2^\ell - 1}$, i.e., $\ell \geq \Omega(\log \log d)$.

Key technical ingredients:
1. The isomorphism $\Phi$ translates the problem into analysis over set functions.
2. The **conforming condition** is reformulated as: $F \in \mathcal{C}_{\mathcal{L}}$ (i.e., $F(S)$ and $F(T)$ cannot have opposite signs when $S \subseteq T$).
3. **Lemma 4.5**: "Pushing down supports"—positive and negative supports can be pushed to lower levels.
4. Decomposition and induction over sublattices of the Boolean lattice.

### Combinatorial Proof for $d=4$

**Theorem 5.2**: $\mathcal{M}_{\mathcal{B}_d}^2(2) = \mathcal{V}_{\mathcal{B}_d}(4)$.

That is, a two-layer $\mathcal{B}_d^0$-conforming ReLU network can exactly represent $\max\{0, x_1, \ldots, x_4\}$. Previously, this was only verified computationally via MIP; this paper provides the first purely combinatorial proof.

### Unexpected Expressivity of Maxout

**Theorem 6.2**: $\max\{0, x_1, \ldots, x_6\}$ can be computed by a rank-(3,2) maxout network.

This shows that the standard upper bound $\prod r_i = 3 \times 2 = 6$ is not tight—a rank-(3,2) network can represent the max of 7 values (one more than expected), contradicting naive intuition.

Construction: Two functions $f_1, f_2 \in \mathcal{V}_{\mathcal{B}_7}(3)$ are identified such that $\max\{f_1, f_2\} \in \mathcal{V}_{\mathcal{B}_7}(7) \setminus \mathcal{V}_{\mathcal{B}_7}(6)$.

## Key Experimental Results

This is a purely theoretical work with no numerical experiments. Core results are summarized below:

| Result | Content |
|--------|---------|
| Depth lower bound | $\Omega(\log \log d)$ (under $\mathcal{B}_d^0$-conforming) |
| Prev. SOTA | Constant 2 |
| Upper bound | $\lceil \log_3(d-1) \rceil + 1$ (Bakaev et al.) |
| Exact result for $d=4$ | 2 layers $\Leftrightarrow$ $\mathcal{V}_{\mathcal{B}_d}(4)$ |
| Maxout surprise | rank-(3,2) can compute max of 7 values (standard bound: 6) |

### Comparison with Existing Lower Bounds

| Method | Restriction | Lower Bound |
|--------|------------|-------------|
| Averkov et al. (2025) | N-ary fractional weights | $\Omega(\log d / \log \log N)$ |
| Haase et al. (2023) | Integer weights | $\lceil \log_2(d+1) \rceil$ |
| Bakaev et al. (2025a) | Non-negative weights | Dedicated lower bound |
| **Ours** | **Breakpoints on braid arrangement** | $\Omega(\log \log d)$ |

This work is the **only** non-constant depth lower bound that imposes no restriction on weights.

## Highlights & Insights
- **Although $\log \log d$ grows slowly, it breaks the barrier of constant lower bounds**—representing the first substantial progress in this direction in several years.
- The proof methodology is elegant: a geometric problem (CPWL functions on polyhedral fans) is fully reduced to a combinatorial one (set functions on Boolean lattices).
- The combinatorial proof for $d=4$ supersedes the MIP-based computational verification of Hertrich et al. (2023), providing deeper structural insight.
- **The unexpected expressivity of maxout networks** has independent significance: it shows that the expressive power of higher-order activation functions cannot be characterized by simple product formulas.
- The approach is potentially generalizable to other fan structures, offering tools for understanding depth lower bounds for general networks.

## Limitations & Future Work
- The $\mathcal{B}_d^0$-conforming constraint is a **genuine restriction**: Bakaev et al. (2025b) have shown that general (non-conforming) 2-layer networks can compute the max of 5 values.
- The $\log \log d$ growth is extremely slow (10 layers are not required until $d = 2^{2^{10}-1}$), leaving a large gap from the expected $\Theta(\log d)$ upper bound.
- As a purely theoretical work, its practical implications for architecture design are limited.
- The proof techniques rely heavily on the structure of the braid arrangement and do not directly extend to general breakpoint configurations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First non-constant depth lower bound without weight restrictions; of major academic significance.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work with no numerical experiments (compensated by constructive proofs).
- Writing Quality: ⭐⭐⭐⭐ Clear structure; illustrations of Boolean lattice decompositions are helpful for comprehension.
- Value: ⭐⭐⭐⭐ Advances the core question of depth vs. expressivity in deep learning theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Revisiting Bi-Linear State Transitions in Recurrent Neural Networks](revisiting_bi-linear_state_transitions_in_recurrent_neural_networks.md)
- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)
- [\[NeurIPS 2025\] Tight Bounds On the Distortion of Randomized and Deterministic Distributed Voting](tight_bounds_on_the_distortion_of_randomized_and_deterministic_distributed_votin.md)

</div>

<!-- RELATED:END -->
