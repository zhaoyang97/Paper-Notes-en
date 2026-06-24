---
title: >-
  [Paper Note] Reconstruction and Secrecy under Approximate Distance Queries
description: >-
  [NeurIPS 2025 Spotlight][AI Safety][Reconstruction attacks] Under the approximate distance query model, this paper studies the reconstruction game from a learning-theoretic perspective, proves a geometric characterization of the optimal reconstruction error as the Chebyshev radius, and provides a complete classification of pseudo-finiteness for Euclidean convex spaces.
tags:
  - "NeurIPS 2025 Spotlight"
  - "AI Safety"
  - "Reconstruction attacks"
  - "privacy protection"
  - "approximate distance queries"
  - "metric spaces"
  - "Chebyshev radius"
  - "pseudo-finite spaces"
date: 2026-05-08
content_hash: 5e85e9f481a696d9
---

# Reconstruction and Secrecy under Approximate Distance Queries

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2511.06461](https://arxiv.org/abs/2511.06461)  
**Authors**: Shay Moran (Technion & Google Research), Elizaveta Nesterova (Technion)
**Code**: Not released  
**Area**: AI Security
**Keywords**: Reconstruction attacks, privacy protection, approximate distance queries, metric spaces, Chebyshev radius, pseudo-finite spaces

## TL;DR

Under the approximate distance query model, this paper studies the reconstruction game from a learning-theoretic perspective, proves a geometric characterization of the optimal reconstruction error as the Chebyshev radius, and provides a complete classification of pseudo-finiteness for Euclidean convex spaces.

## Background & Motivation

### State of the Field
Consider a fundamental localization problem: identifying an unknown target point in a metric space via approximate distance queries. In each round, the reconstructor selects a reference point and receives a noisy distance from that point to the target. This setting arises broadly in GPS localization, sensor networks, and privacy-aware data access.

The model encompasses two opposing perspectives:
- **Reconstructor's perspective**: seeks to accurately recover unknown information (e.g., navigation, search-and-rescue, remote sensing)
- **Responder's perspective**: seeks to limit information leakage (e.g., privacy protection, security)

### Limitations of Prior Work
- Dinur & Nissim (2003) pioneered the study of reconstruction attacks under counting queries, but restricted their analysis to counting queries on Boolean hypercubes
- The metric dimension literature focuses primarily on noise-free, static settings on finite graphs
- No prior work provides a **precise geometric characterization** of optimal reconstruction error in **general metric spaces** under **noisy** conditions
- A systematic theoretical framework for the **convergence rate** of reconstruction error with respect to query count is lacking

### Root Cause
The goal is to establish a unified, learning-theoretic framework that precisely characterizes the limiting optimal error in the reconstruction game, addressing two core questions:
1. What is the optimal error under unlimited queries? (asymptotic behavior)
2. Can finite queries achieve optimality? (pseudo-finiteness classification)

## Method

### Reconstruction Game Formulation
- **Setting**: In metric space $(X, \mathrm{dist}_X)$, the reconstructor selects query point $q_t$ each round and receives a noisy distance $\hat{d}_t$
- **Noise model**: $\hat{d}_t =_{\epsilon,\delta} \mathrm{dist}_X(q_t, x^\star)$, where $\epsilon \geq 0$ controls multiplicative error and $\delta \geq 0$ controls additive error
- **Notation**: $x =_{\epsilon,\delta} y$ if and only if $x \leq (1+\epsilon)y + \delta$ and $y \leq (1+\epsilon)x + \delta$
- **Feasible region**: After each interaction, the set $\Phi_T \subseteq X$ of points consistent with all query-response pairs
- **Objective**: The reconstructor minimizes $\mathrm{dist}_X(\hat{x}_T, x^\star)$; the responder maximizes it

The paper adopts the **a posteriori responder** model, in which the responder selects the secret point after the reconstructor commits to a guess, making the problem maximally difficult for the reconstructor.

### Core Theoretical Contribution 1: Geometric Characterization of Optimal Reconstruction Error (Theorem 2)

**Chebyshev radius**: For a subset $S \subseteq X$, the Chebyshev radius is $r(S) = \inf_{x \in X} \sup_{y \in S} \mathrm{dist}_X(x,y)$, i.e., the radius of the smallest enclosing ball of $S$.

**Diameter–radius profile**: $\mathtt{e}_X(\alpha) = \sup_{S: \mathrm{diam}(S) \leq \alpha} r(S)$, characterizing the worst-case Chebyshev radius over sets with diameter at most $\alpha$.

**Main theorem**: For all totally bounded metric spaces $X$,

$$\mathrm{OPT}_X(\epsilon, \delta) = \mathtt{e}_X\big((2+\epsilon)\delta\big)$$

and if $(2+\epsilon)\delta$ is achievable in the space, then

$$\frac{1}{2}(2+\epsilon)\delta \leq \mathrm{OPT}_X(\epsilon,\delta) \leq (2+\epsilon)\delta$$

**Proof sketch**:
- **Upper bound**: Uses total boundedness to construct a finite $\alpha$-cover as the query set, shows that the feasible region diameter approaches $(2+\epsilon)\delta$, and invokes hyperspace theory to establish right-continuity of $\mathtt{e}_X$
- **Lower bound**: Constructs a responder strategy maintaining an extremal set $S_m$ with diameter at most $(2+\epsilon)\delta$ inside the feasible region, using the triangle inequality and noise parameters to ensure consistency

### Core Theoretical Contribution 2: Pseudo-Finiteness Classification (Theorem 6)

**Definition**: A metric space $X$ is $(\epsilon,\delta)$-pseudo-finite if there exists a finite $T$ such that $\mathrm{OPT}_X(T,\epsilon,\delta) = \mathrm{OPT}_X(\epsilon,\delta)$, i.e., optimality is achieved with finitely many queries.

**Main theorem**: For bounded convex sets $X \subset \mathbb{R}^n$, $X$ is $(\epsilon,\delta)$-pseudo-finite if and only if $\dim(X)=1$ and $\epsilon=0$.

**Key techniques**:
- The responder maintains the $\alpha$-neighborhood of a regular simplex $\Delta$ within the feasible region, keeping the Chebyshev radius strictly above $\mathrm{OPT}(\epsilon,\delta) + \alpha_T$
- **Multiplicative noise** ($\epsilon > 0$): Translating the simplex away from the query point guarantees a surviving neighborhood $\alpha^\star(\Delta', q) \geq \alpha_{t+1}$
- **Pure additive noise** ($\epsilon = 0$): Translation alone is insufficient to enlarge the surviving neighborhood; **rotation** of the simplex is required. In dimension $n \geq 2$, rotation can sustain the surviving neighborhood indefinitely; in dimension $n = 1$ there is no non-trivial rotation, so one-dimensional intervals are indeed pseudo-finite
- Convergence rate lower bounds: exponential decay for $\epsilon \neq 0$, doubly exponential decay for $\epsilon = 0$

### Feasible-Region Calculus
For a given set $S$ and query $q$, the consistency window $[r_q^{\min}(S), r_q^{\max}(S)]$ is defined as:
- $r_q^{\min}(S) = \frac{\sup_{s \in S} \mathrm{dist}(s,q) - \delta}{1+\epsilon}$ (the farthest point lies exactly on the outer boundary)
- $r_q^{\max}(S) = (1+\epsilon) \inf_{s \in S} \mathrm{dist}(s,q) + \delta$ (the nearest point lies exactly on the inner boundary)

A response exists that keeps $S$ entirely within the feasible region if and only if $r_q^{\min}(S) \leq r_q^{\max}(S)$.

## Key Experimental Results

### Result 1: Optimal Reconstruction Error Across Metric Spaces

| Metric space $X$ | $\mathtt{e}_X(\alpha)$ | $\mathrm{OPT}_X(\epsilon,\delta)$ | Pseudo-finite? |
|---|---|---|---|
| $\mathbb{R}^n$, $\ell_2$ norm | $\sqrt{\frac{n}{2(n+1)}} \cdot \alpha$ | $\sqrt{\frac{n}{2(n+1)}} \cdot (2+\epsilon)\delta$ | Yes iff $n=1, \epsilon=0$ |
| $[0,1]$, Euclidean distance | $\alpha$ (small $\alpha$) | $(2+\epsilon)\delta$ (small $\delta$) | Yes iff $\epsilon=0$ |
| $\{0,1\}^n$, Hamming distance | — | Equivalent to Dinur–Nissim counting queries | Finite space, always pseudo-finite |
| $\{0,1\}^{\mathbb{N}}$, ultrametric | — | — | Not $(0,0)$-pseudo-finite |
| $\mathbb{N}$, discrete metric | — | $\mathrm{OPT} = 1$ (diameter) | Always pseudo-finite (trivially) |

### Result 2: Complete Pseudo-Finiteness Classification (Euclidean Convex Spaces)

| Condition | $\epsilon = 0$ | $\epsilon > 0$ |
|---|---|---|
| $\dim(X) = 1$ (interval) | **Pseudo-finite** — querying the endpoints suffices to restrict the feasible region to an interval of length $2\delta$ | **Not pseudo-finite** — the responder can employ a binary-search-like strategy to keep a feasible region of length $> (2+\epsilon)\delta$ indefinitely |
| $\dim(X) \geq 2$ (higher-dimensional convex set) | **Not pseudo-finite** — rotating a regular simplex maintains reconstruction error strictly above optimal; convergence is doubly exponential | **Not pseudo-finite** — translating the simplex away from query points sustains the surviving neighborhood; convergence is exponential |

Convergence rate summary:
- $\epsilon > 0$: $\mathrm{OPT}_X(T,\epsilon,\delta) - \mathrm{OPT}_X(\epsilon,\delta) \geq \Omega(e^{-cT})$
- $\epsilon = 0, \dim(X) \geq 2$: $\mathrm{OPT}_X(T,0,\delta) - \mathrm{OPT}_X(0,\delta) \geq \Omega(e^{-e^{cT}})$ (doubly exponential lower bound)
- $\epsilon = 0, \delta = 0$ (exact queries): $\mathrm{OPT}_X(T,0,0) = 0$ for $T \geq n+1$ (optimality achieved in finite queries)

## Highlights & Insights

- **Elegant geometric characterization**: The optimal reconstruction error is precisely identified with the Chebyshev radius, a classical geometric quantity, via a clean formula that applies to all totally bounded metric spaces
- **Complete pseudo-finiteness dichotomy**: A full classification is given for Euclidean convex spaces in terms of dimension and noise parameters — finite queries suffice only in the one-dimensional, pure additive noise case; all other cases require infinitely many queries
- **Deep connection to privacy theory**: The counting-query reconstruction attack of Dinur–Nissim is subsumed as a special case of this framework; distance queries on the Hamming cube are equivalent to counting queries up to a factor of two, providing a new geometric perspective for privacy analysis
- **Technically refined proofs**: The pseudo-finiteness lower bound requires a unified analysis over all query types (good/bad), constructing the responder strategy via translations and rotations of a regular simplex, with careful geometric arguments throughout

## Limitations & Future Work

- **Unmatched convergence rate bounds**: Whether the exponential/doubly exponential lower bounds are tight for $\delta > 0$ remains open; the upper and lower bounds match only in the pure multiplicative case $\delta = 0$
- **Restricted to Euclidean convex spaces**: The pseudo-finiteness classification applies only to convex subsets of $\mathbb{R}^n$; classification for non-convex and general metric spaces remains an open problem
- **Unresolved open questions**: In totally bounded spaces, does "finite" imply "pseudo-finite for all $(\epsilon,\delta)$"? (Open Question 11)
- **No algorithmic implementation or numerical experiments**: The paper is purely theoretical, with no concrete reconstruction algorithms or numerical validation
- **Deterministic reconstructor assumption**: Main results are stated for deterministic reconstructors; extension to randomized settings is claimed but not developed

## Related Work & Insights

- **Dinur & Nissim (2003)**: Pioneered the study of reconstruction attacks under counting queries; this paper subsumes their work as the special case of Hamming distance queries and extends the framework to arbitrary metric spaces
- **Metric dimension literature (Harary 1975, Slater 1975, Seager 2013)**: Studies exact localization on noise-free graphs, corresponding to the $\epsilon=\delta=0$ special case of this model; this paper admits noise and studies convergence rates
- **Differential privacy (DMNS06)**: Focuses on limiting information leakage; this paper studies the limits of leakage from the reconstructor's perspective — the two are complementary
- **Cohen et al. (2025)**: Proposes the Narcissus Resiliency definitional framework and connects it to Kolmogorov complexity and differential privacy
- **Remote sensing literature (Twomey 1977, Rodgers 2000)**: Reconstructs physical quantities from noisy measurements as inverse problems; this paper offers a complementary learning-theoretic and geometric perspective
- **Learning curve theory**: The decay of excess reconstruction error $\mathrm{OPT}_X(T) - \mathrm{OPT}_X$ is analogous to learning curves in statistical learning theory

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to precisely characterize the optimal reconstruction error in the reconstruction game as the Chebyshev radius; the pseudo-finiteness concept is novel and the classification is complete
- Experimental Thoroughness: ⭐⭐ — Purely theoretical work with no numerical experiments or algorithmic implementations
- Writing Quality: ⭐⭐⭐⭐⭐ — Exceptionally clear exposition, well-motivated contributions, highly readable technical overview (Section 3), and abundant illustrative examples
- Value: ⭐⭐⭐⭐ — Establishes an elegant bridge between reconstruction attacks and geometry, though the purely theoretical nature limits direct applied impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OmniFC: Rethinking Federated Clustering via Lossless and Secure Distance Reconstruction](omnifc_rethinking_federated_clustering_via_lossless_and_secure_distance_reconstr.md)
- [\[AAAI 2026\] Truth, Justice, and Secrecy: Cake Cutting Under Privacy Constraints](../../AAAI2026/ai_safety/truth_justice_and_secrecy_cake_cutting_under_privacy_constraints.md)
- [\[NeurIPS 2025\] Fairness under Competition](fairness_under_competition.md)
- [\[ICCV 2025\] Backdoor Mitigation by Distance-Driven Detoxification](../../ICCV2025/ai_safety/backdoor_mitigation_by_distance-driven_detoxification.md)
- [\[NeurIPS 2025\] Unifying Re-Identification, Attribute Inference, and Data Reconstruction Risks in Differential Privacy](unifying_re-identification_attribute_inference_and_data_reconstruction_risks_in_.md)

</div>

<!-- RELATED:END -->
