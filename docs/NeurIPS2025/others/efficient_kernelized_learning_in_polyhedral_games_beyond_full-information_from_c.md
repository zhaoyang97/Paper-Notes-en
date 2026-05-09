---
title: >-
  [Paper Note] Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games
description: >-
  [NeurIPS 2025][polyhedral games] This paper proposes a kernelization-based framework for designing computationally efficient no-regret learning algorithms for polyhedral games (Colonel Blotto, graphic matroid congestion games, and network congestion games) under partial-information feedback, significantly improving the runtime complexity for learning coarse correlated equilibria (CCE).
tags:
  - NeurIPS 2025
  - polyhedral games
  - kernelization
  - coarse correlated equilibrium
  - Colonel Blotto
  - congestion games
date: 2026-05-08
content_hash: 73123783df7b0dee
---

# Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games

**Conference**: NeurIPS 2025
**arXiv**: [2509.20919](https://arxiv.org/abs/2509.20919)
**Code**: None
**Area**: Other
**Keywords**: polyhedral games, kernelization, coarse correlated equilibrium, Colonel Blotto, congestion games

## TL;DR

This paper proposes a kernelization-based framework for designing computationally efficient no-regret learning algorithms for polyhedral games (Colonel Blotto, graphic matroid congestion games, and network congestion games) under partial-information feedback, significantly improving the runtime complexity for learning coarse correlated equilibria (CCE).

## Background & Motivation

**Polyhedral games** are normal-form games with combinatorial structure and exponentially large action spaces. Each player's action is a $d$-dimensional binary vector (with at most $m$ ones), and the cost function is linear over actions. Canonical examples include:

- **Colonel Blotto games**: distributing $n$ soldiers across $k$ battlefields, with $N \approx n^k$ actions
- **Graphic matroid congestion games**: selecting a spanning tree as a strategy, with $N \approx |E|^{|V|}$
- **Network congestion games**: selecting an $s\to t$ path, with $N \approx |E|^K$

Since $N$ grows exponentially in $m$, classical no-regret algorithms (MWU, FTRL, etc.) incur per-round complexity of $\text{poly}(N)$, rendering them computationally infeasible in these games.

The **full-information setting** (observing the cost of every action each round) has been addressed via kernelization [Farina et al., 2022], but practical settings more commonly involve **partial-information feedback**:
- **Bandit feedback**: only the cost of the chosen action is observed
- **Semi-bandit feedback**: the cost of each component of the chosen action is observed

Existing methods under partial information suffer from prohibitive runtime complexity:
- [ZL22]'s bandit algorithm for Colonel Blotto: $\tilde{O}(n^{18}k^{11}/\varepsilon^2)$
- [LLWZ20]'s CCE learning: runtime dependent on $d^{10}$

**Core Problem**: Can kernelization be extended to partial-information settings to yield no-regret learning dynamics with optimal runtime complexity?

## Method

### Overall Architecture

The proposed kernelization framework addresses three main challenges:
1. **Efficient computation of loss estimators** for policy updates
2. **Efficient sampling** from the MWU distribution
3. **High-probability no-swap-regret guarantees** to ensure convergence to CCE

**Theorem 2.1 (informal)**: If all $|P|$ players employ no-regret learning, the time-averaged joint action $\sigma^* = \frac{1}{T}\sum_t v_1^{(t)} \otimes \cdots \otimes v_{|P|}^{(t)}$ constitutes an approximate CCE.

### Key Designs

**1. Bandit Setting: Kernelized GeometricHedge**

The key innovation lies in the observation that, under bandit feedback, constructing an unbiased combinatorial bandit estimator requires the **second-order moment** of the MWU distribution (rather than the first-order moment sufficient in the full-information setting).

**Theorem 3.1**: The required second-order moments can be efficiently computed via $\Theta(d^2)$ kernel evaluations.

Regret bound: $\tilde{O}(d^{2/3}m^{4/3}T^{2/3})$ (high probability), improving over baselines in dependence on game parameters.

**2. Semi-Bandit Setting: Implicit Exploration Loss Estimator**

The implicit exploration loss estimator of Neu [2015] is adopted and shown to be kernel-compatible with the first-order moments of MWU.

Regret bound: $\tilde{O}(m\sqrt{Td})$ (high probability).

**3. Efficient Sampling**

A general kernelized sampling procedure is proposed, requiring only an additional $\Theta(d)$ kernel evaluations. The core idea is to exploit the combinatorial structure of the games (e.g., generating functions, Matrix-Tree Theorem) to reduce operations over the exponentially large action space to polynomial-time kernel computations.

### Game-Specific Kernelizations

**Colonel Blotto games**:
- Kernelization via generating functions
- Operations performed directly on the $\Theta(nk)$ representation, avoiding the $O(n^2k)$ overhead of DAG-based representations
- Bandit runtime: $\tilde{O}(n^{2+\omega}k^{6+\omega}/\varepsilon^3)$, where $\omega \approx 2.372$ is the matrix multiplication exponent

**Graphic matroid congestion games**:
- Kernel designed via the Matrix-Tree Theorem
- Amortized kernel computation time reduced via rank-1 updates to the LU decomposition of the Laplacian matrix
- Incremental kernelization enables exact sampling

**Network congestion games**:
- First efficient implementation of GeometricHedge for general network congestion games beyond shortest-path problems

### Loss & Training

This is a theoretical work; the core "training strategy" consists of online learning dynamics:
- Each round, an action is sampled from the MWU distribution
- After observing feedback, a loss estimator is constructed
- MWU distribution weights are updated
- After $T$ rounds, the time-averaged strategy converges to an $\varepsilon$-CCE

## Key Experimental Results

### Main Results

This is a purely theoretical paper; results are presented as runtime complexity comparison tables.

**Table 1: Runtime Comparison for Colonel Blotto Games**

| Algorithm | Runtime for $\varepsilon$-CCE | Representation | Feedback |
|:---:|:---:|:---:|:---:|
| [BHK+23] | $\tilde{O}(nk^4/\varepsilon^2)$ | $O(k\log n)$ | Full-info |
| **Ours** | $\tilde{O}(\|P\|nk^3/\varepsilon)$ | $O(nk)$ | Full-info |
| [ZL22] | $\tilde{O}(n^{18}k^{11}/\varepsilon^2)$ | $O(n^2k)$ | Bandit |
| [LE21] | $\tilde{O}(n^4k^5/\varepsilon^3 \cdot \max\{1/\lambda_{\min}, n^2\}^{3/2})$ | $O(n^2k)$ | Bandit |
| **Ours** | $\tilde{O}(n^{2+\omega}k^{6+\omega}/\varepsilon^3)$ | $O(nk)$ | Bandit |
| **Ours** | $\tilde{O}(n^2k^4/\varepsilon^2)$ | $O(nk)$ | Semi-bandit |

**Comparison with [ZL22]**: In the bandit setting, the dependence on $n$ and $k$ improves from $n^{18}k^{11}$ to approximately $n^{4.4}k^{8.4}$, yielding an improvement factor of approximately $n^{13}k^2$.

**Table 2: Runtime Comparison for Graphic Matroid Congestion Games**

| Algorithm | Runtime for $\varepsilon$-CCE | Feedback |
|:---:|:---:|:---:|
| [ZL22] | $\tilde{O}(\|V\|^{29}/\varepsilon^2)$ | Bandit |
| **Ours** | $\tilde{O}(\|E\|^3\|V\|^6(\|V\|^{\omega-1}+\|E\|)/\varepsilon^3)$ | Bandit |
| **Ours** | $\tilde{O}(\|E\|^2\|V\|^{2+\omega}/\varepsilon^2)$ | Semi-bandit |

**Table 3: Runtime Comparison for Network Congestion Games**

| Algorithm | Runtime for $\varepsilon$-CCE | Feedback |
|:---:|:---:|:---:|
| [ZL22] | $\tilde{O}(\|E\|^9K^{10}/\varepsilon^2)$ | Bandit |
| **Ours** | $\tilde{O}(\|E\|^{2+\omega}K^4/\varepsilon^3)$ | Bandit |
| [GLLO07a] | $\tilde{O}(\|E\|^{1+\omega}K^3/\varepsilon^2)$ | Semi-bandit |
| **Ours** | $\tilde{O}(\|E\|^{1+\omega}K^2/\varepsilon^2)$ | Semi-bandit |

### Ablation Study

There are no ablation experiments in the conventional sense; instead, the contribution of each component is demonstrated as follows:
- **Kernelized vs. non-kernelized**: kernelization reduces per-round complexity from $\text{poly}(N)$ to $\text{poly}(d,m)$
- **Second-order vs. first-order moment kernel**: the bandit setting requires computing second-order moments of MWU, incurring $\Theta(d)$ additional kernel evaluations compared to the full-information setting
- **Direct Blotto representation vs. DAG representation**: operating directly on the $\Theta(nk)$ representation yields a more compact alternative to the $\Theta(n^2k)$ DAG representation used in [ZL22]

### Key Findings

1. **Kernelization can be successfully extended to partial-information settings**, directly answering the paper's core question.
2. **Runtime improvements are substantial**: approximately $n^{13}k^2$ improvement over [ZL22] in Colonel Blotto games; a reduction from $|V|^{29}$ to polynomial complexity in graphic matroid games.
3. **Multiple open problems resolved**: the paper answers the open question of [BHK+23] on $1/\varepsilon$ convergence for Colonel Blotto under full information, and the open question of [CBL12] on efficiently implementing GeometricHedge over spanning trees.

## Highlights & Insights

1. **Generality of the unified framework**: the same kernelization paradigm applies to three structurally distinct classes of games—Colonel Blotto, graphic matroid congestion games, and network congestion games.
2. **Second-order moment kernelization is the core technical contribution**: while full-information kernelization requires only first-order moments, this work identifies that the bandit setting requires second-order moments and proves they can be computed efficiently via kernelization.
3. **Compact representations are critical**: operating directly on the game's geometric structure (the $nk$ representation) rather than an indirect DAG representation is a key source of the runtime improvements.
4. **Resolution of multiple open problems**: significantly amplifies the theoretical impact of the work.

## Limitations & Future Work

1. The regret bound in the bandit setting is $T^{2/3}$, falling short of the optimal $\sqrt{T}$ (which is achieved in the semi-bandit setting).
2. The runtime dependence on $\varepsilon$ is $1/\varepsilon^3$ (bandit); whether this can be improved to $1/\varepsilon^2$ remains an open problem.
3. As a purely theoretical work, numerical experiments are absent.
4. The framework requires cost functions to be linear in the actions, and may not apply to games with nonlinear cost functions.
5. The practical efficiency of kernel computations depends on the specific game structure, limiting the degree of generalization.

## Related Work & Insights

- **[Farina et al., 2022]**: Kernelized MWU framework for the full-information setting; the direct precursor of this work.
- **GeometricHedge/ComBand** [Dani et al., 2007]: Classical bandit linear optimization algorithms; the target of kernelization in this paper.
- **[Beaglehole et al., 2023]**: Fast approximate sampling for Colonel Blotto under full information, but incompatible with optimism-based methods.
- **[Zimmert & Lattimore, 2022]**: Continuous-action-space algorithms applied to polyhedral games, with impractical runtimes.
- **Matrix-Tree Theorem** [Tutte, 2001]: Mathematical foundation for kernelization in graphic matroid congestion games.

## Rating

- **Novelty**: ★★★★★ — First extension of kernelization to partial-information settings in polyhedral games, resolving multiple open problems.
- **Technical Depth**: ★★★★★ — Deep integration of online learning, combinatorial optimization, and matrix theory.
- **Experimental Thoroughness**: ★★☆☆☆ — Purely theoretical; no numerical experiments.
- **Writing Quality**: ★★★★☆ — Theoretical exposition is clear; tabular comparisons are easy to parse; mathematical notation is dense.
- **Value**: ★★★☆☆ — Major theoretical contribution; empirical validation in practical game-theoretic settings remains to be conducted.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Evolutionary Prediction Games](evolutionary_prediction_games.md)
- [\[NeurIPS 2025\] Optimism Without Regularization: Constant Regret in Zero-Sum Games](optimism_without_regularization_constant_regret_in_zero-sum_games.md)
- [\[AAAI 2026\] Deviation Dynamics in Cardinal Hedonic Games](../../AAAI2026/others/deviation_dynamics_in_cardinal_hedonic_games.md)
- [\[NeurIPS 2025\] Johnson-Lindenstrauss Lemma Beyond Euclidean Geometry](johnson-lindenstrauss_lemma_beyond_euclidean_geometry.md)
- [\[NeurIPS 2025\] Active Measurement: Efficient Estimation at Scale](active_measurement_efficient_estimation_at_scale.md)

<!-- RELATED:END -->
