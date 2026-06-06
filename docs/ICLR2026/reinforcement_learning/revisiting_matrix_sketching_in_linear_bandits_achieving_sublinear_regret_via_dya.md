---
title: >-
  [Paper Note] Revisiting Matrix Sketching in Linear Bandits: Achieving Sublinear Regret via Dyadic Block Sketching
description: >-
  [ICLR 2026][Reinforcement Learning][Linear Bandits] This paper identifies a fundamental flaw in existing sketch-based linear bandit methods: when the spectrum of the streaming matrix has a heavy tail…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Linear Bandits"
  - "Matrix Sketching"
  - "Frequent Directions"
  - "Multi-Scale Sketching"
  - "Sublinear Regret"
  - "Dyadic Block Sketching"
date: 2026-05-08
content_hash: 1470d84ac607f6fb
---

# Revisiting Matrix Sketching in Linear Bandits: Achieving Sublinear Regret via Dyadic Block Sketching

**Conference**: ICLR 2026
**arXiv**: [2410.10258](https://arxiv.org/abs/2410.10258)  
**Authors**: Dongxie Wen, Hanyan Yin, Xiao Zhang, Peng Zhao, Lijun Zhang, Zhewei Wei (Renmin University of China & Nanjing University)
**Code**: None  
**Area**: Reinforcement Learning / Online Learning / Bandits
**Keywords**: Linear Bandits, Matrix Sketching, Frequent Directions, Multi-Scale Sketching, Sublinear Regret, Dyadic Block Sketching

## TL;DR

This paper identifies a fundamental flaw in existing sketch-based linear bandit methods: when the spectrum of the streaming matrix has a heavy tail, these methods degenerate to linear regret. To address this, the paper proposes the Dyadic Block Sketching (DBS) framework, which dynamically doubles the sketch size to control the global approximation error within a user-specified parameter $\epsilon$. The resulting algorithm guarantees sublinear regret without requiring prior knowledge of the spectral structure of the streaming matrix, and adaptively recovers the computational efficiency of single-scale methods when the spectrum is favorable.

## Background & Motivation

**Background**: Stochastic Linear Bandits (SLB) constitute a core framework in online learning. The classical OFUL algorithm achieves $\widetilde{O}(d\sqrt{T})$ regret via regularized least squares with upper confidence bounds, but incurs $\Omega(d^2)$ per-round update cost. In high-dimensional settings this is computationally prohibitive, motivating the use of matrix sketching to reduce per-round complexity to $O(dl + l^2)$, where $l < d$ denotes the sketch size.

**Limitations of Prior Work**: The regret bounds of sketch-based methods such as SOFUL and CBSCFD depend on the spectral error $\Delta_T$. When the streaming matrix has a heavy-tailed spectrum, a fixed small sketch cannot retain sufficient spectral information, causing $\Delta_T$ to grow rapidly and the regret to degenerate to **linear**—fundamentally undermining the goal of online learning.

**Key Challenge**: The optimal sketch size depends on the spectral structure of the streaming matrix, which is unknown a priori. A sketch that is too small leads to linear regret, while one that is too large sacrifices efficiency. The authors prove that in locally convex arm spaces with geometric constant $q \geq 1/3$, **any** fixed-$l$ SOFUL with $l < d$ must incur linear regret (Observation 1).

**Key Insight**: Drawing on the dyadic decomposition framework from streaming algorithms, the paper approximates the streaming matrix using multiple sketch blocks whose sizes grow geometrically, bounding the global error by a user-specified parameter $\epsilon$ and thereby decoupling performance from unknown spectral properties.

## Method

### Overall Architecture: DBSLinUCB

The method comprises three layers:
1. **Bottom layer — Dyadic Block Sketching (Algorithm 1)**: The data stream is partitioned into blocks, each approximated by an independent matrix sketch (FD/RFD), with sketch sizes doubling across blocks.
2. **Middle layer — Sketch merging**: Leveraging the decomposability property (Lemma 3), all block sketches are merged into a global sketch $S^{(t)}$ and an auxiliary matrix $M^{(t)}$.
3. **Top layer — UCB decision-making**: Arm selection is performed using a multi-scale sketch-based RLS estimator and a multi-scale confidence ellipsoid.

### Key Designs

**1. Dyadic Block Partitioning Strategy**

The initial block sketch size is $l_0$, and each new block doubles to $2l$. The algorithm maintains an active block $\mathcal{B}^\star$ (receiving new rows) and a list of inactive (frozen) blocks $\mathcal{L}$. Two invariants are enforced: (i) the sketch size of an inactive block is at least its rank, or its block size is less than $\epsilon l_0$; (ii) the total number of blocks is at most $\lfloor \log(d/l_0 + 1) \rfloor$. When the rank of the active block would exceed its sketch size and the block size is at least $\epsilon l_0$, the current block is frozen and a new block with doubled sketch size is created. When the block count reaches the upper limit (extreme heavy-tail case), the algorithm falls back to rank-1 exact updates, equivalent to OFUL.

**2. Decomposability (Lemma 3)**

If each block sketch satisfies $\|X_i^\top X_i - S_i^\top S_i\|_2 \leq \epsilon_i \|X_i\|_F^2$, the global error of the merged sketch is bounded by the sum of per-block errors—enabling multi-scale quality guarantees to compose into a global bound.

**3. Global Error Control (Theorem 1)**

$$\|X^\top X - S^\top S\|_2 \leq 2\epsilon$$

This bound is independent of the spectral properties of the streaming matrix. The block count $B = \lceil \min\{\log(k/l_0), \|X\|_F^2 / (\epsilon l_0)\} \rceil$ adapts to the data: few blocks and small sketches for low-rank inputs, more blocks for heavy-tailed inputs until falling back to exact updates.

**4. Multi-Scale Confidence Ellipsoid (Theorem 2)**

$$\hat{\beta}_t(\delta) \lesssim R\sqrt{d \ln(1 + \epsilon/\lambda) + 2l_{B_t}} \cdot \sqrt{1 + \epsilon/\lambda} + \frac{H(\lambda + \epsilon)}{\sqrt{\lambda}}$$

The ellipsoid radius depends only on $\epsilon$ and the current sketch size $l_{B_t}$, rather than the uncontrollable quantity $\Delta_T$.

### Theoretical Guarantees

**Regret bound of DBSLinUCB-FD (Theorem 3)**:

$$\text{Regret}_T = \widetilde{O}\left(\left(1 + \frac{\epsilon}{\lambda}\right)^{3/2} \cdot (d + l_{B_T}) \cdot \sqrt{T}\right)$$

Setting $\epsilon = O(1)$ yields $\widetilde{O}(\sqrt{T})$, matching the order of OFUL.

**Regret bound of DBSLinUCB-RFD (Theorem 4)**:

$$\text{Regret}_T = \widetilde{O}\left(\left(1 + \frac{\epsilon}{\lambda}\right)^{1/2} \cdot \sqrt{l_{B_T} T} + \sqrt{d l_{B_T} T}\right)$$

The exponent on $\epsilon$ is reduced from $3/2$ to $1/2$, and $d$ is decoupled from $\epsilon$, benefiting from the positive-definiteness monotonicity and well-conditioned structure of RFD. Setting $\epsilon = O(T^{(2\gamma-1)/3})$ yields any regret of order $O(T^\gamma)$ for $\gamma \in [0.5, 1)$.

## Key Experimental Results

### Experiment 1: Linear Regret Verification on Synthetic Data

Setting: $d=500$, 100 arms, Gaussian distribution $\mathcal{N}(0, I_d)$, sketch sizes $l \in \{50, 450\}$.

| Algorithm | Sketch Size | Regret Trend | Spectral Error $\log(\Delta_T)/\log t$ |
|-----------|------------|-------------|----------------------------------------|
| OFUL | No sketch | Sublinear (optimal baseline) | — |
| SOFUL | $l=450$ | Sublinear | $< 1/3$ ✔ |
| SOFUL | $l=50$ | **Near-linear** ❌ | $> 1/3$ ❌ exceeds threshold |
| CBSCFD | $l=450$ | Sublinear | $< 1/3$ ✔ |
| CBSCFD | $l=50$ | **Near-linear** ❌ | $> 1/3$ ❌ exceeds threshold |
| DBSLinUCB-FD | $l_0=50, \epsilon=8$ | **Sublinear** ✔ | Adaptively controlled |
| DBSLinUCB-RFD | $l_0=50, \epsilon=8$ | **Sublinear** ✔ | Adaptively controlled |

Key finding: At $l=50$, the spectral error of SOFUL/CBSCFD exceeds the $1/3$ threshold, confirming linear regret as predicted by Observation 1. DBSLinUCB maintains sublinear regret with the same initial sketch size.

### Experiment 2: MNIST Real Data + Pareto Frontier

Setting: $d=784$, $M=10$ classes, 60,000 samples, 2,000 rounds of online classification.

| Method | Configuration | Regret (2000 rounds) | Time Saving | Space Saving |
|--------|--------------|---------------------|-------------|--------------|
| OFUL | No sketch | ~200 (optimal) | 0% (baseline) | 0% (baseline) |
| SOFUL | $l=600$ | ~250 | ~30% | ~25% |
| SOFUL | $l=50$ | >500 ❌ | ~85% | ~90% |
| DBSLinUCB-FD | $\epsilon=4, l_0=50$ | ~220 | ~60% | ~80% |
| DBSLinUCB-RFD | $\epsilon=4, l_0=50$ | ~210 | ~60% | ~80% |
| DBSLinUCB-FD | $\epsilon=25, l_0=50$ | ~300 | ~80% | ~90% |

Key findings: (1) DBSLinUCB dominates SOFUL on the Pareto frontier (regret vs. time/space), reducing regret by up to 40% or reducing time by 60% and space by 80% at equivalent regret. (2) Regret remains below 300, whereas SOFUL with a small sketch exceeds 500. (3) With very small $\epsilon$, performance converges across different values of $l_0$ (constrained by Invariant 2 to rely more on exact updates).

## Highlights & Insights

- **Paradigm shift from "guessing sketch size" to "specifying error tolerance"**: Users directly control the precision $\epsilon$ rather than guessing unknown spectral properties, elegantly transferring problem complexity from parameter selection to adaptive computation.
- **Graceful degradation at both extremes**: In the best case, the method recovers the optimal $O(dk)$ sketch complexity; in the worst case, it degrades to $O(d^2)$ OFUL—both endpoints are known to be optimal, with a smooth transition in between.
- **Framework generality**: The approach is not tied to any specific sketching method; any sketch satisfying a covariance error guarantee (FD/RFD/random projection) can be plugged in directly, yielding a modular and extensible design.
- **Tight theory-experiment correspondence**: The spectral threshold condition from Observation 1 is precisely reproduced experimentally, and the Pareto frontier intuitively illustrates the efficiency-accuracy tradeoff.

## Limitations & Future Work

- **Manual tuning of $\epsilon$ remains necessary**: The optimal $\epsilon$ depends on the problem instance and $T$; fully adaptive $\epsilon$ selection is an open problem.
- **No speedup in the heavy-tail regime**: When $k=d$, the method degrades to $O(d^2)$, matching OFUL—an information-theoretic inevitability, though finer practical solutions may exist.
- **Limited experimental scale**: $d=784$ (MNIST) is relatively small; further validation in high-dimensional settings such as recommendation systems ($d \geq 10{,}000$) is needed.
- **Restricted to stationary stochastic settings**: Block-splitting strategies for non-stationary environments or adversarial noise require redesign.
- **Frobenius norm bound is not tightest possible**: The authors note that the adaptive spectral tail bounds of FD could be leveraged to improve block allocation, representing a concrete direction for theoretical improvement.

## Related Work & Insights

- **vs. SOFUL (Kuzborskij et al., 2019)**: Uses a fixed FD sketch; regret depends on $\Delta_T$ and may be linear. DBSLinUCB decouples the error via multi-scale sketching and recovers equivalent efficiency in low-rank cases.
- **vs. CBSCFD (Chen et al., 2020)**: Replaces FD with RFD to reduce the order of $\Delta_T$, but the fundamental issue of a fixed sketch size remains. DBSLinUCB-RFD combines the advantages of RFD with adaptive sketch sizing.
- **vs. OFUL (Abbasi-Yadkori et al., 2011)**: The exact method with $O(d^2)$ cost. DBSLinUCB accelerates substantially when the spectrum is favorable and degrades to OFUL in the worst case—effectively a computationally adaptive generalization of OFUL.
- **Dyadic framework origin**: Dyadic decomposition in streaming algorithms (Wang et al., 2013; Wei et al., 2016). Adapting this to bandits requires non-trivial additional analysis of confidence ellipsoids and regret bounds.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-scale sketching idea originates in streaming algorithms, but its application to bandits with a sublinear regret guarantee constitutes a non-trivial contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic and MNIST experiments clearly validate the theory, but large-scale high-dimensional real-world datasets are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured and logically coherent, from the identification of the pitfall to the proposed solution; figures and tables are clear and informative.
- Value: ⭐⭐⭐⭐ Addresses a fundamental flaw in sketch-based bandits with a general-purpose framework, though the application domain is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits](online_minimization_of_polarization_and_disagreement_via_low-rank_matrix_bandits.md)
- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](../../NeurIPS2025/reinforcement_learning/generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICML 2026\] Practical and Optimal Algorithm for Linear Contextual Bandits with Rare Parameter Updates](../../ICML2026/reinforcement_learning/practical_and_optimal_algorithm_for_linear_contextual_bandits_with_rare_paramete.md)

</div>

<!-- RELATED:END -->
