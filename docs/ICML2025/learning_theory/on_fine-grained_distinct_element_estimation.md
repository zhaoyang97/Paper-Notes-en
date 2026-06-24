---
title: >-
  [Paper Note] On Fine-Grained Distinct Element Estimation
description: >-
  [ICML2025][Others/Algorithmic Theory][distinct element estimation] Proposes using the **pairwise collisions** $C$ as a fine-grained complexity parameter for the distributed distinct element counting problem, designs a protocol whose communication decreases significantly as $C$ decreases, breaking the prior worst-case lower bound of $\Omega(\alpha/\varepsilon^2)$, and provides matching lower bounds across all parameter regimes.
tags:
  - "ICML2025"
  - "Others/Algorithmic Theory"
  - "distinct element estimation"
  - "distributed computing"
  - "communication complexity"
  - "parameterized complexity"
  - "streaming algorithms"
date: 2026-05-08
content_hash: 91bfcce93bea732d
---

# On Fine-Grained Distinct Element Estimation

**Conference**: ICML2025  
**arXiv**: [2506.22608](https://arxiv.org/abs/2506.22608)  
**Code**: None  
**Area**: Others/Algorithmic Theory  
**Keywords**: distinct element estimation, distributed computing, communication complexity, parameterized complexity, streaming algorithms

## TL;DR

Proposes using the **pairwise collisions** $C$ as a fine-grained complexity parameter for the distributed distinct element counting problem, designs a protocol whose communication decreases significantly as $C$ decreases, breaking the prior worst-case lower bound of $\Omega(\alpha/\varepsilon^2)$, and provides matching lower bounds across all parameter regimes.

## Background & Motivation

**Problem Definition**: Let $\alpha$ servers each hold a subset of the universe $[n]$. The goal is to estimate the number of distinct elements $F_0(S)$ within a $(1+\varepsilon)$-approximation with minimal total communication.

**Known Optimal Bounds**: [KNW10, Bla20] provided an upper bound of $O(\alpha/\varepsilon^2 + \alpha\log n)$, and [WZ14] proved a matching lower bound of $\Omega(\alpha/\varepsilon^2 + \alpha\log n)$, suggesting that the problem had been settled.

**Motivation**: The lower bound construction in [WZ14] requires a constant fraction of the elements to appear on a constant fraction of the servers, which is unrealistic in practical scenarios (e.g., network traffic, recommendation systems, federated learning). Real-world data typically follows a **Zipfian distribution**, where only a small number of elements are highly frequent. Thus, a natural question arises: can this lower bound be bypassed under "real-world" distributions with few collisions?

## Method

### Core Parameterization

Define the **number of pairwise collisions** $C$: for all vectors $v^{(1)}, \ldots, v^{(\alpha)} \in \{0,1\}^n$,

$$C = \left|\{(a,b,i) : 1 \le a < b \le \alpha,\; v_i^{(a)} = v_i^{(b)} = 1\}\right|$$

Let $C = \beta \cdot F_0(S)$, where $\beta \ge 1$ reflects the "collision density." When $\beta \ll \alpha^2$, the collisions are sparse, and existing lower bounds no longer apply.

### Protocol 1: General Protocol (Theorem 1.1)

**Framework**: Classic level-based subsampling.

1. **Constant-factor approximation**: Perform geometrically decreasing subsampling on the universe $[n]$ to obtain $S_0 \supset S_1 \supset \cdots$, where each $S_i$ survives with probability $1/2$. Starting from the sparsest level, each server reports the elements falling into this level to the coordinator until the coordinator observes $\Theta(1)$ distinct elements. The communication cost is $O(\alpha \log n)$.
2. **$(1+\varepsilon)$-approximation**: Follow the same process but stop when the coordinator accumulates $\Theta(1/\varepsilon^2)$ distinct elements. Scale the unbiased estimator using the inverse sampling probability, which guarantees accuracy via Chebyshev's bound.
3. **Collision-parameterized analysis**: Key insight—if $C = \beta \cdot F_0(S)$, each coordinate on average appears in only $\sqrt{\beta}$ servers. Thus, the total number of element identifiers sent by all servers is only $O(\sqrt{\beta}/\varepsilon^2)$ instead of $O(\alpha/\varepsilon^2)$.

**Communication**:

$$O\!\left(\alpha \log n + \frac{\sqrt{\beta}}{\varepsilon^2} \log n\right)$$

### Protocol 2: Further Improvement with Few Collisions (Theorem 1.2)

When $C \le F_0(S)$ (i.e., $\beta < 1$), the protocol utilizes $F_0(S) = F_1(S) - D$, where $D$ is the excess mass (redundancy).

- $F_1(S) = \sum_i \|v^{(i)}\|_1$ can be computed exactly with $O(\alpha \log n)$ bits.
- Estimate the excess mass $D$ through sampling: sample the universe with a sampling rate of $100C / (\varepsilon^2 X^2)$ (where $X$ is a constant-factor approximation). The coordinator observes the frequency redundancy of the sampled elements and performs inverse scaling to obtain an additive $\varepsilon F_0(S)$ approximation of $D$.

**Communication**:

$$\tilde{O}\!\left(\alpha \log n + \max\!\left(\frac{1}{F_0(S)},\, \varepsilon^2\right) \cdot \frac{C}{\varepsilon^2} \log n\right)$$

### Lower Bound Proofs

| Theorem | Applicable Regime | Lower Bound |
|------|---------|------|
| Theorem 1.3 | $C = \Omega(\beta \cdot F_0(S)),\; \beta \ge 1$ | $\Omega(\alpha + \sqrt{\beta}/\varepsilon^2)$ |
| Theorem 1.4 | $C \in [\varepsilon F_0(S),\, F_0(S)]$ | $\Omega(\alpha + C/(\varepsilon^2 F_0(S)))$ |
| Theorem 1.5 | Distributed Duplicate Detection | $\Omega(\alpha s / (C\varepsilon^2))$ |

- **Theorem 1.3**: Embeds the SUM-DISJ lower bound from [WZ14] parametrically, embedding $\alpha$-player instances into $\sqrt{\beta}$ players.
- **Theorem 1.4** (The most difficult technical contribution of this work): Defines a composition problem $\mathsf{GapSet}$, where the outer layer is $\mathsf{GapAnd}$ (a variant of Gap-Hamming) and the inner layer is a multi-party pairwise intersection disjointness problem. An $\Omega(nt)$ communication lower bound is proven via information complexity analysis, which is then reduced to duplicate detection.

### Streaming Algorithm Extensions

Parameterizing $C$ as the number of coordinates with frequency $>1$ (analogous to distributed collisions):

| Model | Space Complexity | Notes |
|------|-----------|------|
| Two-pass | $\tilde{O}(C + 1/\varepsilon)$ | Matches the lower bound $\Omega(C + 1/\varepsilon)$ |
| Single-pass | $\tilde{O}(C/\varepsilon)$ | Breaks $\Omega(1/\varepsilon^2)$ when $C < 1/\varepsilon$ |

Two-pass algorithm: The first pass uses CountSketch to identify high-frequency elements (outliers) and performs multi-scale subsampling on intermediate-frequency elements. The second pass exactly reconstructs their frequencies. Single-pass algorithm: Increases the number of CountSketch buckets to $O(C/\varepsilon)$ and incorporates robust mean estimation techniques.

## Summary of Theoretical Results

| Parameter Regime | Upper Bound | Lower Bound | Status |
|---------|------|------|------|
| $\beta \ge 1$, $F_0 \ge 1/\varepsilon^2$ | $O(\alpha\log n + \sqrt{\beta}\log n/\varepsilon^2)$ | $\Omega(\alpha + \sqrt{\beta}/\varepsilon^2)$ | Tight (up to $\log n$) |
| $\beta \ge 1$, $F_0 < 1/\varepsilon^2$ | $O(\alpha\log n + \sqrt{\beta} F_0 \log n)$ | $\Omega(\alpha + \sqrt{\beta} F_0)$ | Tight (up to $\log n$) |
| $\beta < 1$, $F_0 \ge 1/\varepsilon^2$ | $O(\alpha\log n + \beta F_0 \log n)$ | $\Omega(\alpha + \beta F_0)$ | Tight (up to $\log n$) |
| $\beta < 1$, $F_0 < 1/\varepsilon^2$ | $O(\alpha\log n + \beta\log n/\varepsilon^2)$ | $\Omega(\alpha + \beta/\varepsilon^2)$ | Tight (up to $\log n$) |
| Zipfian ($s>1$) | $\tilde{O}(\alpha\log n)$ | — | Bypasses worst-case lower bound |

**Key Findings**: When the number of collisions is $C = O(\alpha) \cdot F_0(S)$ (as in a Zipfian distribution), the communication cost is only $\tilde{O}(\alpha\log n)$, which is orders of magnitude lower than the worst-case $\Theta(\alpha/\varepsilon^2 + \alpha\log n)$. This is empirically validated on real CAIDA network traffic data.

## Highlights & Insights

- **Parameterized Complexity Perspective**: Reopens a classic, seemingly "closed" distributed statistical estimation problem using the pairwise collision parameter, revealing why worst-case lower bounds fail to reflect practical scenarios.
- **Comprehensive Matching of Upper and Lower Bounds**: Provides tight characterizations (differing only by a $\log n$ factor) across all parameter regimes of $\beta$ and $\varepsilon$. In particular, the composition lower bound in Theorem 1.4 is the most sophisticated technical contribution of this work.
- **Clear Practical Significance**: Zipfian distributions are prevalent in natural language word frequencies, internet traffic, and social network degree distributions. The proposed algorithm can reduce communication overhead by several orders of magnitude in these scenarios.
- **Synergy with Streaming Algorithms**: Generalizes the collision parameterization to streaming models, similarly breaking known streaming space lower bounds in both two-pass and single-pass settings.

## Limitations & Future Work

- **Prior Knowledge of Collisions**: The upper bound protocols (especially Theorem 1.2) require a loose upper bound of $C$ as input. The paper notes that estimating $C$ itself might require high communication (Theorem 1.5), dryly presenting a "chicken-and-egg" dilemma that requires auxiliary information (such as historical data).
- **$\log n$ Gap Between Upper and Lower Bounds**: The upper bound contains a $\log n$ factor while the lower bound does not. Can this gap be eliminated?
- **Strictly Insertion-Only Streams**: The streaming results are limited to insertion-only streams, leaving sliding-window or turnstile models unaddressed.
- **No Multi-party Collisions**: The current parameterization only considers pairwise collisions, lacking a fine-grained characterization of $k$-wise ($k > 2$) collisions.
- **Limited Experimental Scale**: Validated only on a single CAIDA dataset, lacking comprehensive experimental evaluation.

## Related Work & Insights

- **[WZ14]** established the worst-case optimal bound of $\Theta(\alpha/\varepsilon^2 + \alpha\log n)$ for distributed distinct element counting, which is the main target this work aims to beat.
- **[KNW10, Bla20]** provided single-stream optimal distinct elements algorithms with $O(1/\varepsilon^2 + \log n)$ space.
- **[BJKS04, CKS03]** are foundational works on the communication complexity of set disjointness, serving as primary analytical tools for the lower bounds in this work.
- **[CR12]** provided the Gap-Hamming communication lower bound, which governs the outer layer of the composition lower bound in this work.
- **Learning-augmented algorithms [Mit18]**: Improve algorithm performance using auxiliary predictions, echoing the idea of utilizing prior knowledge of collisions in this work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Re-evaluates a "closed" problem through the lens of parameterized complexity, which is highly novel.
- Experimental Thoroughness: ⭐⭐⭐ — Validated only on the CAIDA dataset; the ratio of theory to experiments is highly unbalanced.
- Writing Quality: ⭐⭐⭐⭐ — Clear technical overview, precise theorem statements, and complete structure.
- Value: ⭐⭐⭐⭐⭐ — Highly valuable for the theoretical community (communication complexity and streaming algorithms).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Multiple-Policy Evaluation via Density Estimation](multiple-policy_evaluation_via_density_estimation.md)
- [\[ICLR 2026\] Enabling Fine-Tuning of Direct Feedback Alignment via Feedback-Weight Matching](../../ICLR2026/learning_theory/enabling_fine-tuning_of_direct_feedback_alignment_via_feedback-weight_matching.md)
- [\[ICLR 2026\] Information Estimation with Discrete Diffusion](../../ICLR2026/learning_theory/information_estimation_with_discrete_diffusion.md)
- [\[ICLR 2026\] Learning-Augmented Moment Estimation on Time-Decay Models](../../ICLR2026/learning_theory/learning-augmented_moment_estimation_on_time-decay_models.md)
- [\[ICLR 2026\] Score-Based Density Estimation from Pairwise Comparisons](../../ICLR2026/learning_theory/score-based_density_estimation_from_pairwise_comparisons.md)

</div>

<!-- RELATED:END -->
