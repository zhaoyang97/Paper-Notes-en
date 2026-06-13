---
title: >-
  [Paper Note] Decision Tree Learning on Product Spaces
description: >-
  [ICML 2026][top-down greedy heuristic] This paper extends the theoretical guarantees of the "top-down greedy decision tree heuristic" from Blanc et al. (ITCS'20) from uniform distributions to **any product distribution**…
tags:
  - "ICML 2026"
  - "top-down greedy heuristic"
  - "influence splitting"
  - "product distribution"
  - "PAC learning"
  - "parameter-free"
date: 2026-05-08
content_hash: aa0cc6bf296077c2
---

# Decision Tree Learning on Product Spaces

**Conference**: ICML 2026  
**arXiv**: [2605.12983](https://arxiv.org/abs/2605.12983)  
**Code**: None (Theoretical paper)  
**Area**: Learning Theory / Decision Tree Learning  
**Keywords**: top-down greedy heuristic, influence splitting, product distribution, PAC learning, parameter-free

## TL;DR
This paper extends the theoretical guarantees of the "top-down greedy decision tree heuristic" from Blanc et al. (ITCS'20) from uniform distributions to **any product distribution**, providing an upper bound of size $\exp(\Delta_\mathrm{opt} D_\mathrm{opt}\log(e/\epsilon))$ (which is strictly better than ITCS'20 in the full binary tree case) and being **completely parameter-free**—running without prior knowledge of the optimal tree size or depth.

## Background & Motivation

**Background**: Decision trees (ID3 / C4.5 / CART) using "top-down greedy + influence (or equivalent entropy/Gini) splitting" dominate practice across numerous tasks. However, theoretical analysis has long been disconnected from practice—algorithms by Ehrenfeucht-Haussler, Mehta-Raghavan, and Blanc are either brute-force searches or require prior knowledge of $s$ (optimal tree size), which differs significantly from real-world algorithms.

**Limitations of Prior Work**: (a) Blanc et al. ITCS'20 provided the first strict guarantees for top-down greedy, but the analysis heavily relied on **uniform distribution + Boolean Fourier analysis**, limiting its applicability; (b) Feature distributions in real-world data are often highly non-uniform, making prior theoretical guarantees insufficient as explanations; (c) Even the implementation of Blanc et al. requires knowing $s$ beforehand to select hyperparameters, making it unusable in engineering.

**Key Challenge**: The gap between practical algorithms (adaptive, splitting by local maximum influence, no global parameters) versus theoretical algorithms (global optimization, reliance on uniform distributions, requiring prior knowledge of $s$).

**Goal**: (1) Extend top-down greedy guarantees to any product distribution $\mu=\mu_1\times\cdots\times\mu_n$; (2) Strictly tighten the upper bound of Blanc et al. in the full binary tree case; (3) Provide a parameter-free implementation and a robust version (tolerating sample estimation errors).

**Key Insight**: Avoid Fourier-analytic tools and instead use "two depth parameters"—maximum depth $D_\mathrm{opt}$ (used for the total influence ≤ depth × variance inequality in Lemma 4.2) and average depth $\Delta_\mathrm{opt}$ (used for the max-influence ≥ variance / average depth inequality from O'Donnell 2005 in Lemma C.1). The product $\Delta_\mathrm{opt} D_\mathrm{opt}$ serves as the mixed driving term.

**Core Idea**: Use "cost = $\sum_\mathrm{leaves} p_v \cdot \mathrm{Inf}(f_v)$" as a potential function to prove (a) error ≤ cost, (b) each split step strictly reduces the cost by an amount equal to the leaf's score, and (c) provide lower bounds for the score in two distinct cost intervals to bound the total number of steps.

## Method

As a purely theoretical paper, the "Method" comprises the algorithm (derived from Blanc et al. ITCS'20), new analysis, and a parameter-free implementation.

### Overall Architecture
Algorithm `BuildTopDownDT(f, ε)`: Starting from a single-leaf tree, each round calculates score = $p_v \cdot \max_i \mathrm{Inf}^\mu_i(f_v)$ for every leaf and selects the leaf with the highest score to split on its most influential variable. The process stops when the $f$-completion (by majority label) $\epsilon$-approximates $f$. The analysis framework centers on a two-phase reduction of the cost potential function: first from $\mathrm{Inf}(f)$ to $\epsilon D_\mathrm{opt}$ (Phase 1, Lemma 4.6), then from $\epsilon D_\mathrm{opt}$ until error ≤ $\epsilon$ (Phase 2, Lemma 4.7).

### Key Designs

1.  **Influence + Cost Potential under Product Distributions**:
    - **Function**: Translates "what the algorithm does at each step" into a monotonically decreasing scalar and provides a link where error ≤ cost (Lemma 4.1).
    - **Mechanism**: Influence is defined as $\mathrm{Inf}^\mu_i(f)=\Pr_{x\sim\mu}[f(x)\neq f(x^{(i)})]$, where $x^{(i)}$ is obtained by re-sampling the $i$-th bit according to $\mu_i$. The leaf score is $\mathrm{Score}(v)=p_v\cdot \max_i \mathrm{Inf}_i(f_v)$, and the tree cost is $\mathrm{cost}(T^\circ)=\sum_{v\in\mathrm{leaves}} p_v\cdot \mathrm{Inf}(f_v)$. Lemma 4.3 proves that splitting leaf $v$ reduces cost by exactly $\mathrm{Score}(v)$, translating "algorithmic greed" into a "cost reduction rate."
    - **Design Motivation**: By relying only on the probabilistic definition of influence and the product structure rather than Fourier coefficients, the analysis naturally extends to arbitrary product distributions.

2.  **Upper Bound Driven by Mixed Depth Parameters**:
    - **Function**: Achieves a tighter bound of $\max\bigl((e\Delta_\mathrm{opt}/(\epsilon D_\mathrm{opt}))^{\Delta_\mathrm{opt} D_\mathrm{opt}}, e^{\Delta_\mathrm{opt} D_\mathrm{opt}}\bigr)$ (Theorem 1.1).
    - **Mechanism**: $D_\mathrm{opt}$ enters via $\mathrm{Inf}(f)\le D(T)\cdot \mathrm{Var}(f)$ (Lemma 4.2); $\Delta_\mathrm{opt}$ enters via the max-influence inequality $\max_i \mathrm{Inf}_i(f)\ge \mathrm{Var}(f)/\Delta(T)$. Two score lower bounds (Lemma 4.4 for cost ≤ ε$D_\mathrm{opt}$ and Lemma 4.5 for cost ≥ ε$D_\mathrm{opt}$) provide step bounds for each phase. For path-like trees ($\Delta_\mathrm{opt}$ constant, $D_\mathrm{opt}=n$), $\Delta_\mathrm{opt} D_\mathrm{opt}$ is much smaller than $D_\mathrm{opt}^2$, yielding an exponentially better bound than using $D_\mathrm{opt}$ alone. For balanced trees ($D_\mathrm{opt}=\Delta_\mathrm{opt}=\log s$), the bound is $s^{\log s\log(e/\epsilon)}$, slightly better than Blanc et al.'s $s^{O(\log(s/\epsilon)\log(1/\epsilon))}$.
    - **Design Motivation**: Under uniform distributions, $D_\mathrm{opt}=\Delta_\mathrm{opt}$, so prior work did not need to distinguish them. Under non-uniform distributions, they can differ exponentially, making separate tracking critical for tightening the bound.

3.  **Parameter-free + Robust Approximate Implementation**:
    - **Function**: Runs without requiring prior knowledge of $s$ or $D_\mathrm{opt}$ and tolerates score estimation errors (selecting a leaf with score ≥ 1/4 of the true maximum is sufficient).
    - **Mechanism**: Theorem 5.1 proves that if the selected leaf satisfies $\mathrm{Score}(l')\ge \frac14 \max_l \mathrm{Score}(l)$, the exponent only degrades to $4\Delta_\mathrm{opt} D_\mathrm{opt}$. Scores are estimated using unbiased samples $\widehat{\mathrm{Score}}(l,i,E_i)=\frac{1}{|E_i|}\sum_{(x,x^{(i)})}\mathbf 1[x,x^{(i)}\to l]\mathbf 1[f(x)\neq f(x^{(i)})]$ with Chernoff bounds. The sample complexity per step is $M_S(j,\delta,\epsilon,n)=\frac{12(j+1)n}{\epsilon}\log\frac{4j^2(j+1)n}{\delta}$. Tree error is estimated via majority-vote ERM as a stopping criterion.
    - **Design Motivation**: Theoretical algorithms previously required "prior knowledge of $s$" for termination conditions or hyperparameters, which is impractical. This provides the first directly runnable version.

### Loss & Training
N/A (Theoretical paper). The optimization objective of the algorithm itself is to "split the leaf with the maximum score," which is equivalent to greedily reducing cost. Termination occurs when estimated error ≤ ε (using ERM majority labels + Chernoff sampling).

## Key Experimental Results

### Main Results (Theoretical results, not empirical data)

| Setting | Upper Bound | Comparison with Blanc et al. ITCS'20 |
|---|---|---|
| Arbitrary Product Distribution, General Tree | $\max((e\Delta_\mathrm{opt}/(\epsilon D_\mathrm{opt}))^{\Delta_\mathrm{opt} D_\mathrm{opt}}, e^{\Delta_\mathrm{opt} D_\mathrm{opt}})$ | Generalizes to non-uniform distributions |
| Uniform Distribution + Full Binary Tree ($\Delta_\mathrm{opt}=D_\mathrm{opt}=\log s$) | $s^{\log s\cdot\log(e/\epsilon)}$ | Slightly tighter than $s^{O(\log(s/\epsilon)\log(1/\epsilon))}$ |
| Balanced Tree ($D_\mathrm{opt},\Delta_\mathrm{opt}\in O(\log s)$) | $s^{O(\log s\cdot\log(e/\epsilon))}$ | Same as above |
| Path-like Tree ($\Delta_\mathrm{opt}$ constant, $D_\mathrm{opt}=n$) | Exponentially better than $D_\mathrm{opt}^2$ bound | Benefit of separate parameter tracking |

### Key Robustness Results

| Configuration | Upper Bound | Description |
|---|---|---|
| Exact Score | $\Delta_\mathrm{opt} D_\mathrm{opt}$ exponent | Theorem 1.1 |
| Selected Leaf Score ≥ ¼ Max | $4\Delta_\mathrm{opt} D_\mathrm{opt}$ exponent | Theorem 5.1, tolerates sample estimation |
| Per-step Sample Complexity | $\tilde O((j+1)n/\epsilon)$ | $\delta$/total failure probability = δ/2 |

### Key Findings
- $D_\mathrm{opt}$ and $\Delta_\mathrm{opt}$ must be tracked separately; doing so brings exponential improvements for path-like trees, a phenomenon otherwise hidden by the symmetry of uniform distributions.
- The greedy algorithm is highly robust—selecting a ¼-approximate optimal leaf is sufficient, allowing for practical sample estimation. This explains why CART/C4.5 remains effective on noisy data.
- Since Koch et al. (2023) proved that decision tree learning lacks poly-size algorithms, the quasi-polynomial dependence on $s$ is nearly tight within a constant gap—the bounds in this paper closely approach the lower limits.

## Highlights & Insights
- Completely eliminates Boolean Fourier tools—providing a "non-Fourier path" for further theoretical analysis (noise stability, agnostic learning, etc.) on product spaces.
- Separately bounding the cost using $D_\mathrm{opt}$ and $\Delta_\mathrm{opt}$ in two inequalities is a rare "mixed parameter exponential bound" technique that can be transferred to any analysis of greedy algorithms with monotonic potential functions.
- The parameter-free implementation and 1/4-approximation robustness bridge the gap between "theoretical algorithms" and "runnable algorithms," a step often overlooked in prior theory papers.

## Limitations & Future Work
- The bound remains quasi-polynomial ($s^{\log s}$), not polynomial. Lower bounds by Koch et al. suggest this is unavoidable in the worst case, but distribution-specific tight bounds for real-world scenarios are still missing.
- Limited to product distributions $\mu=\mu_1\times\cdots\times\mu_n$. Extending this to non-product (Markov or general) distributions where features are correlated remains an open problem.
- The error metric is 0-1 loss (Boolean functions $\{\pm 1\}$); regression trees or soft labels are not directly covered.
- Sample complexity still depends on $n$ (number of features) in the worst case, leading to high overhead in high-dimensional sparse scenarios.

## Related Work & Insights
- **vs Blanc et al. (ITCS'20)**: Also analyzes top-down greedy but only supports uniform distributions and relies on Fourier; Ours bypasses this using product-space influence and variance-depth inequalities.
- **vs Mehta-Raghavan (TCS'02)**: Provides $n^{O(\log(s/\epsilon))}$ DP algorithm, but only covers uniform distributions and deviates from practical algorithms.
- **vs Blanc et al. (FOCS'22)**: Designs a polylog-influential variable algorithm reaching $n^{O(\log\log n)}$ runtime, which is more complex and non-greedy; Ours provides strict analysis for the actual greedy algorithm used in practice.
- **vs Koch et al. (SODA'23, COLT'24)**: Provides superpolynomial / NP-hard lower bounds, proving that the quasi-poly upper bound in this paper is nearly tight.

## Rating
- Novelty: ⭐⭐⭐⭐ While not a new algorithm, the use of "mixed dual-depth + non-Fourier" techniques extends guarantees to arbitrary product distributions and enables a parameter-free implementation.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical paper with no empirical experiments; however, theoretical analysis is well-aligned with upper and lower bounds.
- Writing Quality: ⭐⭐⭐⭐ Clear progression of lemmas; proof sketches provide good intuition, though some notation is slightly dense.
- Value: ⭐⭐⭐⭐ Serves as a meaningful bridge for decision tree theory—providing the first strict guarantees for the actual greedy algorithm in settings reflecting realistic data distributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices](../../AAAI2026/others/dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[ICML 2026\] Structure-Induced Information for Rerooting Levin Tree Search](structure-induced_information_for_rerooting_levin_tree_search.md)
- [\[ICML 2026\] HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces](haste_hardware-aware_dynamic_sparse_training_for_large_output_spaces.md)
- [\[AAAI 2026\] From Sequential to Recursive: Enhancing Decision-Focused Learning with Bidirectional Feedback](../../AAAI2026/others/from_sequential_to_recursive_enhancing_decision-focused_learning_with_bidirectio.md)

</div>

<!-- RELATED:END -->
