---
title: >-
  [Paper Note] Matroid Algorithms Under Size-Sensitive Independence Oracles
description: >-
  [ICML 2026][Algorithm Theory / Combinatorial Optimization][Matroid] The authors propose a "size-sensitive matroid oracle" model where the query cost grows linearly with the size of the query set. They prove that under th…
tags:
  - "ICML 2026"
  - "Algorithm Theory / Combinatorial Optimization"
  - "Matroid"
  - "independence oracle"
  - "size-sensitive query cost"
  - "lower bound"
  - "bounded girth"
date: 2026-05-08
content_hash: b3ae746e753ae712
---

# Matroid Algorithms Under Size-Sensitive Independence Oracles

**Conference**: ICML 2026  
**arXiv**: [2605.00201](https://arxiv.org/abs/2605.00201)  
**Code**: None (Theoretical paper)  
**Area**: Algorithm Theory / Combinatorial Optimization  
**Keywords**: Matroid, independence oracle, size-sensitive query cost, lower bound, bounded girth

## TL;DR
The authors propose a "size-sensitive matroid oracle" model where the query cost grows linearly with the size of the query set. They prove that under this model, the optimal query cost for finding a basis, estimating rank, and estimating the partition number is $\tilde{\Theta}(n^2)$. Furthermore, for matroids with a bounded girth $c$, they present a maximum weight basis algorithm with a complexity of $\mathcal{O}(n^{2-1/c}\log n)$, breaking the quadratic lower bound.

## Background & Motivation

**Background**: Matroids are core abstractions in combinatorial optimization for characterizing "constrained subset selection." In ML, they are widely used for feasibility constraints in bandits and online learning, submodular maximization, preference-guided allocation mechanisms, and more. Current algorithmic analyses almost exclusively adopt the "independence oracle" model: given a set $Q\subseteq E$, the oracle returns whether $Q\in\mathcal{I}$ in $\mathcal{O}(1)$ time, and the "query complexity" is measured by the number of queries.

**Limitations of Prior Work**: Constant-time oracles are unrealistic in practice. For instance, in a graphic matroid, determining if an edge set forms a forest requires $\Theta(|Q|)$ work using Union-Find or DFS. Oracles for other "natural" matroid classes (bicircular, transversal, scheduling) are also near-linear rather than constant. This implies that published "$\mathcal{O}(n)$ query" algorithms may actually consume $\mathcal{O}(n^2)$ real time, leading to a serious disconnect between theoretical analysis and practical execution.

**Key Challenge**: To ensure analysis provides practical guidance, the oracle cost must be explicitly modeled as a function of $|Q|$. However, this immediately renders classic "query counting" lower bounds inapplicable—large queries become more expensive than small ones, and algorithms might use many small queries to save total cost. Matching upper and lower bounds must be re-established.

**Goal**: Analyze three fundamental matroid tasks under the size-sensitive model (where querying $Q$ costs $|Q|$): (i) finding a basis; (ii) approximating the rank; (iii) computing/approximating the partition number $k(M)$. Additionally, investigate general non-decreasing cost functions $f(|Q|)$.

**Key Insight**: The authors observe that the "greedy algorithm" naturally costs $\mathcal{O}(n^2)$ in this model. The question becomes: "Can more clever query strategies break this quadratic barrier?" They construct a family of matroid instances where all small queries "automatically return yes," forcing any informative query to be large ($\Theta(n)$), thus driving the cost to quadratic.

**Core Idea**: Construct hard instance families using "free matroid + union of uniform matroids + truncation" (for rank tasks) and "partition matroid + $\ell$-relaxation + truncation" (for partition tasks). Yao's minimax principle is used to convert deterministic decision tree lower bounds into randomized ones. Upper bounds utilize existing base-covering algorithms combined with adaptive truncation.

## Method

### Overall Architecture
The paper follows two trajectories. Lower bound track: (1) Define the size-sensitive oracle; (2) Construct a hard instance distribution $\mathcal{D}_{m,\epsilon}$; (3) Argue that the "witness" needed to distinguish instances must be large, using counting arguments to show any decision tree requires $\Omega(m)$ large queries, each costing $\Omega(m)$, totaling $\Omega(n^2)$; (4) Upgrade to randomized algorithms via Yao’s Principle. Upper bound track: (a) For partition numbers, apply the base-covering method of Quanrud (2024) truncated at rank $\lceil n/k\rceil$ to achieve $\tilde{\mathcal{O}}(n^2)$; (b) For maximum weight bases with bounded girth $c$, use a sub-quadratic algorithm combining randomized sub-sampling and binary search to locate "minimum weight circuit elements."

### Key Designs

1.  **Hard Instance Construction with "Uninformative Small Queries"**:
    - **Function**: Forces any query smaller than $m$ to be automatically independent, compelling the algorithm to perform large queries.
    - **Mechanism**: For rank tasks, fix $n=3m$, choose a subset $S\subseteq[3m]$ of size $m$, and define $M_{m,S}$ as the union of a "free matroid on $S$" and a "uniform matroid of rank $m$ on $T=[3m]\setminus S$." Its rank is $2m$, and any set of size $\le m$ is independent (Lemma 4.2). This is then truncated to rank $2m-\epsilon m$ to obtain $M'_{m,S,\epsilon}$. To distinguish $M_{m,S}$ from $M'_{m,S,\epsilon}$, one must find a witness $W$, such that $|W|>2m-\epsilon m$ and $|W\setminus S|\le m$. For partition tasks, a similar "relaxation + truncation" method is used.
    - **Design Motivation**: The core of the oracle model is how much information a query distinguishes. Designing all low-cost queries to return "yes" indiscriminately blocks all paths for cheap algorithms.

2.  **Witness Counting + Yao’s Principle for Randomized Lower Bounds**:
    - **Function**: Translates the number of large queries required by a deterministic decision tree into a lower bound for randomized algorithms.
    - **Mechanism**: For a fixed witness $W$, the number of sets $S$ for which $W$ serves as a witness is strictly bounded by binomial coefficients (Lemma 4.5: at most $\binom{2m-\delta m}{m-\delta m}\binom{2m+\delta m}{\delta m}$). A decision tree of depth $q$ explores at most $2^{q+1}$ candidate sets, so its success probability under the uniform distribution $\mathcal{D}_{m,\epsilon}$ is bounded by $\frac{1}{2}+\frac{2^q\cdot\binom{2m}{m}\binom{2m+\epsilon m}{2m}}{\binom{3m}{m}}$. To raise this probability to $2/3$, $q=\Omega(m)$ is required. Yao's Principle then yields an $\Omega(m^2)=\Omega(n^2)$ cost lower bound for randomized algorithms.
    - **Design Motivation**: This pipeline (hard distribution + witness counting + decision tree depth + Yao) is standard for combinatorial lower bounds, but this is its first successful application to basic matroid tasks in a size-sensitive model.

3.  **Randomized Basis Algorithm Breaking Quardatic Cost for Bounded Girth**:
    - **Function**: Finds a maximum weight basis with an expected cost of $\mathcal{O}(n^{2-1/c}\log n)$ when all circuits have size $\le c$.
    - **Mechanism**: Algorithm 1 operates in reverse—starting from $B\leftarrow E$, it runs $n\ln n$ rounds. In each round, elements are independently included in $S$ with probability $n^{-1/c}$. If $S$ is dependent, elements are sorted by weight, and binary search finds the last element of the minimum dependent prefix to remove it from both $B$ and $S$. For any $d\notin B^*$, the probability that its fundamental circuit $C_d$ is fully contained in $S$ is $\ge(n^{-1/c})^c=n^{-1}$, so the probability $d$ survives $n\ln n$ rounds is $\le 1/n$. Each round has an expected $|S|$ of $n^{1-1/c}$ and requires $\mathcal{O}(\log n)$ queries.
    - **Design Motivation**: The quadratic lower bound stems from the fact that a single non-basis element might require a very large dependent set to be identified. Bounded girth ensures each element's "fingerprint" is small ($\le c$), allowing sparse sampling to "trap" the circuit with high probability.

### Loss & Training
This is a purely theoretical paper with no training. Lower bounds use Yao's Principle; the primary upper bound algorithm is a randomized sketch with binary search (Algorithm 1). The partition number upper bound is derived by applying Quanrud’s (2024) $\tilde{\mathcal{O}}(nk)$ query algorithm to a matroid truncated at rank $\lceil n/k\rceil$, limiting each query size to $\mathcal{O}(n/k)$, resulting in $\tilde{\mathcal{O}}(n \cdot k \cdot n/k) = \tilde{\mathcal{O}}(n^2)$.

## Key Experimental Results

### Main Results (Theoretical Bound Summary)

| Task | Upper Bound | Lower Bound | Remarks |
| :--- | :--- | :--- | :--- |
| Find Basis / Est. Rank (General) | $\mathcal{O}(n^2)$ (Greedy) | $\Omega(n^2)$ (Thm 1.1.1) | Quadratic even for $1\pm 1/40$ approx. |
| Partition Number (General) | $\tilde{\mathcal{O}}(n^2)$ (Thm 1.1.2) | $\Omega(n^2)$ (Distinguish $3$ vs $4$) | Quadratic for $(1+\epsilon)$-approx ($\epsilon<1/3$) |
| Max Weight Basis (Girth $\le c$) | $\mathcal{O}(n^{2-1/c}\log n)$ (Alg 1) | —— | First sub-quadratic result |
| Gen. Cost $f(|Q|)$ (Rank) | —— | $\Omega(n\cdot f(n/3))$ (Thm 1.2) | Simplifies to $\Omega(n\cdot f(n))$ if $f$ is poly. |
| Gen. Cost $f(|Q|)$ (Partition) | —— | $\Omega(n\cdot f(n/12))$ | Same as above |

### Ablation Study (Model Comparisons)

| Model Variant | Basis Complexity | Background |
| :--- | :--- | :--- |
| Classic $\mathcal{O}(1)$ Oracle | $\mathcal{O}(n)$ queries | Disconnected from real runtimes in practice |
| Dynamic Oracle (Blikstad 2023) | Sub-quadratic Greedy | Requires stateful oracle; differs from this work |
| **Ours** (Size-sensitive) | $\Theta(n^2)$ (Tight) | Matches linear oracles in graphic matroids |
| **Ours** + Bounded Girth $c$ | $\mathcal{O}(n^{2-1/c}\log n)$ | Degenerates to $\tilde{\mathcal{O}}(n^2)$ as $c\to\infty$ |

### Key Findings
- **"Approximation saves no money"**: Even for a $1\pm 1/40$ rank approximation, quadratic cost is required. This aligns with actual algorithmic costs for spanning forest tasks on dense graphs.
- **Bounded Girth is a structural breakthrough**: It provides "circuit fingerprints" of size $\le c$ for non-basis elements, which sparse sampling can locate efficiently.
- **Lower bound robustness**: The $\Omega(n\cdot f(n))$ bound for polynomial $f$ shows that the conclusions are robust across various cost curves of oracle implementations.

## Highlights & Insights
- Changing the oracle cost model from "counting" to "pay-as-you-go" is a subtle but impactful shift. It forces a re-evaluation of $\mathcal{O}(n)$ query algorithms and aligns the real runtime of special cases (like graphic matroids) with general theory.
- The "uninformative small queries" strategy is a reusable template for lower bounds: using unions of free and uniform matroids to force small-set independence can be extended to other "set-cost" oracle scenarios.
- The $n^{-1/c}$ sampling rate in Algorithm 1 is finely tuned to ensure circuits of size $\le c$ are trapped with probability $\ge n^{-1}$, which, over $n \ln n$ rounds, clears non-basis elements. This "probabilistic circuit trapping" could inspire other sparse identification algorithms.

## Limitations & Future Work
- Lower bounds apply to the **memoryless** (stateless) model. In a **dynamic oracle** setting (Blikstad 2023), greedy can be cheaper, so these results do not directly generalize there.
- The $\mathcal{O}(n^{2-1/c}\log n)$ result only applies to maximum weight basis; a "unified framework for any matroid task under bounded girth" is missing.
- Cache mechanisms (re-querying the same set) are not considered; in real systems, such locality could significantly reduce effective costs.
- Purely theoretical; no numerical experiments or comparisons on real-world matroid instances (e.g., dense graphs) were provided.

## Related Work & Insights
- **vs. Eberle et al. (2024) (Budgeted oracles)**: They also focus on costs but from an "augmented oracle" perspective; this work redefines costs on the standard interface.
- **vs. Blikstad et al. (2023) (Dynamic oracles)**: Dynamic models allow stateful oracles where greedy is cheaper; the stateless model here is more suited for distributed or REST API scenarios.
- **vs. Quanrud (2024) (Base covering)**: This paper effectively ports the $\tilde{\mathcal{O}}(nk)$ query algorithm into the size-sensitive model by using truncation to cap query sizes at $\mathcal{O}(n/k)$, elegantly reusing existing results.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Redefining the cost model is simple but long-overlooked; the resulting bounds are entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ For a theory paper, the bounds for three tasks are matched up to log factors and extended to general costs; lacks empirical validation.
- **Writing Quality**: ⭐⭐⭐⭐ Definitions and theorems are clear; intuition for lower bounds is well-explained, though some counting details are relegated to appendices.
- **Value**: ⭐⭐⭐⭐ High impact for the combinatorial optimization community, harmonizing "query-based" analysis with real-world runtimes and initiating size-sensitive complexity studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[ICML 2026\] Parsimonious Learning-Augmented Online Metric Matching](parsimonious_learning-augmented_online_metric_matching.md)

</div>

<!-- RELATED:END -->
