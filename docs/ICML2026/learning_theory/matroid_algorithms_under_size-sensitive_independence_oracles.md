---
title: >-
  [Paper Note] Matroid Algorithms Under Size-Sensitive Independence Oracles
description: >-
  [ICML 2026][learning_theory][Paper Note] The authors propose a size-sensitive matroid oracle model where the "query cost grows linearly with the size of the query set." They prove that under this model, the optimal query costs for finding a basis, estimating the rank, and estimating the partition number are all $\tilde{\Theta}(n^2)$. Furthermore, for matroids
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: 52c84f2a67bfb508
---
# Matroid Algorithms Under Size-Sensitive Independence Oracles

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.00201](https://arxiv.org/abs/2605.00201)  
**Code**: None (Theoretical paper)  
**Area**: Algorithm Theory / Combinatorial Optimization  
**Keywords**: Matroid, Independence Oracle, Size-Sensitive Query Cost, Lower Bound, Bounded Girth  

## TL;DR
The authors propose a size-sensitive matroid oracle model where the "query cost grows linearly with the size of the query set." They prove that under this model, the optimal query costs for finding a basis, estimating the rank, and estimating the partition number are all $\tilde{\Theta}(n^2)$. Furthermore, for matroids with a bounded girth $c$, they provide a maximum weight basis algorithm with a complexity of $\mathcal{O}(n^{2-1/c}\log n)$, breaking the quadratic lower bound.

## Background & Motivation

**Background**: Matroids are core abstractions in combinatorial optimization for characterizing "subset selection under constraints." In machine learning, they are widely used for feasibility constraints in bandits/online learning, submodular maximization, preference guidance, and allocation mechanisms. Algorithm analysis almost exclusively adopts the "independence oracle" model: given a set $Q \subseteq E$, the oracle answers whether $Q \in \mathcal{I}$ in $\mathcal{O}(1)$ time, and the entire literature uses the "number of queries" as the complexity metric.

**Limitations of Prior Work**: Constant-time oracles are often unrealistic in practice. For instance, in a graphic matroid, determining whether an edge set forms a forest requires $\Theta(|Q|)$ work using Union-Find or DFS. Oracles for other "natural" matroid classes (bicyclic, transversal, scheduling) are also closer to linear rather than constant. This implies that published algorithms with "$\mathcal{O}(n)$ queries" may actually take $\mathcal{O}(n^2)$ real time, leading to a serious disconnect between theoretical analysis and practical execution.

**Key Challenge**: To make analysis guiding for practice, the oracle cost must be explicitly modeled as a function of $|Q|$. However, this immediately invalidates classic "query counting" lower bounds—large queries are more expensive than small ones, and algorithms might use many small queries to save on total cost. Matching upper and lower bounds must be re-established.

**Goal**: Analyze three fundamental matroid tasks under the size-sensitive model (where querying $Q$ costs $|Q|$): (i) finding a basis; (ii) approximating the rank; (iii) computing/approximating the partition number $k(M)$; while also considering general non-decreasing cost functions $f(|Q|)$.

**Key Insight**: The authors noticed that "greedy algorithms" are naturally $\mathcal{O}(n^2)$ under this model. The problem then becomes: "Can smarter query strategies break the quadratic barrier?" They constructed a family of matroid instances where all small queries are "automatically yes"—thus any informative query must be large ($\Theta(n)$), forcing the cost to be quadratic.

**Core Idea**: Use "Free Matroid + Union of Uniform Matroids + Truncation" (for the rank task) and "Partition Matroid + $\ell$-relaxation + Truncation" (for the partition task) to construct hard instances. These are combined with Yao's Principle to convert deterministic decision tree lower bounds into randomized lower bounds. The upper bounds utilize existing base-covering algorithms adapted with truncation.

## Method

### Overall Architecture
The paper follows two main threads. Lower bound thread: (1) Define the size-sensitive oracle; (2) Construct the hard instance distribution $\mathcal{D}_{m,\epsilon}$; (3) Argue that "the witness needed to distinguish instances must be large" and use counting arguments to prove any decision tree requires $\Omega(m)$ large queries, each costing $\Omega(m)$, resulting in a total cost of $\Omega(n^2)$; (4) Upgrade to randomized algorithms via Yao's Principle. Upper bound thread: (a) For the partition number, apply the base-cover method from Quanrud (2024) truncated to rank $\lceil n/k \rceil$, yielding $\tilde{\mathcal{O}}(n^2)$; (b) For maximum weight basis with bounded girth $c$, use a sub-quadratic algorithm combining random subsampling and binary search to locate the "minimum weight circuit element."

### Key Designs

**1. "Uninformative Small Queries" Hard Instance: Forcing Large Queries**

In the size-sensitive model, large queries are expensive and small queries are cheap. To prove a quadratic lower bound, one must block the path of "saving total cost with many small queries." The core trick is constructing a family of matroids where any query no larger than $m$ is automatically judged as independent. Since small queries are all indistinguishable "yes" answers, cheap algorithms have no recourse. For the rank task, fix $n=3m$, pick a subset $S \subseteq [3m]$ of size $m$, and define $M_{m,S}$ as the union of a "free matroid on $S$" and a "uniform matroid of rank $m$ on $T=[3m]\setminus S$": its rank is $2m$, and any set of size $\le m$ is independent (Lemma 4.2). Truncating this to rank $2m-\epsilon m$ yields $M'_{m,S,\epsilon}$. To distinguish these two matroids, one must find a witness $W$ that is "independent in the original but dependent after truncation," satisfying $|W| > 2m-\epsilon m$ and $|W\setminus S| \le m$—such a witness is destined to be large. The partition task uses an "equally sized $m$-segment partition of size $\alpha+1$" + $\ell=m/\alpha$-relaxation + rank-minus-1 truncation, similarly making queries $\le m/\alpha$ automatically independent. The essence of the oracle model is "how many instances a query can distinguish"; by designing all low-cost queries as indistinguishable "yes" responses, all cheap algorithms are effectively neutralized.

**2. Witness Counting + Yao's Principle: Translating Decision Tree Depth to Randomized Lower Bounds**

After constructing hard instances, the requirement of "how many large queries a deterministic decision tree must make" is upgraded to a lower bound for randomized algorithms. The key is witness counting: after fixing a witness $W$, the number of $S$ that can make it a witness is bounded by a strict binomial coefficient upper bound (Lemma 4.5: at most $\binom{2m-\delta m}{m-\delta m}\binom{2m+\delta m}{\delta m}$). A decision tree of depth $q$ explores at most $2^{q+1}$ candidate sets. Its success probability under the uniform distribution $\mathcal{D}_{m,\epsilon}$ is bounded by:

$$\frac{1}{2}+\frac{2^q\cdot\binom{2m}{m}\binom{2m+\epsilon m}{2m}}{\binom{3m}{m}}$$

To increase the success rate from $1/2$ to $2/3$, $q=\Omega(m)$ is required, with each large query costing $\Omega(m)$. Yao's Principle then provides the $\Omega(m^2)=\Omega(n^2)$ cost lower bound for randomized algorithms on the worst-case instance. "Construct hard distribution + count witness + decision tree exponentiality + Yao" is a standard pipeline for combinatorial lower bounds, but here it is applied for the first time to fundamental matroid tasks in a size-sensitive model.

**3. Sub-quadratic Basis Algorithm for Bounded Girth: Locating Non-basis Elements via "Probabilistic Circuit Sniping"**

The root of the quadratic lower bound is that "locating a single non-basis element might require a very large dependent set." However, if all circuit sizes are $\le c$, every non-basis element has a "circuit fingerprint" of size at most $c$. Sparse sampling can efficiently "snip" it. Algorithm 1 works backward from $B \leftarrow E$ for $n \ln n$ rounds: in each round, elements are independently included in $S$ with probability $n^{-1/c}$; if $S$ is dependent, they are sorted by weight descending, and binary search finds the last element of the minimum dependent prefix (which must be the minimum weight element in some circuit, i.e., a non-basis element), which is then removed from both $B$ and $S$. For each $d \notin B^*$, its fundamental circuit $C_d$ (size $\le c$) falls entirely into $S$ with probability $\ge (n^{-1/c})^c = n^{-1}$. Thus, the probability that $d$ survives $n \ln n$ rounds is $\le 1/n$, meaning the expected number of residual non-basis elements is only 1. In each round, the expected size of $|S|$ is $n^{1-1/c}$, and binary search takes $\mathcal{O}(\log n)$ queries, leading to a total cost of $\mathcal{O}(n^{2-1/c}\log n)$. The sampling probability $n^{-1/c}$ is carefully tuned to allow circuits $\le c$ to be caught with probability $\ge n^{-1}$, replacing expensive large queries with a multitude of small queries.

### Loss & Training
This is a pure theory paper with no training. Lower bounds use Yao's Principle and decision tree arguments. The main upper bound algorithm is a randomized sketch with binary search (Algorithm 1). The partition number upper bound is achieved by applying the $\tilde{\mathcal{O}}(nk)$ query complexity algorithm of Quanrud (2024) to a matroid truncated to rank $\lceil n/k \rceil$, limiting each query size to $\mathcal{O}(n/k)$, resulting in a total cost of $\tilde{\mathcal{O}}(n \cdot k \cdot n/k) = \tilde{\mathcal{O}}(n^2)$.

## Key Experimental Results

### Main Results (Theoretical Summary)

| Task | Upper Bound | Lower Bound | Remarks |
|---|---|---|---|
| Basis / Rank (General) | $\mathcal{O}(n^2)$ (Greedy) | $\Omega(n^2)$ (Thm 1.1.1) | Even $1\pm 1/40$ approx is quadratic |
| Partition (General) | $\tilde{\mathcal{O}}(n^2)$ (Thm 1.1.2) | $\Omega(n^2)$ (Distinguish $3$ vs $4$) | $(1+\epsilon)$-approx ($\epsilon<1/3$) is quadratic |
| Max Weight Basis (Girth $\le c$) | $\mathcal{O}(n^{2-1/c}\log n)$ (Algo 1) | —— | First sub-quadratic result |
| General Cost $f(|Q|)$ (Rank) | —— | $\Omega(n\cdot f(n/3))$ (Thm 1.2) | If $f$ is polynomial, simplifies to $\Omega(n\cdot f(n))$ |
| General Cost $f(|Q|)$ (Partition) | —— | $\Omega(n\cdot f(n/12))$ | Same as above |

### Ablation Study (Model Comparison)

| Model Variant | Basis Complexity | Description |
|---|---|---|
| Classic $\mathcal{O}(1)$ oracle | $\mathcal{O}(n)$ queries | Disconnected from practice; cannot reflect runtime |
| Dynamic oracle (Blikstad 2023) | Greedy can be sub-quadratic | Requires oracle to maintain state; different from stateless |
| Ours (Size-sensitive) | $\Theta(n^2)$ (Tight bound) | Naturally matches linear oracles like graphic matroids |
| Ours + Bounded Girth $c$ | $\mathcal{O}(n^{2-1/c}\log n)$ | Degenerates to $\tilde{\mathcal{O}}(n^2)$ as $c \to \infty$, consistent with general case |

### Key Findings
- "Approximation does not save money" is a strong conclusion of this model: even for a $1\pm 1/40$ rank approximation, the cost remains quadratic. This perfectly matches the actual algorithmic costs of spanning forest tasks on dense graphs.
- Bounded girth is one of the few structural assumptions that can truly break the quadratic barrier—it provides "circuit fingerprints" of size $\le c$ for non-basis elements, allowing sparse sampling to locate them efficiently.
- The lower bound of $\Omega(n \cdot f(n))$ for general cost functions (where $f$ is polynomial) shows the robustness of the conclusions across various oracle implementation cost curves.

## Highlights & Insights
- Changing the oracle cost model from "counting" to "paying by size" is a seemingly small but far-reaching shift in perspective. It immediately subjects a large class of "$\mathcal{O}(n)$ query" algorithms to re-examination and aligns the theoretical analysis of matroids with the actual runtime of special cases like graphic matroids.
- "Making all small queries uninformative" is a transferable template for lower bound construction: the union of a free matroid and a uniform matroid forces small sets to be independent, with truncation and witnesses serving as the means of distinction. This approach can be generalized to other "pay by set size" oracle complexity scenarios.
- The $n^{-1/c}$ random sampling in Algorithm 1 is meticulously tuned—ensuring that circuits of size $\le c$ are "caught" in their entirety with probability $\ge n^{-1}$, which, combined with $n \ln n$ rounds, eliminates non-basis elements with high probability. This "probabilistic circuit sniping" concept can inspire other sparse identification algorithms with local structures.

## Limitations & Future Work
- The lower bounds are for the memoryless (stateless) model. The authors explicitly point out that in the dynamic oracle setting (Blikstad 2023), greedy can be cheaper, so the conclusions here do not directly extrapolate there.
- The $\mathcal{O}(n^{2-1/c}\log n)$ result applies only to the maximum weight basis algorithm and does not provide a "unified framework for all matroid tasks under bounded girth."
- Neither the lower nor the upper bounds account for caching mechanisms where the same set is queried multiple times. In real-world systems, such locality could significantly reduce the effective cost.
- The paper does not provide numerical experiments or comparisons on real matroid instances (such as dense graphs), remaining purely theoretical.

## Related Work & Insights
- **vs. Eberle et al. (2024) (Oracles with Budgets)**: They also focus on oracle costs but approach it from an "augmented oracle" perspective; this paper redefines costs on the existing oracle interface.
- **vs. Blikstad et al. (2023) (Dynamic Oracles)**: The dynamic model allows the oracle to maintain state, making greedy cheaper; the stateless model in this paper is more suitable for distributed systems or REST APIs.
- **vs. Quanrud (2024) (Base Covering)**: This paper directly incorporates its $\tilde{\mathcal{O}}(nk)$ query complexity algorithm into the size-sensitive model. By using truncation to cap each query size at $\mathcal{O}(n/k)$, it derives a partition number upper bound, cleverly reusing existing results.

## Rating
- Novelty: ⭐⭐⭐⭐ Redefining the oracle cost model is a simple but long-overlooked perspective; the entire set of bounds is new.
- Experimental Thoroughness: ⭐⭐⭐⭐ As a theoretical paper, the bounds for three tasks are aligned up to logarithmic factors and extended to general cost functions, providing comprehensive coverage; no empirical evidence.
- Writing Quality: ⭐⭐⭐⭐ Definitions, lemmas, and theorems are clearly structured. The intuitive explanation of lower bound construction is well-executed, though some counting argument details are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Significant impact on the combinatorial optimization theory community: it provides a more realistic runtime comparison for existing "pay-per-query" matroid algorithms and initiates a new generation of size-sensitive complexity research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2025\] Principled Algorithms for Optimizing Generalized Metrics in Binary Classification](../../ICML2025/learning_theory/principled_algorithms_for_optimizing_generalized_metrics_in_binary_classificatio.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)

</div>

<!-- RELATED:END -->
