---
title: >-
  [Paper Note] Matroid Algorithms Under Size-Sensitive Independence Oracles
description: >-
  [ICML 2026][learning_theory][Paper Note] The authors propose a "size-sensitive matroid oracle" model where the query cost grows linearly with the size of the query set. They prove that under this model, the optimal query costs for finding a basis, estimating the rank, and estimating the partition number are all $\tilde{\Theta}(n^2)$. Furthermore, for matroids
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: e014c794c7ef2011
---
# Matroid Algorithms Under Size-Sensitive Independence Oracles

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.00201](https://arxiv.org/abs/2605.00201)  
**Code**: None (Theoretical paper)  
**Area**: Algorithm Theory / Combinatorial Optimization  
**Keywords**: Matroids, independence oracle, size-sensitive query cost, lower bounds, bounded girth

## TL;DR
The authors propose a "size-sensitive matroid oracle" model where the query cost grows linearly with the size of the query set. They prove that under this model, the optimal query costs for finding a basis, estimating the rank, and estimating the partition number are all $\tilde{\Theta}(n^2)$. Furthermore, for matroids with a bounded girth $c$, they provide a maximum weight basis algorithm with a complexity of $\mathcal{O}(n^{2-1/c}\log n)$, breaking the quadratic lower bound.

## Background & Motivation

**Background**: Matroids are core abstractions in combinatorial optimization used to characterize "subset selection under constraints." In machine learning, they are widely applied in feasibility constraints for bandits and online learning, submodular maximization, preference guidance, and allocation mechanisms. Algorithm analysis almost exclusively adopts the "independence oracle" model: given a set $Q\subseteq E$, the oracle returns whether $Q\in\mathcal{I}$ in $\mathcal{O}(1)$ time, and the complexity is measured by the "number of queries" throughout the literature.

**Limitations of Prior Work**: Constant-time oracles are unrealistic in practice. For example, in graphic matroids, determining whether an edge set forms a forest inherently requires $\Theta(|Q|)$ work using Union-Find or DFS. Oracles for other "natural" matroid classes (bicircular, transversal, scheduling) are also nearly linear rather than constant. This implies that published algorithms with "$\mathcal{O}(n)$ queries" may actually take $\mathcal{O}(n^2)$ real time, causing a serious disconnect between theoretical analysis and practical execution.

**Key Challenge**: To ensure analysis guides practice, the oracle cost must be explicitly modeled as a function of $|Q|$. However, this immediately invalidates classical "query counting" lower bounds—large queries are more expensive than small ones, and algorithms might use many small queries to save total cost. A new matching of upper and lower bounds needs to be established.

**Goal**: Analyze three fundamental matroid tasks under the size-sensitive model (querying $Q$ costs $|Q|$): (i) finding a basis; (ii) approximating the rank; (iii) computing/approximating the partition number $k(M)$. The study also considers a general non-decreasing cost function $f(|Q|)$.

**Key Insight**: The authors observe that the "greedy algorithm" naturally takes $\mathcal{O}(n^2)$ in this model. The question becomes, "Can a cleverer query strategy break the quadratic barrier?" They construct a family of matroid instances where all small queries are "automatically yes"—thus, any informative query must be large ($\Theta(n)$), forcing the cost to be quadratic.

**Core Idea**: Hard instances are constructed using "free matroids + union of uniform matroids + truncation" (for rank tasks) and "partition matroids + $\ell$-relaxation + truncation" (for partition tasks). Yao’s minimax principle is used to convert deterministic decision tree lower bounds into randomized lower bounds. The upper bound utilizes existing base-covering algorithms combined with adaptive truncation.

## Method

### Overall Architecture
The paper follows two main threads. Lower bound thread: (1) Define the size-sensitive oracle; (2) Construct a hard instance distribution $\mathcal{D}_{m,\epsilon}$; (3) Argue that the "witness required to distinguish instances must be large" and use counting arguments to prove any decision tree requires $\Omega(m)$ large queries, each costing $\Omega(m)$, resulting in a total cost of $\Omega(n^2)$; (4) Upgrade to randomized algorithms via Yao's principle. Upper bound thread: (a) For the partition number, apply the base-cover algorithm from Quanrud (2024) truncated to rank $\lceil n/k\rceil$, yielding $\tilde{\mathcal{O}}(n^2)$; (b) For maximum weight basis with bounded girth $c$, use a sub-quadratic algorithm combining randomized sub-sampling and binary search to locate the "minimum weight circuit element."

### Key Designs

**1. Hard instance construction where "small queries are uninformative": Forcing large queries**

In the size-sensitive model, large queries are expensive and small queries are cheap. To prove a quadratic lower bound, one must block the path of "saving total cost by using many small queries." The core trick is constructing a family of matroids where any query no larger than $m$ is automatically judged as independent. Since small queries satisfy all instances with a "yes," cheap algorithms have no recourse. For the rank task, fix $n=3m$, pick a subset $S\subseteq[3m]$ of size $m$, and define $M_{m,S}$ as the matroid union of a "free matroid on $S$" and a "uniform matroid of rank $m$ on $T=[3m]\setminus S$." Its rank is $2m$, and any set of size $\le m$ is independent (Lemma 4.2). Truncating this to rank $2m-\epsilon m$ yields $M'_{m,S,\epsilon}$. To distinguish these two matroids, one must find a witness $W$ that is independent in the original but dependent after truncation, satisfying $|W|>2m-\epsilon m$ and $|W\setminus S|\le m$—such a witness is destined to be large. The partition task uses an "equal partition of $m$ segments of size $\alpha+1$" + $\ell=m/\alpha$-relaxation + rank-minus-1 truncation, similarly making queries $\le m/\alpha$ automatically independent. The essence of the oracle model is "how many instances a query can distinguish"; by designing all low-cost queries as indistinguishable "yes" answers, all cheap algorithms are effectively neutralized.

**2. Witness Counting + Yao’s Principle: Translating decision tree depth to randomized lower bounds**

After constructing hard instances, the requirement for "how many large queries a deterministic decision tree must make" is upgraded to a lower bound for randomized algorithms. The key is witness counting: after fixing a witness $W$, the number of $S$ that can make it a witness is strictly upper-bounded by binomial coefficients (Lemma 4.5: at most $\binom{2m-\delta m}{m-\delta m}\binom{2m+\delta m}{\delta m}$). A decision tree of depth $q$ explores at most $2^{q+1}$ candidate sets. Thus, its success probability under the uniform distribution $\mathcal{D}_{m,\epsilon}$ is controlled by:

$$\frac{1}{2}+\frac{2^q\cdot\binom{2m}{m}\binom{2m+\epsilon m}{2m}}{\binom{3m}{m}}$$

To increase the success rate from $1/2$ to $2/3$, $q=\Omega(m)$ is required, with each large query costing $\Omega(m)$. Yao's principle then provides the $\Omega(m^2)=\Omega(n^2)$ cost lower bound for randomized algorithms on the worst-case instance. "Constructing hard distributions + counting witnesses + decision tree exponent + Yao" is a standard pipeline for combinatorial lower bounds, but this is the first application to fundamental matroid tasks in the size-sensitive model.

**3. Randomized basis algorithm breaking quadratic for bounded girth: "Probabilistic circuit trapping"**

The root of the quadratic lower bound is that "locating a single non-basis element might require a very large dependent set." However, if all circuit sizes are $\le c$, every non-basis element has a "circuit fingerprint" of size at most $c$, which can be efficiently trapped via sparse sampling. Algorithm 1 starts backwards from $B\leftarrow E$ and runs for $n\ln n$ rounds: in each round, elements are independently included in $S$ with probability $n^{-1/c}$. If $S$ is dependent, elements are sorted by weight in descending order, and binary search finds the last element of the minimum dependent prefix (which must be the minimum weight element in a circuit, i.e., a non-basis element), which is then removed from both $B$ and $S$. For each $d\notin B^*$, the probability that its fundamental circuit $C_d$ (size $\le c$) falls entirely into $S$ is $\ge(n^{-1/c})^c=n^{-1}$. Thus, the probability that $d$ survives $n\ln n$ rounds is $\le 1/n$, leaving an expected 1 non-basis element remaining. Each round has an expected $|S|$ of $n^{1-1/c}$ and requires $\mathcal{O}(\log n)$ queries for binary search, leading to a total cost of $\mathcal{O}(n^{2-1/c}\log n)$. The sampling probability $n^{-1/c}$ is carefully tuned to allow circuits of size $\le c$ to be trapped with probability $\ge n^{-1}$, replacing expensive large queries with a multitude of small ones.

### Loss & Training
This is a purely theoretical paper with no training involved. Lower bounds use Yao's principle and decision tree arguments. The primary upper-bound algorithm is a randomized sketch with binary search (Algorithm 1). The partition number upper bound is achieved by applying the $\tilde{\mathcal{O}}(nk)$ query complexity algorithm from Quanrud (2024) to the matroid truncated to rank $\lceil n/k\rceil$, limiting each query size to $\mathcal{O}(n/k)$ for a total cost of $\tilde{\mathcal{O}}(n\cdot k\cdot n/k)=\tilde{\mathcal{O}}(n^2)$.

## Key Experimental Results

### Main Results (Summary of Theoretical Results)

| Task | Upper Bound | Lower Bound | Remarks |
|---|---|---|---|
| Basis / Rank Estimation (General) | $\mathcal{O}(n^2)$ (Greedy) | $\Omega(n^2)$ (Theorem 1.1.1) | Quadratic even with $1\pm 1/40$ approx. |
| Partition Number (General) | $\tilde{\mathcal{O}}(n^2)$ (Theorem 1.1.2) | $\Omega(n^2)$ (Distinguish $3$ vs $4$) | Quadratic for $(1+\epsilon)$-approx ($\epsilon<1/3$) |
| Max Weight Basis (Girth $\le c$) | $\mathcal{O}(n^{2-1/c}\log n)$ (Alg 1) | —— | First sub-quadratic result |
| General Cost $f(|Q|)$ (Rank) | —— | $\Omega(n\cdot f(n/3))$ (Theorem 1.2) | Simplifies to $\Omega(n\cdot f(n))$ if $f$ is poly |
| General Cost $f(|Q|)$ (Partition) | —— | $\Omega(n\cdot f(n/12))$ | Same as above |

### Ablation Study (Comparison of Applicable Models)

| Model Variant | Basis Complexity | Description |
|---|---|---|
| Classic $\mathcal{O}(1)$ Oracle | $\mathcal{O}(n)$ queries | Disconnected from this model; fails to reflect real runtime |
| Dynamic Oracle (Blikstad 2023) | Greedy can be sub-quadratic | Requires oracle state maintenance; differs from this stateless model |
| Ours (Size-Sensitive) | $\Theta(n^2)$ (Tight) | Naturally matches linear oracles like those for graphic matroids |
| Ours + Bounded Girth $c$ | $\mathcal{O}(n^{2-1/c}\log n)$ | Upper bound degrades to $\tilde{\mathcal{O}}(n^2)$ as $c\to\infty$, consistent with general case |

### Key Findings
- "Approximation does not save money" is a strong conclusion of this model: even a $1\pm 1/40$ rank approximation requires quadratic cost. This perfectly aligns with the actual algorithmic costs for spanning forest tasks on dense graphs.
- Bounded girth is a genuine structural assumption capable of breaking the quadratic barrier—it provides "circuit fingerprints" of size $\le c$ for non-basis elements, allowing sparse sampling to locate them efficiently.
- The general cost function lower bound $\Omega(n\cdot f(n))$ (for polynomial $f$) indicates that the conclusions are robust against various cost curves of oracle implementations.

## Highlights & Insights
- Shifting the oracle cost model from "counting" to "paying by size" is a seemingly small but profound perspective shift—it immediately subjects a large class of "$\mathcal{O}(n)$ query" algorithms to re-examination and aligns the theoretical analysis of general matroids with the actual runtime of special cases like graphic matroids.
- "Making all small queries uninformative" is a transferable template for constructing lower bounds: utilizing the union of free and uniform matroids to force independence in small sets, with truncation and witnesses providing the means of distinction. This construction can be generalized to other "pay-per-set-size" oracle complexity scenarios.
- The randomized sampling $n^{-1/c}$ in Algorithm 1 is meticulously tuned to ensure circuits of size $\le c$ are trapped with probability $\ge n^{-1}$. Combined with $n\ln n$ rounds, this clears non-basis elements with high probability. This "probabilistic circuit trapping" idea could inspire other sparse identification algorithms with local structures.

## Limitations & Future Work
- The lower bounds apply to the memoryless (stateless) model. The authors explicitly note that in a dynamic oracle setting (Blikstad 2023), greedy could be cheaper; thus, these conclusions do not directly extrapolate.
- The $\mathcal{O}(n^{2-1/c}\log n)$ result is specifically for maximum weight basis algorithms and does not provide a "unified framework for arbitrary matroid tasks under bounded girth."
- Neither the upper nor lower bounds consider caching mechanisms for "repeatedly querying the same set"; in real systems, such locality could significantly reduce effective costs.
- The paper lacks numerical experiments or comparisons on real matroid instances (e.g., dense graphs); it is purely theoretical.

## Related Work & Insights
- **vs Eberle et al. (2024) (Budgeted Oracle)**: They also focus on oracle costs but intervene from an "augmented oracle" perspective; this work redefines costs on the original oracle interface.
- **vs Blikstad et al. (2023) (Dynamic Oracle)**: The dynamic model allows oracles to maintain state, making greedy cheaper. The stateless model in this work is better suited for distributed or REST API scenarios.
- **vs Quanrud (2024) (Base Covering)**: This work directly ports Quanrud's $\tilde{\mathcal{O}}(nk)$ query complexity algorithm into the size-sensitive model. By truncating the rank to $\mathcal{O}(n/k)$, the cost per query is limited, yielding the partition number upper bound and cleverly repurposing existing results.

## Rating
- Novelty: ⭐⭐⭐⭐ Redefining the oracle cost model is a simple but long-overlooked perspective; the entire set of matching bounds is new.
- Experimental Thoroughness: ⭐⭐⭐⭐ As a theoretical paper, the matching bounds for three tasks (up to logarithmic factors) and the extension to general cost functions make it very comprehensive; no empirical results.
- Writing Quality: ⭐⭐⭐⭐ Clarity in definitions, lemmas, and theorems is excellent, and the intuitive explanations for lower bound constructions are well done, though some counting details are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Significant impact on the combinatorial optimization community: it provides a more realistic runtime benchmark for existing "query-based" matroid algorithms and initiates a new generation of size-sensitive complexity research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Data Manifold under the Microscope](the_data_manifold_under_the_microscope.md)
- [\[ICML 2026\] Sequential Kernel-based Conditional Independence Testing via Adaptive Betting](sequential_kernel-based_conditional_independence_testing_via_adaptive_betting.md)
- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)
- [\[ICML 2026\] Quantum Algorithms for Triangle Cut Sparsification](quantum_algorithms_for_triangle_cut_sparsification.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)

</div>

<!-- RELATED:END -->
