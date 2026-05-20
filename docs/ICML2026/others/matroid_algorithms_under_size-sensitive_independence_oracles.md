---
title: >-
  [Paper Note] Matroid Algorithms Under Size-Sensitive Independence Oracles
description: >-
  [ICML 2026][Matroid] The authors propose a size-sensitive matroid oracle model where the query cost grows linearly with the size of the queried set, and prove that under this model…
tags:
  - "ICML 2026"
  - "Matroid"
  - "independence oracle"
  - "size-sensitive query cost"
  - "lower bound"
  - "bounded circuit size"
date: 2026-05-08
content_hash: 69a47229e962edf6
---

# Matroid Algorithms Under Size-Sensitive Independence Oracles

**Conference**: ICML 2026  
**arXiv**: [2605.00201](https://arxiv.org/abs/2605.00201)  
**Code**: None (theoretical paper)  
**Area**: Algorithm Theory / Combinatorial Optimization  
**Keywords**: Matroid, independence oracle, size-sensitive query cost, lower bound, bounded circuit size

## TL;DR
The authors propose a size-sensitive matroid oracle model where the query cost grows linearly with the size of the queried set, and prove that under this model, the optimal query cost for finding a basis, estimating rank, and estimating the partition number is all $\tilde{\Theta}(n^2)$. For matroids with bounded circuit size $c$, they provide an $\mathcal{O}(n^{2-1/c}\log n)$ algorithm for maximum weight basis, breaking the quadratic lower bound.

## Background & Motivation

**Background**: Matroids are a core abstraction for "subset selection under constraints" in combinatorial optimization, widely used in ML for bandit/online learning feasibility constraints, submodular maximization, preference-guided allocation, etc. Algorithm analysis almost universally adopts the "independence oracle" model: given a set $Q\subseteq E$, the oracle answers whether $Q\in\mathcal{I}$ in $\mathcal{O}(1)$ time, and the literature measures complexity by "number of queries".

**Limitations of Prior Work**: The constant-time oracle is unrealistic in practice. For example, in graphic matroids, checking whether a set of edges forms a forest requires $\Theta(|Q|)$ work (union-find/DFS); other "natural" matroid classes (bicircular, transversal, scheduling) also have nearly linear, not constant, oracle costs. This means published "$\mathcal{O}(n)$ query" algorithms may actually take $\mathcal{O}(n^2)$ real time, creating a serious disconnect between theory and practice.

**Key Challenge**: To make analysis practically relevant, the oracle cost must be explicitly modeled as a function of $|Q|$; but this immediately invalidates classic "query count" lower bounds—large queries are more expensive than small ones, so algorithms may use many small queries to save total cost. New matching upper/lower bounds are needed.

**Goal**: Analyze three fundamental matroid tasks under the size-sensitive model (querying $Q$ costs $|Q|$): (i) finding a basis; (ii) approximate rank; (iii) compute/approximate partition number $k(M)$; also consider general non-decreasing cost functions $f(|Q|)$.

**Key Insight**: The authors observe that "greedy algorithms" naturally cost $\mathcal{O}(n^2)$ in this model, so the question becomes "can smarter query strategies break quadratic cost?" They construct a family of matroid instances where all small queries are "automatically yes"—so any informative query must be large ($\Theta(n)$), directly pushing the cost to quadratic.

**Core Idea**: Construct hard instances using "free matroid + uniform matroid union + truncation" (for rank tasks) and "partition matroid + $\ell$-relaxation + truncation" (for partition tasks), and use Yao's principle to convert deterministic decision tree lower bounds to randomized ones; the upper bound uses existing base-covering algorithms adapted to truncation.

## Method

### Overall Architecture
The paper has two main threads. Lower bound thread: (1) define the size-sensitive oracle; (2) construct hard instance distribution $\mathcal{D}_{m,\epsilon}$; (3) argue that "witnesses required to distinguish instances must be large" and use counting arguments to show any decision tree needs $\Omega(m)$ large queries, each costing $\Omega(m)$, so total cost is $\Omega(n^2)$; (4) upgrade to randomized algorithms via Yao's principle. Upper bound thread: (a) for partition number, use Quanrud (2024)'s base-cover + truncation to rank $\lceil n/k\rceil$, yielding $\tilde{\mathcal{O}}(n^2)$; (b) for maximum weight basis with bounded circuit size $c$, use random subsampling + binary search to locate the "minimum weight circuit element" for a sub-quadratic algorithm.

### Key Designs

1. **Hard Instance Construction with "Small Queries Give No Information"**:

    - **Function**: Ensures any query of size at most $m$ is automatically judged independent, forcing the algorithm to use large queries.
    - **Mechanism**: For the rank task, fix $n=3m$, pick a subset $S\subseteq[3m]$ of size $m$, define $M_{m,S}$ as the union of the "free matroid on $S$" and the "uniform matroid of rank $m$ on $T=[3m]\setminus S$"; its rank is $2m$, and any set of size $\le m$ is independent (Lemma 4.2). Truncate it to rank $2m-\epsilon m$ to get $M'_{m,S,\epsilon}$. To distinguish $M_{m,S}$ from $M'_{m,S,\epsilon}$, one must find a witness $W$ that is independent in the original but dependent after truncation, with $|W|>2m-\epsilon m$ and $|W\setminus S|\le m$. For the partition task, similarly use "partition into $m$ blocks of size $\alpha+1$" + $\ell=m/\alpha$-relaxation + rank minus 1 truncation, so any query of size $\le m/\alpha$ is automatically independent.
    - **Design Motivation**: The core of the oracle model is "how many instances can a query distinguish"; designing all low-cost queries to be "indistinguishable yes" blocks all cheap algorithmic approaches.

2. **Witness Counting + Yao's Principle for Randomized Lower Bound**:

    - **Function**: Translates "how many large queries a deterministic decision tree needs" into a lower bound for randomized algorithms.
    - **Mechanism**: Fix a witness $W$; the number of $S$ making $W$ a witness is strictly upper bounded by binomial coefficients (Lemma 4.5: at most $\binom{2m-\delta m}{m-\delta m}\binom{2m+\delta m}{\delta m}$). A decision tree of depth $q$ explores at most $2^{q+1}$ candidate sets, so its success probability under uniform distribution $\mathcal{D}_{m,\epsilon}$ is bounded by $\frac{1}{2}+\frac{2^q\cdot\binom{2m}{m}\binom{2m+\epsilon m}{2m}}{\binom{3m}{m}}$; to boost success from $1/2$ to $2/3$ requires $q=\Omega(m)$. Yao's principle then gives a randomized lower bound of $\Omega(m^2)=\Omega(n^2)$ on the worst-case instance.
    - **Design Motivation**: This "hard distribution + witness counting + decision tree exponent + Yao" pipeline is standard for combinatorial lower bounds, but this is the first successful application to basic matroid tasks in the size-sensitive model.

3. **Randomized Basis Algorithm Breaking Quadratic Bound for Bounded Circuit Size**:

    - **Function**: When all circuits have size $\le c$, finds a maximum weight basis in expected cost $\mathcal{O}(n^{2-1/c}\log n)$.
    - **Mechanism**: Algorithm 1 works in reverse: start with $B\leftarrow E$, run $n\ln n$ rounds; in each round, independently include each element in $S$ with probability $n^{-1/c}$; if $S$ is dependent, sort by decreasing weight, binary search for the last element in the minimal dependent prefix, and remove it from both $B$ and $S$ (it must be the minimum-weight element in some circuit, i.e., a non-basis element). For each $d\notin B^*$, its fundamental circuit $C_d$ has size $\le c$, and the probability all of $C_d$ falls into $S$ is at least $(n^{-1/c})^c=n^{-1}$, so the chance $d$ survives $n\ln n$ rounds is at most $1/n$, and the expected number of remaining non-basis elements is 1. Each round, $|S|$ has expected size $n^{1-1/c}$, binary search takes $\mathcal{O}(\log n)$ queries, so total cost is $\mathcal{O}(n^{2-1/c}\log n)$.
    - **Design Motivation**: The root of the quadratic lower bound is that a single non-basis element may require a large dependent set to identify, but bounded circuit size ensures each non-basis element has a "circuit fingerprint" of size at most $c$, so sparse sampling can "trap" the entire circuit with high probability, replacing expensive large queries with many small ones.

### Loss & Training
Purely theoretical paper, no training. Lower bounds use Yao's principle + decision tree arguments; main upper bound algorithm is a randomized sketch with binary search (Algorithm 1). The partition number upper bound applies Quanrud (2024)'s $\tilde{\mathcal{O}}(nk)$ query algorithm to the matroid truncated to rank $\lceil n/k\rceil$, limiting each query size to $\mathcal{O}(n/k)$, so total cost is $\tilde{\mathcal{O}}(n\cdot k\cdot n/k)=\tilde{\mathcal{O}}(n^2)$.

## Key Experimental Results

### Main Results (Summary of Theoretical Results)

| Task | Upper Bound | Lower Bound | Remarks |
|---|---|---|---|
| Find basis / estimate rank (general matroid) | $\mathcal{O}(n^2)$ (greedy) | $\Omega(n^2)$ (Theorem 1.1.1) | Even $1\pm 1/40$ approximation still quadratic |
| Partition number (general matroid) | $\tilde{\mathcal{O}}(n^2)$ (Theorem 1.1.2 upper bound) | $\Omega(n^2)$ (distinguishing $3$ vs $4$) | $(1+\epsilon)$-approximation ($\epsilon<1/3$) also quadratic |
| Maximum weight basis (circuit size $\le c$) | $\mathcal{O}(n^{2-1/c}\log n)$ (Algorithm 1) | —— | First sub-quadratic result |
| General cost $f(|Q|)$ (rank) | —— | $\Omega(n\cdot f(n/3))$ (Theorem 1.2) | If $f$ is polynomial, simplifies to $\Omega(n\cdot f(n))$ |
| General cost $f(|Q|)$ (partition) | —— | $\Omega(n\cdot f(n/12))$ | Same as above |

### Ablation Study (Comparison of Applicable Models)

| Model Variant | Basis Finding Complexity | Notes |
|---|---|---|
| Classic $\mathcal{O}(1)$ oracle | $\mathcal{O}(n)$ queries | Disconnects from this model, cannot reflect real runtime |
| Dynamic oracle (Blikstad 2023) | Greedy can be sub-quadratic | Requires oracle to maintain state, different from this stateless model |
| This paper's size-sensitive | $\Theta(n^2)$ (tight bound) | Naturally matches graphic matroid and other linear oracles |
| This paper + bounded circuit size $c$ | $\mathcal{O}(n^{2-1/c}\log n)$ | Upper bound degenerates to $\tilde{\mathcal{O}}(n^2)$ as $c\to\infty$, matching the general case |

### Key Findings
- "Approximation does not save cost" is a strong conclusion of this model: even for $1\pm 1/40$ rank approximation, quadratic cost is still required; this matches the actual algorithmic cost for spanning forest tasks on dense graphs.
- Bounded circuit size is one of the few structural assumptions that truly break quadratic cost—it provides a "circuit fingerprint" of size $\le c$ for non-basis elements, enabling efficient localization via sparse sampling.
- The general cost function lower bound $\Omega(n\cdot f(n))$ (for polynomial $f$) shows the results are robust to various oracle implementation cost curves.

## Highlights & Insights
- Shifting the oracle cost model from "counting" to "pay by size" is a seemingly small but far-reaching perspective shift—it immediately forces a re-examination of many "$\mathcal{O}(n)$ query" algorithms, and aligns real runtime for graphic matroids and other special cases with general matroid theory.
- "Making all small queries uninformative" is a transferable lower bound construction template: free matroid + uniform matroid union forces small sets to be independent, truncation + witness is the distinguishing tool; the same approach applies to partition tasks. This construction can be generalized to other "pay by set size" oracle complexity scenarios.
- The randomized sampling $n^{-1/c}$ in Algorithm 1 is carefully tuned—just enough so that circuits of size $\le c$ are "trapped" with probability $\ge n^{-1}$, and $n\ln n$ rounds clear non-basis elements with high probability. This "probabilistic circuit trapping" idea may inspire other sparse recognition algorithms with local structure.

## Limitations & Future Work
- The lower bound is for the stateless (memoryless) model; the authors note that in the dynamic oracle setting (Blikstad 2023), greedy can be cheaper, so these results do not directly extend there.
- The $\mathcal{O}(n^{2-1/c}\log n)$ result is only for the maximum weight basis algorithm; no unified framework is given for arbitrary matroid tasks under bounded circuit size.
- Both upper and lower bounds do not consider caching mechanisms for repeated queries to the same set; in real systems, such locality could greatly reduce effective cost.
- The paper does not provide numerical experiments or comparisons on real matroid instances (e.g., dense graphs), being purely theoretical.

## Related Work & Insights
- **vs Eberle et al. (2024) (budgeted oracle)**: They also focus on oracle cost, but approach from an "augmented oracle" perspective; this paper redefines cost within the original oracle interface.
- **vs Blikstad et al. (2023) (dynamic oracle)**: The dynamic model allows the oracle to maintain state, making greedy cheaper; this paper's stateless model is more suitable for distributed/REST API scenarios.
- **vs Quanrud (2024) (base covering)**: This paper directly adapts their $\tilde{\mathcal{O}}(nk)$ query algorithm to the size-sensitive model, using truncation to cap each query size at $\mathcal{O}(n/k)$ for the partition number upper bound, cleverly reusing existing results.

## Rating
- Novelty: ⭐⭐⭐⭐ Redefining the oracle cost model is a simple but long-overlooked perspective; the entire set of upper/lower bounds is new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Theoretical paper, upper and lower bounds for three tasks match up to logarithmic factors, and extend to general cost functions, covering the topic comprehensively; no empirical results.
- Writing Quality: ⭐⭐⭐⭐ Definitions, lemmas, and theorems are clearly structured; intuitive explanations for lower bound constructions are well done, though some counting arguments are squeezed into the appendix.
- Value: ⭐⭐⭐⭐ High impact for the combinatorial optimization theory community: brings existing "query-charged" matroid algorithm analyses closer to real runtime, and opens the door for a new generation of size-sensitive complexity research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics](local_and_mixing-based_algorithms_for_gaussian_graphical_model_selection_from_gl.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[ICML 2025\] Softmax is not Enough (for Sharp Size Generalisation)](../../ICML2025/others/softmax_is_not_enough_for_sharp_size_generalisation.md)
- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](../../AAAI2026/others/parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)

</div>

<!-- RELATED:END -->
