---
title: >-
  [Paper Note] A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport
description: >-
  [ICLR2026][Optimization][Optimal Transport] This paper provides the first truly quadratic-time constant-factor approximation algorithm for **all** $p \in [1, \infty]$ (including $p = \infty$): on any metric space, it computes a $(4+\varepsilon)$-approximation for $W_p$ optimal transport in $O(n^2+(n^{3/2}\varepsilon^{-1}\log n\log\Delta)^{1+o(1)}\log U)$ time, reducing the previous $O(\log n)$ approximation ratio to a constant.
tags:
  - "ICLR2026"
  - "Optimization"
  - "Optimal Transport"
  - "Approximation Algorithms"
  - "$W_p$ Distance"
  - "Directed Spanner"
  - "Bichromatic Closest Pair"
date: 2026-05-08
content_hash: 437e3b3ae8a62fa8
---

# A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=RPQKJxrEPs](https://openreview.net/forum?id=RPQKJxrEPs)  
**Code**: https://anonymous.4open.science/r/anon_matching-2B3B/ (Anonymous repository, proof-of-concept Python implementation)  
**Area**: optimization  
**Keywords**: Optimal Transport, Approximation Algorithms, $W_p$ Distance, Directed Spanner, Bichromatic Closest Pair

## TL;DR
This paper provides the first truly quadratic-time constant-factor approximation algorithm for **all** $p \in [1, \infty]$ (including $p = \infty$): on any metric space, it computes a $(4+\varepsilon)$-approximation for $W_p$ optimal transport in $O(n^2+(n^{3/2}\varepsilon^{-1}\log n\log\Delta)^{1+o(1)}\log U)$ time, reducing the previous $O(\log n)$ approximation ratio to a constant.

## Background & Motivation
**Background**: Given two discrete distributions $\mu, \nu$ supported on finite point sets $A, B \subseteq X$, the cost of moving mass $\delta$ from $a$ to $b$ is $\delta \, d(a,b)^p$. $W_p$ optimal transport aims to find a transport plan $\sigma$ that minimizes the total cost $w_p(\sigma)=\big(\sum_{a,b}\sigma(a,b)\,d(a,b)^p\big)^{1/p}$; as $p \to \infty$, this degrades to $w_\infty(\sigma)=\max_{\sigma(a,b)>0}d(a,b)$. Different $p$ values have various applications: $W_1$ measures total displacement (Word Mover's Distance, WMD), $W_2$ possesses monotonicity and translation invariance, and $W_\infty$ has recently been used for topological data analysis and convergence analysis of neural network layers.

**Limitations of Prior Work**: Exact solutions can be formulated as Minimum Cost Flow (MCF) and solved in $n^{2+o(1)}$ time using the interior point method by Chen et al. (2022), but the algorithm is extremely complex and impractical—even simpler problems like determining the existence of perfect matchings in dense graphs in quadratic time remain open challenges. Consequently, research has shifted toward approximations. However, existing approximations have significant drawbacks: Charikar (2002) achieved an $O(\log n)$-approximation by embedding metrics into Hierarchically Well-Separated Trees (HSTs); Sinkhorn entropy regularization (Cuturi 2013) only provides **additive** error $\varepsilon \cdot \mathrm{diam}$, and its running time deteriorates with $p$ ($n^2/\varepsilon^{O(1)}$ for $W_1$, becoming $n^2/\varepsilon^{O(p)}$ for $W_p$), and it **cannot handle $p=\infty$ at all**; Lahn et al. (2025), the closest work to this paper, provides an $O(\log n)$-relative approximation for any metric in $O(n^2\log U\log\Delta\log n)$ time, but the approximation ratio remains logarithmic.

**Key Challenge**: It is difficult to simultaneously balance the approximation ratio, running time, and the range of $p$. Either the ratio is good but only holds for Euclidean/low-dimensional spaces (failing to generalize to $p \ge 2$), or the method is general but has an $O(\log n)$ ratio, or it fails to handle $p=\infty$. The root cause is that when $p>1$, the transport cost $d(\cdot,\cdot)^p$ **is no longer a metric** (violating the triangle inequality), rendering classic undirected metric spanners and distance oracles invalid.

**Goal**: To reduce the approximation ratio from $O(\log n)$ to a constant under the three constraints of **any metric**, **all $p \in [1, \infty]$**, and **truly quadratic time**.

**Key Insight**: The authors found that instead of approximating the non-metric $d^p$, it is better to first approximate the underlying metric $d$ with high precision in sub-quadratic space and then "lift" this approximation to $d^p$. This requires a new approximation structure capable of tolerating power-of-$p$ amplification.

**Core Idea**: Construct a **two-level clustering** using Bourgain multi-scale sampling to derive a $(4+\varepsilon)$-approximate proxy distance $d_C$; then weave this into a **directed spanner** to approximate $d^p$ (directionality bypasses the non-metric obstacle of $d^p$); finally, all algorithms access data only through the "Bichromatic Closest Pair (BCP)" primitive without explicitly calculating all-pairs distances, ensuring both speed and implementability.

## Method

### Overall Architecture
The method consists of three layers and results in two sets of algorithms. The first layer is **distance approximation**: constructing a two-level clustering $\mathcal{C}$ on the point set $P=A \cup B$ and defining a proxy distance $d_C(x,y)$. It is proven that the true metric is sandwiched as $d \le d_C \le (4+\varepsilon)d$, requiring only $\sim n^{3/2}$ storage instead of $O(n^2)$. The second layer is **structuralization**: weaving the clusters into a directed graph (spanner) such that the shortest path from $B$ to $A$ approximates $d^p$, lifting the "approximate $d$" to "approximate $d^p$". The third layer is **solving**: running algorithms on this sparse directed graph—either using the MCF interior point method of Chen et al. (2022) (theoretically optimal, proving Theorem 1.1, but not implementable) or a set of combinatorial algorithms relying solely on the BCP primitive (implementable, proving Theorems 1.2/1.3).

The key constraints of the pipeline are $\Delta$ (point set spread, the ratio of maximum to minimum non-zero distance) and $U$ (ratio of maximum to minimum probability); $\log\Delta$ and $\log U$ appear throughout the running time. Final conclusion: the MCF version gives a $(4+\varepsilon)$-approximation in $O(n^2+(n^{3/2}\varepsilon^{-1}\log n\log\Delta)^{1+o(1)}\log U)$ time for all $p$ (including $\infty$); under the query model, after one preprocessing of $X$, $O(k)$-approximate queries for any distribution pairs on $X$ can be answered in sub-quadratic time. The authors also provide a conditional lower bound: for $p = \infty$, if a relative factor better than $2$ could be achieved within $O(n^2)$, it would solve the perfect matching problem for any bipartite graph in quadratic time—indicating that the $(4+\varepsilon)$ constant is near the boundary of hardness.

### Key Designs

**1. Two-level Clustering + Proxy Distance: Sandwiching $(4+\varepsilon)$ Metric Approximation with $n^{3/2}$ Space**

The pain point is that quadratic time prohibits storing a $O(n^2)$ distance matrix, yet accurate distances must be accessible. The authors use Bourgain multi-scale sampling to construct clusters. Let $P_0=P$, and sample each point into $P_1$ independently with probability $n^{-1/2}$, yielding expected $|P_1|=\sqrt{n}$. Take a geometric sequence of radii $r_i=(1+\tfrac{\varepsilon}{4})^i$ for $0 \le i \le t = \lceil\log_{1+\varepsilon/4}\Delta\rceil$. Two types of clusters are generated: (i) for each unsampled point $q \in P_0 \setminus P_1$, within its Voronoi set $V(q,P_1)=\{y:d(y,q)<d(y,P_1)\}$, cut clusters $C_q[i]=\{x \in V(q,P_1):d(x,q) \le r_i\}$; (ii) for each sampled point $q \in P_1$, cut clusters $C_q[i]=\{x \in P_0:d(x,q) \le r_i\}$.

The proxy distance is defined elegantly: $d_C(x,y)=2r_i$, where $i$ is the **minimum index of a cluster containing both $x$ and $y$**. Lemma 2.2 proves $d(x,y) \le d_C(x,y) \le (4+\varepsilon)d(x,y)$ by case analysis: if $x$ is "separated" by some point $a \in P_1$ (there exists $a \in P_1$ such that $d(x,a) < d(x,y)$), then by triangle inequality $d(y,a) \le 2d(x,y)$, so a type-(ii) cluster $C_a[i]$ contains both $x,y$ with $2r_i \le 2(1+\tfrac{\varepsilon}{4})d(y,a) \le 4(1+\tfrac{\varepsilon}{4})d(x,y)$; otherwise $x \in V(y,P_1)$, and a type-(i) cluster $C_y[i]$ gives $2r_i \le 2(1+\tfrac{\varepsilon}{4})d(x,y)$. Crucial cost control comes from **Degree Lemma 2.1**: the number of clusters each point participates in is $\deg_{\mathcal{C}}(p)=O(\sqrt{n}\,\varepsilon^{-1}\log n\log\Delta)$ with probability $\ge 1-1/n$. Total space is $O(n^{3/2}\varepsilon^{-1}\log\Delta)$ and construction time is $O(n^2)$. This is the foundation: trading off for a constant-factor distance approximation in sub-quadratic space.

**2. Directed Spanner: Bypassing Non-metric $d^p$ and Lifting Metric Approx to Cost Approx**

When $p>1$, $d^p$ violates the triangle inequality, making classic undirected spanners unusable. The authors' approach is to orient every edge. For each cluster $C$ (index $i$), two new Steiner vertices $a_C, b_C$ are added with three types of edges: $a_C \to b_C$ weight $(2r_i)^p$; zero-weight edges $a \to a_C$ for $a \in A \cap C$; and zero-weight edges $b_C \to b$ for $b \in B \cap C$. This graph might not be strongly connected but guarantees a path from any $b \in B$ to any $a \in A$. Lemma 2.3 proves: the shortest path distance in the graph satisfies $d^p(a,b) \le d_{G,p}(a,b) \le (4+\varepsilon)^p d^p(a,b)$—lifting the $(4+\varepsilon)$ from Lemma 2.2 to the $p$-th power. The differences from classic spanners are why it is fast and simple: first, inserting carefully chosen **Steiner points** at multiple scales significantly simplifies the structure and reduces construction time to $O(n^2+kn^{1+1/k}\log\Delta)$ (classic spanners require super-quadratic preprocessing); second, **directionality** specifically expresses the asymmetric transport of $d^p$. The graph size is $|V|=O(n\varepsilon^{-1}\log\Delta)$, $|E|=O(n^{3/2}\varepsilon^{-1}\log n\log\Delta)$, sparse enough to feed directly into MCF: add super source/sink, set capacities by $\mu(a), \nu(b)$, and run Chen et al. (2022) to get min-cost flow in $(|E|)^{1+o(1)}\log U$ time, then extract the plan $\sigma$ from flow paths (Theorem 1.1). For $p=\infty$, this is modified to **binary search + a sequence of max flows** over $O(\varepsilon^{-1}\log\Delta)$ radius values: verifying if flow is full on subgraph $G_i$ (excluding all edges with cost $>2r_i$) to find the minimum feasible $i^*$, which is the key to extending quadratic-time algorithms to $p=\infty$.

**3. BCP Primitive + Combinatorial Algorithm: Never Explicitly Calculate Distances, Only Query "Weighted BCP"**

The MCF interior point method is theoretically optimal but not implementable. The authors created an implementable combinatorial algorithm that encapsulates the expensive "all-pairs distance" operation into a **dynamic weighted Bichromatic Closest Pair (BCP)** primitive: given weighted distance $d_w(a,b)=d_C^p(a,b)-w(a)-w(b)$, maintain $\mathrm{BCP}_w(A,B)=\arg\min d_w$ under insertions/deletions of $A, B$. The data structure reuses the same clusters: each cluster maintains a max-heap for $A \cap C$ and $B \cap C$ sorted by weight; take roots $a_C, b_C$, calculate $\phi_C=(2r_i)^p-w(a_C)-w(b_C)$, and use a global min-heap $H$ with $\phi_C$ as the key; the root of $H$ is the global BCP (Lemma 2.4). Updating a point involves $\deg_{\mathcal{C}}(q)$ clusters, taking $O(\deg_{\mathcal{C}}(q)\log n)$ time. With BCP, the Gabow–Tarjan cost scaling/Hungarian search can be applied: Hungarian search is essentially Dijkstra using slack as edge weight; each $\arg\min(s(a',b')+\ell_{b'})$ corresponds to one weighted BCP query (Appendix D). OT requires $O(n\log U)$ Hungarian searches, each $O(\sum_q\deg_{\mathcal{C}}(q)\log n)=O(kn^{1+1/k}\log^2 n)$, yielding Theorem 1.2's $O(n^{2+1/k}\log^2 n\log\Delta\log U)$; for equal-mass $W_p$-**matching**, search count drops to $O(\sqrt n)$, yielding Theorem 1.3's $O(n^2\varepsilon^{-3/2}\log^2 n\log\Delta)$. The algorithm **never explicitly calculates any $d(a,b)$**, only calling BCP—the reason it remains scalable in high dimensions (e.g., MNIST 784-dim).

## Key Experimental Results

### Main Results
Proof-of-concept Python implementation on an 8-core Apple M1 / 16GB. Data: uniform and truncated normal distributions on unit cubes ($\le 10$ dimensions) plus MNIST (784 dimensions). Baseline: HST proxy distance from Lahn et al. (2025). Both use the same primal-dual framework, differing only in the proxy distance.

| Comparison ($p=2$, Uniform) | Ours (cluster-dist) | Lahn 2025 (HST-dist) | Conclusion |
|--------|------|----------|------|
| Max distance distortion | Always $\le (4+\varepsilon)$ | Significantly larger | Ours has theoretical bound and is tighter |
| Avg distance distortion | Near $2$ | Significantly larger | Tested better than worst-case bound |
| Opt matching cost ratio | Baseline | $> 4.5 \times$ larger | cluster-dist error is an order of magnitude smaller |

### Approximation Ratio and Efficiency

| Configuration | Observed Value | Description |
|------|---------|------|
| Approximation ratio ($p \in \{1,2,3,4,5,\infty\}$, Uniform/Normal/MNIST) | Typical $1.5 \sim 2$ | Always within $(4+\varepsilon)$ and stable across $p$ |
| Avg degree ($d \le 10$) | Close to Lemma 2.1 bound | Efficient two-level clustering space, stable across settings |
| BCP query count (Dominant time) | Growth scales as predicted; nearly invariant to $p$ | Combined with $\tilde O(n^{1/2})$ per query, overall empirical quadratic time |

### Key Findings
- **Worst-case bound is loose**: Theoretical guarantee is $(4+\varepsilon)$, but empirical ratios are often $1.5 \sim 2$, with avg distortion near $2$; shows worst-case analysis is pessimistic and engineering performance is much better.
- **Insensitive to $p$**: Approximation ratio and BCP query count remain almost constant from $p=1$ to $p=\infty$, validating the "unified $p$ handling" promise.
- **Proxy distance quality is the key variable**: Within the same solver framework, simply replacing HST-dist with cluster-dist reduces the optimal matching cost by over $4.5\times$—quantifying the benefit of a more accurate proxy distance.

## Highlights & Insights
- **The reordering "approximate metric first, then lift to $d^p$" is clever**: Directly approximating non-metric $d^p$ is rejected by classic spanner tools; the authors take a step back to approximate $d$ with high precision using clusters (Lemma 2.2) and then lift the $(4+\varepsilon)$ by $p$-th power via directed spanner (Lemma 2.3), solving "general $p$" in one go.
- **Directionality is key to bypassing non-metric obstacles**: $d^p$ lacks triangle inequality, but as long as $B \to A$ is directed-reachable and weights encode $(2r_i)^p$, shortest paths approximate $d^p$ without requiring strong connectivity—a reusable idea for generalizing metric spanners to non-metric costs.
- **Encapsulating "expensive distance calculation" into the BCP primitive**: The algorithm accesses data only through BCP and never calculates $d(a,b)$, making it scalable in high dimensions (MNIST 784-dim), which is instructive for any geometric algorithm where distances are expensive but only nearest pairs are needed.
- **First quadratic-time method for $p=\infty$**: Binary search on radius + a series of max flows fills the gap left by additive methods like Sinkhorn.
- **Tightness of conditional lower bound**: Theorem 1.4 shows that for $p=\infty$, pushing the relative factor below $2$ within $O(n^2)$ would solve "perfect matching in bipartite graphs in quadratic time," suggesting $(4+\varepsilon)$ is near the complexity ceiling.

## Limitations & Future Work
- **Constant factor gap**: $(4+\varepsilon)$ is constant but not $(1+\varepsilon)$; the lower bound only blocks factors below $2$ for $p=\infty$. Whether $2 \sim 4$ can be improved or whether smaller constants for finite $p$ are possible remains unsolved.
- **Theoretical optimal algorithm is not implementable**: Theorem 1.1 relies on the interior point method by Chen et al. (2022), which has no practical implementation; the implementable combinatorial algorithms in Theorem 1.2/1.3 have $1/k$ powers and multiple $\log$ factors, and their constant overhead might not be faster than optimized Sinkhorn implementations for medium scales.
- **Dependence on spread $\Delta$ and probability ratio $U$**: Running time includes $\log\Delta \cdot \log U$, which may expand when point scales vary greatly or probabilities are highly imbalanced; discrete mapping is needed for continuous distributions.
- **Randomization and high probability guarantees**: Degree bounds and overall complexity are Las Vegas results holding with probability $\ge 1-1/n$; in the worst case, single-point degrees could still be $\Theta(n)$.

## Related Work & Insights
- **vs Lahn et al. (2025)**: Both use a general metric and primal-dual framework, but their HST proxy distance has $O(\log n)$ distortion and excludes $p=\infty$; this paper uses clustering to reduce the ratio to constant $(4+\varepsilon)$, handles all $p$ uniformly, and shows $>4.5\times$ better proxy distance quality.
- **vs Sinkhorn / Entropy Regularization (Cuturi 2013 et seq.)**: Sinkhorn gives additive error $\varepsilon \cdot \mathrm{diam}$, its time worsens with $p$ ($n^2/\varepsilon^{O(p)}$), and it cannot handle $p=\infty$; this paper gives relative approximation, is insensitive to $p$, and supports $W_\infty$.
- **vs Euclidean $(1+\varepsilon)$ Approximations (Agarwal et al. 2022/2024, Andoni–Zhang 2023, etc.)**: Those methods have better ratios but are restricted to Euclidean/fixed dimensions and hard to generalize to $p \ge 2$; this paper sacrifices the factor to constant for the sake of any metric and general $p$.
- **vs Classic Spanners / Distance Oracles (Thorup–Zwick, Baswana–Sen, etc.)**: Classic constructions only approximate metrics, require super-quadratic preprocessing, and cannot represent non-metric $d^p$; this paper uses multi-scale Steiner points + directionality to achieve sub-quadratic preprocessing and support dynamic BCP queries, representing a tailored modification of this toolchain.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First constant-factor quadratic-time for all $p$ (incl. $\infty$); "metric approx first + directed spanner + BCP primitive" is highly original.
- Experimental Thoroughness: ⭐⭐⭐ Validates theoretical bounds and proxy distance advantages, but lacks wall-clock comparison with optimized implementations due to proof-of-concept Python.
- Writing Quality: ⭐⭐⭐⭐ Theorems are clear; proof logic (Lemmas 2.1/2.2/2.3) is well-organized and progressive.
- Value: ⭐⭐⭐⭐ Reduces OT approximation ratio from log to constant and enables $p=\infty$, particularly useful for query scenarios with shared supports like WMD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems](a_memory-efficient_hierarchical_algorithm_for_large-scale_optimal_transport_prob.md)
- [\[ICLR 2026\] HOTA: Hamiltonian Framework for Optimal Transport Advection](hota_hamiltonian_framework_for_optimal_transport_advection.md)
- [\[ICLR 2026\] Neural Hamilton–Jacobi Characteristic Flows for Optimal Transport](neural_hamilton--jacobi_characteristic_flows_for_optimal_transport.md)
- [\[ICLR 2026\] Neural Optimal Transport Meets Multivariate Conformal Prediction](neural_optimal_transport_meets_multivariate_conformal_prediction.md)
- [\[ICLR 2026\] Hyperparameter Trajectory Inference with Conditional Lagrangian Optimal Transport](hyperparameter_trajectory_inference_with_conditional_lagrangian_optimal_transpor.md)

</div>

<!-- RELATED:END -->
