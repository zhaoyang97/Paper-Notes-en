---
title: >-
  [Paper Note] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering
description: >-
  [ICML 2026][learning_theory][Correlation Clustering] This paper provides two simple 2-approximation algorithms for the "Bad Triangle Transversal" (BTT) problem on signed graphs that require only a single LP solve. It proves a unified NP-hard inapproximability bound of $\tfrac{2137}{2136}$ for BTT, Correlation Clustering (CC), MinSTC, and Cluster Deletion on complete grap
tags:
  - ICML 2026
  - learning_theory
  - Correlation Clustering
  - LP rounding
date: 2026-05-08
content_hash: 57ae3fd5023825d1
---
# Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2602.04463](https://arxiv.org/abs/2602.04463)  
**Code**: None  
**Area**: Algorithm Theory / Approximation Algorithms / Graph Clustering  
**Keywords**: Bad Triangle Transversal, Correlation Clustering, Approximation Algorithms, LP rounding, Hardness proof  

## TL;DR
This paper provides two simple 2-approximation algorithms for the "Bad Triangle Transversal" (BTT) problem on signed graphs that require only a single LP solve. It proves a unified NP-hard inapproximability bound of $\tfrac{2137}{2136}$ for BTT, Correlation Clustering (CC), MinSTC, and Cluster Deletion on complete graphs. Additionally, it constructs a new pivot procedure to convert any feasible BTT cover into a clustering with at most $\tfrac{3}{2}|F|$ errors, tightening the gap between BTT and CC optima from 2 to $3/2$.

## Background & Motivation
**Background**: Signed graphs $G=(V, E^+, E^-)$ are widely used in social networks, Ising models, and clustering. A "bad triangle" is a triangle with exactly one negative edge—the minimal unbalanced unit in structural balance theory. Correlation Clustering (CC) aims to partition nodes into clusters minimizing the sum of inter-cluster positive edges and intra-cluster negative edges. Bansal et al. initially used "packing of disjoint bad triangles" to achieve a constant-ratio approximation for CC, followed by Ailon's pivot as a 3-approximation benchmark.

**Limitations of Prior Work**: (1) BTT is a lower bound for CC. On large-scale graphs where LP is difficult to solve, researchers seek fast BTT covers to "convert" into CC clusters. However, known BTT algorithms are either 3-approximations (taking all edges of a maximal disjoint set) or use Krivelevich's (1995) 2-approximation for unsigned triangle covers—the latter requires solving LPs $\mathcal{O}(m)$ times with a bottleneck of $\widetilde{\mathcal{O}}(m^{\alpha+1})$, making it impractical for large graphs. (2) Whether BTT can strictly outperform a 2-approximation on complete graphs or has an explicit hardness lower bound remained unexplored. (3) The existing MatchFlipPivot (Veldt, 2022) for converting BTT covers to CC clusters only guarantees $\text{OPT}_{CC} \le 2\,\text{OPT}_\Delta$, causing the CC approximation ratio to degrade from $\alpha$ to $2\alpha$.

**Key Challenge**: BTT on a 3-uniform hypergraph is equivalent to a vertex cover with "bipartite" constraints (each hyperedge contains exactly one negative edge node). It is neither a standard VC nor a $k$-partite VC, so Lovász's (1975) randomized VC algorithm cannot be directly applied. Achieving an approximation ratio better than 3 requires leveraging structural information from the LP relaxation $\text{LP}_\Delta$.

**Goal**: (1) Develop a 2-approximation that rounds the solution of $\text{LP}_\Delta$ only once without repeated solving. (2) Prove a unified hardness lower bound for BTT/CC/MinSTC/CD on complete graphs. (3) Improve the cover-to-cluster conversion ratio from 2 to $3/2$, thereby improving the CC approximation ratio from $6$ to $3+\epsilon$.

**Key Insight**: The authors observe that the constraint structure of $\text{LP}_\Delta$ possesses "bipartite" properties where each bad triangle contains exactly one negative edge. By setting different rounding thresholds for "positive" and "negative" edges—e.g., $x_e \ge 1/2$ for positive and $x_e > 0$ for negative (or using a random threshold $r$ with $x_e > 1-r$ vs $x_e \ge r/2$)—the bipartite constraint can be utilized to achieve a 2-approximation in a single rounding round.

**Core Idea**: Replace Krivelevich's "iterative rounding" with "one-pass LP rounding with asymmetric thresholds"; replace Veldt's "flip-then-pivot" with a "categorized budget attack and modified pivot probabilities."

## Method

### Overall Architecture
The paper presents four interconnected contributions regarding BTT: two 2-approximation covering algorithms that **solve the LP only once** (Algorithm 2 deterministic, Algorithm 3 randomized), followed by an **improved pivot** to transform covers into clusterings (Algorithm 4, tightening the ratio from 2 to $3/2$), and finally a **unified hardness lower bound on complete graphs** (using a 2SAT gadget for BTT/CC/MinSTC/CD). The first two address "speed and accuracy," the third addresses "converting covers to CC with minimal loss," and the fourth defines the "theoretical limits."

### Key Designs

**1. Simple Deterministic 2-approximation (Algorithm 2): Replacing Iterative Solves**

Existing 2-approximations (Krivelevich 1995) require solving the LP $\mathcal{O}(m)$ times, presenting a bottleneck of $\widetilde{\mathcal{O}}(m^{\alpha+1})$. The key observation is that BTT on a 3-uniform hypergraph has a bipartite structure where each bad triangle contains exactly one negative edge. Thus, positive and negative edges can be handled in one round using distinct thresholds: solve $\text{LP}_\Delta$ once to get fractional solutions $\{x_e\}$, and output $E^-_{>0}\cup E^+_{\ge 1/2}$—all negative edges with non-zero values and positive edges $\ge 1/2$. Correctness follows from complementary slackness: the dual Packing LP has tight constraints only on edges where $x_e > 0$. Combined with bipartiteness, the total edges are $\le 2\cdot\text{LP}_\Delta\le 2\cdot\text{OPT}_\Delta$. By solving the LP once, time is dominated by the LP solver ($\widetilde{\mathcal{O}}(m^\alpha)$), roughly $m$ times faster than Krivelevich.

**2. Randomized 2-approximation (Algorithm 3): Robustness via Combinatorial Solvers**

While the deterministic version is faster, solving the LP exactly is still a bottleneck on large graphs and does not support weighted edges or approximate solutions. Algorithm 3 uses a randomized threshold: draw $r\in[0,1]$ and select $\{e\in E^+:x_e\ge r/2\}\cup\{e\in E^-:x_e>1-r\}$ as the cover. The proof uses integration: the probability of covering any bad triangle $t=\{e_1,e_2,e_3\}$ is directly linked to $\sum_{e\in t}x_e\ge 1$, yielding an expected edge count of $2\sum_e x_e$. It can be derandomized (Remark 3.4) by sorting edges by $x_e$ into two scanning lines and checking $\mathcal{O}(|E|)$ candidate thresholds. Notably (Remark 3.5), it provides a $(2+2\epsilon)$-approximation even when $\{x_e\}$ is only $(1+\epsilon)$-optimal, allowing integration with the $\widetilde{\mathcal{O}}(\epsilon^{-7}m^{3/2})$ combinatorial solver from Cao et al. (2024).

**3. Improved Pivot: Tightening the Cover-to-Cluster Ratio to $3/2$ (Algorithm 4 / Theorem 5.1)**

To obtain a CC clustering from a cover $F$, Veldt's (2022) MatchFlipPivot flips the signs of all edges in $F$ and performs a pivot on the auxiliary graph, guaranteeing only $\text{OPT}_{CC}\le 2\,\text{OPT}_\Delta$. This paper instead uses **probabilistic rules depending on the cover $F$**: if $uv \in F$, positive edges are sucked into the cluster with probability $1/4$ and negative edges with $3/4$; if $uv \notin F$, Ailon's deterministic rule is used. By assigning a unit budget $b(uv)=1$ to each edge in $F$, the analysis shows that for any triplet $\{u,v,w\}$, the ratio of "expected errors" to "expected budget" is $\le \tfrac{3}{2}$. This yields $\text{OPT}_{CC}\le\tfrac32\text{OPT}_\Delta$, improving the CC approximation ratio from 6 to $3+\epsilon$ when paired with Theorem 1.1's cover.

**4. Unified Hardness Lower Bound: One Gadget for Four Problems (Theorem 1.2 / 4.6-4.7)**

The paper establishes the first **explicit constant hardness** on complete graphs. Using a hexagonal gadget and clause edges, a gap-preserving reduction from Minimum 2CNF Deletion (MD) is performed—each variable is represented by a 12-node hexagon such that BTT optima correspond to MD optima. The MD gap of $2\delta n$ vs $3\delta n$ translates to a BTT gap of $(11+2\delta)n$ vs $(11+3\delta)n$. With $\delta=1/194$, it is NP-hard to approximate BTT within $<\tfrac{2137}{2136}$. Crucially, the same construction matches the optima for MinSTC+, CC, and CD, providing a unified lower bound for all four problems.

### Loss & Training
This is a theoretical paper on combinatorial optimization; it does not involve training loops. The 2-approximations are rounding processes on $\text{LP}_\Delta$ solutions. Analyses rely on LP complementary slackness, integration for probabilities, and the Ailon triplet-amortization framework.

## Key Experimental Results

### Main Results
This is a purely theoretical work. The table below summarizes the approximation ratios and time complexities ($m$ edges, $\alpha \ge 2$ for matrix multiplication):

| Algorithm | Approx. Ratio | LP Solves | Time Complexity | Work |
|-----------|---------------|-----------|-----------------|------|
| Standard 3-approx | 3 | 0 | $\mathcal{O}(m^{3/2})$ | Multiple |
| Krivelevich 1995 | 2 | $\mathcal{O}(m)$ | $\widetilde{\mathcal{O}}(m^\alpha+1)$ | Krivelevich 1995 |
| Algorithm 2 (Det.) | 2 | 1 | $\widetilde{\mathcal{O}}(m^\alpha)$ | **Ours** |
| Algorithm 3 (Rand.) | 2 (Exp.) | 1 (Approx. LP) | $\widetilde{\mathcal{O}}(\epsilon^{-7} m^{3/2})$ | **Ours** |
| Algorithm 3 + Derand. | 2 | 1 | + $\mathcal{O}(|E|\log n)$ | **Ours** |

### Hardness & Conversion Ratios

| Result | Conclusion | Scope | Type |
|--------|------------|-------|------|
| Theorem 4.2 | BTT is as hard as Vertex Cover; UGC-hard $\ge 2$ | General Graphs | Hardness |
| Theorem 4.6 | BTT is NP-hard to approximate within $< \tfrac{2137}{2136}$ | Complete Graphs | Hardness |
| Theorem 4.7 | $\tfrac{2137}{2136}$ bound applies to CC / MinSTC / CD | Complete Graphs | Hardness |
| Theorem 5.1 | $\text{OPT}_{CC} \le \tfrac{3}{2}\,\text{OPT}_\Delta$ (improves Veldt's 2) | Complete Graphs | Conversion |
| Lemma 4.1 | $\text{LP}_\Delta$ integrality gap $\ge 2$ | Complete Graphs | Tightness |

### Key Findings
- **One-pass LP rounding is sufficient for a 2-approximation**: Algorithms 2 and 3 prove that the bipartite structure of BTT allows for single-round rounding, reducing complexity from $m^{\alpha+1}$ to $m^\alpha$ (or $m^{3/2}$ on complete graphs).
- **2-approximation is the limit on general graphs**: Combining Theorem 4.2 with Khot–Regev (2008) implies that breaking the 2-approximation barrier is UGC-hard.
- **Unified reduction for four problems**: Using the 2CNF gadget to provide a $\tfrac{2137}{2136}$ bound for BTT, CC, MinSTC, and CD is a significant conceptual contribution, as MinSTC previously lacked an explicit lower bound.
- **$\tfrac{3}{2}$ pivot improves CC ratio**: Linking the $(2+\epsilon)$ BTT cover to the $\tfrac{3}{2}$ pivot improves the CC approximation from 6 to $3+\epsilon$.

## Highlights & Insights
- **"Bipartite hypergraph VC" perspective**: Reinterpreting BTT as a 3-uniform hypergraph VC with exactly one negative node per edge allows for different rounding thresholds—a paradigm applicable to other labeled covering problems.
- **"Two-line scanning" derandomization**: A clean paradigm to derandomize LP rounding by checking $\mathcal{O}(|E|)$ discrete candidate thresholds.
- **Budget-pivot analysis framework**: Assigning budgets to cover edges and proving triplet error-to-budget ratios $\le 3/2$ simplifies the proof and improves benchmarks.
- **Gadget reuse**: Showing that four seemingly different clustering/editing problems are homogeneous in hardness on complete graphs.
- **Weighted and approx-LP friendliness**: Support for weighted BTT and $(1+\epsilon)$-LP solutions allows the algorithms to be applied to practical problems without re-proof.

## Limitations & Future Work
- The $\epsilon^{-7}$ factor in the $\widetilde{\mathcal{O}}(\epsilon^{-7} m^{3/2})$ complexity is still costly for small $\epsilon$.
- 2 is the optimal approximation on general graphs (UGC-hard); improvements are restricted to complete graphs.
- The open question of whether $\text{OPT}_{CC} = \text{OPT}_\Delta$ remains unsolved.
- Providing a $(1+\epsilon)$ polynomial-time conversion would translate the $(2+\epsilon)$ BTT cover into a $(2+2\epsilon)$ CC approximation, reaching the integrality gap lower bound.
- Lack of empirical evaluation on real-world networks to compare heuristic clustering quality.

## Related Work & Insights
- **vs Krivelevich (1995)**: Same ratio, but this work is $\mathcal{O}(m)$ times faster by using one-pass rounding on signed structures.
- **vs Veldt (2022)**: Improves the conversion bound from 2 to $3/2$ using budget-based pivot analysis.
- **vs Cao et al. (2024b)**: Seamlessly integrates with their combinatorial LP solver to achieve $\widetilde{\mathcal{O}}(m^{3/2})$ complexity.
- **vs Cohen-Addad et al. (2022)**: While CC has a 1.485-approximation, this work confirms that BTT cannot break the 2-approximation barrier using $\text{LP}_\Delta$, even on complete graphs.
- **vs Charikar et al. (2005)**: Extends CC APX-hardness to the first explicit constant lower bound $\tfrac{2137}{2136}$.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2026\] Quantum Algorithms for Triangle Cut Sparsification](quantum_algorithms_for_triangle_cut_sparsification.md)
- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[ICML 2025\] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](../../ICML2025/learning_theory/sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)

</div>

<!-- RELATED:END -->
