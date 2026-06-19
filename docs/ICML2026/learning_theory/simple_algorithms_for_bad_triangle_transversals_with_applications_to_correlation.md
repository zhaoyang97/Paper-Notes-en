---
title: >-
  [Paper Note] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering
description: >-
  [ICML 2026][learning_theory][Correlation Clustering] This paper provides two simple 2-approximation algorithms for the "Bad Triangle Transversal" (BTT) problem on signed graphs that require only a single LP solve. It proves that on complete graphs, BTT, Correlation Clustering, MinSTC, and Cluster Deletion share a uniform NP-hardness of approximation lower bound of $\tfra
tags:
  - ICML 2026
  - learning_theory
  - Correlation Clustering
  - LP rounding
date: 2026-05-08
content_hash: 99569e7cce69d297
---
# Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2602.04463](https://arxiv.org/abs/2602.04463)  
**Code**: None  
**Area**: Algorithm Theory / Approximation Algorithms / Graph Clustering  
**Keywords**: Bad Triangle Transversal, Correlation Clustering, Approximation Algorithms, LP rounding, Hardness Proofs  

## TL;DR
This paper provides two simple 2-approximation algorithms for the "Bad Triangle Transversal" (BTT) problem on signed graphs that require only a single LP solve. It proves that on complete graphs, BTT, Correlation Clustering, MinSTC, and Cluster Deletion share a uniform NP-hardness of approximation lower bound of $\tfrac{2137}{2136}$. Furthermore, it constructs a new pivot process to convert any feasible BTT cover into a clustering with at most $\tfrac{3}{2}|F|$ errors, thereby tightening the gap between BTT and CC optimal values from 2 to $3/2$.

## Background & Motivation
**Background**: Signed graphs $G=(V, E^+, E^-)$ are widely used in social networks, Ising models, and clustering. A "bad triangle" refers to a triangle with exactly one negative edge—the smallest unit of imbalance in structural balance theory. Correlation Clustering (CC) requires partitioning nodes into clusters to minimize the total number of inter-cluster positive edges and intra-cluster negative edges. Bansal et al. initially used "packing disjoint bad triangles" to obtain constant approximations for CC, and Ailon's pivot later became the benchmark for 3-approximations.

**Limitations of Prior Work**: (1) BTT serves as a lower bound for CC. Therefore, on large-scale graphs where LP is difficult to solve, researchers seek fast BTT covers to "convert" into CC clusterings. However, existing BTT algorithms are either 3-approximations (taking all edges of a set of edge-disjoint bad triangles) or follow Krivelevich's (1995) 2-approximation for unsigned triangle covers—the latter requires repeatedly solving LPs $\mathcal{O}(m)$ times, with a time bottleneck of $\widetilde{\mathcal{O}}(m^{\alpha+1})$, which is impractical for large graphs. (2) Whether BTT can strictly outperform a 2-approximation on complete graphs or if hardness lower bounds exist has been a long-standing void. (3) Existing cover-to-cluster conversions like MatchFlipPivot (Veldt, 2022) only guarantee $\text{OPT}_{CC} \le 2\,\text{OPT}_\Delta$, causing the CC approximation ratio to degrade from $\alpha$ to $2\alpha$, resulting in significant loss.

**Key Challenge**: BTT on a 3-uniform hypergraph is equivalent to vertex cover with a "bipartite" constraint (each hyperedge contains exactly one negative-edge node). It is neither a standard VC nor a $k$-partite VC, so Lovász's 1975 randomized VC algorithm cannot be directly applied. To obtain an approximation better than 3, one must rely on the structural information of the LP relaxation $\text{LP}_\Delta$.

**Goal**: (1) Provide 2-approximations that can directly round the $\text{LP}_\Delta$ solution in a single pass without repeated re-solving. (2) Prove a unified hardness lower bound for BTT/CC/MinSTC/CD on complete graphs. (3) Improve the cover-to-cluster conversion ratio, tightening it from 2 to $3/2$, thereby improving the CC approximation ratio from $6$ to $3+\epsilon$.

**Key Insight**: The authors observe that the constraint structure of $\text{LP}_\Delta$ possesses a bipartite nature where "each bad triangle contains exactly 1 negative edge." By setting different rounding thresholds for "positive edges" and "negative edges"—using $x_e \ge 1/2$ for positive edges and $x_e > 0$ for negative edges (or using $x_e > 1-r$ vs $x_e \ge r/2$ for a random threshold $r$)—this bipartite constraint can be exploited to obtain a 2-approximation within a single round of rounding.

**Core Idea**: Replace Krivelevich's "low-speed iterative rounding" with "single-round LP rounding with asymmetric thresholds"; replace Veldt's "edge flipping followed by pivot" with "categorized budget attack + improved pivot probabilities."

## Method

### Overall Architecture
The paper presents four interconnected contributions regarding BTT: first, two 2-approximation algorithms that **solve the LP only once** (Algorithm 2 deterministic, Algorithm 3 randomized, replacing Krivelevich's 1995 iterative re-solving); second, an **improved pivot** for converting covers to clusterings (Algorithm 4, tightening the cover-to-cluster ratio from 2 to $3/2$); and finally, a **unified hardness lower bound on complete graphs** (using a 2SAT gadget to simultaneously address BTT/CC/MinSTC/CD). The first two parts solve "how to calculate quickly and accurately," the third solves "how to convert the calculated cover into CC clustering with minimal loss," and the fourth defines "where the theoretical limits lie." These four core designs are detailed below.

### Key Designs

**1. Simple Deterministic 2-Approximation (Algorithm 2): Single LP Rounding Replacing Krivelevich's Iterative Re-solving**

Existing 2-approximations (Krivelevich 1995) require solving LPs $\mathcal{O}(m)$ times, with a time bottleneck of $\widetilde{\mathcal{O}}(m^{\alpha+1})$, making them unusable on large graphs. The key observation of this paper is that BTT on a 3-uniform hypergraph has a bipartite structure where "each bad triangle contains exactly one negative edge." Thus, positive and negative edges can be handled in one round using two sets of thresholds: solve $\text{LP}_\Delta$ once to get fractional solutions $\{x_e\}$, and directly output $E^-_{>0}\cup E^+_{\ge 1/2}$—all negative edges with non-zero values plus all positive edges with values $\ge 1/2$. Correctness is based on complementary slackness: the dual Packing LP has tight constraints only on edges where $x_e>0$, and combined with bipartiteness, it can be proved that the total number of edges is $\le 2\cdot\text{LP}_\Delta\le 2\cdot\text{OPT}_\Delta$. Since the LP is solved only once, the time is dominated by the LP solver ($\widetilde{\mathcal{O}}(m^\alpha)$), approximately $m$ times faster than Krivelevich.

**2. Randomized 2-Approximation (Algorithm 3): Robustness to Weighted and Approximate LP Solutions, Running in $m^{3/2}$ with Modern Combinatorial Solvers**

While the deterministic version is fast, solving the LP itself remains a bottleneck on large graphs, and it does not support weighted or "only approximately optimal" settings. Algorithm 3 addresses these using a random threshold: draw $r\in[0,1]$, and take $\{e\in E^+:x_e\ge r/2\}\cup\{e\in E^-:x_e>1-r\}$ as the cover. The proof uses integration techniques—the probability of covering any bad triangle $t=\{e_1,e_2,e_3\}$ is directly linked to $\sum_{e\in t}x_e\ge 1$, and the expected number of edges is exactly $2\sum_e x_e$. It can also be cleanly derandomized (Remark 3.4): sort positive and negative edges by $x_e$ into two scanlines; all $r$ intervals generate only $\mathcal{O}(|E|)$ candidate solutions, and picking the minimum takes only an additional $\mathcal{O}(|E|\log n)$. A significant advantage in Remark 3.5 is that when $\{x_e\}$ is only a $(1+\epsilon)$-approximate solution, it still yields a $(2+2\epsilon)$-approximation. This allows integration with the $\widetilde{\mathcal{O}}(\epsilon^{-7}m^{3/2})$ combinatorial LP solver from Cao et al. 2024, pushing the overall complexity on complete graphs to $\widetilde{\mathcal{O}}(\epsilon^{-7}m^{3/2})$, nearly matching the lower bound time for "finding a maximal set of disjoint bad triangles."

**3. Improved Pivot: Tightening the Cover-to-Cluster Ratio from 2 to $3/2$ (Algorithm 4 / Theorem 5.1)**

The previous two algorithms only solve for "finding a small cover $F$," but the practical goal is CC clustering—converting $F$ into clusters. Veldt's 2022 MatchFlipPivot follows the path of "flipping the signs of all edges in $F$ and then pivoting on the auxiliary graph," which only guarantees $\text{OPT}_{CC}\le 2\,\text{OPT}_\Delta$, losing a factor of two. This paper does not flip edges but instead replaces the deterministic absorption rule of Ailon pivot with a **probabilistic rule dependent on the cover $F$**: when the edge $uv$ under consideration is in $F$, a positive edge is absorbed into the current pivot's cluster with probability $1/4$, and a negative edge with probability $3/4$; if $uv\notin F$, Ailon's deterministic rule is used (absorb positive, reject negative). The analytical ingenuity lies in a new "cost sharing" scheme—assigning a unit budget $b(uv)=1$ to each $uv\in F$ and 0 otherwise, charging pivot errors to the budget rather than the LP value. By proving that any triplet $\{u,v,w\}$ satisfies "sum of error probabilities / sum of budget probabilities $\le\tfrac32$" (verified across four categories of triangles based on positive edge counts in complete graphs), the Ailon framework immediately yields an expected $\tfrac32|F|$ errors, i.e., Theorem 1.4: $\text{OPT}_{CC}\le\tfrac32\text{OPT}_\Delta$. Connecting this to the $(2+\epsilon)$ cover from Theorem 1.1, the CC approximation ratio drops from Veldt's 6 to $3+\epsilon$ with comparable time complexity.

**4. Unified Hardness Lower Bound on Complete Graphs: One Gadget for Four Problems (Theorem 1.2 / 4.6-4.7)**

With algorithms pushing the upper bound to 2, it is natural to ask "where is the lower bound." This paper provides the first **explicit constant hardness** on complete graphs: using a hexagonal gadget + clause edges, it performs a gap-preserving reduction from Minimum 2CNF Deletion (MD) by Chlebík & Chlebíková. Each variable is assigned a 12-node hexagon (containing 6 crowns), and clauses connect crowns to clause nodes such that the optimal BTT solution corresponds exactly to the optimal MD solution. The $2\delta n$ vs $3\delta n$ gap in MD translates to an $(11+2\delta)n$ vs $(11+3\delta)n$ gap for BTT; setting $\delta=1/194$ yields "NP-hard to approximate within $<\tfrac{2137}{2136}$." The true conceptual contribution is **reusability**: under the same construction, the optimal values for MinSTC+/CC/CD all equal $\text{OPT}_\Delta(G)$, so a single reduction provides the same lower bound for all four problems. Previously, MinSTC+ had no explicit bound, and CC only had a weaker randomized bound. Combined with Lemma 4.1 ($\text{LP}_\Delta$ integrality gap $\ge 2$ even on complete graphs), this defines the hard boundary where "it is impossible to break the 2-approximation based on $\text{LP}_\Delta$."

### Loss & Training
This is a paper on combinatorial optimization and approximation algorithms and does not involve training loops. All 2-approximations are rounding processes for $\text{LP}_\Delta$ solutions, and all $\tfrac{3}{2}$-approximations are randomized pivots based on Algorithm 4. Analysis relies on standard LP complementary slackness, integral probability arguments, and triplet-based sharing within the Ailon framework.

## Key Experimental Results

### Main Results
This paper is a purely theoretical work with no experimental code. The following table summarizes the approximation ratios and time complexities of various algorithms for BTT ($m$ is the number of edges, $\alpha \ge 2$ is the matrix multiplication constant for LP solving):

| Algorithm | Approximation Ratio | LP Solves | Time Complexity | Work |
|-----------|---------------------|-----------|-----------------|------|
| Standard 3-approximation (Maximal disjoint triangles) | 3 | 0 | $\mathcal{O}(m^{3/2})$ (Complete graph) | Various |
| Krivelevich 1995 (Algorithm 1) | 2 | $\mathcal{O}(m)$ | $\widetilde{\mathcal{O}}(m^{\alpha+1})$ | Krivelevich 1995 |
| Algorithm 2 (Deterministic) | 2 | 1 | $\widetilde{\mathcal{O}}(m^\alpha)$ | **Ours** |
| Algorithm 3 (Randomized) | 2 (Expected) | 1 (Approx LP) | $\widetilde{\mathcal{O}}(\epsilon^{-7} m^{3/2})$ (Complete graph, Theorem 1.1) | **Ours** |
| Algorithm 3 + Derandomization | 2 | 1 | + $\mathcal{O}(|E|\log n)$ | **Ours** |

### Lower Bounds and Conversion Ratios

| Result | Conclusion | Scope | Type |
|--------|------------|-------|------|
| Theorem 4.2 | BTT is as hard to approximate as Vertex Cover, UGC-hard $\ge 2$ | General Graph | Hardness |
| Theorem 4.6 | BTT on complete graphs is NP-hard to approximate $< \tfrac{2137}{2136}$ | Complete Graph | Hardness |
| Theorem 4.7 | Same gadget gives $\tfrac{2137}{2136}$ lower bound for CC / MinSTC / CD | Complete Graph | Hardness |
| Theorem 5.1 / 1.4 | $\text{OPT}_{CC} \le \tfrac{3}{2}\,\text{OPT}_\Delta$ (Improving Veldt 2022's 2) | Complete Graph | Gain |
| Lemma 4.1 | $\text{LP}_\Delta$ integrality gap $\ge 2$ (even on complete graphs) | Complete Graph | Gap |

### Key Findings
- **Single LP rounding suffices for 2-approximation**: Algorithms 2/3 prove that the bipartite structure of BTT allows completing rounding in one round, improving the $m^{\alpha+1}$ complexity of Krivelevich to $m^\alpha$ (or $m^{3/2}$ on complete graphs).
- **2-approximation is the limit for general graphs**: Combining Theorem 4.2 with Khot–Regev 2008 implies that unless UGC is false, the 2-approximation cannot be broken—shifting research focus to whether $< 2$ is possible on complete graphs.
- **Unified reduction for four problems**: Using the Chlebík 2CNF gadget to provide a $\tfrac{2137}{2136}$ lower bound for BTT/CC/MinSTC/CD simultaneously is a conceptual contribution. Previously, MinSTC had no explicit bound, and CC only had the randomized $24/23$ bound from Cao et al. 2024—this paper provides a stronger deterministic reduction.
- **$\tfrac{3}{2}$ pivot directly improves CC approximation**: Combining Theorem 1.1 and Theorem 1.4, the $(2+\epsilon)$ approximation of BTT translates to a $(3+\epsilon)$ approximation for CC, a significant improvement over the 6-approximation of Veldt's MatchFlipPivot with similar time complexity.

## Highlights & Insights
- **"Bipartite hypergraph VC" Perspective**: The authors re-interpret BTT as a specific 3-uniform hypergraph VC (each hyperedge contains exactly 1 negative node) and use this structural information to set rounding thresholds independently for positive and negative edges—this "binning by edge type" approach can be transferred to any covering problem with "label / color" constraints.
- **"Dual Scanlines" Derandomization**: Remark 3.4 discretizes the continuous random variable $r$ in Algorithm 3 into two scanlines of positive and negative edges sorted by LP value, checking only $\mathcal{O}(|E|)$ candidate thresholds—a clean paradigm for derandomizing LP rounding algorithms.
- **Budget-Pivot Analysis Framework**: Setting a unit budget for each cover edge and proving a "error vs budget" ratio $\le \tfrac{3}{2}$ for arbitrary triplets allows directly obtaining the cover-to-cluster ratio. This small variation in "cost-sharing" pulls Veldt's 2-bound down to $3/2$, suggesting further potential in "how to choose pivot probabilities" for the CC family.
- **Gadget Reuse for Multi-problem Lower Bounds**: Using the same hexagonal + clause structure to simultaneously hit the hardness of BTT/CC/MinSTC/CD makes the observation explicit: these four different clustering/editing problems on complete graphs are homogeneous in their hardness.
- **Friendliness to Weighted and Approximate LP**: Algorithm 3 supports both weighted BTT and "$(1+\epsilon)$-approximate LP solutions," allowing it to connect directly to practical weighted problems like LambdaSTC and temporal MinSTC+ without needing re-proofs.

## Limitations & Future Work
- The "speed" described in the paper refers to asymptotic complexity after LP solving, which is insensitive to constants or specific LP solver choices; however, the $\epsilon^{-7}$ factor in $\widetilde{\mathcal{O}}(\epsilon^{-7} m^{3/2})$ remains challenging for small $\epsilon$ in engineering practice.
- For general graphs, 2 is the optimal approximation (UGC-hard), so further improvement is restricted to complete graphs. The key open question $\text{OPT}_{CC} = \text{OPT}_\Delta$ has neither a proof nor a counterexample.
- The $\tfrac{3}{2}$ ratio in Algorithm 4 depends on randomized pivots. If a polynomial-time conversion with ratio $(1+\epsilon)$ can be found, the $(2+\epsilon)$ BTT approximation would translate directly into a $(2+2\epsilon)$ CC approximation, approaching the LP integrality gap lower bound.
- No empirical experiments were conducted: it remains unknown how Algorithm 2/3 vs MatchFlipPivot performs regarding speed or clustering quality on real social/biological network data.

## Related Work & Insights
- **vs Krivelevich (1995)**: Also a 2-approximation but requires iteratively solving $\mathcal{O}(m)$ LPs. This paper’s Algorithm 2/3 solve the LP only once and explicitly utilize the signed structure (exactly one negative edge), making them simpler and faster than Krivelevich's unsigned triangle cover algorithm.
- **vs Veldt (2022) MatchFlipPivot**: Veldt used "flipping cover edges then pivoting" to get $\text{OPT}_{CC} \le 2\,\text{OPT}_\Delta$. This paper uses budget-pivot analysis to reach $\tfrac{3}{2}$, improving the CC approximation ratio from 6 to $3+\epsilon$.
- **vs Cao et al. (2024b)**: They provide a $(1+\epsilon)$ combinatorial approximate LP solver for $\text{LP}_\Delta$ on complete graphs and a $2.4$-approximation for CC. Algorithm 3 in this paper seamlessly integrates with Cao’s solver (Remark 3.5) to obtain $(2+\epsilon)$ BTT and $(3+\epsilon)$ CC.
- **vs Chawla et al. (2015)**: Chawla gave a 2.06-approximation for CC on complete graphs but requires solving $\text{LP}_{\text{CC}}$ ($\Theta(n^3)$ constraints). This paper takes the opposite direction—trading a higher constant factor for better scalability via the lightweight $\text{LP}_\Delta$.
- **vs Cohen-Addad et al. (2022)**: They broke the 2-approximation for CC (reaching 1.485). This paper provides a tightness result (Lemma 4.1 integrality gap $\ge 2$) showing that a $< 2$-approximation for BTT is impossible based on $\text{LP}_\Delta$, implying that any breakthrough must move beyond the $\text{LP}_\Delta$ framework.
- **vs Charikar et al. (2005)**: Used a similar hexagram gadget to prove CC is APX-hard on complete graphs. This paper provides the first **explicit constant lower bound** of $\tfrac{2137}{2136}$ and extends it to MinSTC+ / CD.
- **vs Bansal et al. (2002) and Ailon et al. (2008)**: Early CC algorithms used "packing edge-disjoint bad triangles" as fractional duals or for direct pivots. This paper follows Ailon's pivot framework but modifies budgets and probabilities, simultaneously improving the cover→cluster ratio and CC approximation ratio.

## Related Papers

- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[ICML 2025\] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](../../ICML2025/learning_theory/sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)
- [\[ICML 2026\] Matroid Algorithms Under Size-Sensitive Independence Oracles](matroid_algorithms_under_size-sensitive_independence_oracles.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[ICML 2026\] Matroid Algorithms Under Size-Sensitive Independence Oracles](matroid_algorithms_under_size-sensitive_independence_oracles.md)
- [\[ICML 2026\] Conditional KRR: Injecting Unpenalized Features into Kernel Methods with Applications to Kernel Thresholding](conditional_krr_injecting_unpenalized_features_into_kernel_methods_with_applicat.md)
- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)

</div>

<!-- RELATED:END -->
