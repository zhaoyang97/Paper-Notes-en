---
title: >-
  [Paper Note] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering
description: >-
  [ICML 2026][Bad Triangle Transversal] This paper presents two simple 2-approximation algorithms for the "Bad Triangle Transversal" (BTT) problem on signed graphs that require only a single LP solve. It proves a unified N…
tags:
  - "ICML 2026"
  - "Bad Triangle Transversal"
  - "Correlation Clustering"
  - "Approximation Algorithms"
  - "LP rounding"
  - "Hardness Proof"
date: 2026-05-08
content_hash: 6254d74ec97653e3
---

# Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering

**Conference**: ICML 2026  
**arXiv**: [2602.04463](https://arxiv.org/abs/2602.04463)  
**Code**: None  
**Area**: Algorithmic Theory / Approximation Algorithms / Graph Clustering  
**Keywords**: Bad Triangle Transversal, Correlation Clustering, Approximation Algorithms, LP rounding, Hardness Proof  

## TL;DR
This paper presents two simple 2-approximation algorithms for the "Bad Triangle Transversal" (BTT) problem on signed graphs that require only a single LP solve. It proves a unified NP-hardness lower bound of $\tfrac{2137}{2136}$ for BTT, Correlation Clustering (CC), MinSTC, and Cluster Deletion on complete graphs. Additionally, it introduces a new pivot process that converts any feasible BTT cover into a clustering with at most $\tfrac{3}{2}|F|$ errors, tightening the gap between BTT and CC optimal values from 2 to $3/2$.

## Background & Motivation
**Background**: Signed graphs $G=(V, E^+, E^-)$ are widely used in social networks, Ising models, and clustering. A "bad triangle" refers to a triangle with exactly one negative edge—the smallest unit of imbalance in structural balance theory. Correlation Clustering (CC) aims to partition nodes into clusters to minimize the sum of inter-cluster positive edges and intra-cluster negative edges. Bansal et al. initially used "packing disjoint bad triangles" to obtain constant approximations for CC, and Ailon's subsequent pivot method remains a 3-approximation benchmark.

**Limitations of Prior Work**: (1) Since BTT is a lower bound for CC, researchers seek fast BTT covers to "convert" into CC clusters for large-scale graphs where the CC LP is hard to solve. However, known BTT algorithms are either 3-approximations (taking all edges of disjoint bad triangles) or adapt Krivelevich's (1995) 2-approximation for unsigned triangle covers—the latter requires solving LPs $\mathcal{O}(m)$ times with a time bottleneck of $\widetilde{\mathcal{O}}(m^{\alpha+1})$, which is impractical for large graphs. (2) There has long been an absence of hardness lower bounds for BTT on complete graphs and whether it can strictly outperform a 2-approximation. (3) Existing conversion methods like MatchFlipPivot (Veldt, 2022) only guarantee $\text{OPT}_{CC} \le 2\,\text{OPT}_\Delta$, causing the CC approximation ratio to lose a factor of 2.

**Key Challenge**: BTT on a 3-uniform hypergraph is equivalent to vertex cover with a "bipartite" constraint (each hyperedge contains exactly one negative-edge node). This is neither a standard VC nor a $k$-partite VC, meaning Lovász's (1975) randomized VC algorithm cannot be directly applied. Achieving an approximation ratio better than 3 requires leveraging the structural information of the LP relaxation $\text{LP}_\Delta$.

**Goal**: (1) Provide 2-approximation algorithms using a single rounding of the $\text{LP}_\Delta$ solution without iterative solving. (2) Prove a unified hardness lower bound for BTT/CC/MinSTC/CD on complete graphs. (3) Improve the cover-to-cluster conversion ratio from 2 to $3/2$, thereby improving the CC approximation ratio from $6$ to $3+\epsilon$.

**Key Insight**: The authors observe that the constraint structure of $\text{LP}_\Delta$ possesses a bipartite nature where "each bad triangle contains exactly 1 negative edge." By setting different rounding thresholds for "positive" and "negative" edges—using $x_e \ge 1/2$ for positive and $x_e > 0$ for negative (or $x_e > 1-r$ vs $x_e \ge r/2$ for a random threshold $r$)—one can exploit this bipartite constraint to obtain a 2-approximation in a single rounding round.

**Core Idea**: Replace Krivelevich's slow iterative rounding with "asymmetric threshold single-round LP rounding"; replace Veldt's edge-flipping pivot with a "category-based budget attack + improved pivot probabilities."

## Method

### Overall Architecture
The paper presents three BTT algorithms (Algorithm 1 restates Krivelevich, while Algorithms 2 and 3 are original contributions) and one cover-to-cluster algorithm (Algorithm 4). The remaining results consist of hardness proofs and theoretical analysis.

### Key Designs

1.  **Simple Deterministic 2-Approximation (Algorithm 2)**:
    - **Function**: Outputs a 2-approximate cover in one shot after solving $\text{LP}_\Delta$ once using asymmetric thresholds.
    - **Mechanism**: Solves $\text{LP}_\Delta$ for the optimal fractional solution $\{x_e\}$ and outputs $E^-_{>0} \cup E^+_{\ge 1/2}$ (all negative edges with non-zero values + all positive edges with values at least $1/2$). The proof relies on complementary slackness: tight constraints in the dual PackingLP only exist for edges where $x_e > 0$. Combined with the property that "each bad triangle contains one negative edge," it is proven that the total number of edges does not exceed $2 \cdot \text{LP}_\Delta \le 2 \cdot \text{OPT}_\Delta$. The complexity is dominated by the LP solver ($\widetilde{\mathcal{O}}(m^\alpha)$), which is $m$ times faster than Krivelevich's $\widetilde{\mathcal{O}}(m^{\alpha+1})$.
    - **Design Motivation**: To avoid iterative rounding and exploit the bipartite structure of BTT on 3-uniform hypergraphs.

2.  **Randomized 2-Approximation (Algorithm 3) + Weighted/Approx-LP Compatibility**:
    - **Function**: Achieves a 2-approximation in expectation and supports weighted BTT and approximately optimal LP solutions.
    - **Mechanism**: Samples $r \in [0,1]$ and selects $\{e \in E^+: x_e \ge r/2\} \cup \{e \in E^-: x_e > 1-r\}$ as the cover. The proof uses integration: the probability of covering any bad triangle $t = \{e_1, e_2, e_3\}$ is tied to $\sum_{e\in t} x_e \ge 1$, resulting in an expected size of $2\sum_e x_e$. Remark 3.4 provides a derandomization method in $\mathcal{O}(|E|\log n)$ by scanning sorted LP values. As per Remark 3.5, if $\{x_e\}$ is a $(1+\epsilon)$-approximate solution, the algorithm yields a $(2+2\epsilon)$-approximation. This allows the use of fast combinatorial LP solvers (Cao et al. 2024), reducing BTT complexity on complete graphs to $\widetilde{\mathcal{O}}(\epsilon^{-7} m^{3/2})$.
    - **Design Motivation**: To support large-scale instances where exact LP solving is a bottleneck, using a rounding scheme robust to "approximate LP solutions."

3.  **Improved Pivot: $\tfrac{3}{2}$-Approximation for Cover-to-Cluster (Algorithm 4) + Unified Hardness**:
    - **Function**: Given any feasible BTT cover $F$, outputs a clustering with errors $\le \tfrac{3}{2}|F|$; proves a $\tfrac{2137}{2136}$ NP-hardness lower bound for BTT/CC/MinSTC/CD on complete graphs.
    - **Mechanism**: Replaces the rigid "if u-v is positive, pool into cluster" rule of Ailon pivot with an $F$-dependent probabilistic rule: if $uv \in F \cap E^+$, pick with probability $1/4$; if $uv \in F \cap E^-$, pick with probability $3/4$; if $uv \notin F$, follow deterministic rules. The analysis assigns a budget $b(uv)=1$ for each $uv \in F$ and proves that for any triplet $\{u,v,w\}$, the ratio of expected errors to budget is $\le 3/2$. Hardness is proven via a "hexagonal gadget + clause edges" reduction from Minimum 2CNF Deletion (MD).
    - **Design Motivation**: (a) Improving the cover-to-cluster ratio to $3/2$ immediately yields a $(3+\epsilon)$ CC approximation. (b) A single gadget setup suffices for four problems because their optimal values are equivalent to $\text{OPT}_\Delta(G)$ in this construction.

## Key Experimental Results

### Main Results
This is a theoretical paper. The following table summarizes the BTT algorithms' approximation ratios and complexities:

| Algorithm | Approx. Ratio | LP Solves | Time Complexity | Work |
| :--- | :--- | :--- | :--- | :--- |
| Standard 3-approx (Maximal Disjoint $\Delta$) | 3 | 0 | $\mathcal{O}(m^{3/2})$ (complete) | Various |
| Krivelevich 1995 (Algorithm 1) | 2 | $\mathcal{O}(m)$ | $\widetilde{\mathcal{O}}(m^{\alpha+1})$ | Krivelevich 1995 |
| Algorithm 2 (Deterministic) | 2 | 1 | $\widetilde{\mathcal{O}}(m^\alpha)$ | **Ours** |
| Algorithm 3 (Randomized) | 2 (Exp.) | 1 (Approx. LP) | $\widetilde{\mathcal{O}}(\epsilon^{-7} m^{3/2})$ | **Ours** |
| Algorithm 3 + Derandomized | 2 | 1 | + $\mathcal{O}(|E|\log n)$ | **Ours** |

### Key Findings
- **Single LP Rounding suffices for 2-approximation**: Algorithms 2 and 3 prove that the bipartite structure of BTT allows for one-shot rounding, improving complexity scale.
- **2-approximation is the limit for general graphs**: Theorem 4.2 combined with Khot–Regev 2008 shows that BTT is UGC-hard to approximate better than 2.
- **Unified Reduction**: Using the Chlebík 2CNF gadget to provide a $\tfrac{2137}{2136}$ bound for BTT/CC/MinSTC/CD is a conceptual contribution, providing the first explicit lower bound for MinSTC.
- **$3/2$ Pivot Impact**: The BTT-to-CC conversion improvement significantly refines the CC approximation ratio to $(3+\epsilon)$ compared to the previous 6-approximation by MatchFlipPivot.

## Highlights & Insights
- **"Bipartite Hypergraph VC" Perspective**: Interpreting BTT as a specific hypergraph VC where each edge has exactly one negative node allows the use of dual-threshold rounding.
- **"Two-Line Scanning" Derandomization**: Derandomizing the continuous variable $r$ via discrete LP value sorting is a clean paradigm for LP-based approximation algorithms.
- **Budget-Pivot Framework**: Assigning unit budgets to cover edges and analyzing triplet error-to-budget ratios suggests that pivot probabilities in CC problems still have room for optimization.
- **Weighted and Approx-LP Friendliness**: These properties allow Algorithm 3 to integrate directly with modern combinatorial LP solvers and weighted problem variants like LambdaSTC.

## Limitations & Future Work
- The time complexity factor $\epsilon^{-7}$ remains heavy for practical applications with small $\epsilon$.
- 2-approximation is likely the best achievable on general graphs (UGC-hard); thus, future research should focus on whether $< 2$ is possible specifically for complete graphs.
- The $\tfrac{3}{2}$ ratio in Algorithm 4 relies on a randomized pivot; a polynomial-time $(1+\epsilon)$ conversion would bring the CC approximation closer to the LP integrality gap of 2.
- Lack of empirical testing on real-world social or biological networks to evaluate practical performance against MatchFlipPivot.

## Related Work & Insights
- **vs Krivelevich (1995)**: Achieving the same 2-approximation but reducing $\text{LP}$ solve iterations from $\mathcal{O}(m)$ to 1 by exploiting the signed structure.
- **vs Veldt (2022) MatchFlipPivot**: Improved the $\text{OPT}_{CC} \le 2\,\text{OPT}_\Delta$ bound to $\tfrac{3}{2}\,\text{OPT}_\Delta$ via budget analysis.
- **vs Cao et al. (2024b)**: Integrates with their $(1+\epsilon)$ combinatorial LP solver to achieve $\widetilde{\mathcal{O}}(m^{3/2})$ complexity.
- **vs Cohen-Addad et al. (2022)**: While they improved CC approximations to 1.485, this paper provides a tightness result (integrality gap $\ge 2$) showing that BTT-based methods cannot break the 2-approximation barrier.
- **vs Charikar et al. (2005)**: Provides the first explicit constant lower bound $\tfrac{2137}{2136}$ for CC and related problems on complete graphs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/others/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/others/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[ICLR 2026\] Distributed Algorithms for Euclidean Clustering](../../ICLR2026/others/distributed_algorithms_for_euclidean_clustering.md)
- [\[ICML 2026\] Riemannian Networks over Full-Rank Correlation Matrices](riemannian_networks_over_full-rank_correlation_matrices.md)

</div>

<!-- RELATED:END -->
