---
title: >-
  [Paper Note] An Approximation Algorithm for Graph Label Selection
description: >-
  [ICML2026][Graph Learning][Graph Label Selection] This paper provides the first $\tilde{O}(\log^{1.5} n)$ approximation algorithm for Graph Label Selection without relaxing the labeling budget. It transforms the globally…
tags:
  - "ICML2026"
  - "Graph Learning"
  - "Graph Label Selection"
  - "Active Learning"
  - "Approximation Algorithms"
  - "Tree Cut Sparsification"
  - "Dynamic Programming"
date: 2026-05-08
content_hash: f3081d866a802e96
---

# An Approximation Algorithm for Graph Label Selection

**Conference**: ICML2026  
**arXiv**: [2605.18623](https://arxiv.org/abs/2605.18623)  
**Code**: https://github.com/josia-john/icml2026-graph-label-selection  
**Area**: Graph Learning  
**Keywords**: Graph Label Selection, Active Learning, Approximation Algorithms, Tree Cut Sparsification, Dynamic Programming  

## TL;DR
This paper provides the first $\tilde{O}(\log^{1.5} n)$ approximation algorithm for Graph Label Selection without relaxing the labeling budget. It transforms the globally coupled node selection problem into a solvable combinatorial optimization pipeline via tree cut sparsification, flow-based decision making, and dynamic programming on trees.

## Background & Motivation
**Background**: Active learning on graphs often represents the similarity between samples as a weighted graph. The goal is to select a small set of $k$ vertices for labeling such that they are representative of the entire graph. Graph Label Selection (GLS) is a classic formalization of this problem: select a labeled set $L$ to maximize the worst-case cut sparsity $\Psi(L)$ of the remaining unlabeled subset. Intuitively, this prevents the existence of a large unlabeled cluster that is weakly connected to known information.

**Limitations of Prior Work**: The target function is neither submodular nor supermodular, making point-wise greedy selection difficult to provide reliable guarantees. Existing theoretical methods mostly employ resource augmentation, allowing the algorithm to select more than $k$ labels to compare against the optimal solution with budget $k$. This is unnatural for practical active learning where the labeling budget is typically a hard constraint.

**Key Challenge**: The difficulty of GLS lies in the fact that the value of a point is not a local property. The utility of a point depends on which sparse clusters it helps "cut" in conjunction with other candidate points. Examples like star graphs show that local greedy choices can be significantly suboptimal. To obtain an approximation guarantee under a fixed budget, the algorithm must explicitly handle the global interactions between labeled points.

**Goal**: The authors aim to answer an open question posed by Cohen-Addad et al.: whether there exists a polynomial-time approximation algorithm that strictly uses $k$ labels while competing with $\mathrm{OPT}_k$. This paper provides an affirmative answer with an approximation factor of $\tilde{O}(\log^{1.5} n)$.

**Key Insight**: Instead of point-wise greedy selection, the authors reduce the general graph to a binary tree using a tree cut sparsifier. They then transform the decision problem of "whether a label set is good enough for a threshold $\tau$" into a max-flow problem and utilize the tree structure for dynamic programming.

**Core Idea**: Use tree cut sparsification to preserve the sparsity of all cuts, and then use dynamic programming on the tree to select a set of leaf labels all at once, thereby capturing the global combinatorial effects among labels.

## Method
The methodology can be understood as a three-layer reduction. The first layer transforms GLS on general graphs into a Leaf Label Selection (LLS) problem on trees. The second layer converts the feasibility check (whether the worst-case sparsity reaches at least $\tau$) into a flow problem with source/sink edges. The third layer designs a DP for this flow problem using the binary tree structure and finds the optimal threshold via binary search. The theoretical approximation factor stems from the distortion of the tree cut sparsifier, while the subsequent DP solves the tree problem exactly.

### Overall Architecture
The input consists of a weighted undirected graph $G=(V,E,w)$ and a labeling budget $k$. The algorithm first constructs a tree cut sparsifier $T$ where the leaves correspond to the vertices of the original graph, such that any cut weight in the original graph is approximately preserved in the tree. Thus, GLS is transformed into LLS: select a leaf label set $|L|\leq k$ to maximize the worst-case tree cut sparsity $\widehat{\Psi}_T(L)$.

To solve LLS, the algorithm constructs a graph $T_{L,\tau}$ for a candidate threshold $\tau$: all leaves receive a capacity $\tau$ from a source, selected leaves are connected to a sink with infinite capacity, and tree edges retain their original capacities. The authors prove that $\widehat{\Psi}_T(L)\geq \tau$ if and only if the $s$-$t$ max-flow in this graph reaches $n\tau$. The problem thus becomes: select the minimum number of leaves as sinks such that all source flow is routable within the budget.

On a binary tree, this sink selection problem can be solved exactly using DP. The state $\mathrm{DP}[v][k]$ represents the maximum additional flow that can be injected from the subtree root $T_v$ given a budget of $k$ sinks while remaining routable. Internal nodes distribute the budget between left and right subtrees and use edge capacity functions to truncate the flow each subtree can handle. If $\mathrm{DP}[root][k]\geq 0$ at the root, the threshold $\tau$ is feasible for budget $k$. Binary search on $\tau$ yields the LLS solution.

### Key Designs
1. **Tree cut sparsifier reduction**:
    - **Function**: Transforms GLS on general graphs to LLS on tree leaves while losing only the approximation factor of the tree cut sparsifier.
    - **Mechanism**: Construct a tree $T$ whose leaf set is identical to the graph's vertices $V$, such that for any $A\subseteq V$, the graph cut weight $w_G(A,V\setminus A)$ is bounded by the minimum tree cut $\lambda_T(A, \mathcal{L}_T\setminus A)$ separating $A$ from its complement leaves, within an $\alpha$ factor expansion. Thus, $\Psi_G(L)\leq\widehat{\Psi}_T(L)\leq\alpha\Psi_G(L)$.
    - **Design Motivation**: The tree structure enables subsequent DP, while the cut sparsifier ensures the tree solution does not deviate too far from the original graph. Binarization of the tree normalizes the DP structure without changing approximation properties.

2. **Thresholded flow / sink selection decision**:
    - **Function**: Converts the non-linear minimum sparsity objective into a feasibility problem for a given threshold $\tau$.
    - **Mechanism**: In $T_{L,\tau}$, each leaf receives $\tau$ capacity from the source. Leaves selected as labels have infinite capacity to the sink. If a set of unlabeled leaves forms an overly sparse cut, the $s$-$t$ min-cut will be less than $n\tau$. Conversely, a max-flow of $n\tau$ implies all unlabeled sets have a tree cut sparsity $\geq \tau$.
    - **Design Motivation**: This equivalence compresses the global constraint (that all unlabeled subsets must be "good") into the language of max-flow/min-cut, providing a composable flow conservation perspective for the tree DP.

3. **Dynamic programming on trees and binary search**:
    - **Function**: Exactly solves for the minimum sinks needed under a fixed $\tau$ and determines if the budget $k$ is sufficient.
    - **Mechanism**: The leaf base case is simple: an unlabeled leaf contributes $-\tau$ net demand, while a labeled leaf can absorb arbitrary flow. Internal nodes enumerate budget allocations $a$ and $k-a$, using $\mathrm{bound}_{(v,c)}(x)$ to truncate the routable flow based on edge capacity. The total state space is $O(nk)$, and with budget enumeration, the DP runtime is $O(nk^2)$.
    - **Design Motivation**: While standard max-flow is hard for the combinatorial "sink selection" problem on general graphs, the tree structure allows subproblems to exchange flow only through a single edge, letting DP capture global interactions completely.

### Loss & Training
This is a combinatorial optimization algorithm rather than a neural network training process. The objective is to maximize $\Psi(L)=\min_{C\subseteq V\setminus L} w(C,V\setminus C)/|C|$, ensuring no unlabeled cluster is both large and isolated. The theoretical algorithm uses binary search on $\tau$. In the experimental implementation, due to a lack of open-set tree cut sparsifiers, the authors use hierarchical decomposition heuristics like Fiedler vectors and METIS.

## Key Experimental Results

### Main Results
The primary theoretical result is the fixed-budget approximation guarantee. The experiments demonstrate the speed advantages of the heuristic tree-decomposition version on real SNAP graphs. The table below summarizes theoretical and runtime results for `ca-GrQc`.

| Dataset / Setting | Metric | Ours | Prev. SOTA / Baseline | Gain |
|-------------------|------|------|-----------------|------|
| Any weighted graph, budget $k$ | Approx. Guarantee | $\tilde{O}(\log^{1.5} n)$, with $|L|\leq k$ | Cohen-Addad et al. $O(\log n)$ resource augmentation | First polynomial approx w/o budget relaxation |
| ca-GrQc, $k=10$ | Real time | 22s (Ours Fiedler) | 144s Guillory-Bilmes / 4967s Cohen-Addad | ~6.5x / 225x faster |
| ca-GrQc, $k=50$ | Real time | 21s (Ours Fiedler) | 842s Guillory-Bilmes / 15586s Cohen-Addad | ~40x / 742x faster |
| ca-GrQc, $k=100$ | Real time | 22s (Ours Fiedler) | 1835s Guillory-Bilmes / 22956s Cohen-Addad | ~83x / 1043x faster |
| com-dblp, 317,080 nodes | $\Psi$ Quality | 0.030 / 0.048 / 0.083 for $k=50/500/5000$ | Most baselines fail to scale | Practical solutions on large graphs |

### Ablation Study
The primary analysis focuses on the impact of different sparse-cut heuristics on quality and scalability. The authors do not claim these heuristics maintain worst-case theoretical guarantees, but they demonstrate a practical version of the theoretical framework.

| Configuration | Key Metrics | Description |
|------|---------|------|
| Fiedler sweep | Quality near existing methods on ca-GrQc, ~20s runtime | Spectral methods provide stable cut quality but require Fiedler vector computation |
| FiedlerBalanced, $\beta\in\{0.01,0.1\}$ | Faster speed, lower quality | Enforced balanced tree decomposition reduces recursion depth but sacrifices cut quality |
| METIS, samples $\in\{\sqrt n,10\sqrt n,100\sqrt n\}$ | Scalable to com-dblp, finished in hours with $\sqrt n$ samples | Multiple METIS runs with varied weights maintain utility on large graphs |
| Original tree cut sparsifier | Theoretical $\tilde{O}(\log^{1.5} n)$ factor | Theoretical version is separated from engineering version due to lack of open-source implementation |

### Key Findings
- The fixed-budget approximation is the most significant theoretical breakthrough: it avoids requiring "extra labels," which is often unrealistic in active learning.
- The tree DP runtime is primarily $O(nk^2)$; thus, when $k$ is not excessively large, it has higher scalability potential than iteratively solving complex flow/cut problems on general graphs.
- Experimental quality is strongly correlated with the sparse-cut heuristic. Fiedler and METIS perform well overall, while Balanced versions trade quality for speed. This indicates the practical bottleneck has shifted from "how to select labels" to "how to construct a good tree decomposition."

## Highlights & Insights
- The most ingenious aspect is transforming the budget-constrained selection problem into "selecting sinks to make flow routable." This perspective treats each label's role as absorbing flow rather than locally improving a greedy score.
- The paper clearly distinguishes between the theoretical algorithm and the engineering implementation. Theoretically, it relies on tree cut sparsifiers; experimentally, it admits the lack of implementation and uses Fiedler/METIS for proof-of-concept—an honest approach compared to masquerading heuristics as theory.
- The design of the DP states is highly reusable. Many active learning or representative selection problems under tree decompositions could use "subtree injectable/absorbable flow" as a composable state.
- For graph active learning, this paper serves as a reminder: if the objective is not submodular, stacking greedy heuristics may not yield guarantees. Switching to an equivalent decision problem can open up new algorithmic spaces.

## Limitations & Future Work
- The experiments do not use an actual worst-case tree cut sparsifier implementation; thus, the experimental version does not strictly inherit the $\tilde{O}(\log^{1.5} n)$ guarantee.
- The runtime tables are not strictly controlled benchmarks (as the authors note other processes were running and implementation was unoptimized); the results should be viewed as order-of-magnitude trends.
- The objective function focuses on graph cut sparsity, ideal for label propagation on similarity graphs. If real-world tasks involve heavy label noise or class imbalance, graph structure alone may be insufficient.
- The DP has a $k^2$ dependency on the budget. Further optimization—such as approximate DP, parallelized METIS sampling, or incremental updates—is needed for very large budgets or dynamic graphs.

## Related Work & Insights
- **vs Guillory and Bilmes**: Early work proposed the GLS objective with practical heuristics; this paper addresses it from an approximation perspective for fixed budgets.
- **vs Cesa-Bianchi et al.**: They provide guarantees on specific structures like unweighted trees; this work extends the applicability via tree cut sparsifiers for general graphs.
- **vs Cohen-Addad et al.**: Previous work provided resource augmentation and left the fixed-budget setting as an open problem; this paper directly answers it (at the cost of the tree cut sparsifier factor).
- **Insight**: For graph selection problems, seeking a cut-preserving tree or hierarchical decomposition and then performing exact DP on that structure is a viable route that may yield provable results more easily than designing greedy rules on original graphs.

## Rating
- Novelness: ⭐⭐⭐⭐⭐ First polynomial-time approximation for GLS without resource augmentation; clear pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple SNAP graphs but lacks implementation of the theoretical sparsifier.
- Writing Quality: ⭐⭐⭐⭐☆ Complete reduction chain and clear proofs; pseudo-code is compact and requires careful reading alongside the text.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for graph active learning, representative selection, and graph combinatorial optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Posterior Label Smoothing for Node Classification](../../AAAI2026/graph_learning/posterior_label_smoothing_for_node_classification.md)
- [\[ICML 2026\] Identifying and Correcting Label Noise for Robust GNNs via Influence Contradiction](identifying_and_correcting_label_noise_for_robust_gnns_via_influence_contradicti.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](../../AAAI2026/graph_learning/echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)
- [\[AAAI 2026\] RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](../../AAAI2026/graph_learning/rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)
- [\[ICML 2026\] Aitchison Embeddings for Learning Compositional Graph Representations](aitchison_embeddings_for_learning_compositional_graph_representations.md)

</div>

<!-- RELATED:END -->
