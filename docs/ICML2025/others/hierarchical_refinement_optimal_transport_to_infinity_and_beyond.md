---
title: >-
  [Paper Note] Hierarchical Refinement: Optimal Transport to Infinity and Beyond
description: >-
  [ICML 2025 Oral][Optimal Transport] Proposes the Hierarchical Refinement (HiRef) algorithm, which dynamically constructs multi-scale data partitions by recursively solving low-rank optimal transport subproblems to obtain a full bijective Monge map in log-linear time and linear space complexity, scaling optimal transport to million-scale datasets.
tags:
  - "ICML 2025 Oral"
  - "Optimal Transport"
  - "Low-Rank Factorization"
  - "Hierarchical Refinement"
  - "Monge Map"
  - "Large-Scale Alignment"
date: 2026-05-08
content_hash: 509c0ad50606b90d
---

# Hierarchical Refinement: Optimal Transport to Infinity and Beyond

**Conference**: ICML 2025 Oral  
**arXiv**: [2503.03025](https://arxiv.org/abs/2503.03025)  
**Code**: None  
**Area**: Optimal Transport / Algorithm Optimization  
**Keywords**: Optimal Transport, Low-Rank Factorization, Hierarchical Refinement, Monge Map, Large-Scale Alignment

## TL;DR
Proposes the Hierarchical Refinement (HiRef) algorithm, which dynamically constructs multi-scale data partitions by recursively solving low-rank optimal transport subproblems to obtain a full bijective Monge map in log-linear time and linear space complexity, scaling optimal transport to million-scale datasets.

## Background & Motivation

**Background**: Optimal Transport (OT) is widely used in machine learning for data alignment, with the Sinkhorn algorithm being the most common solver, possessing $O(n^2 \log n)$ time and $O(n^2)$ space complexity. Low-rank OT reduces the space complexity to $O(nr)$ but fails to yield a one-to-one point correspondence.

**Limitations of Prior Work**: (1) The quadratic space complexity of Sinkhorn limits its capability to process datasets beyond ~$10^4$ points; (2) Although low-rank OT is efficient, it only yields approximate couplings with limited resolution and cannot recover exact bijective maps; (3) Mini-batch OT introduces systematic bias and does not compute global alignment; (4) Neural OT methods have limitations in recovering faithful maps.

**Key Challenge**: Full-rank OT produces exact Monge maps but incurs quadratic computational cost, while low-rank OT is computationally efficient but fails to provide one-to-one correspondences—requiring a method that combines the benefits of both.

**Goal**: Design an algorithm that recovers a full bijective Monge map and scales to million-scale data while maintaining the linear space complexity of low-rank OT.

**Key Insight**: The authors observe that the optimal factor matrices $(Q^*, R^*)$ of low-rank OT possess a key invariant: they assign each source point and its image under the Monge map to the same cluster. This implies that low-rank OT can be leveraged to correctly "co-cluster" source-target point pairs.

**Core Idea**: Recursively partition the data using low-rank OT into increasingly finer co-clusters until each cluster contains exactly one point pair, thereby recovering the full Monge map—with equivalent performance to full-rank OT but under log-linear complexity.

## Method

### Overall Architecture
HiRef adopts a top-down, multi-scale strategy. Initially, the entire source dataset $\mathsf{X}$ and target dataset $\mathsf{Y}$ form a coarse-grained co-cluster. At each layer, a low-rank OT subproblem is solved for each co-cluster to further subdivide it into $r$ smaller co-clusters. After $\kappa = \log_r n$ layers of recursive refinement, each co-cluster contains exactly one $(x_i, T(x_i))$ point pair, thus recovering the Monge map.

### Key Designs

1. **Co-Clustering Invariant (Proposition 3.1)**:

    - **Function**: Provides theoretical guarantees for hierarchical refinement—ensuring that low-rank OT at each layer correctly partitions Monge pairs into the same cluster.
    - **Mechanism**: Proves that when the optimal low-rank factors $(Q^*, R^*)$ correspond to hard clustering, the cluster label of a source point $q^*(x)$ equals the cluster label of its Monge image $r^*(T^*(x))$. The key condition is that the cost matrix satisfies "strict $r$-Monge separability", under which the optimal factors can factorize into a symmetric form $Q = P^\dagger R$ (where $P^\dagger$ is the Monge coupling).
    - **Design Motivation**: This theoretical result is the cornerstone of the algorithm's correctness—if low-rank OT cannot correctly co-cluster Monge pairs, recursive refinement cannot converge to the correct map.

2. **Rank-Annealing Schedule**:

    - **Function**: Controls the refinement granularity at each layer, balancing computational efficiency and memory constraints.
    - **Mechanism**: Selects a sequence of factors $(r_1, r_2, \ldots, r_\kappa)$ such that $\prod r_i = n$. Since rank is inversely related to the entropy regularization parameter $\epsilon$, progressing from small to large ranks is equivalent to annealing from high to low $\epsilon$. An optimal schedule is found via dynamic programming in $O(r_{\max} \kappa n)$ time to minimize the cumulative total rank of subproblems.
    - **Design Motivation**: The choice of rank directly impacts complexity and correctness. A uniform rank constraint $g = 1_r/r$ ensures uniform partitioning; a binary schedule ($r=2$) automatically satisfies the hard partitioning condition.

3. **Hierarchical Block Coupling and Diminishing Costs (Proposition 3.4)**:

    - **Function**: Guarantees that each refinement layer improves the transport cost.
    - **Mechanism**: Defines an implicit block coupling $P^{(t)}$ at scale $t$, proving that the cost difference between adjacent layers satisfies $\Delta_{t,t+1} \geq 0$ and $\Delta_{t,t+1} \leq \|\nabla c\|_\infty \cdot \text{mean\_diam}(\Gamma_t)$. This implies refinement always reduces transport cost, with the improvement bounded by the mean diameter of the current clusters.
    - **Design Motivation**: Provides theoretical guarantees for the algorithm's convergence and solution quality.

### Loss & Training
HiRef is a purely algorithmic (non-learning) method that does not involve training. The core optimization objective is the low-rank OT subproblem at each layer: $\min_{Q,R} \langle C, Q \text{diag}(1/g) R^\top \rangle_F$, subject to a uniform distribution constraint on $g$.

## Key Experimental Results

### Main Results

| Dataset | HiRef Cost | Sinkhorn Cost | Remarks |
|--------|-----------|--------------|------|
| Half-Moon (1K) | Optimal/Near-optimal | Close | Comparable, but HiRef outputs exactly 1024 non-zero entries |
| Half-Moon (1M) | 14.2 | **Cannot Run** | HiRef scales to $2^{20}$ data size |
| MOSTA E12-13.5 (51K) | **14.35** | - | Sinkhorn out of memory (OOM) |
| MOSTA E15-16.5 (121K) | **12.79** | - | Only HiRef and low-rank/mini-batch are executable |

### Ablation Study

| Method Comparison | Cost Quality | Scalability | Notes |
|---------|---------|---------|------|
| HiRef vs Sinkhorn | Comparable or better | HiRef >> | HiRef scales up to millions, Sinkhorn limited to ~16K |
| HiRef vs LOT/FRLC (r=40) | HiRef better | Comparable | Refinement step reduces the coarse cost of low-rank OT |
| HiRef vs Mini-batch | HiRef better | Comparable | HiRef is unbiased, mini-batch has systematic bias |
| HiRef vs MOP | HiRef much better | Comparable | MOP cost is 2x higher on Checkerboard |

### Key Findings
- HiRef outputs a strictly bijective map (exactly $n$ non-zero entries in the coupling matrix), whereas Sinkhorn/ProgOT outputs 620k–680k non-zero entries—occupying most of the dense matrix space.
- On single-cell transcriptomic data, HiRef achieves lower transport costs across all time points compared to other methods, including FRLC which serves as its subprocess.
- Execution time scales linearly, whereas Sinkhorn scales quadratically; successfully scaled to >1M points.

## Highlights & Insights
- **Elegant Fusion of Theory and Algorithm**: The co-clustering invariant in Proposition 3.1 translates the algebraic structure of low-rank OT into recursively exploitable divide-and-conquer guarantees, making the design highly elegant.
- **Rank-Entropy Duality Insight**: The equivalence of low-rank $\leftrightarrow$ high regularization makes hierarchical refinement conceptually equivalent to $\epsilon$-annealing, but more structured and memory-efficient in execution.
- In practical large-scale applications like transcriptomics, HiRef achieves full-rank alignment for >100k points for the first time, bridging the gap between low-rank approximation and exact solving.

## Limitations & Future Work
- Theoretical guarantees rely on the "strict $r$-Monge separability" assumption, which empirical data may not satisfy; the algorithm remains effective in practice but lacks a general proof.
- When the cost matrix cannot undergo low-rank factorization ($d = O(n)$), complexity degenerates to $O(n^2)$.
- The uniform rank constraint $g = 1_r/r$ forces uniform split at each layer, which may perform poorly on highly imbalanced data distributions.
- Currently processes only equal-sized datasets ($|X| = |Y|$); extensions to non-bijective scenarios are required.

## Related Work & Insights
- **vs Sinkhorn**: Sinkhorn is the $O(n^2)$ gold standard but unscalable; HiRef achieves comparable or better transport costs in log-linear time.
- **vs Low-Rank OT (Scetbon et al.)**: Low-rank OT serves as a subprogram for HiRef; HiRef transcends the limited resolution of low-rank approximations via hierarchical refinement.
- **vs Gerber & Maggioni (2017)**: Both are multi-scale OT methods, but MOP requires predefined multi-scale partitions and relies on ambient space grids; HiRef constructs partitions automatically and is fully data-driven.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical discovery of the co-clustering invariant provides brilliant structural insight, perfectly bridging low-rank and full-rank OT.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covering synthetic and transcriptomic data ranging from small to million-scale, with comprehensive comparisons against various baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and highly illustrative figures/tables.
- Value: ⭐⭐⭐⭐⭐ Scales full-rank OT to the million-scale for the first time, bearing significant practical relevance for fields like computational biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](../../ICCV2025/others/lacoot_layer_collapse_through_optimal_transport.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [\[NeurIPS 2025\] Estimation of Stochastic Optimal Transport Maps](../../NeurIPS2025/others/estimation_of_stochastic_optimal_transport_maps.md)
- [\[NeurIPS 2025\] Bispectral OT: Dataset Comparison using Symmetry-Aware Optimal Transport](../../NeurIPS2025/others/bispectral_ot_dataset_comparison_using_symmetry-aware_optimal_transport.md)
- [\[ACL 2025\] Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](../../ACL2025/others/quantifying_lexical_semantic_shift_via_unbalanced_optimal_transport.md)

</div>

<!-- RELATED:END -->
