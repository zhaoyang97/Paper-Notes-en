---
title: >-
  [Paper Note] TANGO: Clustering with Typicality-Aware Nonlocal Mode-Seeking and Graph-Cut Optimization
description: >-
  [ICML2025][Density clustering] Proposes the concept of "typicality" to quantify the confidence of data points serving as modes (cluster centers) from a global perspective. Combined with an improved path similarity and graph-cut optimization, it achieves automatic mode detection and clustering without manual threshold tuning.
tags:
  - "ICML2025"
  - "Density clustering"
  - "Mode seeking"
  - "Typicality"
  - "Graph-cut optimization"
  - "Spectral clustering"
date: 2026-05-08
content_hash: b8b960baf4646fe8
---

# TANGO: Clustering with Typicality-Aware Nonlocal Mode-Seeking and Graph-Cut Optimization

**Conference**: ICML2025  
**arXiv**: [2408.10084](https://arxiv.org/abs/2408.10084)  
**Code**: [GitHub](https://github.com/SWJTU-ML/TANGO_code)  
**Area**: Clustering  
**Keywords**: Density clustering, Mode seeking, Typicality, Graph-cut optimization, Spectral clustering

## TL;DR

Proposes the concept of "typicality" to quantify the confidence of data points serving as modes (cluster centers) from a global perspective. Combined with an improved path similarity and graph-cut optimization, it achieves automatic mode detection and clustering without manual threshold tuning.

## Background & Motivation

Density clustering is a key method for processing complex data distributions. Quick Shift and DPC (Density Peaks Clustering) are two representative mode-seeking methods:

- **Quick Shift**: Connects each point to its nearest neighbor with higher density, and partitions dependencies using a distance threshold to obtain modes and clusters. However, it is highly sensitive to the threshold parameter and easily misidentifies outliers as modes.
- **DPC**: Selects cluster centers via a decision graph of density and distance, but still relies on manual threshold setting or manual operations.

**Core Problem**: Existing methods rely solely on local information (density, local distances) to determine whether a point is a mode. However, whether a point can serve as a mode is actually influenced by the **global data structure**. For example, for the same local density peak point $P$, it should be identified as a mode when surrounded by a large number of dependent points, but should be merged into a neighboring cluster when it has only a few dependent points. Existing methods cannot handle such scenarios automatically, requiring per-dataset parameter tuning or manual intervention.

## Method

### 1. Typicality Definition

The core innovation of TANGO is to introduce **typicality** $T_i$ to quantify the confidence of point $x_i$ serving as a mode. The design follows two axioms:

- **(P1)** Points with higher density should have higher typicality.
- **(P2)** All points that eventually aggregate to a point through density-ascent dependencies, along with their dependency strengths, should contribute to the typicality.

The recursive definition is:

$$T_i = \rho_i + \sum_j B_{ji} T_j$$

where $\rho_i$ is the density of point $x_i$, and $B_{ji}$ is the dependency weight from $x_j$ to $x_i$ in the dependency matrix. The vector form is $T = B^\top T + \rho$, and the closed-form solution is $T = (I - B^\top)^{-1}\rho$.

> Connection to PageRank: Data points can be viewed as web pages, and density-ascent dependencies as hyperlinks. Thus, typicality is analogous to a specialized PageRank centrality.

### 2. Hierarchical Dependency-Based Similarity and Density

**Similarity (based on Shared Nearest Neighbors, SNN)**:

$$A(x_i, x_j) = \sum_{p \in \text{SNN}_k(x_i, x_j)} \exp\left(-\left(\frac{d(p,x_i)+d(p,x_j)}{2d_{\max}}\right)^2\right)$$

This is non-zero only if $x_i \in N_k(x_j)$ or $x_j \in N_k(x_i)$. This definition considers the varying contributions of different shared nearest neighbors, enhancing robustness.

**Density (Normalized)**: $\rho(x_i) = \frac{\sum_{p \in L(x_i)} A(x_i, p)}{\max_{x} \sum_{p \in L(x)} A(x, p)}$

### 3. Leader Relationships and Dependency Matrix

The **leader** of each point is its neighbor with higher density and the highest similarity. To eliminate bias in dense/sparse regions, the dependency strength is defined using the **rank** of the leader in the similarity-sorted neighbor list, rather than the direct similarity:

$$B_{ij} = \exp\left(-\left(\frac{\text{rank}(x_i)}{\max(\text{rank})}\right)^2\right), \quad \text{if } x_j = \text{leader}(x_i)$$

**Theorem 1**: For $B$ defined above, there exists a unique typicality vector $T$ satisfying $T = B^\top T + \rho$.

**Theorem 2**: When sorted in ascending order of density, $T$ can be computed via forward substitution in $O(n)$ time.

### 4. Typicality-Aware Mode Seeking

For point $x_i$ and its leader $x_j$:

- If $T(x_i) < T(x_j)$: Retain the dependency ($x_j$ can represent $x_i$).
- If $T(x_i) \geq T(x_j)$: Sever the dependency, and $x_i$ becomes a mode.

Intuition: A point with higher typicality should not be represented by a point with lower typicality. This criterion is **completely automatic** and requires no manual threshold.

### 5. Path Similarity and Graph-Cut Aggregation

Mode detection partitions the data into multiple tree-like sub-clusters. The relationship between sub-clusters is measured using an **improved path similarity**:

$$\text{PBSim}(r_i, r_j) = \max_{P \in \mathcal{P}_{ij}} \left\{ \min_{1 \leq h < |P|} C(p_h, p_{h+1}) \right\}$$

where the edge weight $C(p_h, p_{h+1}) = A(p_h, p_{h+1}) \cdot \rho(p_h) \cdot \rho(p_{h+1})$ (cross-sub-cluster) and is set to 1 within the same sub-cluster. This is efficiently computed using a variant of Kruskal's algorithm, with complexity $O(nk\log(nk))$.

Finally, **spectral clustering (Normalized Cut)** is applied to the path similarity matrix of the sub-clusters to obtain the final clustering result.

### 6. Overall Complexity

The overall time complexity is $O(nk^2d)$ ($k, q \ll n$), where the bottleneck lies in the similarity matrix calculation accelerated by KD-Tree, which can be parallelized.

## Key Experimental Results

### Synthetic Data Visualization

On 4 synthetic datasets, TANGO automatically detects modes and correctly clusters the data via typicality, while Quick Shift and DPC perform poorly because they only consider local structures.

### Comparison on Real Datasets (16 Datasets, 10 Comparison Methods)

| Dataset | Size n | Dimension d | Clusters | Method Description |
|--------|--------|--------|------|----------|
| wdbc | 569 | 30 | 2 | Medical diagnosis |
| semeion | 1593 | 256 | 10 | Handwritten digits |
| waveform | 5000 | 21 | 3 | Waveform classification |
| isolet1234 | 6238 | 617 | 26 | Speech alphabets |
| MNIST (AE) | 10000 | 64 | 10 | Handwritten digits |
| Umist (AE) | 575 | 64 | 20 | Faces |

The comparison methods include LDP-SC, DCDP-ASC, LDP-MST, NDP-Kmeans, DEMOS, CPF, QKSPP, DPC-DBFN, USPEC, KNN-SC. Evaluation metrics are ARI, NMI, ACC.

**Statistical Tests**: Friedman test p < 0.05. In the subsequent Nemenyi test, the average rank differences between TANGO and all comparison methods exceeded the critical development (CD) threshold of 2.30, demonstrating statistically significant advantages.

**Image Segmentation Experiments**: Clustering was performed on 154,401 pixels (5-dimensional features: 2 spatial + 3 RGB) of the Berkeley Segmentation Benchmark, verifying both effectiveness and computational efficiency.

## Highlights & Insights

1. **Highly Innovative Concept**: The concept of "typicality" elevates the local density dependency to a global perspective. By recursively weighting and aggregating density information from downstream nodes, it elegantly solves the global-vs-local dilemma in mode detection.
2. **Fully Automatic**: It eliminates the mode detection thresholds heavily present in Quick Shift / DPC methods, requiring only a single parameter $k$ (the number of nearest neighbors).
3. **Rich Theoretical Guarantees**: Proves the uniqueness of typicality (Theorem 1), $O(n)$ computational efficiency (Theorem 2), efficient calculation of path similarity (Theorem 3), and power-law tail behavior of typicality.
4. **Connection to PageRank**: Maps the local dependencies in density clustering to a random walk / PageRank framework on graphs, providing a fresh theoretical perspective.
5. **Sound Framework Design**: The local-to-global two-stage strategy—first partitioning sub-clusters via typicality, then aggregating them using path similarity + spectral clustering—balances local structure preservation with global optimal partitioning.

## Limitations & Future Work

1. **Automatic Selection of Parameter $k$**: The number of nearest neighbors $k$ still needs to be searched (2~100). The paper does not provide an automatic determination method, which the authors also note for future work.
2. **Requires Presetting the Number of Clusters**: The final spectral clustering stage requires specifying the number of target clusters, limiting its applicability as a fully unsupervised method.
3. **KD-Tree Degradation on High-Dimensional Data**: The similarity calculation relies on KD-Tree, the efficiency of which degenerates in high-dimensional spaces.
4. **Image Data Requires Preprocessing**: MNIST and Umist require dimensionality reduction to 64 dimensions using an AutoEncoder; the method is not directly applicable to raw high-dimensional images.
5. **Density Weighting of Path Similarity**: Cross-cluster connections in low-density areas may be overly penalized, which might be disadvantageous for clusters connected by wide but sparse bridge areas.

## Related Work & Insights

- The evolution from **Quick Shift → DPC → TANGO** is clear: from purely local to introducing global information.
- The recursive definition of typicality + PageRank perspective can be extended to other scenarios requiring local-to-global information propagation (e.g., node importance in Graph Neural Networks).
- The sub-cluster aggregation strategy of path similarity + spectral clustering can serve as a general post-processing module applicable to other methods that produce over-segmented results.

## Rating
- Novelty: ⭐⭐⭐⭐ (Highly original typicality concept and its theoretical analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐ (16 datasets + 10 comparison methods + statistical tests + image segmentation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-balanced between theory and intuitive explanation)
- Value: ⭐⭐⭐⭐ (Provides an effective solution for automated mode detection in density clustering)** The higher the self-density of a point, the greater its confidence of serving as a mode.
- **(P2)** The more descendant points aggregate to this point through dependency relationships, and the stronger the dependencies, the greater the confidence.

Recursive definition:

$$T_i = \rho_i + \sum_j B_{ji} T_j$$

where $B_{ji} \neq 0$ if and only if $x_j$ depends on $x_i$. In vector form, $T = B^\top T + \rho$, i.e., $T = (I - B^\top)^{-1} \rho$.

### Similarity and Dependency Matrix

**Shared Nearest Neighbor Similarity (Definition 1)**: For two points $x_i, x_j$, a weighted sum based on their k-nearest neighbor intersection:

$$A(x_i, x_j) = \sum_{p \in \text{SNN}_k(x_i, x_j)} \exp\!\Big(-\Big(\frac{d(p, x_i) + d(p, x_j)}{2d_{\max}}\Big)^2\Big)$$

**Density (Definition 2)**: For $x_i$, sum the similarities of its k most similar neighbors and normalize: $\rho(x_i) = \frac{\sum_{p \in L(x_i)} A(x_i, p)}{\max_x \sum_{p \in L(x)} A(x, p)}$

**Leader Relationship (Definition 3)**: Each point depends on its neighbor with higher density and the highest similarity.

**Dependency Weight (Eq. 6)**: Defined by the rank of the leader in the neighbor ranking to avoid bias in dense regions:

$$B_{ij} = \exp\!\Big(-\Big(\frac{\text{rank}(x_i)}{\max(\text{rank})}\Big)^2\Big), \quad \text{if } x_j = \text{leader}(x_i)$$

### Typicality-Aware Mode Seeking (Algorithm 2)

1. Initialize $T(x_i) \leftarrow \rho(x_i)$
2. Traverse in ascending order of density, propagating to the leader: $T(x_j) \leftarrow T(x_j) + T(x_i) \cdot B_{ij}$
3. Mode determination: If $T(x_i) \geq T(\text{leader}(x_i))$, then $x_i$ is a mode (i.e., points with high confidence should not be represented by leaders with lower confidence)

**Key Theorems**:
- **Uniqueness (Theorem 1)**: The typicality vector $T$ exists and is unique.
- **Efficiency (Theorem 2)**: When sorted by density, $T$ can be solved using forward substitution in $O(n)$ time.

### Sub-cluster Aggregation: Improved Path Similarity + Spectral Clustering

**Path Similarity (Definition 5)**: The similarity between two sub-clusters is the "maximum-minimum connectivity" among all paths connecting their modes:

$$\text{PBSim}(r_i, r_j) = \max_{P \in \mathcal{P}_{ij}} \min_{1 \leq h < |P|} C(p_h, p_{h+1})$$

where edge weight $C = 1$ within the same sub-cluster, and cross-sub-cluster edge weight $C(p_h, p_{h+1}) = A(p_h, p_{h+1}) \cdot \rho(p_h) \cdot \rho(p_{h+1})$. Solved in $O(nk \log(nk))$ using a Kruskal variant.

Finally, perform **Normalized Cut spectral clustering** on the graph composed of sub-cluster modes.

### Overall Time Complexity

$O(nk^2d)$, dominated by the similarity matrix calculation (which can be parallelized), where $k, q \ll n$.

## Key Experimental Results

### Dataset Overview

| Dataset | Samples $n$ | Dimension $d$ | Clusters |
|---|---|---|---|
| wdbc | 569 | 30 | 2 |
| segmentation | 2100 | 19 | 7 |
| semeion | 1593 | 256 | 10 |
| waveform | 5000 | 21 | 3 |
| isolet1234 | 6238 | 617 | 26 |
| MNIST (AE) | 10000 | 64 | 10 |
| Umist (AE) | 575 | 64 | 20 |

### Main Results

- Compared with **10 SOTA methods** (LDP-SC, DCDP-ASC, LDP-MST, NDP-Kmeans, DEMOS, CPF, QKSPP, DPC-DBFN, USPEC, KNN-SC) on **16 datasets**.
- Evaluation metrics: ARI, NMI, ACC.
- Friedman test $p < 0.05$; in the Nemenyi post-hoc test, the average rank differences between TANGO and all comparison methods exceeded the critical difference CD = 2.30, **demonstrating statistically significant superiority over all comparison methods**.
- Synthetic data visualization shows that TANGO outperforms Quick Shift and DPC on complex manifold structures (such as rings, spirals, and heterogeneous densities).
- Image segmentation tasks (Berkeley dataset, 154,401 pixels) also verified the effectiveness and computational efficiency.

### Single Hyperparameter

Only needs to set the number of neighbors $k$ (search range 2~100, step size 1), without manually selecting modes or setting thresholds.

## Highlights & Insights

1. **Elegant Design of Typicality**: It reinterprets locally defined density dependencies from a global perspective, with a deep connection to PageRank—where it can be viewed as information propagation on a dependency graph with density as weights.
2. **Rigorous Theory**: Proves properties such as the uniqueness of typicality, $O(n)$ computability, and the power-law tail distribution.
3. **Elimination of Manual Intervention**: No longer requires manual threshold tuning or checking decision graphs to select modes, significantly improving automation.
4. **Two-Stage Decoupling**: Over-segmentation followed by graph-cut aggregation balances local refinement and global optimality (similar to the superpixel + graph-cut image segmentation paradigm).
5. **Only One Hyperparameter $k$**: Easier to tune compared to DPC series methods.

## Limitations & Future Work

1. **Requires Presetting the Number of Clusters**: The spectral clustering stage still needs to specify the target number of clusters. Future work can combine automatic cluster number determination methods (such as eigenvalue gaps).
2. **Selection of $k$**: Although there is only one hyperparameter, the optimal $k$ still needs to be searched, and the paper does not provide an adaptive selection strategy.
3. **High-Dimensional Data**: SNN similarity and KD-Tree may degrade on extremely high-dimensional data (curse of dimensionality).
4. **Large-Scale Scaling**: $O(nk^2d)$ is still challenging for million-scale data. Although similarity computation can be parallelized, the spectral clustering part $O(q^3)$ also needs attention when the number of sub-clusters is large.
5. **Dependency on a Single Leader**: Each point depends on only one leader (the most similar higher-density neighbor), which may not be robust enough in density plateaus or symmetric structures.

## Related Work & Insights

- **Quick Shift** → The direct target of improvement in this paper. TANGO retains its dependency structure but replaces the threshold with typicality.
- **DPC (Rodriguez & Laio, 2014)** → Semi-automation of decision graphs vs. full automation of TANGO.
- **LDP-SC / DCDP-ASC** → Both belong to the "over-segmentation followed by aggregation" paradigm, but TANGO innovates in mode detection and path similarity.
- **PageRank** → The theoretical foundation of typicality. Insight: Information propagation on graphs + density estimation can yield powerful global indicators.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel typicality concept, elevating mode-seeking from a local to a global perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐ (16 datasets + 10 comparison methods + statistical tests + ablation + image segmentation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with definitions, theorems, and algorithms, intuitive illustrations)
- Value: ⭐⭐⭐⭐ (Addresses the key pain point of density clustering relying on manual intervention for mode detection)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees](../../NeurIPS2025/others/fsnet_feasibility-seeking_neural_network_for_constrained_optimization_with_guara.md)
- [\[ICML 2025\] Exploiting Similarity for Computation and Communication-Efficient Decentralized Optimization](exploiting_similarity_for_computation_and_communication-efficient_decentralized_.md)
- [\[ICML 2025\] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)
- [\[NeurIPS 2025\] Generalized Linear Mode Connectivity for Transformers](../../NeurIPS2025/others/generalized_linear_mode_connectivity_for_transformers.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)

</div>

<!-- RELATED:END -->
