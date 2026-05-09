---
title: >-
  [Paper Note] Robust Graph Condensation via Classification Complexity Mitigation
description: >-
  [NeurIPS 2025][AI Safety][Graph Condensation] This paper reveals that graph condensation (GC) is fundamentally a process of reducing classification complexity, and that adversarial attacks precisely undermine this property. Based on this insight, the authors propose the MRGC framework, which enhances GC robustness through three manifold-based regularization modules: intrinsic dimensionality regularization, curvature-aware manifold smoothing, and inter-class manifold decoupling. This work represents the first systematic study of GC robustness under simultaneous perturbations of structure, features, and labels.
tags:
  - NeurIPS 2025
  - AI Safety
  - Graph Condensation
  - Adversarial Robustness
  - Classification Complexity
  - Manifold Learning
  - Graph Neural Networks
date: 2026-05-08
content_hash: cd1fcf1c3b1f0a82
---

# Robust Graph Condensation via Classification Complexity Mitigation

**Conference**: NeurIPS 2025
**arXiv**: [2510.26451](https://arxiv.org/abs/2510.26451)
**Code**: [GitHub](https://github.com/RingBDStack/MRGC)
**Area**: AI Security
**Keywords**: Graph Condensation, Adversarial Robustness, Classification Complexity, Manifold Learning, Graph Neural Networks

## TL;DR

This paper reveals that graph condensation (GC) is fundamentally a process of reducing classification complexity, and that adversarial attacks precisely undermine this property. Based on this insight, the authors propose the MRGC framework, which enhances GC robustness through three manifold-based regularization modules: intrinsic dimensionality regularization, curvature-aware manifold smoothing, and inter-class manifold decoupling. This work represents the first systematic study of GC robustness under simultaneous perturbations of structure, features, and labels.

## Background & Motivation

Graph condensation (GC) improves GNN training efficiency by compressing large graphs into smaller synthetic ones, and has been applied in settings such as neural architecture search and continual graph learning. However, existing GC methods **assume the original graph is clean**. When the original graph is corrupted by adversarial attacks, the quality of the condensed graph degrades substantially.

The authors systematically address three key questions:

**Q1: Can existing robust graph learning techniques improve GC robustness?** The answer is no. Integrating robust GNNs (e.g., MedianGCN) as the GC backbone or applying them during training on the condensed graph both perform worse than standard GC methods. This is because most robust GNNs rely on attention mechanisms, which perform poorly in the GC setting.

**Q2: What critical property of GC do attacks destroy?** Through the lens of classification complexity theory, the authors find that GC is intrinsically a **classification complexity reduction** process — after condensation, three complexity metrics (intrinsic dimensionality, Fisher discriminant ratio, hyperspherical coverage ratio) decrease by an average of 89.25%. Adversarial attacks, however, cause these metrics to increase by an average of 547.54% in the condensed graph, effectively **reversing the complexity reduction** achieved by GC.

**Q3: How can a general defense strategy be designed?** Protecting the complexity-reducing capability of GC requires three complementary directions: reducing intrinsic dimensionality, simplifying class boundary complexity, and eliminating inter-class ambiguity.

## Method

### Overall Architecture

MRGC is a plug-and-play framework that augments existing GC methods (e.g., GCond) with three manifold learning regularization modules. The total loss is:
$$\mathcal{L} = \mathcal{L}_{GC} + \alpha\mathcal{L}_{dim} + \beta\mathcal{L}_{cur} + \gamma\mathcal{L}_{sep}$$

### Key Designs

1. **Intrinsic Dimensionality Manifold Regularization ($\mathcal{L}_{dim}$)**: Theorem 1 establishes that GC is an intrinsic dimensionality reduction process: $\mathrm{ID}(\mathcal{G}') < \mathrm{ID}(\mathcal{G})$; Theorem 2 further proves that adversarial attacks increase the intrinsic dimensionality of the condensed graph: $\mathrm{ID}(\mathcal{G}') < \mathrm{ID}(\mathcal{G}'_*)$.

   To constrain intrinsic dimensionality, node representations $\mathbf{Z}' = (\mathbf{A}')^2\mathbf{X}'$ (embeddings after two rounds of message passing) are used, and manifold dimensionality is estimated via Laplacian approximation:
    $\mathcal{L}_{dim} = \sqrt{\det(\mathbf{\Sigma}_{Z'} + \mathbf{I})} \cdot \sum_{i=1}^{d'} S_{\mathbf{Z}'}(\alpha_i)$
   The first term estimates manifold volume via the determinant of the covariance matrix; the second term approximates the gradient integral of coordinate functions via the graph Laplacian operator. The entire process is differentiable and compatible with end-to-end training.

2. **Curvature-Aware Manifold Smoothing ($\mathcal{L}_{cur}$)**: The complexity of class boundaries is governed by the curvature of class manifolds. For each node $i$, Gaussian curvature is estimated by fitting a quadratic hypersurface $f_\Theta(\mathbf{o}) = \mathbf{o}^\top\Theta\mathbf{o}$ over its local neighborhood: $K(i) = 2\det(\mathrm{Mat}(\mathbf{Q}^{-1}\mathbf{p}))$. The procedure involves estimating the normal vector $\mathbf{u}_i$ (eigenvector corresponding to the smallest eigenvalue), projecting neighbors onto the tangent space, and fitting the hypersurface to obtain the curvature.

   To differentiate the importance of individual nodes, Ollivier Ricci curvature $\kappa(i,j) = 1 - \mathcal{W}(m_i^\alpha, m_j^\alpha)/\mathcal{D}(i,j)$ is used to assign higher weights to boundary-bridging nodes:
    $\mathcal{L}_{cur} = \sum_c \sum_{i \in \mathcal{V}_c} \mathrm{Norm}(-\kappa(i)) \cdot |K(i)|$

3. **Inter-Class Manifold Decoupling ($\mathcal{L}_{sep}$)**: Inter-class overlap is eliminated by minimizing the discrepancy between the sum of per-class manifold volumes and the total data manifold volume:
    $\mathcal{L}_{sep} = \left(\sum_c |\mathcal{M}(\mathcal{G}'_c)| - |\mathcal{M}(\mathcal{G}')|\right)^2$
   When class manifolds are completely non-overlapping, the sum of volumes equals the total volume and $\mathcal{L}_{sep}=0$. Manifold volumes are again estimated via the determinant of the covariance matrix.

### Complexity Analysis

- Intrinsic dimensionality module: $\mathcal{O}(n'd' + (d')^3)$
- Curvature computation: $\mathcal{O}(n'((d')^6 + k))$ (Gaussian curvature) + $\mathcal{O}((n')^2)$ (Ricci curvature)
- Inter-class decoupling: $\mathcal{O}(c(d')^3)$
- In practice, $n'$ is small and PCA dimensionality reduction keeps $d'$ manageable.

## Key Experimental Results

### Main Results — Robustness at Different Condensation Ratios (Attack: S.1%/F.10%/L.20%)

| Dataset | Ratio | GCond | SGDD | SFGC | GEOM | RobGC | MRGC |
|--------|------|-------|------|------|------|-------|------|
| Cora | 1.30% | 70.69 | 68.28 | 39.70 | 40.43 | 71.60 | **77.43** |
| Cora | 2.60% | 71.39 | 68.48 | 50.10 | 54.00 | 72.19 | **76.72** |
| CiteSeer | 0.90% | 60.77 | 47.13 | 36.23 | 36.70 | 59.88 | **65.12** |
| CiteSeer | 1.80% | 61.03 | 56.12 | 48.69 | 47.17 | 61.77 | **63.87** |
| PubMed | 0.08% | 70.81 | 48.75 | 47.26 | 50.17 | — | **74.85** |

### Robustness under Different Attack Budgets (Cora, Ratio 1.30%)

| Attack Type | GCond | SGDD | RobGC | MRGC | Gain |
|---------|-------|------|-------|------|------|
| Structure S.5% | 60.59 | 59.16 | 62.03 | **65.57** | +3.54 |
| Structure S.10% | 59.09 | 53.68 | 61.24 | **64.02** | +2.78 |
| Feature F.20% | 69.36 | 68.54 | 71.13 | **74.79** | +3.66 |
| Feature F.30% | 65.22 | 58.34 | 69.15 | **72.36** | +3.21 |
| Label L.30% | 64.97 | 68.02 | 65.99 | **70.47** | +4.48 |
| Label L.40% | 57.49 | 61.97 | 57.81 | **63.89** | +6.08 |

### Changes in Classification Complexity Metrics

| Metric | Change after GC | Change in Condensed Graph after Attack |
|------|---------|----------------|
| Intrinsic Dimensionality | ↓89.25% (avg.) | ↑547.54% (avg.) |
| Fisher Discriminant Ratio | Notable decrease | Substantial increase |
| Hyperspherical Coverage Ratio | Notable decrease | Substantial increase |

### Key Findings

- MRGC achieves best performance across nearly all condensation ratios and attack configurations on Cora, CiteSeer, PubMed, and DBLP.
- On Ogbn-arxiv, trajectory matching methods (SFGC/GEOM) retain an advantage due to their superior clean-graph baseline performance, a difference attributable to backbone choice rather than robustness.
- Graph denoising preprocessing (Jaccard/SVD/KNN) provides limited improvement, as these methods assume clean features and labels.
- RobGC is effective only against structural attacks and performs poorly under feature and label attacks.
- MRGC achieves its most significant gains under label attack budgets of 30%/40% (+4.48/+6.08), indicating that the inter-class manifold decoupling module is particularly effective against label attacks.

## Highlights & Insights

- **Novel theoretical perspective**: Classification complexity theory is introduced into GC robustness research, establishing a clear analytical framework through three complexity factors (intrinsic dimensionality, boundary complexity, and class ambiguity).
- **Elegant construction of two theorems**: Theorem 1 shows that GC naturally reduces complexity, while Theorem 2 shows that attacks reverse this process, together pinpointing the objective of the defense.
- **Plug-and-play design**: The regularization modules are compatible with most GC methods and are not restricted to GCond.

## Limitations & Future Work

- Gaussian curvature computation has complexity $\mathcal{O}(n'(d')^6)$; although $n'$ is small and PCA is used for dimensionality reduction, overhead may still be non-trivial for large-scale graphs.
- Experiments use only GCond as the backbone; compatibility with trajectory matching methods (SFGC/GEOM) remains unexplored.
- The three hyperparameters $\alpha, \beta, \gamma$ require grid search over the range 1e-3 to 1e2, incurring non-trivial tuning cost.
- The defense assumes the presence of known attacks; adaptability to unknown attack types warrants further investigation.

## Related Work & Insights

- Compared to RobGC, MRGC is the first GC robustification method capable of simultaneously defending against structural, feature, and label attacks.
- The application of classification complexity theory (Ho & Basu, 2002) to graph learning represents a novel cross-domain transfer.
- The manifold regularization paradigm is generalizable to other data distillation settings, such as dataset distillation and model compression.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Both the classification complexity perspective and the manifold constraint designs constitute entirely new research directions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five datasets, multiple attack types and budgets, multiple condensation ratios, and comparisons against 10 baselines.
- **Writing Quality**: ⭐⭐⭐⭐ The Q&A-driven paper structure is engaging, with tight integration between theory and experiments.
- **Value**: ⭐⭐⭐⭐ The work has significant implications for the practical deployment of GC in adversarial environments and advances the research agenda of robust graph condensation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Enhancing Graph Classification Robustness with Singular Pooling](enhancing_graph_classification_robustness_with_singular_pooling.md)
- [\[NeurIPS 2025\] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[NeurIPS 2025\] Improved Balanced Classification with Theoretically Grounded Loss Functions](improved_balanced_classification_with_theoretically_grounded_loss_functions.md)
- [\[NeurIPS 2025\] DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning](design_encrypted_gnn_inference_via_server-side_input_graph_pruning.md)

<!-- RELATED:END -->
