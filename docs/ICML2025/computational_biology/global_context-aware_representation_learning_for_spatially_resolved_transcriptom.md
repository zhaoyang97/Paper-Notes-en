---
title: >-
  [Paper Note] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics
description: >-
  [ICML2025][Computational Biology][Spatially Resolved Transcriptomics] The proposed Spotscape framework captures global similarity relations among spots via the Similarity Telescope module (rather than solely relying on spatial local neighbors). By introducing prototypical contrastive learning and a similarity scale matching strategy to handle multi-slice batch effects, it comprehensively outperforms existing methods in tasks such as spatial domain identification…
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Spatially Resolved Transcriptomics"
  - "Graph Neural Networks"
  - "Self-Supervised Learning"
  - "Global Similarity"
  - "Batch Effect Correction"
  - "Multi-slice Integration"
date: 2026-05-08
content_hash: bccaf3c063b3d0eb
---

# Global Context-aware Representation Learning for Spatially Resolved Transcriptomics

**Conference**: ICML2025  
**arXiv**: [2506.15698](https://arxiv.org/abs/2506.15698)  
**Code**: [yunhak0/Spotscape](https://github.com/yunhak0/Spotscape)  
**Area**: Spatially Resolved Transcriptomics / Graph Representation Learning  
**Keywords**: Spatially Resolved Transcriptomics, Graph Neural Networks, Self-Supervised Learning, Global Similarity, Batch Effect Correction, Multi-slice Integration

## TL;DR

The proposed Spotscape framework captures global similarity relations among spots via the Similarity Telescope module (rather than solely relying on spatial local neighbors). By introducing prototypical contrastive learning and a similarity scale matching strategy to handle multi-slice batch effects, it comprehensively outperforms existing methods in tasks such as spatial domain identification, trajectory inference, and multi-slice integration and alignment.

## Background & Motivation

Spatially Resolved Transcriptomics (SRT) enables the simultaneous acquisition of spatial coordinates and gene expression profiles of cells, serving as a frontier technology for studying tissue spatial architecture. Current graph-based representation learning methods (e.g., SEDR, STAGATE, GraphST) aggregate local information through spatial nearest neighbor (SNN) graphs, but suffer from key limitations:

**Insufficient discriminability of local similarity**: Gene expressions in biological systems change continuously along spatial coordinates, leading to minimal feature differences among local neighbors. This makes boundary spots between different spatial domains difficult to distinguish.

**Ineffectiveness of attention mechanisms for boundary spots**: Experiments demonstrate that while GAT improves the overall clustering accuracy (Total CA), it degrades the clustering accuracy of boundary spots (Boundary CA).

**Limited gains from Oracle edge weights**: Even when constructing perfect edge weights using ground-truth labels (weight of 1 for same-class, 0 for different-class), the improvement in boundary CA remains insignificant. This indicates that information acquired solely from a local perspective is fundamentally insufficient.

**Multi-slice batch effects**: During multi-slice integration, expression profiles from the same slice abnormally aggregate, masking the true biological signals.

## Method

### Overall Architecture

Spotscape adopts a Siamese network structure: two random augmentations (node feature masking + edge masking) are applied to the original SNN graph $\mathcal{G}=(X,A)$, yielding two augmented views $\tilde{\mathcal{G}}$ and $\tilde{\mathcal{G}}'$. These are processed by a shared GNN encoder $f_\theta$ to generate representations $\tilde{Z}$ and $\tilde{Z}'$, respectively.

### Similarity Telescope (Core Module)

A relational consistency loss is proposed to capture global relationships by aligning the cosine similarity matrices of all spot pairs between the two augmented views:

$$\mathcal{L}_{\text{SC}}(\tilde{Z}, \tilde{Z}') = \text{MSE}\left(\tilde{Z}_{\text{norm}}(\tilde{Z}'_{\text{norm}})^T,\ \tilde{Z}'_{\text{norm}}(\tilde{Z}_{\text{norm}})^T\right)$$

where $\tilde{Z}_{\text{norm}}$ is the L2-normalized representation. This loss forces the model to learn consistent global similarity relationships under different augmentations, directly optimizing the relative distances between spots.

### Reconstruction Loss (Preventing Degradation)

The original features are reconstructed via a shared MLP decoder $g_\theta$ to prevent representation collapse:

$$\mathcal{L}_{\text{Recon}} = \text{MSE}(X, \hat{X}) + \text{MSE}(X, \hat{X}')$$

Total single-slice loss: $\mathcal{L}_{\text{Single}} = \lambda_{\text{SC}}\mathcal{L}_{\text{SC}} + \lambda_{\text{Recon}}\mathcal{L}_{\text{Recon}}$

### Prototypical Contrastive Learning (Multi-slice)

K-means clustering is applied to representations to obtain prototypes (centroids), treating spots in the same cluster as positive pairs and those in different clusters as negative pairs. Clusterings at different granularities are repeated $T$ times to capture multi-scale semantics:

$$l_{\text{PCL}}(\tilde{Z}_i, P_{\text{set}}) = \frac{1}{T}\sum_{t=1}^{T}\log\frac{e^{\text{sim}(\tilde{Z}_i, p_{\text{map}_t(i)}^t)/\tau}}{\sum_{j=1}^{K_t}e^{\text{sim}(\tilde{Z}_i, p_j^t)/\tau}}$$

This is enabled after a warm-up period (500 epochs) to avoid interference from inaccurate early prototypes.

### Similarity Scale Matching (Eliminating Batch Effects)

Core idea: Force the mean of the top-k similarities of each spot within its own slice to match the mean of its top-k similarities across other slices:

$$l_{\text{SS}}(H_i, \mathcal{G}^{(j)}) = \left(\text{Mean}(S_{\text{top}}^{(c)}) - \text{Mean}(S_{\text{top}}^{(j)})\right)^2$$

Total multi-slice loss: $\mathcal{L}_{\text{Multi}} = \lambda_{\text{SC}}\mathcal{L}_{\text{SC}} + \lambda_{\text{Recon}}\mathcal{L}_{\text{Recon}} + \lambda_{\text{PCL}}\mathcal{L}_{\text{PCL}} + \lambda_{\text{SS}}\mathcal{L}_{\text{SS}}$

## Key Experimental Results

### Single-slice Spatial Domain Identification (SDI)

| Dataset | Method | ARI | NMI | CA |
|--------|------|-----|-----|-----|
| DLPFC (P1, Slice 151673) | SpaceFlow | 0.42 | 0.57 | 0.57 |
| | **Spotscape** | **0.48** | **0.64** | **0.61** |
| DLPFC (P2, Slice 151507) | SpaceFlow | 0.55 | 0.68 | 0.71 |
| | **Spotscape** | **0.60** | **0.72** | **0.76** |
| MTG-Control | SpaceFlow | 0.66 | 0.74 | 0.70 |
| | **Spotscape** | **0.73** | **0.78** | **0.75** |
| MTG-AD | CAST (runner-up) | 0.54 | 0.71 | 0.65 |
| | **Spotscape** | **0.68** | **0.75** | **0.77** |

Spotscape achieves the optimal ARI/NMI/CA across all 16 slices and 4 datasets.

### Multi-slice Homogeneous Integration (DLPFC)

| Patient | Method | ARI | NMI | CA |
|------|------|-----|-----|-----|
| Patient 1 | SpaceFlow | 0.48 | 0.60 | 0.60 |
| | **Spotscape** | **0.57** | **0.70** | **0.67** |
| Patient 3 | SpaceFlow | 0.51 | 0.60 | 0.69 |
| | **Spotscape** | **0.63** | **0.68** | **0.75** |

### Heterogeneous Integration (MTG, CT+AD)

| Method | ARI | NMI | CA | Silhouette |
|------|-----|-----|-----|------------|
| CAST | 0.48 | 0.52 | 0.59 | 0.45 |
| STAligner | 0.38 | 0.54 | 0.49 | 0.62 |
| **Spotscape** | **0.72** | **0.76** | **0.81** | **0.69** |

### Multi-slice Alignment (Mouse Embryo, LTARI)

| PASTE2 | CAST | STAligner | SLAT | **Spotscape** |
|--------|------|-----------|------|---------------|
| 0.21 | 0.10 | 0.46 | 0.41 | **0.51** |

## Highlights & Insights

1. **Deep problem discovery**: Through oracle experiments, the authors prove that local graph structures cannot effectively distinguish boundary spots even with perfect edge weights, fundamentally demonstrating the necessity of global similarity learning.
2. **Simple yet effective Similarity Telescope**: Aligns the global similarity matrices of two augmented views using only MSE without complex negative sampling, directly optimizing the relative distances between spots.
3. **Novel similarity scale matching strategy**: Eliminates batch effects by matching top-k similarity means. This simple concept yields remarkable results (clustering performance drops drastically upon removal).
4. **Comprehensive downstream task coverage**: Quantitative evaluations are conducted on each task, including SDI, trajectory inference, gene imputation, homogeneous/heterogeneous integration, and cross-technology alignment.
5. **High scalability**: Maintains reasonable training time on 100K spot-scale datasets.

## Limitations & Future Work

1. **$O(N_s^2)$ complexity of the global similarity matrix**: Although scalability is demonstrated, computing and storing the global similarity matrix may still become a bottleneck on ultra-large-scale datasets (millions of spots).
2. **PCL's dependency on K-means**: Prototype quality depends on K-means clustering results, which are sensitive to the number of clusters $K$ and initialization.
3. **Absence of PCL in single-slice scenarios**: The authors bypassed PCL for single-slice tasks due to a runtime-performance trade-off, leaving potential gains in single slices underexplored.
4. **Domain classification accuracy depends on downstream clustering**: Since representation learning itself does not generate domain labels, downstream clustering algorithms like K-means are still required, with clustering quality influenced by the choice of $K$.
5. **Validation restricted to specific tissues/technologies**: Evaluated primarily on brain tissues (DLPFC, MTG) and a few other tissues; generalization to more tissue types and sequencing platforms remains to be validated.

## Related Work & Insights

- **STAGATE** (Dong & Zhang, 2022): Uses GAT to learn attention weights between spots; this paper points out its poor performance on boundary spots.
- **SpaceFlow** (Ren et al., 2022): Uses DGI + spatial regularization, serving as the strongest single-method baseline.
- **GraphST** (Long et al., 2023): Uses DGI for batch correction; this paper significantly outperforms it in multi-slice tasks.
- **STAligner** (Zhou et al., 2023): Uses mutual nearest neighbors + triplet loss for slice integration.
- **CAST** (Tang et al., 2024): Uses CCA-SSG for heterogeneous slice integration and alignment.

Insight: In continuous feature scenarios (such as biological tissues, remote sensing), the information bottleneck of local graph structures is a common problem, and global similarity consistency constraints offer a promising general solution.

## Rating

- Novelty: ⭐⭐⭐⭐ — The Similarity Telescope and similarity scale matching strategy are innovative, and the problem analysis (via oracle experiments) is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 datasets, 6+ downstream tasks, comprehensive ablation/sensitivity/scalability analyses, 10 runs with statistical testing.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivational arguments and great intuitive explanation in Figure 1.
- Value: ⭐⭐⭐⭐ — Provides practical advancements to representation learning in spatially resolved transcriptomics, with highly generalizable methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](../../AAAI2026/computational_biology/dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)
- [\[ICML 2025\] ComRecGC: Global Graph Counterfactual Explainer through Common Recourse](comrecgc_global_graph_counterfactual_explainer_through_common_recourse.md)
- [\[ICLR 2026\] Fusing Pixels and Genes: Spatially-Aware Learning in Computational Pathology](../../ICLR2026/computational_biology/fusing_pixels_and_genes_spatially-aware_learning_in_computational_pathology.md)
- [\[CVPR 2025\] Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning](../../CVPR2025/computational_biology/unsupervised_foundation_model-agnostic_slide-level_representation_learning.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)

</div>

<!-- RELATED:END -->
