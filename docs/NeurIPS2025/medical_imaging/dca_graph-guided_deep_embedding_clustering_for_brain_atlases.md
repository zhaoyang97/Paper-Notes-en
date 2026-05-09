---
title: >-
  [Paper Note] DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases
description: >-
  [NeurIPS 2025][Medical Imaging][Brain atlas] DCA (Deep Cluster Atlas) proposes a graph-guided deep embedding clustering framework that combines voxel-level spatiotemporal embeddings from a pretrained Swin-UNETR with KNN graph spatial regularization. By aligning soft assignments with atlas clustering auxiliary labels via KL divergence, the framework generates functionally homogeneous and spatially contiguous individualized brain atlases. On the HCP dataset, DCA achieves 98.8% improvement in homogeneity and 29% improvement in silhouette coefficient, and outperforms existing atlases on downstream tasks including autism diagnosis and cognitive decoding.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Brain atlas
  - deep clustering
  - Swin-UNETR
  - graph regularization
  - individualized parcellation
date: 2026-05-08
content_hash: 7be466f126b955d9
---

# DCA: Graph-Guided Deep Embedding Clustering for Brain Atlases

**Conference**: NeurIPS 2025
**arXiv**: [2509.01426](https://arxiv.org/abs/2509.01426)
**Code**: [https://github.com/ncclab-sustech/DCA](https://github.com/ncclab-sustech/DCA)
**Area**: Brain Imaging / Brain Atlas Construction
**Keywords**: Brain atlas, deep clustering, Swin-UNETR, graph regularization, individualized parcellation

## TL;DR
DCA (Deep Cluster Atlas) proposes a graph-guided deep embedding clustering framework that combines voxel-level spatiotemporal embeddings from a pretrained Swin-UNETR with KNN graph spatial regularization. By aligning soft assignments with atlas clustering auxiliary labels via KL divergence, the framework generates functionally homogeneous and spatially contiguous individualized brain atlases. On the HCP dataset, DCA achieves 98.8% improvement in homogeneity and 29% improvement in silhouette coefficient, and outperforms existing atlases on downstream tasks including autism diagnosis and cognitive decoding.

## Background & Motivation

**State of the Field**: Brain atlases are fundamental tools for dimensionality reduction and interpretable analysis of neuroimaging data. Hundreds of atlases (Yeo, Schaefer, AAL, MMP, etc.) have been proposed over the decades, based on anatomical, functional, or cytoarchitectural criteria, with resolutions ranging from fewer than 10 to over 1,000 regions.

**Limitations of Prior Work**: (a) Most atlases are defined on the cortical surface, ignoring subcortical and white matter structures; (b) resolutions are fixed and predefined, offering no flexibility to the user; (c) group-level templates are derived from averaged data, ignoring substantial inter-individual variability; (d) traditional clustering methods (K-Means, hierarchical clustering, spectral clustering) do not account for spatial continuity, resulting in fragmented and anatomically implausible parcellations.

**Root Cause**: fMRI data suffer from low signal-to-noise ratio and extremely high dimensionality (gray matter masks contain tens of thousands of voxels), making it difficult to simultaneously optimize functional similarity and spatial continuity. Distance penalty strategies require careful tuning; otherwise, functional homogeneity is compromised.

**Paper Goals**: To design a scalable deep clustering framework capable of generating voxel-level, individualized brain atlases while jointly ensuring functional homogeneity and spatial continuity.

**Starting Point**: A pretrained Swin-UNETR is employed to learn voxel-level spatiotemporal embeddings via masked reconstruction, after which spatial priors from a KNN graph constrain the clustering process—encouraging voxels that are functionally similar and spatially adjacent to be assigned to the same region.

**Core Idea**: Pretrained Swin-UNETR for voxel embedding extraction + 26-neighborhood graph regularization + KL divergence joint optimization = functionally homogeneous and spatially contiguous individualized brain atlases.

## Method

### Overall Architecture
**Input**: 4D fMRI data ($96\times96\times96\times300$, spatial + temporal) + binary ROI mask $M$. **Output**: $K$-class voxel-level parcellation labels. **Pipeline**: (A) Swin-UNETR masked reconstruction pretraining → (B) frozen encoder for voxel embedding extraction → learnable cluster centers + soft assignments → 26-neighborhood graph construction + spectral clustering auxiliary labels → KL divergence optimization (alternating updates of embeddings, graph weights, and cluster centers).

### Key Designs

1. **4D fMRI Masked Reconstruction Pretraining**:

    - Function: Learn voxel-level spatiotemporal embeddings capturing both local and global context.
    - Mechanism: On a Swin-UNETR encoder–decoder architecture, 80% of spatiotemporal patches are randomly masked, and the encoder is trained to reconstruct the masked regions. The encoder outputs a feature map of resolution $96\times96\times96\times256$, with each voxel corresponding to a 256-dimensional embedding.
    - Design Motivation: The hierarchical structure and window attention of the Swin Transformer preserve spatial hierarchy, making it more suitable for voxel-level modeling than standard ViT. The high masking rate of 80% compels the model to capture long-range dependencies.

2. **Learnable Cluster Centers + Soft Assignments**:

    - Function: Associate voxel embeddings with $K$ cluster centers.
    - Mechanism: A trainable matrix $\{\boldsymbol{\mu}_j\}_{j=1}^K \subset \mathbb{R}^d$ is maintained (orthogonal initialization + L2 normalization). The Euclidean distance from each voxel to all centers is computed as $\Delta_{ij} = \|\mathbf{z}_i - \boldsymbol{\mu}_j\|_2$, and soft assignments $\mathbf{q}_i \in \Delta^{K-1}$ are obtained via min-max normalization followed by softmax.
    - Design Motivation: Soft assignments permit gradient backpropagation, enabling joint optimization of cluster centers and the encoder.

3. **26-Neighborhood Graph Construction and Spatial Regularization**:

    - Function: Encourage spatially adjacent voxels to be assigned to the same region.
    - Mechanism: A 26-neighborhood graph $G=(V,E)$ is constructed over voxels within the ROI mask (all neighbors in a $3\times3\times3$ cube, excluding the center). Edge weights are defined as the cosine similarity of mean-subtracted embeddings: $a_{ij} = \cos(\mathbf{z}_i - \bar{\mathbf{z}}_i, \mathbf{z}_j - \bar{\mathbf{z}}_j)$. **Spectral clustering** is applied to the weighted graph (computing the $K$ smallest eigenvectors of the Laplacian $L=D-W$ + K-Means) to obtain hard auxiliary labels $\mathbf{p}$.
    - Design Motivation: Spectral clustering naturally accounts for graph structure (spatial neighborhoods), producing spatially contiguous labels. The Hungarian algorithm is used to align labels across iterations, avoiding the label permutation problem.

4. **KL Divergence Objective**:

    - Function: Drive soft assignments (derived from embedding distances) toward hard auxiliary labels (derived from atlas clustering).
    - Mechanism: $\mathcal{L} = \text{KL}(\mathbf{P} \| \mathbf{Q}) = \frac{1}{N} \sum_i \sum_j p_{ij} \log \frac{p_{ij}}{q_{ij}}$, where $\mathbf{P}$ is the one-hot encoding of auxiliary labels and $\mathbf{Q}$ is the soft assignment matrix. Gradients are backpropagated only to the cluster centers $\{\boldsymbol{\mu}_j\}$ and the final projection layer of Swin-UNETR; the remaining encoder weights are frozen.
    - Design Motivation: KL divergence is the standard objective in deep clustering (originating from DEC); auxiliary labels introduce spatial priors so that optimized soft assignments naturally yield spatially contiguous parcellations.

5. **Group-Level Atlas Generation**:

    - Function: Aggregate individualized atlases into a comparable group-level atlas.
    - Mechanism: A three-step procedure — (a) select $K$ template label vectors; (b) assign each gray matter voxel to the template with highest label similarity; (c) retain the largest connected component of each parcel, reassigning isolated small regions to neighboring parcels.

### Loss & Training
- Pretraining: 8 epochs, 2× A100, batch size 4, Adam lr=0.01, masking rate 0.8.
- Clustering fine-tuning: only cluster centers and the final projection layer are updated, Adam lr=0.01.
- Alternating iterations: update embeddings → recompute graph weights → re-run spectral clustering for auxiliary labels → KL optimization.
- Supports multiple resolutions: $K \in \{41, 100, 200, 360, 400, 500, 800\}$.

## Key Experimental Results

### Main Results — Homogeneity and Silhouette Coefficient (100 HCP Subjects)

| Atlas | Parcels | Homogeneity ↑ | Silhouette ↑ |
|-------|---------|---------------|--------------|
| Yeo | 7 | ~0.02 | ~0.005 |
| Brodmann | 41 | low | low |
| Schaefer200 | 200 | baseline | baseline |
| MMP | 360 | baseline | baseline |
| **DCA200** | 200 | **+77.7%** vs. Schaefer200 | **+19.5%** vs. Schaefer200 |
| **DCA (mean)** | 41–800 | **+98.8%** mean gain | **+29%** mean gain |

### Ablation Study — Component Contributions

| Method | Homogeneity ↑ | Silhouette ↑ | Connected Components/Parcel ↓ |
|--------|---------------|--------------|-------------------------------|
| fMRI + K-Means | 0.074±0.019 | 0.006±0.005 | 447.9 |
| fMRI + Graph Cut | 0.086±0.020 | 0.018±0.005 | 8.9 |
| Embedding + K-Means | 0.079±0.020 | 0.015±0.007 | 322.3 |
| Embedding + Graph Cut | 0.090±0.021 | 0.021±0.006 | 4.9 |
| **DCA (full)** | **0.100±0.022** | **0.030±0.007** | **1.005** |

### Downstream Tasks — Classification Accuracy

| Task | DCA100 | DCA200 | DCA360 | Schaefer Counterpart | Notes |
|------|--------|--------|--------|----------------------|-------|
| Sex prediction (HCP) | competitive | superior | superior | baseline | resting-state FC |
| Fluid intelligence (HCP) | competitive | superior | superior | baseline | resting-state FC |
| 7-task decoding (HCP) | superior | top | top | baseline | task FC |
| ASD diagnosis (ABIDE) | superior | **best** | superior | baseline | resting-state FC |

### Key Findings
- DCA outperforms the best baseline at all resolutions, with larger gains at lower resolutions.
- Applying K-Means directly to raw fMRI data produces an average of 447.9 connected components per parcel—severe fragmentation; DCA yields only 1.005.
- Graph regularization is critical for spatial continuity: Graph Cut reduces fragmentation from 447.9 to 8.9, and DCA further reduces it to 1.005.
- Pretrained embeddings improve homogeneity by approximately 5–10% over raw fMRI, demonstrating the importance of feature learning.
- Task-specific atlases (DCA$^{\text{gender}}_{100}$) improve sex classification by up to +12% (CNN) and +10% (k-GNN).

## Highlights & Insights
- **First application of deep clustering to brain atlas construction**: The DEC paradigm is introduced into neuroimaging, with KL divergence + graph priors enabling joint optimization of embeddings and clustering — a technically elegant transfer.
- **Near-perfect spatial continuity**: An average of 1.005 connected components per parcel is achieved without post-processing, far surpassing traditional methods.
- **High flexibility**: The framework supports arbitrary ROI masks, arbitrary resolution $K$, and switching between individualized and group-level atlases — a truly general-purpose framework.
- **Transferable methodology**: The combination of Swin-UNETR masked pretraining and graph-regularized clustering can be directly applied to region segmentation in other 3D/4D medical images (e.g., cardiac functional parcellation, pulmonary region delineation).

## Limitations & Future Work
- Voxel-level clustering incurs high memory and computational costs, making whole-brain scaling challenging.
- The fixed 26-neighborhood KNN graph may suppress long-range functional connectivity.
- Only unimodal fMRI is used; structural MRI, diffusion MRI, or electrophysiological data are not integrated.
- The optimal resolution differs across tasks (no universal optimal $K$), requiring user selection.
- The spectral clustering step requires computation of Laplacian eigenvectors, which may become a bottleneck for large-scale graphs (whole brain, ~100,000 voxels).

## Related Work & Insights
- **vs. Schaefer (gradient-driven clustering)**: Schaefer performs group-level clustering using spatially weighted functional connectivity; DCA performs individualized clustering with deep embeddings + graph priors, achieving 77.7% higher homogeneity.
- **vs. GIANT (genetics-driven atlas)**: GIANT integrates genetic information; DCA uses only fMRI yet achieves superior performance on functional metrics.
- **vs. brain segmentation methods (Swin-UNETR/DDParcel)**: Segmentation is supervised (with known labels), whereas atlas construction is unsupervised (requiring discovery of new parcellations); DCA leverages the feature learning capacity of pretrained segmentation models while performing unsupervised clustering.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of deep clustering to brain atlas construction; the graph regularization + KL optimization design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparison against 12 baseline atlases, multiple resolutions, 6 downstream tasks, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the benchmarking platform has community value, though notation is dense.
- Value: ⭐⭐⭐⭐ A significant advance in brain atlas research; configurable resolution, individualization, and open-source code offer high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)
- [\[CVPR 2026\] SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](../../CVPR2026/medical_imaging/spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)
- [\[NeurIPS 2025\] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)
- [\[NeurIPS 2025\] MedMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph](medmkg_benchmarking_medical_knowledge_exploitation_with_multimodal_knowledge_gra.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)

<!-- RELATED:END -->
