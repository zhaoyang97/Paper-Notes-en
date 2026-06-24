---
title: >-
  [Paper Note] A Flag Decomposition for Hierarchical Datasets
description: >-
  [CVPR 2025][Image Restoration][Flag Manifold] This paper proposes Flag Decomposition (FD), an algorithm that decomposes hierarchically structured data into flag manifold representations (Stiefel coordinates) while preserving hierarchical relationships. It demonstrates advantages over standard methods like SVD in denoising, clustering, and few-shot learning tasks.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Flag Manifold"
  - "Hierarchical Data"
  - "Matrix Decomposition"
  - "Denoising"
  - "Few-Shot Learning"
date: 2026-05-08
content_hash: 2012a1afc7f2e88c
---

# A Flag Decomposition for Hierarchical Datasets

**Conference**: CVPR 2025  
**arXiv**: [2502.07782](https://arxiv.org/abs/2502.07782)  
**Code**: [https://github.com/nmank/FD](https://github.com/nmank/FD)  
**Area**: Image Restoration / Subspace Learning  
**Keywords**: Flag Manifold, Hierarchical Data, Matrix Decomposition, Denoising, Few-Shot Learning

## TL;DR
This paper proposes Flag Decomposition (FD), an algorithm that decomposes hierarchically structured data into flag manifold representations (Stiefel coordinates) while preserving hierarchical relationships. It demonstrates advantages over standard methods like SVD in denoising, clustering, and few-shot learning tasks.

## Background & Motivation

**Background**: Hierarchical structures widely exist in data—taxonomic hierarchies, spectral hierarchies, feature hierarchies, etc. Flag manifolds encode sequences of nested subspaces and have shown potential in tasks such as motion averaging, subspace clustering, and dimensionality reduction.

**Limitations of Prior Work**: Most existing flag manifold applications are limited to using standard matrix decomposition methods like SVD to extract flags, lacking general algorithms specifically designed for decomposing hierarchical data while preserving its hierarchical structure. Standard dimensionality reduction methods (such as PCA) flatten the hierarchical structure, losing intrinsic part-whole relationship information.

**Key Challenge**: Hierarchical relationships in data encode important structural information (e.g., coarse-to-fine neighborhoods, organization of spectral bands), but standard linear algebra tools treat all dimensions equally, failing to preserve these relationships.

**Goal**: To design a general algorithm that decomposes data matrices with hierarchical index annotations into flag representations that preserve the hierarchy, and to demonstrate its advantages across various applications.

**Key Insight**: The mathematical structure of the flag manifold—nested subspace sequences $\mathcal{V}_1 \subset \mathcal{V}_2 \subset \cdots \subset \mathcal{V}_k$—naturally corresponds to the column index hierarchy of hierarchically structured data.

**Core Idea**: Given a data matrix $\mathbf{D}$ and the hierarchical grouping of its columns $\mathcal{A}_1 \subset \mathcal{A}_2 \subset \cdots \subset \mathcal{A}_k$, it is decomposed as $\mathbf{D}\mathbf{P} = \mathbf{Q}\mathbf{R}$, where $\mathbf{Q}$ is a flag represented in Stiefel coordinates, $\mathbf{R}$ is a block upper triangular matrix, and $\mathbf{P}$ is a permutation matrix. This decomposition ensures that the nested subspace hierarchy $[[\mathbf{Q}]]$ precisely corresponds to the column hierarchy of the data.

## Method

### Overall Architecture
Input: Data matrix $\mathbf{D} \in \mathbb{R}^{n \times p}$ + column hierarchy $\mathcal{A}_1 \subset \cdots \subset \mathcal{A}_k$. The algorithm rearranges the columns according to the hierarchical groups, step-by-step performs orthogonalization on each hierarchy layer (QR decomposition after projecting out existing subspaces) to obtain the hierarchy-preserving flag coordinates $[\![\mathbf{Q}]\!]$. Applications: denoising (projection onto flags), clustering (using flag distance), and few-shot learning (flags as prototypes).

### Key Designs

1. **Hierarchy-Preserving Layer-by-Layer Orthogonalization**:

    - **Function**: Decomposes the data matrix into flag coordinates that preserve nested subspaces.
    - **Mechanism**: The new columns $\mathbf{B}_i$ of the $i$-th layer are first projected onto the orthogonal complement of the subspace of the previous $i-1$ layers, and then QR decomposition is performed on the projected result to obtain $\mathbf{Q}_i$. Finally, $\mathbf{Q} = [\mathbf{Q}_1 | \mathbf{Q}_2 | \cdots | \mathbf{Q}_k]$ ensures $[\mathbf{Q}_1] \subset [\mathbf{Q}_1, \mathbf{Q}_2] \subset \cdots$.
    - **Design Motivation**: Standard QR decomposition does not consider column hierarchical grouping. FD ensures the hierarchical structure is preserved through the ordered projection of grouped hierarchies.

2. **Flag Distance for Robust Clustering**:

    - **Function**: Leverages geometric distances on flag manifolds to improve subspace clustering.
    - **Mechanism**: Flag chordal distance considers the alignment of each layer in the nested subspaces simultaneously, providing a finer-grained similarity measure than the Grassmannian distance (which only considers the overall subspace). For noisy data or data with outliers, flag representations are naturally more robust.
    - **Design Motivation**: Hierarchical information acts as an additional constraint; using it allows better distinction between noise and signal.

3. **Flag Prototypes for Few-Shot Learning**:

    - **Function**: Replaces single prototype vectors with flags as class representations.
    - **Mechanism**: Constructs flag prototypes (utilizing feature hierarchies) for few-shot sample features of each class, and calculates the distance from test samples to each flag prototype during classification. Flag prototypes contain richer class subspace structures than single vectors.
    - **Design Motivation**: In few-shot learning where sample sizes are small, keeping the hierarchical structure of features provides a more informative class representation.

### Loss & Training
Not applicable for training—FD is a deterministic matrix decomposition algorithm. In few-shot learning, flag chordal distance is used as the metric for nearest neighbor classification.

## Key Experimental Results

### Main Results

| Application | Method | Performance |
|---|---|---|
| Hyperspectral Denoising | SVD Reconstruction | Baseline |
| Hyperspectral Denoising | **FD Reconstruction** | **Better (preserves spectral hierarchy)** |
| Subspace Clustering (Noisy) | Grassmannian Distance | Sensitive to noise |
| Subspace Clustering (Noisy) | **Flag Distance** | **More robust** |
| Few-Shot Classification | Single Vector Prototype | Baseline |
| Few-Shot Classification | **Flag Prototype** | **Higher accuracy** |

### Ablation Study

| Configuration | Performance | Description |
|---|---|---|
| SVD (No hierarchy awareness) | Baseline | Loses hierarchical structure |
| FD (Hierarchy-preserving) | Superior | Hierarchical information is valuable |
| Flag Distance vs Grassmannian Distance | Flag is superior | Multi-layer alignment is more discriminative |

### Key Findings
- On noisy data, the clustering performance of FD is significantly better than SVD—hierarchical constraints provide natural robustness to noise.
- In hyperspectral denoising, preserving the spectral hierarchy is more reasonable than directly truncating SVD singular values.
- Flag prototypes consistently outperform standard prototype networks in few-shot learning.
- The computational overhead of the algorithm is comparable to QR decomposition, making it applicable to large-scale data.

## Highlights & Insights
- **Generalization of flag manifold applications**: Transforms the abstract differential geometry concept into a practical tool for data analysis. The FD algorithm is simple and practical.
- **Mathematical formalization of hierarchical structures**: Strictly expresses the "part-to-whole" relationship using nested subspaces, providing stronger theoretical guarantees than ad-hoc hierarchical processing.
- **Broad applicability**: The same decomposition framework is applicable to three different tasks: denoising, clustering, and few-shot learning.

## Limitations & Future Work
- Requires prior knowledge of the data's hierarchical structure (the grouping of columns)—cannot automatically discover hierarchies.
- Assumes the hierarchy corresponds to the nesting of linear subspaces; non-linear hierarchical relationships cannot be handled directly.
- Currently verified only on small-to-medium-scale data; effectiveness in large-scale deep learning scenarios remains to be tested.
- Future work can explore combination with deep learning feature extractors to automatically construct feature hierarchies.

## Related Work & Insights
- **vs SVD/PCA**: Do not preserve hierarchical structures; FD is strictly superior on hierarchical data.
- **vs Grassmannian methods**: Consider only a single subspace; flags consider nested subspace sequences, providing richer information.
- The flag manifold perspective is transferable to any data analysis task with a natural hierarchical structure.

## Rating
- Novelty: ⭐⭐⭐⭐ First general hierarchy-preserving flag decomposition algorithm
- Experimental Thoroughness: ⭐⭐⭐ Covers three applications but the depth of each is limited
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear intuitive examples
- Value: ⭐⭐⭐⭐ Provides a new mathematical tool for handling hierarchical data

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](../../NeurIPS2025/image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](../../ICLR2026/image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)
- [\[ICLR 2026\] Taming Hierarchical Image Coding Optimization: A Spectral Regularization Perspective](../../ICLR2026/image_restoration/taming_hierarchical_image_coding_optimization_a_spectral_regularization_perspect.md)
- [\[NeurIPS 2025\] scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy](../../NeurIPS2025/image_restoration/scsplit_bringing_severity_cognizance_to_image_decomposition_in_fluorescence_micr.md)
- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](../../NeurIPS2025/image_restoration/learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)

</div>

<!-- RELATED:END -->
