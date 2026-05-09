---
title: >-
  [Paper Note] Autoencoding Random Forests
description: >-
  [NeurIPS 2025][Medical Imaging][Random Forests] RFAE is the first principled encode-decode framework for random forests. It exploits the positive-definiteness and universality of the RF kernel to derive low-dimensional encodings via diffusion-map spectral decomposition, and decodes back to the original feature space through k-NN regression in leaf-node space. Across 20 tabular datasets, RFAE achieves an average reconstruction rank of 1.80, substantially outperforming TVAE (3.38) and AE (3.27), and is successfully applied to MNIST reconstruction and scRNA-seq batch-effect removal.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Random Forests
  - Autoencoder
  - Diffusion Maps
  - Spectral Decomposition
  - Tabular Data
date: 2026-05-08
content_hash: 6d3baf8807518713
---

# Autoencoding Random Forests

**Conference**: NeurIPS 2025
**arXiv**: [2505.21441](https://arxiv.org/abs/2505.21441)
**Code**: [https://github.com/bips-hb/RFAE](https://github.com/bips-hb/RFAE)
**Area**: Representation Learning / Tree Models
**Keywords**: Random Forests, Autoencoder, Diffusion Maps, Spectral Decomposition, Tabular Data

## TL;DR
RFAE is the first principled encode-decode framework for random forests. It exploits the positive-definiteness and universality of the RF kernel to derive low-dimensional encodings via diffusion-map spectral decomposition, and decodes back to the original feature space through k-NN regression in leaf-node space. Across 20 tabular datasets, RFAE achieves an average reconstruction rank of 1.80, substantially outperforming TVAE (3.38) and AE (3.27), and is successfully applied to MNIST reconstruction and scRNA-seq batch-effect removal.

## Background & Motivation

**Background**: Deep learning dominates representation learning (AE, VAE), yet random forests frequently outperform deep models on tabular data. Tabular datasets contain mixed continuous/categorical features, which tree models handle more naturally than neural networks.

**Limitations of Prior Work**: Random forests lack a principled encode-decode pipeline—there is no established way to project RF-learned representations into a low-dimensional space and decode them back. Existing approaches (e.g., applying MDS directly to the RF similarity matrix) offer no theoretical guarantees and provide no decoding capability.

**Key Challenge**: RFs implicitly learn rich data similarity structure through leaf-node co-occurrence frequencies, but this structure is locked inside the trees and cannot be exploited for compression, visualization, denoising, or other tasks that require encoding and decoding.

**Goal**: To construct a theoretically grounded encoder (low-dimensional embedding) and decoder (recovery of original features from the embedding) for random forests.

**Key Insight**: The RF kernel $k_n^{RF}(\mathbf{x}, \mathbf{x}')$ (the fraction of trees in which two samples share a leaf node) is shown to be positive-definite, asymptotically universal, and characteristic (Theorem 3.4), making it directly amenable to spectral decomposition via diffusion maps.

**Core Idea**: Positive-definiteness of the RF kernel → diffusion-map spectral decomposition = encoder; k-NN regression in leaf-node space = decoder; theoretical guarantees of universal consistency.

## Method

### Overall Architecture
Training data → Random Forest → RF similarity kernel matrix → **Encoding**: diffusion-map spectral decomposition $\mathbf{Z} = \sqrt{n}\mathbf{V}\bm{\Lambda}^t$ → **Decoding**: k-NN regression (neighbors found in low-dimensional space, reconstruction via distance-weighted average of original features) → Reconstructed data

### Key Designs

1. **Theoretical Guarantees for the RF Kernel (Theorem 3.4)**:

    - Function: Establishes that the RF kernel is suitable for diffusion maps.
    - Mechanism: Proves that $k_n^{RF}$ is positive-definite (enabling spectral decomposition), asymptotically universal (approximating any continuous kernel), and characteristic (distinct distributions yield distinct embeddings), guaranteeing the quality of the diffusion-map embedding.
    - Design Motivation: Without these guarantees, applying diffusion maps to the RF kernel could produce degenerate or inconsistent embeddings.

2. **Diffusion-Map Encoder**:

    - Function: Decomposes the RF similarity matrix into a low-dimensional embedding.
    - Mechanism: $\mathbf{Z} = \sqrt{n}\mathbf{V}\bm{\Lambda}^t$, where $\mathbf{V}$ contains the eigenvectors of the kernel matrix, $\bm{\Lambda}$ contains the eigenvalues, and $t$ controls the diffusion time scale. New samples are projected via the Nyström extension.
    - Design Motivation: Diffusion maps naturally capture nonlinear manifold structure, making them more suitable for complex data than PCA.

3. **k-NN Regression Decoder (Theorem 4.4)**:

    - Function: Recovers original features from the low-dimensional embedding.
    - Mechanism: Identifies $k$ nearest neighbors in the low-dimensional space and computes a distance-weighted average of their original features as the reconstruction. Theorem 4.4 proves universal consistency of the k-NN decoder—reconstruction error converges to zero as sample size grows.
    - Design Motivation: Achieves a better accuracy-efficiency trade-off than ILP (exact but slow) and split relabeling (fast but less accurate).

### Loss & Training
- Reconstruction evaluation: $1 - R^2$ for continuous features; classification error rate for categorical features.
- Aggregated DICE metric: $\text{DICE} = 1 - \frac{2|A \cap B|}{|A| + |B|}$

## Key Experimental Results

### Main Results (Average Rank across 20 Tabular Datasets)

| Method | Avg. Rank | # Best Datasets |
|--------|-----------|-----------------|
| **RFAE** | **1.80** | **12/20** |
| TTVAE | 2.45 | — |
| AE | 3.27 | — |
| TVAE | 3.38 | — |
| VAE | 4.10 | — |

### Ablation Study

| Decoding Method | Speed | Accuracy | Characteristics |
|-----------------|-------|----------|-----------------|
| ILP (integer linear programming) | Slow | Highest | Exact leaf-node assignment |
| Split relabeling | Fast | Moderate | CART synthetic data |
| **k-NN regression** | **Fast** | **High** | **Best trade-off** |

### Key Findings
- RFAE ranks first on 12/20 tabular datasets, substantially outperforming deep autoencoder methods.
- At $d_\mathcal{Z}=32$ on MNIST, recognizable digit reconstructions are obtained, demonstrating that the RF kernel retains sufficient visual information.
- scRNA-seq batch-effect removal succeeds, achieving batch harmonization across the Zeisel (2,874 samples) and Tasic (1,590 samples) datasets.
- The framework applies to both fully random forests and supervised RFs.

## Highlights & Insights
- **Solid theoretical foundation**: The triple guarantees of positive-definiteness, universality, and characteristicness for the RF kernel are the central theoretical contribution, legitimizing the RF kernel as a proper kernel method.
- **Another demonstration that tree models >> deep models on tabular data**: RFAE substantially outperforms AE/VAE/TVAE on tabular benchmarks.
- **Cross-domain applicability**: From tabular reconstruction to genomic denoising, the results demonstrate the generality of RFAE.

## Limitations & Future Work
- Decoding is computationally non-trivial—k-NN requires storing and querying the original RF.
- The choice of $k$ in k-NN is sensitive.
- The framework is not suited to end-to-end optimization for high-dimensional data such as images.
- The model cannot be distilled into a smaller model, as the original RF must be retained.

## Related Work & Insights
- **vs. TVAE/CTGAN**: Deep generative models generally underperform tree models on tabular data; RFAE extends this advantage to the autoencoding setting.
- **vs. Kernel PCA**: RFAE employs an RF-specific kernel with stronger theoretical guarantees and automatic handling of mixed feature types.
- **Insight**: The RF kernel can serve as a general nonparametric similarity measure, applicable not only to classification but also to representation learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First complete autoencoding framework for random forests, with substantial theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ 20 tabular datasets + MNIST + scRNA-seq across three application domains.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with clearly stated theorems.
- Value: ⭐⭐⭐⭐ Opens a new direction for combining tree models with representation learning.

### Supplementary Notes on the Method
- **Complexity of the encoding process**: Constructing the kernel matrix requires $O(n^2 d)$ time, and spectral decomposition requires $O(n^3)$; for large datasets, the Nyström approximation reduces this to $O(nm^2)$ using $m$ anchor points.
- **Consistency guarantee for k-NN decoding (Theorem 4.4)**: As $n \to \infty$ with $k/n \to 0$ and $k \to \infty$, the reconstruction error converges to the Bayes optimum—the first universal consistency result for decoding with random forests.
- **Fundamental difference from deep autoencoders**: Both the encoder (determined by the RF kernel) and the decoder (k-NN) in RFAE are nonparametric, eliminating overfitting risk at the cost of flexibility being bounded by the kernel.
- **Significance of the scRNA-seq application**: Batch effects are a central challenge in single-cell omics; RFAE's nonlinear embedding removes technical batch variation while preserving cell-type structure.
- **Multi-task potential**: The low-dimensional embedding of RFAE can simultaneously support clustering, visualization, and denoising—multiple uses from a single training run.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[NeurIPS 2025\] Learning Conformational Ensembles of Proteins Based on Backbone Geometry](learning_conformational_ensembles_of_proteins_based_on_backbone_geometry.md)
- [\[NeurIPS 2025\] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] Fractional Diffusion Bridge Models](fractional_diffusion_bridge_models.md)

</div>

<!-- RELATED:END -->
