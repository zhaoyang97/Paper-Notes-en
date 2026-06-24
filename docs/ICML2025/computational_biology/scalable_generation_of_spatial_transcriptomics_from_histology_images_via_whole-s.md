---
title: >-
  [Paper Note] Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching
description: >-
  [ICML2025][Computational Biology][Spatial Transcriptomics] STFlow is proposed, a generative model based on flow matching that explicitly captures inter-cellular interactions by modeling the joint distribution of gene expressions across the entire slide. It leverages local spatial attention for efficient whole-slide encoding, achieving an 18% improvement over the best baseline on HEST-1k and STImage-1K4M.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Spatial Transcriptomics"
  - "Flow Matching"
  - "Whole-Slide Modeling"
  - "Inter-cellular Interactions"
  - "E(2)-Invariance"
  - "Generative Models"
date: 2026-05-08
content_hash: 9f364016c1cb7a9f
---

# Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching

**Conference**: ICML2025  
**arXiv**: [2506.05361](https://arxiv.org/abs/2506.05361)  
**Code**: [GitHub](https://github.com/Graph-and-Geometric-Learning/STFlow)  
**Area**: Spatial Transcriptomics / Computational Pathology  
**Keywords**: Spatial Transcriptomics, Flow Matching, Whole-Slide Modeling, Inter-cellular Interactions, E(2)-Invariance, Generative Models

## TL;DR

STFlow is proposed, a generative model based on flow matching that explicitly captures inter-cellular interactions by modeling the joint distribution of gene expressions across the entire slide. It leverages local spatial attention for efficient whole-slide encoding, achieving an 18% improvement over the best baseline on HEST-1k and STImage-1K4M.

## Background & Motivation

Spatial transcriptomics (ST) technologies enable the simultaneous acquisition of spatial location and gene expression profiles in tissue sections, providing new insights into cellular interactions and microenvironments. However, traditional ST experiments suffer from low throughput and high equipment requirements, driving the research direction of predicting gene expression from H&E stained whole-slide images (WSIs).

Existing methods face three major bottlenecks:

**Independent Prediction**: Modeling the gene expression of each spot independently as $p(\boldsymbol{Y}_i|\boldsymbol{I}_i)$, thereby ignoring the regulatory interactions between neighboring cells.

**Memory Bottleneck**: Slide-level methods apply global attention over all spots (often exceeding 10,000 spots), leading to O(N²) computational and memory overhead, resulting in out-of-memory (OOM) errors on standard hardware.

**Fragile Coordinate Encoding**: Using coordinates directly as positional encodings, which are highly sensitive to numerical noise and batch effects.

The core motivation of STFlow is to reformulate the regression task into a **generative modeling problem of the joint distribution**. It uses a flow matching iterative denoising framework to explicitly model $p(\boldsymbol{Y}_0,\cdots,\boldsymbol{Y}_N|\boldsymbol{I}_0,\cdots,\boldsymbol{I}_N)$. Each denoising step uses the current gene expression estimation as a context, thereby naturally capturing inter-cellular interactions.

## Method

### Overall Architecture

STFlow consists of three steps:

1. **Spot Encoding**: Extract visual features for each spot $\boldsymbol{Z}_i = f_{\text{PFM}}(\boldsymbol{I}_i)$ using a pretrained pathology foundation model $f_{\text{PFM}}$.
2. **Slide-level Spatial Encoding**: Aggregate k-nearest neighbor information using local spatial attention, integrating coordinate geometric relationships.
3. **Flow Matching Iterative Optimization**: Sample an initial "gene expression guess" from a prior distribution and converge to the final prediction through multi-step denoising.

### Flow Matching Learning Framework

**Training Objective**: Given WSI coordinates $\boldsymbol{C}\in\mathbb{R}^{N\times 2}$, spot images $\boldsymbol{I}$, and ground-truth gene expression $\boldsymbol{Y}\in\mathbb{R}^{N\times G}$, the denoising loss is minimized:

$$\min_{\theta} \text{MSE}\left(\boldsymbol{Y},\; f_{\theta}(\boldsymbol{Y}_t, \boldsymbol{I}, \boldsymbol{C}, t)\right)$$

where $t \sim \text{Uniform}[0,1]$, and $\boldsymbol{Y}_t = t \cdot \boldsymbol{Y} + (1-t) \cdot \boldsymbol{Y}_0$ is the linear interpolation between the ground-truth expression and the prior sample.

**Inference Process** (Euler ODE Solving): Starting from $\boldsymbol{Y}_0 \sim \mathcal{Z}(\mu,\phi,\pi)$, after $S$ iterations:

$$\boldsymbol{Y}_{t_2} = \boldsymbol{Y}_{t_1} + \frac{\hat{\boldsymbol{Y}} - \boldsymbol{Y}_{t_1}}{1 - t_1} \cdot (t_2 - t_1)$$

The last step directly outputs $\hat{\boldsymbol{Y}}$ as the prediction.

### Prior Distribution: Zero-Inflated Negative Binomial (ZINB)

The authors observe two characteristics of gene expression data: (1) a large number of genes are inactive (zeros dominate), and (2) overdispersion (variance > mean). Therefore, the ZINB distribution $\mathcal{Z}(\mu, \phi, \pi)$ is adopted as the prior, where $\pi$ models the zero-inflation ratio and the negative binomial component models the overdispersion. This highlights a key advantage of flow matching over diffusion models—the flexibility to choose non-Gaussian priors.

### E(2)-Invariant Spatial Attention

**Local Context**: For each spot $i$, only its k-nearest neighbors $\mathcal{N}(i)$ are attended to. The spatial relationship is encoded using the directional vector $\boldsymbol{C}_{i\to j} = \boldsymbol{C}_i - \boldsymbol{C}_j$.

**Frame Averaging for Invariance**: PCA is performed on the set of directional vectors to extract 4 frames (combinations of $\pm 1$ of the two principal components). The coordinates are projected onto each frame and then averaged:

$$\boldsymbol{C}'_{i\to j} = \frac{1}{|\mathcal{F}|} \sum_g \text{MLP}(\boldsymbol{C}^{(g)}_{i\to j})$$

This ensures invariance to slide rotation, translation, and reflection (E(2) transformations).

**Attention Computation**: Visual features (Q, K, V), spatial encodings, and gene expression differences $(Y_{t,i} - Y_{t,j})$ are jointly fed into an MLP attention layer:

$$\boldsymbol{A}_{ij} = \text{Softmax}_i\left(\text{MLP}(\boldsymbol{Z}_{Q,i} \| \boldsymbol{Z}_{K,j} \| \boldsymbol{C}'_{i\to j} \| (\boldsymbol{Y}_{t,i} - \boldsymbol{Y}_{t,j}))\right)$$

**Aggregation & Update**:

$$\boldsymbol{Z}'_i = \text{MLP}\left(\sum_{j\in\mathcal{N}(i)} \boldsymbol{A}_{ij}\boldsymbol{Z}_{V,j} \| \sum_{j\in\mathcal{N}(i)} \boldsymbol{A}_{ij}\boldsymbol{C}'_{i\to j}\right) + \boldsymbol{Z}_i$$

Each layer simultaneously outputs a gene expression update $\boldsymbol{Y}'_{t,i} = \text{MLP}(\boldsymbol{Z}'_i)$, and the outputs of multiple layers are averaged to obtain the final prediction.

## Key Experimental Results

### Benchmarks & Metrics

- **Benchmarks**: HEST-1k (9 cancer types) + STImage-1K4M (8 tissue types), totaling 17 datasets.
- **Metrics**: Pearson correlation coefficient (averaged across the gene dimension).
- **Baselines**: 5 spot-based models (Ciga, UNI, Gigapath, STNet, BLEEP) + 3 slide-based models (Gigapath-slide, HisToGene, TRIPLEX).

### Main Results (Pearson Corr, Higher is Better)

| Dataset | UNI | Gigapath | BLEEP | TRIPLEX | **STFlow** |
|--------|------|----------|-------|---------|-----------|
| IDC | 0.520 | 0.513 | 0.533 | 0.606 | **0.587** |
| PRAD | 0.371 | 0.384 | 0.382 | 0.402 | **0.421** |
| PAAD | 0.432 | 0.436 | 0.459 | 0.492 | **0.507** |
| SKCM | 0.629 | 0.590 | 0.566 | 0.699 | **0.704** |
| COAD | 0.285 | 0.290 | 0.303 | 0.319 | **0.326** |
| CCRCC | 0.178 | 0.187 | 0.298 | 0.289 | **0.332** |
| HCC | 0.052 | 0.051 | 0.086 | 0.062 | **0.124** |
| LUNG | 0.559 | 0.569 | 0.588 | 0.601 | **0.610** |
| **HEST Avg** | 0.344 | 0.344 | 0.368 | 0.395 | **0.415** |

- On HEST-1k, STFlow achieves an average of 0.415, a **+5.1%** improvement over the strongest baseline, TRIPLEX (0.395).
- An improvement of approximately **+20.6%** compared to pathology foundation models (UNI/Gigapath ~0.344).
- On the HCC dataset, performance increases from 0.062 to 0.124, representing an improvement of up to **100%**.
- Gigapath-slide encounters OOM errors on large-scale datasets such as IDC/COAD/Stomach, whereas STFlow avoids this issue.

### Efficiency Analysis

- Compared to global attention methods like HisToGene/TRIPLEX, STFlow utilizes local k-nearest neighbor attention, reducing the memory complexity from O(N²) to O(Nk).
- It operates effectively on slides with over 10,000 spots, where other slide-based methods suffer from OOM errors.

## Highlights & Insights

1. **Regression-to-Generative Paradigm Shift**: Reformulating gene expression prediction from single-step regression into a flow matching generative process, where iterative denoising naturally encodes inter-cellular gene expression dependencies. This approach is highly novel and theoretically grounded.
2. **ZINB Prior**: Capitalizing on flow matching's flexibility to utilize arbitrary priors, a prior matching the characteristics of gene expression distribution (zero-inflation + overdispersion) is introduced, which is more reasonable than a standard Gaussian prior.
3. **E(2)-Invariant Design**: The combination of Frame Averaging and local spatial attention both encodes spatial dependencies and ensures invariance to coordinate transformations, possessing clear physical significance.
4. **Scalability**: Local k-NN attention addresses the memory bottleneck for whole slides with tens of thousands of spots, while long-range dependencies can still be captured via multi-layer stacking.
5. **Strong Biological Interpretability**: Injecting gene expression differences $(Y_{t,i} - Y_{t,j})$ into attention weights directly models inter-cellular gene regulatory signals.

## Limitations & Future Work

1. **Inference Speed**: Flow matching requires multi-step iterative sampling (solving an S-step ODE), making its inference slower than single-step regression methods. Future directions could explore acceleration using distillation or consistency models.
2. **Prior Parameter Estimation**: The parameters $\mu, \phi, \pi$ of the ZINB prior are estimated from the training set, and their generalization capability across datasets remains to be verified.
3. **Limitations of Local Attention**: The k-NN mechanism may overlook long-range intercellular interactions across different regions (e.g., immune cell infiltration). Although multi-layer stacking partially alleviates this, information decay remains an issue.
4. **Dataset Diversity**: The framework is primarily validated on Visium 10x platform data, and its applicability to higher-resolution ST technologies (e.g., MERFISH, Slide-seq) has not been explored.
5. **Lack of Downstream Evaluation**: Although biomarker gene prediction is mentioned, systematic evaluation of downstream tasks such as cell-type annotation and spatial domain identification is lacking.

## Related Work & Insights

- **TRIPLEX** (Chung et al., 2024): Currently the strongest slide-based baseline, but global attention causes OOM errors.
- **BLEEP** (Xie et al., 2023): A contrastive-learning-based spot-level method that performs well on certain datasets.
- **Flow Matching** (Lipman et al., 2022): The foundational generative framework; STFlow demonstrates its successful application in non-image domains.
- **Frame Averaging** (Puny et al., 2021): A general technique for achieving E(n) invariance in geometric deep learning.
- **Insights**: The paradigm of using generative models as "structured regression" is worth expanding to other multi-output prediction tasks (e.g., joint multi-omics prediction).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Regression-to-generative paradigm shift + ZINB prior + E(2)-invariant spatial attention, a triple innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison across 17 datasets and 8 baselines, though lacking downstream task validation.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, rigorous technical descriptions, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ — Establishes a new SOTA for spatial transcriptomics prediction with high methodological generality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Predicting Spatial Transcriptomics from Histology Images via High-Order Multi-Cell Interaction Modeling](../../CVPR2026/computational_biology/predicting_spatial_transcriptomics_from_histology_images_via_high-order_multi-ce.md)
- [\[ICML 2025\] Efficient Molecular Conformer Generation with SO(3)-Averaged Flow Matching and Reflow](efficient_molecular_conformer_generation_with_so3-averaged_flow_matching_and_ref.md)
- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)
- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/computational_biology/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[CVPR 2026\] From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images](../../CVPR2026/computational_biology/from_spots_to_pixels_dense_spatial_gene_expression_prediction_from_histology_ima.md)

</div>

<!-- RELATED:END -->
