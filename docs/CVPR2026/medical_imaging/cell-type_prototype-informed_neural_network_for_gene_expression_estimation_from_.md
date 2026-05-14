---
title: >-
  [Paper Note] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images
description: >-
  [CVPR 2026][Medical Imaging][gene expression estimation] This paper proposes CPNN, which constructs cell-type prototypes from publicly available single-cell RNA-seq data and models slide/patch-level gene expression as a…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "gene expression estimation"
  - "pathology images"
  - "single-cell RNA sequencing"
  - "cell-type prototype"
  - "multiple instance learning"
date: 2026-05-08
content_hash: 233d274167003487
---

# Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images

**Conference**: CVPR 2026
**arXiv**: [2603.18461](https://arxiv.org/abs/2603.18461)
**Code**: [https://github.com/naivete5656/CPNN](https://github.com/naivete5656/CPNN)
**Area**: Medical Imaging
**Keywords**: gene expression estimation, pathology images, single-cell RNA sequencing, cell-type prototype, multiple instance learning

## TL;DR

This paper proposes CPNN, which constructs cell-type prototypes from publicly available single-cell RNA-seq data and models slide/patch-level gene expression as a weighted combination of these prototypes, achieving state-of-the-art performance on gene expression estimation while providing interpretability.

## Background & Motivation

Directly predicting gene expression from whole slide images (WSI) is an important low-cost alternative to RNA sequencing. Existing methods fall into two categories: slide-level (bulk transcriptomics, using MIL architectures) and patch-level (spatial transcriptomics, using Transformers/GNNs). However, both approaches learn only at the aggregated level, **without explicitly modeling the data generative process of gene expression**—that is, observed expression is actually aggregated from the expression of individual underlying cells.

**Key Challenge**: Single-cell RNA-seq data provides cell-level expression information, but it is noisy, subject to batch effects, and lacks corresponding pathology images, preventing its direct use in WSI regression.

**Core Idea**: Extract stable **cell-type prototypes** (mean expression profiles per cell type) from single-cell data, and use these prototypes as prior constraints on the prediction space. The model estimates cell-type composition weights for each patch from the image, then obtains gene expression predictions via matrix multiplication of weights and prototypes.

## Method

### Overall Architecture

The core assumption of CPNN is that slide/patch-level gene expression equals a weighted combination of cell-type prototype expressions. Training proceeds in three steps: (1) estimate cell-type prototypes $\bar{T}$ from scRNA-seq data; (2) estimate compositional weights $w$ from WSI patch images; (3) optimize the overall model via negative binomial distribution likelihood.

### Key Designs

1. **Batch-Agnostic Prototype Generation**: Single-cell data suffers from severe batch effects. A negative binomial regression model is employed: $\mu_{c,g}^{\mathrm{sc}} = (t_{c,g} + b_d)s_d$, where $s_d$ and $b_d$ are scale and shift parameters for experimental conditions, respectively. Normalized prototypes $\bar{T}$ are obtained after regression fitting. **Design Motivation**: Disentangle technical variation to retain only biologically stable gene–gene covariation patterns.

2. **Compositional Weight Estimation**: Features $\mathbf{h}_i^{(n)}$ are extracted from WSI patches using a pretrained encoder (CONCH), and cell-type composition weights $w(\mathbf{x}_i^{(n)})$ are obtained via MLP + softmax. The gene expression mean is modeled as: $\mu_g^{\mathrm{b}}(\mathcal{X}^{(n)}) = \alpha_g \sum_i \sum_c w(\mathbf{x}_i^{(n)})_c \bar{T}_{c,g} + \beta_g$, where $\alpha_g, \beta_g$ are gene-specific scale and shift parameters that bridge the modality gap between single-cell and bulk data.

3. **Modality Correction & Prototype Update**: Prototypes $\bar{T}$ are fine-tuned during training to adapt to the target distribution, while a regularization term constrains them from deviating too far from their initial values: $L_R = \|\bar{T}^0 - \bar{T}\|^2 + \mathbb{E}_n[\|\mathbf{W}^{(n)} - \bar{\mathbf{W}}^{(n)}\|^2]$. This regularization preserves the interpretability of prototypes, ensuring that weights still correspond to true cell-type compositions.

4. **Patch-Level Extension**: In the spatial transcriptomics (ST) setting, where ST data is noisier, Pearson correlation loss replaces the negative binomial likelihood. CPNN can be incorporated as a plug-and-play module into existing ST models (e.g., STNet, TRIPLEX).

### Loss & Training

Total loss: $L_{\text{total}} = L_{\text{NB}} + \lambda L_R$, where $L_{\text{NB}}$ is the negative binomial negative log-likelihood and $L_R$ regularizes prototypes and weights. AdamW optimizer, batch size 16, 500 epochs, $\lambda = 10^3$. 4-fold cross-validation.

## Key Experimental Results

### Main Results

**Slide-Level Gene Expression Estimation**

| Dataset | Metric (SCC) | CPNN | Prev. SOTA | Gain |
|---------|--------------|------|------------|------|
| BRCA | SCC | **0.338** | 0.314 (MOSBY) | +0.024 |
| KIRC | SCC | **0.318** | 0.292 (HE2RNA) | +0.026 |
| LUAD | SCC | **0.304** | 0.286 (SRMambaMIL) | +0.018 |

**Patch-Level Gene Expression Estimation (integrated into TRIPLEX)**

| Dataset | Metric (SCC) | TRIPLEX+CPNN | TRIPLEX | Gain |
|---------|--------------|--------------|---------|------|
| CSCC | SCC | **0.1821** | 0.1239 | +0.0582 |
| Her2st | SCC | **0.1194** | 0.0861 | +0.0333 |
| STNet | SCC | **0.0621** | 0.0546 | +0.0075 |

### Ablation Study

| Configuration | SCC (BRCA) | Notes |
|---------------|-----------|-------|
| w/o PI, MC, R (trained from scratch) | 0.305 | No prototype guidance, similar to MOSBY |
| w/o MC, U, R (prototype not updated or corrected) | 0.174 | Collapses due to large modality gap |
| w/o U, R (with modality correction) | 0.248 | Correction alone insufficient to fully bridge gap |
| w/o R (prototype updatable) | 0.336 | Close to full model |
| **Full CPNN** | **0.338** | Regularization maintains interpretability |

### Key Findings

- Cell-type label granularity: coarse (8 types) SCC = 0.317, medium (29 types) 0.336, fine (49 types) 0.338; overly coarse granularity causes information loss.
- Biological validation: compositional weights across BRCA subtypes are consistent with known biological characteristics—Basal-like subtype exhibits the highest Cycling prototype weight (high proliferation), and LumB > LumA.

## Highlights & Insights

- "Indirect utilization" of single-cell data: rather than direct pairing, stable prototypes are extracted as priors, circumventing noise and modality mismatch.
- Built-in interpretability: weights can be directly interpreted as "which cell type primarily drives this patch," providing practical utility for pathological analysis.
- The plug-and-play design allows seamless integration with existing ST methods, offering flexibility in practical applications.

## Limitations & Future Work

- Relies on annotated single-cell datasets; not applicable to tissue types lacking publicly available scRNA-seq data.
- Prototypes represent mean expression per cell type and cannot capture intra-type expression variability.
- Absolute SCC values remain modest (~0.3), reflecting the inherent difficulty of mapping morphology to expression.
- More advanced MIL aggregators (e.g., graph-based) have not been explored.

## Related Work & Insights

- Distinction from cell deconvolution: deconvolution estimates cell proportions to reconstruct known expression, whereas this work inversely uses proportions and prototypes to predict unknown expression from images.
- Inspired by PINN philosophy: prior knowledge (cell-type prototypes) is used to constrain the model output space.
- This approach can inspire other scenarios involving auxiliary data with modality mismatch.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The introduction of cell-type prototypes represents a structural innovation in gene expression prediction
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six datasets, both slide- and patch-level settings, complete ablation study
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and mathematical derivations are rigorous
- **Value**: ⭐⭐⭐⭐ Combines interpretability with performance gains, offering practical significance for computational pathology

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/medical_imaging/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[CVPR 2026\] Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control](beyond_pixel_simulation_pathology_image_generation_via_diagnostic_semantic_token.md)
- [\[CVPR 2026\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](prototypebased_knowledge_guidance_for_finegrained.md)

</div>

<!-- RELATED:END -->
