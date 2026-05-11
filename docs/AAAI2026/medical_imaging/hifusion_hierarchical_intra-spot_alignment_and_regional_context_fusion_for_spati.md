---
title: >-
  [Paper Note] HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology
description: >-
  [AAAI 2026][Medical Imaging][Spatial Transcriptomics] This paper proposes HiFusion, a framework comprising two complementary modules — Hierarchical Intra-Spot Modeling (HISM) and Context-Aware Cross-Scale Fusion (CCF) —…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Spatial Transcriptomics"
  - "Gene Expression Prediction"
  - "Multi-Scale Feature Fusion"
  - "Histopathology"
  - "Cross-Attention"
date: 2026-05-08
content_hash: 59f8d24b160bbaab
---

# HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology

**Conference**: AAAI 2026
**arXiv**: [2511.12969](https://arxiv.org/abs/2511.12969)
**Code**: [GitHub](https://github.com/Advanced-AI-in-Medicine-and-Physics-Lab/HiFusion)
**Area**: Medical Imaging / Spatial Transcriptomics
**Keywords**: Spatial Transcriptomics, Gene Expression Prediction, Multi-Scale Feature Fusion, Histopathology, Cross-Attention

## TL;DR

This paper proposes HiFusion, a framework comprising two complementary modules — Hierarchical Intra-Spot Modeling (HISM) and Context-Aware Cross-Scale Fusion (CCF) — to accurately predict spatial gene expression from H&E-stained whole-slide images, achieving state-of-the-art performance on two benchmark datasets under both 2D cross-validation and 3D sample-specific evaluation settings.

## Background & Motivation

- **Spatial transcriptomics (ST)** enables genome-wide expression profiling with spatial localization preserved, but its clinical adoption is hindered by high cost, specialized equipment, and limited scalability.
- **H&E-stained WSIs** are routinely acquired in clinical pathology at low cost and encode rich morphological features closely associated with gene expression (e.g., the correlation between ERBB2 overexpression and specific morphological phenotypes in HER2-positive breast cancer).
- **Limitations of existing methods**:
  - Most methods treat each spot as a homogeneous region, neglecting intra-spot hierarchical structure (a 55–100 μm spot contains diverse cell types, nuclear textures, and subcellular patterns).
  - Regional context information is used only as auxiliary input without explicitly modeling the semantic relationship between a spot and its surrounding tissue.
  - Methods such as TRIPLEX and ASIGN employ large regional patches (exceeding 1000×1000 pixels), where large receptive fields may introduce morphological noise.
- **Core Problem**: How to simultaneously capture fine-grained morphological heterogeneity within spots and biologically relevant contextual information from surrounding tissue.

## Method

### Overall Architecture

HiFusion is a dual-branch framework consisting of two key components:

1. **HISM (Hierarchical Intra-Spot Modeling)**: hierarchical intra-spot modeling and alignment.
2. **CCF (Context-Aware Cross-Scale Fusion)**: context-aware cross-scale fusion.

The inputs are a spot image $X^S \in \mathbb{R}^{n \times H_S \times W_S \times 3}$ and a regional neighborhood patch $X^N \in \mathbb{R}^{n \times H_N \times W_N \times 3}$. The objective is to learn the mapping $\phi: \{X^S, X^N\} \rightarrow Y$ to predict the gene expression vector $Y \in \mathbb{R}^{n \times m}$.

### HISM Module: Hierarchical Intra-Spot Modeling

The core idea is to capture intra-spot morphological heterogeneity through multi-resolution sub-patch decomposition:

- **Level-0**: The original spot image (224×224) is encoded by a shared encoder $f_\theta$ to produce a global feature map $\mathbf{F}_0^S \in \mathbb{R}^{d \times h \times w}$.
- **Level-1**: The spot is decomposed into $p \times p$ (2×2) non-overlapping sub-patches to capture sub-tissue/regional structures.
- **Level-2**: The spot is decomposed into $q \times q$ (7×7) non-overlapping sub-patches to capture cellular/subcellular information.
- All sub-patches are encoded by a **shared encoder** (ResNet-18), and the resulting features are reassembled according to their original spatial positions; bilinear interpolation is applied when the reconstructed resolution does not match Level-0.

**Feature alignment loss**: enforces cross-scale semantic consistency:

$$\mathcal{L}_{\text{align}} = \sum_{s=1}^{2} \|\tilde{\mathbf{F}}_s^S - \mathbf{F}_0^S\|_1$$

Exploiting the translation invariance of CNNs, this encourages fine-grained features to maintain global semantic consistency.

### CCF Module: Context-Aware Cross-Scale Fusion

- **Regional encoding**: The neighborhood patch is processed by a lightweight encoder (ResNet-10) followed by global average pooling to obtain a regional representation $\mathbf{Q}_i^N \in \mathbb{R}^{1 \times d}$.
- **Learnable weighted fusion**: Spot features from three scales are adaptively fused via softmax-normalized learnable weights:

$$\mathbf{F}_{\text{fused}}^S = \sum_{s=0}^{2} \omega_s \cdot \mathbf{F}_s^S, \quad \omega_s = \frac{\exp(\alpha_s)}{\sum_{j=0}^{2} \exp(\alpha_j)}$$

- **Cross-attention fusion**: The regional representation serves as Query, while the fused multi-scale spot features (pooled into $k^2$ tokens via adaptive average pooling) serve as Key and Value.
- **Residual connection + prediction head**:

$$\hat{\mathbf{y}}_i = \text{FC}(\text{LayerNorm}(\mathbf{Q}_i^N + \phi_{ca}(\mathbf{Q}_i^N, \mathbf{K}_i^S, \mathbf{V}_i^S)))$$

This design enables the model to selectively attend to biologically relevant contextual information while suppressing spatial noise.

### Loss & Training

The total loss consists of three components:

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{main}} + \mathcal{L}_{\text{aux}}}_{\mathcal{L}_{\text{reg}}} + \lambda \mathcal{L}_{\text{align}}$$

- $\mathcal{L}_{\text{main}}$: MSE regression loss.
- $\mathcal{L}_{\text{aux}}$: Multi-scale auxiliary supervision, where each scale independently predicts gene expression.
- $\mathcal{L}_{\text{align}}$: Cross-scale feature alignment regularization ($\lambda=1$).

## Key Experimental Results

### Datasets

| Dataset | Description | # Samples | # Spots |
|---------|-------------|-----------|---------|
| HER2 | HER2-positive breast tumors | 36 WSIs (8 patients) | 13,620 |
| ST-Data | Breast cancer (ST-Net) | 16 samples | 41,544 |

The top-250 highly expressed genes are predicted; gene expression values are processed with spot-wise normalization followed by log transformation.

### Main Results

| Method | HER2-2D MSE↓ | HER2-2D PCC↑ | HER2-3D MSE↓ | HER2-3D PCC↑ | ST-2D MSE↓ | ST-2D PCC↑ | ST-3D MSE↓ | ST-3D PCC↑ |
|--------|-------------|-------------|-------------|-------------|-----------|-----------|-----------|-----------|
| ST-Net | 0.6523 | 0.4621 | 0.5323 | 0.7042 | 0.5798 | 0.5304 | 0.4939 | 0.7443 |
| TRIPLEX | 0.5715 | 0.4750 | 0.2899 | 0.7471 | 0.5389 | 0.5387 | 0.2857 | 0.7780 |
| ASIGN-2D | 0.5830 | 0.4601 | 0.3116 | 0.7316 | 0.5449 | 0.5373 | 0.2822 | 0.7741 |
| **HiFusion** | **0.5459** | **0.4961** | **0.2846** | **0.7492** | **0.5095** | **0.5613** | **0.2711** | **0.7838** |

- Under 2D evaluation, HiFusion achieves 2.1–2.6% lower MSE and over 2% higher PCC than TRIPLEX on HER2.
- Under 3D evaluation, improvements of 22–25% over ST-Net are observed; the 3D intra-sample learning strategy outperforms the complex 3D alignment scheme of ASIGN-3D.

### Ablation Study

**HISM decomposition levels**:
- 1×1 (no decomposition) already approaches the second-best baseline (TRIPLEX).
- The combination of 1×1 + 2×2 + 7×7 is optimal, with complementary spatial granularities spanning global tissue → sub-regional structure → cellular level.

**Feature alignment loss**:
- Incorporating the alignment loss reduces MSE by approximately 2% and improves PCC by more than 2% on HER2.

**CCF module**:
- Number of spot tokens: 2×2 (4 tokens) is optimal; more tokens introduce noise.
- Neighborhood patch size: 2× the spot size (448×448) is optimal; larger regions introduce irrelevant tissue signals.

### Cancer Marker Gene Visualization

- ERBB2: HiFusion MAE=0.711, PCC=0.518 vs. ASIGN MAE=1.074, PCC=−0.035.
- KRT19: HiFusion MAE=0.446, PCC=0.230 (best).
- CD74: HiFusion MAE=0.584, PCC=0.357 (best).
- Visually, HiFusion most accurately localizes high-expression regions.

## Highlights & Insights

1. The **hierarchical intra-spot modeling** approach is novel, leveraging multi-resolution decomposition with a shared encoder for computational efficiency and semantic consistency.
2. **Moderate context outperforms large-scale context**: a neighborhood size of 448 (2× spot) outperforms larger receptive fields such as 1120, demonstrating that overly large receptive fields introduce noise.
3. The **3D intra-sample learning** strategy is counterintuitive yet effective: training on a single section and predicting adjacent sections outperforms more complex cross-sample 3D strategies.
4. **Simplicity and practicality**: ResNet-18/10 backbones enable training on a single RTX 4090, offering strong scalability.
5. The cross-scale feature alignment loss serves as an important regularizer, ensuring semantic consistency across multi-resolution features.

## Limitations & Future Work

- Validation is limited to breast cancer datasets (HER2 and ST-Data); other tissue types and diseases remain unexplored.
- The encoder adopts a fixed ResNet architecture; pretrained foundation model encoders (e.g., UNI, CONCH) have not been explored.
- Regional context is captured via a single neighborhood patch, without considering more flexible spatial graph modeling (e.g., GNN).
- Only the top-250 genes are predicted; genome-wide prediction remains an open challenge.

## Related Work & Insights

- **ST-Net** (He et al. 2020): DenseNet-based independent spot prediction; a pioneering work.
- **HisToGene** (Pang et al. 2021): ViT for modeling long-range dependencies.
- **Hist2ST** (Zeng et al. 2022): ConvMixer + GNN for neighborhood modeling.
- **EGN/BLEEP**: Image similarity retrieval / contrastive learning approaches, sensitive to staining variation.
- **TRIPLEX** (Chung et al. 2024): Three-branch architecture (spot / neighborhood / global); CVPR 2024.
- **ASIGN** (Zhu et al. 2025): 3D tissue section alignment + graph model; current state of the art.

## Rating ⭐⭐⭐⭐

The method is clearly designed and well-motivated; the combination of hierarchical decomposition and cross-attention fusion effectively addresses the multi-scale modeling problem. Experiments are comprehensive with complete ablations, and cancer marker gene visualization enhances clinical interpretability. However, dataset diversity is limited, stronger pretrained encoders are not explored, and generalizability requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](../../ICLR2026/medical_imaging/histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)
- [\[AAAI 2026\] SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection](spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/medical_imaging/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/medical_imaging/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)

</div>

<!-- RELATED:END -->
