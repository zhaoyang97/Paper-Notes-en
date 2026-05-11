---
title: >-
  [Paper Note] SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection
description: >-
  [AAAI 2026][Medical Imaging][Cancer region detection] SpaCRD is proposed as a transfer learning-based multimodal deep fusion framework that integrates histology images and spatial transcriptomics (ST) data through a Vari…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Cancer region detection"
  - "spatial transcriptomics"
  - "histology images"
  - "multimodal fusion"
  - "transfer learning"
  - "variational autoencoder"
  - "cross-attention"
date: 2026-05-08
content_hash: 86821a7ef1fa72da
---

# SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection

**Conference**: AAAI 2026  
**arXiv**: [2603.06186](https://arxiv.org/abs/2603.06186)  
**Code**: [github.com/wenwenmin/SpaCRD](https://github.com/wenwenmin/SpaCRD)  
**Area**: Medical Imaging / Computational Pathology  
**Keywords**: Cancer region detection, spatial transcriptomics, histology images, multimodal fusion, transfer learning, variational autoencoder, cross-attention

## TL;DR
SpaCRD is proposed as a transfer learning-based multimodal deep fusion framework that integrates histology images and spatial transcriptomics (ST) data through a Variational Reconstruction-guided Bidirectional Cross-Attention (VRBCA) fusion network. It achieves state-of-the-art performance in cancer tissue region (CTR) detection across samples, platforms, and batches on 23 paired datasets.

## Background & Motivation

**Background**: CTR detection is a critical step in tumor diagnosis, directly relevant to surgical margin delineation, radiotherapy dose delivery, and tumor microenvironment analysis. Traditional approaches either rely on manual pathologist annotation—which is costly and time-consuming—or on histology-based anomaly detection algorithms that suffer from high false-positive rates due to morphological similarity between tissue types.

**Opportunities and Challenges with Spatial Transcriptomics (ST)**: ST technology enables comprehensive transcriptomic profiling of tissue sections while preserving spatial context, providing rich cell phenotype and spatial localization information. However, background noise introduced during ST sequencing severely affects downstream algorithm performance, and marker-gene-based methods dependent on expert prior knowledge lack generalizability.

**Limitations of Prior Work**:
   - SpaCell relies on simple feature concatenation, ignoring cross-modal interaction and global spatial context
   - STANDS/MEATRD adopt computer vision anomaly detection paradigms using reconstruction error for detection, but perform poorly on structured, contiguous cancer regions
   - Batch heterogeneity limits cross-dataset generalization

**Key Insight**: Transfer learning is used to align heterogeneous ST datasets, while multimodal deep fusion compensates for morphological ambiguity and ST noise. This is the first work to combine multimodal deep fusion with transfer learning for CTR detection.

## Method

### Overall Architecture

SpaCRD consists of three training stages:
- **Stage I: Modality-Aligned Representation Learning** — Histology image features are extracted using the pretrained pathology foundation model UNI, and the image and ST modalities are aligned via CLIP-style contrastive learning
- **Stage II: VRBCA Fusion Network** — Bidirectional cross-attention combined with a category-regularized VAE learns compact, class-consistent multimodal embeddings
- **Stage III: Cancer Likelihood Estimation** — Cancer probability is predicted for each spot based on the fused representation

### Stage I: Modality-Aligned Representation Learning

1. **Image Feature Extraction**: Patches are cropped from histology images according to the spatial coordinates of each spot in the ST data. H&E embeddings are extracted using UNI (a pathology foundation model) as $\mathbf{x}_i^{\text{img}} = f_{\text{UNI}}(I_i)$, without fine-tuning to reduce computational cost.

2. **Contrastive Alignment**: Two lightweight three-layer MLP encoders (image encoder $f_{c1}$ and gene encoder $f_{c2}$) are designed and trained with an InfoNCE contrastive loss to pull together co-located paired samples and push apart samples from different locations:
$$\mathcal{L}_{\text{contrast}} = \alpha \cdot \mathcal{L}_{\text{img→gene}} + (1-\alpha) \cdot \mathcal{L}_{\text{gene→img}}, \quad \alpha=0.5$$

### Stage II: VRBCA Fusion Network

The core module comprises two components: **Bidirectional Cross-Attention (BCA)** and a **Regularized Variational Autoencoder (RVAE)**:

1. **Bidirectional Cross-Attention (BCA)**:
    - Two independent multi-head cross-attention modules are defined: gene-guided CA and H&E-guided CA
    - For each spot $i$ and its neighborhood spots, cross-modal interaction representations are obtained from two perspectives—using image features as query with gene features as key/value, and vice versa
    - The outputs from both directions are concatenated and passed through an MLP to obtain the fused representation $\mathbf{h}_i^*$

2. **Category-Regularized VAE (RVAE)**:
    - The fused representation $\mathbf{h}_i^*$ is mapped through an encoder to a latent variable $\mathbf{z}_i \sim \mathcal{N}(\boldsymbol{\mu}_i, \boldsymbol{\sigma}_i^2)$
    - Learnable class-specific latent centers $\boldsymbol{\mu}_{y_i}$ are introduced, replacing the standard KL divergence with a category-regularized KL divergence:
   $$\mathcal{D}_{\text{KL}}^{\text{cls}}(q_i \| p_{y_i}) = \frac{1}{2}\sum_j [\sigma_{i,j}^2 + (\mu_{i,j} - \mu_{y_i,j})^2 - \log\sigma_{i,j}^2 - 1]$$
   - This encourages intra-class compactness and inter-class separation, while denoising through reconstruction

### Stage III: Cancer Likelihood Discriminator

The concatenated $\boldsymbol{\mu}_i$ and $\log\boldsymbol{\sigma}^2$ from the RVAE encoder are fed into a two-layer MLP classifier to predict cancer probability.

### Loss & Training

$$\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{BCE}} + \gamma \cdot \mathcal{L}_{\text{fused}}, \quad \gamma=0.1$$

where $\mathcal{L}_{\text{fused}}$ includes the reconstruction loss and the category-regularized KL loss ($\beta=0.5$).

During inference, a Gaussian Mixture Model (GMM) is used to automatically determine the classification threshold.

## Experiments

### Datasets
- **23 paired histology–ST datasets** spanning multiple platforms and batches
- Breast cancer datasets: STHBC (8 sections), 10XHBC, XeHBC, IDC
- Colorectal cancer datasets: CRC (12 sections, 6 pairs from different platforms)

### Baselines
8 state-of-the-art methods: SimpleNet (image-based), Spatial-ID/STAGE (ST-based), SpaCell-Plus/iStar/TESLA/MEATRD/STANDS (multimodal)

### Main Results

| Scenario | SpaCRD vs. 2nd (AUC↑) | SpaCRD vs. 2nd (AP↑) | SpaCRD vs. 2nd (F1↑) |
|------|------------------------|----------------------|----------------------|
| Cross-sample (20 datasets) | avg. +13.5% | avg. +14.1% | avg. +14.0% |
| Cross-platform & batch | avg. +12.1% | avg. +11.8% | avg. +13.8% |

Representative results:
- CRC_A1: AUC 0.953 vs. SpaCell-Plus 0.821
- STHBC_A: AUC 0.979 vs. SpaCell-Plus 0.929
- ViHBC (cross-platform): AUC 0.900 vs. SpaCell-Plus 0.784
- IDC (cross-platform): AUC 0.891 vs. SpaCell-Plus 0.803

### Ablation Study

| Variant | HBC AUC | CRC AUC |
|------|---------|---------|
| Image-based only | 0.789 | 0.606 |
| ST-based only | 0.832 | 0.782 |
| w/o BCA | 0.849 | 0.797 |
| w/o RVAE | 0.887 | 0.831 |
| w/o VRBCA | 0.815 | 0.771 |
| w/o CL (contrastive learning) | 0.892 | 0.824 |
| **Full model** | **0.923** | **0.869** |

- Removing VRBCA leads to the largest performance drop, confirming the central role of the fusion module
- Contrastive pre-alignment also contributes significantly

**Feature extractor ablation**: UNI >> HIPT > ResNet50 > Swin-Tiny, validating the superiority of pathology foundation models

### Key Findings

1. **Cross-platform generalization**: Strong performance is maintained when training on ST platforms and testing on Visium/Xenium platforms
2. **Downstream analysis**: Spots scored as high-cancer by SpaCRD but annotated as normal exhibit elevated expression of breast cancer marker genes (e.g., ERBB2), suggesting these may represent potential early-stage lesion regions
3. **KS distance analysis**: SpaCRD achieves far greater separation between predicted score distributions of healthy and tumor regions (median KS = 0.754) compared to SpaCell-Plus (0.494) and MEATRD (0.348)

## Highlights & Insights

1. **First framework to combine multimodal deep fusion with transfer learning for CTR detection**, without relying on marker gene prior knowledge
2. **Elegant VRBCA design**: BCA models cross-modal interactions from complementary perspectives, while RVAE enforces a class-structured latent space alongside denoising
3. **GMM adaptive thresholding**: No manual threshold setting is required during inference
4. Substantial improvement over baselines in cross-platform scenarios (training on ST → testing on Visium/Xenium), demonstrating robustness to batch effects
5. All experiments are conducted on a single RTX 3090, with manageable computational overhead

## Limitations & Future Work

1. Validation is limited to breast and colorectal cancers; generalizability to other cancer types (e.g., lung, brain tumors) remains unknown
2. The contrastive alignment stage uses a fixed symmetric weight $\alpha=0.5$; whether different tissue types require adjustment is not discussed
3. The number of neighborhood spots $k$ is a hyperparameter that may require different settings for ST platforms with varying spot densities
4. Reliance on UNI pretrained weights means that domain biases in UNI may propagate to downstream tasks

## Related Work & Insights

- **Spatial transcriptomics analysis**: SpaCell (simple concatenation), TESLA (requires marker genes), iStar (requires prior knowledge)
- **Visual anomaly detection**: STANDS/MEATRD adopt CV anomaly detection paradigms but are ill-suited to contiguous cancer regions
- **Pathology foundation models**: UNI provides strong histological features; CTransPath and CONCH are alternative choices
- **Transfer learning**: Knowledge transfer from source to target domain aligns representation spaces across different platforms and batches

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ — The cross-modal fusion design (BCA+RVAE) is original; introducing transfer learning into CTR detection is a first
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 23 datasets with rigorous cross-sample and cross-platform/batch settings
- **Writing Quality**: ⭐⭐⭐⭐ — Clear framework presentation with rich figures and tables
- **Value**: ⭐⭐⭐⭐ — Trainable on a single GPU with open-source code

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology](hifusion_hierarchical_intra-spot_alignment_and_regional_context_fusion_for_spati.md)
- [\[ICLR 2026\] HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](../../ICLR2026/medical_imaging/histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)
- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/medical_imaging/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[AAAI 2026\] Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)
- [\[AAAI 2026\] Dual-Path Knowledge-Augmented Contrastive Alignment Network for Spatially Resolved Transcriptomics](dual-path_knowledge-augmented_contrastive_alignment_network_for_spatially_resolv.md)

</div>

<!-- RELATED:END -->
