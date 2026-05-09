---
title: >-
  [Paper Note] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data
description: >-
  [CVPR 2026][Segmentation][Incomplete multimodal] This paper proposes SGMA—a Semantic-Guided Modality-Aware segmentation framework—that employs Semantic-Guided Fusion (SGF) to reduce intra-class variance and reconcile cross-modal conflicts, and Modality-Aware Sampling (MAS) to balance training frequency for vulnerable modalities. On ISPRS, SGMA achieves Average mIoU +9.20% and Last-1 mIoU +18.26% for weak modalities compared to the SOTA method IMLT.
tags:
  - CVPR 2026
  - Segmentation
  - Incomplete multimodal
  - semantic-guided fusion
  - modality-aware sampling
  - remote sensing segmentation
  - vulnerable modality
date: 2026-05-08
content_hash: 12e3f904a3b24c69
---

# SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data

**Conference**: CVPR 2026
**arXiv**: [2603.02505](https://arxiv.org/abs/2603.02505)
**Code**: Unavailable
**Area**: Image Segmentation
**Keywords**: Incomplete multimodal, semantic-guided fusion, modality-aware sampling, remote sensing segmentation, vulnerable modality

## TL;DR

This paper proposes SGMA—a Semantic-Guided Modality-Aware segmentation framework—that employs Semantic-Guided Fusion (SGF) to reduce intra-class variance and reconcile cross-modal conflicts, and Modality-Aware Sampling (MAS) to balance training frequency for vulnerable modalities. On ISPRS, SGMA achieves Average mIoU +9.20% and Last-1 mIoU +18.26% for weak modalities compared to the SOTA method IMLT.

## Background & Motivation

**State of the Field**: Remote sensing multimodal segmentation integrates complementary information from RGB, DSM, NIR, SAR, and other sources. However, in real-world systems, sensor failures or incomplete coverage frequently lead to missing modalities, giving rise to the Incomplete Multimodal Semantic Segmentation (IMSS) problem. Existing approaches include modality dropout (M3L), MAE-based pretraining (IMLT), and contrastive alignment (MAGIC).

**Limitations of Prior Work**: IMSS simultaneously faces three intertwined challenges: (1) **Modality imbalance**: strong modalities such as RGB suppress vulnerable modalities such as DSM, NIR, and SAR; (2) **Intra-class variance**: buildings of the same category exhibit large variation in scale, orientation, and shape in remote sensing imagery, with small buildings yielding sparse features; (3) **Cross-modal heterogeneity**: rooftops and ground surfaces appear similarly colored in RGB but differ in DSM elevation, while grassland and bare ground share similar elevation in DSM but differ in RGB color—semantic cues conflict across modalities.

**Root Cause**: Contrastive alignment (IMLT/MAGIC) forces all modalities into a shared space, discarding modality-specific discriminative information; dropout strategies increase exposure to missing modalities but do not selectively reinforce learning for vulnerable modalities; no existing method addresses all three challenges simultaneously.

**Paper Goals**: Achieve balanced multimodal learning, reduced intra-class variance, and reconciled cross-modal inconsistency simultaneously under arbitrary modality-missing scenarios.

**Starting Point**: Global semantic prototypes serve as cross-modal "intermediate anchors"—compressing dense pixel representations into class-level semantic representations to reduce intra-class variance, while using prototype–feature alignment to measure modality reliability for adaptive fusion, and employing reliability-inverted sampling to prioritize training on vulnerable modalities.

**Core Idea**: Semantic prototypes jointly address intra-class variance (by providing class-level consistent representations) and cross-modal heterogeneity (through adaptive weighting). Their byproduct—modality reliability scores—drives the sampling strategy to resolve modality imbalance.

## Method

### Overall Architecture

A shared-weight encoder extracts four-scale features for each modality → SGF module: modality-specific projector (MP) → class-aware semantic filter (CSF) → global prototype construction → spatial perceiver (SP, MHA) → robustness perceiver (RP, MHA) → MAS module: invert robustness scores → sampling probabilities → stochastically select a vulnerable modality for independent training. During training, outputs from both the SGF and MAS branches are jointly optimized; at inference, only SGF is retained.

### Key Designs

1. **Semantic-Guided Fusion (SGF)**:
    - **Function**: Constructs global class-level semantic prototypes and fuses multimodal features via attention mechanisms while estimating modality reliability.
    - **Mechanism**: A $1\times1$ Conv projects features from $C_i$ dimensions to $K$ classes, yielding a compact representation $c_m^i$; matrix multiplication with the semantic feature matrix produces global prototypes $p_{se}^{i,k} \in \mathbb{R}^C$. The spatial perceiver (SP) performs MHA using prototypes as queries and multimodal features as keys/values, producing semantically activated outputs $a_{se}^{i,k}$. The robustness perceiver (RP) performs a second MHA using the fused features as queries, outputting the fused feature $f_{SGF}^i$ and per-modality robustness maps $r_m^i$.
    - **Design Motivation**: Global prototypes provide class-level consistent representations that reduce intra-class variance (small and large buildings share the same prototype); the attention weights of RP naturally reflect each modality's contribution to each class (DSM contributes more to buildings, NIR to vegetation), enabling category-dependent adaptive fusion.

2. **Modality-Aware Sampling (MAS)**:
    - **Function**: Dynamically adjusts modality selection probabilities during training using robustness scores produced as a byproduct of SGF.
    - **Mechanism**: Robustness scores are inverted as $\hat{r}_m^i = \frac{1/r_m^i}{\sum_{m'} 1/r_{m'}^i}$; spatial averaging yields sampling probabilities $s_m^i$. At each training iteration, one modality is selected according to these probabilities and independently passed through SGF to obtain $f_{MAS}^i$, which is supervised separately.
    - **Design Motivation**: Vulnerable modalities with low robustness are selected more frequently—effectively a SoftMin operation computed directly on normalized attention weights, avoiding the smoothing effect of applying SoftMin to SoftMax outputs. No additional hyperparameters are required, and the training overhead consists of only one additional forward pass.

### Loss & Training

$$\mathcal{L}_{IMSS} = 2 \cdot \mathcal{L}_{SGF} + 1 \cdot \mathcal{L}_{MAS}$$

- Both terms are standard cross-entropy losses.
- AdamW optimizer, lr = 6e-5, polynomial decay (power 0.9), 200 epochs, 10-epoch warmup at 10% lr.
- Trained on 4× A100 GPUs; plug-and-play design adds only 4.79M parameters and 0.79G FLOPs.

## Key Experimental Results

### Main Results

| Dataset | Metric | SGMA | IMLT | MAGIC | Gain (vs. SOTA) |
|--------|------|------|------|-------|-------------|
| ISPRS (PVT) | Avg mIoU | **79.55%** | 70.35% | 67.43% | **+9.20%** |
| ISPRS (PVT) | Top-1 mIoU | **86.84%** | 85.12% | 84.75% | +0.34% |
| ISPRS (PVT) | Last-1 mIoU | **57.05%** | 38.78% | 34.34% | **+18.26%** |
| ISPRS (ResNet) | Avg mIoU | **76.42%** | 62.75% | 66.21% | **+10.21%** |
| DFC2023 | Avg mIoU | **81.91%** | 74.25% | — | +7.66% |
| DELIVER | Avg mIoU | **55.49%** | 47.17% | — | +8.31% |

### Ablation Study

| SGF | MAS | Avg mIoU | Last-1 mIoU |
|-----|-----|----------|-------------|
| ✗ | ✗ | 46.51% | 2.61% |
| ✓ | ✗ | 49.13% | 7.01% |
| ✗ | ✓ | 62.07% | 29.86% |
| ✓ | ✓ | **79.55%** | **57.05%** |

| Analysis Dimension | w/o SGF | w/ SGF | Improvement |
|----------|-------|-------|------|
| Building intra-class variance | 0.84 | 0.74 | −12% |
| DSM silhouette score | 0.03 | 0.30 | 10× |
| NIR silhouette score | 0.05 | 0.31 | 6.2× |

### Key Findings

- Improvements on vulnerable modality Last-1 are remarkable (+18.26% / +50.04% absolute), confirming MAS as the key driver for balancing vulnerable modalities.
- SGF and MAS are highly complementary: SGF reduces intra-class variance (0.84 → 0.74), while MAS strengthens vulnerable modalities (silhouette score 0.03 → 0.30).
- Cross-backbone generalization: SGMA consistently outperforms all baselines on both PVT-v2-b2 and ResNet-50.
- Cross-domain generalization: effective on both remote sensing (ISPRS/DFC2023) and autonomous driving (DELIVER) benchmarks.

## Highlights & Insights

- Plug-and-play design with only 4.79M additional parameters and 0.79G FLOPs, offering strong practical utility.
- Semantic prototypes simultaneously serve to reduce intra-class variance, assess modality reliability, and guide fusion weights—a single design addressing three distinct problems.
- The reliability-inversion sampling design is elegant and parameter-free, requiring no additional hyperparameters or modality-specific architectural modifications.
- Combinations of two vulnerable modalities (e.g., DSM + SAR) can even surpass a single strong modality, demonstrating exploitable complementarity even among weak modalities.

## Limitations & Future Work

- The framework assumes all modalities are available during training, whereas practical scenarios may also involve missing modalities at training time.
- Interpretability mechanisms for modality-specific learning dynamics are lacking.
- Validation on temporal multimodal sequences (e.g., video remote sensing) has not been conducted.
- Modality selection in MAS relies on random sampling—a deterministic curriculum strategy warrants future investigation.

## Related Work & Insights

- **MAGIC (ECCV 2024)**: Modality-agnostic segmentation, 67.43% → Ours 79.55%. MAGIC employs contrastive alignment but over-alignment discards modality-specific discriminative information.
- **IMLT (IEEE TGRS 2024)**: The first method specifically targeting remote sensing IMSS, using contrastive learning and MAE pretraining, 70.35% → Ours 79.55%. MAE focuses on low-level pixel reconstruction rather than high-level semantic understanding.
- **Insights**: The "bootstrapping" design paradigm—using task byproducts (robustness scores) to guide training strategy—is a transferable and inspiring design principle.

## Rating

- ⭐⭐⭐⭐ **Novelty**: Semantic prototype-guided fusion combined with inverted reliability sampling addresses three intertwined problems within a single framework.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Evaluated on 3 remote sensing and 1 autonomous driving datasets, with 2 backbones, detailed ablation studies, and visualization analyses.
- ⭐⭐⭐⭐ **Writing Quality**: Problem decomposition is clear (modality imbalance / intra-class variance / cross-modal conflict), with method components explicitly mapped to each challenge.
- ⭐⭐⭐⭐⭐ **Value**: Delivers significant practical contributions to incomplete multimodal remote sensing segmentation; the plug-and-play design lowers the barrier to adoption.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [\[CVPR 2026\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportionaware_dynamic_adaptive_sali.md)
- [\[CVPR 2026\] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)
- [\[CVPR 2026\] SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons](semlayer_semantic-aware_generative_segmentation_and_layer_construction_for_abstr.md)
- [\[CVPR 2026\] PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation](pixdlm_uav_reasoning_segmentation.md)

<!-- RELATED:END -->
