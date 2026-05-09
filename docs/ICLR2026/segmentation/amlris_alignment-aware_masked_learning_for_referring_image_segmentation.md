---
title: >-
  [Paper Note] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation
description: >-
  [ICLR 2026][Segmentation][referring image segmentation] This paper proposes an Alignment-aware Masked Learning (AML) strategy that quantifies vision-language patch-level alignment and filters low-alignment pixels, enabling RIS models to focus on reliable regions during training. Without any architectural modifications, AML achieves state-of-the-art performance across all 8 splits of RefCOCO benchmarks.
tags:
  - ICLR 2026
  - Segmentation
  - referring image segmentation
  - vision-language alignment
  - masked learning
  - cross-modal similarity
date: 2026-05-08
content_hash: e10eac8222f7f9dd
---

# AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation

**Conference**: ICLR 2026
**arXiv**: [2602.22740](https://arxiv.org/abs/2602.22740)
**Code**: [GitHub](https://github.com/pipashu1/AMLRIS)
**Area**: Image Segmentation
**Keywords**: referring image segmentation, vision-language alignment, masked learning, cross-modal similarity

## TL;DR
This paper proposes an Alignment-aware Masked Learning (AML) strategy that quantifies vision-language patch-level alignment and filters low-alignment pixels, enabling RIS models to focus on reliable regions during training. Without any architectural modifications, AML achieves state-of-the-art performance across all 8 splits of RefCOCO benchmarks.

## Background & Motivation

### Root Cause

**Root Cause**: **State of the Field**: 1. Referring Image Segmentation (RIS) requires accurately segmenting target objects in images based on natural language expressions, relying on fine-grained cross-modal alignment.
2. RIS training typically provides only a single annotated target per sample, resulting in sparse supervision signals.
3. Understanding expressions such as "the giraffe closest to the person" requires reasoning about spatial relationships among other objects in the visual context.
4. Existing methods (LAVT/CARIS/DETRIS) enhance alignment through complex fusion modules, but applying loss over all pixels introduces unreliable gradients.
5. Under dense supervision, models tend to overfit to regions irrelevant to the referring expression.
6. Common data augmentation strategies (flipping, color jittering) can disrupt the semantic consistency of referring expressions.

## Method

**Overall Architecture**: Two-stage training with shared parameters — Stage 1 computes the alignment map and generates masks via a forward pass; Stage 2 trains normally on the masked images.

**PatchMax Matching Evaluation (PMME)**:
- $\ell_2$-normalizes visual features $V$ and text features $T$ respectively
- Projects both into a shared $D_a$-dimensional space using random Gaussian matrices $W_i, W_t$ (distance-preserving via Johnson-Lindenstrauss)
- Computes $S_{norm} = \text{SoftMax}(V'T'^{\top})$, and assigns each patch the maximum similarity score with its best-matching token

**Alignment-Aware Filtering Mask (AFM)**:
- Bilinearly upsamples patch-level similarity scores to pixel level
- Pixels below threshold $\tau$ are marked as weakly aligned; a random fraction $1-\rho$ is retained to prevent over-filtering
- Masks are aggregated at the block level (any weakly aligned pixel causes the entire block to be masked), and the corresponding input image regions are zeroed out

**Key Hyperparameters**: $\tau=0.4$, $\rho=0.25$, block size $32\times32$, $D_a=2048$

**Loss**: Standard cross-entropy segmentation loss $\mathcal{L}_{seg}$, with no additional loss terms

## Key Experimental Results

### Main Results

| Method | RefCOCO val | RefCOCO+ val | RefCOCOg val | Avg mIoU |
|--------|-------------|--------------|--------------|----------|
| CARIS* | 76.77 | 69.33 | 68.87 | 71.8 |
| MagNet | 77.43 | 70.10 | 68.53 | 72.1 |
| **AMLRIS** | **77.89** | **71.33** | **69.24** | **72.9** |

- oIoU: RefCOCO val 75.45 (+0.80 vs. CARIS), RefCOCO+ val 67.37 (+1.83)
- Achieves SOTA on all 8 splits
- Cross-dataset robustness: trained only on RefCOCO+, outperforms baseline under 7 perturbation scenarios
- Overhead: only 4.9% additional memory and 17.2% additional training time; zero inference overhead

## Key Experimental Results

### Main Results (mIoU)

| Method | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val | test | Avg |
|--------|-------------|-------|-------|--------------|-------|-------|--------------|------|-----|
| LAVT | 74.46 | 76.89 | 70.94 | 65.81 | 70.97 | 59.23 | 63.34 | 63.62 | 68.0 |
| CGFormer | 76.93 | 78.70 | 73.32 | 68.56 | 73.76 | 61.72 | 67.57 | 67.83 | 71.1 |
| CARIS* | 76.77 | 79.03 | 74.56 | 69.33 | 74.51 | 62.69 | 68.87 | 68.51 | 71.8 |
| MagNet | 77.43 | 79.43 | 74.11 | 70.10 | 74.50 | 63.59 | 68.53 | 69.15 | 72.1 |
| **AMLRIS** | **77.89** | **79.53** | **74.99** | **71.33** | **75.61** | **64.61** | **69.24** | **69.73** | **72.9** |

### Ablation Study

| Configuration | RefCOCO val mIoU | Notes |
|---------------|-----------------|-------|
| CARIS baseline | 76.77 | No masking |
| + Random Mask | 76.92 | Marginal improvement |
| + PMME + AFM (full AML) | **77.89** | Alignment-aware masking is effective |
| AML on DETRIS | 75.64→76.12 | Consistent gain across architectures |
| AML on ReLA | +0.5–1.0 | Also effective |

### Cross-Dataset Robustness

| Perturbation | CARIS baseline | AMLRIS |
|-------------|---------------|--------|
| Standard eval | 69.33 | **71.33** |
| Occlusion | 65.1 | **68.4** |
| Noise | 64.8 | **67.9** |
| Blur | 66.2 | **69.1** |
| Color shift | 67.5 | **70.2** |

### Key Findings
- SOTA on all 8 splits, with average mIoU of 72.9 (+0.8 vs. MagNet)
- oIoU metrics are also comprehensively best, reaching 67.37 on RefCOCO+ val (+1.83 vs. CARIS)
- Random masking is nearly ineffective (+0.15), confirming that alignment-aware mask selection is the critical factor
- Advantages are more pronounced under perturbation scenarios such as occlusion and noise (+3.1–3.3), indicating that the model learns more robust alignment features
- Overhead is minimal: only 4.9% additional memory and 17.2% additional training time; zero inference overhead (masking stage is skipped at inference)
- Can be seamlessly integrated into various RIS frameworks including DETRIS, CARIS, and ReLA

## Highlights & Insights
- **Plug-and-play training strategy**: No architectural modifications and no inference cost — a purely training-stage improvement with zero deployment overhead.
- **Theoretical guarantee**: The Johnson-Lindenstrauss lemma is formally applied to prove that random projection preserves cross-modal inner products, providing a mathematical foundation for the alignment metric.
- **Counter-intuitive effectiveness**: The model never sees complete images during training (some regions are always masked), yet performs better on complete images at inference — demonstrating that filtering weakly aligned regions genuinely eliminates misleading gradients.
- **PatchMax matching strategy**: Assigning each patch the similarity score of its best-matching token better captures local alignment quality compared to average matching.

## Limitations & Future Work
- The threshold $\tau=0.4$ and dropout ratio $\rho=0.25$ require manual tuning, and different datasets may necessitate different configurations.
- The random-projection-based alignment metric relies on initial feature similarity and may miss deep semantic alignment as the feature space evolves during later training stages.
- The two-stage forward pass introduces a 17.2% increase in training time, which may become a bottleneck at large data scales.
- Evaluation is limited to the RefCOCO series; generalization to open-vocabulary, large-scale, or more complex scenarios remains unverified.
- Block-level masking (32×32) may inadvertently cover target regions in small-object scenarios.

## Related Work & Insights
- **vs. CARIS/LAVT/DETRIS**: These serve as the baseline and comparison methods, improving alignment via fusion architectures but all applying full-pixel losses — AML innovates from the perspective of optimization signals.
- **vs. MaskRIS/NeMo/MagNet**: Data augmentation approaches for RIS improvement that still apply loss over all pixels; AML directly suppresses low-quality gradients.
- **vs. CRIS**: A CLIP-based pixel-level adaptation method that performs alignment in the pretrained feature space; AML is applicable to arbitrary backbones.

## Rating
- Novelty: ⭐⭐⭐⭐ The alignment-aware masking idea is concise and novel; PatchMax + JL projection is theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full-split SOTA + robustness evaluation + cross-architecture validation + complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; algorithmic pseudocode is complete.
- Value: ⭐⭐⭐⭐ A general training strategy that can be plug-and-play integrated into existing RIS methods.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](../../CVPR2026/segmentation/phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](../../CVPR2026/segmentation/spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)
- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](../../CVPR2026/segmentation/sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](../../ICCV2025/segmentation/latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](../../NeurIPS2025/segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)

<!-- RELATED:END -->
