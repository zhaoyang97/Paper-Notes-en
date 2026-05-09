---
title: >-
  [Paper Note] AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer
description: >-
  [AAAI 2026][Image Generation][Anomaly Generation] This work formulates zero-shot anomaly generation as a text-guided localized style transfer problem. A lightweight U-Net trained with CLIP-based losses stylizes masked regions of normal images into semantically aligned anomalous images. With only 263M total parameters (0.61M trainable), AnoStyler surpasses diffusion-based baselines on MVTec-AD and VisA while significantly improving downstream anomaly detection performance.
tags:
  - AAAI 2026
  - Image Generation
  - Anomaly Generation
  - Zero-Shot
  - Style Transfer
  - CLIP
  - Industrial Defect Detection
date: 2026-05-08
content_hash: 35380a696fa20425
---

# AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer

**Conference**: AAAI 2026
**arXiv**: [2511.06687v1](https://arxiv.org/abs/2511.06687v1)
**Code**: [https://github.com/yulimso/AnoStyler](https://github.com/yulimso/AnoStyler)
**Area**: Image Generation
**Keywords**: Anomaly Generation, Zero-Shot, Style Transfer, CLIP, Industrial Defect Detection

## TL;DR
This work formulates zero-shot anomaly generation as a text-guided localized style transfer problem. A lightweight U-Net trained with CLIP-based losses stylizes masked regions of normal images into semantically aligned anomalous images. With only 263M total parameters (0.61M trainable), AnoStyler surpasses diffusion-based baselines on MVTec-AD and VisA while significantly improving downstream anomaly detection performance.

## Background & Motivation

**State of the Field**: Real anomaly images are extremely scarce and highly diverse in industrial anomaly detection. Existing anomaly generation methods suffer from three major limitations: (1) heuristic methods (CutPaste, DRAEM, etc.) generate anomalies lacking visual realism; (2) diffusion-based methods (AnoDiff, AnomalyAny, etc.) produce more realistic results but require large numbers of normal images and impose heavy computational costs (>1B parameters); (3) few-shot methods still require a small set of real anomaly images, which are costly to collect.

**Starting Point**: Style transfer is naturally suited for anomaly generation—it can modify local visual attributes while preserving the overall image content. However, it has never been applied to anomaly generation prior to this work.

## Core Problem
**How to generate visually realistic and semantically aligned localized anomaly images in a zero-shot, lightweight setting, using only a single normal image and a text description (object category + defect type)?**

## Method

### Overall Architecture
AnoStyler consists of three stages:
1. **Shape-Guided Mask Generation**: Meta-Shape Priors (Line, Dot, Freeform) are used to generate anomaly region masks $\mathbf{M}_a$.
2. **Dual-Class Text Prompt Generation**: Based on category $[c]$ and defect type $[d]$, 165 normal prompts $\mathcal{T}_n$ and 165 anomaly prompts $\mathcal{T}_a$ are generated.
3. **Text-Guided Localized Anomaly Generation**: A lightweight U-Net $\mathcal{F}$, guided by CLIP-based losses, stylizes the masked region of $\mathbf{I}_n$ into an anomalous image.

**Input**: A single normal image + category label + defect type text
**Output**: Synthesized anomaly image $\mathbf{I}_a$ + anomaly mask $\mathbf{M}_a$

### Key Designs

1. **Meta-Shape Priors**: Three parameter-free geometric primitives (Line, Dot, Freeform) cover diverse anomaly morphologies. Final masks are generated via random composition and intersection with the foreground. These priors are more realistic than Perlin noise or rectangular crops and are extremely lightweight (mask generation takes only 0.09–115 ms). Object-type and texture-type categories are handled separately: SAM extracts foreground for the former, while the entire image serves as foreground for the latter.

2. **Mask-Weighted Co-Directional Loss**: An extension of CLIPstyler's directional alignment loss. The global term $\mathcal{L}_{gdir}$ aligns the cosine distance between image change direction $\Delta\mathbf{h}_I$ and text change direction $\Delta\mathbf{h}_T$. The patch-level term $\mathcal{L}_{pdir}$ applies similar alignment to randomly cropped patches, but weights each patch by its mask coverage ratio $r_j$—patches lying within the anomaly region contribute more, ensuring stylization focuses on the masked area.

3. **Masked CLIP Loss**: An additional loss $\mathcal{L}_{mclip}$ computes the cosine distance between the masked region $\mathbf{I}_a \odot \mathbf{M}_a$ and the anomaly prompts, further reinforcing semantic alignment within the local region.

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_{mwcd} + \lambda_{mclip} \cdot \mathcal{L}_{mclip} + \lambda_c \cdot \mathcal{L}_c + \lambda_{tv} \cdot \mathcal{L}_{tv}$

where $\mathcal{L}_c$ is a VGG content loss (preserving structure) and $\mathcal{L}_{tv}$ is a total variation loss (spatial smoothness).

**Key**: A separate U-Net is optimized per image (only 75 steps of Adam), with 0.61M trainable parameters and the CLIP encoder frozen. This test-time optimization paradigm allows the model to adapt to the specific characteristics of each input image.

## Key Experimental Results

**Anomaly Generation Quality:**

| Dataset | Metric | AnoStyler | AnomalyAny | AnoDiff (few-shot) | RealNet |
|---------|--------|-----------|------------|-------------------|---------|
| MVTec-AD | IS↑ | **2.04** | 2.02 | 1.80 | 1.64 |
| MVTec-AD | IC-L↑ | 0.32 | **0.33** | 0.32 | 0.22 |
| VisA | IS↑ | **1.55** | 1.41 | 1.50 | 1.53 |
| VisA | IC-L↑ | **0.32** | 0.19 | 0.29 | 0.29 |

**Downstream Anomaly Detection:**

| Dataset | Metric | AnoStyler | AnomalyAny | RealNet | AnoDiff (few-shot) |
|---------|--------|-----------|------------|---------|-------------------|
| MVTec-AD | I-AUC | **98.0** | 95.2 | 95.2 | 99.2 |
| MVTec-AD | P-AUC | **94.4** | 89.0 | 94.0 | 99.1 |
| VisA | I-AUC | **93.9** | 88.9 | 92.6 | 86.9 |
| VisA | P-AUC | **93.8** | 90.4 | 92.2 | 93.2 |

### Ablation Study
- **Incremental loss addition**: Baseline (original CLIPstyler) IS=1.70, I-AUC=88.2 → +$\mathcal{L}_{gdir}$: IS=1.86, I-AUC=95.2 → +$\mathcal{L}_{pdir}$: IS=1.96, I-AUC=96.7 → +$\mathcal{L}_{mclip}$: IS=2.04, I-AUC=98.0. Each component contributes positively.
- **Computational efficiency**: AnoStyler requires 9.5 TFLOPs vs. AnomalyAny's 22.8 TFLOPs, a ~58% reduction.
- **Parameter count**: 263M total (including frozen CLIP and SAM), with only 0.61M trainable. All diffusion-based baselines exceed 1B parameters.
- **Statistical significance**: Friedman and Wilcoxon tests confirm that AnoStyler significantly outperforms most methods on both IS and IC-L.

## Highlights & Insights
- **Elegant problem formulation**: This is the first work to model anomaly generation as localized style transfer—a more principled framing than generation from scratch, since anomalies are fundamentally local attribute modifications.
- **Extreme lightweight design**: With only 0.61M trainable parameters, AnoStyler runs on a single RTX 2080Ti (11 GB), making it highly practical for industrial deployment.
- **Zero-shot setting**: Only a single normal image is required to generate anomalies, eliminating the need for large-scale data collection or dataset-level model training.
- **Mask-weighted patch loss**: Using mask coverage ratio as a soft weighting scheme is a simple yet effective localization strategy.
- **Meta-Shape Priors**: The three geometric primitives cover linear (scratches), dot-shaped (spots), and freeform (diffuse) anomaly morphologies, better approximating real defect patterns than Perlin noise.

## Limitations & Future Work
- **Per-image optimization**: Although lightweight, generating each anomaly image still requires 75 optimization steps, precluding feed-forward instantaneous inference.
- **CLIP semantic limitations**: CLIP's understanding of industrial defects is limited (e.g., domain-specific terms such as "contamination" or "thread"), and the precision of text guidance is constrained by CLIP's pretraining data distribution.
- **Misalignment between generated masks and real defect locations**: The generated masks are random geometric shapes, independent of where real defects actually occur (e.g., defects on screws are more likely to appear on the thread region).
- **Coarse handling of texture categories**: The entire image is used as foreground without accounting for semantic differences across regions in texture images.
- **Fixed downstream detector**: Evaluation uses only a U-Net-based anomaly detector; compatibility with stronger detectors (e.g., PatchCore, EfficientAD) remains unvalidated.

## Related Work & Insights

| Method | Core Idea | Key Difference from AnoStyler |
|--------|-----------|-------------------------------|
| **CutPaste/DRAEM** | Heuristic image manipulation (copy-paste, texture injection) | Insufficient realism; AnoStyler achieves semantic alignment via CLIP guidance |
| **AnoDiff** (few-shot) | Diffusion model + few real anomaly images | Requires real anomaly images, >1B parameters; AnoStyler is zero-shot with 0.61M trainable params |
| **AnomalyAny** | Stable Diffusion + text guidance | Also zero-shot and text-guided, but relies on heavy diffusion models; AnoStyler uses style transfer for greater efficiency |
| **RealNet** | Diffusion model + denoising perturbation | Requires large-scale normal images to train the diffusion model; AnoStyler operates from a single image |

AnoStyler's core advantage lies in replacing generative models with style transfer, substantially reducing resource requirements while maintaining competitive quality.

The style transfer approach to data augmentation can generalize to other data-scarce scenarios (e.g., lesion generation in medical imaging, target synthesis in remote sensing). The Meta-Shape Priors concept can enhance other methods requiring anomaly region masks. The mask-weighted CLIP loss strategy is transferable to any text-guided image editing task requiring localized modification.

## Rating
- Novelty: ⭐⭐⭐⭐ First application of style transfer to anomaly generation; novel problem formulation
- Experimental Thoroughness: ⭐⭐⭐⭐ Two standard benchmarks, multiple baselines, complete ablation analysis, and statistical significance tests
- Writing Quality: ⭐⭐⭐⭐ Clear method description with rich figures and tables
- Value: ⭐⭐⭐⭐ Direct practical value for industrial anomaly detection deployment; lightweight design addresses real-world bottlenecks

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](../../CVPR2026/image_generation/ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](../../ICCV2025/image_generation/domain_generalizable_portrait_style_transfer.md)
- [\[ACL 2026\] MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](../../ACL2026/image_generation/mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)
- [\[CVPR 2026\] TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering](../../CVPR2026/image_generation/textpecker_rewarding_structural_anomaly_quantification_for_enhancing_visual_text.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)

<!-- RELATED:END -->
