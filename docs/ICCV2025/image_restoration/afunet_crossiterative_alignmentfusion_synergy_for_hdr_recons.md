---
title: >-
  [Paper Note] AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm
description: >-
  [ICCV 2025][Image Restoration][HDR imaging] This paper formulates multi-exposure HDR reconstruction from a MAP estimation perspective, decomposes the problem into two alternating subproblems—alignment and fusion—via a spatial correspondence prior, and unfolds them into an end-to-end trainable AFUNet comprising SAM (spatial alignment), CFM (channel fusion), and DCM (data consistency) modules. The method achieves state-of-the-art performance on three HDR benchmarks…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "HDR imaging"
  - "deep unfolding network"
  - "MAP estimation"
  - "alignment-fusion alternating optimization"
  - "deghosting"
date: 2026-05-08
content_hash: d664a96ace419043
---

# AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm

**Conference**: ICCV 2025
**arXiv**: [2506.23537](https://arxiv.org/abs/2506.23537)  
**Code**: [https://github.com/eezkni/AFUNet](https://github.com/eezkni/AFUNet)  
**Area**: Other
**Keywords**: HDR imaging, deep unfolding network, MAP estimation, alignment-fusion alternating optimization, deghosting

## TL;DR
This paper formulates multi-exposure HDR reconstruction from a MAP estimation perspective, decomposes the problem into two alternating subproblems—alignment and fusion—via a spatial correspondence prior, and unfolds them into an end-to-end trainable AFUNet comprising SAM (spatial alignment), CFM (channel fusion), and DCM (data consistency) modules. The method achieves state-of-the-art performance on three HDR benchmarks, reaching PSNR-μ of 44.91 dB on the Kalantari dataset.

## Background & Motivation
Existing HDR reconstruction methods fall into two paradigms: "align-then-fuse" (pre-alignment followed by fusion, but pre-alignment may discard useful information) and "fusion-only" (bypassing explicit alignment, leading to ghosting artifacts). Both are empirically designed without a principled mathematical foundation. The core insight is that interleaving alignment within the fusion process through alternating iterations outperforms executing the two steps independently.

## Core Problem
How to provide a theoretically grounded framework for multi-exposure HDR reconstruction such that alignment and fusion mutually reinforce each other through progressive optimization?

## Method

### Overall Architecture
Three multi-exposure LDR images $(y_1, y_2, y_3)$ → SFEM shallow feature extraction → $T=4$ stages of alternating alignment-fusion unfolding network (AFM) → residual HDR image reconstruction. Each AFM stage: SAM aligns non-reference features → SFM spatial fusion → CFM channel fusion → DCM data consistency update → MLP + residual update.

### Key Designs
1. **MAP-based formulation + unfolding**: HDR reconstruction is modeled as MAP estimation (Eq. 2) with a spatial correspondence prior constraint. The HQS method decouples it into an alignment subproblem (gradient descent) and a fusion subproblem (proximal operator). Each iteration is unfolded into one AFM module with independently learnable parameters.
2. **Spatial Alignment Module (SAM)**: Based on window-based cross-attention, SAM aligns non-reference features $f_{\alpha_1}/f_{\alpha_3}$ with the intermediate reconstruction feature $f_x$. Keys and Values incorporate information from the degradation transform $D_i$ (learned via MLP), enabling the alignment process to be aware of exposure differences.
3. **Channel Fusion Module (CFM)**: Based on a channel-attention Transformer, CFM performs adaptive channel-wise fusion after spatial fusion (SFM), combining the previous-stage reconstruction feature $f_x^{t-1}$ with the aligned features.

### Loss & Training
- $\mathcal{L} = \mathcal{L}_1\text{(tone-mapped)} + 0.005 \times \mathcal{L}_\text{perceptual}\text{(VGG-19)}$
- Tone mapping uses the $\mu$-law function ($\mu = 5000$)
- Adam optimizer, batch size = 6, lr = $5\times10^{-4} \to 5\times10^{-6}$ cosine decay, 400 epochs
- Training patch: $128\times128$; data augmentation: random crop/rotation/flip
- Single RTX 4090 GPU

## Key Experimental Results

### Kalantari Dataset

| Method | PSNR-μ↑ | PSNR-l↑ | SSIM-μ↑ | HDR-VDP2↑ |
|--------|---------|---------|---------|-----------|
| CA-ViT | 44.32 | 42.18 | 0.9916 | 66.03 |
| SCTNet | 44.43 | 42.21 | 0.9918 | 66.64 |
| SAFNet | 44.66 | 43.18 | 0.9919 | 66.69 |
| LFDiff | 44.76 | 42.59 | 0.9919 | 66.54 |
| **AFUNet** | **44.91** | 42.59 | **0.9923** | **66.75** |

### Hu Dataset

| Method | PSNR-μ↑ | PSNR-l↑ |
|--------|---------|---------|
| LFDiff | 48.74 | 52.10 |
| **AFUNet** | **48.83** | **52.13** |

### Tel Dataset

| Method | PSNR-μ↑ | PSNR-l↑ |
|--------|---------|---------|
| SCTNet | 42.55 | 47.51 |
| **AFUNet** | **43.31** | **47.83** |

### Ablation Study
- SFM only: PSNR-μ = 43.94 → +SAM: 44.48 → +CFM: 44.62 → +DCM: 44.45 → Full (AFUNet): 44.91
- Alignment-then-Fusion (AF) order outperforms Fusion-then-Alignment (FA): 44.91 vs. 44.72
- Number of stages: 2→44.40, 3→44.83, 4→44.91 (default), 5→44.85, 6→44.93 (4 stages offers the best efficiency-performance trade-off)
- 3 stages already surpasses prior SOTA, demonstrating the intrinsic effectiveness of the proposed framework

## Highlights & Insights
- **Theory-driven architecture design**: MAP formulation + HQS unfolding provides a principled theoretical basis for alternating alignment and fusion, rather than purely empirical design.
- **Alignment as an iterative process rather than preprocessing**: The core innovation—alignment and fusion alternate, with each fusion result guiding the subsequent alignment step.
- **Window-based cross-attention for alignment**: Local windows are better suited for spatial alignment than global attention, as alignment primarily involves local structure and high-frequency details.
- **Practical value of deep unfolding**: Unfolding iterative algorithms into fixed-stage neural networks yields both theoretical interpretability and end-to-end trainability.

## Limitations & Future Work
- Only three-exposure inputs are validated; generalization to more exposures remains unexplored.
- SAM relies on window-based attention, which may limit alignment capacity in regions with large motion.
- The Kalantari dataset contains only 15 test samples, making the evaluation scale relatively small.
- No direct fair comparison is made against other unfolding-based methods (e.g., GAN-style iterations in MERF).

## Related Work & Insights
- **vs. CA-ViT/SCTNet (Transformer-based)**: These methods still follow the "align-then-fuse" or "fusion-only" paradigm; AFUNet's alternating iterative paradigm proves more effective (+0.48–0.59 dB).
- **vs. LFDiff (Diffusion-based)**: AFUNet requires no additional diffusion sampling cost yet achieves superior PSNR-μ (44.91 vs. 44.76).
- **vs. Mai et al. (DUN-based)**: Prior unfolding methods treat HDR reconstruction as low-rank completion, imposing overly strong assumptions; AFUNet is more flexible and general.

## Relevance to My Research
- The deep unfolding paradigm—transforming iterative optimization into a learnable architecture—is transferable to other complex reconstruction tasks.
- The alternating alignment-fusion iterative paradigm can be adapted to video inpainting, multi-view fusion, and related problems.

## Rating
- Novelty: ⭐⭐⭐⭐ — The MAP formulation → unfolding idea is relatively novel in the HDR domain, though deep unfolding itself is a mature technique.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets + ablations + paradigm analysis + stage count analysis; reasonably comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivation is clear; the path from MAP to unfolding is complete and traceable.
- Value: ⭐⭐⭐ — The unfolding idea is instructive, but HDR reconstruction is not a core research focus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](../../CVPR2025/image_restoration/vision-language_gradient_descent-driven_all-in-one_deep_unfolding_networks.md)
- [\[ECCV 2024\] Intrinsic Single-Image HDR Reconstruction](../../ECCV2024/image_restoration/intrinsic_single-image_hdr_reconstruction.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)
- [\[CVPR 2026\] LRHDR: Learning Representation-enhanced HDR Video Reconstruction](../../CVPR2026/image_restoration/lrhdr_learning_representation-enhanced_hdr_video_reconstruction.md)
- [\[CVPR 2026\] LRDUN: A Low-Rank Deep Unfolding Network for Efficient Spectral Compressive Imaging](../../CVPR2026/image_restoration/lrdun_a_low-rank_deep_unfolding_network_for_efficient_spectral_compressive_imagi.md)

</div>

<!-- RELATED:END -->
