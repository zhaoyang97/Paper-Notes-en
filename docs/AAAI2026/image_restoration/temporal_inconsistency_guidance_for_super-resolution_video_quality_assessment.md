---
title: >-
  [Paper Note] Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment
description: >-
  [AAAI 2026][Image Restoration][Video Quality Assessment] This paper proposes TIG-SVQA, a framework that, for the first time, incorporates temporal inconsistency as an explicit guidance signal for super-resolution video quality assessment. The framework introduces an Inconsistency-Highlighted Spatial Module (IHSM) and an Inconsistency-Guided Temporal Module (IGTM), achieving SRCC scores of 0.950, 0.942, and 0.939 on the SFD, MFD, and Combined-VSR datasets, respectively, surpassing all existing IQA/VQA methods.
tags:
  - AAAI 2026
  - Image Restoration
  - Video Quality Assessment
  - Super-Resolution
  - Temporal Inconsistency
  - Transformer
  - Visual Working Memory
date: 2026-05-08
content_hash: 12385ab107555c5f
---

# Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment

**Conference**: AAAI 2026
**arXiv**: [2412.18933](https://arxiv.org/abs/2412.18933)
**Code**: [Lighting-YXLI/TIG-SVQA-main](https://github.com/Lighting-YXLI/TIG-SVQA-main)
**Area**: Image Restoration
**Keywords**: Video Quality Assessment, Super-Resolution, Temporal Inconsistency, Transformer, Visual Working Memory

## TL;DR

This paper proposes TIG-SVQA, a framework that, for the first time, incorporates temporal inconsistency as an explicit guidance signal for super-resolution video quality assessment. The framework introduces an Inconsistency-Highlighted Spatial Module (IHSM) and an Inconsistency-Guided Temporal Module (IGTM), achieving SRCC scores of 0.950, 0.942, and 0.939 on the SFD, MFD, and Combined-VSR datasets, respectively, surpassing all existing IQA/VQA methods.

## Background & Motivation

### Unique Distortions in Super-Resolution Video

With the rapid advancement of super-resolution (SR) techniques, SR videos introduce a distinctive class of distortions — hallucinated textures and temporal flickering — that are fundamentally different from traditional compression artifacts or user-generated degradations. Existing VQA methods are primarily designed for conventional distortions and inadequately model the temporal inconsistency inherent in SR videos, necessitating dedicated evaluation approaches.

### The Criticality of Temporal Inconsistency

Temporal inconsistency refers to irregular variations between consecutive frames in dynamic scenes, such as motion artifacts, abrupt transitions, and unnatural visual changes. Existing VQA methods model temporal relationships via frame differencing, optical flow analysis, and 3D-CNNs, but none explicitly quantify the level of temporal inconsistency or investigate its correlation with human perception. Notably, the SR enhancement process amplifies temporal inconsistency, making this issue particularly prominent in SR-VQA.

### Motion vs. Temporal Inconsistency

Through empirical analysis, the authors find that motion complexity exhibits weak correlation with perceptual quality — because scene content tends to mask temporal artifacts — whereas the differential of motion information, i.e., temporal inconsistency, correlates highly with perceptual quality. On the Combined-VSR dataset, motion signals achieve SRCC/PLCC of 0.885/0.913, while temporal inconsistency reaches 0.939/0.942. This finding provides strong justification for using temporal inconsistency as a guidance signal in SR-VQA.

## Core Problem

How to design an SR video quality assessment method that explicitly leverages temporal inconsistency information to guide spatial feature extraction and temporal feature aggregation, thereby better aligning with human perceptual preferences?

## Method

### Overall Architecture

TIG-SVQA consists of three core components: (1) temporal inconsistency quantification, (2) the Inconsistency-Highlighted Spatial Module (IHSM), and (3) the Inconsistency-Guided Temporal Module (IGTM).

### Temporal Inconsistency Quantification

Given an SR video $V_D$ and a reference video $V_R$, temporal inconsistency information is computed via optical flow difference:

$$V_I = \|OF(V_R) - OF(V_D)\|_2$$

where $OF(\cdot)$ denotes optical flow computation. $V_I$ is then decomposed into coarse-grained (low-pass filtered, capturing large-scale motion changes) and fine-grained (high-pass filtered, capturing subtle inconsistencies) components:

$$V_I^C = \mathcal{F}^{-1}(H_L \cdot \mathcal{F}(V_I)), \quad V_I^F = \mathcal{F}^{-1}((1-H_L) \cdot \mathcal{F}(V_I))$$

After normalization, these are weighted onto the SR video to highlight inconsistent regions: $\hat{V_D}^{C/F} = \text{Norm}(V_I^{C/F}) \times V_D + V_D$.

### Inconsistency-Highlighted Spatial Module (IHSM)

- **Coarse-grained branch**: Employs a modified Swin Transformer that introduces a Deformable Window Super-Attention (DW-SA) block at the third stage, adjusting window positions via learnable offsets and upsampling intra-window features with sub-pixel convolution to enhance modeling of large-scale inconsistent regions.
- **Fine-grained branch**: Uses a ResNet to capture local subtle distortions.
- Features from both branches are concatenated to form the final spatial feature $F_S \in \mathbb{R}^{F \times 5632}$.

### Inconsistency-Guided Temporal Module (IGTM)

**Stage 1: Consistency-aware Fusion**

Inspired by psychological findings on the limited capacity of visual working memory (VWM) — humans can retain approximately 3–7 objects — the authors design a Visual Memory Capacity Block governed by two principles:

1. The memory threshold is dynamically allocated within a defined range.
2. Higher temporal inconsistency leads to a lower memory threshold.

Temporal inconsistency complexity is computed by combining the standard deviation of optical flow magnitude and directional consistency (balanced by $\alpha = 0.5$). The adaptive threshold is defined as $T_D^i = \tau - \eta \times \frac{C_I^i - \min(C_I)}{\max(C_I) - \min(C_I)}$ (with $\tau=5, \eta=4$), partitioning the temporal sequence into segments based on cumulative complexity.

Temporal relationships within segments are modeled via a Graph Attention Network (GAT), where neighbor information is aggregated using attention coefficients $\alpha_{ij} = \text{softmax}(e_{ij})$, followed by GRU-based sequential encoding.

**Stage 2: Informative Filtering**

Self-attention is applied to select the Top-K most informative features, followed by another round of temporal modeling to regress the second-stage quality score. The final prediction is $S = \gamma S_1 + (1-\gamma) S_2$.

## Key Experimental Results

### Table 1: Comparison with SOTA Methods on Combined-VSR

| Method | Type | SRCC↑ | PLCC↑ | KRCC↑ | RMSE↓ |
|--------|------|-------|-------|-------|-------|
| PSNR | Handcrafted | 0.645 | 0.655 | 0.468 | 0.200 |
| SSIM | Handcrafted | 0.696 | 0.710 | 0.525 | 0.189 |
| VIF | Handcrafted | 0.746 | 0.753 | 0.579 | 0.165 |
| VSFA | Learning | 0.808 | 0.812 | 0.630 | 0.152 |
| GSTVQA | Learning | 0.828 | 0.825 | 0.645 | 0.147 |
| STI-VQA | Learning | 0.823 | 0.829 | 0.648 | 0.147 |
| FAST-VQA | Learning | 0.845 | 0.856 | 0.651 | 0.132 |
| MBVQA | Learning | 0.840 | 0.853 | 0.644 | 0.127 |
| VSR-QAD | SR-specific | 0.860 | 0.868 | 0.687 | 0.125 |
| ReLaX-VQA | Learning | 0.924 | 0.936 | 0.782 | 0.091 |
| **TIG-SVQA** | **SR-specific** | **0.939** | **0.942** | **0.794** | **0.083** |

TIG-SVQA outperforms the second-best method ReLaX-VQA by 1.5% in SRCC and surpasses the latest SR-specific method VSR-QAD by 7.9%.

### Table 2: Ablation Study (Combined-VSR)

| Variant | SRCC↑ | PLCC↑ | KRCC↑ | RMSE↓ |
|---------|-------|-------|-------|-------|
| w/o Guidance in IHSM | 0.891 | 0.909 | 0.716 | 0.116 |
| w/o Guidance in IGTM | 0.908 | 0.921 | 0.736 | 0.095 |
| w/o both Guidance | 0.878 | 0.901 | 0.707 | 0.107 |
| Coarse Branch only | 0.789 | 0.846 | 0.609 | 0.131 |
| Fine Branch only | 0.926 | 0.927 | 0.771 | 0.106 |
| w/o DW-SA-T block | 0.891 | 0.909 | 0.716 | 0.116 |
| **Full TIG-SVQA** | **0.939** | **0.942** | **0.794** | **0.083** |

### Table 3: Model Complexity Comparison

| Model | FLOPs (G) | Params (M) | SRCC↑ |
|-------|-----------|------------|-------|
| DISQ | 606.69 | 76.18 | 0.642 |
| FAST-VQA | 70.90 | 27.55 | 0.845 |
| STI-VQA | 103087.70 | 89.37 | 0.823 |
| MBVQA | 2149.90 | 93.23 | 0.840 |
| VSR-QAD | 678.95 | 23.74 | 0.860 |
| **TIG-SVQA** | **171.63** | **24.96** | **0.939** |

TIG-SVQA ranks second only to FAST-VQA in FLOPs, with parameter count comparable to VSR-QAD, yet achieves substantially superior performance.

### Adaptive Memory Threshold Ablation

| Threshold Setting | SRCC↑ | PLCC↑ | KRCC↑ | RMSE↓ |
|-------------------|-------|-------|-------|-------|
| Fixed = 1 | 0.931 | 0.935 | 0.772 | 0.095 |
| Fixed = 5 | 0.925 | 0.932 | 0.770 | 0.104 |
| Fixed = 15 | 0.895 | 0.905 | 0.727 | 0.104 |
| **Adaptive 1→5** | **0.939** | **0.942** | **0.794** | **0.083** |
| Adaptive 1→10 | 0.933 | 0.936 | 0.782 | 0.083 |

## Highlights & Insights

1. **First explicit quantification of temporal inconsistency for SR-VQA**: By quantifying temporal inconsistency via optical flow difference, the paper empirically demonstrates its high correlation with perceptual quality (SRCC=0.939), far exceeding raw motion signals (SRCC=0.885), offering a new perspective for SR video quality assessment.
2. **Dual-granularity spatial feature modeling**: The coarse-grained DW-SA Transformer captures large-scale inconsistencies while the fine-grained ResNet detects subtle distortions; these two branches are complementary — ablation results show SRCC of 0.789/0.926 for each branch alone, rising to 0.939 when fused.
3. **Visual working memory capacity mechanism**: Drawing on the cognitive science finding of limited VWM capacity (3–7 objects), adaptive segmentation outperforms fixed segmentation by 0.8–4.4% in SRCC, representing an elegant perceptually-aligned design.
4. **Exceptional efficiency-performance ratio**: With 171.63G FLOPs and 24.96M parameters at SRCC 0.939, compared to STI-VQA (103,087G FLOPs, 89.37M parameters, SRCC 0.823), the efficiency-performance ratio improves by orders of magnitude.

## Limitations & Future Work

1. **Requires reference video**: Temporal inconsistency information is computed from the optical flow difference between SR and reference videos, making TIG-SVQA a Full-Reference VQA method that cannot be deployed in no-reference scenarios.
2. **Limited dataset scale**: Evaluation is conducted only on SFD (1,193 videos) and MFD (1,067 videos), totaling 2,260 videos, without validation on larger or more diverse SR datasets.
3. **Limited SR method coverage**: Training and test data cover only 10 SR methods (5 single-frame + 5 multi-frame); generalization to distortions from next-generation SR approaches such as diffusion-based models remains unknown.
4. **Optical flow computation overhead**: Temporal inconsistency quantification relies on optical flow computation (e.g., RAFT), introducing additional preprocessing cost.

## Related Work & Insights

- **Conventional VQA methods** (VSFA, GSTVQA, FAST-VQA, etc.): Not designed for SR distortions; performance degrades when temporal inconsistency is amplified in SR scenarios. TIG-SVQA lifts SRCC from 0.808–0.856 to 0.939 through explicit temporal inconsistency modeling.
- **SR-specific methods** (VSR-QAD): Although designed for SR video, they still rely on indirect modeling such as temporal slicing, yielding SRCC=0.860 vs. TIG-SVQA's 0.939 — a significant gap.
- **ReLaX-VQA**: As the strongest non-SR-specific baseline (SRCC=0.924), TIG-SVQA still surpasses it by 1.5% with fewer parameters.
- **VWM in VQA** (VM-VQA): Considers only saliency-driven memory modeling, overlooking the critical cognitive constraint of memory capacity limits. TIG-SVQA's Visual Memory Capacity Block is the first to introduce this constraint into VQA.

The concept of temporal inconsistency as a guidance signal is extensible to video generation quality assessment (e.g., evaluating models such as Sora), where generated videos similarly suffer from severe temporal flickering. The VWM capacity mechanism may inspire adaptive segmentation strategies in other temporal sequence tasks. The effectiveness of dual-granularity spatial modeling (Transformer + CNN complementarity) in SR quality assessment may further benefit related tasks such as video inpainting and video enhancement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to explicitly quantify temporal inconsistency and leverage it for SR-VQA; the introduction of the VWM capacity mechanism is novel, though the overall paradigm remains a dual-branch Transformer+CNN feature fusion framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparison with 18 methods, comprehensive ablation, complexity analysis, and hyperparameter sensitivity study; dataset scale is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is rigorously argued (empirical analysis of motion vs. inconsistency), methodology is clearly described, and figures and tables are well-presented.
- **Value**: ⭐⭐⭐⭐ — Achieves significant improvements on the SR-VQA task; the temporal inconsistency guidance paradigm has broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] SpatioTemporal Difference Network for Video Depth Super-Resolution](spatiotemporal_difference_network_for_video_depth_super-resolution.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](../../CVPR2026/image_restoration/sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[AAAI 2026\] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios](hq-svc_towards_high-quality_zero-shot_singing_voice_conversion_in_low-resource_s.md)
- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](../../CVPR2026/image_restoration/raw-domain_degradation_models_for_realistic_smartphone_super-resolution.md)

<!-- RELATED:END -->
