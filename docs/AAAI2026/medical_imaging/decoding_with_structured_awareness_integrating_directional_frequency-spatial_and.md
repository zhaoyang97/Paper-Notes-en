---
title: >-
  [Paper Note] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation
description: >-
  [AAAI 2026][Medical Imaging][Medical image segmentation] This paper proposes a novel decoder framework for medical image segmentation comprising three modules: Adaptive Cross-Fusion Attention (ACFA) for directional awareness, Triple Feature Fusion Attention (TFFA) for spatial-frequency-wavelet fusion, and Structural-aware Multi-scale Masking Module (SMMM), achieving state-of-the-art performance across multiple benchmark datasets.
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Medical image segmentation"
  - "decoder design"
  - "frequency-spatial fusion"
  - "directional-aware attention"
  - "multi-scale feature fusion"
date: 2026-05-08
content_hash: b72df9d2619a2268
---

# Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation

**Conference**: AAAI 2026
**arXiv**: [2512.05494](https://arxiv.org/abs/2512.05494)  
**Code**: N/A  
**Area**: Medical Imaging
**Keywords**: Medical image segmentation, decoder design, frequency-spatial fusion, directional-aware attention, multi-scale feature fusion

## TL;DR

This paper proposes a novel decoder framework for medical image segmentation comprising three modules: Adaptive Cross-Fusion Attention (ACFA) for directional awareness, Triple Feature Fusion Attention (TFFA) for spatial-frequency-wavelet fusion, and Structural-aware Multi-scale Masking Module (SMMM), achieving state-of-the-art performance across multiple benchmark datasets.

## Background & Motivation

- Accurate delineation of organs, tumors, and lesions in medical image segmentation is critical for surgical planning and radiotherapy dose design.
- **Limitations of Transformer decoders**:
    - Insufficient edge detail capture: self-attention excels at global dependencies but is weak in local texture modeling.
    - Limited local texture recognition: fixed receptive fields struggle with ambiguous boundaries.
    - Inadequate spatial continuity modeling: simple additive skip connections lead to spatial detail loss and redundant information.
- **Issue with U-Net skip connections**: conventional skip connections rely on simple addition, failing to balance global and local features.
- CNN fixed receptive fields limit long-range dependency modeling; ViTs excel globally but are weak at short-range dependencies.
- A decoder framework is needed that enhances edge and structural detail representation while maintaining global perception.

## Method

### Overall Architecture

The encoder uses PVTv2-b2 (ImageNet pre-trained), and the decoder consists of three core modules:

1. **ACFA** (Adaptive Cross-Fusion Attention): directional awareness module
2. **TFFA** (Triple Feature Fusion Attention): frequency-spatial fusion module
3. **SMMM** (Structural-aware Multi-scale Masking Module): skip connection optimization module

### Key Designs

**Module 1: ACFA — Adaptive Cross-Fusion Attention**

Enhances model responsiveness to critical regions and structural directional modeling:

- For input feature map $X \in \mathbb{R}^{B \times C \times H \times W}$, channel gating and spatial gating are applied:
    - Channel gating: $\hat{X}_{l-1}^{CG} = X \odot \sigma(CG_{avg}(X) + CG_{max}(X))$
    - Spatial gating: $\hat{X}_{l-1}^{SG} = X \odot \sigma(f_{7 \times 7}^{Conv}(SG(X)))$

- Spatially gated features are split into 4 groups along the channel dimension.
- **Three directional branches** with learnable directional parameters:
    - Planar direction: $Tensor^{HW} \in [1, C/4, H, W]$
    - Vertical direction: $Tensor^{H} \in [1, C/4, H, 1]$
    - Horizontal direction: $Tensor^{W} \in [1, C/4, 1, W]$
    - Each direction employs depth-wise separable convolutions to extract key responses.

- **Fourth branch**: standard convolution captures general contextual information, complementing details potentially missed by the directional branches.
- Four-branch features are concatenated and fused via LayerNorm and convolution.

**Design Motivation**: Structural directionality of organs and lesions in medical images (e.g., vessel orientation) is important; the module learns the most data-appropriate directional attention patterns end-to-end.

**Module 2: TFFA — Triple Feature Fusion Attention**

Fuses spatial-domain, Fourier-domain, and wavelet-domain features for joint frequency-spatial representation:

- **Wavelet branch**: employs DoG (Difference of Gaussians) and Mexican Hat wavelets.
    - DoG highlights regions of significant gray-level change, enhancing edge and contour perception:
    $$\psi_{a,b}^{DoG}(x) = -\frac{1}{\sqrt{a}} \frac{x-b}{a} e^{-\frac{(x-b)^2}{2a^2}}$$
    - Mexican Hat detects edge zero-crossings via second-order derivatives while suppressing noise:
    $$\psi_{a,b}^{MH}(x) = \frac{2}{\sqrt{3a}\pi^{1/4}} (1 - (\frac{x-b}{a})^2) e^{-\frac{(x-b)^2}{2a^2}}$$
    - Scale parameter $a$ and shift parameter $b$ are both learnable.

- **Fourier branch**: transforms images from the spatial domain to the frequency domain.
    - High-frequency components → edges and textures; low-frequency components → contours and background.
    - Learnable weight matrices modulate frequency-domain features.
    - Compensates for the deficiency of convolutional models in perceiving large-scale structures.

- **Spatial branch**: point-wise convolution extracts spatial features.

- **Attention-gated fusion**: three branch outputs are adaptively fused via dynamic attention weights, avoiding the over-smoothing of conventional fusion strategies.

**Module 3: SMMM — Structural-aware Multi-scale Masking Module**

Optimizes skip connections between encoder and decoder:

- Encoder and decoder features are each activated via point-wise convolutions to highlight spatial cues.
- **Multi-scale perception**:
    - Dual-path extraction using 3×3 and 5×5 depth-wise separable convolutions.
    - Two-stage channel splitting with ReLU activation to enlarge the receptive field.
    - Two-level feature concatenation followed by 3×3 and 5×5 convolutions for further fusion.

- **Spatial saliency mask**:
    - Three distinct channel-gated filters identify the most discriminative spatial regions.
    - Softmax weighting emphasizes high-response areas.
    - Effectively handles ambiguous lesions and indistinct contours.

- Filtered features are summed and processed by a dilated convolution with dilation=2 to expand the receptive field.
- Final normalization and point-wise convolution perform channel alignment.

### Loss & Training

- Encoder: PVTv2-b2, ImageNet pre-trained.
- Optimizer: AdamW, lr = 1e-4.
- Batch size = 12; masks normalized to [0, 1]; no data augmentation.
- Training epochs: ISIC 2017/2018 → 200; Synapse → 300; ACDC → 400.
- Hardware: NVIDIA A100 GPU (40 GB).

## Key Experimental Results

### Main Results (Synapse Multi-organ Segmentation)

| Method | DSC↑ | HD95↓ | Spl | RKid | LKid | Pan |
|--------|------|-------|-----|------|------|-----|
| TransUNet | 77.49 | 31.69 | 85.08 | 77.02 | 81.87 | 55.86 |
| Swin-UNet | 79.13 | 21.55 | 90.66 | 79.61 | 83.28 | 56.58 |
| EMCAD | 83.63 | 15.68 | 92.17 | 84.10 | 88.08 | 68.51 |
| AD-LA Former | 83.48 | 21.31 | 88.72 | 70.82 | 86.50 | 84.69 |
| **Ours** | **83.92** | 18.91 | **92.46** | **86.47** | **89.26** | 69.95 |

**ISIC 2017 Skin Lesion Segmentation**:

| Method | DSC↑ | SE | SP | ACC |
|--------|------|----|----|-----|
| LKA | 90.99 | 90.55 | 98.49 | 96.98 |
| EMCAD | 90.06 | 93.70 | 96.81 | 96.55 |
| **Ours** | **91.40** | 92.75 | 97.78 | **97.26** |

**ACDC Cardiac Segmentation**:

| Method | DSC↑ | RV | Myo | LV |
|--------|------|----|----|-----|
| DMSA-UNet | 92.28 | 90.32 | 90.49 | 96.02 |
| EMCAD | 92.12 | 90.65 | 89.68 | 96.02 |
| **Ours** | **92.75** | **91.18** | 90.40 | **96.67** |

### Ablation Study

**Module Ablation on Synapse**:

| Configuration | DSC↑ | HD95↓ |
|---------------|------|-------|
| Baseline | 81.35 | 20.42 |
| + ACFA | 82.04 | 21.46 |
| + ACFA + TFFA | 83.23 | 16.72 |
| + ACFA + TFFA + SMMM | **83.92** | 18.91 |

**Module Ablation on ISIC 2017**:

| Configuration | DSC↑ | SE | ACC |
|---------------|------|----|----|
| Baseline | 85.95 | 83.68 | 96.05 |
| + ACFA | 87.82 | 85.12 | 95.92 |
| + ACFA + TFFA | 89.15 | 89.83 | 96.85 |
| + ACFA + TFFA + SMMM | **91.40** | **92.75** | **97.26** |

**TFFA Internal Ablation (ISIC 2018)**:

| Fourier | Mexican Hat | DoG | DSC↑ | SE |
|---------|-------------|-----|------|----|
| ✗ | ✗ | ✗ | 90.32 | 89.31 |
| ✓ | ✗ | ✗ | 90.48 | 91.43 |
| ✓ | ✓ | ✗ | 90.57 | 92.63 |
| ✓ | ✓ | ✓ | **90.71** | **93.34** |

**Computational Cost**: Full model — 42.52M parameters, 18.29 GMac (Baseline: 25.07M / 11.85 GMac).

### Key Findings

- ACFA improves DSC by 0.7% on Synapse with only 5.6M additional parameters, validating the effectiveness of directional awareness.
- TFFA contributes the largest gain: DSC from 82.04 → 83.23 (Synapse); SE from 85.12 → 89.83 (ISIC 2017).
- SMMM is more effective on detail-rich datasets (ISIC 2017 DSC +2.25).
- The synergy of all three modules substantially exceeds the sum of their individual contributions.

## Highlights & Insights

1. **Frequency-domain fusion design rationale**: DoG performs band-pass filtering to enhance textures; Mexican Hat performs second-order derivative edge detection; Fourier captures global dependencies — the three are mutually complementary.
2. **Learnability of directional awareness**: all three directional parameters are learnable, eliminating the need for hand-crafted directional filters.
3. **Deep-level skip connection optimization**: SMMM replaces simple addition with spatial saliency masking, reducing redundant information propagation.
4. **Cross-dataset validation**: state-of-the-art or near state-of-the-art performance is achieved on abdominal multi-organ, skin lesion, and cardiac segmentation tasks.

## Limitations & Future Work

- HD95 on Synapse is not optimal (18.91 vs. HiFormer-B's 14.70), indicating room for further improvement in boundary precision.
- Parameter count increases from 25M to 42.5M (+70%), and computation from 11.85 to 18.29 GMac (+54%), incurring a non-trivial efficiency cost.
- Validation is limited to 2D segmentation; extension to 3D medical image segmentation remains unexplored.
- Gallbladder segmentation performance lags behind AD-LA Former (67.51 vs. 83.30), indicating small organ segmentation remains a weakness.
- The initialization and convergence behavior of the wavelet scale parameter $a$ and shift parameter $b$ are not thoroughly analyzed.

## Related Work & Insights

- Compared to EMCAD (multi-scale attention decoder), this work provides a complementary perspective on multi-scale information via frequency-domain fusion.
- The value of frequency-domain analysis in medical imaging is reaffirmed: Fourier + wavelet combinations outperform single-transform approaches.
- The saliency masking concept in SMMM can serve as an alternative to conventional attention gating, more directly addressing feature redundancy.
- The directional awareness design is generalizable to tasks with strong directional structure, such as vessel segmentation and crack detection.

## Rating

- Novelty: ⭐⭐⭐ — The module combination is relatively novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, comprehensive ablation, and computational cost analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete mathematical formulations.
- Value: ⭐⭐⭐⭐ — Provides a practical decoder design paradigm for medical image segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration](../../CVPR2025/medical_imaging/sacb-net_spatial-awareness_convolutions_for_medical_image_registration.md)
- [\[AAAI 2026\] FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing](fia-edit_frequency-interactive_attention_for_efficient_and_high-fidelity_inversi.md)
- [\[AAAI 2026\] FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation](funkan_functional_kolmogorov-arnold_network_for_medical_image_enhancement_and_se.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)
- [\[ICLR 2026\] HFSTI-Net: Hierarchical Frequency-spatial-temporal Interactions for Video Polyp Segmentation](../../ICLR2026/medical_imaging/hfsti-net_hierarchical_frequency-spatial-temporal_interactions_for_video_polyp_s.md)

</div>

<!-- RELATED:END -->
