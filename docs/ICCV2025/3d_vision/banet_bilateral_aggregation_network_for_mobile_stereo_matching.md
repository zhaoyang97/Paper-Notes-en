---
title: >-
  [Paper Note] BANet: Bilateral Aggregation Network for Mobile Stereo Matching
description: >-
  [ICCV 2025][3D Vision][stereo matching] This paper proposes BANet, a Bilateral Aggregation Network that decomposes the cost volume into a high-frequency detail volume and a low-frequency smooth volume via spatial attention and aggregates them separately. Using only 2D convolutions, BANet runs in real time on mobile devices while substantially outperforming MobileStereoNet-2D (35.3% accuracy improvement on KITTI 2015). Its 3D variant achieves the highest accuracy among real-time methods on GPU.
tags:
  - ICCV 2025
  - 3D Vision
  - stereo matching
  - mobile deployment
  - bilateral aggregation
  - 2D convolution
  - spatial attention
date: 2026-05-08
content_hash: 762b848fcdb4459f
---

# BANet: Bilateral Aggregation Network for Mobile Stereo Matching

**Conference**: ICCV 2025
**arXiv**: [2503.03259](https://arxiv.org/abs/2503.03259)
**Code**: [GitHub](https://github.com/gangweix/BANet)
**Area**: 3D Vision / Stereo Matching
**Keywords**: stereo matching, mobile deployment, bilateral aggregation, 2D convolution, spatial attention

## TL;DR

This paper proposes BANet, a Bilateral Aggregation Network that decomposes the cost volume into a high-frequency detail volume and a low-frequency smooth volume via spatial attention and aggregates them separately. Using only 2D convolutions, BANet runs in real time on mobile devices while substantially outperforming MobileStereoNet-2D (35.3% accuracy improvement on KITTI 2015). Its 3D variant achieves the highest accuracy among real-time methods on GPU.

## Background & Motivation

Stereo matching is critical for drone navigation, mobile photography, robotic surgery, and related applications. State-of-the-art methods (e.g., IGEV, ACVNet) rely heavily on 3D convolutions for cost volume aggregation, achieving high accuracy but remaining infeasible for deployment on mobile SoCs such as Qualcomm Snapdragon. Prior efforts to reduce computational cost fall into three categories:

- **Downsampled / sparse cost volumes**: Methods such as StereoNet and Fast-ACVNet construct low-resolution or sparse cost volumes but still depend on 3D convolutions, making them mobile-unfriendly.
- **Complex 2D alternative operations**: AANet uses deformable convolutions and HITNet uses iterative warping to replace 3D convolutions and alleviate edge blurring, but these operations incur high deployment costs on mobile hardware.
- **Pure 2D convolutions**: MobileStereoNet-2D adopts purely 2D convolutions and is mobile-friendly, yet suffers significant accuracy degradation—edge blurring, loss of fine detail, and mismatching in textureless regions.

**Key Challenge**: How can high accuracy be maintained using only mobile-friendly operations? The authors observe that a scene simultaneously contains high-frequency detail regions and low-frequency smooth/textureless regions, and that a single 2D aggregation network struggles to handle both. This motivates a divide-and-conquer bilateral aggregation strategy.

## Method

### Overall Architecture

BANet consists of four stages: (1) feature extraction with MobileNetV2, (2) correlation volume construction, (3) bilateral aggregation (the core contribution), and (4) disparity prediction. The entire pipeline uses only 2D convolutions and standard mobile-friendly operations, enabling direct deployment on Qualcomm Snapdragon SoCs.

### Key Designs

1. **Bilateral Aggregation**:
   The core idea is divide-and-conquer. A spatial attention map $\mathbf{A}$ decomposes the full correlation volume $\mathbf{C}_{cor}$ into two components:
   $$\mathbf{C}_d = \mathbf{A} \odot \mathbf{C}_{cor}, \quad \mathbf{C}_s = (1 - \mathbf{A}) \odot \mathbf{C}_{cor}$$
   where $\mathbf{C}_d$ focuses on high-frequency details and edges, and $\mathbf{C}_s$ focuses on low-frequency smooth and textureless regions. Two independent aggregation branches $\mathbf{G}_d$ and $\mathbf{G}_s$ process each component separately, and their outputs are fused by weighted summation:
   $$\mathbf{C}_{agg} = \mathbf{A} \odot \mathbf{C}_d' + (1 - \mathbf{A}) \odot \mathbf{C}_s'$$
   Both branches share the same architecture but do not share weights. Each branch consists of inverted residual blocks from MobileNetV2: 4 blocks at 1/4 resolution, 6 blocks at 1/8 resolution, and 8 blocks at 1/16 resolution, with channel counts of 32/64/128 respectively.

2. **Scale-aware Spatial Attention**:
   Accurately distinguishing high-frequency from low-frequency regions is essential for bilateral aggregation. The authors observe that **fine-scale features capture more high-frequency details, while coarse-scale features encode more low-frequency smooth information**. Accordingly, multi-scale features are fused to generate the attention map:
   $$\mathbf{S} = \text{Concat}[\text{Conv}(\mathbf{F}_{l,16}^{up}), \text{Conv}(\mathbf{F}_{l,8}^{up}), \text{Conv}(\mathbf{F}_{l,4})]$$
   $$\mathbf{A} = \sigma(\text{Conv}(\mathbf{S}))$$
   Features from three scales (1/16, 1/8, 1/4 resolution) are upsampled to 1/4 resolution, concatenated, and passed through a convolution followed by sigmoid to produce the spatial attention map.

3. **3D Extension (BANet-3D)**:
   The bilateral aggregation concept extends seamlessly to 3D convolutions. The 3D aggregation network comprises 3 downsampling blocks (2 layers of $3\times3\times3$ convolutions each) and 3 upsampling blocks ($4\times4\times4$ transposed convolution followed by 2 layers of $3\times3\times3$ convolutions). BANet-3D achieves the highest accuracy among real-time methods on high-end GPUs, though it is not suitable for mobile deployment.

### Loss & Training

- Loss function: Smooth L1 Loss
  $$\mathcal{L} = \lambda_0 \text{SmoothL1}(\mathbf{d}_0 - \mathbf{d}_{gt}) + \lambda_1 \text{SmoothL1}(\mathbf{d}_1 - \mathbf{d}_{gt})$$
  where $\lambda_0=0.3$, $\lambda_1=1.0$; $\mathbf{d}_0$ denotes the 1/4-resolution disparity and $\mathbf{d}_1$ denotes the full-resolution disparity (upsampled via superpixel weighting).
- Pre-trained on Scene Flow for 200K steps (batch size 16), then fine-tuned on the mixed KITTI 2012+2015 set for 50K steps.
- Training crop size: $256\times512$; $D_{max}=192$.
- Optimizer: AdamW with a one-cycle learning rate schedule; maximum learning rate $8\times10^{-4}$.
- Hardware: RTX 3090 GPU.

## Key Experimental Results

### Main Results

| Method | KITTI 2015 D1-all (%) | KITTI 2012 3-noc (%) | Scene Flow EPE (px) | MACs (G) |
|---|---|---|---|---|
| MobileStereoNet-2D | 2.83 | — | 1.11 | 127→136 |
| Fast-ACVNet+ | 2.01 | 1.45 | 0.59 | 85→93 |
| HITNet | 1.98 | 1.41 | — | 47 |
| CoEx | 2.13 | 1.55 | 0.67 | 49→53 |
| **BANet-2D (Ours)** | **1.83** | **1.38** | **0.57** | **36→39** |
| **BANet-3D (Ours)** | **1.77** | **1.27** | **0.51** | **78→85** |

### Ablation Study

| Aggregation Type | Bilateral Agg. | Scale Attention | Scene Flow EPE | Bad 3.0 (%) | MACs (G) |
|---|---|---|---|---|---|
| 2D Baseline | ✗ | ✗ | 0.63 | 2.75 | 29 |
| 2D + BA | ✓ | ✗ | 0.59 | 2.57 | 38 |
| 2D + BA + SSA | ✓ | ✓ | 0.57 | 2.49 | 39 |
| 3D Baseline | ✗ | ✗ | 0.56 | 2.43 | 57 |
| 3D + BA | ✓ | ✗ | 0.53 | 2.27 | 80 |
| 3D + BA + SSA | ✓ | ✓ | 0.51 | 2.21 | 85 |

| Method | KITTI 2015 Foreground D1-fg (%) | Gain |
|---|---|---|
| 2D w/o BA | 3.67 | — |
| BANet-2D | 3.03 | **17%↑** |
| 3D w/o BA | 3.87 | — |
| BANet-3D | 3.02 | **22%↑** |

### Key Findings

- **Most significant improvement in foreground regions**: D1-fg on the KITTI test set improves by 17–22%, as foreground objects contain more high-frequency edges and fine details.
- **Plug-and-play compatibility**: Bilateral aggregation can be integrated into PSMNet (29% EPE improvement), GwcNet (12%), and Fast-ACVNet+ (10%).
- **Mobile latency**: On a Qualcomm Snapdragon 8 Gen 3 with $512\times512$ input, inference takes only 45 ms (feature extraction 16 ms + correlation volume construction 6.5 ms + bilateral aggregation 22.5 ms), less than one-third the latency of MobileStereoNet-2D.
- **Lowest computational cost**: BANet-2D achieves the lowest MACs (36G) among all compared methods while attaining the highest accuracy.

## Highlights & Insights

- **Elegant divide-and-conquer strategy**: Rather than designing a more powerful unified aggregation network, BANet partitions the problem domain and solves each part separately, achieving the effect of complex operations with simple ones.
- **Multi-scale perception → frequency separation**: The method exploits the natural correspondence between feature scale and spatial frequency to generate a decomposition attention map—a notably clean design choice.
- **High practical deployment value**: Latency is validated on real mobile hardware; 45 ms at $512\times512$ fully satisfies real-time requirements.

## Limitations & Future Work

- The bilateral decomposition assumes that scenes can be clearly partitioned into high-frequency and low-frequency regions; more nuanced decomposition strategies may be needed for challenging cases such as semi-transparent or reflective surfaces.
- Both aggregation branches share the same architecture with only independent weights; whether distinct optimal architectures should be designed for the detail and smooth branches remains unexplored.
- Transfer to related tasks such as multi-view stereo and optical flow estimation is not investigated (noted by the authors as future work).
- The superpixel upsampling strategy has limited capacity to recover extremely fine structures.

## Related Work & Insights

- **vs. AANet**: AANet performs adaptive aggregation via deformable convolutions, which incurs high computational overhead and is mobile-unfriendly; BANet separates the cost volume via attention and aggregates each part independently using standard 2D convolutions throughout.
- **vs. HITNet**: HITNet avoids an explicit cost volume entirely by recovering disparity through iterative warping, but iterative operations are similarly unsuitable for mobile deployment.
- The concept of bilateral filtering and bilateral networks has a long history in image processing; applying this paradigm to cost volume aggregation represents a valuable and well-motivated contribution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The bilateral aggregation idea is concise and effective, with pioneering significance for mobile stereo matching deployment.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers synthetic and real-world data, 2D and 3D variants, mobile SoC latency measurements, and plug-and-play validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; visualizations are persuasive.
- **Value**: ⭐⭐⭐⭐⭐ — A genuinely engineering-oriented contribution and a strong solution for real-time, high-accuracy stereo matching on mobile devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts](learning_robust_stereo_matching_in_the_wild_with_selective_mixture-of-experts.md)
- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [\[ICCV 2025\] Stereo Any Video: Temporally Consistent Stereo Matching](stereo_any_video_temporally_consistent_stereo_matching.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)

</div>

<!-- RELATED:END -->
