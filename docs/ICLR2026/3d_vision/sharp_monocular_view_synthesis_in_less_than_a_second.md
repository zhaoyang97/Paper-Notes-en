---
title: >-
  [Paper Note] Sharp Monocular View Synthesis in Less Than a Second
description: >-
  [ICLR 2026][3D Vision][view synthesis] SHARP generates approximately 1.2 million 3D Gaussians from a single image via a single feedforward pass…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "view synthesis"
  - "3D Gaussian splatting"
  - "monocular depth"
  - "real-time rendering"
  - "feedforward"
date: 2026-05-08
content_hash: 3044c8987737131c
---

# Sharp Monocular View Synthesis in Less Than a Second

**Conference**: ICLR 2026
**arXiv**: [2512.10685](https://arxiv.org/abs/2512.10685)
**Code**: [github.com/apple/ml-sharp](https://github.com/apple/ml-sharp)
**Area**: 3D Vision
**Keywords**: view synthesis, 3D Gaussian splatting, monocular depth, real-time rendering, feedforward

## TL;DR

SHARP generates approximately 1.2 million 3D Gaussians from a single image via a single feedforward pass, completing inference in under one second on an A100 GPU with rendering speeds exceeding 100 FPS. It achieves state-of-the-art zero-shot generalization across 6 datasets, reducing LPIPS by 25–34% and synthesis time by three orders of magnitude compared to the strongest prior method.

## Background & Motivation

**Background**: Novel view synthesis has evolved from multi-image optimization methods (NeRF, 3DGS) to single-image feedforward approaches (Splatter Image, Flash3D) and diffusion-based methods (Gen3C, ViewCrafter, SVC). The former are fast but limited in quality; the latter achieve higher fidelity but require minutes of processing.

**Limitations of Prior Work**: (1) Feedforward methods (e.g., Flash3D) exhibit significantly lower visual fidelity than diffusion-based methods; (2) diffusion-based methods (e.g., Gen3C requiring ~15 minutes) are too slow for interactive browsing; (3) at close-range viewpoints, diffusion-based outputs are often less sharp than the input photograph; (4) most methods lack metric scale, preventing coupling with physical devices.

**Key Challenge**: How can high-fidelity, photorealistic rendering at close viewpoints be achieved while maintaining sub-second interactive speed?

**Goal**: An end-to-end regression network is trained on a Depth Pro encoder backbone to predict dual-layer depth maps and fine-grained residuals for all Gaussian attributes. A learned depth adjustment module is introduced to resolve monocular depth ambiguity, a carefully designed loss configuration suppresses artifacts, and self-supervised fine-tuning is applied to adapt to real-world images.

## Method

### Overall Architecture

Given input $\mathbf{I} \in \mathbb{R}^{3 \times 1536 \times 1536}$, the network outputs $\mathbf{G} \in \mathbb{R}^{14 \times 2 \times 768 \times 768}$ (approximately 1.2 million 3D Gaussians with 14 attributes: 3 position + 3 scale + 4 rotation + 3 color + 1 opacity). The network consists of four learnable modules: a pretrained encoder, a depth decoder, a depth adjustment module, and a Gaussian decoder. The total parameter count is 702M (340M trainable), with inference time under 1 second.

### Key Design 1: End-to-End Depth-to-Gaussian Regression Architecture

**Encoder**: A dual ViT backbone based on Depth Pro. The low-resolution encoder (326M parameters) is unfrozen during training to adapt to the view synthesis task, while the patch encoder remains frozen.

**Dual-Layer Depth Decoder**: Based on the DPT architecture (~20M parameters), the final convolutional layer is duplicated to produce two depth channels—the first representing visible surfaces and the second representing occluded regions and view-dependent effects.

**Gaussian Initializer**: Initial 3D coordinates are obtained by unprojecting the depth map as $\mu(i,j) = [i \cdot \bar{D}'(i,j),\, j \cdot \bar{D}'(i,j),\, \bar{D}'(i,j)]^T$, colors are taken from input pixel values, and scale is proportional to depth via $s = s_0 \cdot \bar{D}'$. A key detail: **the source view intrinsic matrix is not used**, allowing the network to reason in normalized space.

**Gaussian Decoder**: Also based on DPT (~7.8M parameters trained from scratch), it predicts fine-grained residuals for all attributes. Attribute-specific activation functions are combined as:

$$\mathbf{G}_{\text{attr}} = \gamma_{\text{attr}}\Big(\gamma_{\text{attr}}^{-1}(\mathbf{G}_{0,\text{attr}}) + \eta_{\text{attr}} \Delta\mathbf{G}_{\text{attr}}\Big)$$

### Key Design 2: Learned Depth Adjustment Module

Fundamental ambiguities in monocular depth estimation—particularly severe for transparent and reflective surfaces—introduce artifacts in view synthesis. Inspired by Conditional VAEs, a small U-Net (2M parameters) is introduced that takes both the predicted inverse depth $\hat{D}^{-1}$ and the ground-truth inverse depth $D^{-1}$ as input and outputs a scaling map $\mathbf{S}$:

$$\bar{D} = \mathbf{S}(\hat{D}, D) \odot \hat{D}$$

At inference time, the identity function replaces this module (no ground-truth depth required). An MAE regularizer $\mathcal{L}_{\text{scale}} = \mathbb{E}[|\mathbf{S}(p) - 1|]$ and multi-scale TV regularization serve as an information bottleneck, encouraging the network to learn the most compact representation for resolving ambiguity.

### Key Design 3: Two-Stage Training and Loss Design

**Stage 1 — Synthetic Data Training**: Approximately 700K procedurally generated scenes (~8M images) with perfect image and depth ground truth are used.

**Stage 2 — Self-Supervised Fine-Tuning (SSFT)**: The already-trained model generates pseudo novel views of real images; these pseudo views serve as inputs while the original real images serve as supervision targets (view-swapping strategy), requiring no stereo pairs.

**Loss Configuration**:

- **Rendering losses**: L1 color $\mathcal{L}_{\text{color}}$, perceptual loss $\mathcal{L}_{\text{percep}}$ (including a Gram matrix term for sharpness), and BCE alpha loss $\mathcal{L}_{\text{alpha}}$
- **Depth loss**: Inverse depth L1 $\mathcal{L}_{\text{depth}}$ applied to the first layer only
- **Regularizers**: Second-layer depth TV $\mathcal{L}_{\text{tv}}$, disparity gradient floater suppression $\mathcal{L}_{\text{grad}}$, offset constraint $\mathcal{L}_{\text{delta}}$, and Gaussian screen projection variance $\mathcal{L}_{\text{splat}}$

An engineering optimization for the perceptual loss: a Computation Graph Surgery mechanism is proposed, which precomputes gradients and releases the ResNet computation graph to resolve out-of-memory issues on 40GB GPUs.

## Key Experimental Results

### Main Results

Zero-shot evaluation on 6 datasets unseen during training (lower is better):

| Method | Middlebury DISTS | Middlebury LPIPS | ScanNet++ DISTS | ScanNet++ LPIPS | WildRGBD DISTS | WildRGBD LPIPS |
|--------|-----------------|-----------------|-----------------|-----------------|---------------|---------------|
| Flash3D | 0.359 | 0.581 | 0.374 | 0.572 | 0.159 | 0.345 |
| TMPI | 0.158 | 0.436 | 0.128 | 0.309 | 0.114 | 0.327 |
| LVSM | 0.274 | 0.555 | 0.145 | 0.302 | 0.095 | 0.257 |
| Gen3C | 0.164 | 0.545 | 0.090 | 0.227 | 0.106 | 0.285 |
| **SHARP** | **0.097** | **0.358** | **0.071** | **0.154** | **0.069** | **0.190** |

Inference time comparison: SHARP 0.91s (+ rendering ~5ms/frame); Gen3C ~830s; ViewCrafter ~120s.

### Ablation Study

**Loss component ablation** (ScanNet++/Tanks and Temples DISTS):

| Configuration | ScanNet++ DISTS | T&T DISTS |
|---------------|-----------------|-----------|
| color + alpha only | 0.229 | 0.301 |
| + depth | 0.162 | 0.239 |
| + perceptual | 0.063 | 0.126 |
| + regularizers | 0.064 | 0.126 |

**Depth adjustment ablation**:

| Depth Adjustment | ScanNet++ DISTS | ScanNet++ LPIPS |
|------------------|-----------------|-----------------|
| Without | 0.077 | 0.154 |
| With | 0.064 | 0.147 |

**Gaussian count ablation**:

| Count | ScanNet++ DISTS | ScanNet++ LPIPS |
|-------|-----------------|-----------------|
| 2×192² (~74K) | 0.110 | 0.199 |
| 2×384² (~295K) | 0.077 | 0.160 |
| 2×768² (~1.2M) | 0.064 | 0.147 |

### Key Findings

1. **LPIPS reduced by 25–34% compared to Gen3C**: SHARP achieves the best performance across all 6 datasets, while Gen3C requires approximately 900× more inference time.
2. **Perceptual loss contributes most**: DISTS drops from 0.162 to 0.063 on ScanNet++; the Gram matrix term is the key driver of sharpness improvement.
3. **Depth adjustment improves detail clarity**: Particularly on transparent and reflective surfaces, eliminating blur caused by depth ambiguity.
4. **1.2M Gaussians substantially outperform 74K**: Performance improves consistently with increasing Gaussian count.
5. **SSFT yields clear qualitative improvements**: Although metric changes are modest, visual sharpness is significantly enhanced.

## Highlights & Insights

- **The triumph of pure regression**: This work demonstrates that, for close-range view synthesis, a carefully designed feedforward regression approach can surpass diffusion-based methods that require three orders of magnitude more computation.
- **Key engineering insights**: Prediction in normalized space (without source view intrinsics), Computation Graph Surgery to resolve perceptual loss OOM issues, and the view-swapping strategy for self-supervised learning without stereo pairs.
- **Metric scale support**: The output 3D representation carries absolute scale, enabling direct coupling with the physical motion of AR/VR headsets.
- **Information bottleneck role of depth adjustment**: Regularization forces the network to learn only the minimal necessary information for resolving depth ambiguity, allowing complete removal at inference time.

## Limitations & Future Work

- Designed specifically for close-range viewpoints (~0.5m displacement); quality degrades at large displacements—complementary integration with diffusion models may be necessary.
- Spherical harmonics (SH) are not used, precluding modeling of view-dependent effects (reflections, specular highlights).
- The depth model may fail in extreme scenarios such as macro photography, night-sky imaging, and complex water surface reflections.
- Directly applying the synthetic training data to Flash3D yields no improvement, suggesting that architectural design rather than data is the primary factor.
- A unified framework for video or multi-image input has not been explored.

## Related Work & Insights

- **3D Gaussian Splatting (Kerbl et al., 2023)**: Provides the 3D representation adopted by SHARP, though the original method requires multi-image optimization.
- **Depth Pro (Bochkovskii et al., 2025)**: Serves as the SHARP backbone, providing metric-scale monocular depth estimation capability.
- **Flash3D (Szymanowicz et al., 2025)**: A predecessor in the same paradigm that introduces pretrained depth networks for scene-level generalization, but with insufficient fidelity.
- **Gen3C (Ren et al., 2025)**: The strongest diffusion baseline; less sharp than SHARP at close viewpoints, but with stronger large-displacement synthesis capability.

## Rating

⭐⭐⭐⭐⭐

This work achieves Pareto-optimal quality-speed trade-offs for single-image novel view synthesis, attaining comprehensive state-of-the-art zero-shot performance across 6 datasets at sub-second inference speed. Every module in the end-to-end architecture—dual-layer depth, depth adjustment, and Gaussian refinement—is motivated clearly and supported by ablation studies. Engineering contributions (Computation Graph Surgery, SSFT) demonstrate substantial systems expertise. Produced by Apple with open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](../../CVPR2026/3d_vision/movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](dynamic_novel_view_synthesis_in_high_dynamic_range.md)
- [\[ICLR 2026\] 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)
- [\[CVPR 2026\] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](../../CVPR2026/3d_vision/dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](../../NeurIPS2025/3d_vision/hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)

</div>

<!-- RELATED:END -->
