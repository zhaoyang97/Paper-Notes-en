---
title: >-
  [Paper Note] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation
description: >-
  [ECCV 2024][3D Vision][Monocular Depth Estimation] This paper introduces the diffusion denoising process to monocular depth estimation for the first time. By performing visually conditioned iterative denoising in the latent depth space, and proposing a self-diffusion mechanism to resolve the mode collapse issue caused by sparse Ground Truth (GT) depths, it achieves SOTA performance on KITTI and NYU-Depth-V2.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Diffusion Models"
  - "Denoising Process"
  - "Depth Refinement"
  - "Self-Diffusion"
date: 2026-05-08
content_hash: b198bce9fb29ab2c
---

# DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation

**Conference**: ECCV 2024  
**arXiv**: [2303.05021](https://arxiv.org/abs/2303.05021)  
**Code**: [https://github.com/duanyiqun/DiffusionDepth](https://github.com/duanyiqun/DiffusionDepth)  
**Area**: 3D Vision  
**Keywords**: Monocular Depth Estimation, Diffusion Models, Denoising Process, Depth Refinement, Self-Diffusion

## TL;DR

This paper introduces the diffusion denoising process to monocular depth estimation for the first time. By performing visually conditioned iterative denoising in the latent depth space, and proposing a self-diffusion mechanism to resolve the mode collapse issue caused by sparse Ground Truth (GT) depths, it achieves SOTA performance on KITTI and NYU-Depth-V2.

## Background & Motivation

**Background**: Monocular depth estimation is a fundamental visual task that predicts pixel-wise depth from a single 2D image, widely applied in autonomous driving, robotics, and augmented reality. Existing methods are mainly categorized into: (1) Regression methods (e.g., BTS, PixelFormer) which directly regress depth values; (2) Classification methods (e.g., AdaBins, BinsFormer) which discretize continuous depth before classification.

**Limitations of Prior Work**: (1) Pure regression methods are prone to overfitting, struggle to generate fine depth details, and often result in blurry object boundaries and shapes. (2) Classification-based methods discretize depth values using bin centers, leading to discontinuities and fuzziness in depth maps. (3) Both categories adopt static, single-step prediction, lacking the capability for progressive refinement.

**Key Challenge**: Depth estimation requires capturing both coarse-grained scene layout and fine-granular object details, but existing single-step prediction paradigms struggle to accommodate both simultaneously. While diffusion models have demonstrated powerful progressive refinement capabilities in generative tasks, applying them to depth estimation faces a critical barrier—GT depths in outdoor scenes are extremely sparse (e.g., KITTI has only 3.75%-5% valid pixels), and training generative models directly on sparse GT leads to mode collapse.

**Goal**: (1) How to introduce the iterative refinement capability of diffusion models into depth estimation? (2) How to effectively train diffusion models under sparse GT depth scenarios?

**Key Insight**: Reframe depth estimation as a conditional denoising diffusion process—starting from a random depth distribution, it progressively denoises it into an accurate depth map guided by monocular visual conditions. Concurrently, a self-diffusion strategy is proposed to add noise to the model's own refined depths instead of the sparse GT, bypassing the sparsity issue.

**Core Idea**: Reframe depth estimation as a visually-conditioned iterative denoising process, and resolve the training challenge under sparse GT scenarios via a self-diffusion mechanism, thereby achieving progressive, high-quality depth refinement.

## Method

### Overall Architecture

The overall architecture of DiffusionDepth consists of three core components: (1) A visual feature extractor (Swin Transformer + FPN) to construct multi-scale visual conditions $c$ from the input image; (2) A depth encoder-decoder to map depth maps to the latent space and back; (3) A Monocular Conditional Denoising Block (MCDB) to perform conditional denoising in the latent space. During inference, starting from random noise $x_T$, a refined depth latent representation $x_0$ is obtained through a $T$-step denoising process, which is then decoded to output the final depth map.

### Key Designs

1. **Task Reformulation - Depth Estimation as Denoising**:
    - **Function**: Transitions depth estimation from conventional regression/classification to a conditional generative paradigm.
    - **Mechanism**: Defines the conditional denoising process as $p_\theta(x_{t-1}|x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \sigma_t^2 I)$, where $c$ represents the monocular visual conditions. DDIM is employed to accelerate inference, with the random variance set to 0 to ensure deterministic predictions.
    - **Design Motivation**: The iterative refinement of diffusion models is naturally suited for depth estimation—the initial steps establish the coarse structure, while subsequent steps progressively correct details and distance relationships.

2. **Monocular Conditional Denoising Block (MCDB)**:
    - **Function**: Performs denoising on the depth latent representation guided by visual conditions.
    - **Mechanism**: The visual conditions $c \in \mathbb{R}^{H/4 \times W/4 \times c}$ are upsampled to match the size of the depth latent $x_t \in \mathbb{R}^{H/2 \times W/2 \times d}$ via a local projection layer, fused using element-wise addition, and further processed by CNN blocks and self-attention layers. This employs a lightweight design to ensure inference efficiency.
    - **Design Motivation**: Visual conditions provide scene semantics and structural information. Hierarchical fusion allows the denoising process to utilize visual cues in the image to guide depth refinement.

3. **Self-Diffusion Mechanism**:
    - **Function**: Resolves the mode collapse issue of training diffusion models under sparse GT depths.
    - **Mechanism**: During training, instead of adding noise to the sparse GT depths, noise is added to the refined depth latent $x_0$ output by the model's current denoising process to produce $x_t$. The supervision signal aligns the refined depth with the GT using a sparse validity mask. Random cropping, jittering, and flipping augmentations are applied.
    - **Design Motivation**: Outdoor depth GT covers only 3.75%-5% of the pixels; training diffusion models directly on sparse GT results in collapse (experimentally verified as non-convergent on KITTI). Self-diffusion enables the model to learn to "organize" the entire depth map rather than merely regressing known parts.

### Loss & Training

The total loss is a weighted sum of three terms: $L = \lambda_1 L_{ddim} + \lambda_2 L_{pixel} + \lambda_3 L_{latent}$. Among them, $L_{ddim} = \|x_{t-1} - \mu_\theta(x_t, t, c)\|^2$ supervises the denoising step; $L_{pixel}$ is the pixel-level depth loss (a variant of SILog); $L_{latent} = \|x_0 - \hat{x}_0\|^2$ aligns the latent space. The diffusion process involves 1000 steps of training and 20 steps of inference. The first 50% of iterations utilize L1+L2 auxiliary losses. An AdamW optimizer is used to train on 8×A100 GPUs for 30 epochs.

## Key Experimental Results

### Main Results

| Dataset | Metric | DiffusionDepth | Prev. SOTA (URCDC-Depth) | Gain |
|--------|------|----------------|----------------------|------|
| KITTI Eigen (0-80m) | RMSE↓ | **2.016** | 2.032 | 0.8% |
| KITTI Eigen (0-80m) | Abs Rel↓ | **0.050** | 0.050 | - |
| KITTI Official (0-50m) | RMSE↓ | **1.418** | 1.528 | 7.2% |
| KITTI Official (0-50m) | Abs Rel↓ | **0.041** | 0.049 | 16.3% |
| NYU-Depth-V2 | RMSE↓ | **0.295** | - | - |
| NYU-Depth-V2 | Abs Rel↓ | **0.085** | - | - |

### Ablation Study

| Configuration | Rel.↓ | RMSE↓ | Description |
|------|-------|-------|------|
| 20 inference steps (Standard) | 0.086 | 0.298 | Default configuration |
| 15 steps (Direct change) | 0.118 | 0.455 | Severe performance degradation |
| 10 steps (Direct change) | 0.182 | 0.751 | Unusable |
| 15 steps (Retrained) | - | ~0.30 | Only minor degradation |
| Diffusion on sparse GT (KITTI) | - | Not converging | Mode collapse |
| Diffusion on sparse GT (NYU) | - | Comparable | Feasible in dense GT scenes |

### Key Findings

- The self-diffusion mechanism is the key to successful training in outdoor scenes—direct diffusion on sparse GT does not converge on KITTI.
- Visualization of the denoising process shows: the first 10 steps establish scene structure, and subsequent steps refine distance relationships and boundaries.
- Using $\times 2$ downsampling in the depth latent space is slightly superior to $\times 4$ downsampling.
- The framework is compatible with various visual backbones (ResNet, Swin) and can be combined with classification-based methods like Bins.
- Inference speed: 14 FPS with ResNet backbone, 5 FPS with Swin backbone (RTX 3090, 20 steps).

## Highlights & Insights

- **First to introduce diffusion models to depth estimation**: Opens up a new paradigm of generative methods in 3D perception tasks.
- **Ingeniously designed self-diffusion mechanism**: Solves the core difficulty of training generative models under sparse GT with a simple and clean idea.
- **Visualization reveals denoising semantics**: Visualizations demonstrate physical interpretability, first identifying shape boundaries and then correcting depth relationships.
- **Highly versatile**: The diffusion head can be freely integrated with different backbones and depth feature extraction methods.

## Limitations & Future Work

- Inference speed is constrained by the number of denoising steps; while 20 steps is practical, it remains slower than direct regression methods.
- Directly changing the number of inference steps (without retraining) leads to severe performance drops, limiting flexibility.
- Performance on the online benchmark (KITTI Online) is slightly lower than VA-Depth and URCDC-Depth, likely due to not using advanced feature extraction techniques such as long-range attention.
- Faster sampling strategies (such as Consistency Models) can be explored to reduce inference steps to 1-4.

## Related Work & Insights

- **DiffusionDet**: A pioneer in object detection using diffusion models, framing bounding box generation as a denoising process.
- **Latent Diffusion**: The concept of performing diffusion in the latent space directly inspired the design of the depth latent space in this study.
- **VA-Depth**: Performs depth refinement using variational inference, representing a non-diffusion refinement paradigm.
- **Insights**: The iterative refinement paradigm of diffusion models can be extended to other 3D perception tasks such as stereo matching and depth completion.

## Rating

- Novelty: ⭐⭐⭐⭐ Establishes diffusion models for depth estimation for the first time; the self-diffusion mechanism is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual indoor-outdoor datasets + extensively ablated + denoising process visualisations.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with well-articulated motivations.
- Value: ⭐⭐⭐⭐ Pioneering work that paves the way for subsequent methods like Marigold.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)
- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)
- [\[CVPR 2026\] Iris: Integrating Language into Diffusion-based Monocular Depth Estimation](../../CVPR2026/3d_vision/iris_integrating_language_into_diffusion-based_monocular_depth_estimation.md)
- [\[ECCV 2024\] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)
- [\[ECCV 2024\] P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)

</div>

<!-- RELATED:END -->
