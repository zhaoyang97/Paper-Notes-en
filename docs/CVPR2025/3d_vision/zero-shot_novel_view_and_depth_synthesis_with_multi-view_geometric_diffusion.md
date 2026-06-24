---
title: >-
  [Paper Note] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion
description: >-
  [CVPR 2025][3D Vision][Novel View Synthesis] MVGD proposes a multi-view geometric framework based on pixel-level diffusion, which directly generates novel-view images and scale-consistent depth maps from an arbitrary number of known-view images without intermediate 3D representations, achieving state-of-the-art results through training on over 60 million multi-view samples.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Depth Estimation"
  - "Diffusion Models"
  - "Multi-View Geometry"
  - "Multi-Task Learning"
date: 2026-05-08
content_hash: 12e08fbd9606c2f6
---

# MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2501.18804](https://arxiv.org/abs/2501.18804)  
**Code**: [Project Page](https://mvgd.github.io/)  
**Area**: 3D Vision  
**Keywords**: Novel View Synthesis, Depth Estimation, Diffusion Models, Multi-View Geometry, Multi-Task Learning

## TL;DR

MVGD proposes a multi-view geometric framework based on pixel-level diffusion, which directly generates novel-view images and scale-consistent depth maps from an arbitrary number of known-view images without intermediate 3D representations, achieving state-of-the-art results through training on over 60 million multi-view samples.

## Background & Motivation

Reconstructing 3D scenes from sparse posed images is a core problem. Existing methods primarily rely on intermediate 3D representations (NeRF, 3DGS, voxel grids) to ensure multi-view consistency.

Limitations of prior work:
- **NeRF/3DGS-based methods**: Require constructing explicit or implicit 3D scene representations, and their generalization capability is limited by the distribution of input views.
- **Diffusion model methods**: Struggle to guarantee multi-view consistency when applied to novel view synthesis (due to the lack of constraints from intermediate 3D representations).
- **Depth estimation**: Single-frame methods cannot estimate absolute scale, while multi-frame methods face scale heterogeneity issues when training across different datasets.
- **Training data diversity**: Different datasets have different calibration schemes (metric/non-metric scales), varied depth annotation densities, and diverse scene types.

Core Motivation: Can a unified diffusion model be trained to directly generate multi-view consistent images and depths at the pixel level, without relying on any intermediate 3D representations?

## Method

### Overall Architecture

MVGD achieves pixel-level diffusion based on the highly efficient RIN (Recurrent Interface Networks) Transformer architecture, avoiding the need for latent autoencoders. Given $N$ conditional views, an image encoder and a ray encoder generate scene tokens. These tokens are combined with the target camera's ray embeddings and learnable task embeddings to directly generate the image or depth map of the target view through a diffusion process.

### Key Designs

**Design 1: Scene Scale Normalization (SSN) — Unifying Scales Across Datasets**

- **Function**: Automatically extracts the scene scale and injects it into the diffusion process, yielding multi-view consistent metric depth maps.
- **Mechanism**: Expresses all extrinsic camera parameters of the conditioning views relative to the target camera's pose as $\tilde{T}_c^n = T_c^n T_t^{-1}$. The scene scale is defined as the maximum absolute translation component, $s = \max\{|\tilde{x}|, |\tilde{y}|, |\tilde{z}|\}$. All translation vectors and depth values are normalized by dividing them by $s$. During inference, the generated depth is multiplied back by $s$ to restore its metric scale.
- **Design Motivation**: Training across multiple datasets introduces huge scale discrepancies due to varying calibration methods. SSN achieves translation and rotation invariance through relative pose normalization. This allows the model to learn within a unified scale space, ensuring that the generated depths are geometrically consistent with the conditioning cameras.

**Design 2: Learnable Task Embeddings — Unified Multi-Task Generation**

- **Function**: Enables a single model to simultaneously generate both images and depth maps, supporting joint training on datasets with or without depth annotations.
- **Mechanism**: Uses a learnable task embedding $E^{task} \in \mathbb{R}^{D_{task}}$ to guide the diffusion process toward a specific modality (RGB or Depth), which is appended to the predicted tokens. An L2 loss is used for the RGB task, and an L1 loss (only on valid ground-truth pixels) is used for the depth task.
- **Design Motivation**: Simple joint RGB-D generation limits the training datasets to only those with dense depth. Conditioning the latent tokens decouples appearance and geometry priors. The task embedding program permits mixed training while maintaining a shared implicit 3D representation.

**Design 3: RIN Pixel-Level Diffusion + Raymap Conditioning — Efficient Multi-View Geometric Reasoning**

- **Function**: Efficiently performs diffusion in pixel space, supporting an arbitrary number of conditioning views (up to 100+).
- **Mechanism**: Employs a fixed number $L$ of latent tokens for self-attention computation, which interact with input/output tokens via cross-attention. A Raymap (Fourier encoding of ray origins and directions) is utilized both to enhance the spatial position information of conditioning views and to designate the target novel view. An incremental multi-view generation strategy maintains historically generated images as additional conditioning sources.
- **Design Motivation**: RIN decouples the computational complexity from the number of input tokens ($O(L^2)$ instead of $O(N^2)$), allowing efficient handling of a large number of conditioning views. Pixel-level diffusion avoids the loss of detail associated with autoencoders and the requirement for dense grid inputs.

### Loss & Training

The RGB task is trained using an L2 loss, and the depth task is trained using an L1 loss (only on valid ground-truth pixels), both following the standard DDPM noise prediction framework. Depth is parameterized in log scale as $P_D = 2(\log(\frac{D}{s \cdot d_{min}}) / \log(\frac{d_{max}}{d_{min}})) - 1$, with range limits $d_{min}=0.1$ and $d_{max}=200$.

## Key Experimental Results

### Main Results: 2-View Novel View Synthesis

| Method | RE10K PSNR ↑ | RE10K SSIM ↑ | RE10K LPIPS ↓ | ACID PSNR ↑ |
|------|-------------|-------------|--------------|-------------|
| PixelNeRF | 20.43 | 0.589 | 0.550 | 20.97 |
| MuRF | 26.10 | 0.858 | 0.143 | 28.09 |
| PixelSplat | 25.89 | 0.858 | 0.142 | 28.14 |
| MVSplat | 26.39 | 0.869 | 0.128 | 28.25 |
| **MVGD** | **28.41** | **0.891** | **0.107** | **29.98** |

### Ablation Study: Training Data and Model Scaling

| Configuration | PSNR ↑ | Description |
|------|--------|------|
| Base (256 latents) | Baseline | Standard configuration |
| Removing dynamic datasets | Decrease | Diversity is important |
| Without SSN | Decrease | Inconsistent depth scale |
| 512 latents (Incremental fine-tuning) | Gain | Saves 70% training time |

### Key Findings

- On RealEstate10K, the PSNR improves by 2.02 dB compared to MVSplat, and 2.52 dB compared to PixelSplat.
- Capable of processing 100+ conditioning views without increasing computational complexity (thanks to the fixed number of latent tokens).
- The incremental fine-tuning strategy reduces the training time of larger models by 70%.
- Jointly training image and depth generation promotes implicit geometric understanding, enhancing the quality of novel view synthesis.
- Achieves state-of-the-art (SOTA) performance on ScanNet multi-view stereo mapping and video depth estimation.

## Highlights & Insights

1. **End-to-End Generation Without Intermediate 3D Representations**: Demonstrates that diffusion models can implicitly learn multi-view geometric consistency.
2. **Large-Scale Heterogeneous Training**: Trained on over 60 million samples covering diverse scenarios such as driving, indoors, robotics, and synthetic environments, demonstrating robust zero-shot generalization capability.
3. **Incremental Model Scaling Strategy**: Efficient model scaling is achieved via latent token replication and fine-tuning.

## Limitations & Future Work

- Does not model dynamic objects, despite dynamic scenes being present in the training data.
- The current resolution is limited to 256 pixels (longest side), with higher resolutions demanding more computation.
- It can be extended to downstream tasks such as video prediction and scene editing in the future.

## Related Work & Insights

- **PixelSplat/MVSplat**: Generalizable novel view synthesis based on 3DGS, requiring explicit 3D representations.
- **CAT3D/Reconfusion**: Diffusion + 3D reconstruction pipelines that rely on intermediate representations to guarantee multi-view consistency.
- **RIN**: An efficient Transformer architecture that decouples computational complexity from input size.
- **Insights**: Given sufficiently diverse training data, the diffusion model itself is capable of learning implicit 3D geometric reasoning.

## Rating

⭐⭐⭐⭐⭐ — A truly systematic piece of work: showing innovations across architecture design (RIN pixel diffusion), training strategy (SSN + multi-task embedding + 60M data), and scaling strategy (incremental fine-tuning). It sweeps new SOTAs across multiple benchmarks while maintaining an elegant and concise methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [\[CVPR 2025\] MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)
- [\[CVPR 2025\] Cross-View Completion Models are Zero-shot Correspondence Estimators](cross-view_completion_models_are_zero-shot_correspondence_estimators.md)

</div>

<!-- RELATED:END -->
