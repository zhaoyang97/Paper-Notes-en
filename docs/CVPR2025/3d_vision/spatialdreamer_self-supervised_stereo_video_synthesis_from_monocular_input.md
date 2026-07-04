---
title: >-
  [Paper Note] SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input
description: >-
  [CVPR 2025][3D Vision][Stereo Video Synthesis] SpatialDreamer is proposed as a self-supervised stereo video synthesis framework based on video diffusion models. It addresses the shortage of training data through a Depth-guided Video Generation (DVG) module and ensures geometric and temporal consistency via a RefinerNet framework along with a consistency control module (incorporating stereo deviation strength and Temporal Interaction Learning…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Stereo Video Synthesis"
  - "Self-supervised Learning"
  - "Video Diffusion Models"
  - "Spatiotemporal Consistency"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 9aad32c929a3f80a
---

# SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input

**Conference**: CVPR 2025  
**arXiv**: [2411.11934](https://arxiv.org/abs/2411.11934)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Stereo Video Synthesis, Self-supervised Learning, Video Diffusion Models, Spatiotemporal Consistency, Novel View Synthesis

## TL;DR

SpatialDreamer is proposed as a self-supervised stereo video synthesis framework based on video diffusion models. It addresses the shortage of training data through a Depth-guided Video Generation (DVG) module and ensures geometric and temporal consistency via a RefinerNet framework along with a consistency control module (incorporating stereo deviation strength and Temporal Interaction Learning, or TIL). Performance exceeds the Apple Vision Pro 3D converter.

## Background & Motivation

Stereo video synthesis generates target-view videos from monocular inputs and is widely applied in 3D filmmaking and virtual reality (VR) devices such as Apple Vision Pro. This task faces two core challenges: (1) **Lack of high-quality stereo video pairs for training**, as conventional dual-camera acquisition is highly expensive; and (2) **Difficulty in maintaining spatiotemporal consistency**, since generated videos are prone to inter-frame jitter and flickering.

Existing approaches primarily apply single-image Novel View Synthesis (NVS) techniques directly to videos, but struggle to effectively represent dynamic scenes. Layer-based methods (e.g., 3D-photography, MPI) introduce depth discretization artifacts, while NeRF/3DGS-based methods yield limited quality from sparse views. Even the 3D photo converter of Apple Vision Pro produces content flickering and inconsistencies when processing videos.

The core idea of this work is to **leverage the powerful spatiotemporal modeling capability of stable video diffusion (SVD) to simultaneously resolve data scarcity and spatiotemporal inconsistency within a self-supervised framework**.

## Method

### Overall Architecture

SpatialDreamer is built upon Stable Video Diffusion (SVD) and comprises four core modules: (1) The DVG module generates paired training videos via forward-backward rendering; (2) RefinerNet extracts spatial features from the reference view and injects them into the denoising U-Net; (3) The Temporal Interaction Learning (TIL) module fuses features from long-term frames to enhance temporal consistency; and (4) The stereo deviation strength metric controls the intensity of the 3D effect. The inputs are a monocular video and stereo poses, and the output is a stereo video pair.

### Key Designs

**1. Depth-based Video Generation (DVG) — Depth-Guided Video Data Generation**

- **Function**: Self-supervisedly generates paired stereo video training data from monocular videos to address data scarcity.
- **Mechanism**: A forward-backward rendering mechanism is employed: (a) estimate video depth; (b) render the reference-view image to the target view to generate an occlusion mask; (c) use an inpainting model to fill the occluded regions, yielding the target-view image $x_2$; (d) backward-render $x_2$ back to the original view. A key improvement is the propagation of occlusion information across adjacent frames using optical flow $u, v$ and forward-backward consistency confidence $C$: $m^t(i,j) = 1$ when $\sum_{k \in \{t-1,t+1\}} m^k(i+u,j+v) \cdot C(i,j) \geq 1$.
- **Design Motivation**: Direct frame-by-frame rendering leads to inconsistent occlusion masks across frames, inducing jitter. By establishing inter-frame pixel correspondences via optical flow and fusing occlusion information from adjacent frames, the occluded areas are made smoother and more coherent temporally.

**2. RefinerNet + Spatial Attention — Reference View Feature Injection**

- **Function**: Learns spatial distribution discrepancies between paired views and injects reference-view features into the denoising process.
- **Mechanism**: RefinerNet adopts the same architecture as the denoising U-Net (excluding temporal layers) and is initialized with SD2.1 weights. Feature maps $z_t$ from the denoising U-Net and $z_r$ from RefinerNet are concatenated along the height dimension before self-attention is applied, with the first half taken as the output. This allows the U-Net to adaptively learn associated features from RefinerNet within the same feature space.
- **Design Motivation**: Compared to the ControlNet architecture, RefinerNet shares the same structure and weight initialization with U-Net, keeping both in the same feature space and allowing more natural feature fusion of paired views (experiments demonstrate that RefinerNet outperforms ControlNet).

**3. Consistency Control Module — Consistency Control**

- **Function**: Simultaneously ensures geometric and temporal consistency.
- **Mechanism**: Contains two sub-modules: (a) **TIL (Temporal Interaction Learning)**: fuses the reference frame feature $z_r^t$ with those of $N_r$ neighboring frames: $aug_r^t = \lambda \cdot \text{Attn}_{r,r} + (1-\lambda) \cdot \frac{1}{N_r}\sum_{i=1}^{N_r}\text{Attn}_{r,i}$, balancing self-attention and cross-frame attention; (b) **Stereo Deviation Strength**: quantifies the latent space discrepancy between the two views $s(z) = |z_0 - z_{ref}|$, serving as a positional embedding added to the residual blocks, and directly supervised via a stereo-aware loss $l_d = \|s(z_0) - s(\hat{z}_0)\|_2^2$.
- **Design Motivation**: Frame-by-frame spatial guidance from RefinerNet alone is insufficient to guarantee temporal consistency. TIL enhances temporal coherence using global long-term frame information, while the stereo deviation strength allows the model to adaptively control the 3D effect based on scene depth, acknowledging that different scenes should present varying levels of stereoscopic depth even under the same viewpoint.

### Loss & Training

- **Total Loss**: $l = l_\epsilon + \lambda \cdot l_d$
    - $l_\epsilon$: v-prediction MSE loss of SVD.
    - $l_d$: stereo-aware loss, supervising the stereo deviation consistency between the generated and target videos.
- During training, the reference and target views are swapped (both left-to-right and right-to-left serve as training pairs).
- The dataset generated by DVG contains geometric and temporal priors, enabling efficient self-supervised training.

## Key Experimental Results

### Main Results

Comparison of stereo image quality on RealEstate10K:

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 3D-photography | 0.855 | 23.93 | 0.112 |
| Deep3D | 0.808 | 22.94 | 0.183 |
| MVSplat | 0.863 | 25.89 | 0.132 |
| **SpatialDreamer** | **0.916** | **32.26** | **0.038** |

Comparison of stereo video quality (FVD↓ / $E_{warp}^*$↓):

| Method | FVD ↓ | $E_{warp}^*$ ↓ |
|------|-------|-----------------|
| 3D-photography | 155.0 | 3.418 |
| AVP (Apple Vision Pro) | 99.92 | 3.446 |
| NVS-Solver | 249.1 | 5.842 |
| **SpatialDreamer** | **67.09** | **3.374** |

### Ablation Study

Contributions of each component (image-level / video-level):

| Component | SSIM ↑ | PSNR ↑ | LPIPS ↓ | FVD ↓ |
|------|--------|--------|---------|-------|
| U-Net only | 0.880 | 23.73 | 0.06 | - |
| + ControlNet | 0.855 | 24.04 | 0.183 | - |
| + RefinerNet | 0.895 | 30.20 | 0.043 | - |
| + RefinerNet + SDS | 0.916 | 32.26 | 0.038 | - |
| + RN + SDS (video) | - | - | - | 184.0 |
| + RN + SDS + TIL | - | - | - | 123.5 |
| + RN + SDS + DVG | - | - | - | 85.21 |
| + RN + SDS + TIL + DVG | - | - | - | **67.09** |

### Key Findings

1. **RefinerNet significantly outperforms ControlNet**: PSNR improves from 24.04 to 30.20, indicating that feature fusion through a shared architecture is more suitable for stereo synthesis than control signal injection.
2. **DVG is the key component contributing most to video quality improvement**: FVD drops from 184.0 to 85.21 (without TIL), showing that optical flow-guided occlusion mask fusion effectively improves temporal consistency.
3. **Outperforms the Apple Vision Pro 3D converter**: FVD of 67.09 versus 99.92, with lower warp error as well, demonstrating strong industrial application potential.
4. Robust across diverse depth estimation methods (e.g., DepthAnything, Marigold, MiDaS), illustrating the generalizability of the framework.

## Highlights & Insights

1. **The self-supervised paradigm bypasses the bottleneck of acquiring paired stereo video data**: The forward-backward rendering and optical flow refinement in DVG present an elegant data generation paradigm.
2. **The introduction of stereo deviation strength is highly practical**: Different scenes at the same viewpoint should yield distinct 3D effects based on depth, a controllable feature critical for VR applications.
3. **Outperforming the commercial product AVP 3D converter** is highly convincing and underscores the engineering feasibility of the academic approach.

## Limitations & Future Work

1. Reliance on depth maps for rendering implies that depth estimation errors will degrade stereoscopic quality.
2. Massive model parameters hinder real-time deployment.
3. Implicit depth representations can be explored in future work as alternatives to explicit depth maps.
4. Can be extended to a wider range of VR/AR application scenarios.

## Related Work & Insights

- **SVD (Stable Video Diffusion)**: Provides a robust base model for video generation, on top of which SpatialDreamer incorporates stereo viewpoint control.
- **AdaMPI / SinMPI**: Multi-plane image-based NVS methods that exhibit depth discretization artifacts, which SpatialDreamer avoids via diffusion modeling.
- **ControlNet**: Utilized as a baseline; RefinerNet demonstrates that a shared feature space is more suitable for stereo synthesis than control signal injection.
- **DepthCrafter**: A video depth estimation method that supplies temporally consistent depth maps to DVG.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Novel self-supervised stereo video synthesis paradigm, with a complete package of DVG, RefinerNet, and consistency control design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated against multiple methods (including commercial AVP), presenting comprehensive ablations and exploring various depth estimators.
- **Writing Quality**: ⭐⭐⭐ — Overall clear, though some mathematical notations are slightly inconsistent.
- **Value**: ⭐⭐⭐⭐ — Directly valuable for VR content creation, with impressive performance exceeding that of commercial products.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](../../ICCV2025/3d_vision/rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[CVPR 2025\] Sonata: Self-Supervised Learning of Reliable Point Representations](sonata_self-supervised_learning_of_reliable_point_representations.md)
- [\[ICLR 2026\] True Self-Supervised Novel View Synthesis is Transferable](../../ICLR2026/3d_vision/true_self-supervised_novel_view_synthesis_is_transferable.md)
- [\[CVPR 2025\] Consistency-aware Self-Training for Iterative-based Stereo Matching](consistency-aware_self-training_for_iterative-based_stereo_matching.md)
- [\[CVPR 2026\] From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis](../../CVPR2026/3d_vision/from_none_to_all_self-supervised_3d_reconstruction_via_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->
