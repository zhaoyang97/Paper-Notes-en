---
title: >-
  [Paper Note] Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation
description: >-
  [CVPR 2026][Video Generation][4D Generation] This paper proposes Diff4Splat, a feed-forward framework that unifies video diffusion models with deformable 3D Gaussian fields into an end-to-end trainable model…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "4D Generation"
  - "3D Gaussian Splatting"
  - "Video Diffusion Models"
  - "Deformable Gaussian Fields"
  - "Feed-Forward Generation"
date: 2026-05-08
content_hash: 0e6b780267f247d0
---

# Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation

**Conference**: CVPR 2026
**arXiv**: [2511.00503](https://arxiv.org/abs/2511.00503)  
**Code**: [Project Page](https://paulpanwang.github.io/Diff4Splat)  
**Area**: Video Generation
**Keywords**: 4D Generation, 3D Gaussian Splatting, Video Diffusion Models, Deformable Gaussian Fields, Feed-Forward Generation

## TL;DR

This paper proposes Diff4Splat, a feed-forward framework that unifies video diffusion models with deformable 3D Gaussian fields into an end-to-end trainable model, enabling direct generation of dynamic 4D scene representations from a single image in approximately 30 seconds — roughly 60× faster than optimization-based methods.

## Background & Motivation

Dynamic 3D scene generation (4D generation) is a core challenge in computer vision, with broad applications in immersive content creation, robotics, and simulation. Existing approaches face fundamental trade-offs:

**Multi-stage pipeline methods**: These first generate video via video diffusion models and then perform 3D reconstruction. They are slow, error-prone, and lack end-to-end control. For instance, DimensionX requires several GPU hours, and Mosca takes around half an hour.

**Feed-forward generation methods**: While efficient, most are limited to generating 2D video frames or static 3D scenes, and fail to capture explicit dynamic 3D geometry.

**Core gap**: There is no unified framework that can directly and efficiently synthesize explicit, controllable scene representations.

Diff4Splat is motivated by bridging this gap — unifying generation and representation within a single forward pass to achieve both feed-forward efficiency and explicit 3D representation.

## Method

### Overall Architecture

Given a single input image $\mathbf{I}_0$, a camera trajectory $\mathcal{P}$ (in Plücker coordinates), and an optional text prompt $\mathbf{C}_{ctx}$, Diff4Splat predicts a deformable 3D Gaussian field in a single forward pass. The framework consists of four core components: (1) a video diffusion model that generates 3D-aware latent tensors; (2) a Latent Dynamic Reconstruction Model (LDRM) that transforms latent features into Gaussian parameters; (3) a deformable Gaussian field to represent dynamics; and (4) unified multi-task supervision.

### Key Designs

1. **Large-scale 4D data pipeline**: Seven synthetic datasets (TartanAir, MatrixCity, PointOdyssey, etc.) and two real-world datasets (RealEstate10K, Stereo4D) are integrated, yielding approximately 130,000 high-quality 4D training scenes. For real-world datasets, VideoDepthAnything and MegaSaM are used to recover metric-scale depth, with least-squares alignment applied to map relative depth to metric depth. This addresses the lack of metric-scale annotations in real-world data.

2. **Latent Dynamic Reconstruction Model (LDRM)**: Built upon a pretrained video diffusion model (CogVideoX), the LDRM generates 3D-aware latent tensors $\mathbf{z} \in \mathbb{R}^{n \times h \times w \times c}$. It consists of 16 standard Transformer blocks that process concatenated latent features and camera pose tokens, followed by a lightweight decoder that regresses 3D Gaussian attributes. The core design motivation is to avoid per-scene optimization by leveraging the generative prior of the diffusion model to directly predict 3D structure.

3. **Deformable Gaussian fields**: Building on static 3DGS, an inter-frame deformation model is introduced. For each Gaussian at timestep $t$, the model predicts a displacement $\Delta\boldsymbol{\mu}_p^t$, a rotation adjustment $\Delta\mathbf{q}_p^t$, and a scale modification $\Delta\mathbf{s}_p^t$. The deformation parameter dimensionality is $K_d=10$. The LDRM jointly outputs a Gaussian feature map and a deformation map. Pruning based on an opacity threshold ($\tau=0.005$) is applied during both training and inference.

4. **Progressive training strategy**:

    - Stage 1 (40K iterations): Pretrains the LDRM on static scenes at low resolution (256×256) with the deformation module frozen, using only photometric and geometric losses.
    - Stage 2 (40K iterations): Still with the deformation module frozen, refines reconstruction fidelity at high resolution (512×512).
    - Stage 3 (20K iterations): Unfreezes the full model and fine-tunes on dynamic datasets using the complete loss, including the motion loss.

### Loss & Training

The total loss is a weighted sum of four terms:

$$\mathcal{L} = \mathcal{L}_{FM} + \lambda_{photo}\mathcal{L}_{photo} + \lambda_{geo}\mathcal{L}_{geo} + \lambda_{motion}\mathcal{L}_{motion}$$

- **Flow Matching loss** $\mathcal{L}_{FM}$: Applied exclusively to the video diffusion model parameters; fine-tunes on 4D annotated data to align the latent space.
- **Photometric loss** $\mathcal{L}_{photo}$: MSE + LPIPS ($\lambda_p=0.5$), optimizing appearance consistency between rendered and ground-truth images.
- **Geometric loss** $\mathcal{L}_{geo}$: Pearson correlation loss on depth + total variation smoothness loss, with weight $\lambda_{geo}=0.5$.
- **Motion loss** $\mathcal{L}_{motion}$: Based on 3D point tracking data (directly available for synthetic data; obtained via CoTracker for real data), combining L2 loss with L1 regularization, weight $\lambda_{motion}=2.0$.

Training uses AdamW with a learning rate of $10^{-5}$, conducted on 32 A100 GPUs for approximately 7 days.

## Key Experimental Results

### Main Results

| Method | FVD↓ | KVD↓ | CLIP-Score↑ | Reconstruction Time |
|--------|------|------|-------------|---------------------|
| CameraCtrl | 478.2 | 8.11 | 19.37 | 20s |
| AC3D | 339.4 | 6.34 | 20.67 | 28s |
| AC3D + Mosca† | 236.0 | 2.01 | 20.21 | **45min** |
| **Diff4Splat** | **210.2** | **2.32** | **23.12** | **30s** |

| Method | Avg Matches↑ | Subj. Consist.↑ | Bg. Consist.↑ | Time↓ |
|--------|-------------|----------------|---------------|-------|
| AC3D + Mosca† | 4500.7 | 86.23 | 90.43 | 45min |
| **Diff4Splat** | **5114.2** | **88.32** | **89.89** | **30s** |

| Method | RPE (Translation)↓ | RPE (Rotation)↓ | NVS | Depth | Real-time Interaction |
|--------|-------------------|----------------|-----|-------|-----------------------|
| AC3D | 3.001 | 0.810 | ✓ | ✗ | ✗ |
| **Ours** | **0.012** | **0.008** | ✓ | ✓ | ✓ |

### Ablation Study

| Configuration | FVD↓ | KVD↓ | Avg Matches↑ | Note |
|---------------|------|------|-------------|------|
| w/o motion loss | 351.4 | 3.35 | 4821.6 | Significant performance drop without motion loss |
| Full model | **210.2** | **2.32** | **5114.2** | Full model achieves best performance |

### Key Findings

1. Feed-forward generation requires only 30 seconds, approximately **90×** faster than optimization-based methods (Mosca: 45 minutes).
2. The proposed method outperforms optimization-based approaches on both video quality (FVD) and geometric consistency (Avg Matches).
3. The explicit 3DGS representation reduces camera pose error by 250× (RPE Translation: 3.001 → 0.012).
4. The deformable Gaussian field is critical for eliminating ghosting artifacts in dynamic scenes.
5. The progressive training strategy saves 3× training time compared to direct dynamic training, while yielding superior results.

## Highlights & Insights

- **Paradigm innovation**: The first work to unify video diffusion models with deformable 3DGS in a feed-forward framework, entirely eliminating per-scene optimization.
- **Efficiency leap**: 30 seconds vs. 45 minutes, bringing dynamic 3D scene generation to practical usability for the first time.
- **Data pipeline**: A large-scale 4D dataset of 130,000 scenes with metric-scale annotations is constructed, with plans for open-sourcing.
- **Versatility**: A single model simultaneously supports video generation, novel view synthesis, depth extraction, and real-time interaction.
- **Biological analogy**: The spatial relationship head operates analogously to molecular gradients guiding cell differentiation in embryonic development.

## Limitations & Future Work

1. Training costs remain high (32× A100, 7 days), making rapid iteration difficult.
2. The design relies on CogVideoX's latent space; scalability to higher resolutions or longer sequences remains to be validated.
3. Metric depth for real-world data depends on the accuracy of VideoDepthAnything and MegaSaM, introducing the risk of error propagation.
4. The current framework only supports generation from a single image; extension to multi-view input conditioning is worth exploring.
5. Motion is represented as simple displacement + rotation + scale deformation, which may be insufficient for topological changes (e.g., object appearance or disappearance).

## Related Work & Insights

- CogVideoX serves as the video diffusion backbone, demonstrating the potential of video generation priors for 3D understanding.
- The combination of 3DGS and deformation fields provides high-quality, real-time rendering for dynamic scenes.
- The progressive training strategy (static → high-resolution → dynamic) represents effective engineering practice for complex tasks.
- The framework is extensible to downstream applications such as robotics simulation and VR/AR content creation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First feed-forward 4D generation framework unifying diffusion models with deformable 3DGS.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dimensional evaluation is comprehensive, though comparisons with more feed-forward 4D methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with detailed method descriptions.
- Value: ⭐⭐⭐⭐⭐ — Substantial efficiency gains with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_multimodal_multiview_video_diffusion.md)
- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)
- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](../../ICLR2026/video_generation/target-aware_video_diffusion_models.md)
- [\[CVPR 2026\] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)

</div>

<!-- RELATED:END -->
