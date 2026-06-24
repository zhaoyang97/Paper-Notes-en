---
title: >-
  [Paper Note] Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction
description: >-
  [ICCV 2025][3D Vision][4D reconstruction] This paper adapts a pretrained video diffusion model (DynamiCrafter) into a monocular 4D dynamic scene reconstructor that simultaneously predicts three complementary geometric modalities — point maps, disparity maps, and ray maps. Through a multi-modal alignment and fusion algorithm combined with sliding-window inference, the model generalizes zero-shot to real videos despite being trained exclusively on synthetic data…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "4D reconstruction"
  - "video diffusion models"
  - "multi-modal geometry"
  - "point map"
  - "disparity map"
  - "ray map"
  - "dynamic scenes"
date: 2026-05-08
content_hash: 3e98b93702e4c52a
---

# Geo4D: Leveraging Video Generators for Geometric 4D Scene Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2504.07961](https://arxiv.org/abs/2504.07961)  
**Code**: [https://geo4d.github.io](https://geo4d.github.io)  
**Area**: 3D Vision
**Keywords**: 4D reconstruction, video diffusion models, multi-modal geometry, point map, disparity map, ray map, dynamic scenes

## TL;DR
This paper adapts a pretrained video diffusion model (DynamiCrafter) into a monocular 4D dynamic scene reconstructor that simultaneously predicts three complementary geometric modalities — point maps, disparity maps, and ray maps. Through a multi-modal alignment and fusion algorithm combined with sliding-window inference, the model generalizes zero-shot to real videos despite being trained exclusively on synthetic data, substantially outperforming current state-of-the-art video depth estimation methods.

## Background & Motivation

### Core Problem
Feed-forward 4D reconstruction from monocular video — directly recovering the 3D geometry of dynamic scenes (including camera motion and moving objects) from a single-camera video — is an extremely challenging yet consequential problem in computer vision, with broad applications in video understanding, computer graphics, and robotics.

### Limitations of Prior Work

**Iterative optimization methods** (NeRF/3DGS-based): require per-video optimization, incurring large computational overhead and depending on accurate monocular depth priors (e.g., MegaSaM, Uni4D).

**Feed-forward methods** (e.g., MonST3R): extend DUSt3R to dynamic scenes but rely on highly customized architectures and large quantities of 3D-annotated real training data. Such annotations are extremely difficult to obtain for dynamic scenes; falling back to synthetic data introduces a domain gap.

**Depth diffusion models** (e.g., DepthCrafter): estimate depth only, without recovering complete 4D geometry (camera motion + 3D structure).

### Key Insight
Video generation models (as world simulator proxies) implicitly encode an understanding of camera motion, perspective effects, and object dynamics, yet produce only pixels rather than actionable 3D information. The core idea of Geo4D is to make this implicit 3D understanding explicit — by fine-tuning a video diffusion model to directly output geometric modalities.

### Why Multiple Modalities?
A single viewpoint-invariant point map encodes complete 4D geometry but has limited dynamic range: distant objects and the sky (infinite depth) cannot be represented. This motivates the inclusion of:
- **Disparity maps**: zero naturally represents infinite distance, offering better dynamic range
- **Ray maps**: encode camera parameters, are defined for every pixel, and are independent of scene geometry

The three modalities are theoretically redundant but practically complementary — their fusion substantially improves robustness.

## Method

### Overall Architecture
Given a monocular video $\mathcal{I}=\{I^i\}_{i=1}^N$, the network $f_\theta$ simultaneously predicts three geometric modalities per frame:
$$f_\theta: \{I^i\}_{i=1}^N \mapsto \{(D^i, X^i, r^i)\}_{i=1}^N$$
where $D^i \in \mathbb{R}^{H \times W \times 1}$ is the disparity map, $X^i \in \mathbb{R}^{H \times W \times 3}$ is the viewpoint-invariant point map (in the coordinate frame of the first frame), and $r^i \in \mathbb{R}^{H \times W \times 6}$ is the Plücker-coordinate ray map. No camera parameters are required as input.

### Key Designs

#### 1. Multi-Modal Latent Encoding
Built upon DynamiCrafter's VAE encoder-decoder:
- **Disparity maps and ray maps**: directly reuse the pretrained image encoder-decoder without modification
- **Point maps**: the VAE decoder is fine-tuned using an uncertainty-weighted reconstruction loss:
$$\mathcal{L} = -\sum_{uv} \ln \frac{1}{\sqrt{2}\sigma_{uv}} \exp \frac{-\sqrt{2}\ell_1(\mathcal{D}(\mathcal{E}(X))_{uv}, X_{uv})}{\sigma_{uv}}$$
where $\sigma$ is the uncertainty predicted by an auxiliary branch of the decoder. The encoder is kept frozen to minimize changes to the latent space. Point maps are normalized to $[-1,1]$ to be compatible with the pretrained encoder.

#### 2. Video Conditioning Injection (Dual Stream)
- **Global stream**: each frame $I^i$ is encoded via CLIP and injected into each U-Net block via cross-attention through a lightweight query transformer
- **Local stream**: spatial features extracted by the VAE encoder are concatenated channel-wise with the noisy latents of the three geometric modalities

#### 3. Multi-Modal Alignment Fusion (Core Inference Algorithm)

During inference, long videos are divided into overlapping clips using a sliding window ($V=16$ frames, stride $s=4$). Global consistency is achieved by jointly optimizing the following four losses:

**Point map alignment** (group-wise extension of DUSt3R):
$$\mathcal{L}_p = \sum_{g \in \mathcal{G}} \sum_{i \in g} \sum_{uv} \left\| \frac{X^i_{uv} - \lambda_p^g P_p^g X^{i,g}_{uv}}{\sigma^{i,g}_{uv}} \right\|_1$$
recovering per-frame camera intrinsics $K_p^i$, rotation $R_p^i$, center $o_p^i$, and disparity $D_p^i$.

**Disparity map alignment**:
$$\mathcal{L}_d = \sum_{g} \sum_{i \in g} \|D_p^i - \lambda_d^g D_d^{i,g} - \beta_d^g\|_1$$

**Ray map alignment** (camera trajectory alignment):
$$\mathcal{L}_c = \sum_{g} \sum_{i \in g} (\|R_p^{i\top} R_c^g R_c^{i,g} - I\|_f + \|\lambda_c^g o_c^{i,g} + \beta_c^g - o_p^i\|_2)$$

**Trajectory smoothness regularization**:
$$\mathcal{L}_s = \sum_{i=1}^N (\|R_p^{i\top} R_p^{i+1} - I\|_f + \|o_p^{i+1} - o_p^i\|_2)$$

Final objective: $\mathcal{L}_{all} = \alpha_1 \mathcal{L}_p + \alpha_2 \mathcal{L}_d + \alpha_3 \mathcal{L}_c + \alpha_4 \mathcal{L}_s$

### Loss & Training
- Trained exclusively on 5 synthetic datasets (Spring, BEDLAM, PointOdyssey, TarTanAir, VirtualKitti)
- Progressive training schedule: single-modality point map training → multi-resolution training → gradual incorporation of ray maps and disparity maps
- 4× H100 GPUs, approximately one week
- DDIM 5-step sampling at inference

## Key Experimental Results

### Main Results: Video Depth Estimation

| Method | Sintel AbsRel↓ | Sintel δ<1.25↑ | Bonn AbsRel↓ | Bonn δ<1.25↑ | KITTI AbsRel↓ | KITTI δ<1.25↑ |
|------|------|------|------|------|------|------|
| Depth-Anything-V2 | 0.367 | 55.4 | 0.106 | 92.1 | 0.140 | 80.4 |
| DepthCrafter | 0.270 | 69.7 | 0.071 | 97.2 | 0.104 | 89.6 |
| MonST3R | 0.335 | 58.5 | 0.063 | 96.4 | 0.104 | 89.5 |
| **Geo4D** | **0.205** | **73.5** | **0.059** | **97.2** | **0.086** | **93.7** |

Geo4D achieves consistent improvements across all three benchmarks. Compared to DepthCrafter, which shares the same backbone, AbsRel is reduced by 24% on Sintel and 17.3% on KITTI.

### Camera Pose Estimation

| Method | Sintel ATE↓ | Sintel RPE-R↓ | TUM ATE↓ | TUM RPE-R↓ |
|------|------|------|------|------|
| MonST3R | 0.108 | 0.732 | 0.063 | 1.217 |
| **Geo4D** | 0.185 | **0.547** | 0.073 | **0.635** |

Geo4D is the first method to estimate camera parameters for dynamic scenes using a generative model. Rotation estimation (RPE-R) substantially outperforms discriminative methods, while translation estimation is competitive.

### Ablation Study: Multi-Modal Training and Inference

| Training Modalities | Inference Modalities | Sintel AbsRel↓ | ATE↓ | RPE-R↓ |
|------|------|------|------|------|
| Point map only | Point map only | 0.232 | 0.335 | 0.731 |
| All three | Point map only | 0.223 | 0.237 | 0.566 |
| All three | Disparity map only | 0.211 | — | — |
| All three | Full fusion | **0.205** | — | — |

**Key Findings**:
- Multi-modal training improves point-map-only inference, demonstrating an auxiliary task effect
- Disparity maps achieve the best pure depth metrics due to superior dynamic range
- Full three-modality fusion yields the best overall performance

## Highlights & Insights
1. **Paradigm innovation**: the first work to demonstrate that a general-purpose video diffusion model can be effectively repurposed as a 4D geometry reconstructor, without requiring a customized 3D architecture
2. **Elegant complementary multi-modal design**: point maps encode complete structure but suffer from limited dynamic range; disparity maps handle distant regions; ray maps handle camera parameters — each modality covers the weaknesses of the others
3. **Zero-shot generalization from synthetic data**: the strong priors embedded in the video generation model enable reliable generalization to real videos despite purely synthetic training data
4. **Uncertainty-driven alignment**: the uncertainty $\sigma$ predicted by the VAE decoder participates directly in the multi-modal fusion optimization, automatically down-weighting unreliable predictions

## Limitations & Future Work
1. Scale ambiguity of point maps — monocular video cannot determine absolute scale, recovering only up-to-scale geometry
2. Inference speed is constrained by diffusion sampling (DDIM 5-step accelerates inference but is still not real-time)
3. Robustness to extreme dynamic scenes (rapid occlusion or appearance of objects) requires further evaluation
4. Error accumulation in the sliding-window strategy over very long videos

## Related Work & Insights
- **Dynamic scene reconstruction**: DUSt3R → MonST3R → Easi3R (progressive extension from static to dynamic)
- **Geometric diffusion models**: Marigold (image depth), DepthCrafter (video depth), Aether (depth + ray map), GeometryCrafter (point map VAE)
- **Video foundation models**: DynamiCrafter, SVD, etc., serving as backbones for 3D/4D understanding

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Complete pipeline from video diffusion to 4D geometry with a strongly original multi-modal design
- Technical Depth: ⭐⭐⭐⭐⭐ — Comprehensive mathematical framework for multi-modal encoding, decoding, and alignment
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark comparisons and complete ablations; efficiency analysis is lacking
- Value: ⭐⭐⭐⭐ — Strong zero-shot generalization capability, though inference speed requires improvement

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[CVPR 2026\] Complet4R: Geometric Complete 4D Reconstruction](../../CVPR2026/3d_vision/complet4r_geometric_complete_4d_reconstruction.md)
- [\[CVPR 2026\] 4D Primitive-Mâché: Glueing Primitives for Persistent 4D Scene Reconstruction](../../CVPR2026/3d_vision/4d_primitive-mache_glueing_primitives_for_persistent_4d_scene_reconstruction.md)
- [\[CVPR 2025\] Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection](../../CVPR2025/3d_vision/leveraging_3d_geometric_priors_in_2d_rotation_symmetry_detection.md)

</div>

<!-- RELATED:END -->
