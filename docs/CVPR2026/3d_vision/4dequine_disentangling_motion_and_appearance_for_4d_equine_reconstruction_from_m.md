---
title: >-
  [Paper Note] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video
description: >-
  [CVPR 2026][3D Vision][4D reconstruction] This paper proposes the 4DEquine framework, which **disentangles** 4D equine reconstruction from monocular video into two subproblems — dynamic motion estimation (AniMoFormer) and static appearance reconstruction (EquineGS) — achieving SOTA on real-world data while training exclusively on synthetic data.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D reconstruction
  - equine reconstruction
  - 3D Gaussian Splatting
  - parametric model
  - monocular video
  - feed-forward
date: 2026-05-08
content_hash: e9af4e1f98f55544
---

# 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video

**Conference**: CVPR 2026
**arXiv**: [2603.10125](https://arxiv.org/abs/2603.10125)
**Code**: N/A
**Area**: 3D Vision
**Keywords**: 4D reconstruction, equine reconstruction, 3D Gaussian Splatting, parametric model, monocular video, feed-forward

## TL;DR

This paper proposes the 4DEquine framework, which **disentangles** 4D equine reconstruction from monocular video into two subproblems — dynamic motion estimation (AniMoFormer) and static appearance reconstruction (EquineGS) — achieving SOTA on real-world data while training exclusively on synthetic data.

## Background & Motivation

4D reconstruction of equines (horses, donkeys, zebras) from monocular video has significant value in animal welfare and sports analysis, yet existing methods face two fundamental challenges:

**Optimization bottleneck**: Mainstream 4D animal reconstruction methods (GART, SMALR/SMALST, DogRecon, etc.) require joint optimization of motion and appearance over entire video sequences, incurring large computational costs (GART requires 15 minutes for a fixed 10k-step run), and demand near-360° surrounding captures that are extremely difficult to obtain in practice.

**Representation limitations**: Template-free methods (BANMo, RAC) lack explicit structural priors and produce poor geometric detail; SMAL-based methods extract texture directly from images and are sensitive to mesh-image alignment accuracy; feed-forward methods (MagicPony, 3D-Fauna) sacrifice shape fidelity for generalization.

The core insight is that 4D reconstruction can be **decomposed** — an animal's motion changes frame by frame, but its appearance remains nearly constant throughout a single video. There is therefore no need to jointly optimize motion and appearance. This disentanglement yields two advantages: motion estimation can focus on temporal consistency, while appearance reconstruction can be generated feed-forward from a single image, eliminating the dependency on complete multi-view observations.

The key bridge between motion and appearance is the **VAREN model** — a high-fidelity parametric equine body model learned from thousands of 3D scans of 50 real horses (13,873 vertices, 38 joints), incorporating muscle deformation modeling, far surpassing traditional SMAL.

## Method

### Overall Architecture

4DEquine consists of two disentangled components:

- **AniMoFormer**: Spatiotemporal Transformer with post-optimization, recovering per-frame VAREN motion parameters (pose $\theta$, shape $\beta$, global translation $\gamma$) from video.
- **EquineGS**: A feed-forward network that reconstructs a high-fidelity animatable 3D Gaussian avatar from a single image.

The two components are bridged via the VAREN parametric model — AniMoFormer provides per-frame skeletal poses, and EquineGS generates a Gaussian point cloud in canonical space, driven into each frame's pose space via LBS (Linear Blend Skinning). A sliding window strategy is adopted at inference to handle videos of arbitrary length.

### Key Designs 1: AniMoFormer (Motion Recovery)

**VarenPoser dataset construction**: Training data poses a significant challenge — real 4D VAREN annotations do not exist. The authors fit the VAREN model to PFERD, a marker-based equine motion capture dataset, to obtain pose parameters; these are segmented into 600-frame clips with randomly assigned shape parameters for diversity. MV-Adapter is used to generate diverse textures, and three realistic camera trajectories (fix, dolly, orbit) are simulated to render videos. The final dataset contains 1,171 video clips at 512×512 resolution, 60 FPS.

**Spatiotemporal Transformer**:
- **Spatial Transformer**: Extracts per-frame spatial features.
- **Temporal Transformer**: Stacks spatial features from $N=16$ frames and models temporal relationships via self-attention.
- **VAREN Decoder**: Regresses per-frame pose, shape, and camera parameters.

**Post-Optimization (PO)**: While the Transformer output is temporally smooth, it may not be perfectly aligned to the 2D image. A differentiable renderer projects the 3D mesh and compares it against pseudo-GT 2D keypoints (extracted by ViTPose++) and masks (extracted by Samurai). Gradient-based optimization then fine-tunes the parameters to ensure pixel-level alignment.

### Key Designs 2: EquineGS (Appearance Reconstruction)

**Canonical point cloud initialization**: The VAREN template mesh with only 13,873 vertices is insufficiently dense; each edge midpoint is interpolated and each face is subdivided into 4, upsampling to $N_G = 55{,}486$ vertices as initial Gaussian positions.

**Dual-stream feature extraction**:
- Image stream: Pretrained DINOv3 (ViT-Large) extracts multi-scale feature maps, fused via 1×1 convolution into $\mathbf{F}_I \in \mathbb{R}^{784 \times 1024}$.
- Point cloud stream: 3D point coordinates are positionally encoded and passed through an MLP to yield $\mathbf{F}_P \in \mathbb{R}^{N_G \times 1024}$.

**DSTG Decoder (Dual-Stream Transformer Gaussian Decoder)**: Adapted from the MMDiT block in Qwen-Image, operating in three stages:
1. Global context extraction: AvgPool + MLP on image features produces a global context vector.
2. Feature fusion: Image features, point cloud features, and the global context are jointly fed into DSTG, with image information guiding the point features toward appearance alignment.
3. Attribute prediction: An MLP outputs per-Gaussian position offset $\Delta\mu$, rotation $r$, scale $s$, color $c$, and opacity $o$.

**VarenTex dataset**: The texture quality of VarenPoser is insufficient for training high-fidelity avatars, and it consists of monocular rather than multi-view data. UniTex (a multi-view diffusion model) generates reference images from VarenPoser normal maps and canonical coordinate maps (CCM) conditioned via ControlNet, synthesizing 150,000 multi-view training images at 512×512.

### Loss & Training

**AniMoFormer loss**:

$$\mathcal{L} = \lambda_{\text{varen}}\mathcal{L}_{\text{varen}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}} + \lambda_{\text{2D}}\mathcal{L}_{\text{2D}} + \lambda_{\text{3D}}\mathcal{L}_{\text{3D}}$$

where $\mathcal{L}_{\text{smooth}}$ applies an L2 constraint on the difference between adjacent-frame shape and pose parameters to ensure temporal smoothness. The post-optimization stage additionally incorporates a mask L1 loss and pose regularization.

**EquineGS loss**:

$$\mathcal{L} = \lambda_{\text{image}}\mathcal{L}_{\text{image}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$$

The image loss combines L1 and LPIPS perceptual loss to balance pixel accuracy with high-level semantic similarity; the mask loss is a silhouette L1 constraint.

## Key Experimental Results

### Main Results: Motion Estimation (Table 1)

| Method | APT36K PCK@0.05↑ | APT36K PCK@0.1↑ | APT36K Accel↓ | AiM PCK@0.05↑ | AiM PCK@0.1↑ | AiM Accel↓ | VarenPoser CD↓ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 3D-Fauna | 20.1 | 51.4 | 189.3 | 33.3 | 71.8 | 42.3 | 43.0 |
| 4D-Fauna | 25.5 | 53.5 | 177.7 | 46.5 | 74.8 | 32.7 | 38.5 |
| Dessie | 22.0 | 53.1 | 353.1 | 40.3 | 75.9 | 85.8 | 10.0 |
| GenZoo | 27.9 | 60.0 | 190.7 | 42.1 | 80.6 | 43.1 | 22.5 |
| AniMer | 44.5 | 76.6 | 130.5 | 55.5 | 87.7 | 26.2 | 15.2 |
| **AniMoFormer** | **61.8** | **83.9** | **128.6** | **84.2** | **95.3** | **21.8** | **3.4** |

AniMoFormer substantially outperforms all baselines across datasets: on AiM, PCK@0.05 reaches 84.2%, surpassing the strongest baseline AniMer by 28.7 percentage points; Chamfer Distance improves from 15.2 to 3.4, a 4.5× gain.

### Main Results: Appearance Reconstruction (Table 2)

| Method | Horse PSNR↑ | Horse SSIM↑ | Horse LPIPS↓ | Zebra PSNR↑ | Zebra SSIM↑ | Zebra LPIPS↓ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| 3D-Fauna | 12.20 | 0.7205 | 0.2782 | 12.33 | 0.6827 | 0.3318 |
| 4D-Fauna | 13.41 | 0.7550 | 0.2467 | 13.39 | 0.7157 | 0.3055 |
| GVFDiffusion | 12.68 | 0.8189 | 0.2493 | 12.26 | 0.7749 | 0.2897 |
| GART* (few-shot) | 15.42 | 0.7550 | 0.2452 | 14.31 | 0.6485 | 0.2973 |
| GART (full) | 16.19 | 0.7819 | 0.2308 | 15.21 | 0.6752 | 0.2287 |
| **4DEquine** | **15.66** | **0.8364** | **0.1720** | **15.54** | **0.7828** | **0.2000** |

4DEquine comprehensively outperforms all baselines — including fully optimized GART — on perceptual metrics (SSIM, LPIPS). On the zero-shot zebra generalization task, it leads across all three metrics. In terms of efficiency, 4DEquine requires only 11 seconds per frame (A100 GPU), compared to GART's fixed 15-minute optimization.

### Ablation Study (Tables 3 & 4)

| AniMoFormer Variant | APT36K PCK@0.05↑ | APT36K Accel↓ | AiM PCK@0.05↑ | AiM Accel↓ |
|------|:-:|:-:|:-:|:-:|
| w/o PO & Temporal | 37.1 | 134.7 | 45.1 | 30.6 |
| w/o PO | 37.7 | 129.1 | 47.8 | 25.7 |
| w/o Temporal | 57.9 | 143.2 | 82.9 | 24.7 |
| **AniMoFormer (full)** | **61.8** | **128.6** | **84.2** | **21.8** |

| EquineGS Variant | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:-:|:-:|:-:|
| w/o PO | 13.84 | 0.8103 | 0.2170 |
| w/o SubDiv | 15.76 | 0.8237 | 0.1871 |
| w/o DSTG | 15.53 | 0.8353 | 0.1733 |
| **4DEquine (full)** | **15.66** | **0.8364** | **0.1720** |

### Key Findings

- **Post-optimization is critical**: Removing PO drops PCK@0.05 from 61.8 → 57.9 on APT36K and appearance PSNR from 15.66 → 13.84, demonstrating the substantial impact of pixel-level alignment on final reconstruction quality.
- **Temporal modeling improves smoothness**: Removing the Temporal Transformer causes a notable increase in acceleration error (128.6 → 143.2).
- **Point cloud subdivision is necessary, though PSNR may be misleading**: The w/o SubDiv variant yields a marginally higher PSNR (15.76 vs. 15.66), but rendered results exhibit significant holes, as 13,873 points are insufficient to form a continuous surface.
- **DSTG outperforms standard cross-attention**: Replacing DSTG with standard cross-attention degrades all perceptual metrics.

## Highlights & Insights

1. **Elegant disentanglement**: Decomposing 4D reconstruction into independent motion and appearance subproblems — using a temporal Transformer for temporal consistency and a feed-forward network for single-image appearance generation — elegantly exploits the prior that appearance remains constant within a video.
2. **Synthetic-only training, real-world generalization**: Both modules are trained exclusively on synthetic data yet achieve SOTA on real-world benchmarks, demonstrating that high-quality synthetic data combined with strong structural priors can bridge the sim-to-real gap.
3. **Zero-shot cross-species generalization**: Trained only on horse data, the model successfully reconstructs donkeys and zebras, indicating that the model learns generalizable image features rather than memorizing training textures.
4. **Dramatic efficiency gains**: 11 seconds per frame vs. GART's 15 minutes represents over 80× speedup, without relying on multi-frame optimization.
5. **Camera trajectory design in VarenPoser**: Simulating three realistic camera motions (fix/dolly/orbit) for rendering constitutes the first large-scale 4D synthetic video dataset targeting equines.

## Limitations & Future Work

1. **Poor tail and mane reconstruction**: The VAREN model itself provides inadequate modeling of tails and manes; such complex physical structures require additional physics-based representations.
2. **Static lighting assumption**: The current method cannot handle dynamic lighting changes, which are prevalent in real outdoor scenes.
3. **Single-image appearance limitation**: EquineGS infers appearance from a single image, forcing the network to "hallucinate" textures for occluded body regions; future work could incorporate a small number of keyframes to capture distinctive markings.
4. **Dependence on the VAREN prior**: The framework is tightly coupled to VAREN, and generalizing to non-equine quadrupeds requires corresponding parametric models.
5. **Pseudo-GT quality bottleneck**: Post-optimization relies on the detection quality of ViTPose++ and Samurai, which may introduce noise under occlusion or complex poses.

## Related Work & Insights

- **VAREN [61]**: High-fidelity equine parametric body model; the geometric prior backbone of this work.
- **AniMer [22]**: Single-frame Transformer for animal pose estimation; the motion estimation baseline, extended here to a temporal version.
- **GART [13]**: Optimization-based animal avatar using 3DGS; the primary appearance reconstruction comparison method.
- **3D/4D-Fauna [17, 53]**: Template-free generalized animal reconstruction methods.
- **UniTex [19]**: Multi-view diffusion model used to generate VarenTex training data.
- Inspiration: The disentanglement paradigm combined with synthetic data can be extended to 4D reconstruction of other quadrupeds; a high-quality parametric model is a critical prerequisite for high-fidelity reconstruction.

## Rating

| Dimension | Score (1–10) | Note |
|------|:-:|------|
| Novelty | 7 | The disentanglement idea is original, but individual modules (spatiotemporal Transformer, 3DGS avatar) are largely combinations of existing techniques. |
| Technical Depth | 8 | The system is comprehensive: two datasets, two networks, and post-optimization constitute a substantial engineering contribution. |
| Experimental Thoroughness | 8 | Three datasets, multiple baselines, thorough ablations; zero-shot generalization is a highlight. |
| Writing Quality | 7 | Structure is clear, but dual-stream Transformer details rely on supplementary material. |
| Value | 7 | Clear application value for equines; 11 seconds/frame efficiency is practically deployable. |
| **Overall** | **7.5** | A systematic and complete work on equine 4D reconstruction; disentanglement and synthetic-only training are the core contributions. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](../../ICCV2025/3d_vision/shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)

</div>

<!-- RELATED:END -->
