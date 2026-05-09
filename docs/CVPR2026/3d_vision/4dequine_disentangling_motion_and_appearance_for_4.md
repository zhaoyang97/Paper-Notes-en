---
title: >-
  [Paper Note] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video
description: >-
  [CVPR 2026][3D Vision][4D Reconstruction] This work decomposes equine 4D reconstruction into two sub-tasks — motion estimation (AniMoFormer: spatiotemporal Transformer + post-optimization) and appearance reconstruction (EquineGS: feed-forward 3DGS) — bridged by the VAREN parametric model. Trained exclusively on synthetic data (VarenPoser + VarenTex), the method achieves state-of-the-art performance on real-world benchmarks APT-36K and AiM, and generalizes zero-shot to zebras and donkeys.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D Reconstruction
  - 3DGS
  - Motion Disentanglement
  - Appearance Disentanglement
  - Monocular Video
  - VAREN
date: 2026-05-08
content_hash: 63a8c78341ddc619
---

# 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video

**Conference**: CVPR 2026
**arXiv**: [2603.10125](https://arxiv.org/abs/2603.10125)
**Code**: [https://luoxue-star.github.io/4DEquine_Project_Page/](https://luoxue-star.github.io/4DEquine_Project_Page/)
**Area**: 3D Vision / Dynamic Reconstruction / Animal Modeling
**Keywords**: 4D Reconstruction, 3DGS, Motion Disentanglement, Appearance Disentanglement, Monocular Video, VAREN

## TL;DR
This work decomposes equine 4D reconstruction into two sub-tasks — motion estimation (AniMoFormer: spatiotemporal Transformer + post-optimization) and appearance reconstruction (EquineGS: feed-forward 3DGS) — bridged by the VAREN parametric model. Trained exclusively on synthetic data (VarenPoser + VarenTex), the method achieves state-of-the-art performance on real-world benchmarks APT-36K and AiM, and generalizes zero-shot to zebras and donkeys.

## Background & Motivation
Reconstructing the 4D morphology (geometry + motion + appearance) of equines from monocular video has significant value in livestock management, motion analysis, and animal welfare. Existing methods suffer from two fundamental bottlenecks: (1) general-purpose 4D reconstruction methods (e.g., Monst3R, Page-4D) fail to recover complete object geometry from partial observations; (2) SMAL-template-based optimization methods (e.g., GART, 4D-Fauna) require per-video joint optimization, which is computationally expensive and demands near-360° video coverage — a condition rarely met in real-world settings. A solution that is both efficient and robust to sparse viewpoints is therefore needed.

## Core Problem
How to efficiently reconstruct a 4D representation (geometry + motion + appearance) of equines from ordinary monocular video (non-360° capture) while avoiding costly per-video optimization? The core challenge lies in the fact that joint optimization of motion and appearance is both complex and highly sensitive to viewpoint coverage.

## Method

### Overall Architecture
The 4D reconstruction is decomposed into two independent sub-tasks: **motion estimation** (AniMoFormer: spatiotemporal Transformer + post-optimization → outputs per-frame VAREN parameter sequences) and **appearance reconstruction** (EquineGS: feed-forward network from a single image → outputs a canonical 3DGS avatar). The VAREN parametric horse model serves as the bridge — motion provides per-frame pose/shape parameters, while appearance generates a Gaussian point cloud in canonical space that is deformed to each frame's pose via LBS.

### Key Designs
1. **AniMoFormer (Motion Estimation)**: Two-stage design. Stage one is a spatiotemporal Transformer — a Spatial Transformer (ViT-H backbone, pretrained from AniMer) extracts per-frame features, followed by a Temporal Transformer that models temporal relationships over a 16-frame window via self-attention to regress VAREN pose/shape/camera parameters. Stage two is Post-Optimization — a differentiable renderer projects the 3D mesh to 2D, and the result is aligned with 2D keypoints detected by ViTPose++ and masks extracted by Samurai (two steps: first optimizing all parameters with emphasis on keypoint alignment, then freezing pose parameters to emphasize mask alignment).
2. **EquineGS (Appearance Reconstruction)**: A feed-forward network that generates an animatable canonical 3DGS avatar from a single image. The VAREN template mesh is subdivided (13,873 → 55,486 vertices) to initialize Gaussian positions. Dual-stream feature extraction is then applied — DINOv3 ViT-L extracts image features while a Point Transformer encodes 3D point features. A DSTG (Dual-Stream Transformer Gaussian) decoder fuses both feature streams to predict per-Gaussian displacement, rotation, scale, color, and opacity.
3. **Synthetic Dataset Creation**: VarenPoser (for motion training) — VAREN is fitted to PFERD horse motion capture data; MV-Adapter generates textures; three camera trajectories (fixed/orbiting/zoom) are simulated, yielding 1,171 video clips. VarenTex (for appearance training) — UniTex multi-view diffusion model generates consistent multi-view images, totaling 150K images with higher appearance quality than VarenPoser.

### Loss & Training
- AniMoFormer training loss: $\mathcal{L} = \lambda_{varen}\mathcal{L}_{varen} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{2D}\mathcal{L}_{2D} + \lambda_{3D}\mathcal{L}_{3D}$, where the smoothness term constrains parameter variation between adjacent frames.
- Post-optimization loss: $\mathcal{L} = \lambda_{2D}\mathcal{L}_{2D} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{mask}\mathcal{L}_{mask}$, with two stages emphasizing keypoint and mask alignment respectively.
- EquineGS training loss: $\mathcal{L} = \lambda_{image}(\|L1\| + LPIPS) + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{reg}\mathcal{L}_{reg}$
- AniMoFormer is trained on a single RTX 4090 for 10 hours (100K steps); EquineGS is trained on 8×RTX 4090 for 3 days (100K steps).
- Inference speed: 11 seconds/frame (A100), compared to GART's fixed 15 minutes/video.

## Key Experimental Results

| Dataset | Metric | 4DEquine | Prev. SOTA (AniMer) | Gain |
|--------|------|----------|-----------------|------|
| APT-36K | PCK@0.05 | **61.8** | 44.5 | +38.9% |
| APT-36K | PCK@0.1 | **83.9** | 76.6 | +9.5% |
| AiM | PCK@0.05 | **84.2** | 55.5 | +51.7% |
| AiM | Accel↓ | **21.8** | 26.2 | -16.8% |
| VarenPoser | CD↓ | **3.4** | 15.2 | -77.6% |

| Dataset | Metric | 4DEquine | GART (Full Opt.) | Few-shot GART |
|--------|------|----------|-------------|--------------|
| AiM-Horse | SSIM↑ | **0.8364** | 0.7819 | 0.7550 |
| AiM-Horse | LPIPS↓ | **0.1720** | 0.2308 | 0.2452 |
| AiM-Zebra (zero-shot) | PSNR↑ | **15.54** | 15.21 | 14.31 |
| AiM-Zebra (zero-shot) | LPIPS↓ | **0.2000** | 0.2287 | 0.2973 |

### Ablation Study
- Removing post-optimization (PO): PCK@0.05 drops to 37.7 on APT-36K; rendering quality degrades from LPIPS 0.172 → 0.217 — PO is critical for pixel-level alignment.
- Removing the spatiotemporal Transformer: acceleration error (Accel) increases notably — temporal modeling contributes most to motion smoothness.
- Removing mesh subdivision (SubDiv): PSNR is marginally higher but visualizations exhibit severe holes — low-resolution point clouds are insufficient to form a continuous surface.
- DSTG vs. standard cross-attention: DSTG consistently outperforms on SSIM/LPIPS — dual-stream fusion is superior to unidirectional cross-attention.
- Window size $N$: performance improves continuously from 4 → 8 → 16 frames; $N=32$ causes OOM.

## Highlights & Insights
- **Elegant disentanglement**: Decomposing 4D reconstruction into motion and appearance sub-tasks, bridged by the parametric VAREN model, enables independent training and inference for each component.
- **Trained purely on synthetic data with zero-shot generalization to real data and unseen species (donkeys, zebras)** — demonstrating that synthetic data combined with a strong parametric prior can effectively bridge the domain gap.
- Feed-forward appearance reconstruction from a single image eliminates the multi-frame optimization bottleneck of methods such as GART.
- The two-stage post-optimization design (keypoint alignment followed by mask alignment) as an optional post-processing step is worth adopting in related work.

## Limitations & Future Work
- The VAREN model does not adequately represent the physical geometry of tails and manes, limiting reconstruction quality in those regions.
- Dynamic illumination changes cannot be handled.
- EquineGS uses only a single frame as input and fails on severely occluded or truncated frames — future work could incorporate multi-keyframe fusion.
- The method is restricted to equines; generalization to other quadrupeds (cats, dogs, etc.) would require corresponding parametric models.
- Training cost is substantial (8×RTX 4090, 3 days), making it unsuitable for resource-constrained settings.

## Related Work & Insights
- **vs. GART (CVPR 2024)**: GART is a per-video optimization method requiring good viewpoint coverage, with a fixed 15-minute optimization per video. 4DEquine is a feed-forward method with 11-second per-frame inference. GART achieves slightly higher PSNR (16.19 vs. 15.66) but is inferior on perceptual metrics (SSIM/LPIPS), and falls consistently behind on the zero-shot zebra scenario.
- **vs. 4D-Fauna (ICCV 2025)**: 4D-Fauna is a template-free general method capable of reconstructing 100+ quadruped species but lacks geometric precision. 4DEquine leverages the VAREN prior for higher geometric fidelity, at the cost of being limited to equines.
- **vs. AniMer**: AniMer performs single-frame estimation without temporal modeling, resulting in severe motion jitter. 4DEquine's spatiotemporal Transformer achieves 39–52% improvement on PCK@0.05.

The disentanglement + parametric prior paradigm is generalizable to other domains with established parametric models (e.g., SMPL for human bodies, MANO for hands). The synthetic-to-real pipeline (MV-Adapter for texture generation + UniTex for multi-view synthesis) constitutes a reusable general framework. The relevance to the reviewer's research direction is limited, as this work targets a vertical application in species-specific animal reconstruction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The motion–appearance disentanglement strategy is conceptually clear, though individual modules (spatiotemporal Transformer, feed-forward 3DGS) are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple datasets, multiple baselines, detailed ablations, zero-shot generalization, and failure case analysis constitute a highly complete evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure is clear, though the paper is lengthy; supplementary material is extensive.
- **Value**: ⭐⭐⭐ — Practically valuable within the equine reconstruction domain, but with a relatively narrow scope of application.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)

<!-- RELATED:END -->
