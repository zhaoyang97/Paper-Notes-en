---
title: >-
  [Paper Note] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion
description: >-
  [CVPR 2026][3D Vision][Scene Change Detection] This paper presents the first scene change detection (SCD) method that simultaneously achieves online inference, pose-agnosticism, label-free operation, and multi-view consistency. By replacing hard-threshold heuristics with a self-supervised fusion (SSF) loss that integrates pixel-level and feature-level change cues into a 3DGS change representation, the proposed approach surpasses all existing offline methods in detection accuracy while operating in real time at over 10 FPS.
tags:
  - CVPR 2026
  - 3D Vision
  - Scene Change Detection
  - 3D Gaussian Splatting
  - Online Inference
  - Self-Supervised Fusion
  - Scene Update
date: 2026-05-08
content_hash: 32a33ffdf8aa8a46
---

# Changes in Real Time: Online Scene Change Detection with Multi-View Fusion

**Conference**: CVPR 2026
**arXiv**: [2511.12370](https://arxiv.org/abs/2511.12370)
**Code**: [https://chumsy0725.github.io/O-SCD/](https://chumsy0725.github.io/O-SCD/)
**Area**: 3D Vision
**Keywords**: Scene Change Detection, 3D Gaussian Splatting, Online Inference, Self-Supervised Fusion, Scene Update

## TL;DR

This paper presents the first scene change detection (SCD) method that simultaneously achieves online inference, pose-agnosticism, label-free operation, and multi-view consistency. By replacing hard-threshold heuristics with a self-supervised fusion (SSF) loss that integrates pixel-level and feature-level change cues into a 3DGS change representation, the proposed approach surpasses all existing offline methods in detection accuracy while operating in real time at over 10 FPS.

## Background & Motivation

**Background**: Scene change detection (SCD) is a core task in scene understanding, with applications in environmental monitoring, infrastructure inspection, and damage assessment. Recent methods leverage NeRF and 3DGS to build 3D scene representations for pose-agnostic SCD.

**Limitations of Prior Work**:
   - The strongest SCD methods (e.g., MV3DCD, GeSCD) are **offline** — they require all pre- and post-event observations to be collected before inference, making them unsuitable for real-time decision-making scenarios.
   - Existing online methods achieve substantially lower accuracy than offline counterparts, and most fail to maintain real-time performance (<1 FPS).
   - MV3DCD relies on hard thresholds and intersection-based heuristics to fuse change cues, which tends to discard subtle yet important change signals.

**Key Challenge**: Online settings demand real-time, frame-by-frame change detection with cross-view consistency, yet existing methods sacrifice either accuracy (online methods) or real-time capability (offline methods).

**Goal**: (a) How to perform online, real-time change inference? (b) How to avoid information loss caused by hard thresholding? (c) How to update scene representations efficiently?

**Key Insight**: A 3DGS-based change representation serves as a persistent cross-view memory, coupled with a self-supervised loss that automatically learns to fuse multi-view change cues. This is complemented by a lightweight PnP-based pose estimator and change-guided selective scene updating.

**Core Idea**: A self-supervised fusion loss replaces hard-threshold heuristics, allowing change information to accumulate and propagate naturally within the 3DGS representation. Scene updates are further accelerated by reconstructing only the changed regions.

## Method

### Overall Architecture

The system comprises three stages: (1) offline construction of a 3DGS reference scene representation; (2) online processing — for each query frame, estimate camera pose → extract change cues → fuse into the change representation → infer change mask; (3) after all observations are collected, selectively update the scene representation.

### Key Designs

1. **Lightweight PnP Pose Estimation**

    - **Function**: Registers each query frame into the reference scene coordinate system.
    - **Mechanism**: XFeat is used to extract keypoints and descriptors from reference images, which are pre-triangulated into a 3D point set. For each query frame, descriptors are extracted and matched against the top-4 reference frames; camera pose is estimated via PnP+RANSAC from 2D–3D correspondences and refined with GPU-parallelized miniBA.
    - **Design Motivation**: Pose estimation operates over a fixed-size reference frame set, achieving $O(1)$ complexity with no drift accumulation — significantly faster than methods such as SplatPose that optimize pose directly against the 3DGS representation.

2. **Pixel-Level and Feature-Level Change Cue Extraction**

    - **Function**: Extracts complementary change signals from query–render image pairs.
    - **Mechanism**:
        - Pixel-level cue: $C_{\text{pixel}}^k = (1-\lambda)L_1 + \lambda L_{\text{D-SSIM}}$, capturing fine-grained appearance differences but sensitive to illumination, reflections, and shadows.
        - Feature-level cue: dense feature maps extracted via SAM2-Tiny, $C_{\text{feature}}^k = \sum_i |f_{\text{inf}}^{k,i} - f_{\text{ren}}^{k,i}|$, more robust to distractors but may miss subtle changes between semantically similar objects.
        - Final combination: $C^k = C_{\text{pixel}}^k + C_{\text{feature}}^k$, preserving the complementary strengths of both cues through simple addition.
    - **Design Motivation**: MV3DCD hard-thresholds each cue separately and takes their intersection, discarding valid change evidence captured by only one cue. Simple addition followed by the SSF loss enables more effective information integration.

3. **Self-Supervised Fusion (SSF) Loss**

    - **Function**: Fuses multi-view change cues into the 3DGS change representation and infers a multi-view-consistent change mask.
    - **Mechanism**: The change representation $\mathcal{R}_{\text{change}}$ is initialized from the reference 3DGS (color parameters discarded; learnable change parameter $c$ introduced). For each query frame, the SSF loss optimizes $\mathcal{R}_{\text{change}}$ for $n=16$ steps:
    $$L_{\text{SSF}} = C^i \odot (1 - \tilde{M}^i) + \log(1 + \text{mean}(\tilde{M}^i)^2)$$
    The first term encourages high predicted change probability in regions with strong change cues; the second term regularizes against the trivial solution $\tilde{M}=1$. At each step, a historical frame $i$ is randomly sampled, with 1/3 probability biased toward the latest frame $k$.
    - **Design Motivation**: $\mathcal{R}_{\text{change}}$ acts as persistent memory, automatically accumulating change information from all observed viewpoints while enforcing 3D consistency — avoiding the information loss inherent in hard thresholding and intersection operations.

4. **Change-Guided Selective Scene Update**

    - **Function**: Efficiently updates the reference scene representation to reflect the current scene state.
    - **Mechanism**: The refined change mask is used to reconstruct only the changed regions: $\hat{I}_{\text{inf}}^k = I_{\text{inf}}^k \odot M_{\text{refined}}^k$. The reconstructed change-region Gaussians $\mathcal{R}_{\text{change}}^*$ are merged with the unchanged reference Gaussians $\mathcal{R}_{\text{ref}}^*$, followed by one round of constrained global optimization with adaptive density control applied only to Gaussians corresponding to changed pixels.
    - **Design Motivation**: This avoids full scene reconstruction after each inspection by reusing high-quality Gaussians from unchanged regions, achieving rendering speeds exceeding 400 FPS and completing the overall update in tens of seconds — 8–13× faster than reconstruction from scratch.

### Loss & Training

- **SSF Loss**: $L_{\text{SSF}} = C^i \odot (1 - \tilde{M}^i) + \log(1 + \text{mean}(\tilde{M}^i)^2)$
- Reference scene construction: standard 3DGS (accelerated via Speedy-Splat) with SfM-based pose estimation.
- Online inference: 16-step optimization of the change representation per frame, with sampling biased toward the latest frame.
- Scene update: standard 3DGS optimization pipeline with constrained adaptive density control.

## Key Experimental Results

### Main Results

SCD results on the PASLCD dataset (10 indoor/outdoor room-scale scenes, 20 instances):

| Method | Label-Free | Pose-Agnostic | Multi-View | Online | mIoU ↑ | F1 ↑ | Speed |
|--------|-----------|--------------|-----------|--------|--------|------|-------|
| GeSCD (offline) | ✓ | ✗ | ✗ | ✗ | 0.477 | 0.611 | 298s |
| MV3DCD (offline) | ✓ | ✓ | ✓ | ✗ | 0.478 | 0.628 | 479s |
| **Ours (offline)** | ✓ | ✓ | ✓ | ✗ | **0.552** | **0.694** | 156s |
| SplatPose+ (online) | ✓ | ✓ | ✗ | ✓ | 0.237 | 0.358 | <1 FPS |
| CS+CYWS2D (online) | ✗ | ✗ | ✗ | ✓ | 0.243 | 0.360 | 8.2 FPS |
| **Ours (online)** | ✓ | ✓ | ✓ | ✓ | **0.486** | **0.638** | 11.2 FPS |

Scene representation update (PASLCD + CL-Splats):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Time (s) ↓ |
|--------|--------|--------|---------|-----------|
| 3DGS (from scratch) | 22.21 | 0.756 | 0.243 | 550 |
| 3DGS-LM | 22.26 | 0.756 | 0.242 | 340 |
| CLNeRF | 22.27 | 0.624 | 0.391 | 451 |
| **Ours** | **23.70** | **0.787** | **0.249** | **42** |

### Ablation Study

| Variant | mIoU ↑ | F1 ↑ |
|---------|--------|------|
| Full model | 0.486 | 0.638 |
| w/o $L_1$ | 0.320 | 0.464 |
| w/o $L_{\text{D-SSIM}}$ | 0.447 | 0.620 |
| $C_{\text{pixel}}$ only | ✗ (no convergence) | ✗ |
| $C_{\text{feature}}$ only | ✗ (no convergence) | ✗ |
| w/o regularization term | ✗ (trivial solution) | ✗ |
| MV3DCD hard threshold + intersection | 0.350 | 0.495 |

### Key Findings

- **Both pixel-level and feature-level cues are indispensable**: Using either cue alone prevents the SSF loss from converging, confirming that the two modalities provide complementary supervision signals.
- **SSF Loss vs. hard thresholding**: Replacing the SSF loss with MV3DCD's hard-threshold intersection strategy reduces F1 from 0.638 to 0.495, demonstrating the clear advantage of learned fusion over heuristic aggregation.
- **Online model surpasses offline SOTA**: The online variant achieves mIoU of 0.486, exceeding the strongest offline competitor MV3DCD (0.478) — a particularly noteworthy result.
- **Speed–accuracy trade-off**: Reducing the number of fusion iterations allows throughput to be tuned between 11 and 20 FPS, with only a 3.6% drop in F1.
- **Scene update is 8–13× faster than reconstruction from scratch**: Reusing Gaussians from unchanged regions is the key enabler, while also yielding superior PSNR.

## Highlights & Insights

- **3DGS as persistent change memory**: Embedding change parameters into 3DGS primitives allows multi-view change information to accumulate and propagate naturally in 3D space. This design is both elegant and effective, and is transferable to any task requiring temporal information fusion within 3DGS.
- **Elegant SSF loss design**: A two-term loss achieves end-to-end multi-modal cue fusion, multi-view consistency, and trivial-solution prevention — all without any manual annotation. The key insight is to let the loss function *learn* to fuse, rather than relying on hand-crafted aggregation rules.
- **Selective reconstruction and merging for scene update**: Changed regions require only a small number of Gaussians to model, enabling rendering at >400 FPS and greatly accelerating optimization. This provides a practical solution for long-term scene monitoring.

## Limitations & Future Work

- XFeat matching may fail under extreme appearance changes (e.g., seasonal variation), adversely affecting pose estimation.
- The current approach uses only SAM2-Tiny features as semantic cues; stronger vision foundation models may further improve detection accuracy.
- The scene update strategy assumes changes within a single inspection are static, making it unsuitable for continuously dynamic scenes.
- Detection of small-object-level changes remains an area for further improvement.

## Related Work & Insights

- **vs. MV3DCD**: The most direct competitor. MV3DCD fuses cues via hard thresholds and intersection heuristics; this work replaces that with a learnable SSF loss, improving mIoU by approximately 15%, with the online variant also surpassing MV3DCD's offline performance.
- **vs. SplatPose/SplatPose+**: These methods optimize camera pose directly against the 3DGS representation, resulting in very low throughput (<1 FPS). The proposed PnP-based approach operates at $O(1)$ complexity with no drift accumulation.
- **vs. CL-Splats/GaussianUpdate**: Competing approaches for scene updating that require substantially longer training times. The proposed selective reconstruction strategy is simpler and more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First method integrating online inference, pose-agnosticism, label-free operation, and multi-view consistency for SCD.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons against online and offline baselines, detailed ablations, and thorough speed analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical structure, rich figures and tables, with explicit problem–solution correspondence throughout.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to robotic inspection and long-term scene monitoring scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] r4det 4d radar camera fusion 3d detection](r4det_4d_radar_camera_fusion_3d_detection.md)
- [\[CVPR 2026\] Coherent Human-Scene Reconstruction from Multi-Person Multi-View Video in a Single Pass](coherent_humanscene_reconstruction_from_multiperso.md)
- [\[CVPR 2026\] EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding](embodiedsplat_online_feed-forward_semantic_3dgs_for_open-vocabulary_3d_scene_und.md)
- [\[CVPR 2026\] VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection](vggt-det_mining_vggt_internal_priors_for_sensor-geometry-free_multi-view_indoor_.md)
- [\[CVPR 2026\] ForgeDreamer: Industrial Text-to-3D Generation with Multi-Expert LoRA and Cross-View Hypergraph](forgedreamer_industrial_text-to-3d_generation_with_multi-expert_lora_and_cross-v.md)

</div>

<!-- RELATED:END -->
