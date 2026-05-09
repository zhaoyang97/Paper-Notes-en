---
title: >-
  [Paper Note] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View
description: >-
  [ICCV 2025][Video Understanding][panoramic optical flow] This paper proposes PriOr-Flow, a dual-branch framework that leverages the low-distortion prior of orthogonal views to compensate for severe distortions in polar regions of ERP panoramic images, achieving significant improvements in panoramic optical flow estimation — reducing EPE by 30.0% on MPFDataset and 29.6% on FlowScape.
tags:
  - ICCV 2025
  - Video Understanding
  - panoramic optical flow
  - equirectangular projection
  - dual-branch
  - distortion compensation
  - orthogonal view
date: 2026-05-08
content_hash: a5abd40e952a50f0
---

# PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View

**Conference**: ICCV 2025
**arXiv**: [2506.23897](https://arxiv.org/abs/2506.23897)
**Code**: [GitHub](https://github.com/longliangLiu/PriOr-Flow)
**Area**: Video Understanding / Optical Flow Estimation / Panoramic Vision
**Keywords**: panoramic optical flow, equirectangular projection, dual-branch, distortion compensation, orthogonal view

## TL;DR

This paper proposes PriOr-Flow, a dual-branch framework that leverages the low-distortion prior of orthogonal views to compensate for severe distortions in polar regions of ERP panoramic images, achieving significant improvements in panoramic optical flow estimation — reducing EPE by 30.0% on MPFDataset and 29.6% on FlowScape.

## Background & Motivation

Panoramic optical flow estimation aims to compute dense pixel motion fields from consecutive frames of panoramic video, with important applications in autonomous driving, video frame interpolation, and 3D reconstruction. Equirectangular projection (ERP) is the dominant representation for panoramic images, but mapping the sphere to a plane introduces severe geometric distortion, particularly in the **polar regions** (near the north and south poles), following a cosine decay pattern.

Existing methods fall into three categories:

**Weight transformation methods** (LiteFlowNet360, OmniFlowNet): adapt convolution weights for ERP but introduce additional computational overhead.

**Tangent plane methods** (TanImg): project the sphere onto multiple tangent planes, but suffer from cross-plane discontinuities and motion inconsistencies.

**ERP-based methods** (PanoFlow, SLOF, MPF-Net): directly process ERP images using techniques such as deformable convolutions to compensate for distortion.

**Core Problem**: None of the above methods explicitly address the severe distortion in polar regions — the area of highest distortion in ERP (pixels are overstretched), leading to noisy cost volumes and large optical flow estimation errors.

**Core Insight**: Rotating the panoramic image 90° around the x-axis on the sphere yields an **orthogonal view** whose distortion distribution is complementary to the original — regions of high distortion in the original polar areas become low-distortion regions in the orthogonal view. This complementarity enables the low-distortion information from the orthogonal view to compensate for polar-region errors in the original view.

## Method

### Overall Architecture

PriOr-Flow adopts a **dual-branch structure** that can be integrated into iterative optical flow networks such as RAFT, GMA, and SKFlow. Using PriOr-RAFT as the representative instance:

- **Primitive Branch**: processes the original ERP frame pair and constructs a primitive cost volume pyramid.
- **Orthogonal Branch**: rotates the ERP frame pair by 90° to obtain the orthogonal view, independently extracts features, and constructs an orthogonal cost volume pyramid.
- **DCCL operator**: jointly retrieves correlation information from both cost volumes during iterative updates.
- **ODDC module**: performs confidence-guided motion feature fusion to compensate polar-region optical flow in the primitive branch using motion information from the orthogonal branch.

### Key Designs

#### 1. Orthogonal View Generation

Implemented via a spherical rotation operation $\mathcal{R}$:
- Map ERP pixel coordinates $\mathbf{x}$ to 3D Cartesian coordinates $P(\mathbf{x})$.
- Apply a 90° rotation around the x-axis: $R_x(90°) \cdot P(\mathbf{x})$.
- Re-project back to the ERP plane using bilinear interpolation: $I^o = T_p^o(I^p)$.

Key property of the orthogonal view: its distortion distribution is **complementary** to that of the original view — polar regions become low-distortion, and equatorial regions become high-distortion.

#### 2. Dual-Cost Collaborative Lookup (DCCL)

Conventional methods retrieve correlations from a single cost volume, where distortion noise in polar regions severely degrades retrieval quality. DCCL performs correlation lookup on a unified sphere:

- Locate the correspondence $\hat{\mathbf{x}}^p$ from the current flow estimate $\mathcal{F}^p$ (with mod W for horizontal boundary continuity).
- Define a local grid $\mathcal{N}(\hat{\mathbf{x}}^p)_r^p$ around the correspondence.
- Index the primitive cost volume pyramid to obtain $\mathcal{C}^p$.
- Transform the local grid to the orthogonal coordinate system via spherical rotation: $\mathcal{N}(\hat{\mathbf{x}}^p)_r^o$.
- Index the orthogonal cost volume pyramid to obtain $\mathcal{C}^o$, then convert back to the primitive format as $\mathcal{C}^{o2p}$.

Both correlation streams are jointly fed into a ConvGRU to guide flow recovery.

#### 3. Ortho-Driven Distortion Compensation (ODDC)

ODDC further exploits the low-distortion prior of the orthogonal view to compensate polar-region optical flow reconstruction:

- **Confidence computation**: group-wise correlation is used to compute confidence maps for the primitive flow $G^p$ and the transformed orthogonal flow $G^{o2p}$.
- **Motion feature encoding**: three shallow encoders — $\text{En}_c$ (correlation), $\text{En}_g$ (confidence), $\text{En}_f$ (flow conditioning).
- **Fusion strategy**: $m^p = [\text{En}_c(\mathcal{C}^p + \mathcal{C}^{o2p}), \text{En}_g([G^p, G^{o2p}]), \text{En}_f([\mathcal{F}^p, \mathcal{F}^{o2p}]), \mathcal{F}^p, \mathcal{F}^{o2p}]$
- The fused motion features are fed into the ConvGRU to update the hidden state and decode residual flow.

### Loss & Training

- **Spherical-weighted L1 loss**: accounts for the non-uniform sampling of ERP projection by multiplying the per-pixel L1 loss by the corresponding spherical area weight $\omega^j$.
- **Dual-branch joint supervision**: the primitive branch is supervised with the original GT; the orthogonal branch with the rotated GT.
- **Exponentially increasing weights**: weights increase along the iteration direction as $\gamma^{N-i}$ ($\gamma=0.8$).
- **Total loss**: $\mathcal{L} = \mathcal{L}_p + \mathcal{L}_o$

Training details:
- AdamW optimizer, gradient clipping $[-1, 1]$, one-cycle learning rate schedule.
- Initial learning rate $1\text{e-}4$; initialized from RAFT pretrained weights on FlyingThings.
- MPFDataset: batch size 4, 60k training steps.
- FlowScape: batch size 6, 100k training steps.
- Number of iterations: 12 for both training and inference.

## Key Experimental Results

### Main Results

Comparison on MPFDataset and FlowScape benchmarks:

| Method | Baseline | MPF-EFT EPE | MPF-EFT SEPE | MPF-City EPE | MPF-All EPE | FlowScape-All EPE | FlowScape-All SEPE |
|--------|----------|-------------|--------------|--------------|-------------|-------------------|--------------------|
| SphereNet+RAFT | RAFT | 13.2 | 15.7 | 8.28 | 10.7 | 12.9 | 21.0 |
| TanImg+RAFT | RAFT | 4.38 | 9.52 | 3.13 | 3.76 | 18.3 | 25.3 |
| SLOF | RAFT | 4.98 | 8.20 | 1.35 | 3.17 | 7.59 | 5.79 |
| PanoFlow | RAFT | - | - | - | - | 3.38 | 4.78 |
| PanoFlow | CSFlow | - | - | - | - | 3.31 | 4.44 |
| **PriOr-RAFT** | **RAFT** | **3.30** | **6.23** | **1.13** | **2.22** | **2.33** | **3.49** |

- On MPFDataset: EPE reduced by **30.0%** (vs. SLOF), SEPE reduced by **20.9%**.
- On FlowScape: EPE reduced by **29.6%**, SEPE reduced by **21.4%** (vs. PanoFlow).

Polar vs. equatorial region performance (FlowScape):

| Method | Equatorial EPE | Equatorial SEPE | Polar EPE | Polar SEPE | All EPE | All SEPE |
|--------|----------------|-----------------|-----------|------------|---------|----------|
| PanoFlow | 0.52 | 2.87 | 6.25 | 6.68 | 3.38 | 4.78 |
| PriOr-RAFT | 0.53 | 2.94 | **4.13** | **4.03** | **2.33** | **3.49** |

Polar EPE improves by **39.7%**, while equatorial performance remains on par.

### Ablation Study

#### Module Effectiveness Ablation (MPFDataset EFT Scene)

| Model | Orthogonal View | DCCL | ODDC | Polar EPE | Polar SEPE | All EPE | All SEPE |
|-------|----------------|------|------|-----------|------------|---------|----------|
| RAFT Baseline | ✗ | ✗ | ✗ | 7.90 | 8.56 | 4.49 | 7.43 |
| +Ortho+DCCL | ✓ | ✓ | ✗ | 7.56 | 8.16 | 4.32 | 7.20 |
| **PriOr-RAFT (Full)** | ✓ | ✓ | ✓ | **5.57** | **6.47** | **3.30** | **6.23** |

The ODDC module contributes a **26.3%** improvement in polar EPE, making it the most critical component.

#### Generalizability Ablation

| Model | EPE | EPE Reduction | Runtime |
|-------|-----|--------------|---------|
| RAFT | 4.49 | - | 0.07s |
| PriOr-RAFT (4-iter) | 3.89 | ↓13.4% | 0.10s |
| PriOr-RAFT | 3.30 | ↓26.5% | 0.20s |
| GMA | 4.26 | - | 0.07s |
| PriOr-GMA (4-iter) | 3.51 | ↓17.6% | 0.10s |
| PriOr-GMA | 3.25 | ↓23.7% | 0.20s |
| SKFlow | 3.79 | - | 0.12s |
| PriOr-SKFlow (4-iter) | 3.33 | ↓12.1% | 0.13s |
| PriOr-SKFlow | 3.19 | ↓15.8% | 0.30s |

PriOr-GMA achieves a 17.6% improvement with only 4 iterations, at virtually no additional inference cost.

### Key Findings

1. **Orthogonal view selection**: 90° rotation around the x-axis performs best (EPE 3.30); y-axis rotation disrupts polar continuity (EPE 3.55); 45° rotation yields a smaller low-distortion region (EPE 3.41).
2. **Iteration count**: PriOr-RAFT surpasses the baseline's 12-iteration result with only 3 iterations, demonstrating that orthogonal branch motion features accelerate convergence.
3. **Confidence type**: dynamic confidence based on warp + group-wise correlation (EPE 3.30) outperforms a fixed distortion-map confidence (EPE 3.34), as it better reflects actual flow reliability.

## Highlights & Insights

1. **Elegant exploitation of complementary distortion**: The polar high-distortion problem in ERP has long been a challenge in panoramic vision. The proposed 90° spherical rotation to produce a distortion-complementary orthogonal view is both intuitive and elegant.
2. **Strong generalizability**: The DCCL and ODDC modules can be plug-and-play integrated into multiple iterative optical flow networks (RAFT, GMA, SKFlow), consistently yielding significant improvements.
3. **Accelerated convergence**: Incorporating orthogonal branch information allows the model to reach or exceed baseline accuracy with fewer iterations, enabling accuracy gains without sacrificing inference speed.
4. **Geometrically aligned retrieval on the sphere**: DCCL performs coordinate transformation and correlation lookup on a unified sphere, ensuring geometric alignment between the two cost volumes.

## Limitations & Future Work

1. **Slight degradation in equatorial regions**: Due to the polar–equatorial distortion trade-off, PriOr-RAFT's equatorial EPE is marginally worse than PanoFlow (0.53 vs. 0.52); adaptive region-weighting strategies may address this.
2. **Computational overhead**: The dual-branch structure approximately doubles inference time (0.07s → 0.20s), which may be a bottleneck in latency-sensitive applications.
3. **Quantitative evaluation limited to synthetic data**: Real-scene evaluation is qualitative only due to the lack of real-world panoramic optical flow GT datasets; generalization requires further validation.
4. **Fixed rotation angle**: Only a single 90° rotation is used; the potential gains from multi-angle orthogonal views remain unexplored.
5. **Polar boundary topology not explicitly handled**: The special topological structure at the upper and lower boundaries (poles) of ERP may require additional treatment.

## Related Work & Insights

- **RAFT** (2020): The classic iterative optical flow framework upon which this work directly builds.
- **PanoFlow** (2023): Uses deformable convolutions and optical flow distortion augmentation, but does not explicitly handle polar regions.
- **SLOF** (2022): Employs twin representation learning and rotation augmentation with cross-view flow similarity constraints.
- **Inspiration**: The idea of multi-view distortion complementarity generalizes to other panoramic vision tasks (depth estimation, semantic segmentation, etc.), where different projection views can compensate each other's distortions to improve overall performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The complementary distortion concept via orthogonal view is concise and effective; the DCCL+ODDC fusion mechanism is well-designed.
- **Technical Quality**: ⭐⭐⭐⭐ — Comprehensive ablations, multi-baseline integration validates generalizability, with sufficient quantitative and qualitative results.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple datasets, baselines, and ablation dimensions; however, real-scene quantitative evaluation is absent.
- **Practicality**: ⭐⭐⭐⭐ — Plug-and-play, open-sourced, with tangible contributions to the panoramic optical flow field.
- **Overall**: ⭐⭐⭐⭐ — Elegant idea, significant results (30% EPE↓), thorough ablations; a solid contribution to panoramic optical flow estimation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](../../CVPR2026/video_understanding/u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)

<!-- RELATED:END -->
