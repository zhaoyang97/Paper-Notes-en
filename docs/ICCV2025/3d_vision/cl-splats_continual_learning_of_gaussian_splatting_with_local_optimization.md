---
title: >-
  [Paper Note] CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization
description: >-
  [ICCV 2025][3D Vision][Gaussian Splatting] This paper proposes CL-Splats, a continual learning framework built on 3D Gaussian Splatting that incrementally updates scene reconstructions from sparse novel views via DINOv2-…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Gaussian Splatting"
  - "continual learning"
  - "local optimization"
  - "scene update"
  - "change detection"
date: 2026-05-08
content_hash: 2893112999d3b101
---

# CL-Splats: Continual Learning of Gaussian Splatting with Local Optimization

**Conference**: ICCV 2025  
**arXiv**: [2506.21117](https://arxiv.org/abs/2506.21117)  
**Code**: [https://cl-splats.github.io/](https://cl-splats.github.io/)  
**Area**: 3D Vision / Scene Reconstruction  
**Keywords**: Gaussian Splatting, continual learning, local optimization, scene update, change detection

## TL;DR

This paper proposes CL-Splats, a continual learning framework built on 3D Gaussian Splatting that incrementally updates scene reconstructions from sparse novel views via DINOv2-based change detection, 2D-to-3D mask lifting, and sphere-constrained local optimization. CL-Splats substantially outperforms CL-NeRF and related methods on both synthetic and real scenes (PSNR: 40.1 vs. 30.1 dB) while supporting applications such as historical state recovery and concurrent updates.

## Background & Motivation

In applications such as robotics, mixed reality, and embodied AI, scenes change continuously over time (objects are moved, added, or removed), necessitating efficient updates to 3D scene representations. The naive approach of re-running 3DGS/NeRF from scratch discards existing reconstruction information and requires re-capturing the entire scene.

Existing continual learning methods exhibit notable shortcomings:
- **CL-NeRF**: Based on implicit NeRF representations, it suffers from catastrophic forgetting, cannot accurately recover historical states, and requires camera poses for unchanged regions.
- **CLNeRF**: Requires frames outside the changed region, is insufficiently efficient, and renders at <<1 FPS.
- **Direct 3DGS retraining**: Destroys existing reconstructions in unobserved regions under sparse viewpoints.

The core objective of this work is to efficiently and accurately update an existing 3DGS reconstruction using only a small number of new images capturing the **locally changed region**, while preserving the integrity of unchanged areas.

## Method

### Overall Architecture

CL-Splats operates in three stages:
1. **2D Change Detection**: DINOv2 features are used to compare new images against rendered images from the existing reconstruction, producing 2D change masks.
2. **3D Mask Lifting**: The 2D masks are projected into 3D space via majority voting to identify which Gaussians belong to changed regions, and new points are sampled for newly appeared objects.
3. **Locally Constrained Optimization**: Only Gaussians within the changed region are optimized; sphere constraints prevent Gaussians from escaping the local region, and an efficient rendering kernel avoids full-scene computation.

### Key Designs

1. **DINOv2-based Change Detection**: Given a new-view image $I_i^t$ and the corresponding rendered image $\hat{I}_i^{t-1}$ from the existing reconstruction $\mathcal{G}^{t-1}$, per-patch feature maps are extracted with DINOv2, and cosine similarity is computed. Regions falling below threshold $\tau_1$ are marked as changed. The mask is then dilated to fill noisy holes. DINOv2 is preferred over pixel-level L2 or SSIM because it is more robust to lighting variation and rendering errors, improving recall from 0.745/0.761 to **0.961**.

2. **Majority-Voting 3D Mask + New Point Sampling**: Existing Gaussians are projected onto each 2D mask and counted; Gaussians appearing in more than $K$ view masks are labeled as the changed region $\mathcal{O}^t$. For newly appearing objects (with no corresponding existing Gaussians), a recursive sampling algorithm (Algorithm 1) first samples randomly and then samples near the 3D mask region, ensuring adequate initial points for new areas.

3. **Sphere-Constrained Local Optimization**: HDBSCAN clustering is applied to the changed-region Gaussians, and a bounding sphere is fitted to each cluster. During optimization, whether each Gaussian center lies within the union of spheres is dynamically checked; Gaussians that escape are pruned. This ensures strictly local optimization without affecting unchanged regions. Spheres are preferred over axis-aligned bounding boxes: membership checking requires only 1/3 of the FLOPS.

### Loss & Training

- The standard 3DGS photometric loss is used, but computed **only within a dynamically generated 2D rendering mask**.
- At each optimization step: (1) $\mathcal{O}^t$ is projected onto the image plane to generate a dynamic rendering mask; (2) rendering and gradient computation are performed only on masked pixels; (3) backpropagation updates only the Gaussians contributing to masked pixels.
- Key property: the gradients produced by the local optimization kernel are exactly equivalent to those of full-scene optimization restricted to $\mathcal{O}^t$, while substantially reducing computation.
- Freezing all Gaussian parameters outside the changed region is the single most critical factor (without freezing, PSNR drops from 40.8 to 20.8 dB).

## Key Experimental Results

### Main Results (Tables)

**CL-Splats Dataset (Synthetic + Real)**

| Method | Syn. PSNR↑ | Syn. LPIPS↓ | Syn. SSIM↑ | Syn. FPS↑ | Real PSNR↑ | Real LPIPS↓ | Real SSIM↑ | Real FPS↑ |
|------|-----------|------------|-----------|----------|-----------|------------|-----------|----------|
| 3DGS | 21.993 | 0.189 | 0.838 | 221 | 11.764 | 0.376 | 0.399 | 125 |
| 3DGS+M | 15.127 | 0.303 | 0.737 | 254 | 8.585 | 0.461 | 0.271 | 151 |
| GaussianEditor | 19.801 | 0.197 | 0.871 | 227 | 24.133 | 0.143 | 0.867 | 137 |
| CLNeRF | 26.758 | 0.322 | 0.738 | <1 | 24.541 | 0.373 | 0.658 | <1 |
| CL-NeRF | 30.063 | 0.058 | 0.939 | <1 | 23.268 | 0.290 | 0.725 | <1 |
| **CL-Splats** | **40.125** | **0.015** | **0.985** | **223** | **28.249** | **0.065** | **0.930** | **135** |

CL-Splats surpasses the second-best method by 10 dB PSNR on synthetic data (40.1 vs. 30.1), approaching the upper bound of densely re-sampled full-scene 3DGS (~42 dB), while maintaining real-time rendering (>120 FPS).

**CL-NeRF Dataset**

| Method | PSNR↑ | LPIPS↓ | SSIM↑ |
|------|-------|--------|-------|
| 3DGS | 11.072 | 0.356 | 0.537 |
| CL-NeRF | 27.302 | 0.177 | 0.829 |
| **CL-Splats** | **29.984** | **0.156** | **0.839** |

### Ablation Study (Tables)

**Optimization Component Ablation (CL-Splats Dataset Level 2)**

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Time |
|------|-------|-------|--------|------|
| (a) No background freezing | 20.773 | 0.811 | 0.176 | 8 min |
| (b) All-view voting | 35.611 | 0.881 | 0.102 | 5 min |
| (c) No local kernel | 40.812 | 0.978 | 0.018 | **8 min** |
| (d) Bounding box | 40.717 | 0.979 | 0.018 | 5 min |
| **(e) Full method** | **40.833** | **0.980** | **0.018** | **5 min** |

Background freezing is the single most critical factor (+20 dB); the local optimization kernel reduces training time from 8 min to 5 min (−60%) without loss of accuracy.

**Mask Quality Comparison**

| Method | Recall↑ | Precision↑ |
|------|---------|-----------|
| Color L2 | 0.761 | 0.281 |
| SSIM | 0.745 | 0.332 |
| DINOv2 mask | **0.961** | 0.370 |
| Full method (after 3D projection) | 0.942 | **0.609** |

### Key Findings

- **Rapid convergence**: CL-Splats achieves high-quality reconstruction in 5K iterations (40 seconds on an RTX Quadro 6000), whereas CL-NeRF requires 25K iterations and 50 minutes (75× slower).
- **3DGS+M performs worse than unconstrained 3DGS**: Applying 2D masks directly to constrain the 3DGS photometric loss (without 3D lifting) performs worse than the unconstrained baseline (15.1 vs. 22.0 dB), validating the necessity of 3D spatial awareness in mask construction.
- **Object removal is easiest; multi-object changes are hardest**: Removal only requires deleting Gaussians, whereas multi-object changes involve synchronized optimization across multiple clusters.
- **Historical recovery requires only 36 MB per step**: Compared to the naive per-step storage of 1173 MB, exploiting locality by storing only changed regions and indices yields a 32× storage efficiency improvement.

## Highlights & Insights

- **Advantages of explicit representation**: The explicit Gaussian representation in 3DGS naturally supports local editing, historical state recovery, and concurrent updates — capabilities that are difficult to realize with implicit NeRF representations.
- **Elegant local optimization kernel design**: Dynamic projection combined with a rendering mask and local backpropagation guarantees gradient equivalence to full-scene optimization while substantially reducing computation.
- **HDBSCAN + sphere constraints**: Automatic clustering of changed regions and bounding sphere fitting elegantly resolves the problem of defining 3D optimization boundaries.
- **Broad application potential**: Concurrent updates (multiple independent changes can be optimized in parallel and merged) and historical recovery (efficient storage of scene evolution) are highly significant for robotics and mixed reality.

## Limitations & Future Work

- The method assumes changes are local and cannot handle global illumination shifts (e.g., day-to-night transitions).
- COLMAP is relied upon for estimating new-view poses, which may fail in scenes with large-scale changes.
- The reconstruction quality of 3DGS itself under extremely sparse views (2–3 images) is inherently limited, constraining overall performance.
- Scalability to large-scale outdoor scenes (e.g., autonomous driving data) remains unclear.
- Sphere constraints may be ill-suited for non-convex changed regions; multi-sphere coverage is possible but increases complexity.

## Related Work & Insights

- Unlike 3D editing methods such as GaussianEditor, CL-Splats drives updates from real observations rather than user instructions, making it better suited for autonomous systems.
- CL-NeRF and CLNeRF are the most direct competitors, but their NeRF-based representations limit speed and flexibility.
- Using DINOv2 for change detection is a lightweight yet highly effective choice, substantially outperforming pixel-level comparisons.
- Implications for robotics: domestic robots could maintain continuously updated 3D maps of home environments through periodic sparse image capture, tracking changes in object positions over time.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The local optimization framework introducing continual learning into 3DGS is well-designed and genuinely novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Synthetic and real datasets, multiple baselines, comprehensive ablations, and a newly contributed dataset
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, well-illustrated, and the pipeline is presented completely
- **Value**: ⭐⭐⭐⭐⭐ High practical value with significant implications for 3D reconstruction in dynamic environments

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GaussianUpdate: Continual 3D Gaussian Splatting Update for Changing Environments](gaussianupdate_continual_3d_gaussian_splatting_update_for_changing_environments.md)
- [\[ICCV 2025\] A Lesson in Splats: Teacher-Guided Diffusion for 3D Gaussian Splats Generation with 2D Supervision](a_lesson_in_splats_teacher-guided_diffusion_for_3d_gaussian_splats_generation_wi.md)
- [\[ICCV 2025\] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt](3dgslm_faster_gaussiansplatting_optimization_with_levenbergm.md)
- [\[AAAI 2026\] Splats in Splats: Robust and Effective 3D Steganography towards Gaussian Splatting](../../AAAI2026/3d_vision/splats_in_splats_robust_and_effective_3d_steganography_towards_gaussian_splattin.md)
- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)

</div>

<!-- RELATED:END -->
