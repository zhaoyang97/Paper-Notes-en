---
title: >-
  [Paper Note] SAP: Segment Any 4K Panorama
description: >-
  [CVPR 2026][Segmentation][Panoramic Segmentation] This paper proposes SAP (Segment Any 4K Panorama), which converts panoramic images into perspective pseudo-video sequences sampled along fixed spherical trajectories…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Panoramic Segmentation"
  - "SAM2"
  - "4K High Resolution"
  - "Topology-Memory Alignment"
  - "Perspective Video Reconstruction"
date: 2026-05-08
content_hash: 627ea08a64afb1d6
---

# SAP: Segment Any 4K Panorama

**Conference**: CVPR 2026
**arXiv**: [2603.12759](https://arxiv.org/abs/2603.12759)  
**Code**: Available (Project Page)  
**Area**: Panoramic Image Segmentation
**Keywords**: Panoramic Segmentation, SAM2, 4K High Resolution, Topology-Memory Alignment, Perspective Video Reconstruction

## TL;DR

This paper proposes SAP (Segment Any 4K Panorama), which converts panoramic images into perspective pseudo-video sequences sampled along fixed spherical trajectories, addressing the structural mismatch of SAM2's streaming memory mechanism on 360° images. By synthesizing a 183K instance-annotated 4K panoramic dataset for fine-tuning, SAP achieves a zero-shot mIoU improvement of +17.2 on real-world panoramic benchmarks.

## Background & Motivation

With the growing adoption of 360° cameras in robotics, AR/VR, and embodied intelligence, the demand for high-quality instance segmentation of panoramic images has increased substantially. However, existing segmentation foundation models (e.g., SAM/SAM2) face three major challenges:

1. **Resolution loss**: SAM supports only $1024^2$ inputs; a 4K 2:1 panorama ($4096 \times 2048$) is downsampled to $1024 \times 512$ and padded, losing significant detail.
2. **Geometric distortion**: Equirectangular projection (ERP) introduces severe polar distortion and left-right seam discontinuities.
3. **Violation of structural assumptions**: SAM2's streaming memory mechanism assumes consecutive frames correspond to smooth camera motion and overlapping visual content, whereas ERP panoramas have **no intrinsic temporal order**—sliding-window cropping disrupts physical viewpoint continuity.

OmniSAM attempts to apply SAM2 with a sliding window directly on ERP, but distortion and seam issues are merely surface-level symptoms. The key insight of this paper is that **spherical panoramas fundamentally violate the structural assumptions of streaming memory models**, and this must be addressed from a topology-memory alignment perspective.

## Method

### Overall Architecture

The SAP pipeline consists of four steps:
1. Project the ERP panorama and prompt points into a perspective pseudo-video sequence sampled along a fixed trajectory.
2. Process the pseudo-video with fine-tuned SAM2 to generate per-frame segmentation masks.
3. Back-project perspective frame masks and fuse them onto the ERP plane.
4. Output the final panoramic segmentation result.

### Key Designs

1. **Panorama-to-perspective video conversion**: Given an ERP image $I^{ERP} \in \mathbb{R}^{H \times W \times 3}$, $N$ perspective views are generated via a three-stage geometric projection:
    - Define camera intrinsics (FoV $\beta = 90°$, focal length $f = \frac{L-1}{2\tan(\beta/2)}$)
    - Back-project pixels to ray directions: $\mathbf{r}^{cam} \propto \mathbf{K}^{-1}[u,v,1]^T$
    - Rotate to world coordinates and convert to spherical coordinates for sampling: $[x,y,z]^T = \text{Normalize}(\mathbf{R}_i \mathbf{K}^{-1}[u,v,1]^T)$

2. **Column-first zigzag scanning trajectory**: This is the central design innovation of the paper. Compared to row-first scanning, column-first scanning possesses an **infinite-loop** property: starting from any point, alternating up-and-down motion returns exactly to the starting point. Formally, the visiting order for column $j$ is:
    $$\mathcal{O}_j = \begin{cases} (j,1),(j,2),\dots,(j,N_{pitch}), & j \bmod 2 = 1 \\ (j,N_{pitch}),\dots,(j,2),(j,1), & j \bmod 2 = 0 \end{cases}$$
    Consecutive frames differ in only one angular dimension (yaw or pitch), ensuring smooth video-like transitions. The sampling grid is determined by FoV and overlap ratio $r=0.5$: $\Delta_{yaw} = \beta_h(1-r)$, $N_{yaw} = \lceil 360°/\Delta_{yaw} \rceil$.

3. **Cyclic extension for arbitrary starting points**: The trajectory is duplicated to $2 \cdot N$ frames; during training, a random start index $s \in \{0, \dots, N-1\}$ is sampled and a contiguous window of $N$ frames is extracted, guaranteeing that any window covers all viewpoints at least once.

4. **183K synthetic dataset**: Using the InfiniGen engine and 40,000 GPU hours, 183,440 panoramic images at $4096 \times 2048$ resolution were synthesized, comprising 6,409,732 instance masks. Object size distribution: small 37.84%, medium 25.70%, large 36.47%.

5. **Prompt point projection**: A user prompt point $\mathbf{p} = (u_p, v_p)$ on the ERP is first converted to a spherical direction vector $\mathbf{d} = [\cos\theta_p\cos\phi_p, \cos\theta_p\sin\phi_p, \sin\theta_p]^T$, then projected onto each perspective frame to determine visibility.

### Mask Fusion

Per-frame masks are fused back onto the ERP plane via max-value aggregation:

$$M^{ERP}(u,v) = \max_{i: (u,v) \in \mathcal{V}_i} \tilde{M}_i(u,v)$$

### Loss & Training

- Built on SAM2 (Hiera-Large encoder)
- **Image encoder is frozen**; only the memory attention, memory encoder, mask decoder, and prompt encoder are updated
- Mixed training: synthetic panoramic data + SAM2 original training data (SA-1B + SA-V) to prevent catastrophic forgetting
- AdamW optimizer, batch size 128, lr $2 \times 10^{-4}$ (cosine schedule), weight decay 0.1, gradient clipping 0.1

## Key Experimental Results

### PAV-SOD Real-World 4K Panoramic Benchmark (Zero-Shot)

| Method | 1-click Overall | 1-click Small | 1-click Large | 3-click Overall |
|--------|----------------|--------------|--------------|----------------|
| SAM2-tiny | 51.6 | 46.3 | 49.1 | 82.2 |
| SAM2-tiny+scan | 65.1 | 49.6 | 70.0 | 83.0 |
| **SAP-tiny** | **75.8** | **53.9** | **79.7** | **84.8** |
| Δ(SAP-SAM2) | **+24.2** | +7.6 | +30.6 | +2.6 |
| SAM2-large | 66.3 | 50.7 | 64.4 | 84.3 |
| SAM2-large+scan | 69.0 | 58.4 | 73.8 | 84.1 |
| **SAP-large** | **77.3** | **61.1** | **81.7** | **86.1** |
| Δ(SAP-SAM2) | **+11.0** | +10.4 | +17.3 | +1.8 |

### InfiniGen Synthetic 4K Panoramic Benchmark

| Method | 1-click Overall | 1-click Small | 1-click Large | 3-click Overall |
|--------|----------------|--------------|--------------|----------------|
| SAM2-base | 62.0 | 57.6 | 59.8 | 81.4 |
| SAP-base | **81.8** | **72.3** | **89.6** | **88.9** |
| Δ(SAP-SAM2) | **+19.8** | +14.7 | +29.8 | +7.5 |
| SAM2-large | 62.8 | 59.7 | 60.7 | 81.4 |
| SAP-large | **81.9** | **72.5** | **90.7** | **89.0** |
| Δ(SAP-SAM2) | **+19.1** | +12.8 | +30.0 | +7.6 |

### Key Findings

- SAP substantially outperforms SAM2 across all model sizes, with an **average gain of +17.2 mIoU** across four model variants (PAV-SOD 1-click).
- The scanning strategy alone (without fine-tuning) yields +13.5 improvement on the tiny model, but fine-tuning provides larger and more consistent gains.
- Improvement is most pronounced for large objects (PAV-SOD tiny: +30.6), indicating that cross-view propagation is especially critical for large-scale instances.
- On HunyuanWorld (cartoon-style 8K panoramas), applying the scan strategy without fine-tuning actually degrades performance, underscoring the necessity of fine-tuning.
- Ablation studies confirm that mixing SAM2 original training data significantly improves generalization (PAV-SOD: 67.3 → 77.3).

## Highlights & Insights

- **Precise problem formulation**: Reframing "ERP distortion and seam discontinuity" as a "topology-memory alignment" problem provides a fundamental explanation for SAM2's failure on panoramic inputs.
- **Elegant column-first zigzag trajectory design**: Satisfies the infinite-loop constraint and guarantees full coverage from any starting point—an elegant engineering solution.
- **Effective large-scale synthetic data**: The 183K synthetic images with SAM2 fine-tuning prove effective not only on synthetic test sets but also on real-world data, validating synthetic-to-real transfer.
- **Essential distinction from prior work**: OmniSAM applies a sliding window directly on ERP; SAP operates entirely in perspective space, avoiding distortion altogether.

## Limitations & Future Work

- The large number of perspective frames ($N_{yaw} \times N_{pitch}$ frames × 2 for cyclic extension) results in high inference cost.
- The fixed FoV of $90°$ and overlap ratio of $50\%$ are manually selected rather than adaptively optimized.
- Evaluation is limited to SAM2 as the foundation model; other segmentation foundation models are not assessed.
- Cross-frame instance consistency relies on SAM2's memory mechanism, which may still struggle in complex occlusion scenarios.
- Although synthetic data is effective, the domain gap persists, particularly for small objects where improvement is relatively limited.

## Related Work & Insights

- **SAM2 [Meta 2024]**: A video segmentation foundation model providing a streaming memory mechanism—the backbone of this work.
- **OmniSAM [2024]**: Applies SAM2 with a sliding window on ERP for semantic segmentation; the primary baseline improved upon in this paper.
- **InfiniGen [2024]**: A data engine for generating large-scale synthetic panoramic images.
- **Trans4PASS / PanoFormer**: Deformable embedding / tangent patch methods for handling spherical distortion.
- Inspiration: The topology-memory alignment paradigm can be generalized to adapt other foundation models to non-standard geometries such as spherical, cylindrical, and fisheye imaging.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](../../ICCV2025/segmentation/wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[CVPR 2026\] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] Unified Spherical Frontend: Learning Rotation-Equivariant Representations of Spherical Images from Any Camera](unified_spherical_frontend_learning_rotation-equivariant_representations_of_sphe.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[CVPR 2026\] Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](boundary_segment_action_segmentation.md)

</div>

<!-- RELATED:END -->
