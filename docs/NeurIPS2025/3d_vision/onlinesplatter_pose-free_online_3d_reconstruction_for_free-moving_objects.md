---
title: >-
  [Paper Note] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects
description: >-
  [NeurIPS 2025][3D Vision][Online 3D Reconstruction] This paper proposes OnlineSplatter, a feed-forward online 3D reconstruction framework that requires no camera poses, depth priors, or global optimization. It achieves constant-time incremental reconstruction of free-moving objects via a dual-key memory module combining appearance-geometry latent keys and orientation keys.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Online 3D Reconstruction
  - 3D Gaussian Splatting
  - Pose-Free Reconstruction
  - Free-Moving Objects
  - Memory Module
date: 2026-05-08
content_hash: 373aa7b69d1e8560
---

# OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects

**Conference**: NeurIPS 2025
**arXiv**: [2510.20605](https://arxiv.org/abs/2510.20605)
**Code**: [https://markhh.com/OnlineSplatter](https://markhh.com/OnlineSplatter)
**Area**: 3D Vision
**Keywords**: Online 3D Reconstruction, 3D Gaussian Splatting, Pose-Free Reconstruction, Free-Moving Objects, Memory Module

## TL;DR
This paper proposes OnlineSplatter, a feed-forward online 3D reconstruction framework that requires no camera poses, depth priors, or global optimization. It achieves constant-time incremental reconstruction of free-moving objects via a dual-key memory module combining appearance-geometry latent keys and orientation keys.

## Background & Motivation
Real-time monocular reconstruction of free-moving objects is a fundamental challenge in computer vision, with applications in robotics, AR, and beyond. Existing methods suffer from the following limitations:

**Optimization-based methods** (BARF, BundleSDF, Fmov): Require global bundle adjustment and cannot run online in real time; BundleSDF additionally requires ground-truth depth input.

**Diffusion-based generative methods** (LRM, InstantMesh): Rely on learned priors to "hallucinate" unobserved regions, making them unsuitable for perception tasks.

**Feed-forward point-map methods** (DUSt3R, NoPoSplat, Spann3R): Assume static scenes and treat moving objects as outliers; implicitly rely on large background surfaces.

**Key Challenge**: Online reconstruction requires causal processing (updating upon each arriving frame), yet existing methods either demand global optimization, assume static scenes, or require additional sensors.

**Key Insight**: The paper designs an object-centric feed-forward framework that defines a canonical coordinate system from the first frame and incrementally fuses temporal information in constant time via a dual-key memory module, without requiring poses, depth, or background information.

## Method

### Overall Architecture
At each timestep $t$: an RGB frame $V_t$ is input → online video segmentation yields an object mask → a dual encoder extracts patch features → the OnlineSplatter Transformer jointly processes the reference frame, current frame, and memory tokens → outputs pixel-aligned 3D Gaussians → updates the object memory. The entire pipeline is feed-forward with constant time complexity.

### Key Designs

1. **Dual-Encoder Image Feature Extraction**:

    - A frozen DINO encoder provides strong self-supervised appearance cues.
    - A trainable encoder of the same architecture captures complementary geometric cues.
    - Features are concatenated: $f_{vt} = \text{Concat}(\text{Encoder}_1^I(V_t'), \text{Encoder}_2^I(V_t'))$
    - **Design Motivation**: DINO offers strong visual priors but lacks 3D awareness.

2. **Dual-Key 3D Object Memory**: Core contribution.

    - **Latent key** $\mathbf{k}_t^{(L)}$: Learned by a lightweight encoder from patch features, capturing visual-geometric cues.
    - **Orientation key** $\mathbf{k}_t^{(D)}$: Derived from a pretrained zero-shot 3D orientation estimator, converted to a unit direction vector.
    - **Value** $\mathbf{v}_t^{(L)}$: Encoded from Transformer output tokens.
    - **Dual-purpose retrieval**:
        - Orientation-aligned retrieval: Retrieves memory entries with similar latent and orientation keys (current-viewpoint information).
        - Orientation-complementary retrieval: Retrieves entries with similar latent keys but opposite orientation keys (complementary-viewpoint information).
    - Similarity formula: $s_{i,t}^{(\text{align})} = (\mathbf{q}_t^{(L)\top}\mathbf{k}_i^{(L)}) \cdot \mathbf{q}_t^{(D)\top}\mathbf{k}_i^{(D)} \cdot \frac{1}{\tau_t}$

3. **Memory Sparsification Mechanism**:

    - When memory reaches capacity $S$, the 20% least useful entries are pruned.
    - Two criteria are jointly considered: utilization (accumulated cross-attention weights) and spatial coverage (mean angular distance of orientation keys).
    - Low-utilization entries are removed from a high-coverage subset, balancing the retention of unique viewpoints and the removal of redundant ones.

4. **Gaussian Decoding and Rendering**:

    - Transformer outputs are decoded into $4N$ Gaussians: $\mathbf{G}_{obj,t}^{4N} = \{\mathbf{G}_{mem,t}^{2N}, \mathbf{G}_{ref,t}^{N}, \mathbf{G}_{src,t}^{N}\}$
    - Non-accumulative: each step directly outputs a complete object representation, avoiding global aggregation.
    - Frame-level subsets are rendered from their respective viewpoints, encouraging each Gaussian group to specialize on the corresponding visible region.

### Loss & Training
- **Two-stage training**: Warm-up (no memory module, 250K steps) → Main (with memory module, 500K steps).
- **Photometric loss** $\mathcal{L}_\text{photo}$: MSE between ground-truth and rendered images, plus a background penalty term.
- **Geometric loss** $\mathcal{L}_\text{geo}$: Ray alignment $\mathcal{L}_\text{ray}$ + relative depth $\mathcal{L}_\text{depth}$.
- Training data: 100K objects from Objaverse; diverse trajectories generated via custom scripts.

## Key Experimental Results

### Main Results (GSO Dataset)

| Method | Phase | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|-------|-------|--------|
| FreeSplatter-dist4 | Late | 23.751 | 0.873 | 0.120 |
| NoPoSplat-dist3 | Late | 24.141 | 0.863 | 0.125 |
| **OnlineSplatter** | **Late** | **31.737** | **0.969** | **0.075** |
| FreeSplatter-dist4 | Early | 22.365 | 0.874 | 0.119 |
| **OnlineSplatter** | **Early** | **26.329** | **0.921** | **0.084** |

Late-phase results on HO3D: PSNR 27.928 vs. best baseline 22.947 (+4.981).

### Ablation Study

| Configuration | Early $\mathcal{M}_{avg}$↑ | Mid $\mathcal{M}_{avg}$↑ | Late $\mathcal{M}_{avg}$↑ |
|------|--------------------------|-------------------------|--------------------------|
| Full model | 0.699 | 0.734 | **0.810** |
| w/o latent key | 0.545 | 0.582 | 0.596 |
| w/o orientation key | 0.699 | 0.701 | 0.723 |
| w/o staged training | 0.545 | 0.582 | 0.588 |
| w/o ray loss | 0.562 | 0.599 | 0.682 |
| random pruning | 0.697 | 0.728 | 0.764 |

### Key Findings
- OnlineSplatter achieves PSNR gains of +7.596 (GSO) and +4.981 (HO3D) in the Late phase, substantially outperforming all baselines.
- Performance improves consistently as more observations accumulate, whereas baselines frequently plateau or fluctuate.
- Removing the latent key causes the largest performance drop (−0.214); the orientation key primarily affects later phases (−0.087), confirming their complementarity.
- Staged training is critical: single-stage training yields only 0.588 Late-phase performance vs. 0.810 with staged training.

## Highlights & Insights
- **Constant-time online reconstruction**: Each frame incurs $O(1)$ update cost independent of sequence length, making the method genuinely suitable for real-time applications.
- **Complementarity of the dual-key design**: The latent key captures "what is relevant" while the orientation key encodes "from where it is observed"—together they enable comprehensive spatial coverage.
- **Non-accumulative paradigm**: Unlike conventional methods that accumulate predictions and then globally optimize, the proposed approach directly outputs a complete representation at each step, fundamentally eliminating redundancy and optimization overhead.

## Limitations & Future Work
- Only rigid objects are supported; non-rigid deformable objects are not handled.
- The quality of the initial frame affects subsequent reconstruction (severe occlusion or blur in the first frame degrades overall performance).
- Converting the output 3DGS representation to explicit meshes remains challenging.
- Resolution is limited to 256×256; scaling to higher resolutions requires further investigation.

## Related Work & Insights
- **vs. BundleSDF**: BundleSDF requires ground-truth depth and keyframe-matching optimization; OnlineSplatter operates as a purely RGB feed-forward method.
- **vs. DUSt3R/NoPoSplat**: These methods assume static scenes, whereas OnlineSplatter is specifically designed for free-moving objects.
- **vs. FreeSplatter**: FreeSplatter processes four frames at a time and requires a frame selection strategy; OnlineSplatter naturally accumulates information through its memory module.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The dual-key memory design is novel; integrating orientation estimation into memory retrieval is particularly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on both synthetic and real datasets with phased assessment, comprehensive ablations, and mesh comparisons.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, though the dense notation requires some adaptation on first reading.
- Value: ⭐⭐⭐⭐⭐ The first truly pose-free online object reconstruction feed-forward framework, with direct applicability to robotic perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](../../CVPR2026/3d_vision/e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[NeurIPS 2025\] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](../../ICCV2025/3d_vision/no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](../../ICCV2025/3d_vision/pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)

</div>

<!-- RELATED:END -->
