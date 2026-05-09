---
title: >-
  [Paper Note] Pano360: Perspective to Panoramic Vision with Geometric Consistency
description: >-
  [CVPR2026][3D Vision][panorama stitching] Pano360 extends panoramic image stitching from conventional 2D pairwise matching to the 3D photogrammetric space, leveraging a Transformer architecture to achieve globally geometrically consistent multi-view alignment. It attains a 97.8% success rate under challenging scenarios including weak texture, large parallax, and repetitive patterns.
tags:
  - CVPR2026
  - 3D Vision
  - panorama stitching
  - 3D geometric consistency
  - transformer
  - multi-view alignment
  - seam detection
date: 2026-05-08
content_hash: b68937ba4c594fef
---

# Pano360: Perspective to Panoramic Vision with Geometric Consistency

**Conference**: CVPR2026
**arXiv**: [2603.12013](https://arxiv.org/abs/2603.12013)
**Code**: [KiMomota/Pano360](https://github.com/KiMomota/Pano360)
**Area**: 3D Vision
**Keywords**: panorama stitching, 3D geometric consistency, transformer, multi-view alignment, seam detection

## TL;DR

Pano360 extends panoramic image stitching from conventional 2D pairwise matching to the 3D photogrammetric space, leveraging a Transformer architecture to achieve globally geometrically consistent multi-view alignment. It attains a 97.8% success rate under challenging scenarios including weak texture, large parallax, and repetitive patterns.

## Background & Motivation

Panoramic image stitching is in high demand for downstream tasks such as autonomous driving, VR, and 3D Gaussian Splatting. Existing methods suffer from the following core issues:

**Error accumulation in pairwise matching**: Both traditional pipelines (SIFT/ORB/LoFTR + RANSAC) and learning-based methods (UDIS/UDIS2) are confined to establishing 2D feature correspondences pairwise. When stitching multiple images, projection errors accumulate progressively, causing severe distortion.

**Feature matching failure under challenging conditions**: In scenes with weak texture, large parallax, or repetitive patterns, reliable feature matches are scarce, making homography estimation prone to failure.

**Neglect of 3D projection geometry**: Existing methods pursue visual seamlessness while ignoring global 3D projection consistency, leading to geometric distortion.

**High cost of post-processing**: CNN-based methods (e.g., UDIS2) require complex post-processing to complete multi-image alignment, limiting practical utility.

**Core Insight**: Multi-view geometric correspondences can be constructed directly in 3D space, yielding greater accuracy and global consistency than 2D correspondences. The authors therefore lift the 2D alignment task into the 3D photogrammetric space, fundamentally addressing error accumulation.

## Method

### Overall Architecture

Pano360 adopts a dual-branch Transformer architecture. Given $N$ partially overlapping input images, it jointly predicts all stitching parameters in a single forward pass:

$$f(\{I_i\}_{i=1}^N) = \{P_i, W_i, M_i\}_{i=1}^N$$

where $P_i$ denotes the global projection transformation, $W_i$ the local deformation field (handling parallax), and $M_i$ the seam mask. The complete pixel mapping is:

$$\mathcal{W}_i(\mathbf{u}) = P_i(\mathbf{u}) + W_i(\mathbf{u})$$

Pipeline: (a) project perspective images into a unified panoramic coordinate system using camera parameters → (b) extract overlapping regions → (c) seam decoder generates per-image seam masks → (d) blend aligned images using masks to produce the final panorama.

### Key Designs

**1. Feature Backbone**

- Each image is first tokenized via a pretrained DINO encoder.
- Learnable camera tokens are prepended to all image embedding sequences to capture cross-image global geometric relationships.
- $L$ layers of alternating attention (global attention + frame attention) from pretrained VGGT process the concatenated sequence.
- Two outputs are produced: camera tokens (encoding 3D geometric correspondences, fed to the projection head) and feature tokens (retaining local detail, fed to the seam head).

**2. Projection Head**

- Decodes per-image intrinsics $\mathbf{K}_i$ and extrinsics $\{\mathbf{R}_i, \mathbf{t}_i\}$ from predicted camera tokens.
- Assumes a shared focal length across cameras with principal points at image centers; the first image is fixed as the reference frame ($\mathbf{R}_1=\mathbf{I}, \mathbf{t}_1=\mathbf{0}$).
- Supports adaptive selection of projection format: planar, equirectangular, spherical, etc.
- For large-parallax scenes, an additional local mesh warp $W_i$ is computed to correct residual misalignment.

**3. Seam Head — Multi-Feature Joint Optimization**

Seam detection is formulated as an energy minimization problem:

$$E(\mathcal{I}) = E_l(\mathcal{I}) + E_c(\mathcal{I})$$

- $E_l$: label cost — hard constraint ensuring each pixel originates from a valid image region.
- $E_c$: continuity cost — penalizes adjacent pixels with different labels, encouraging seams to be continuous and inconspicuous.

The pixel-level cost function fuses three types of information:

$$C(p) = F_{color}(p) + F_{gradient}(p) \times F_{ratio}(p)$$

| Cost Term | Definition | Role |
|-----------|-----------|------|
| $F_{color}$ | Color difference between overlapping images $\|I_i(p) - I_j(p)\|$ | Guides seams away from color discontinuities |
| $F_{gradient}$ | Gradient magnitude $|\nabla I_i(p)| + |\nabla I_j(p)|$ | Guides seams away from sharp object edges |
| $F_{ratio}$ | Texture complexity map | Heavily penalizes visually complex regions (with parallax/depth variation), steering seams toward uniform areas |

Key advantage: **all overlapping images** are considered simultaneously when computing color differences and gradients, avoiding the local optima inherent in pairwise computation. The resulting seam masks serve as pseudo-labels for supervising the seam decoder.

### Loss & Training

The multi-task loss comprises three terms:

| Loss | Formula | Description |
|------|---------|-------------|
| $\mathcal{L}_{cam}$ | $\sum_{i=1}^N \|\hat{\mathbf{g}}_i - \mathbf{g}_i\|_\epsilon$ (Huber loss) | Supervises camera parameter prediction |
| $\mathcal{L}_{seam}$ | $\sum_{i=1}^N \|\hat{M}_i - M_i\|$ (L1 loss) | Supervises seam mask prediction |
| $\mathcal{L}_{proj}$ | Predefined projection format loss | Adapts the network to different projection formats; enabled from the start of training to ensure gradient continuity |

Training details:
- VGGT alternating attention weights are initialized from pretrained checkpoints and **frozen**.
- Uncertainty terms are removed to accelerate convergence.
- Data normalization: all quantities are expressed in the first-frame coordinate system, ensuring permutation invariance of inputs.
- Data augmentation: random rotational jitter of up to 2° is applied to yaw/pitch/roll.

**Pano360 Dataset**: 200 real-world scenes (50% tourism, 30% extreme sports, 20% extreme lighting), each with 3 focal lengths × 24 frames = 72 images (2048×2048), totaling 14,400 frames with annotated ground-truth camera parameters and full 360° FoV coverage.

## Key Experimental Results

### Main Results: Panorama Quality Comparison on Pano360 Dataset

| Method | QA_q ↑ | QA_a ↑ | BRIS ↓ | NIQE ↓ | Notes |
|--------|--------|--------|--------|--------|-------|
| AutoStitch | 3.28 | 2.81 | 49.84 | 5.01 | Traditional features |
| APAP | 3.53 | 3.66 | 45.66 | 3.77 | Traditional features |
| GES-GSP | 3.74 | 3.72 | 44.22 | 3.95 | Traditional features |
| UDIS2‡ | 2.87 | 2.34 | 58.62 | 4.91 | Pairwise only |
| **Pano360 (Ours)** | **4.09** | **3.94** | **37.96** | **3.37** | — |

(Results shown for Scene (c), which includes repetitive textures, abnormal lighting, and large FoV challenges.)

### Success Rate and Speed Comparison

| Method | Requires Geometric Features | Success Rate (%) | Runtime |
|--------|-----------------------------|-----------------|---------|
| LoFTR+RANSAC | ✓ | 63.4 | ~13s |
| LightGlue+RANSAC | ✓ | 66.7 | ~11s |
| ELA | ✓ | 80.1 | ~90s |
| GES-GSP | ✓ | 83.3 | ~20s |
| APAP | ✓ | 30.0 | >300s |
| **Pano360 (Ours)** | ✗ | **97.8** | **~5s** |

### Generalization on UDIS-D Dataset

| Method | PSNR ↑ | SSIM ↑ | PIQE ↓ | NIQE ↓ |
|--------|--------|--------|--------|--------|
| UDIS2‡ | 25.43 | 0.838 | 48.09 | 6.11 |
| DHS‡ | 25.88 | 0.845 | 45.73 | 6.18 |
| **Pano360 (Ours)** | **25.97** | **0.852** | **42.12** | **5.78** |

(Pano360 was not trained on UDIS-D; it still outperforms methods specifically fine-tuned for pairwise scenarios when generalized to that setting.)

### Ablation Study

| $\mathcal{L}_{cam}$ | $\mathcal{L}_{proj}$ | $\mathcal{L}_{seam}$ | QA_q ↑ | BRIS ↓ | NIQE ↓ |
|:---:|:---:|:---:|--------|--------|--------|
| ✗ | ✗ | ✗ | 2.76 | 62.47 | 5.31 |
| ✓ | ✗ | ✗ | 3.45 | 47.43 | 4.65 |
| ✓ | ✓ | ✗ | 3.68 | 43.71 | 3.97 |
| ✗ | ✗ | ✓ | 3.01 | 51.12 | 4.83 |
| ✓ | ✓ | ✓ | **4.09** | **37.96** | **3.37** |

**Key Findings**:
- Pose-guided alignment ($\mathcal{L}_{cam}$) contributes the most, raising QA_q from 2.76 to 3.45.
- The projection function further eliminates non-perspective distortion, reducing BRIS by approximately 4 points.
- The combination of all three terms is optimal; seam optimization alone without alignment yields limited benefit, as precise alignment is a prerequisite for high-quality seams.
- In seam ablations: removing the color term causes noticeable color artifacts; removing the texture map causes ghosting (seams passing through foreground subjects); the traditional graph-cut approach produces the most severe structural distortion.

## Highlights & Insights

1. **Paradigm shift**: Transitioning from 2D pairwise matching to 3D global alignment represents a significant advance in panoramic stitching, directly filtering unreliable matches via multi-view geometric consistency in 3D space.
2. **Elegant architectural reuse**: The pretrained VGGT alternating attention module — inherently 3D-aware — is frozen and repurposed, achieving powerful cross-view feature aggregation at minimal training cost.
3. **Scalability**: The method supports inputs ranging from a few to hundreds of images and is more than 10× faster than pairwise methods at large scale.
4. **Multi-feature joint seam optimization**: Simultaneously incorporating color, gradient, and texture information from all overlapping images avoids the local optima of pairwise computation.
5. **High-quality dataset**: 14,400 frames of real-world scenes covering extreme motion and nighttime conditions fill a significant data gap in the field.

## Limitations & Future Work

1. **No support for distorted inputs**: The model assumes distortion-free input images, precluding direct application to fisheye lenses and other camera types.
2. **Limitations under extreme parallax**: When the same object is captured from drastically different viewpoints, full 3D reconstruction is still required for correct stitching; image-level warping alone is insufficient.
3. **Trade-off of freezing VGGT**: Although freezing the pretrained attention module reduces training cost, it may limit further task-specific adaptation for panoramic stitching.
4. Promising future directions: (a) integrating depth estimation modules to handle more complex parallax; (b) extension to video panorama stitching and real-time scenarios; (c) support for heterogeneous lens configurations (mixed fisheye and perspective inputs).

## Related Work & Insights

- **VGGT** [Wang et al.]: Provides 3D-aware Transformer features, adopted as the backbone foundation in this work.
- **UDIS/UDIS2** [Nie et al.]: Representative CNN-based learning methods, limited to pairwise stitching.
- **GES-GSP** [Du et al.]: A geometry-structure-preserving traditional method that still fails under repetitive textures.
- **LoFTR/LightGlue**: Modern feature matching methods; combined with RANSAC, their success rates remain only 60–67%.
- Key takeaway: **The strategy of lifting 2D problems into 3D space is broadly applicable to other geometric vision tasks**, such as image registration and optical flow estimation.

## Rating

| Dimension | Score (1–10) | Comments |
|-----------|:------------:|---------|
| Novelty | 8 | The 2D→3D paradigm shift is original; the architectural reuse of VGGT is elegant |
| Technical Depth | 8 | Projection head + seam head + multi-task loss form a complete design with clear theoretical derivation |
| Experimental Thoroughness | 8 | Multi-dataset validation, comprehensive ablation, generalization experiments, and qualitative comparisons |
| Practical Value | 8 | 97.8% success rate and ~5s runtime make it applicable at large scale |
| Writing Quality | 7 | Generally clear; some formula typesetting is slightly dense |
| **Overall** | **8.0** | Solid contribution to panoramic stitching: paradigm innovation backed by strong experiments |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GAP: Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](action-geometry_prediction_with_3d_geometric_prior_for_bimanual_manipulation.md)
- [\[CVPR 2026\] No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency](no_calibration_no_depth_no_problem_cross-sensor_view_synthesis_with_3d_consisten.md)
- [\[CVPR 2026\] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty](rethinking_pose_refinement_in_3d_gaussian_splatting_under_pose_prior_and_geometr.md)
- [\[CVPR 2026\] A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering](a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)
- [\[CVPR 2026\] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)

</div>

<!-- RELATED:END -->
