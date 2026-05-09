---
title: >-
  [Paper Note] 3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image
description: >-
  [CVPR 2026][3D Vision][Single-image 3D scene generation] This paper proposes a novel paradigm termed *in-place completion*, which extends pretrained object-level generative priors to the scene level, directly completing fragmented geometry at its original spatial location without explicit pose alignment. The authors also construct ARSG-110K, a 110K-scale scene-level dataset, and substantially outperform baselines such as MIDI and Gen3DSR.
tags:
  - CVPR 2026
  - 3D Vision
  - Single-image 3D scene generation
  - in-place completion
  - coarse-to-fine completion
  - occlusion robustness
  - large-scale scene dataset
date: 2026-05-08
content_hash: 2a6293c53d658ffb
---

# 3D-Fixer: Coarse-to-Fine In-place Completion for 3D Scenes from a Single Image

**Conference**: CVPR 2026
**arXiv**: [2604.04406](https://arxiv.org/abs/2604.04406)
**Code**: [Project Page](https://zx-yin.github.io/3dfixer) (coming soon)
**Area**: 3D Vision
**Keywords**: Single-image 3D scene generation, in-place completion, coarse-to-fine completion, occlusion robustness, large-scale scene dataset

## TL;DR
This paper proposes a novel paradigm termed *in-place completion*, which extends pretrained object-level generative priors to the scene level, directly completing fragmented geometry at its original spatial location without explicit pose alignment. The authors also construct ARSG-110K, a 110K-scale scene-level dataset, and substantially outperform baselines such as MIDI and Gen3DSR.

## Background & Motivation
**Background**: Compositional 3D scene generation from a single image is a core task in robotics, AR/VR, and related domains.

**Limitations of Prior Work — Two Main Paradigms**:
   - **Feed-forward generation** (e.g., MIDI, SceneGen): end-to-end efficient but poor generalization, with multi-instance attention complexity scaling quadratically with the number of objects.
   - **Divide-and-conquer** (e.g., Gen3DSR): generates or retrieves individual objects and optimizes pose alignment — good generalization but time-consuming optimization that accumulates errors.

**Key Challenge**: How can generalization be maintained while avoiding costly pose alignment?

**Key Observation**: Geometry estimation models can already accurately recover the 3D geometry of visible regions, which encodes both the spatial layout and the visible portion of each instance. This makes it possible to complete the invisible parts directly in place, without first generating and then aligning.

**Core Idea**: Instead of "generate then align," perform *in-place completion* — using fragmented geometry as spatial anchors to complete full 3D assets at their original positions via object-level generative priors.

## Method

### Overall Architecture
Single image input → scene decomposition (instance segmentation + geometry estimation) → per-instance progressive completion (coarse structure → fine geometry → texture) → complete 3D scene.

### Key Designs
1. **Contextual Conditioning**:

    - **Geometric conditioning**: The fragmented point cloud $G_{\text{frag}}$ and its mask are directly fed as 3D spatial anchors, providing scale and orientation information. Self-attention with depth-ratio embeddings and cross-attention with global features handle varying degrees of distortion.
    - **Texture conditioning (GAFP)**: High-resolution 2D image features from MoGe v2 are projected onto the 3D voxel coordinates of visible point clouds, establishing precise spatial correspondences, and are injected into DiT blocks layer-by-layer to guide texture generation.
    - **Design Motivation**: Existing methods relying solely on 2D information suffer from scale/orientation ambiguity; explicit 3D information provides strong constraints.

2. **Coarse-to-Fine Generation**:

    - **Coarse stage**: Computes the AABB of the visible point cloud, expands it by a factor of 4 to obtain a conservative bounding box $B_{\text{exp}}$, and predicts the full bounding box $B_{\text{full}}$ within this range.
    - **Fine stage**: Generates high-resolution, high-fidelity geometry within the predicted tight bounding box.
    - **Design Motivation**: Occlusion causes severe bounding box ambiguity (the visible portion may be far smaller than the complete object); decoupling boundary prediction from detail generation allows each stage to specialize.

3. **Occlusion-Robust Feature Alignment (ORFA)**:
   A frozen pretrained TRELLIS model serves as a teacher for layer-wise knowledge distillation to the student model:
    $$\mathcal{L}_{\text{AL}} = -\mathbb{E}\Big[\frac{1}{N}\sum_{n=1}^{N} \text{sim}(\mathbf{h}_s, \mathbf{h})\Big]$$
   The teacher receives clean images while the student receives occluded inputs; aligning intermediate representations stabilizes training.
    - **Design Motivation**: Object-level priors are trained on occlusion-free data; scene-level occlusion introduces a severe domain gap, and direct adaptation leads to training instability.

### Loss & Training
- Base loss: Flow Matching loss $L_{\text{FM}}$
- Alignment loss: $L_{\text{AL}}$ — cosine similarity between teacher and student intermediate features
- Built upon the TRELLIS architecture, extended to a dual-branch design (frozen original branch + trainable scene branch)

## Key Experimental Results

### Main Results

| Dataset | Metric | 3D-Fixer | MIDI | Gen3DSR | Gain (vs. MIDI) |
|--------|------|------|------|---------|------|
| MIDI testset | CD_S ↓ | **0.069** | 0.080 | 0.123 | +13.8% |
| MIDI testset | FS_S ↑ | **78.67** | 50.19 | 40.07 | +56.8% |
| MIDI testset | CD_O ↓ | **0.032** | 0.103 | 0.157 | +68.9% |
| MIDI testset | FS_O ↑ | **94.39** | 53.58 | 38.11 | +76.2% |
| MIDI testset | Inference Time | **30s** | 40s | 9min | Faster |

### Ablation Study

| Configuration | Key Metric | Remarks |
|------|---------|------|
| w/o ORFA | CD_O increases | Occlusion causes training instability |
| w/o coarse-to-fine | Bounding box prediction fails | Cannot handle heavy occlusion |
| w/o geometric conditioning | Scale/orientation ambiguity | 2D conditioning insufficient |
| w/o GAFP | Texture quality degrades | Lack of precise spatial correspondence |

### Key Findings
- Object-level CD drops from 0.103 to 0.032 (69% reduction), demonstrating that in-place completion avoids accumulated alignment errors.
- F-Score increases from 53.58 to 94.39, achieving near-perfect geometric recovery.
- Inference takes 30 seconds — 18× faster than Gen3DSR and faster than MIDI.
- Generalizes to complex scenes, real-world scenes, and outdoor scenes.

## Highlights & Insights
- **Paradigm innovation**: In-place completion cleverly leverages the visible portion recovered by geometry estimation as spatial anchors, entirely bypassing pose alignment as a source of error accumulation.
- **ARSG-110K**: 110K scenes, 180K+ assets, and 3 million annotated images — currently the largest scene-level dataset.
- **Coarse-to-fine decoupling**: Separating scale prediction from geometry generation is an elegant design for handling occlusion.

## Limitations & Future Work
- The pipeline depends on the quality of the geometry estimation model (MoGe v2); failures in estimation propagate through the entire pipeline.
- Currently limited to rigid objects; deformable objects (e.g., cloth, human bodies) are not covered.
- ARSG-110K is synthetic, and a domain gap with real scenes remains.
- Occlusion relationship reasoning during parallel multi-instance completion may lack sufficient granularity.

## Related Work & Insights
- Compared to MIDI (multi-instance diffusion) and Gen3DSR (divide-and-conquer), 3D-Fixer achieves a superior trade-off between the two paradigms.
- Advances in geometric foundation models (MoGe v2, UniDepth) are what make the in-place completion paradigm feasible.
- The idea of "anchoring generation with partial observations" is generalizable to other generation tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The in-place completion paradigm is original; the ORFA training strategy is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset comparisons with complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and precise problem formulation.
- Value: ⭐⭐⭐⭐⭐ Dual contributions of paradigm and dataset; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](../../ICLR2026/3d_vision/generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers](affordmatcher_affordance_learning_in_3d_scenes_from_visual_signifiers.md)

</div>

<!-- RELATED:END -->
