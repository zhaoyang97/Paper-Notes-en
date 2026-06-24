---
title: >-
  [Paper Note] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction
description: >-
  [CVPR 2025][Autonomous Driving][Temporal Fusion] This paper proposes GDFusion, which re-interprets RNNs as gradient descent steps to unify the fusion of three temporal cues (scene-level, motion, and geometry). It improves performance by 1.4-4.8% mIoU over non-temporal baselines on Occ3D while reducing inference memory by 27-72%, demonstrating superior efficiency compared to multi-frame methods like SOLOFusion.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Temporal Fusion"
  - "Gradient Descent"
  - "RNN Re-interpretation"
  - "3D Semantic Occupancy"
  - "Memory-efficient"
date: 2026-05-08
content_hash: 0993baecc3eee03c
---

# GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction

**Conference**: CVPR 2025  
**arXiv**: [2411.16718](https://arxiv.org/abs/2411.16718)  
**Code**: [https://cdb342.github.io/GDFusion](https://cdb342.github.io/GDFusion)  
**Area**: Autonomous Driving / 3D Occupancy Prediction  
**Keywords**: Temporal Fusion, Gradient Descent, RNN Re-interpretation, 3D Semantic Occupancy, Memory-efficient

## TL;DR

This paper proposes GDFusion, which re-interprets RNNs as gradient descent steps to unify the fusion of three temporal cues (scene-level, motion, and geometry). It improves performance by 1.4-4.8% mIoU over non-temporal baselines on Occ3D while reducing inference memory by 27-72%, demonstrating superior efficiency compared to multi-frame methods like SOLOFusion.

## Background & Motivation

### Background

**Background**: Camera-only 3D semantic occupancy prediction requires integrating multi-frame information to compensate for the depth estimation uncertainty of single frames. Existing temporal fusion approaches (e.g., SOLOFusion, BEVDet4D) must cache multi-frame BEV features—requiring $N$ times the memory for $N$ frames—which constrains the frame capacity and real-time performance.

**Limitations of Prior Work**: (1) Caching multiple frames exhibits a massive memory footprint; (2) different temporal cues (scene priors, motion compensation, and depth consistency) are mixed together during representation learning, lacking structured understanding; (3) adapting to different occupancy prediction frameworks (e.g., BEVDetOcc, FlashOCC) requires redesigning custom fusion modules.

**Key Challenge**: While more comprehensive temporal fusion (with more frames) leads to better performance, the memory consumption scales linearly. A temporal fusion approach with a fixed memory footprint is highly desirable.

**Key Insight**: The RNN hidden state update $h^t = Ah^{t-1} + Bx^t$ can be re-interpreted as a gradient descent step $h^t = h^{t-1} - \eta \nabla \mathcal{L}^t$, where the hidden state serves as a compressed, fixed-size summary of all historical information.

**Core Idea**: RNN = Gradient Descent $\rightarrow$ leveraging a fixed-size hidden state to fuse scene, motion, and geometry temporal cues $\rightarrow$ memory-efficient temporal occupancy prediction.

## Method

### Key Designs

1. **Scene-level Fusion**: Self-supervised learning of environmental priors, where the hidden state step-by-step accumulates knowledge of "what the scene usually looks like." This can also adapt at test-time (similar to TTA).

2. **Motion Fusion**: Dynamic object motions are compensated using warp and residual prediction: $\mathcal{L}_m = \|\text{Warp}(H_m^{t-1}) - M^t\|^2$, using motion distillation via the Jacobian of sampling operations.

3. **Geometry Fusion**: Depth consistency check—if the warped depth from the previous frame is consistent with the current depth estimation, the geometry is deemed reliable. An adaptive gating mechanism $\eta_g = \text{sigmoid}(f(H_g^{t-1}, G^t))$ is utilized.

### Loss & Training

Independent loss functions for each fusion branch + final occupancy cross-entropy. The RNN hidden state is of a fixed size (equal to the single-frame feature size), eliminating the need for caching multiple frames.

## Key Experimental Results

| Method | Occ3D mIoU | Memory | FPS |
|------|-----------|------|-----|
| BEVDetOcc Baseline | 41.9 | - | - |
| + SOLOFusion | 43.4 | 1205MB | 32 |
| **+ GDFusion** | **43.6** | **980MB** | **40** |
| FlashOCC + GDFusion | +1.4-4.8 mIoU | -27~72% Memory | — |

### Ablation Study

| Temporal Cues | mIoU Contribution |
|---------|----------|
| Scene-level | ~0.5% |
| Motion | ~0.5% |
| Geometry | ~0.5% |
| All | **1.4-4.8%** |

### Key Findings
- Each of the three cues contributes around 0.5% individually, but combined they yield a 1.4-4.8% improvement, demonstrating a non-linear synergistic effect.
- Fixed-size hidden state vs. multi-frame caching: reduces memory by 27-72% with a higher FPS.
- Proposition 1 establishes a precise mathematical equivalence between RNN steps and gradient descent.

## Highlights & Insights
- **Re-interpreting RNN as Gradient Descent** provides a novel paradigm for understanding temporal fusion, where each update step acts as a "gradient correction" on historical knowledge.
- **Plug-and-play** — can be directly incorporated into various frameworks such as BEVDetOcc and FlashOCC.

## Limitations & Future Work
- Requires camera ego-motion transformation matrices.
- Motion estimation assumes smooth motions.
- Test-time scene adaptation requires a few iterations during inference.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel RNN-as-GD perspective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensively evaluated on Occ3D, SurroundOcc, and OpenOcc
- Writing Quality: ⭐⭐⭐⭐ Sound integration of theory and practice
- Value: ⭐⭐⭐⭐ Practical and versatile temporal fusion scheme

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)
- [\[CVPR 2025\] SDGOcc: Semantic and Depth-Guided BEV Transformation for 3D Multimodal Occupancy Prediction](sdgocc_semantic_and_depth-guided_birds-eye_view_transformation_for_3d_multimodal.md)
- [\[CVPR 2025\] ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [\[CVPR 2025\] OccMamba: Semantic Occupancy Prediction with State Space Models](occmamba_semantic_occupancy_prediction_with_state_space_models.md)
- [\[CVPR 2025\] M²-Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs](m2-occ_resilient_3d_semantic_occupancy_prediction_for_autonomous_driving_with_in.md)

</div>

<!-- RELATED:END -->
