---
title: >-
  [Paper Note] Geometry-aware 4D Video Generation for Robot Manipulation
description: >-
  [ICLR 2026][Video Generation][4D video generation] This paper proposes a geometry-aware 4D video generation framework that trains a video diffusion model via cross-view pointmap alignment supervision…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "4D video generation"
  - "robot manipulation"
  - "cross-view consistency"
  - "pointmap alignment"
  - "pose estimation"
date: 2026-05-08
content_hash: a3f85a9a1b34a1c3
---

# Geometry-aware 4D Video Generation for Robot Manipulation

**Conference**: ICLR 2026
**arXiv**: [2507.01099](https://arxiv.org/abs/2507.01099)
**Code**: [Project Page](https://robot4dgen.github.io/)
**Area**: Video Generation
**Keywords**: 4D video generation, robot manipulation, cross-view consistency, pointmap alignment, pose estimation

## TL;DR
This paper proposes a geometry-aware 4D video generation framework that trains a video diffusion model via cross-view pointmap alignment supervision, jointly predicting RGB and pointmap sequences to achieve spatiotemporally consistent multi-view RGB-D videos. Without requiring camera pose inputs at inference, the framework generates consistent videos from novel viewpoints and recovers robot end-effector trajectories using an off-the-shelf 6DoF pose tracker.

## Background & Motivation

1. **Background**: Video generation models (e.g., SVD) are increasingly used as visual dynamics models for robot planning. Approaches to extracting robot actions from predicted videos include inverse dynamics models, behavior cloning, and RGB-based pose tracking.
2. **Limitations of Prior Work**: (1) Pixel-space video models excel at short-horizon motion but lack 3D structural understanding, leading to flickering, distortion, and object disappearance; (2) 3D-aware methods enforce geometric constraints but are restricted to simple static backgrounds and struggle to scale to complex multi-object scenes; (3) existing methods suffer significant performance degradation under novel camera viewpoints.
3. **Key Challenge**: Temporal consistency and 3D consistency are difficult to achieve simultaneously. Single-view prediction lacks geometric grounding, while multi-view methods either optimize temporal and spatial consistency separately or handle only single objects against white backgrounds.
4. **Goal**: How to generate 4D videos that are simultaneously temporally coherent and cross-view 3D consistent, and how to recover robot manipulation trajectories from them?
5. **Key Insight**: Drawing inspiration from DUSt3R's cross-view pointmap alignment, this work adapts the idea to video generation by supervising the model during training to project pointmap predictions from one viewpoint into another viewpoint's coordinate system.
6. **Core Idea**: Cross-view pointmap alignment serves as geometric supervision for training a video diffusion model to learn a shared 3D scene representation. At inference, the model generates cross-view consistent 4D videos without requiring camera pose inputs.

## Method

### Overall Architecture
Built upon Stable Video Diffusion (SVD). Each viewpoint independently predicts RGB video and pointmap sequences. Pointmap prediction has two output branches: the pointmap $X_t^n$ of viewpoint $v_n$ in its own coordinate system, and the projected pointmap $X_t^{m \to n}$ of viewpoint $v_m$ expressed in $v_n$'s coordinate system. Both branches are supervised jointly during training to enforce 3D consistency. The U-Net decoder employs a dual-branch architecture with cross-attention mechanisms.

### Key Designs

1. **Cross-view Geometric Consistency Supervision**:
    - Function: Enforces the model to learn a shared 3D scene representation across viewpoints.
    - Mechanism: The reference viewpoint $v_n$ predicts its own pointmap $X_t^n$; the second viewpoint $v_m$ predicts not in its own coordinate system but **projected into $v_n$'s coordinate system** as $X_t^{m \to n}$. Both branches are supervised with diffusion losses: $\mathcal{L}_{\text{3D-diff}}(t') = \mathbb{E}\|z_{t'}^n(0) - f_\theta(z_{t'}^n(k), k, c^n)\|^2 + \mathbb{E}\|z_{t'}^{m \to n}(0) - f_\theta(z_{t'}^{m \to n}(k), k, c^m)\|^2$. Camera poses are required during training to define the projection relationship, but at inference the model directly predicts the pointmap of another viewpoint in the reference coordinate system from a single-frame RGB-D input, **without camera pose as input**.
    - Design Motivation: Inspired by the success of DUSt3R, cross-view pointmap alignment provides the most direct supervision signal for enforcing 3D consistency. The model internalizes the inter-view geometric mapping during training.

2. **Multi-view Cross-attention Mechanism**:
    - Function: Enables cross-view information transfer in the pointmap prediction U-Net decoder.
    - Mechanism: RGB video prediction shares a single U-Net across all viewpoints, as each viewpoint predicts independently in its own coordinate system. However, pointmap prediction requires alignment to a reference coordinate system, so two decoder branches with independent weights are used, augmented with cross-attention layers: intermediate features from $v_n$'s decoder are passed via cross-attention to $v_m$'s decoder, enabling the $v_m$ branch to attend to geometric cues from $v_n$ and accurately predict pointmaps in $v_n$'s coordinate system.
    - Design Motivation: RGB prediction can proceed independently per viewpoint, but the asymmetric nature of pointmap prediction ($v_n$ predicts its own; $v_m$ predicts in $v_n$'s frame) necessitates separate decoders and information transfer. Cross-attention realizes asymmetric geometric information propagation.

3. **Joint Temporal–3D Consistency Optimization**:
    - Function: Unifies temporal coherence and 3D spatial consistency within a single framework.
    - Mechanism: The total loss combines RGB diffusion loss and pointmap 3D diffusion loss with a weight: $\mathcal{L} = \sum_{t'}[\mathcal{L}_{\text{diff}}^n(t') + \mathcal{L}_{\text{diff}}^m(t') + \lambda \cdot \mathcal{L}_{\text{3D-diff}}(t')]$, with $\lambda=1$. The Pointmap VAE is initialized from a pretrained RGB VAE and fine-tuned on pointmap data. Pretrained SVD weights provide a strong temporal prior.
    - Design Motivation: SVD's temporal prior and pointmap alignment's 3D supervision are complementary. Joint optimization allows the model to leverage both motion knowledge from large-scale video pretraining and geometric constraints.

### Loss & Training
DDPM denoising formulation (direct clean data prediction). Pointmap VAE fine-tuning. Dual-view training requires known camera poses to compute projection ground truth. Training uses 25 demonstrations × 16 camera viewpoints = 400 videos per task (12 viewpoints for training, 4 for testing).

## Key Experimental Results

### Main Results

| Method | Cross-view mIoU↑ | FVD-nn↓ | FVD-mm↓ | AbsRel-nn↓ | δ1-nn↑ |
|--------|------|------|----------|------|------|
| 4D Gaussian | 0.39–0.46 | 1208–1396 | 815–1192 | 0.18–0.33 | 0.43–0.80 |
| SVD | — | 370–977 | 417–743 | — | — |
| SVD w/ MV attn | — | 536–942 | 445–767 | — | — |
| Ours w/o MV attn | 0.26–0.44 | 451–597 | 302–607 | 0.10–0.15 | 0.75–0.89 |
| **Ours** | **0.64–0.70** | **378–491** | **258–561** | **0.03–0.06** | **0.95–0.98** |

### Ablation Study

| Configuration | mIoU↑ | AbsRel↓ | Note |
|------|---------|------|------|
| Full model | 0.64–0.70 | 0.03–0.06 | Cross-attention + cross-view supervision |
| w/o MV attention | 0.26–0.44 | 0.10–0.15 | Removing cross-view attention drastically degrades consistency |
| SVD baseline | — | — | RGB only, no 3D supervision |

Robot manipulation success rate (novel viewpoint):

| Task | Ours | Baseline |
|------|------|------|
| StoreCerealBoxUnderShelf | Higher | Lower |
| PutSpatulaOnTable | Higher | Lower |
| PlaceAppleFromBowlIntoBin | Higher | Lower |

### Key Findings
- Cross-view attention is critical for 3D consistency: removing it drops mIoU from 0.70 to 0.41 (Task 1).
- The proposed method maintains strong consistency on novel viewpoints unseen during training, demonstrating that the model learns a generalizable 3D representation.
- Pointmap depth quality is exceptionally high: AbsRel of only 0.03–0.06, far outperforming 4D Gaussian's 0.20+.
- The ability to operate without camera pose input at inference is highly practical for deployment, eliminating the need for pose calibration.
- End-effector trajectories recovered from 4D videos via FoundationPose can directly control robots to execute tasks.

## Highlights & Insights
- The design of **using poses during training but not at inference** is elegant: the model internalizes inter-view geometric mappings.
- A natural transfer of DUSt3R's ideas from static reconstruction to 4D video generation.
- Joint RGB + pointmap prediction (rather than RGB-only or depth-only) provides the most complete 4D information.
- End-effector pose tracking closes the full loop from generation to control.
- Bimanual manipulation tasks (PlaceAppleFromBowlIntoBin) validate effectiveness over long time horizons.

## Limitations & Future Work
- Supports only dual-view settings; extension to more viewpoints remains unexplored.
- Data acquisition is non-trivial, requiring 25 demonstrations × 16 viewpoints per task.
- The underlying SVD model may limit visual quality.
- Gripper state inference relies on a simple distance threshold, which is not sufficiently robust.
- Real-robot experiments are conducted only in simulation; real-world validation is limited.

## Related Work & Insights
- **vs. DUSt3R**: DUSt3R targets static 3D reconstruction; this work extends cross-view pointmap alignment to video generation.
- **vs. 4D Gaussian**: Optimizes temporal and spatial consistency separately; the proposed method performs tighter joint optimization.
- **vs. UniPi/SuSIE**: Extract actions from video without considering 3D consistency.
- **vs. CamAnimate/CameraCtrl**: Require camera poses as inference inputs.

## Rating
- Novelty: ⭐⭐⭐⭐ Transfer of DUSt3R to 4D video generation + pose-free inference design
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 simulation tasks + 4 real tasks, though real-world manipulation experiments are limited
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and method description is detailed
- Value: ⭐⭐⭐⭐ The complete loop from 4D video generation to robot manipulation carries significant practical importance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control](learning_video_generation_for_robotic_manipulation_with_collaborative_trajectory.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](../../CVPR2026/video_generation/seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](../../CVPR2026/video_generation/geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[ICLR 2026\] Target-Aware Video Diffusion Models](target-aware_video_diffusion_models.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)

</div>

<!-- RELATED:END -->
