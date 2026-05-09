---
title: >-
  [Paper Note] DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction
description: >-
  [CVPR 2026][3D Vision][Human motion reconstruction] DuoMo decomposes world-space human motion reconstruction into two independent diffusion models: a camera-space model that extracts generalizable motion estimates from video in camera coordinates, and a world-space model that refines the noisy lifted proposals into globally consistent world-space motion. By directly generating mesh vertex motion rather than SMPL parameters, DuoMo reduces W-MPJPE by 16% on EMDB and 30% on RICH.
tags:
  - CVPR 2026
  - 3D Vision
  - Human motion reconstruction
  - diffusion model
  - world coordinate system
  - camera space
  - mesh vertices
date: 2026-05-08
content_hash: f0e4bd72e2f75b7d
---

# DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.03265](https://arxiv.org/abs/2603.03265)
**Code**: [Project Page](https://yufu-wang.github.io/duomo/)
**Area**: 3D Vision
**Keywords**: Human motion reconstruction, diffusion model, world coordinate system, camera space, mesh vertices

## TL;DR

DuoMo decomposes world-space human motion reconstruction into two independent diffusion models: a camera-space model that extracts generalizable motion estimates from video in camera coordinates, and a world-space model that refines the noisy lifted proposals into globally consistent world-space motion. By directly generating mesh vertex motion rather than SMPL parameters, DuoMo reduces W-MPJPE by 16% on EMDB and 30% on RICH.

## Background & Motivation

Reconstructing human motion in world coordinates from monocular video is fundamental to understanding human behavior, embodied AI, and human-computer interaction. Recent research has shifted focus from isolated pose sequence analysis to recovering motion in a consistent world coordinate system. However, existing methods face a **fundamental trade-off**:

**Direct prediction methods** (WHAM, GVHMR, GENMO): End-to-end models learn a mapping from video to world-space motion. While they capture strong global priors, their generalization to complex in-the-wild scenes is limited by the motion diversity of lab-collected training data.

**Lifting methods** (TRAM, PromptHMR + post-processing): These first estimate human pose in camera coordinates, then lift the result to world coordinates using estimated camera parameters. Camera-space estimation generalizes well (leveraging abundant 2D supervision), but the motion prior is inherently local and cannot guarantee global physical plausibility (e.g., foot skating, drift).

**Root Cause**: Generalizability (from camera space) and global consistency (requiring world-space priors) are difficult to achieve simultaneously within a single model.

**Starting Point**: Rather than forcing a single end-to-end model to address both 2D-to-3D lifting and global consistency, the paper decomposes the problem into a two-stage generative process. Two core insights guide this design:
- **Connection mechanism**: The camera-space output is explicitly lifted to world coordinates via a known geometric transformation (camera pose), rather than requiring the model to implicitly learn this geometric relationship.
- **World coordinate definition**: Instead of a fixed canonical coordinate system (which requires ground-plane alignment and is error-prone), the world coordinate system for each video is defined by the first-frame camera pose, allowing the model to denoise across diverse coordinate frames.

## Method

### Overall Architecture

DuoMo follows a two-stage pipeline:

1. **Camera-space Model** $\mathcal{D}_{\text{cam}}$: Extracts video features and generates human motion $\mathbf{C}$ in camera coordinates.
2. **Explicit Lifting**: Lifts $\mathbf{C}$ to world coordinates using estimated camera poses $\mathbf{g}_t$, yielding a noisy proposal $\hat{\mathbf{X}}_t^1 = \mathbf{g}_t(\mathbf{X}_t^t)$.
3. **World-space Model** $\mathcal{D}_{\text{world}}$: Conditioned on the noisy proposal, generates clean, globally consistent world-space motion $\mathbf{W}$.

### Key Designs

1. **Motion Representation — Direct Mesh Vertex Generation**: Rather than relying on the SMPL parametric model, DuoMo directly generates 3D coordinates of 595 mesh vertices (LOD6 sparse mesh). In camera space, each frame is decomposed into a root-centered mesh $\mathbf{P}_t$ and a root position $\mathbf{r}_t$. In world space, since the root position $\mathbf{r}_t^1$ grows unboundedly over time, the model instead generates velocity $\mathbf{v}_t^1 = \mathbf{r}_t^1 - \mathbf{r}_{t-1}^1$, with position recovered by integration. This general representation enables the approach to generalize to other object categories.

2. **Camera-space Diffusion Model**:

    - **Input features**: Two types of features are extracted per frame — (a) dense keypoints converted to ray directions $\gamma(\mathbf{K}_t^{-1} \cdot \mathbf{L}_t)$ (implicitly encoding camera intrinsics), processed by an MLP to yield $\mathbf{f}_t^{\text{kpt}}$; (b) image encoder features $\mathbf{f}_t^{\text{img}}$; the two are summed.
    - **Architecture**: Standard DiT with RoPE relative positional encoding and windowed attention, enabling long-video inference without segmentation.
    - **Height conditioning**: Body height can optionally be provided to resolve the scale ambiguity of monocular reconstruction. Height is encoded via an MLP and added to the diffusion timestep embedding. Experiments show height conditioning improves MPJPE by approximately 10%.

3. **World-space Diffusion Model**:

    - Conditioned on the lifted noisy motion $\hat{\mathbf{X}}_t^1$ (encoded by an MLP), the model generates clean world-space motion.
    - **Masked modeling**: During training, conditions for randomly selected frames are replaced with a learnable mask token, simulating person-invisible scenarios and enabling the model to generate plausible motion during occlusion.
    - **Per-video coordinate system**: World coordinates are defined relative to the first-frame camera of each video, eliminating the need for alignment to a fixed canonical space and greatly simplifying in-the-wild video processing.

4. **Guided Sampling (Test-time Guidance)**:

    - **2D reprojection guidance**: $\mathcal{L}_{\text{repro}} = \sum_t \|\mathbf{L}_t - \mathbf{K}_t \cdot \mathbf{g}_t^{-1}(\mathbf{X}_t^1)\|$, reprojects world-space motion back to the original video to correct drift from velocity integration.
    - **Displacement guidance**: During long occlusion segments, ensures that the integrated velocity displacement matches the person's disappearance and reappearance positions.

### Loss & Training

Camera-space model loss:
$$\mathcal{L}_{\text{Camera}} = \mathcal{L}_{\text{vertices}} + \mathcal{L}_{\text{position}} + \mathcal{L}_{\text{joints}}$$

World-space model loss (trained with the camera-space model frozen):
$$\mathcal{L}_{\text{World}} = \mathcal{L}_{\text{vertices}} + \mathcal{L}_{\text{velocity}} + \mathcal{L}_{\text{contact}}$$

Here $\mathcal{L}_{\text{contact}}$ is a **training-time contact loss** (distinct from post-hoc foot locking): an L1 loss on world-space foot vertices is applied only at frames where the feet contact the ground, reducing foot-skating artifacts at the source.

Both models are trained with AdamW for 1 million steps, learning rate $10^{-4}$, batch size 256, and sequence length $T=120$.

## Key Experimental Results

### Main Results

| Dataset | Metric | DuoMo | Prev. SOTA | Gain |
|---------|--------|-------|------------|------|
| EMDB | W-MPJPE (mm)↓ | **167.1** | 202.1 (GENMO) | −16.3% |
| EMDB | Foot Skating↓ | **3.7** | 3.5 (GVHMR) | Comparable |
| EMDB | Jitter↓ | **8.7** | 16.7 (GVHMR) | −47.9% |
| RICH | W-MPJPE (mm)↓ | **80.8** | 118.6 (GENMO) | −31.9% |
| RICH | Foot Skating↓ | **3.1** | 3.0 (GVHMR) | Comparable |
| EMDB | PA-MPJPE (mm)↓ | **41.7** | 42.5 (GENMO) | −1.9% |

Note: DuoMo (w/ height) further reduces W-MPJPE to 167.1 and MPJPE to 59.5 on EMDB.

### Ablation Study

| Configuration | WA-MPJPE | W-MPJPE | RTE | Jitter | FS | Notes |
|---------------|----------|---------|-----|--------|----|-------|
| World-model only (one stage) | 153.5 | 445.1 | 6.7 | 9.1 | 4.8 | Poor accuracy; single model struggles to balance all objectives |
| Cam-model + Lifting | 67.0 | 180.2 | 1.3 | 32.6 | 9.2 | Good accuracy but poor motion quality |
| **DuoMo** | **66.0** | **167.1** | **1.1** | **8.7** | **3.7** | Complementary strengths of both stages |

### Key Findings

- **Value of dual priors**: The world-space model alone has strong motion priors but poor accuracy; lifting alone yields good accuracy but severe jitter and foot skating. DuoMo achieves the best of both.
- **Mesh vs. SMPL representation**: World-Model-Mesh outperforms World-Model-SMPL by 17.7 mm in W-MPJPE (164.8 vs. 182.5), demonstrating that direct vertex generation is more accurate than parametric regression.
- **Robustness**: In occlusion scenarios from Egobody, DuoMo achieves W-MPJPE-Occ of 193.1, substantially outperforming Cam+Lifting at 688.1.
- **Robustness to camera noise**: W-MPJPE degrades far more slowly with increasing camera noise compared to the lifting baseline, with the world-space model acting as a generative regularizer.
- **Speed**: For a 20-second video at 30 FPS, total runtime on an H200 is approximately 36.5 s (keypoints 2 s + dense keypoints 3 s + image features 30 s + diffusion 1.5 s).

## Highlights & Insights

- The **decomposition strategy** is particularly elegant: rather than building one all-purpose model, each component is specialized — the camera-space model handles generalization while the world-space model enforces global consistency.
- The **per-video coordinate system** design is simple yet effective, avoiding the complexity of canonical space alignment and enabling the model to handle diverse terrains.
- **Training-time contact loss** is more principled than post-hoc foot locking.
- Bypassing the parametric model to directly generate mesh vertices opens a more general path toward motion modeling applicable to other object categories.

## Limitations & Future Work

- Image feature extraction takes 30 s (PromptHMR encoder), constituting the primary computational bottleneck.
- The world-space model outputs root velocity; integration over long sequences accumulates error, though guided sampling partially mitigates this.
- The method requires camera intrinsics and estimated camera poses, making it dependent on the quality of camera estimation for in-the-wild videos.
- Only single-person scenarios are supported; multi-person interaction is not addressed.
- Training data for the world-space model (AMASS + BEDLAM) primarily covers flat-ground motion, with limited coverage of complex terrains such as stairs or slopes.

## Related Work & Insights

- **vs. GENMO**: GENMO uses a single end-to-end conditional generative model, whereas DuoMo employs two decoupled generative models.
- **vs. SLAHMR**: SLAHMR applies optimization to post-process lifting results, whereas DuoMo uses a generative model to refine them.
- **Insight**: For complex visual estimation tasks, a strategy of "decomposition into multi-stage models + injection of known geometric transformations" is more robust than end-to-end approaches.
- The success of mesh vertex representation suggests that motion reconstruction for non-human objects (e.g., animals) need not rely on parametric models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Dual diffusion model decomposition, per-video coordinate system, and direct mesh vertex generation constitute a novel and meaningful combination of ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple datasets (EMDB/RICH/Egobody), detailed ablations, and robustness analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem definition is clear, trade-off analysis is thorough, and method presentation is fluent.
- **Value**: ⭐⭐⭐⭐⭐ — Significant advancement in world-space human reconstruction with large performance gains and strong methodological generality.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space](raynova_scale-temporal_autoregressive_world_modeling_in_ray_space.md)
- [\[CVPR 2026\] AnthroTAP: Learning Point Tracking with Real-World Motion](anthrotap_learning_point_tracking_with_real-world_motion.md)
- [\[CVPR 2026\] Iris: Bringing Real-World Priors into Diffusion Model for Monocular Depth Estimation](iris_bringing_realworld_priors_into_diffusion_model_for_monocular_depth_estimation.md)
- [\[CVPR 2026\] Fall Risk and Gait Analysis using World-Spaced 3D Human Mesh Recovery](fall_risk_gait_analysis_hmr.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)

<!-- RELATED:END -->
