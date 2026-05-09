---
title: >-
  [Paper Note] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions
description: >-
  [ICCV 2025][3D Vision][Human-scene interaction] This work formally introduces the scene-aware motion in-betweening problem and proposes the SceneMI framework, which comprehensively encodes scene context via a dual-layer scene descriptor (global voxels + local BPS). By leveraging the denoising capability of diffusion models to handle noisy keyframes, SceneMI reduces the collision frame rate by 56.9% on TRUMANS, and reduces foot skating by 37.5% and jitter by 56.5% on the real-world GIMO dataset.
tags:
  - ICCV 2025
  - 3D Vision
  - Human-scene interaction
  - motion in-betweening
  - diffusion models
  - scene encoding
  - keyframe animation
date: 2026-05-08
content_hash: 3e5f1c856ab0ed6b
---

# SceneMI: Motion In-betweening for Modeling Human-Scene Interactions

**Conference**: ICCV 2025  
**arXiv**: [2503.16289](https://arxiv.org/abs/2503.16289)  
**Code**: [inwoohwang.me/SceneMI](http://inwoohwang.me/SceneMI)  
**Area**: 3D Vision  
**Keywords**: Human-scene interaction, motion in-betweening, diffusion models, scene encoding, keyframe animation

## TL;DR

This work formally introduces the scene-aware motion in-betweening problem and proposes the SceneMI framework, which comprehensively encodes scene context via a dual-layer scene descriptor (global voxels + local BPS). By leveraging the denoising capability of diffusion models to handle noisy keyframes, SceneMI reduces the collision frame rate by 56.9% on TRUMANS, and reduces foot skating by 37.5% and jitter by 56.5% on the real-world GIMO dataset.

## Background & Motivation

### Problem Definition

Given a 3D scene $\mathcal{G}$ and a sparse set of keyframe poses $\mathbf{s} \in \mathbb{R}^{N \times D}$ with keyframe indicators $\mathbf{m} \in \{0,1\}^N$, the goal is to synthesize a complete motion sequence $\mathbf{x} = \{x^n\} \in \mathbb{R}^{N \times D}$ that simultaneously satisfies keyframe constraints and environmental constraints imposed by the 3D scene (e.g., obstacle avoidance, physical plausibility).

### Limitations of Prior Work

**Lack of controllability in scene-aware motion synthesis**: Methods such as HUMANISE and TRUMANS generate motion from text or action labels, but do not allow users to precisely control motion details (i.e., what action occurs, when, and at which location).

**Scene-agnostic motion in-betweening**: Conventional motion in-betweening methods (RNN, Transformer, etc.) assume an open environment; directly applying them to 3D scenes leads to human-object penetration artifacts.

**Strong assumptions in existing scene-aware approaches**: The few methods that incorporate scene context (e.g., goal-pose reaching) rely on conditional VAEs, which suffer from limited expressiveness and scalability.

**Noise in real-world data**: In practice, keyframes may be obtained from imprecise motion capture sensors or video-based pose estimation, yet existing methods assume clean keyframes.

### Core Problem

**Key insight**: Motion in-betweening is a more controllable and practically useful task than unconstrained motion generation — keyframes provide sufficient constraints to reduce task complexity. Scene-aware motion in-betweening supports a variety of real-world applications: (1) animators can create character animations in 3D scenes via keyframes; (2) motion artifacts in real-world human-scene interaction data can be automatically repaired; and (3) human-scene interaction quality reconstructed from monocular video can be enhanced. Furthermore, diffusion models possess an inherent denoising capability that can be analogously applied to handle noisy keyframes.

## Method

### Overall Architecture

Global voxel features $\mathbf{c}_g$ and keyframe-centered local BPS features $\mathbf{c}_l$ are extracted from the 3D scene. During training, keyframes in the motion sequence are handled via imputation (replacement). The model integrates scene features, body shape features $\mathbf{b}$, and diffusion timestep $t$ to synthesize a complete motion sequence that satisfies both keyframe and scene constraints.

### Key Designs

#### 1. **Dual-Layer Scene Encoding**

- **Function**: Compactly and comprehensively encodes both global and local scene context.
- **Mechanism**:

  **Global scene features**: The entire scene is represented as a coarse occupancy voxel grid $\mathbf{c}_g \in \{0,1\}^{d_x \times d_y \times d_z}$ (0.1 m/voxel), centered at the root position and orientation of the first frame. A ViT encoder processes this representation ($48 \times 24 \times 48$ input → 512-dimensional feature vector), providing global spatial layout to guide overall motion trajectory planning.

  **Keyframe-centered local scene features**: The Basis Point Set (BPS) method is employed. Sixty-four anchor points are determined on the T-pose SMPL mesh surface via farthest point sampling (FPS), with fixed indices. For each keyframe, the offset vector from the nearest scene point to each anchor is computed as $\mathbf{c}_l^n \in \mathbb{R}^{64 \times 3}$. After MLP embedding, these features are concatenated with the corresponding keyframe features at their respective positions.

- **Design Motivation**: Global features capture the "big picture" (room layout, furniture positions), while local features encode "nearest obstacles at keyframe locations." BPS features are insensitive to point ordering, mesh topology, and resolution, enhancing generalization across scenes from different sources (handcrafted vs. scanned). The two feature types are complementary: global features guide path planning, while local features prevent fine-grained collisions.

#### 2. **Scene-Aware Motion In-betweening Diffusion Model**

- **Function**: Leverages a diffusion model to generate motion sequences satisfying both keyframe and scene constraints.
- **Mechanism**:

  **Motion representation**: Each frame consists of global joint positions $J \in \mathbb{R}^{22 \times 3}$, 6D root orientation $\phi \in \mathbb{R}^6$, and SMPL pose parameters $\psi \in \mathbb{R}^{21 \times 6}$, totaling 201 dimensions. Body shape features $\mathbf{b} \in \mathbb{R}^7$ encode pairwise distances between representative joints.

  **Training-time imputation**: $k$ keyframes (including the first and last frames) are randomly selected. At keyframe positions in the noisy sample $\mathbf{x}_t$, the noisy values are replaced with clean values:
  $$\mathbf{x}_t' = \mathbf{m} \odot \mathbf{x}_0 + (1-\mathbf{m}) \odot \mathbf{x}_t$$
  Local scene features are visible only at keyframe positions: $\mathbf{c}_l' = \mathbf{m} \odot \mathbf{c}_l$

  **Model architecture**: U-Net + Adaptive Group Normalization + 1D convolution. AdaGN dynamically adapts normalization, and 1D convolution learns sequential motion patterns.

  **Inference**: Imputation is applied at each denoising step, and classifier-free guidance ($\mathbf{w}=2.5$) is used:
  $$\hat{\mathbf{x}}_0 = \mathbf{w} \cdot \mathcal{D}_\theta(\tilde{\mathbf{x}}_t, t, \mathbf{b}, \mathbf{c}_g) + (1-\mathbf{w}) \cdot \mathcal{D}_\theta(\tilde{\mathbf{x}}_t, t, \mathbf{b}, \emptyset)$$

- **Design Motivation**: Imputation is the most natural way to incorporate keyframe constraints within a diffusion framework — it directly injects clean signals into specific positions of the noisy sequence. U-Net + AdaGN has been shown to be effective for global motion representations.

#### 3. **Noisy Keyframe Handling**

- **Function**: Exploits the denoising capability of diffusion models to handle imprecise real-world keyframes.
- **Mechanism**:

  The diffusion/sampling timesteps are divided into two stages:
  - **Stage 1** $[T, T^*+1]$: Imputation is performed with noisy keyframes $\mathbf{s}^{\text{noisy}}$, guiding the overall motion structure.
  - **Stage 2** $[T^*, 1]$: Imputation is stopped; the diffusion model freely denoises the entire sequence (including keyframe positions) to correct for noise.

  $$\mathbf{x}_t' = \begin{cases} \mathbf{m} \odot \mathbf{x}_0^{\text{noisy}} + (1-\mathbf{m}) \odot \mathbf{x}_t, & t \in [T, T^*+1] \\ \mathbf{x}_t, & t \in [T^*, 1] \end{cases}$$

  During training, noisy training data is generated by adding random noise ($l \sim \mathcal{U}(0, 1.0)$) to clean motion sequences, and the model learns to recover the clean motion.

- **Design Motivation**: The noise present in noisy keyframes can be treated as analogous to additive noise in the diffusion process. $T^*=20$ (with $T=1000$) is the optimal threshold — larger $T^*$ yields smoother motion but reduces keyframe fidelity, while smaller $T^*$ preserves keyframe accuracy at the cost of insufficient denoising.

### Loss & Training

Total training loss:
$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda_{\text{joints}} \mathcal{L}_{\text{joints}} + \lambda_{\text{vel}} \mathcal{L}_{\text{vel}}$$

- $\mathcal{L}_{\text{simple}}$: Standard diffusion reconstruction loss (predicting $\mathbf{x}_0$)
- $\mathcal{L}_{\text{joints}}$: 3D joint position loss computed via forward kinematics
- $\mathcal{L}_{\text{vel}}$: Joint velocity loss (promotes motion smoothness)
- $\lambda_{\text{joints}}=2.0$, $\lambda_{\text{vel}}=10.0$

Training uses $T=1000$ steps on a single RTX 3090 GPU. Training data: TRUMANS.

## Key Experimental Results

### Main Results

**TRUMANS evaluation with clean keyframes** (keyframe interval: 60 frames):

| Method | FID↓ | Foot Skating↓ | Jerk↓ | MJPE Key(m)↓ | MJPE All(m)↓ | Collision Rate↓ | Penetration Max(m)↓ |
|--------|------|--------------|-------|-------------|-------------|----------------|-------------------|
| MDM | 1.422 | 0.316 | 0.972 | 0.568 | 0.576 | 0.317 | 0.112 |
| OmniControl | 0.371 | 0.294 | 0.274 | 0.217 | 0.294 | 0.211 | 0.081 |
| CondMDI | 0.943 | 0.281 | 0.305 | 0.452 | 0.457 | 0.262 | 0.087 |
| **SceneMI** | **0.123** | **0.248** | **0.194** | **0.006** | **0.023** | **0.113** | **0.043** |

**Real-world GIMO evaluation** (keyframe interval: 15 frames):

| Method | Foot Skating↓ | Accel↓ | Jerk↓ | Collision Rate↓ | Penetration Max(m)↓ |
|--------|--------------|-------|-------|----------------|-------------------|
| GIMO original data | 0.261 | 0.347 | 0.573 | 0.057 | 0.048 |
| CondMDI | 0.312 | 0.359 | 0.498 | 0.091 | 0.083 |
| **SceneMI** | **0.163** | **0.165** | **0.249** | **0.060** | **0.047** |

### Ablation Study

**Scene encoding component ablation** (TRUMANS, clean keyframes):

| Configuration | FID↓ | MJPE All(m)↓ | Collision Rate↓ |
|--------------|------|-------------|----------------|
| w/o scene awareness ($\mathbf{c}_g, \mathbf{c}_l$) | 0.136 | 0.059 | 0.131 |
| w/o global features $\mathbf{c}_g$ | 0.138 | 0.051 | 0.128 |
| w/o local features $\mathbf{c}_l$ | 0.125 | 0.036 | 0.119 |
| **Full model** | **0.123** | **0.023** | **0.113** |

**Noise-aware $T^*$ ablation** (TRUMANS, noisy keyframes, interval: 3 frames):

| $T^*$ | FID↓ | Foot Skating↓ | Jerk↓ | MJPE Key(m)↓ | Collision Rate↓ |
|-------|------|--------------|-------|-------------|----------------|
| 0 (no noise awareness) | 0.157 | 0.265 | 0.230 | 0.015 | 0.119 |
| 10 | 0.123 | 0.253 | 0.199 | 0.013 | 0.110 |
| **20** | **0.118** | **0.247** | **0.198** | **0.013** | **0.108** |
| 40 | 0.121 | 0.249 | 0.187 | 0.014 | 0.112 |
| 60 | 0.122 | 0.250 | 0.189 | 0.015 | 0.114 |

**Component decomposition on GIMO**:

| Configuration | Foot Skating↓ | Jerk↓ | Collision Rate↓ |
|--------------|--------------|-------|----------------|
| w/o scene awareness | 0.192 | 0.245 | **0.082** |
| w/o noise awareness | 0.391 | 0.301 | 0.072 |
| **Full model** | **0.163** | **0.249** | 0.060 |

### Key Findings

1. **Substantial reduction in collision rate**: SceneMI achieves a collision frame rate of only 0.113 on TRUMANS (vs. 0.262 for CondMDI and 0.317 for MDM).
2. **Highly accurate keyframe alignment**: MJPE Key is only 0.006 m, two orders of magnitude lower than OmniControl (0.217 m).
3. **Significant repair on real-world GIMO data**: Foot skating is reduced by 37.5% (0.261→0.163) and jitter by 56.5% (0.573→0.249).
4. **Noise awareness is critical**: On GIMO, removing noise awareness degrades foot skating to 0.391 — worse than the original data — demonstrating that naively applying noisy keyframes via imputation is harmful.
5. **Scene awareness and noise awareness are complementary**: Scene awareness primarily reduces collisions, while noise awareness primarily improves motion quality.

## Highlights & Insights

1. **Practically valuable problem formulation**: Recasting HSI modeling as a motion in-betweening task reduces problem complexity while increasing practical utility (animation production, data augmentation, video reconstruction).
2. **Elegant BPS local scene encoding**: Fixed anchor points combined with offsets to the nearest scene points yield a representation that is agnostic to scene representation format, which is key to generalizing to real-world scanned scenes.
3. **Analogy between diffusion denoising and noisy keyframes**: The approach cleverly leverages the inherent denoising capability of diffusion models, framing noisy keyframe handling as "early stopping of imputation followed by free denoising."
4. **Generalization from synthetic to real**: The model, trained solely on TRUMANS (handcrafted scenes + MoCap data), generalizes to GIMO (phone-scanned scenes + IMU data) — enabled by the scene-agnostic properties of BPS and voxel representations.
5. **First monocular video HSI reconstruction pipeline**: Combines image-to-3D, depth estimation, human pose estimation, and SceneMI into a complete end-to-end pipeline.

## Limitations & Future Work

1. **Dependence on full-body keyframes**: Flexibility is limited when only partial joint positions are available.
2. **No text conditioning**: Motion style or semantics cannot be guided via natural language.
3. **Limitations of feature-level fusion**: Human-scene interaction is modeled primarily through feature concatenation; model-level fusion may be more expressive.
4. **Inference speed**: Sampling with $T=1000$ steps is slow; acceleration strategies such as DDIM have not been explored.
5. **Absence of hand interactions**: The 22-joint representation does not include fingers, precluding the modeling of fine-grained object manipulation.

## Related Work & Insights

- **Distinction from CondMDI**: CondMDI relies on motion velocity features as input, which are difficult to obtain in practice; SceneMI requires only keyframe poses.
- **Distinction from SceneDiffuser**: SceneDiffuser combines diffusion with reinforcement learning to extend motion from a starting pose and lacks keyframe alignment capability.
- **Elegant use of BPS**: Encoding offsets from fixed body surface anchors to nearest scene points provides a point-order-invariant representation of the relative scene-body relationship.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First formal study of scene-aware motion in-betweening; the noisy keyframe handling design is particularly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers clean keyframes, noisy keyframes, real-world data, and video reconstruction, with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated and experiments are well organized.
- **Value**: ⭐⭐⭐⭐⭐ — High practical applicability; establishes a useful new paradigm for the HSI research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ATLAS: Decoupling Skeletal and Shape Parameters for Expressive Parametric Human Modeling](atlas_decoupling_skeletal_and_shape_parameters_for_expressive_parametric_human_m.md)
- [\[ICCV 2025\] Global Motion Corresponder for 3D Point-Based Scene Interpolation under Large Motion](global_motion_corresponder_for_3d_point-based_scene_interpolation_under_large_mo.md)
- [\[ICCV 2025\] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)
- [\[ICCV 2025\] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)
- [\[ICCV 2025\] Estimating 2D Camera Motion with Hybrid Motion Basis](estimating_2d_camera_motion_with_hybrid_motion_basis.md)

</div>

<!-- RELATED:END -->
