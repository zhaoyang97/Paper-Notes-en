---
title: >-
  [Paper Note] RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space
description: >-
  [CVPR 2026][3D Vision][World Model] This paper proposes RayNova, a geometry-agnostic multi-view world model based on dual-causal (scale + temporal) autoregressive modeling. By leveraging relative Plücker ray positional e…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "World Model"
  - "Multi-View Video Generation"
  - "Autoregressive"
  - "Plücker Rays"
  - "Autonomous Driving"
date: 2026-05-08
content_hash: 3c1352b97bb323a1
---

# RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space

**Conference**: CVPR 2026
**arXiv**: [2602.20685](https://arxiv.org/abs/2602.20685)  
**Authors**: Yichen Xie, Chensheng Peng, Mazen Abdelfattah, Yihan Hu et al. (Applied Intuition, UC Berkeley)
**Code**: [Project Page](https://raynova-ai.github.io/)  
**Area**: 3D Vision
**Keywords**: World Model, Multi-View Video Generation, Autoregressive, Plücker Rays, Autonomous Driving

## TL;DR

This paper proposes RayNova, a geometry-agnostic multi-view world model based on dual-causal (scale + temporal) autoregressive modeling. By leveraging relative Plücker ray positional encodings, RayNova achieves unified 4D spatiotemporal reasoning and attains state-of-the-art multi-view video generation performance on nuScenes.

## Background & Motivation

World foundation models (WFMs) aim to simulate the physical evolution of real-world environments. Existing approaches suffer from fundamental limitations:

1. **Decoupled spatial-temporal design**: Spatial relationships are modeled via multi-view adjacency while temporal dynamics rely on video generation techniques, handled separately — limiting adaptability to novel camera configurations and fast motion.
2. **Reliance on strong 3D priors**: Methods depend on explicit 3D representations such as point clouds or BEV, which restricts generalization to open-world settings.
3. **Fixed camera configuration binding**: Most methods assume a fixed sensor layout and adjacency structure.

## Core Problem

How can a world model be constructed with minimal inductive bias that remains physically plausible while generalizing to arbitrary camera configurations and motions?

## Method

### 3.1 Next-Scale Prediction (Foundation)

Building on visual autoregressive models, each image is quantized into $K$ multi-scale token maps $X_{1:K}$, generated autoregressively from coarse to fine:

$$p(X_1, \ldots, X_K) = \prod_{k=1}^K p(X_k | X_1, \ldots, X_{k-1})$$

### 3.2 Dual-Causal Autoregression

**Scale causality**: All views within a single frame are modeled jointly (as they depict the same 3D space), with generation proceeding scale by scale:

$$p(X_1^{1:V}, \ldots, X_K^{1:V}) = \prod_{k=1}^K p(X_k^{1:V} | X_1^{1:V}, \ldots, X_{k-1}^{1:V})$$

**Temporal causality**: The current frame is conditioned on all views from all historical frames, without assuming strong dependencies between frames of the same camera:

$$p(X_{1:K}^{1:V,1:T}) = \prod_{t=1}^T \prod_{k=1}^K p(X_k^{1:V,t} | X_{1:k-1}^{1:V,1:t})$$

### 3.3 Isotropic Spatiotemporal Representation

**Core Innovation**: Rotary positional encoding (RoPE) based on relative Plücker rays.

For each token, the Plücker ray is computed as $\mathbf{p}_k^{v,t} = (\mathbf{m}, \mathbf{d}, t) \in \mathbb{R}^7$, where $\mathbf{m} = \mathbf{o}^{v,t} \times \mathbf{d}_k^{v,t}$.

RoPE is extended to 7D space:

$$\mathbf{R} = \begin{bmatrix} \mathbf{R_m} & 0 & 0 \\ 0 & \mathbf{R_d} & 0 \\ 0 & 0 & \text{RoPE}_{d/4}(t) \end{bmatrix}$$

Attention scores are computed based on the **relative** position between tokens:

$$a_{i,j} = \mathbf{q}_i^T \mathbf{R}_\Delta^{i,j} \mathbf{k}_j, \quad \mathbf{R}_\Delta^{i,j} = \mathbf{R}_i^T \mathbf{R}_j$$

**Key advantages**:
- Isotropic across all scales, views, and frames — no camera-specific assumptions required.
- Relative encoding naturally supports extrapolation beyond the training distribution.

### 3.4 Transformer Architecture

Each block contains three attention layers:
1. **Image-wise self-attention**: Processes each image independently with 2D Axial RoPE, preserving per-image fidelity.
2. **Global self-attention**: Unified attention across views and frames with Plücker ray RoPE, ensuring spatiotemporal consistency.
3. **Image-wise cross-attention**: Fuses conditional inputs such as text, 3D bounding boxes, and HD maps.

Condition encoding: 3D bounding box corners are projected into image space and encoded; T5 text embeddings are used for text; HD map points are sampled in 3D, projected, and encoded via PointNet.

### 3.5 Long-Video Recurrent Training

To address distribution shift in long-video generation, a recurrent training strategy is proposed:
- Frame-by-frame forward and backward passes with gradient accumulation followed by a unified parameter update.
- Latent features (rather than KV caches) are cached, reducing GPU memory by 50% while retaining gradients through the KV projection layers.
- Random bit-flip noise is introduced into visual token inputs to simulate inference-time errors.

## Key Experimental Results

| Method | Resolution | FID ↓ | FVD ↓ | Throughput ↑ (img/s) |
|--------|------------|-------|-------|----------------------|
| MagicDrive | 224×400 | 16.2 | - | 1.76 |
| DriveDreamer | 256×448 | 14.9 | 341 | 0.37 |
| Panacea | 256×512 | 17.0 | 139 | 0.67 |
| **RayNova** | **384×672** | **10.5** | **91** | **1.96** |

| Evaluation Dimension | Method | Metric (relative to Oracle) |
|----------------------|--------|-----------------------------|
| Object Conditioning (StreamPETR) | Panacea | 32.1 NDS (68%) |
| | **RayNova** | **41.9 NDS (89%)** |
| Object Conditioning (SparseFusion) | X-Drive | 69.6 NDS (95%) |
| | **RayNova** | **72.0 NDS (99%)** |
| Novel View Synthesis FID (shift 4m) | StreetGaussian | 67.44 |
| | **RayNova** | **17.48** |

## Highlights & Insights

- **Geometry-agnostic design**: No reliance on point clouds, BEV, or depth-based 3D priors; geometric awareness is achieved solely through relative ray positional encodings.
- **Dual-causal autoregression**: A unified scale-temporal causal framework that is more flexible than decoupled spatial-temporal attention designs.
- **Strong novel-view generalization**: Zero-shot adaptation to unseen camera configurations; FID of 17.48 vs. StreetGaussian's 67.44 under a 4m camera shift.
- **Efficient generation**: Throughput of 1.96 img/s significantly outperforms diffusion-based baselines (0.37–1.76 img/s).
- **Heterogeneous data compatibility**: Supports mixed training data with varying sensor configurations, resolutions, and frame rates.

## Limitations & Future Work

- An image-based VAE is used, which may affect FID/FVD metrics.
- Training data volume (~60 hours) remains limited compared to some methods trained on private large-scale datasets.
- Recurrent training incurs longer overall training time.
- The 3D point projection for map conditioning lacks height information.
- Experiments are conducted exclusively in driving scenarios; generalization to indoor or other settings remains unverified.

## Related Work & Insights

- vs. **Panacea**: Panacea assumes strong temporal dependencies between frames of the same camera, binding it to specific camera configurations; RayNova is fully decoupled, achieving FVD 91 vs. 139.
- vs. **X-Drive**: X-Drive relies on point clouds as a 3D prior; RayNova requires no explicit 3D representation.
- vs. **StreetGaussian/OmniRe**: Explicit 3D representations degrade sharply under large camera shifts (FID 67+), whereas RayNova remains robust (17.48).
- vs. **BEVWorld**: BEV representations are tied to a specific ground plane height; RayNova's ray space is more general.

The design principle of relative Plücker ray encoding is transferable to other geometry-aware generative tasks. The dual-causal autoregressive framework provides a unified paradigm for multi-modal and multi-resolution generation. The recurrent training strategy for mitigating distribution shift offers broader inspiration for long-sequence generation. The combination with VAR (Visual Autoregressive Model) warrants further investigation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Dual-causal autoregression combined with relative ray positional encoding represents a fundamentally novel design paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluation covers multiple dimensions (quality, conditioning, novel-view synthesis, motion), though limited to driving scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and figures are well-crafted.
- Value: ⭐⭐⭐⭐⭐ — Establishes a new direction for geometry-agnostic world modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)
- [\[CVPR 2026\] OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)
- [\[CVPR 2026\] GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)
- [\[CVPR 2026\] AvatarPointillist: AutoRegressive 4D Gaussian Avatarization](avatarpointillist_autoregressive_4d_gaussian_avatarization.md)
- [\[ICCV 2025\] UST-SSM: Unified Spatio-Temporal State Space Models for Point Cloud Video Modeling](../../ICCV2025/3d_vision/ust-ssm_unified_spatio-temporal_state_space_models_for_point_cloud_video_modelin.md)

</div>

<!-- RELATED:END -->
