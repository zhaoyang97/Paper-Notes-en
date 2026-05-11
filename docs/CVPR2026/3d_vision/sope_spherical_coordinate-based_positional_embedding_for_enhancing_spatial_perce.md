---
title: >-
  [Paper Note] SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs
description: >-
  [CVPR2026][3D Vision][positional embedding] This paper proposes SoPE, a spherical coordinate-based positional embedding that remaps point cloud tokens from one-dimensional sequence indices to a spherical coordinate space…
tags:
  - "CVPR2026"
  - "3D Vision"
  - "positional embedding"
  - "3D LVLM"
  - "spherical coordinates"
  - "RoPE"
  - "point cloud"
  - "spatial perception"
date: 2026-05-08
content_hash: 9be46b362aafc8c7
---

# SoPE: Spherical Coordinate-Based Positional Embedding for Enhancing Spatial Perception of 3D LVLMs

**Conference**: CVPR2026  
**arXiv**: [2602.22716](https://arxiv.org/abs/2602.22716)  
**Code**: None  
**Area**: 3D Vision / 3D Scene Understanding  
**Keywords**: positional embedding, 3D LVLM, spherical coordinates, RoPE, point cloud, spatial perception

## TL;DR

This paper proposes SoPE, a spherical coordinate-based positional embedding that remaps point cloud tokens from one-dimensional sequence indices to a spherical coordinate space $(t,r,\theta,\phi)$, combined with multi-dimensional frequency allocation and multi-scale frequency mixing strategies, significantly enhancing the spatial perception capabilities of 3D large vision-language models.

## Background & Motivation

Current 3D large vision-language models (3D LVLMs) widely adopt Rotary Position Embedding (RoPE) inherited from LLMs to model positional relationships between tokens. However, RoPE exhibits two fundamental deficiencies in 3D scene settings:

**Loss of 3D spatial structure**: RoPE flattens point cloud tokens into a one-dimensional sequence following a raster scan order and assigns indices by sequence position. This approach completely disregards the true 3D spatial positions of point cloud tokens, causing spatially adjacent tokens to receive non-adjacent position indices and thus breaking 3D neighborhood continuity.

**Absence of directional awareness**: The relative distance computation $\Delta t = t_1 - t_2$ in RoPE only captures temporal changes within the sequence and cannot perceive angular and directional differences between tokens. Consequently, the model is blind to directional variations that are critical for visual representation.

Through attention visualization, the authors identify a "spatial perception bias" phenomenon: cross-modal attention concentrates on a small number of hotspot regions, and large numbers of 3D tokens with clearly distinct positions and orientations receive nearly identical attention weights, causing vast areas of the scene to be effectively ignored. Small objects and structural boundaries are particularly suppressed in large indoor scenes.

Although several works have extended RoPE to multimodal settings (e.g., variants for image/video), they still treat positional indices as sequential or grid coordinates and do not explicitly encode the 3D geometric structure of point cloud tokens. This core gap constitutes the design motivation of the proposed method.

## Method

### Overall Architecture

SoPE serves as a connector-level positional encoding module that replaces the original RoPE in SpatialLM in a plug-and-play manner. The overall framework, SpatialSoPE, consists of a point cloud encoder (Sonata), a two-layer MLP projector, and an LLM (Qwen2.5-0.5B). SoPE reparameterizes the positional encoding between the projector and the LLM. SoPE comprises three core components:

### Key Design 1: Spherical Coordinate Positional Projection

Point cloud tokens are remapped from one-dimensional indices to geometry-aware four-dimensional positions in two steps:

**Step 1 — Position index reassignment**: The Cartesian coordinates $(x,y,z)$ of each point cloud token are extracted while retaining the original RoPE sequence index $t$, yielding a four-dimensional position $(t,x,y,z)$.

**Step 2 — Spherical coordinate mapping**: Cartesian coordinates are converted to spherical coordinates:

- Radius $r = \sqrt{x^2+y^2+z^2}$ (encodes depth/distance information)
- Polar angle $\theta = \arccos(z/r)$ (encodes elevation direction)
- Azimuthal angle $\phi = \text{atan2}(y,x)$ (encodes horizontal orientation)

The final position index is $(t,r,\theta,\phi)$, and the relative position is expanded into four components $(\Delta t, \Delta r, \Delta \theta, \Delta \phi)$. This enables the model to simultaneously capture spatial position and directional angular changes, fundamentally addressing the limitation of original RoPE which relies solely on a one-dimensional temporal index.

### Key Design 2: Multi-Dimensional Frequency Allocation

The $d/2=64$ frequency bands of RoPE are proportionally allocated across the four dimensions. The core idea is:

- **Spherical coordinate components $(r,\theta,\phi)$ → high-frequency sub-bands** (front): capture fine-grained spatial and angular variations
- **Temporal component $t$ → low-frequency sub-bands** (back): preserve long-range temporal dynamics and continuity

Through extensive ablation experiments, the optimal ratio is determined to be $t:r:\theta:\phi = 24:2:3:3$ (totaling 32 frequency band pairs). The four partial rotation matrices are summed to obtain the final relative rotation matrix.

### Key Design 3: Multi-Scale Frequency Mixing Strategy

Single-scale positional encoding struggles to simultaneously capture fine-grained geometry and large-scale architectural layout. For each coordinate component $u \in \{t,r,\theta,\phi\}$, three complementary coordinate transformations are constructed:

- **Linear scale** $g^{\text{lin}}(u)$: preserves absolute positional accuracy
- **Logarithmic compression scale** $g^{\text{log}}(u)$: emphasizes local neighborhood structure
- **Periodic scale** $g^{\text{per}}(u)$: captures global patterns and long-range dependencies

The final phase is a uniform-weight fusion of all three: $\varphi_k(u) = \frac{1}{3}(\omega_k^{\text{lin}}g^{\text{lin}}(u) + \omega_k^{\text{log}}g^{\text{log}}(u) + \omega_k^{\text{per}}g^{\text{per}}(u))$. No additional learnable parameters are introduced, maintaining a lightweight design.

### Loss & Training

Training is conducted within the SpatialLM framework using the Sonata point cloud encoder in a single-stage setup on 4 NVIDIA H20 GPUs. Both training from scratch and pretrain-then-finetune configurations are supported.

## Key Experimental Results

### Main Results: Layout Estimation (Structured3D)

| Method | IoU2D@0.25 ↑ | IoU2D@0.5 ↑ |
|------|-------------|------------|
| RoomFormer | 70.4 | 67.2 |
| SceneScript | 83.1 | 80.8 |
| SpatialLM (ft. SpatialLM→S3D) | 86.5 | 84.6 |
| **SpatialSoPE (ft. SpatialLM→S3D)** | **88.7** | **86.2** |

### Main Results: 3D Object Detection (ARKitScenes)

| Method | IoU3D@0.25 ↑ | IoU3D@0.5 ↑ |
|------|-------------|------------|
| VoteNet | 53.9 | 45.4 |
| H3DNet | 55.7 | 46.3 |
| NeRF-Det | 60.3 | 34.7 |
| UniDet3D | 62.8 | 48.3 |
| SpatialLM | 63.9 | 60.7 |
| **SpatialSoPE** | **66.1** | **63.2** |

### Comparison of Different Positional Encoding Schemes (ARKitScenes + SpatialLM Dataset)

| Method | ARKit@0.25 ↑ | ARKit@0.5 ↑ | SpatialLM@0.25 ↑ | SpatialLM@0.5 ↑ |
|------|-------------|------------|------------------|-----------------|
| SpatialLM (baseline) | 63.9 | 60.7 | 69.7 | 62.0 |
| +MCA | 63.7 | 60.2 | 70.1 | 61.6 |
| +CCA | 64.1 | 60.5 | 69.8 | 62.5 |
| +RoPE-3D | 64.2 | 61.4 | 69.7 | 62.4 |
| **SpatialSoPE** | **66.1** | **63.2** | **71.4** | **63.4** |

### Ablation Study: Frequency Allocation Ratio (ARKitScenes)

| Configuration | $t:r:\theta:\phi$ | IoU3D@0.25 ↑ | IoU3D@0.5 ↑ |
|------|-------------------|-------------|------------|
| Angular-Biased | 8:6:9:9 | 65.5 | 62.7 |
| Uniform | 1:1:1:1 | 63.0 | 59.0 |
| Temporal-Biased | 5:1:1:1 | 65.0 | 62.7 |
| **SpatialSoPE** | **24:2:3:3** | **66.1** | **63.2** |

### Ablation Study: Effect of Multi-Scale Frequency Mixing

| Method | Multi-Scale | ARKit@0.25 | ARKit@0.5 | SpatialLM@0.25 | SpatialLM@0.5 |
|------|--------|-----------|----------|----------------|---------------|
| RoPE-3D | ✗ | 64.2 | 61.7 | 69.4 | 62.3 |
| RoPE-3D | ✓ | 64.8 | 62.1 | 70.3 | 62.9 |
| SoPE | ✗ | 65.4 | 61.4 | 71.0 | 62.5 |
| **SoPE** | **✓** | **66.1** | **63.2** | **71.4** | **63.4** |

### Key Findings

1. SoPE consistently outperforms the SpatialLM baseline across all three benchmarks (ARKitScenes, SpatialLM Dataset, Structured3D), with improvements of +2.2/+1.6 on layout estimation and +2.2/+2.5 on object detection.
2. Compared to 2D projection schemes such as CCA and MCA, direct 3D encoding (RoPE-3D) already surpasses 2D projection, and spherical coordinate encoding (SoPE) widens the gap further.
3. The frequency allocation ratio has a significant impact: uniform allocation yields the worst performance (IoU3D@0.5 of only 59.0), while the configuration with temporal weight 24 and spherical coordinate weight 8 achieves the best results.
4. Multi-scale frequency mixing provides a substantially larger gain for SoPE than for RoPE-3D (+1.8 vs. +0.4 on IoU@0.5), indicating strong synergy between spherical coordinate parameterization and the multi-scale strategy.
5. Cross-modal attention visualization shows that SoPE produces more balanced global attention patterns, effectively mitigating spatial perception bias.

## Highlights & Insights

1. **Precise problem identification**: Addressing 3D LVLM performance from the perspective of positional encoding targets a fundamental yet overlooked issue. The "spatial perception bias" phenomenon revealed through attention flow visualization provides compelling motivation.
2. **Spherical coordinates as a natural choice**: Point cloud data inherently possesses directional and distance properties; spherical coordinates $(r,\theta,\phi)$ naturally decouple distance from angular information more effectively than Cartesian coordinates and exhibit better mathematical alignment with RoPE's rotation mechanism.
3. **Plug-and-play design**: SoPE functions as a drop-in replacement without modifying the model architecture or introducing learnable parameters, making it highly practical.
4. **Comprehensive engineering validation**: In addition to benchmark experiments, end-to-end deployment on a real robotic system is validated (point cloud reconstruction → scene understanding → navigation planning), demonstrating practical applicability.

## Limitations & Future Work

1. **Validation limited to Qwen2.5-0.5B**: Experiments are conducted solely with a 0.5B-scale LLM; scalability to larger models (7B/13B+) remains unverified.
2. **Frequency allocation ratio determined empirically**: The optimal ratio 24:2:3:3 is identified through ablation experiments without theoretical guidance. Adaptive allocation based on data characteristics warrants investigation.
3. **Spherical coordinate origin selection**: The paper computes spherical coordinates relative to the coordinate origin, but the origin position may vary across scenes and affect encoding quality, particularly in multi-room environments.
4. **Fixed multi-scale weights**: The three scales are fused with equal weights; different scenes or object scales may benefit from varying weight assignments.
5. **Focus on indoor scenes**: Experiments are concentrated on indoor scenes (ARKitScenes, Structured3D); performance on large-scale outdoor scenes (e.g., autonomous driving) remains unexplored.

## Related Work & Insights

- **SpatialLM**: The base framework of this work, providing data, architecture, and training designs for 3D spatial reasoning.
- **RoPE variants** (VideoRoPE, M-RoPE, ComRoPE, DRoPE): Extend RoPE along temporal/multi-view dimensions but do not address point cloud geometry.
- **Circle-RoPE**: Decouples cross-modal positional encoding via conical projection; conceptually related but methodologically distinct.
- **Insights**: The spherical coordinate encoding idea is generalizable to other 3D tasks requiring directional awareness (e.g., 3D object tracking, scene flow estimation) and to positional modeling in emerging representations such as 3D Gaussian Splatting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of spherical coordinates and multi-scale frequency mixing to improve RoPE is novel and practical, though the core idea of replacing sequence indices with spatial coordinates has precedents in 2D settings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-benchmark evaluation, comprehensive baseline comparisons, and multi-faceted ablation studies are thorough, though only a 0.5B model is used with no large-model validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation analysis is in-depth, mathematical derivations are clear, and visualizations are convincing.
- **Value**: ⭐⭐⭐⭐ — The drop-in replacement design offers strong practical utility and provides direct reference value to the 3D LVLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PE3R: Perception-Efficient 3D Reconstruction](pe3r_perception-efficient_3d_reconstruction.md)
- [\[ICCV 2025\] RoCo-Sim: Enhancing Roadside Collaborative Perception through Foreground Simulation](../../ICCV2025/3d_vision/roco-sim_enhancing_roadside_collaborative_perception_through_foreground_simulati.md)
- [\[CVPR 2026\] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](long_scope_fully_sparse_long_range_cooperative_3d_perception.md)
- [\[CVPR 2026\] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)
- [\[CVPR 2026\] Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics](efficient_hybrid_se3-equivariant_visuomotor_flow_policy_via_spherical_harmonics_.md)

</div>

<!-- RELATED:END -->
