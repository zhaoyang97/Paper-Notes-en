---
title: >-
  [Paper Note] SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs
description: >-
  [CVPR 2026][Multimodal VLM][3D LVLM] This paper identifies spatial perception bias in RoPE when applied to 3D LVLMs (1D indexing disrupts 3D locality and ignores directionality), and proposes SoPE…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "3D LVLM"
  - "positional encoding"
  - "spherical coordinates"
  - "RoPE"
  - "SpatialLM"
  - "spatial reasoning"
date: 2026-05-08
content_hash: ed8dcdf5fa1586d2
---

# SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs

**Conference**: CVPR 2026
**arXiv**: [2602.22716](https://arxiv.org/abs/2602.22716)  
**Code**: None  
**Area**: 3D Vision / Multimodal VLM / Positional Encoding
**Keywords**: 3D LVLM, positional encoding, spherical coordinates, RoPE, SpatialLM, spatial reasoning

## TL;DR

This paper identifies spatial perception bias in RoPE when applied to 3D LVLMs (1D indexing disrupts 3D locality and ignores directionality), and proposes SoPE, a spherical coordinate-based positional embedding using a four-dimensional index $(t, r, \theta, \phi)$ with multi-dimensional frequency allocation and multi-scale mixing. SoPE achieves state-of-the-art performance on 3D layout estimation and object detection benchmarks built upon SpatialLM.

## Background & Motivation

**Background**: 3D LVLMs encode point clouds and process them jointly with an LLM for 3D scene understanding. Mainstream approaches inherit RoPE from LLMs, flattening point cloud tokens into a 1D sequence via raster-scan ordering.

**Limitations of Prior Work**: Information flow visualization reveals severe spatial perception bias — cross-modal attention concentrates on a few hotspot tokens, the majority of 3D tokens receive approximately uniform weights, and small objects along with structural boundaries are systematically suppressed. Two root causes are identified: (i) 1D raster indexing destroys the 3D spatial continuity of point clouds, causing spatially adjacent tokens to receive non-adjacent positional indices; (ii) the relative distance $\Delta t = t_1 - t_2$ captures only sequential order, with no sensitivity to spatial position or directional change.

**Key Challenge**: RoPE is designed for 1D text and, when naively applied to 3D point clouds, inherently neglects spatial structure and directional information. Existing 2D/video extensions (VideoRoPE, M-RoPE) target image grids and are unsuitable for irregular point clouds.

**Key Insight**: Spherical coordinates $(r, \theta, \phi)$ naturally decouple distance from direction. Mapping 3D tokens into spherical space enables simultaneous encoding of position and orientation.

**Core Idea**: Replace the 1D raster index with spherical coordinates $(t, r, \theta, \phi)$, and allocate RoPE frequency bands functionally across different coordinate components.

## Method

### Overall Architecture

SpatialLM baseline → extract $(x, y, z)$ coordinates of point cloud tokens → convert to spherical coordinates $(r, \theta, \phi)$ while retaining temporal index $t$ → allocate 128-dimensional RoPE frequency bands at ratio $t:r:\theta:\phi = 24:2:3:3$ → apply multi-scale frequency mixing to each component → replace original RoPE → end-to-end training.

### Key Designs

1. **Spherical Coordinate Positional Projection**

    - Function: Remaps 3D tokens from 1D raster indices to geometrically-aware four-dimensional positions $(t, r, \theta, \phi)$.
    - Mechanism: $r = \sqrt{x^2+y^2+z^2}$, $\theta = \arccos(z/r)$, $\phi = \text{atan2}(y, x)$. The relative displacement is decomposed into four components $\Delta t, \Delta r, \Delta\theta, \Delta\phi$, naturally encoding both spatial position change and directional angular change.
    - Design Motivation: Cartesian 3D coordinates (RoPE-3D) encode position but cannot distinguish angular relationships; spherical decomposition renders radial distance and angular direction orthogonal, making directional information explicit.

2. **Multi-Dimensional Frequency Allocation**

    - Function: Distributes 128-dimensional RoPE frequency bands across four coordinate components at ratio $t:r:\theta:\phi = 24:2:3:3$.
    - Mechanism: Spherical components $(r, \theta, \phi)$ are mapped to high-frequency sub-bands (capturing fine-grained spatial and angular variation), while temporal index $t$ is mapped to low-frequency sub-bands (preserving long-range temporal coherence). The rotation matrix is block-diagonalized, with each component encoded independently and combined additively.
    - Design Motivation: The value range of $t$ greatly exceeds that of the angular components, necessitating more low-frequency bands for temporal smoothness; angular variations are typically small and fine-grained, requiring high-frequency bands for discrimination. The ratio is determined via large-scale ablation experiments (Uniform, Angular-Biased, Temporal-Biased).

3. **Multi-Scale Frequency Mixing**

    - Function: Fuses linear, logarithmic, and periodic transformations at the RoPE phase level for each coordinate component.
    - Mechanism: $\varphi_k(u) = \frac{1}{3}(\omega_k^{lin}g^{lin}(u) + \omega_k^{log}g^{log}(u) + \omega_k^{per}g^{per}(u))$. The linear term preserves absolute precision, the logarithmic term emphasizes local neighborhoods, and the periodic term captures global structure. Equal-weight mixing introduces no additional learnable parameters.
    - Design Motivation: Single-scale encoding struggles to simultaneously capture fine-grained geometry and large-scale layout; multi-scale fusion endows the model with discriminative power across different spatial ranges.

### Loss & Training

Training follows the SpatialLM setup. Architecture: Sonata encoder + Qwen2.5-0.5B LLM + 2-layer MLP. Single-stage training on 4 × NVIDIA H20 GPUs. SoPE serves as a drop-in replacement for RoPE with no additional inference overhead.

## Key Experimental Results

### Main Results

| Method | ARKitScenes F1@0.25 | F1@0.50 | SpatialLM Dataset F1@0.25 | F1@0.50 |
|---|---|---|---|---|
| SpatialLM (RoPE) | 63.9 | 60.7 | 69.7 | 62.0 |
| + CCA | 64.1 | 60.5 | 69.8 | 62.5 |
| + RoPE-3D | 64.2 | 61.4 | 69.7 | 62.4 |
| **SpatialSoPE** | **66.1** | **63.2** | **71.4** | **63.4** |

| Method | Structured3D IoU2D@0.25 | IoU2D@0.50 |
|---|---|---|
| RoomFormer | 70.4 | 67.2 |
| SceneScript | 83.1 | 80.8 |
| SpatialLM (ft.) | 86.5 | 84.6 |
| **SpatialSoPE (ft.)** | **88.7** | **86.2** |

### Ablation Study

| Configuration | ARKit F1@0.25 | F1@0.50 | Note |
|---|---|---|---|
| Ratio 24:2:3:3 (optimal) | 66.1 | 63.2 | Ours |
| Ratio 8:6:9:9 (Angular-Biased) | 65.5 | 62.7 | Over-allocation to spherical |
| Ratio 1:1:1:1 (Uniform) | 63.0 | 59.0 | −3 points |
| Ratio 5:1:1:1 (Temporal-Biased) | 65.0 | 62.7 | Temporal-dominant |
| SoPE w/o multi-scale mixing | 65.4 | 61.4 | Multi-scale contributes +1.8 |
| RoPE-3D + multi-scale | 64.8 | 62.1 | Spherical > Cartesian |

### Key Findings

- Multi-scale mixing yields larger gains for SoPE (+0.7/+1.8) than for RoPE-3D — spherical coordinates are a prerequisite for fully benefiting from multi-scale mixing.
- Spherical > Cartesian > 2D projection; directional/angular encoding is the key differentiating factor.
- Information flow visualization confirms that SoPE produces more balanced cross-modal attention, eliminating the hotspot concentration observed with RoPE.

## Highlights & Insights

- Spherical coordinates naturally decouple distance from orientation — geometrically more appropriate than Cartesian coordinates for 3D positional encoding. The idea is direct and effective, yet previously unexplored.
- A simple modification (coordinate transformation + frequency reallocation) yields substantial gains (ARKitScenes +2.2/+2.5), demonstrating that positional encoding is indeed a critical bottleneck in 3D LVLMs.
- Information flow visualization as a diagnostic tool merits broader adoption — identifying which tokens are under-attended before designing targeted encoding improvements.

## Limitations & Future Work

- Validation is limited to a 0.5B small model; effectiveness on larger models (7B+) remains to be confirmed.
- The choice of spherical origin (scene geometric center vs. camera position) is not thoroughly investigated, which may affect encoding quality.
- The frequency allocation ratio is determined manually; adaptive or learnable schemes may yield further improvement.
- Evaluation is restricted to indoor 3D scenes; outdoor and large-scale settings such as autonomous driving remain untested.

## Related Work & Insights

- **vs. RoPE-3D**: Cartesian coordinate encoding improves spatial awareness but lacks directional information; SoPE's spherical decomposition encodes both simultaneously.
- **vs. VideoRoPE/M-RoPE**: These methods perform spatiotemporal decomposition for 2D images/video and are not applicable to the irregular structure of 3D point clouds.
- **vs. DRoPE**: The polar-coordinate directional extension targets task-specific properties such as heading periodicity; SoPE's spherical formulation is more general-purpose.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of spherical coordinate PE in 3D LVLMs
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark full ablation + real-device deployment latency testing
- Writing Quality: ⭐⭐⭐⭐ Thorough motivation analysis with outstanding information flow visualization
- Value: ⭐⭐⭐⭐ Drop-in replacement for RoPE with high cross-domain reference value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models](../../ICML2026/multimodal_vlm/circle-rope_cone-like_decoupled_rotary_positional_embedding_for_large_vision-lan.md)
- [\[ICLR 2026\] PPE: Positional Preservation Embedding for Token Compression in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/ppe_positional_preservation_embedding_for_token_compression_in_multimodal_large_.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](modix_positional_index_scaling.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[CVPR 2026\] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)

</div>

<!-- RELATED:END -->
