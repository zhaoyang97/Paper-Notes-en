---
title: >-
  [Paper Note] Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues
description: >-
  [AAAI 2026][3D Vision][Photometric Stereo] This paper proposes GeoUniPS, which injects geometric priors learned by a large-scale 3D reconstruction model (VGGT) into a universal photometric stereo pipeline. Through a ligh…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Photometric Stereo"
  - "Geometric Priors"
  - "3D Reconstruction Foundation Model"
  - "VGGT"
  - "Perspective Projection"
date: 2026-05-08
content_hash: 22a4cc9a3893384e
---

# Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.13015](https://arxiv.org/abs/2511.13015)  
**Code**: [https://github.com/marcotam2002/geounips](https://github.com/marcotam2002/geounips)  
**Area**: 3D Vision / Photometric Stereo
**Keywords**: Photometric Stereo, Geometric Priors, 3D Reconstruction Foundation Model, VGGT, Perspective Projection

## TL;DR

This paper proposes GeoUniPS, which injects geometric priors learned by a large-scale 3D reconstruction model (VGGT) into a universal photometric stereo pipeline. Through a light–geometry dual-branch encoder, the method recovers plausible surface normals even when multi-illumination cues are unreliable (e.g., shadows, self-occlusions, biased lighting). A new perspective-projection training dataset, PS-Perp, is also introduced to bridge the gap between the orthographic projection assumption and real-world cameras.

## Background & Motivation

### State of the Field

Photometric Stereo (PS) recovers high-fidelity surface normals from multiple images captured under a fixed camera with varying illumination. Its development can be summarized as a progressive relaxation of assumptions:

- **Classical PS**: Assumes Lambertian surfaces and calibrated directional light sources.
- **Uncalibrated PS**: Removes the requirement for light calibration but still assumes a directional light model.
- **Universal PS (UniPS)**: Requires no lighting model at all, learning lighting representations directly from images.

State-of-the-art UniPS methods (SDM-UniPS, LINO-UniPS) have already eliminated most illumination assumptions.

### Limitations of Prior Work

1. **Performance degrades sharply when multi-illumination cues are unreliable**: UniPS fundamentally relies on multi-illumination variation as its primary signal. When illumination is insufficient in certain regions (biased lighting, shadows, self-occlusions), or when indirect lighting dominates in complex real indoor environments, existing methods lack a compensation mechanism.
2. **Synthetic training data lacks real-world geometric context**: Obtaining high-accuracy ground-truth normals for real scenes is prohibitively expensive, so PS models must be trained on clean synthetic data. The statistical distribution of synthetic data diverges significantly from real-world priors (e.g., "building facades are typically piecewise planar").
3. **Orthographic projection assumption does not hold for real perspective cameras**: Existing training datasets (PS-Wild, PS-Mix) are all rendered under orthographic projection and cannot handle perspective distortion.

### Root Cause

How can **high-level real-world geometric priors** be acquired within a **purely synthetic training pipeline** to compensate for insufficient multi-illumination cues?

### Starting Point

Large-scale 3D reconstruction models (e.g., VGGT) pretrained on massive real-world data have implicitly encoded rich geometric knowledge of real scenes. Even from a single input image, these models can recover plausible 3D shapes—demonstrating that they have learned **high-level monocular geometric priors** that go beyond low-level multi-view photometric constraints. Injecting these priors into the PS pipeline provides meaningful geometric guidance when illumination is insufficient.

## Method

### Overall Architecture

GeoUniPS adopts a standard two-stage encoder–decoder design:
1. **Encoder**: Extracts $K$ feature maps from $K$ multi-illumination images (replacing explicit lighting information in classical PS).
2. **Decoder**: Predicts normals at randomly sampled pixel locations.

The core innovation lies in the encoder: the **light–geometry dual-branch encoder**.

### Key Designs

#### 1. Light–Geometry Dual-Branch Encoder

**Function**: Extracts illumination-sensitive features and illumination-invariant geometric prior features in parallel, concatenating them into a unified representation.

$$\mathcal{F} = \text{Concat}(\mathcal{F}_{\text{Geo}}, \mathcal{F}_{\text{IL}})$$

**Geometry Branch (Encoder$_\text{Geo}$)**:
- Uses a **frozen VGGT aggregator** (24 layers of alternating frame and global attention).
- Tokens are extracted from layers [4, 11, 17, 23].
- A learnable DPT head fuses these into a 128-dim feature map (2× downsampled).
- VGGT is pretrained on large-scale real-world datasets (MegaDepth, CO3D-v2, ScanNet, DL3DV), encoding rich scene geometry knowledge.
- Parameters are frozen to preserve geometric knowledge against corruption by synthetic training data.

**Illumination Branch (Encoder$_\text{IL}$)**:
- A VGGT-like Transformer architecture (12 layers), replacing the DINOv2 tokenizer with a lightweight convolutional layer to better capture fine-grained local patterns.
- Alternates between intra-frame attention and **light-axis attention** (attention across different illumination images at the same spatial location).
- Tokens from layers [2, 5, 8, 11] are fused into a 128-dim feature map.
- Trained from scratch to learn normal cues from illumination variation.

**Design Motivation**:
- Geometric priors are illumination-invariant and provide plausible global shape estimates under any lighting condition.
- Illumination cues provide fine-grained surface detail when lighting is sufficient.
- The two branches are complementary: geometric priors compensate for poorly lit regions, while illumination cues supply high-frequency detail.

#### 2. Dual-Scale Normal Decoder

**Function**: Predicts normals at randomly sampled pixel positions from features and raw images.

**Mechanism**: Two-stage prediction—

**Low-scale stage**:
- 5-layer 256-dim light-axis Transformer processes encoder features.
- 1-layer 384-dim light-axis Transformer + PMA aggregates along the light axis.
- 2-layer 384-dim pixel-sampling Transformer enhances spatial consistency.
- MLP (384→192→3) predicts low-frequency normals.

**High-scale stage**:
- Raw RGB values are embedded via MLP (3→256).
- Concatenated with encoder features and processed by a 5-layer 256-dim light-axis Transformer.
- PMA aggregation + fusion with low-scale normals.
- 2-layer pixel-sampling Transformer + MLP (387→384→192→3) predicts refined normals.

**Design Motivation**: The low scale captures global geometric structure; the high scale recovers high-frequency surface detail from raw RGB. MLP-based RGB embedding is more efficient than using patches or raw pixels directly.

#### 3. PS-Perp: Perspective-Projection Training Dataset

**Function**: Constructs the first synthetic universal PS training dataset with perspective projection.

**Mechanism**:
- Rendered with Blender Cycles; focal lengths sampled from 20 mm to 1000 mm.
- 60,297 multi-object scenes, of which 44,220 use focal lengths <70 mm (strong perspective effects).
- 10 16-bit images per scene at 512×512 resolution.
- Random combinations of directional lights, point lights, and environment lighting.
- Shares asset libraries and scene composition strategies with PS-Mix.

**Design Motivation**: SDM-UniPS trained only on orthographic data achieves MAE of 22.18° at 15 mm focal length; mixed training reduces this to 6.98°. Perspective-projection training enables the model to learn spatially varying view directions.

### Loss & Training

- **Loss**: MSE loss computing the $\ell_2$ error between predicted and ground-truth normal vectors, computed at both scales and summed.
- Trained on 4×H100 GPUs for 6 days; AdamW optimizer; initial learning rate 1e-4; decay ×0.8 every 10 epochs.
- FP32 precision; 3–6 input images randomly sampled per batch to improve robustness.
- 2048 pixels sampled during training; increased to 10,000 during inference.

## Key Experimental Results

### Main Results

#### DiLiGenT Benchmark (Orthographic Projection, 96 Images)

| Method | Ball | Bear | Buddha | Cat | Goblet | Harvest | Avg MAE↓ |
|--------|------|------|--------|-----|--------|---------|---------|
| SDM-UniPS | 1.45 | 3.50 | 7.54 | 5.19 | 7.69 | 10.76 | 5.80 |
| LINO-UniPS | 1.77 | 2.62 | 6.22 | 3.38 | 5.14 | 8.60 | 4.75 |
| **GeoUniPS** | 2.63 | **2.46** | **5.95** | **3.27** | **5.00** | **8.54** | **4.65** |

#### LUCES Benchmark (Perspective Projection, 52 Images)

| Method | Ball | Die | Hippo | House | Avg MAE↓ |
|--------|------|-----|-------|-------|---------|
| SDM-UniPS | 11.77 | 7.22 | 8.95 | 25.91 | 12.80 |
| LINO-UniPS | 9.65 | 6.25 | 5.82 | 22.69 | 9.46 |
| **GeoUniPS** | **7.59** | **3.79** | **5.62** | **21.84** | **9.42** |

### Ablation Study

#### Effect of Training Data (Validated with SDM-UniPS)

| Training Data | 15 mm MAE | 35 mm MAE | Orthographic MAE |
|--------------|-----------|-----------|-----------------|
| PS-Mix (orthographic) | 22.18 | 14.09 | **5.52** |
| PS-Perp (perspective) | 7.18 | 5.47 | 8.95 |
| PS-Perp + PS-Mix | **6.98** | **5.53** | 5.62 |

#### Encoder Design Ablation

| Configuration | DiLiGenT (K=96) | DiLiGenT (K=1) | LUCES (K=52) |
|---------------|----------------|----------------|--------------|
| Encoder$_\text{IL}$ only | 5.81 | 18.40 | 11.03 |
| Encoder$_\text{Geo}$ only (VGGT) | 6.97 | 11.67 | 11.48 |
| Dual-Branch (VGGT) | **5.75** | **12.86** | **9.42** |
| Dual-Branch (MoGe) | 5.98 | 14.26 | 9.78 |

### Key Findings

1. **Geometric priors are critical at K=1 (single image)**: Encoder$_\text{IL}$ only yields MAE=18.40° at K=1; adding the geometry branch reduces this to 12.86°—geometric priors serve as a compensating mechanism when multi-illumination cues are entirely absent.
2. **Pure geometric priors do not scale with the number of input images**: Encoder$_\text{Geo}$ only improves only marginally as K increases, confirming that extracting illumination cues in the encoder is indispensable.
3. **Complementarity of perspective-projection training data**: Using PS-Mix alone yields MAE=22.18° under strong perspective; mixed training reduces this to only 6.98°.
4. **VGGT outperforms MoGe as a geometry backbone**: VGGT's multi-view training endows it with stronger geometric priors than monocular depth estimation methods.
5. **Qualitative results on the Multi-Illumination dataset are compelling**: GeoUniPS produces plausible normal maps in regions where classical PS completely fails, such as floors, walls, specular surfaces, and transparent objects.

## Highlights & Insights

1. **The insight that "3D reconstruction models = visual-geometry foundation models" is profound**: These models are reframed not merely as reconstruction tools but as foundation models encoding rich geometric knowledge.
2. **Freeze + learnable projection head strategy**: Preserves pretrained knowledge from corruption by small-scale synthetic data while still allowing adaptation to the downstream task.
3. **Light-axis attention vs. global attention**: The illumination branch uses light-axis attention (intuitively: variation at the same pixel across different illuminations is the key signal), whereas removing global attention from the geometry branch has negligible impact (since VGGT's geometric knowledge is illumination-invariant).
4. **Practical value of PS-Perp**: Bridges the long-standing orthographic–perspective projection gap in the PS community.

## Limitations & Future Work

1. **Slow inference speed**: Processing 16 images at 512×512 takes approximately 13 seconds on an H100, limiting real-time applicability.
2. **Training uses only 3–6 images**: Evaluating with more images may cause performance degradation due to distribution shift.
3. **Dependence on VGGT**: If VGGT is updated or a superior 3D reconstruction model emerges, the projection head must be retrained.
4. Extending the geometric prior injection approach to video PS or dynamic scene PS is a promising avenue.
5. A detailed analysis of computational resource requirements (VGGT itself is large) is lacking.

## Related Work & Insights

- **SDM-UniPS** (Ikehata, 2023): The foundational work in universal PS and the direct baseline of this paper; introduced the GLC concept.
- **VGGT** (Wang et al., 2025): A feed-forward 3D reconstruction model capable of inferring 3D shape from a single image; serves as the geometric prior source in this work.
- **DPT** (Ranftl et al., 2021): Dense Prediction Transformer, used as the projection head for feature fusion.
- **LINO-UniPS** (Li et al., 2025): Improves surface detail through wavelet refinement and gradient loss; complementary to the proposed method.
- **Insight**: Geometric knowledge from large-scale pretrained visual foundation models can be transferred at low cost to tasks that rely on photometric signals.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introducing 3D reconstruction models as geometric priors for PS is a genuinely new direction; PS-Perp fills an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative evaluation on DiLiGenT and LUCES, qualitative evaluation on Multi-Illumination, and comprehensive ablations; however, quantitative evaluation on more real-world scenes is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is articulated excellently; the logical chain for "why geometric priors are needed" is complete and coherent.
- Value: ⭐⭐⭐⭐⭐ — Opens a new direction for injecting foundation model geometric knowledge into PS, with implications for other photometry-dependent tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] 3D-Free Meets 3D Priors: Novel View Synthesis from a Single Image with Pretrained Diffusion Guidance](3d-free_meets_3d_priors_novel_view_synthesis_from_a_single_image_with_pretrained.md)
- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[ICCV 2025\] Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](../../ICCV2025/3d_vision/relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection](../../CVPR2026/3d_vision/vggt-det_mining_vggt_internal_priors_for_sensor-geometry-free_multi-view_indoor_.md)

</div>

<!-- RELATED:END -->
