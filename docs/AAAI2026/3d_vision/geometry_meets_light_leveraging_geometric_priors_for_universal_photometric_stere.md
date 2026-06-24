---
title: >-
  [Paper Note] Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues
description: >-
  [AAAI 2026 Oral][3D Vision][Photometric Stereo] This paper proposes GeoUniPS, which introducesgeometric priors from a large-scale 3D reconstruction model (VGGT) into a universal photometric stereo network for the first time. Through a dual-branch illumination-geometry encoder, geometric priors are leveraged to compensate for insufficient multi-illumination cues. Additionally, a perspective projection training dataset, PS-Perp, is introduced to bridge the gap between orthograp…
tags:
  - "AAAI 2026 Oral"
  - "3D Vision"
  - "Photometric Stereo"
  - "Geometric Priors"
  - "3D Reconstruction Foundation Model"
  - "VGGT"
  - "Perspective Projection"
date: 2026-05-08
content_hash: f890514a6d75f123
---

# Geometry Meets Light: Leveraging Geometric Priors for Universal Photometric Stereo under Limited Multi-Illumination Cues

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.13015](https://arxiv.org/abs/2511.13015)  
**Code**: [https://github.com/marcotam2002/geounips](https://github.com/marcotam2002/geounips)  
**Area**: 3D Vision / Photometric Stereo  
**Keywords**: Photometric Stereo, Geometric Priors, 3D Reconstruction Foundation Model, VGGT, Perspective Projection

## TL;DR

This paper proposes GeoUniPS, which introducesgeometric priors from a large-scale 3D reconstruction model (VGGT) into a universal photometric stereo network for the first time. Through a dual-branch illumination-geometry encoder, geometric priors are leveraged to compensate for insufficient multi-illumination cues. Additionally, a perspective projection training dataset, PS-Perp, is introduced to bridge the gap between orthographic projection assumptions and real-world scenes.

## Background & Motivation

### Background

Photometric Stereo (PS) recovers high-fidelity normal maps from multiple images under varying illumination with a fixed camera. Its development history is characterized by the gradual relaxation of illumination assumptions:

- **Traditional PS**: Requires calibrated directional light + Lambertian BRDF assumptions.
- **Uncalibrated PS**: Removes the requirement for light calibration but still assumes a directional light model.
- **Universal PS**: Eliminates illumination models entirely, learning illumination representations directly from images.

Current SOTA universal PS methods (SDM-UniPS, LINO-UniPS) have eliminated most illumination assumptions.

### Limitations of Prior Work

1. **Severe performance degradation when multi-illumination cues are unreliable**: Universal PS fundamentally relies on multi-illumination variation as the primary cue. When certain regions suffer from insufficient illumination (e.g., biased lighting, shadows, self-occlusion) or when indirect lighting dominates in complex real-world indoor environments, existing methods lack a compensation mechanism.
2. **Synthetic training data lacks real-world geometric context**: Acquiring high-accuracy ground-truth (GT) normals for real scenes is extremely expensive, forcing PS models to train only on clean synthetic data. The statistical distribution of synthetic data differs drastically from real-world scenes (e.g., priors such as "architectural facades are typically piecewise planar").
3. **Orthographic projection assumption is inapplicable to real perspective cameras**: Existing training datasets (PS-Wild, PS-Mix) are rendered entirely with orthographic projection, failing to handle perspective distortion.

### Key Challenge

How to acquire **high-level real-world geometric priors** in a **purely synthetic training pipeline** to compensate for insufficient multi-illumination cues?

### Key Insight

Large-scale 3D reconstruction models (e.g., VGGT) pre-trained on massive real-world data have implicitly encoded rich geometric knowledge of real scenes. Even given a single image, they can recover reasonable 3D shapes—demonstrating that these models have learned **high-level monocular geometric priors** that go beyond low-level multi-view photometric constraints. Injecting these priors into the PS pipeline can provide meaningful geometric guidance when illumination is insufficient.

## Method

### Overall Architecture

GeoUniPS adopts a standard two-stage encoder-decoder design:
1. **Encoder**: Extracts K feature maps from K multi-illumination images (replacing illumination information in traditional PS).
2. **Decoder**: Predicts normals at randomly sampled pixel locations.

The core innovation lies in the encoder side—the **dual-branch illumination-geometry encoder**.

### Key Designs

#### 1. Dual-Branch Illumination-Geometry Encoder

**Function**: Parallelly extracts illumination-sensitive features and illumination-invariant geometric prior features, which are concatenated into a unified representation.

$$\mathcal{F} = \text{Concat}(\mathcal{F}_{\text{Geo}}, \mathcal{F}_{\text{IL}})$$

**Geometric Branch (EncoderGeo)**:
- Uses a **frozen VGGT aggregator** (24 layers of alternating frame/global attention).
- Extracts tokens from layers [4, 11, 17, 23].
- Fuses them into a 128-dim feature map (2× downsampling) via a learnable DPT head.
- VGGT is trained on large-scale real-world datasets such as MegaDepth, CO3D-v2, ScanNet, and DL3DV, thereby encoding rich scene geometric knowledge.
- Keeping parameters frozen preserves the geometric knowledge from being corrupted by synthetic training data.

**Illumination Branch (EncoderIL)**:
- Uses a VGGT-like Transformer architecture (12 layers), but replaces the DINOv2 tokenizer with a lightweight convolution to capture fine-grained local patterns.
- Alternatingly employs intra-frame attention (frame attention) and **light-axis attention** (attention across different illumination images at the same spatial position).
- Extracts and fuses feature maps from layers [2, 5, 8, 11] into a 128-dim representation.
- Trained from scratch to learn normal cues from illumination variations.

**Design Motivation**:
- Geometric priors are illumination-invariant, offering reasonable global shape estimation under any lighting conditions.
- Illumination cues provide fine surface details when light is sufficient.
- Complementarity: Geometric priors compensate for regions with insufficient illumination, while illumination cues provide high-frequency details.

#### 2. Dual-Scale Normal Decoder

**Function**: Predicts normals at randomly sampled pixel locations using features and raw images.

**Mechanism**: Two-stage prediction—

**Low-Scale Stage**:
- A 5-layer 256-dim light-axis Transformer processes encoder features.
- A 1-layer 384-dim light-axis Transformer + PMA aggregates along the light axis.
- A 2-layer 384-dim pixel-sampling Transformer enhances spatial consistency.
- An MLP (384→192→3) predicts low-frequency normals.

**High-Scale Stage**:
- Embeds raw RGB values into a high-dimensional space via an MLP (3→256).
- Concatenates the embedding with encoder features and processes using a 5-layer 256-dim light-axis Transformer.
- Fuses PMA aggregation with the low-scale normals.
- A 2-layer pixel-sampling Transformer + MLP (387→384→192→3) predicts fine normals.

**Design Motivation**: The low-scale stage captures global geometric structure, whereas the high-scale stage recovers high-frequency surface details from raw RGB. Embedding raw RGB via MLP is more efficient than directly using patches or pixels.

#### 3. PS-Perp Perspective Projection Dataset

**Function**: Constructs the first synthetic training dataset for universal PS with perspective projection.

**Mechanism**:
- Employs Blender Cycles renderer with focal lengths sampled from 20mm to 1000mm.
- Contains 60,297 multi-object scenes, with 44,220 using a focal length of <70mm (strong perspective effect).
- Each scene consists of 10 16-bit images at 512×512 resolution.
- Employs random combinations of directional, point, and environmental lights.
- Shares the asset library and scene composition strategies with PS-Mix.

**Design Motivation**: SDM-UniPS has an MAE up to 22.18° under 15mm focal length when trained only on orthographic data, which drops to 6.98° after hybrid training. Training under perspective projection enables the model to learn spatially-varying viewing directions.

### Loss & Training

- **Loss**: MSE loss computes the $\ell_2$ error between forecasted and GT normals, calculated and summed across both scales.
- Trained on 4×H100 GPUs for 6 days using the AdamW optimizer, with an initial learning rate of 1e-4 decaying by 0.8 every 10 epochs.
- FP32 precision training, with 3-6 input images randomly sampled per batch to improve robustness.
- Sampled 2,048 pixels during training, which scales up to 10,000 pixels during inference.

## Key Experimental Results

### Main Results

#### DiLiGenT Benchmark (Orthographic Projection, 96 Images)

| Method | Ball | Bear | Buddha | Cat | Goblet | Harvest | Avg MAE↓ |
|------|------|------|--------|-----|--------|---------|---------|
| SDM-UniPS | 1.45 | 3.50 | 7.54 | 5.19 | 7.69 | 10.76 | 5.80 |
| LINO-UniPS | 1.77 | 2.62 | 6.22 | 3.38 | 5.14 | 8.60 | 4.75 |
| **GeoUniPS** | 2.63 | **2.46** | **5.95** | **3.27** | **5.00** | **8.54** | **4.65** |

#### LUCES Benchmark (Perspective Projection, 52 Images)

| Method | Ball | Die | Hippo | House | Avg MAE↓ |
|------|------|-----|-------|-------|---------|
| SDM-UniPS | 11.77 | 7.22 | 8.95 | 25.91 | 12.80 |
| LINO-UniPS | 9.65 | 6.25 | 5.82 | 22.69 | 9.46 |
| **GeoUniPS** | **7.59** | **3.79** | **5.62** | **21.84** | **9.42** |

### Ablation Study

#### Effectiveness of Training Data (Validated using SDM-UniPS)

| Training Data | 15mm MAE | 35mm MAE | Orthographic MAE |
|---------|---------|---------|---------|
| PS-Mix (Orthographic) | 22.18 | 14.09 | **5.52** |
| PS-Perp (Perspective) | 7.18 | 5.47 | 8.95 |
| PS-Perp + PS-Mix | **6.98** | **5.53** | 5.62 |

#### Ablation on Encoder Designs

| Configuration | DiLiGenT (K=96) | DiLiGenT (K=1) | LUCES (K=52) |
|------|----------------|----------------|--------------|
| EncoderIL only | 5.81 | 18.40 | 11.03 |
| EncoderGeo only (VGGT) | 6.97 | 11.67 | 11.48 |
| Dual-Branch (VGGT) | **5.75** | **12.86** | **9.42** |
| Dual-Branch (MoGe) | 5.98 | 14.26 | 9.78 |

### Key Findings

1. **Geometric priors are crucial when K=1 (single image)**: EncoderIL-only yields an MAE of 18.40° when K=1, which drops to 12.86° upon integrating the geometric branch—demonstrating that geometric priors compensate when multi-illumination cues are completely absent.
2. **Pure geometric priors do not scale with an increased number of input images**: EncoderGeo-only shows limited improvement even as K increases, proving that extracting illumination cues in the encoder remains indispensable.
3. **Complementarity of perspective projection training data**: Using PS-Mix alone under strong perspective yields an MAE of 22.18°, which lands at 6.98° after hybrid training.
4. **VGGT serves as a better geometric backbone than MoGe**: VGGT's multi-view pretraining allows it to learn stronger geometric priors compared to pure monocular depth estimation methods.
5. **Stunning qualitative results on the Multi-Illumination dataset**: In challenging areas (e.g., floors, walls, specular surfaces, and transparent objects) where traditional PS completely fails, GeoUniPS still produces plausible normal maps.

## Highlights & Insights

1. **Profound insight of "3D Reconstruction Models = Vision-Geometry Foundation Models"**: Treating multi-view 3D reconstruction models as foundation models embedded with rich geometric knowledge, rather than mere reconstruction tools.
2. **Frozen + learnable projection head strategy**: Preserves pre-trained knowledge from being corrupted by small-scale synthetic data while allowing adaptation to the downstream task.
3. **Difference between light-axis attention vs. global attention**: Using light-axis attention in the illumination branch (intuition: the variation of the same pixel under different illumination is key), whereas removing global attention in the geometry branch has minimal impact (since VGGT's geometric knowledge is illumination-invariant).
4. **Practical value of the PS-Perp dataset**: Bridges the long-standing orthographic-perspective projection gap in the field.

## Limitations & Future Work

1. **Slow inference speed**: Inferring on 16 512×512 images takes about 13 seconds (H100), limiting real-time applications.
2. **Limited image count during training**: Using only 3-6 images during training may cause performance drops when evaluating on more images due to distribution shifts.
3. **Dependency on VGGT**: If the VGGT encoder is updated or superior 3D reconstruction models emerge, the projection heads must be retrained.
4. Future work could explore incorporating geometric priors into video PS or dynamic scene PS.
5. Lack of detailed analysis on computational resource requirements (VGGT itself is relatively large).

## Related Work & Insights

- **SDM-UniPS** (Ikehata, 2023): Seminal work in universal PS, serving as the direct baseline of this paper; introduced the concept of GLC.
- **VGGT** (Wang et al., 2025): A feedforward 3D reconstruction model capable of inferring 3D shape from a single image, serving as the source of geometric priors in this paper.
- **DPT** (Ranftl et al., 2021): Dense Prediction Transformer, utilized as a projection head for feature fusion.
- **LINO-UniPS** (Li et al., 2025): Improves details through wavelet refinement and gradient loss, which is complementary to this work.
- **Insight**: The geometric knowledge of large-scale pre-trained vision foundation models can be transferred "at low cost" to tasks fundamentally relying on photometric signals.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Introducing a 3D reconstruction model as a geometric prior to PS is a brand-new concept, and the PS-Perp dataset fills a critical gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative evaluations on DiLiGenT+LUCES, qualitative analysis on Multi-Illumination, and thorough ablation studies are provided, though quantitative evaluation on a broader range of real-world scenes is missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The motivation is excellently justified, with a highly complete and logical argument on "why geometric priors are needed".
- **Value**: ⭐⭐⭐⭐⭐ — Opens up a new direction of introducing foundation model geometric knowledge to PS, inspiring other tasks relying on photometric signals.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Light of Normals: Unified Feature Representation for Universal Photometric Stereo](../../ICLR2026/3d_vision/light_of_normals_unified_feature_representation_for_universal_photometric_stereo.md)
- [\[ECCV 2024\] SpectraM-PS: Spectrally Multiplexed Photometric Stereo Under Unknown Spectral Composition](../../ECCV2024/3d_vision/spectram-ps_spectrally_multiplexed_photometric_stereo_under_unknown_spectral_com.md)
- [\[CVPR 2025\] Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection](../../CVPR2025/3d_vision/leveraging_3d_geometric_priors_in_2d_rotation_symmetry_detection.md)
- [\[CVPR 2026\] Geometric-Photometric Event-based 3D Gaussian Ray Tracing](../../CVPR2026/3d_vision/geometric-photometric_event-based_3d_gaussian_ray_tracing.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](../../CVPR2026/3d_vision/anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)

</div>

<!-- RELATED:END -->
