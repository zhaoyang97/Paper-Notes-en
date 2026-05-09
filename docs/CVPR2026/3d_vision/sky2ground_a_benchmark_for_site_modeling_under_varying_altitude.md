---
title: >-
  [Paper Note] Sky2Ground: A Benchmark for Site Modeling under Varying Altitude
description: >-
  [CVPR 2026][3D Vision][Cross-view localization] This paper introduces the Sky2Ground dataset (51 scenes, 80k images, covering satellite/aerial/ground views with both synthetic and real imagery) and the SkyNet model (dual-stream encoder + masked satellite attention + progressive view sampling), presenting the first systematic study of joint camera localization across ground, aerial, and satellite viewpoints. SkyNet achieves a 9.6% improvement in RRA@5 and an 18.1% improvement in RTA@5.
tags:
  - CVPR 2026
  - 3D Vision
  - Cross-view localization
  - satellite–aerial–ground
  - multi-altitude 3D reconstruction
  - Gaussian splatting
  - curriculum learning
date: 2026-05-08
content_hash: 35bd0260b10b7130
---

# Sky2Ground: A Benchmark for Site Modeling under Varying Altitude

**Conference**: CVPR 2026
**arXiv**: [2603.13740](https://arxiv.org/abs/2603.13740)
**Code**: Coming soon
**Area**: 3D Vision / Cross-View Localization
**Keywords**: Cross-view localization, satellite–aerial–ground, multi-altitude 3D reconstruction, Gaussian splatting, curriculum learning

## TL;DR
This paper introduces the Sky2Ground dataset (51 scenes, 80k images, covering satellite/aerial/ground views with both synthetic and real imagery) and the SkyNet model (dual-stream encoder + masked satellite attention + progressive view sampling), presenting the first systematic study of joint camera localization across ground, aerial, and satellite viewpoints. SkyNet achieves a 9.6% improvement in RRA@5 and an 18.1% improvement in RTA@5.

## Background & Motivation
1. **State of the Field**: Multi-view 3D reconstruction and camera localization are fundamental tasks in computer vision. Recent neural approaches such as DUSt3R, MASt3R, and VGGT have achieved notable progress, yet they are primarily trained and evaluated on ground-level and aerial viewpoints.
2. **Limitations of Prior Work**: (1) No existing dataset simultaneously covers ground, aerial, and satellite viewpoints — nuScenes/KITTI are ground-only, AerialMegaDepth lacks satellite imagery, and MatrixCity/BungeeNeRF contain only synthetic data. (2) Joint camera localization across all three viewpoints has not been studied. (3) A large distribution gap exists between satellite images and ground/aerial imagery.
3. **Root Cause**: Satellite imagery provides globally consistent geographic coverage and a stable reference, yet its visual appearance differs drastically from ground/aerial views (near-orthographic projection, kilometer-scale altitude difference). Intuitively, incorporating satellite images should supply more information, but experiments reveal that doing so actually degrades localization performance.
4. **Paper Goals**: (1) Construct the first dataset covering all three viewpoints with both real and synthetic imagery. (2) Analyze why satellite images harm existing model performance. (3) Propose a new architecture that effectively exploits satellite information.
5. **Starting Point**: The authors find that naïvely fine-tuning VGGT with satellite data causes an 18.2% performance collapse, whereas pairwise architectures such as DUSt3R/MASt3R actually benefit from satellite data. This indicates that the problem lies not in distribution shift per se, but in the global attention mechanism, which allows ground/aerial tokens to be disturbed by satellite tokens.
6. **Core Idea**: Masked satellite attention prevents ground/aerial tokens from attending directly to satellite tokens, while a progressive view sampling strategy gradually introduces increasingly distant viewpoints, enabling joint localization across altitudes.

## Method

### Overall Architecture
SkyNet is built upon VGGT and adopts a dual-stream encoder design. The GAS encoder produces a joint representation across all viewpoints while restricting attention interactions between ground/aerial tokens and satellite tokens. The Sat encoder processes satellite images exclusively. The two encoders are connected via additive fusion of satellite features. A shared Camera Head and DPT Head then predict camera parameters and depth maps, respectively.

### Key Designs

1. **Sky2Ground Dataset**:

    - **Function**: Provides the first multi-modal dataset covering satellite, aerial, and ground viewpoints jointly.
    - **Mechanism**: Encompasses 51 geographic locations worldwide. Each scene contains 120 satellite images (altitude 1–2 km, orthorectified), 1,080 synthetic aerial images (captured with a three-camera virtual rig following a helical descent trajectory at 250–800 m altitude), 50–250 synthetic ground-level images, and 120 real aerial/ground images each (manually collected from Google Maps and YouTube travel videos). Synthetic data are rendered via Google Earth Studio; dense depth maps are annotated using COLMAP.
    - **Design Motivation**: Real images introduce illumination variation and weather noise, while synthetic images provide precise camera poses and depth annotations. The two sources are complementary.

2. **Masked Satellite Attention (MSA)**:

    - **Function**: Prevents ground/aerial tokens from being disturbed by the heterogeneous distribution of satellite tokens.
    - **Mechanism**: Within each block of the GAS encoder, standard self-attention (intra-frame) is applied first, followed by MSA: satellite tokens may attend to ground/aerial tokens, but ground/aerial tokens are prohibited from attending to satellite tokens. The attention mask matrix is set to $-\infty$ in the satellite→ground/aerial direction. The self-attention and MSA layers of the GAS encoder are initialized with pretrained VGGT weights and kept frozen.
    - **Design Motivation**: Experiments show that fine-tuning VGGT causes a performance collapse because global attention allows ground/aerial features to be "contaminated" by satellite features. MSA preserves VGGT's zero-shot capability on ground/aerial views (since those tokens never interact with satellite tokens) while enabling satellite tokens to gather information from ground/aerial tokens.

3. **Progressive View Sampling (P-VS)**:

    - **Function**: Gradually increases training difficulty through a curriculum learning strategy.
    - **Mechanism**: In the early stage of training, more aerial images are sampled ($N_a \approx N$), serving as a "bridge" between ground and satellite views. As training progresses, the proportion of aerial images is progressively reduced ($N_a \approx 0$), until only ground and satellite images remain. This guides the model from the easier problem (three-view joint localization) to the harder problem (ground + satellite only).
    - **Design Motivation**: Ground and satellite represent an extreme viewpoint pair, making direct joint training highly challenging. Aerial views serve as an intermediate bridge, enabling the model to first establish progressive associations across ground–aerial–satellite.

### Loss & Training
A multi-task loss is employed: $\mathcal{L} = \mathcal{L}_{\text{cam, sat}} + 0.4 \cdot \mathcal{L}_{\text{cam, gnd/aerial}} + \mathcal{L}_{\text{depth}}$. A Curriculum Aware Camera-Sampling (CA-CS) strategy is also applied: training begins by sampling nearby camera pairs and progressively extends to distant pairs, with distance measured as rotation distance + 0.5 × translation distance.

## Key Experimental Results

### Main Results (GAS setting, RRA@5 / RTA@5 %)

| Method | Training Data | Ground RRA/RTA | Sat RRA/RTA | Aerial RRA/RTA | Avg. RRA/RTA |
|--------|--------------|----------------|-------------|----------------|--------------|
| VGGT | Zero-shot | 75.1/60.9 | 66.6/0.0 | 79.2/72.6 | 73.6/44.5 |
| VGGT | Sky2Ground | 50.0/46.1 | 86.6/53.3 | 29.7/31.5 | 55.4/43.6 |
| SkyNet | Sky2Ground | **76.7/64.2** | **88.9/57.3** | **84.0/78.1** | **83.2/66.5** |

### Ablation Study (G+S setting)

| Configuration | MSA | CA-CS | P-VS | Avg. Performance |
|---------------|-----|-------|------|-----------------|
| VGGT fine-tuned | ✗ | ✗ | ✗ | 47.8 |
| VGGT zero-shot | ✗ | ✗ | ✗ | 52.9 |
| +MSA | ✓ | ✗ | ✗ | 62.7 (+8.2) |
| +P-VS | ✗ | ✗ | ✓ | 61.1 (+7.3) |
| +MSA+CA-CS+P-VS | ✓ | ✓ | ✓ | **65.1 (+12.2)** |

### Key Findings
- **Fine-tuning VGGT with satellite data causes severe degradation**: RRA drops from 73.6% to 55.4% (−18.2%), which is the central finding.
- **MSA is the single most impactful component**: +8.2%, as it protects ground/aerial features from satellite interference.
- **P-VS outperforms CA-CS**: +7.3% vs. +1.4%, indicating that using aerial imagery as a bridge is more critical than near-to-far camera sampling.
- **Pairwise architectures can benefit from satellite data**: DUSt3R/MASt3R improve when satellite images are included, because the high co-visibility rate among satellite–satellite pairs facilitates global alignment.
- **Real images degrade rendering quality**: PSNR consistently decreases after incorporating real images, as the domain gap makes it difficult for Gaussian splatting to blend the two sources.
- **2DGS consistently outperforms 3DGS**: 2D Gaussian splatting achieves superior perceptual quality across all viewpoints and density settings.

## Highlights & Insights
- **The counter-intuitive finding that "more data leads to worse performance"** is highly thought-provoking: adding satellite imagery — an information-theoretically richer source — actually hurts performance, demonstrating that when distribution shift is sufficiently large, more data does not guarantee better results. This challenges the "scale everything" paradigm.
- **The MSA design is broadly transferable**: In any Transformer architecture involving heterogeneous modalities (e.g., text + image, RGB + thermal), asymmetric attention masking can be employed to mitigate interference from modalities with excessive distributional discrepancy.
- **Curriculum learning with aerial views as a "bridge modality"**: This training strategy of transitioning progressively from an intermediate modality to extreme modalities can be generalized to multi-modal alignment tasks at large.

## Limitations & Future Work
- The method is two-stage (pose prediction followed by Gaussian splatting); a unified end-to-end model warrants future exploration.
- 51 scenes may be insufficient for large-scale training.
- Orthorectification of satellite images requires additional preprocessing.
- Poses for real images are estimated via COLMAP, limiting annotation accuracy.
- More advanced domain adaptation techniques for bridging the synthetic-to-real gap have not been explored.

## Related Work & Insights
- **vs. AerialMegaDepth**: The most closely related dataset, but lacks satellite viewpoints; Sky2Ground is a strict superset.
- **vs. VGGT**: SkyNet builds upon VGGT while resolving its catastrophic failure on satellite viewpoints.
- **vs. DUSt3R/MASt3R**: Pairwise processing can leverage satellite information but incurs $O(N^2)$ complexity, making it unsuitable for real-time applications.
- **vs. Dragon**: Dragon also employs a progressive strategy to integrate images from different altitudes, but targets reconstruction only and does not address localization.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic study of joint three-view localization; MSA and P-VS designs are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers both localization and rendering tasks, with multiple baselines and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — In-depth analysis with clearly articulated counter-intuitive findings.
- Value: ⭐⭐⭐⭐ — The dataset and benchmark make an important contribution to the cross-view localization community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FreeArtGS: Articulated Gaussian Splatting Under Free-Moving Scenario](freeartgs_articulated_gaussian_splatting_under_free-moving_scenario.md)
- [\[CVPR 2026\] NimbusGS: Unified 3D Scene Reconstruction under Hybrid Weather](nimbusgs_unified_3d_scene_reconstruction_under_hybrid_weather.md)
- [\[CVPR 2026\] GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](ava_bench_atomic_visual_ability_vfm.md)
- [\[CVPR 2026\] RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations](rng_unified_transformer_complete_3d_modeling_partial_observations.md)

<!-- RELATED:END -->
