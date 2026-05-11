---
title: >-
  [Paper Note] Dynamic Novel View Synthesis in High Dynamic Range
description: >-
  [ICLR 2026][3D Vision][HDR] This paper is the first to formally define the HDR Dynamic Novel View Synthesis (HDR DNVS) problem and proposes the HDR-4DGS framework. Through a dynamic tone mapping module, the framework achieves temporally consistent HDR radiance field reconstruction in time-varying scenes, outperforming existing methods on both synthetic and real-world datasets.
tags:
  - ICLR 2026
  - 3D Vision
  - HDR
  - Dynamic Novel View Synthesis
  - 4D Gaussian Splatting
  - Tone Mapping
  - Radiance Field
date: 2026-05-08
content_hash: 5ba722f9d04604bb
---

# Dynamic Novel View Synthesis in High Dynamic Range

**Conference**: ICLR 2026
**arXiv**: [2509.21853](https://arxiv.org/abs/2509.21853)
**Code**: [prinasi/HDR-4DGS](https://github.com/prinasi/HDR-4DGS)
**Area**: 3D Vision
**Keywords**: HDR, Dynamic Novel View Synthesis, 4D Gaussian Splatting, Tone Mapping, Radiance Field

## TL;DR

This paper is the first to formally define the HDR Dynamic Novel View Synthesis (HDR DNVS) problem and proposes the HDR-4DGS framework. Through a dynamic tone mapping module, the framework achieves temporally consistent HDR radiance field reconstruction in time-varying scenes, outperforming existing methods on both synthetic and real-world datasets.

## Background & Motivation

### State of the Field

**Background**: Existing novel view synthesis methods are constrained by two assumptions: **static scenes** and **low dynamic range (LDR) inputs**.

### Root Cause

**Key Challenge**: **Dynamic Novel View Synthesis (DNVS)** can handle time-varying scenes (e.g., moving objects, changing illumination), but is limited to LDR images, losing information in over-/under-exposed regions under high-contrast conditions (direct sunlight, low-light environments).

### Limitations of Prior Work

**Limitations of Prior Work**: **HDR Novel View Synthesis (HDR NVS)** can reconstruct HDR scenes from multi-exposure LDR images, but existing methods (e.g., HDR-NeRF, HDR-GS, GaussHDR) all assume fully static scenes.

### Starting Point

**Key Insight**: **Real-world demand**: HDR scenes in the real world are inherently dynamic—containing moving objects, changing illumination, and transient phenomena. Existing methods cannot simultaneously handle dynamic geometry and HDR radiance reconstruction.

Although HDR-HexPlane preliminarily explores dynamic HDR reconstruction, it never carefully evaluates HDR output quality nor validates on real-world scenes, leaving substantial gaps.

### Paper Goals

**Goal**: **HDR Dynamic Novel View Synthesis (HDR DNVS)**: Given sparse, time-varying multi-exposure LDR inputs, learn an HDR 4D radiance field model $\mathcal{F}_h$ capable of rendering temporally consistent HDR images at arbitrary timestamps $t'$ and viewpoints $V'$. The core challenges are:

1. Jointly modeling continuously evolving scene structure and HDR radiance
2. Complex spatiotemporal inconsistencies caused by non-rigid motion and temporal variation
3. Severe photometric ambiguity due to the lack of reliable luminance priors in sparse LDR observations

## Method

### Overall Architecture: HDR-4DGS

HDR-4DGS is built upon 4D Gaussian Splatting and comprises two core components:

**1) Dynamic Scene Representation (4DGS)**

- Adopts 4D Gaussian Splatting as the scene representation backbone, introducing the temporal dimension into 3DGS
- Pixel observation $\mathbf{I}(u,v,t)$ depends jointly on spatial coordinates and timestamps
- The mean of each Gaussian is extended to $\mu = (\mu_x, \mu_y, \mu_z, \mu_t)$
- Uses 4D spherical harmonics (4DSH) to model the temporal evolution of appearance, constructing a radiance bank that supports HDR-LDR conversion
- Key improvement: expands the color representation space of vanilla 4DGS from LDR to HDR

**2) Dynamic Tone Mapper (DTM)**

The DTM is the core innovation of this work, inspired by the human visual adaptation mechanism:

- **Radiance Bank**: Stores the mean HDR color statistics $\mathbf{r}_t^h = \frac{1}{N}\sum_{i=1}^N \mathbf{c}_{i,t}^h$ for each timestamp
- **Dynamic Radiance Context Learner (DRCL)**: Uses a GRU to process the radiance signature sequence $\{\mathbf{r}_{t-k:t}^h\}$ over a sliding window of $k$ frames, generating a radiance context embedding $\mathbf{f}_t \in \mathbb{R}^d$
- **Adaptive Tone Mapping**: Concatenates log-domain HDR color with exposure time and radiance context, then maps to LDR color via a per-channel tone mapping function $g_\theta$:
$$\mathbf{c}_t^l = g_\theta([\log \mathbf{c}_t^h + \log e_t, \mathbf{f}_t])$$

**3) Model Optimization**

- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{ldr} + \alpha \mathcal{L}_{hdr}$
- The LDR loss incorporates **dual supervision**: pixel-level (2D tone-mapped LDR) and ray-level (3D rasterized LDR)
- The HDR loss uses $\mu$-law compression to align the HDR and LDR domains
- Image reconstruction loss $= (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$, with $\lambda=0.2$

## Key Experimental Results

### Datasets (Constructed by This Work)

| Dataset | Scenes | Type | Characteristics |
|--------|--------|------|------|
| HDR-4D-Syn | 8 | Synthetic | Multi-exposure video + synchronized multi-view LDR streams + HDR ground truth |
| HDR-4D-Real | 4 | Real | Synchronized capture with 6 iPhone 14 Pro cameras at three exposure levels |

### Core Results on HDR-4D-Syn (LDR Supervision Only)

| Method | HDR PSNR↑ | HDR SSIM↑ | HDR LPIPS↓ | Inference Speed (fps) |
|------|-----------|-----------|------------|---------------|
| HDR-NeRF | 8.54 | 0.062 | 0.552 | 0.061 |
| HDR-GS | 4.64 | 0.158 | 0.645 | 380.38 |
| HDR-HexPlane | 14.70 | 0.649 | 0.287 | 1.61 |
| **HDR-4DGS** | **25.88** | **0.865** | **0.076** | **40.80** |

- HDR PSNR exceeds the second-best method (HDR-HexPlane) by **11.18 dB**
- Inference speed is approximately **25×** faster than HDR-HexPlane and **669×** faster than HDR-NeRF
- With joint LDR+HDR supervision, HDR PSNR further improves to **30.40 dB**

### Ablation Study

- The best two-stage pipeline using independent HDR reconstruction (4DGS + KPNet, etc.) achieves only PSNR 20.92, far below the jointly optimized result of 25.88
- DTM vs. static MLP tone mapper: PSNR 25.88 vs. 23.92, LPIPS 0.076 vs. 0.142
- Contribution of pixel-level supervision: removing it leads to a PSNR drop of 1.03 dB
- Optimal temporal context length is $k=20$; performance degrades with both smaller (5/10) and larger (30) values

## Highlights & Insights

1. **High value in problem formulation**: The paper is the first to formally define the HDR DNVS problem, filling the gap in HDR synthesis for dynamic scenes.
2. **Elegant DTM design**: Inspired by the human visual adaptation mechanism, the DTM uses a GRU to model temporal radiance context for adaptive HDR-LDR conversion; the learned tone mapping curves are interpretable (monotonically increasing, dynamically adjusting with scene luminance).
3. **Comprehensive benchmark construction**: Two new datasets (synthetic + real) are introduced, providing a standardized evaluation platform for future research.
4. **Dual supervision strategy**: Joint pixel-level and ray-level constraints effectively mitigate overfitting in 3D tone mapping.
5. **Significant efficiency advantage**: The method achieves real-time inference speed while substantially improving reconstruction quality.

## Limitations & Future Work

1. **Structural degradation in motion regions**: Structural degradation persists in deformable regions, attributed by the authors to inherent limitations of the underlying 4DGS representation; stronger dynamic representations may be explored in future work.
2. **HDR metrics on real scenes are less prominent**: HDR PSNR on HDR-4D-Real (14.50 under LDR-only supervision) is lower than HDR-HexPlane (9.306 [sic]), though the authors attribute this to HDR ground truth noise and PSNR's preference for blurry reconstructions; visual quality is reportedly superior.
3. **Fixed temporal window**: $k=20$ is a fixed hyperparameter; adaptive window length selection is not explored.
4. **Limited real-world deployment scenarios**: The real dataset covers only 4 indoor scenes and does not include outdoor large-scale scenes, extreme weather, or other more complex conditions.
5. **Relatively longer training time**: HDR-4DGS training takes approximately 69–99 minutes, slower than HDR-GS (14–38 minutes).

## Related Work & Insights

| Dimension | HDR-NeRF / HDR-GS | HDR-HexPlane | HDR-4DGS (Ours) |
|------|-------------------|--------------|----------------|
| Static/Dynamic | Static | Dynamic | Dynamic |
| Tone Mapping | Static MLP | Static Sigmoid | Dynamic adaptive (GRU) |
| Temporal Consistency | N/A | Weak | Strong (radiance context learning) |
| HDR Evaluation | Yes | No | Yes (complete benchmark) |
| Real-time Performance | NeRF slow / GS fast | Slow (~1.6 fps) | Fast (~41 fps) |

## Related Work & Insights

- **Transferability of dynamic tone mapping**: The "radiance bank + sequence model" paradigm of the DTM can be generalized to other tasks requiring temporally adaptive color/radiance conversion, such as video HDR reconstruction and relighting under dynamic illumination.
- **Generality of the dual supervision strategy**: The joint pixel-level and ray-level supervision approach can be applied to other 3DGS-based color space conversion tasks.
- **Benchmark dataset value**: HDR-4D-Syn and HDR-4D-Real can be directly used for evaluation in subsequent dynamic HDR-related research.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel problem formulation; creative design of the dynamic tone mapping module
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic + real datasets, extensive ablation studies and visualizations
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated, rigorous mathematical exposition
- Value: ⭐⭐⭐⭐ — Opens a new direction in HDR DNVS, provides a complete benchmark, and releases code publicly

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)
- [\[CVPR 2026\] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](../../CVPR2026/3d_vision/physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)
- [\[ICLR 2026\] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](../../CVPR2026/3d_vision/instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[NeurIPS 2025\] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](../../NeurIPS2025/3d_vision/reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)

</div>

<!-- RELATED:END -->
