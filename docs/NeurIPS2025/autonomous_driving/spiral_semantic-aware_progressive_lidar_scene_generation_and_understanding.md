---
title: >-
  [Paper Note] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding
description: >-
  [NeurIPS 2025][Autonomous Driving][LiDAR Generation] SPIRAL proposes a semantic-aware range-view LiDAR diffusion model that jointly generates depth maps, reflectance images…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "LiDAR Generation"
  - "Diffusion Models"
  - "Semantic Segmentation"
  - "Range-View"
  - "Closed-Loop Inference"
date: 2026-05-08
content_hash: fb8534225e4d1ecb
---

# SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding

**Conference**: NeurIPS 2025
**arXiv**: [2505.22643](https://arxiv.org/abs/2505.22643)
**Code**: [GitHub](https://github.com/worldbench/SPIRAL)
**Area**: Autonomous Driving / LiDAR Generation
**Keywords**: LiDAR Generation, Diffusion Models, Semantic Segmentation, Range-View, Closed-Loop Inference

## TL;DR

SPIRAL proposes a semantic-aware range-view LiDAR diffusion model that jointly generates depth maps, reflectance images, and semantic segmentation maps. By introducing progressive semantic prediction and a closed-loop inference mechanism to enhance cross-modal consistency, the model achieves state-of-the-art performance with a minimal parameter count of 61M.

## Background & Motivation

Large-scale acquisition and annotation of LiDAR data is prohibitively expensive. Leveraging diffusion models to synthesize LiDAR scenes has emerged as a promising direction to alleviate this data bottleneck. Existing generation approaches fall into two categories: voxel-based methods (e.g., XCube, DynamicCity), which can simultaneously generate geometric structures and semantic labels but incur high memory and computational costs, and range-view methods (e.g., LiDARGen, R2DM), which are computationally efficient but produce only unannotated depth and reflectance images.

**Limitations of Prior Work**: Existing range-view methods that require semantic labels must resort to a two-stage pipeline—first generating an unannotated scene, then predicting semantic maps using a pretrained segmentation model (e.g., RangeNet++). This approach has two critical drawbacks:
1. The generative and segmentation models are trained independently without shared representations, resulting in low training efficiency.
2. Semantic maps are predicted post-hoc and cannot guide the generation of depth and reflectance during the diffusion process, leading to poor cross-modal consistency.

**Key Insight**: The powerful feature learning capacity of diffusion models can be exploited to simultaneously predict semantic labels during denoising, and a closed-loop mechanism can allow semantic predictions to inversely guide geometric generation.

## Method

### Overall Architecture

SPIRAL employs a 4-level Efficient U-Net as the backbone, built upon a continuous-time DDPM framework. The input consists of noisy depth and reflectance images $x_t$ along with a semantic map $y$ (encoded as an RGB image). Two independent branches output the diffusion residual $\hat{\epsilon}_t$ and semantic labels $\hat{y}_t$, respectively. The model alternates between *unconditional steps* and *conditional steps*, controlled by two mutually exclusive switches $\mathcal{A}$ and $\mathcal{B}$.

### Key Designs

1. **Complete Semantic Awareness**:

    - Unconditional step: the model jointly predicts the semantic map $\hat{y}_t$ and noise $\hat{\epsilon}_t$; the loss is MSE + cross-entropy.
    - Conditional step: conditioned on a given semantic map $y$, only the denoising residual $\hat{\epsilon}_t$ is predicted; the loss is MSE.
    - During training, the two step types are randomly switched with 50% probability, with a unified loss: $\mathcal{L} = \mathcal{L}_c \cdot \mathbb{I}(\psi \leq 0.5) + \mathcal{L}_u \cdot \mathbb{I}(\psi > 0.5)$

2. **Progressive Semantic Predictions**:

    - At inference, each unconditional denoising step produces an intermediate semantic map $\hat{y}_t$.
    - Exponential moving average (EMA) smoothing is applied: $\bar{y}_t = \alpha \cdot \hat{y}_t + (1-\alpha) \cdot \bar{y}_{t+1}$
    - This suppresses stochastic fluctuations during diffusion and yields stable per-pixel confidence scores.
    - The final $\bar{y}_0$ serves as the semantic output.

3. **Closed-Loop Inference**:

    - Inference begins in open-loop mode, executing unconditional steps.
    - When the proportion of pixels in $\bar{y}_t$ with confidence exceeding threshold $\delta$ surpasses $\delta$ (default 0.8), the model switches to closed-loop mode.
    - In closed-loop mode, unconditional and conditional steps alternate: unconditional steps predict semantics and noise, while conditional steps use the current semantic map to guide depth/reflectance generation.
    - This achieves joint optimization of semantics and geometry, enhancing cross-modal consistency.

### Semantic-Aware Evaluation Metrics

The paper also introduces a new semantic-aware evaluation framework:
- **Learned features**: Features extracted by a RangeNet++ encoder and a LiDM semantic encoder are concatenated to compute S-FRD, S-FPD, and S-MMD.
- **Rule-based features**: Per-class BEV 2D histograms are computed and aggregated into $h^s \in \mathbb{R}^{C \times B \times B}$ to compute S-JSD and S-MMD.

## Key Experimental Results

### Main Results (SemanticKITTI)

| Method | Params | S-FRD↓ | S-FPD↓ | S-JSD↓ |
|--------|--------|--------|--------|--------|
| LiDARGen + RangeNet++ | 80M | 1216.61 | 710.79 | 28.65 |
| LiDM + RangeNet++ | 325M | — | 458.33 | 16.69 |
| R2DM + RangeNet++ | 81M | 559.26 | 363.16 | 18.13 |
| R2DM + SPVCNN++ | 128M | 555.09 | 351.73 | 18.67 |
| **SPIRAL (Ours)** | **61M** | **382.87** | **153.61** | **9.16** |

### Ablation Study

| Configuration | S-FRD↓ | S-FPD↓ | Notes |
|---------------|--------|--------|-------|
| w/o closed-loop inference | Higher | Higher | Closed-loop mechanism significantly improves cross-modal consistency |
| w/o EMA smoothing | Higher | Higher | EMA suppresses stochasticity during denoising |
| threshold δ=0.8 | Optimal | Optimal | Too low introduces noise contamination; too high delays closed-loop activation |

### Key Findings
- SPIRAL surpasses all two-stage methods (80–372M parameters) with only 61M parameters, achieving 31% improvement in S-FRD, 56% in S-FPD, and 50% in S-JSD.
- Larger segmentation models (SPVCNN++) perform worse than RangeNet++ on generated data, suggesting larger models are more sensitive to distributional noise.
- SPIRAL-generated data effectively augments downstream segmentation training, reducing annotation costs.
- The model generalizes well, achieving state-of-the-art results on the nuScenes dataset as well.

## Highlights & Insights

- **The closed-loop inference mechanism is particularly elegant**: feeding intermediate diffusion predictions back as conditional inputs enables mutual reinforcement between semantics and geometry—a design principle transferable to other multi-modal generation tasks.
- **EMA-based progressive semantic prediction** naturally exploits the iterative nature of the denoising process for gradual confidence accumulation.
- **Unified training rather than two-stage pipelines** eliminates the representational gap between generative and segmentation models, substantially reducing the total parameter count.
- **The proposed semantic-aware evaluation metrics** fill a gap in quality assessment for labeled LiDAR scene generation.

## Limitations & Future Work

- Range-view representations may lose fine-grained details of distant objects at high resolutions.
- Closed-loop inference increases the number of inference steps and overall runtime due to alternating between two step types.
- Semantic prediction relies on the diffusion model's feature learning capacity and may be insufficiently precise for rare object categories.
- Text-conditioned generation and 4D dynamic scene generation remain unexplored.

## Related Work & Insights

- **vs. R2DM**: R2DM generates only depth and reflectance and requires an external segmentation model; SPIRAL unifies the generation of all three modalities.
- **vs. LiDM**: LiDM supports semantic-conditioned generation but requires a semantic map as prior input; SPIRAL autonomously predicts semantics during generation.
- **vs. voxel-based methods (XCube, DynamicCity)**: Voxel-based methods carry large parameter counts and high computational overhead; SPIRAL's range-view representation is substantially more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ Closed-loop inference and progressive semantic prediction are meaningful contributions, though the core diffusion framework is standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two standard benchmarks + new evaluation metrics + extensive ablations + downstream application validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, professional figures, and coherent motivation.
- Value: ⭐⭐⭐⭐ Practically valuable for autonomous driving data generation, though impact is primarily confined to the LiDAR domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](../../ICCV2025/autonomous_driving/hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[NeurIPS 2025\] FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](../../ICCV2025/autonomous_driving/controllable_3d_outdoor_scene_generation_via_scene_graphs.md)

</div>

<!-- RELATED:END -->
