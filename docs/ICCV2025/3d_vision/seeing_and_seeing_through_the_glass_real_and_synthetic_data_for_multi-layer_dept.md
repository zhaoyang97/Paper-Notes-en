---
title: >-
  [Paper Note] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation
description: >-
  [ICCV 2025][3D Vision][multi-layer depth estimation] This paper introduces the novel task of multi-layer depth estimation, constructs the LayeredDepth benchmark comprising 1,500 real-world images…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "multi-layer depth estimation"
  - "transparent objects"
  - "synthetic data"
  - "relative depth"
  - "benchmark dataset"
date: 2026-05-08
content_hash: 33369bf3958226a5
---

# Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation

**Conference**: ICCV 2025
**arXiv**: [2503.11633](https://arxiv.org/abs/2503.11633)  
**Code**: [Project Page](https://layereddepth.cs.princeton.edu)  
**Area**: 3D Vision
**Keywords**: multi-layer depth estimation, transparent objects, synthetic data, relative depth, benchmark dataset

## TL;DR

This paper introduces the novel task of multi-layer depth estimation, constructs the LayeredDepth benchmark comprising 1,500 real-world images, and develops a procedural synthetic data generator. The work reveals severe deficiencies of existing depth estimation methods when applied to transparent objects.

## Background & Motivation

Transparent objects are ubiquitous in daily life, and understanding their multi-layer depth information is critical for practical applications:

**Perceiving the transparent surface itself**: avoiding walking into glass doors/walls, grasping plastic bags.

**Perceiving objects behind the transparent surface**: retrieving items from transparent containers, recognizing scenes through windows.

Core limitations of existing depth datasets:
- **Single-layer depth annotations only**: datasets annotate either the depth of objects behind the transparent surface or the surface itself, but never both simultaneously.
- **Few transparent objects or limited scenes**: most datasets are restricted to tabletop objects or indoor environments.
- **Structured light/LiDAR cannot reliably measure transparent object depth**: emitted light penetrates the surface.

## Method

### Task Definition

Multi-layer depth estimation: given an image $\mathcal{I}$ and a query pixel $p=(x,y)$, predict an ordered depth sequence $\hat{\mathcal{D}} = \{\hat{d}_1, \dots, \hat{d}_n\}$, where $n$ varies per pixel. Each medium transition (e.g., air to water) defines a new layer.

### Real-World Benchmark (LayeredDepth)

- **Scale**: 1,500 CC0 images covering diverse scenes including households, restaurants, laboratories, and urban environments.
- **Annotation strategy**: relative depth annotation (metric depth cannot be accurately obtained for transparent objects).
    - Annotators draw monotonic depth lines or select reference points.
    - Each point is assigned a layer ID indicating its corresponding layer.
    - 10% fake tuples are included to penalize models that predict excessive layers.
- **Annotation volume**: 2.5M pairs, 5.9M triplets, and 5.8M quadruplets.

### Synthetic Data Generator (LayeredDepth-Syn)

A fully procedural generator built upon Infinigen Indoor:
- Infinite variation in materials, shapes, and scene compositions.
- Random material assignment system: any object can be designated as transparent.
- Generates 15,300 images with multi-layer depth annotations.

### Baseline Model Designs

Three baseline architectures for multi-layer depth estimation are proposed:
1. **Multi-head Output**: multiple output heads each predicting a distinct layer.
2. **Layer Index Concatenation**: layer index concatenated to the input.
3. **Recurrent**: layers predicted sequentially via recurrent inference.

## Key Experimental Results

### Main Results — Performance of Existing Methods on Transparent Objects

| Method | Quadruplet Accuracy (All) |
|--------|--------------------------|
| ZoeDepth | 42.98% |
| MiDaS | 52.26% |
| Metric3D V2 | 55.14% |
| Depth Anything V2 | 70.43% |
| **Metric3D ft. (fine-tuned)** | **75.20%** |

All existing state-of-the-art depth estimation methods perform poorly on transparent objects; Metric3D V2 achieves only 55.14%.

### Effect of Fine-Tuning on Synthetic Data

| Method | Before Fine-tuning | After Fine-tuning |
|--------|--------------------|-------------------|
| Metric3D V2 (quadruplet) | 55.14% | **75.20%** |

Fine-tuning on synthetic data yields approximately 20 percentage points improvement in quadruplet accuracy (+36.3% relative gain).

### Baseline Results on the Real-World Benchmark

| Method | All-P | All-T | All-Q | Layer1-P | Mixed-P |
|--------|-------|-------|-------|----------|---------|
| Multi-head | 63.42 | 42.55 | 25.97 | 65.66 | 74.72 |
| Index Concat | 64.46 | 44.00 | 26.00 | 66.95 | 76.70 |
| Recurrent | 62.36 | 41.88 | 24.64 | 68.08 | 73.51 |

Index Concatenation achieves the best overall performance, while the Recurrent approach degrades significantly at higher layer counts (Layers 5 and 7).

## Highlights & Insights

1. **Novel task formulation**: the first systematic definition of multi-layer depth estimation, filling a critical gap in transparent object understanding.
2. **Elegant annotation strategy**: relative depth annotation circumvents the fundamental difficulty of obtaining metric depth for transparent objects.
3. **Effectiveness of synthetic data**: training solely on synthetic data yields strong generalization to the real-world benchmark.
4. **Fake tuple design**: 10% fake tuples prevent models from over-predicting layers that do not exist.

## Limitations & Future Work

- Multi-layer prediction accuracy of the baseline models remains substantially below saturation (quadruplet accuracy is only ~26%).
- A domain gap between synthetic and real-world data persists.
- Annotations are produced manually rather than via crowdsourcing, limiting scalability.
- End-to-end metric multi-layer depth prediction is not explored.

## Related Work & Insights

- Metric3D, Depth Anything: single-layer depth estimation.
- Booster, ClearGrasp: transparent object datasets (single-layer only).
- Infinigen Indoor: synthetic scene generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (pioneering new task and dataset)
- Technical Depth: ⭐⭐⭐ (baselines are relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐ (comparison against multiple SOTA methods and baselines)
- Practical Value: ⭐⭐⭐⭐⭐ (transparent object understanding is a key open problem)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](bootstrap3d_improving_multiview_diffusion_model_with_synthet.md)
- [\[ICCV 2025\] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data](david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)

</div>

<!-- RELATED:END -->
