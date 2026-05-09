---
title: >-
  [Paper Note] 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation
description: >-
  [CVPR 2026][Model Compression][4D understanding] This paper proposes 4D-RGPT and the Perceptual 4D Distillation (P4D) framework, which enhances 4D perception in MLLMs by distilling knowledge of depth and optical flow from frozen 4D perceptual expert models. It also introduces R4D-Bench, the first region-level 4D video question-answering benchmark.
tags:
  - CVPR 2026
  - Model Compression
  - 4D understanding
  - region-level VQA
  - perceptual distillation
  - temporal position encoding
  - depth perception
date: 2026-05-08
content_hash: bad60862d00f6259
---

# 4D-RGPT: Toward Region-level 4D Understanding via Perceptual Distillation

**Conference**: CVPR 2026
**arXiv**: [2512.17012](https://arxiv.org/abs/2512.17012)
**Code**: [GitHub](https://github.com/NVIDIA/4D-RGPT)
**Area**: Model Compression
**Keywords**: 4D understanding, region-level VQA, perceptual distillation, temporal position encoding, depth perception

## TL;DR

This paper proposes 4D-RGPT and the Perceptual 4D Distillation (P4D) framework, which enhances 4D perception in MLLMs by distilling knowledge of depth and optical flow from frozen 4D perceptual expert models. It also introduces R4D-Bench, the first region-level 4D video question-answering benchmark.

## Background & Motivation

Despite significant advances in visual understanding, MLLMs still fall short on tasks requiring fine-grained 3D structure and temporal dynamic reasoning. Key limitations include:

1. **Weak 4D perception**: Existing SFT/RL methods rely solely on text supervision and fail to effectively learn low-level 4D representations such as depth and optical flow.
2. **Lack of region-level prompting**: Existing 3D/4D VQA benchmarks either lack region prompts or do not cover dynamic scenes, making it impossible to evaluate understanding of "specific regions in a 4D context."
3. **Inference overhead**: Methods that inject knowledge via external 3D models (e.g., VG-LLM) introduce additional computational costs at inference time.

Core insight: 4D perception (depth + optical flow + motion segmentation + camera rays) should be an intrinsic capability of MLLMs, acquired through training-time distillation rather than relying on external modules at inference time.

## Method

### Overall Architecture

Video input → VLM visual encoder + timestamp position encoding → LLM backbone → Training-time branch: 4D perceptual decoder extracts latent/explicit 4D representations → P4D distillation aligns with frozen teacher → Inference time: only the standard VLM path is retained with no additional overhead.

### Key Designs

1. **Perceptual 4D Distillation (P4D)**:
    - **Function**: Transfers 4D perceptual knowledge from expert models into the MLLM.
    - **Mechanism**: Dual-branch distillation — latent distillation (aligning the MLLM's intermediate 4D features with the teacher's latent representations) + explicit distillation (aligning predicted depth/optical flow/motion signals with teacher outputs).
    - **Design Motivation**: Latent distillation provides abstract guidance, while explicit distillation ensures interpretable and precise signals; the training-time modules are removed at inference, resulting in zero additional overhead.

2. **Timestamp Position Encoding (TPE)**:
    - **Function**: Provides the MLLM with explicit temporal cues.
    - **Mechanism**: Encodes the sampling timestamp of each frame as a sinusoidal positional encoding, which is added to the visual features before being fed into the multimodal projector.
    - **Design Motivation**: Answering questions such as "the average speed of a vehicle" requires knowledge of the video duration, yet MLLMs do not by default perceive the true inter-frame time intervals.

3. **R4D-Bench Benchmark Construction**:
    - **Function**: The first region-level 4D VQA benchmark.
    - **Mechanism**: Starting from non-region questions in STI-Bench and VLM4D, entity keywords are extracted → GroundingDINO + SAM2 perform segmentation → SoM labeling → Qwen2.5-VL matches regions → manual verification.
    - Contains 1,517 region-prompted VQA instances across 9 task categories covering both static (dimension measurement / 3D localization / spatial relations) and dynamic (counting / translation / rotation / speed / displacement) scenarios.

### Loss & Training

- Total loss = SFT cross-entropy loss + latent distillation loss ($\mathcal{L}_{LD}$) + explicit distillation loss ($\mathcal{L}_{ED}$)
- Teacher model: L4P (frozen), providing four 4D modalities: depth / flow / motion / camray
- Training data: RoboFAC, SAT, VSTI-Bench training split, Wolf
- Baseline model: NVILA-Lite-8B

## Key Experimental Results

### Main Results (Non-Region Benchmarks)

| Benchmark | NVILA Baseline | 4D-RGPT | Gain |
|-----------|---------------|---------|------|
| STI-Bench | 33.8 | 37.6 | +3.8 |
| VLM4D | 46.5 | 52.7 | +6.2 |
| VSTI-Bench | 45.2 | 59.1 | +13.9 |
| Avg. (6 benchmarks) | — | — | +5.3 |

### R4D-Bench

| Method | Static | Dynamic | Overall |
|--------|--------|---------|---------|
| GPT-4o | 30.3 | 47.5 | 42.8 |
| NVILA-Lite-8B | 29.1 | 41.3 | 37.9 |
| 4D-RGPT-8B | 32.9 | 45.7 | 42.2 (+4.3) |

### Ablation Study

| Configuration | STI-Bench | R4D | Note |
|---------------|-----------|-----|------|
| Baseline | 33.8 | 37.9 | No distillation |
| + TPE | 35.5 | 39.8 | Temporal awareness |
| + LD | 36.6 | 41.0 | Latent distillation |
| + ED | 36.9 | 41.5 | Explicit distillation |
| + LD + ED (P4D) | 37.6 | 42.2 | Full method |

### Key Findings

- Latent and explicit distillation are complementary; neither can be omitted.
- TPE contributes most significantly on temporally sensitive tasks such as speed and acceleration estimation.
- P4D outperforms alternatives including direct SFT on 4D data, concatenating 4D features, and 4D positional encoding.
- Distillation modules exist only during training; inference incurs zero additional overhead.

## Highlights & Insights

- The "distill at training time, free at inference time" design paradigm is elegant — it enhances perception without increasing inference cost.
- The dual-branch distillation (latent + explicit) is more effective than either branch alone.
- R4D-Bench fills the gap in region-level 4D VQA; the construction pipeline is reusable.
- The results reveal that even GPT-4o achieves only 42.8% on region-level 4D reasoning, underscoring the considerable difficulty of the task.

## Limitations & Future Work

- The quality of the teacher model L4P directly affects distillation performance; limitations of the teacher are inherited by the student.
- R4D-Bench is converted from existing benchmarks rather than designed natively for 4D region-level tasks.
- Numerical estimation of speed and displacement in dynamic scenes remains insufficiently accurate.
- Validation is conducted only on 8B-scale models; larger models may exhibit different behaviors.

## Related Work & Insights

- **vs. SpaceR/ViLaSR**: RL-based methods optimize via text rewards without direct 4D perceptual supervision.
- **vs. VG-LLM/SD-VLM**: These methods rely on external 3D models at inference time; P4D distills at training time with zero inference overhead.
- **vs. 3DRS**: Handles only static 3D scenes; P4D extends to dynamic 4D settings, incorporating optical flow and motion segmentation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combined innovation of training-time 4D distillation and a region-level 4D benchmark.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Six external benchmarks plus the self-constructed R4D-Bench; comprehensive ablations and alternative comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear method diagrams, modular framework description, and reproducible benchmark construction pipeline.
- **Value**: ⭐⭐⭐⭐ — Provides an efficient and generalizable framework for enhancing 4D perception in MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](../../NeurIPS2025/model_compression/4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)
- [\[CVPR 2026\] PlanaReLoc: Camera Relocalization in 3D Planar Primitives via Region-Based Structure Matching](planareloc_camera_relocalization_in_3d_planar_primitives_via_region-based_struct.md)
- [\[ICLR 2026\] Understanding Dataset Distillation via Spectral Filtering](../../ICLR2026/model_compression/understanding_dataset_distillation_via_spectral_filtering.md)
- [\[CVPR 2026\] Understanding and Enforcing Weight Disentanglement in Task Arithmetic](understanding_and_enforcing_weight_disentanglement_in_task_arithmetic.md)
- [\[AAAI 2026\] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling](../../AAAI2026/model_compression/rethinking_long-tailed_dataset_distillation_a_uni-level_framework_with_unbiased_.md)

</div>

<!-- RELATED:END -->
