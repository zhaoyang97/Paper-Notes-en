---
title: >-
  [Paper Note] On the Generalization Capacities of MLLMs for Spatial Intelligence
description: >-
  [ICLR 2026][Multimodal VLM][Camera Awareness] This paper identifies a fundamental flaw in RGB-only spatial reasoning MLLMs—the focal-length–depth ambiguity arising from the neglect of camera intrinsics—and proposes the Camera-Aware MLLM (CA-MLLM) framework. Through dense camera ray embedding, camera-aware data augmentation, and geometric prior distillation, it improves F1 from 39.1% to 52.1% on cross-camera generalization benchmarks for spatial localization.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Camera Awareness
  - Spatial Intelligence
  - Cross-Camera Generalization
  - 3D Localization
  - Geometric Priors
date: 2026-05-08
content_hash: 5fb4c3e2e1422e5f
---

# On the Generalization Capacities of MLLMs for Spatial Intelligence

**Conference**: ICLR 2026
**arXiv**: [2603.06704](https://arxiv.org/abs/2603.06704)
**Code**: [github.com/Vegetebird/CA-MLLM](https://github.com/Vegetebird/CA-MLLM)
**Area**: 3D Spatial Understanding / MLLM
**Keywords**: Camera Awareness, Spatial Intelligence, Cross-Camera Generalization, 3D Localization, Geometric Priors

## TL;DR

This paper identifies a fundamental flaw in RGB-only spatial reasoning MLLMs—the focal-length–depth ambiguity arising from the neglect of camera intrinsics—and proposes the Camera-Aware MLLM (CA-MLLM) framework. Through dense camera ray embedding, camera-aware data augmentation, and geometric prior distillation, it improves F1 from 39.1% to 52.1% on cross-camera generalization benchmarks for spatial localization.

## Background & Motivation

**State of the Field**: MLLMs are increasingly applied to spatial reasoning tasks (3D localization, depth estimation, navigation). The dominant paradigm trains end-to-end on RGB images/videos without explicit 3D data, achieving competitive performance.

**Limitations of Prior Work**: RGB-only MLLMs ignore camera intrinsics, making them unable to disambiguate "small nearby objects" from "large distant objects" (size–depth ambiguity) or "wide-angle close-up views" from "telephoto distant views" (focal-length–depth ambiguity). As a result, these models overfit to the camera distribution seen during training.

**Root Cause**: In the projection equation $h_{\text{proj}} = fH/Z$, the triplet $(f, H, Z)$ forms an equivalence class $(f, H, Z) \sim (\lambda f, H, \lambda Z)$. Without camera intrinsics, this entanglement cannot be resolved—this is not a problem of model scale or architecture, but a fundamental information deficiency.

**Paper Goals**: Enable MLLMs to perform accurate spatial reasoning across diverse camera configurations, rather than only on cameras seen during training.

**Starting Point**: Drawing lessons from camera-aware monocular metric depth estimation (Metric3D, UniDepth) and generalizing them to universal spatial reasoning at the MLLM level.

**Core Idea**: Inject camera intrinsics as per-token conditioning information into the MLLM, enabling the model to disentangle camera properties from scene content and achieve cross-camera generalization.

## Method

### Overall Architecture

The input image is processed by a Geometry-Aware Visual Encoder: the visual encoder extracts features $F_{\text{vis}}$ → dense camera ray embeddings $E_{\text{cam}}$ are added → geometric prior embeddings $E_{\text{geo}}$ are added → the result is projected into an LLM for multimodal reasoning. Camera-aware geometric augmentation is applied during training.

### Key Designs

1. **Dense Camera Ray Embedding**:
    - Function: Encodes camera intrinsic information into each visual token.
    - Mechanism: For each grid position $(i,j)$, normalized ray directions are computed as $R_x[i,j] = (u_{ij} - c_x) / f_x$ and $R_y[i,j] = (v_{ij} - c_y) / f_y$, together with global focal lengths $f_x, f_y$. A sinusoidal embedding layer generates $E_{\text{cam}} \in \mathbb{R}^{H \times W \times D}$, which is added element-wise to $F_{\text{vis}}$.
    - Design Motivation: Compared to Metric3D's image canonicalization scheme (which is computationally expensive and generates many invalid tokens), directly injecting ray information into each token is more efficient and preserves the original resolution.

2. **Camera-Aware Geometric Augmentation**:
    - Function: Synthesizes diverse camera parameters during training to expand the camera distribution.
    - Mechanism: Two types of transforms are applied to training images: (i) scaling—an image scale factor $s$ with synchronized intrinsic updates $(f_x, f_y, c_x, c_y) \mapsto (sf_x, sf_y, sc_x, sc_y)$; (ii) translation—shifting the principal point $(c_x, c_y)$ to simulate off-center projection. Both the image and intrinsics are updated consistently to preserve geometric correctness.
    - Design Motivation: Existing 3D datasets cover a limited variety of cameras (e.g., ScanNet predominantly uses the Structure Sensor). Training on real data alone is insufficient to disentangle camera properties from scene content.

3. **Geometric Prior Distillation**:
    - Function: Distills 3D geometric knowledge from a pretrained monocular metric depth estimation model.
    - Mechanism: A frozen UniDepth v2 (trained on 10M+ RGB-depth pairs) predicts dense 3D point clouds for each training image, which are encoded as prior embeddings $E_{\text{geo}} \in \mathbb{R}^{H \times W \times D}$ and added to the visual features. At inference, the model remains RGB-only.
    - Design Motivation: UniDepth can estimate intrinsics directly from images, enabling the framework to extend to internet images without available intrinsics and resolving the challenge posed by large 2D datasets lacking camera parameters.

### Loss & Training

Training is based on the VG-LLM baseline, with joint training on multiple datasets including ScanNet, ARKitScenes, Matterport3D, 3RScan, SUN RGB-D, and Objectron. For general spatial reasoning, LLaVA-Video-178k and SPAR data are additionally incorporated.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (4B) | Comparison | Notes |
|--------|------|------|------|------|
| SPAR-Bench (full) | Avg. | 68.35 | 63.25 (SPAR-8B) | Surpasses 8B baseline |
| SPAR-Bench (full) | High-level | 81.74 | 72.92 (VG-LLM-4B) | Large advantage on high-level spatial reasoning |
| VSI-Bench | Abs. Dist. | 71.3 | 66.0 (VG-LLM-4B) | Significant gain in absolute distance estimation |
| CV-Bench-3D | Avg. | 90.7 | 91.3 (VG-LLM-4B) | On par with VG-LLM |
| BLINK-Spatial Multi. View | Multi-view | 87.2 | 54.1 (VG-LLM-4B) | +33.1 gain on multi-view understanding |

### Ablation Study

| Configuration | $F1_{0.25}$ | Notes |
|------|------|------|
| Baseline (no components) | 39.1 | ScanNet-val ×1.2 cross-camera test |
| + Ray Embedding | 41.2 | +2.1, ray embedding is effective |
| + Geom. Augmentation | 42.0 | +2.9, camera distribution diversity is beneficial |
| + Prior Distillation | 43.1 | +4.0, largest individual contribution |
| Ray + Prior | 44.3 | Synergy between the two |
| All Components | 52.1 | +13.0, qualitative leap from joint use |

### Key Findings

- Camera-agnostic MLLMs suffer catastrophic performance drops under simple image scaling (F1: 46.5→25.8 at 0.8× scale), demonstrating that the models learn camera-specific shortcuts rather than general 3D geometric principles.
- Mixed training across multiple datasets actually degrades performance on ScanNet (F1: 46.5→46.0), because conflicting geometric signals from different cameras interfere with each other.
- Ablation results show that all three components must work together: individual components yield limited gains, while their combination produces a qualitative leap (39.1→52.1).

## Highlights & Insights

- The depth of the theoretical analysis is impressive: starting from the projection equation, the paper derives the equivalence class ambiguity, which perfectly explains the generalization failures observed experimentally (simple scaling induces a systematic depth prediction bias $Z_{\text{pred}} \approx Z_{\text{physical}}/s$). The problem diagnosis itself constitutes a significant contribution.
- The elegance of geometric prior distillation lies in enabling the framework to scale to internet images without intrinsics. UniDepth effectively serves as an "intrinsic estimator," substantially expanding the usable training data.

## Limitations & Future Work

- Validation is currently limited to single-frame and video-based 3D detection/localization tasks; broader spatial reasoning tasks such as depth estimation and 3D reconstruction have not been explored.
- Geometric prior distillation depends on the quality of UniDepth v2 and may degrade in scenarios where UniDepth fails (e.g., extreme lighting, specular reflections).
- The model has only 4B parameters; detailed comparisons with large-scale models such as GPT-5 and Gemini-2.5-Pro remain limited.

## Related Work & Insights

- **vs VG-LLM**: VG-LLM represents the RGB-only paradigm; this paper uses it as the baseline to directly demonstrate the gains from camera awareness. VG-LLM exhibits severe performance degradation under cross-camera settings.
- **vs Metric3D / UniDepth**: These works establish the necessity of camera awareness for monocular metric depth estimation. This paper extends that insight to more general MLLM-based spatial reasoning.
- **vs SPAR-Bench**: SPAR provides a comprehensive spatial reasoning benchmark but does not address camera generalization. The proposed method achieves state-of-the-art results on SPAR-Bench.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic analysis and resolution of the camera generalization problem in MLLM spatial reasoning, with rigorous theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-camera generalization experiments are elegantly designed; ablations are comprehensive; multiple benchmarks are evaluated.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative arc from problem diagnosis → theoretical analysis → experimental validation is fluent and cohesive.
- Value: ⭐⭐⭐⭐⭐ Exposes a fundamental issue in the field; the proposed three-component framework is directly applicable to a wide range of spatial MLLMs.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence](../../CVPR2026/multimodal_vlm/spatialscore_towards_comprehensive_evaluation_for_spatial_intelligence.md)
- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](../../CVPR2026/multimodal_vlm/scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[ICLR 2026\] Spatial CAPTCHA: Generatively Benchmarking Spatial Reasoning for Human-Machine Differentiation](spatial_captcha_generatively_benchmarking_spatial_reasoning_for_human-machine_di.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)

<!-- RELATED:END -->
