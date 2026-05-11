---
title: >-
  [Paper Note] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth
description: >-
  [CVPR 2026][Segmentation][RGB-D semantic segmentation] GeomPrompt learns lightweight geometric prompt modules for frozen RGB-D segmentation models, synthesizing task-driven depth proxy signals from RGB (without depth supervision). It achieves gains of +6.1 mIoU under missing depth and up to +3.6 mIoU under degraded depth.
tags:
  - CVPR 2026
  - Segmentation
  - RGB-D semantic segmentation
  - depth missing
  - modality robustness
  - geometric prompt
  - lightweight adaptation
date: 2026-05-08
content_hash: 8ef44f2719f61554
---

# GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth

**Conference**: CVPR 2026
**arXiv**: [2604.11585](https://arxiv.org/abs/2604.11585)
**Code**: [https://geomprompt.github.io](https://geomprompt.github.io)
**Area**: Semantic Segmentation
**Keywords**: RGB-D semantic segmentation, depth missing, modality robustness, geometric prompt, lightweight adaptation

## TL;DR
GeomPrompt learns lightweight geometric prompt modules for frozen RGB-D segmentation models, synthesizing task-driven depth proxy signals from RGB (without depth supervision). It achieves gains of +6.1 mIoU under missing depth and up to +3.6 mIoU under degraded depth.

## Background & Motivation

**Background**: RGB-D semantic segmentation improves performance by fusing depth information, but in real-world deployments depth sensors frequently fail, produce incomplete data, or suffer from severe noise (e.g., reflective/transparent surfaces, sensor malfunctions).

**Limitations of Prior Work**: (1) Distilling depth as privileged information into RGB still requires depth supervision; (2) using monocular depth estimation as a proxy requires additional models and targets depth reconstruction rather than segmentation optimization; (3) lightweight solutions specifically addressing "how to maintain segmentation performance under missing or degraded depth" are lacking.

**Key Challenge**: RGB-D segmenters expect depth inputs to provide geometric priors, yet at deployment depth may be unavailable or unreliable. The key question is whether one can learn a "good enough" geometric signal to satisfy the segmenter without actually reconstructing depth.

**Core Idea**: Learn "task-driven geometric prompts" rather than "reconstruct depth"—train the prompt generation module using only the segmentation loss, allowing it to automatically discover the geometric signal most useful for segmentation.

## Method

### Overall Architecture
A frozen RGB-D segmentation model (e.g., DFormer, GeminiFusion) → the GeomPrompt module generates geometric prompts from RGB to replace missing depth / the GeomPrompt-Recovery module predicts corrective residuals over degraded depth → prompts are normalized, passed through a PromptAdapter and low-pass projection, then fed into the depth channel of the segmenter.

### Key Designs

1. **GeomPrompt (missing depth)**:

    - **Function**: Synthesize task-relevant geometric prompts from RGB.
    - **Mechanism**: A ViT-S/16 encoder extracts RGB features → a lightweight CNN decoder predicts a low-resolution residual map → anti-aliased upsampling to full resolution → a neutral-gray prior of 127.5 is combined with a bounded residual $s \cdot \tanh$ → normalization + PromptAdapter (zero-initialized residual module) + low-pass projection. Trained exclusively with the segmentation loss.
    - **Design Motivation**: Rather than reconstructing depth, the module learns "geometrically informative signals useful for segmentation," avoiding the objective mismatch between depth estimators and segmenters.

2. **GeomPrompt-Recovery (degraded depth)**:

    - **Function**: Repair components of degraded depth that are harmful to segmentation.
    - **Mechanism**: A dual-path design — an RGB ViT branch + a lightweight depth-conditional encoder (4-layer stride-2 CNN); features are concatenated and fused to predict a bounded corrective residual: $p_{raw} = \text{clamp}(\tilde{d} + s \cdot \tanh(\Delta_{full}), 0, 255)$. The correction head is zero-initialized, starting as an identity mapping.
    - **Design Motivation**: Degraded depth may still be largely useful; only the portions harmful to segmentation need repair. Zero initialization ensures the model begins from "no change to depth" and learns only necessary corrections.

3. **Parameterization & Regularization**:

    - **Function**: Ensure the generated prompts are stable and lie within the segmenter's expected input space.
    - **Mechanism**: Bounded residuals (tanh clipping) + progressive scaling factor (small residual magnitude early in training, gradually relaxed) + TV smoothness regularization + L1 magnitude regularization + low-pass projection (suppressing high-frequency artifacts).
    - **Design Motivation**: Since the segmenter is trained on normal depth, the generated prompts must "look like depth" to be processed correctly.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{seg}(\text{OHEM CE}) + \lambda_{tv} \mathcal{L}_{tv}(p_{raw}) + \lambda_\delta \|\Delta\|_1$$
GeomPrompt-Recovery is trained with randomly synthesized depth degradation (spatial dropout, quantization, noise).

## Key Experimental Results

### Main Results

| Setting | Model | Baseline (RGB-only) | + GeomPrompt | Gain |
|---------|-------|---------------------|-------------|------|
| Missing depth | DFormer | 43.8 mIoU | 49.9 mIoU | +6.1 |
| Missing depth | GeminiFusion | 47.2 mIoU | 50.2 mIoU | +3.0 |
| Degraded depth (severe) | DFormer | 45.x mIoU | +3.6 mIoU | improved |

### Ablation Study

| Configuration | mIoU | Latency | Notes |
|---------------|------|---------|-------|
| GeomPrompt | 49.9 | 7.8ms | Lightweight and efficient |
| Depth Anything V2 | 50.1 | 38.3ms | Similar accuracy but 5× slower |
| Metric3Dv2 | 49.6 | 71.9ms | Slower and lower accuracy |
| Neutral-gray fill | 43.8 | 0ms | Baseline |

### Key Findings
- GeomPrompt achieves accuracy competitive with Depth Anything V2 (38.3ms) at only 7.8ms latency, demonstrating a clear efficiency advantage.
- Task-driven geometric prompts need not be precise depth maps—the segmenter only requires "good enough" geometric priors.
- Zero initialization combined with progressive scaling is critical for training stability.

## Highlights & Insights
- **Paradigm shift**: From "estimating depth" to "generating geometrically useful signals for the task," eliminating the need for depth supervision and additional pretraining.
- **Plug-and-play**: Applicable to any frozen RGB-D segmenter without modifying the backbone.

## Limitations & Future Work
- GeomPrompt must be trained separately for each segmenter.
- Recovery capability under extreme degradation remains limited.
- Future work may explore universal geometric prompts transferable across segmenters.

## Related Work & Insights
- **vs. Depth Anything V2**: General-purpose depth estimation objectives are not fully aligned with segmentation objectives; GeomPrompt directly optimizes for segmentation.
- **vs. Privileged information distillation**: Distillation methods modify backbone weights, whereas GeomPrompt leaves the backbone entirely untouched.

## Rating
- Novelty: ⭐⭐⭐⭐ The "task-driven geometric prompt" perspective is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Both missing and degraded depth settings are evaluated.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is exceptionally clear; parameterization design is meticulous.
- Value: ⭐⭐⭐⭐ Practically valuable for robotics and embedded perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[CVPR 2026\] REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion](rel-sf4pass_panoramic_semantic_segmentation_with_rel_depth_representation_and_sp.md)
- [\[CVPR 2026\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)
- [\[CVPR 2026\] Low-Data Supervised Adaptation Outperforms Prompting for Cloud Segmentation Under Domain Shift](low_data_supervised_adaptation_outperforms_prompting_for_cloud_segmentation.md)

</div>

<!-- RELATED:END -->
