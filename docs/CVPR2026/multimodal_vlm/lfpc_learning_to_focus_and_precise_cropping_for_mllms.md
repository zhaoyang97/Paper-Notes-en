---
title: >-
  [Paper Note] LFPC: Learning to Focus and Precise Cropping for MLLMs
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal Large Language Models] LFPC proposes a two-stage pure reinforcement learning framework that addresses the spurious tool-calling behavior ("answer-before-crop") observed in existing agent-based MLLMs. It introduces an information gap mechanism — deliberately downsampling the global image to force the model to rely on high-resolution cropped regions — and a grounding loss to improve cropping precision, achieving state-of-the-art performance on high-resolution VQA benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multimodal Large Language Models
  - Reinforcement Learning
  - Cropping Tool
  - Information Gap
  - High-Resolution VQA
date: 2026-05-08
content_hash: f1bf36160b6dde82
---

# LFPC: Learning to Focus and Precise Cropping for MLLMs

**Conference**: CVPR 2026
**arXiv**: [2603.27494](https://arxiv.org/abs/2603.27494)
**Code**: [https://github.com/XuanPu-Z/LFPC](https://github.com/XuanPu-Z/LFPC)
**Area**: Multimodal VLM
**Keywords**: Multimodal Large Language Models, Reinforcement Learning, Cropping Tool, Information Gap, High-Resolution VQA

## TL;DR

LFPC proposes a two-stage pure reinforcement learning framework that addresses the spurious tool-calling behavior ("answer-before-crop") observed in existing agent-based MLLMs. It introduces an information gap mechanism — deliberately downsampling the global image to force the model to rely on high-resolution cropped regions — and a grounding loss to improve cropping precision, achieving state-of-the-art performance on high-resolution VQA benchmarks.

## Background & Motivation

Fine-grained perception in complex visual scenes remains a challenge for MLLMs. Agent-based approaches equip models with a cropping tool to actively zoom into regions of interest, but existing training strategies suffer from a critical flaw.

**Core Finding**: The authors analyze RL-based models such as DeepEyes and identify a concerning behavioral pattern — the model forms its answer *before* executing a crop, using the cropping action merely to *confirm* a pre-existing conclusion. A dedicated evaluation protocol is constructed to verify this hypothesis, revealing that the model exhibits weak dependence on the content within cropped regions.

**Key Challenge**: SFT+RL methods are bounded by the capability of the teacher model and incur high trajectory generation costs. Pure RL methods do not require a teacher, yet models learn to perform perfunctory cropping rather than genuinely exploiting the information contained in cropped regions.

## Method

### Overall Architecture

A two-stage pure RL training pipeline: Stage 1 trains the model to depend on cropped regions via the information gap mechanism; Stage 2 improves cropping precision via the grounding loss. No trajectory supervision is required.

### Key Designs

1. **Information Gap Mechanism**:

    - **Function**: Forces the model to genuinely rely on information from cropped regions when answering questions.
    - **Mechanism**: Rather than feeding the full high-resolution image directly, LFPC intentionally downsamples the input image to a lower resolution. The degree of downsampling is determined by the model's own uncertainty — specifically, the resolution at which the model begins to produce answers inconsistent with those obtained at full resolution. When the model invokes the cropping tool, however, the cropped patch is extracted from the original high-resolution image. This creates a critical *information gap* between the low-detail global view and the high-detail local view, making the cropped region a necessary source of information for correct answering.
    - **Design Motivation**: If the global image already contains sufficient information, the model has no incentive to genuinely exploit the crop. Only by making the global context informationally insufficient can the model be motivated to actively extract critical details from the cropped region.

2. **Grounding Loss**:

    - **Function**: Improves the precision of predicted cropping coordinates, ensuring the crop is placed at the correct location.
    - **Mechanism**: In Stage 2, a grounding reward signal is introduced using a small number of bounding box annotations. This reward encourages the model not only to invoke the cropping tool, but also to position the crop accurately in regions relevant to the answer. This constitutes a form of weak supervision — a modest amount of annotation suffices to yield substantial gains in cropping precision.
    - **Design Motivation**: Stage 1 addresses the question of *whether* the model depends on the crop; however, the cropping location may still be imprecise. A small amount of localization supervision can efficiently improve this capability.

3. **Uncertainty-Driven Resolution Selection**:

    - **Function**: Adaptively determines the degree of downsampling for each individual image.
    - **Mechanism**: For a given question, answers are sampled at multiple resolutions to identify the resolution threshold at which answers begin to diverge. This threshold defines the information gap boundary for that sample — low enough to create an informational need, yet not so low as to render the image entirely incomprehensible.
    - **Design Motivation**: Different images require different levels of detail; a uniform downsampling ratio may be excessively aggressive for simple questions and insufficiently challenging for complex ones.

### Loss & Training

Pure RL training based on the GRPO algorithm. Stage 1 employs an accuracy reward and a format reward; Stage 2 additionally incorporates a grounding reward. No trajectory data generated by a teacher model is required.

## Key Experimental Results

### Main Results

| Method | HR-Bench 4K | HR-Bench 8K | V* | Visual Tokens |
|--------|------------|------------|-----|---------------|
| DeepEyes | 74.0 | 68.0 | 85.9 | 16384 |
| LFPC (16K tokens) | **SOTA** | **SOTA** | **SOTA** | 16384 |
| LFPC (1K tokens) | Surpasses most 16K methods | Surpasses most 16K methods | Competitive | **1024** |

LFPC achieves state-of-the-art performance under both the 16K and 1K visual token budgets.

### Ablation Study

| Configuration | Crop Dependency | Performance | Note |
|---------------|----------------|-------------|------|
| DeepEyes baseline | Weak (answer-before-crop) | Baseline | Cropping is spurious behavior |
| Stage 1 (Information Gap) | Strong | Significant gain | Model genuinely exploits cropped information |
| Stage 1 + Stage 2 (Grounding) | Strong + Precise | SOTA | Cropping location is more accurate |

### Key Findings

- The information gap mechanism fundamentally alters the model's dependence on cropped regions — shifting from *confirmatory cropping* to *exploratory cropping*.
- Under the 1K token budget, LFPC still outperforms several 16K token methods, demonstrating that precise selection of *what* to crop matters more than the volume of tokens consumed.
- A small number of bounding box annotations in Stage 2 suffices to substantially improve cropping precision, keeping annotation costs low.

## Highlights & Insights

- **Incisive Problem Diagnosis**: The authors identify the "answer-before-crop" failure mode in RL-based agents and construct a dedicated evaluation to validate it. This "question first, then solve" research paradigm is a methodological model worth emulating.
- **Elegant Design of the Information Gap**: Controlling the informational content of the input to guide model behavior is more direct and effective than modifying the reward function. The approach is transferable to any agent tool-use setting.
- **Efficiency Advantage**: The 1K token configuration surpasses the 16K token baseline, confirming that *what* the model looks at is more important than *how much* it sees.

## Limitations & Future Work

- Resolution selection for the information gap mechanism requires pre-sampling, introducing additional preprocessing overhead.
- The current framework supports only a single cropping step; iterative multi-step cropping may yield further improvements.
- The grounding loss requires a modest but non-zero amount of annotation.
- Future work could explore multi-tool agent settings combining cropping, rotation, and image enhancement.

## Related Work & Insights

- **vs. DeepEyes**: A pure RL method that nonetheless suffers from spurious cropping; LFPC resolves this via the information gap mechanism.
- **vs. SFT+RL methods**: These approaches require teacher-generated trajectories, incurring high cost and inheriting the teacher's capability ceiling; LFPC is entirely teacher-free.
- **vs. attention-guided methods**: Such methods identify salient regions via attention maps but lack explicit cropping actions and provide no guarantee that crop information is actually utilized.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Incisive problem diagnosis and an elegantly designed information gap mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive multi-benchmark comparisons; ablations could be further detailed.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated and experimental findings are convincing.
- Value: ⭐⭐⭐⭐⭐ — Offers important insights for tool-use training in agent-based MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](linking_perception_confidence_and_accuracy_in_mllms.md)
- [\[CVPR 2026\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2026\] Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)
- [\[CVPR 2026\] FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy](fluoclip_stain-aware_focus_quality_assessment_in_fluorescence_microscopy.md)

</div>

<!-- RELATED:END -->
