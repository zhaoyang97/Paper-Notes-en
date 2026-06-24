---
title: >-
  [Paper Note] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World
description: >-
  [CVPR 2025][VLM Reasoning][Multimodal Large Language Models] This paper introduces Dyn-Bench, the first large-scale benchmark designed to systematically evaluate the capability of Multimodal Large Language Models (MLLMs) to perceive, track, and reason about dynamics in a physical 4D world. Composing of 1K videos, 7K VQA pairs, and 3K dynamic object localization pairs, it reveals that existing models fail to perform well simultaneously in both spatiotemporal reasoning and dyna…
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "Multimodal Large Language Models"
  - "Spatiotemporal Reasoning"
  - "4D World Understanding"
  - "Dynamic Perception"
  - "Benchmark Evaluation"
date: 2026-05-08
content_hash: fb17da04efafbab8
---

# Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World

**Conference**: CVPR 2025  
**arXiv**: [2603.12746](https://arxiv.org/abs/2603.12746)  
**Code**: [https://dyn-bench.github.io/](https://dyn-bench.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models, Spatiotemporal Reasoning, 4D World Understanding, Dynamic Perception, Benchmark Evaluation

## TL;DR
This paper introduces Dyn-Bench, the first large-scale benchmark designed to systematically evaluate the capability of Multimodal Large Language Models (MLLMs) to perceive, track, and reason about dynamics in a physical 4D world. Composing of 1K videos, 7K VQA pairs, and 3K dynamic object localization pairs, it reveals that existing models fail to perform well simultaneously in both spatiotemporal reasoning and dynamic localization. Furthermore, two structured enhancement methods, Mask-Guided Fusion and ST-TCM, are proposed to significantly improve performance.

## Background & Motivation

**Background**: Multimodal Large Language Models (such as GPT-4V, Gemini, LLaVA, etc.) have achieved remarkable success in static image understanding, enabling complex visual question answering, spatial reasoning, and image description. However, humans live in a physical 4D world (3D space + time dimension) where geometric structures and semantic content evolve dynamically over time.

**Limitations of Prior Work**: Can existing MLLMs truly "think in dynamics"? That is, can they perceive, track, and reason about spatiotemporal dynamics in evolving scenes? A systematic evaluation of this is currently lacking. Existing video datasets typically focus on high-level semantics (such as action recognition) or evaluate only simple temporal sequencing, lacking a comprehensive assessment of **localized dynamic perception** (which object is moving, how it moves, and its impact on other objects) and **spatiotemporal reasoning** (understanding motion trajectories, predicting interactions).

**Key Challenge**: Existing benchmarks fail to reveal the true capabilities of MLLMs in fine-grained dynamic understanding. High-level tasks (such as "what is happening in the video") can be bypassed through sparse keyframe sampling and static image understanding, requiring no genuine dynamic reasoning.

**Goal**: (1) Build a large-scale, high-quality dynamic understanding benchmark to systematically evaluate the spatiotemporal reasoning and dynamic localization capabilities of MLLMs across multiple dimensions; (2) diagnose the bottlenecks of existing models and explore enhancement strategies.

**Key Insight**: The authors decompose "thinking in dynamics" into three progressive capabilities: **Perception**—identifying dynamic elements in a scene; **Tracking**—tracking the movement and transformation of these elements over time; and **Reasoning**—performing causal reasoning and interaction prediction based on dynamic information. The evaluation covers both textual outputs (VQA) and visual outputs (localization).

**Core Idea**: Build a high-quality collection of dynamic scenes, Dyn-Bench, from large-scale 2D video and 4D databases via multi-stage filtering, to systematically evaluate the dynamic understanding capabilities of generalist, spatial, and region-level MLLMs, while introducing two structured enhancement methods.

## Method

### Overall Architecture
The construction and evaluation of Dyn-Bench are divided into three parts: (1) **Data Construction**: 1K high-quality videos of dynamic scenes are collected through multi-stage filtering from real-world and synthetic video datasets, labeled with 7K VQA pairs and 3K dynamic object localization pairs; (2) **Multi-dimensional Evaluation**: evaluation tasks covering dynamic perception, motion tracking, and spatiotemporal reasoning are designed to examine both textual expression (VQA accuracy) and visual expression (localization precision); (3) **Enhancement Methods**: Mask-Guided Fusion and ST-TCM are proposed to improve the dynamic understanding capabilities of MLLMs.

### Key Designs

1. **Multi-stage Filtering**:

    - **Function**: Filter high-quality scenes containing prominent dynamic changes from massive video datasets
    - **Mechanism**: Data sources include real-world video datasets (such as TAO, LaSOT, Waymo, etc.) and 4D synthetic datasets (such as Kubric, Replica, etc.). The filtering process comprises: **Stage 1: Motion Saliency Filtering**—filtering video segments containing significant target motion based on optical flow magnitude; **Stage 2: Scene Diversity Filtering**—ensuring the scenes cover indoor/outdoor environments and diverse object classes based on semantic categories; **Stage 3: Dynamic Complexity Filtering**—retaining segments containing complex dynamics such as multi-object interactions, occlusions, and non-rigid motion. Annotation is completed via a hybrid human-in-the-loop and model-assisted approach, where VQA pairs cover multiple dimensions such as "what is moving," "how does it move," "why does it move," and "what will happen next."
    - **Design Motivation**: Random sampling of videos makes it difficult to guarantee the quality and diversity of dynamic scenes. Multi-stage filtering ensures that each scene exhibits significant and meaningful dynamic behaviors.

2. **Mask-Guided Fusion**:

    - **Function**: Guide the MLLM to focus on dynamic targets using explicit object segmentation masks
    - **Mechanism**: A segmentation model (such as SAM) is applied to keyframes in the video to generate object masks. This mask information is integrated into the input of the MLLM in the form of visual prompts—for example, overlaying translucent mask highlights onto video frames or encoding masks as extra tokens. This enables the MLLM to clearly identify "which object to focus on" during VQA and reasoning.
    - **Design Motivation**: Existing MLLMs often focus on global semantics when processing videos, neglecting the precise locations and movements of localized targets. Mask-Guided Fusion compensates for this limitation through explicit spatial attention guidance.

3. **Spatio-Temporal Textual Cognitive Map (ST-TCM)**:

    - **Function**: Structure the spatiotemporal information of dynamic scenes into textual representations to assist MLLM reasoning
    - **Mechanism**: A textual "cognitive map" is constructed for each video, recording the position of each target at different timesteps, its motion direction, velocity estimates, and changes in spatial relationships with other targets. This structured text is fed into the MLLM as additional context. For example: "[t=1] Object A is on the left of the frame, Object B is on the right; [t=2] Object A moves to the right, and the distance to B decreases; [t=3] Object A makes contact with B." This structured representation helps the MLLM establish a temporal chain of reasoning.
    - **Design Motivation**: Traditional Chain-of-Thought (CoT) and caption-based hints have limited effectiveness in dynamic reasoning because they lack structured encoding of spatiotemporal information. ST-TCM provides a more compact yet information-rich spatiotemporal representation.

### Loss & Training
Dyn-Bench is essentially an evaluation benchmark and does not involve training. Mask-Guided Fusion and ST-TCM are inference-time enhancement strategies that do not require modifying model parameters.

## Key Experimental Results

### Main Results

Evaluated the performance of generalist MLLMs (GPT-4V, Gemini Pro, LLaVA-Next), spatial MLLMs (SpatialVLM), and region-level MLLMs (RegionGPT, Osprey) on Dyn-Bench.

| Model | VQA Dynamic Perception | VQA Spatiotemporal Reasoning | Dynamic Localization mIoU | Overall |
|------|------------|-----------|-------------|------|
| GPT-4V | 62.3 | 48.7 | 18.2 | 43.1 |
| Gemini Pro | 59.8 | 45.2 | 16.5 | 40.5 |
| LLaVA-Next | 55.1 | 40.3 | 22.4 | 39.3 |
| SpatialVLM | 51.2 | 43.8 | 28.6 | 41.2 |
| RegionGPT | 48.5 | 38.1 | 32.1 | 39.6 |
| GPT-4V + ST-TCM | **68.7** | **56.2** | 19.8 | **48.2** |
| RegionGPT + Mask Fusion | 52.1 | 41.6 | **38.5** | 44.1 |

### Ablation Study

| Enhancement Strategy | VQA Reasoning Gain | Localization Gain | Description |
|----------|------------|---------|------|
| No Enhancement (baseline) | — | — | Original model |
| Chain-of-Thought prompt | +1.8 | +0.5 | Limited effectiveness |
| Caption-based hints | +2.3 | +1.2 | Slight improvement |
| Mask-Guided Fusion | +3.5 | **+6.4** | Significant localization gain |
| ST-TCM | **+7.5** | +1.6 | Significant VQA reasoning gain |
| Mask + ST-TCM | +6.8 | +5.9 | Good performance when combined |

### Key Findings
- **An irreconcilable trade-off exists in current MLLMs between dynamic perception and spatiotemporal reasoning**: Generalist models adept at VQA reasoning (such as GPT-4V) perform poorly in localization, while models specialized in region-level localization struggle with reasoning. No single model excels at both simultaneously.
- Traditional prompting strategies like CoT and caption hints provide very limited improvements for dynamic reasoning (only +1.8 to +2.3), indicating that the bottleneck lies not in reasoning strategies, but in the model's capacity to represent and encode spatiotemporal dynamic information.
- **ST-TCM yields the most significant improvement in VQA reasoning (+7.5)**, demonstrating that structuring spatiotemporal dynamic information into text is an effective way to help MLLMs understand dynamics.
- **Mask-Guided Fusion yields the maximum gain in localization (+6.4)**, showing that explicit spatial attention guidance compensates for the general lack of localized target attention in MLLMs.
- Real-world videos are substantially more challenging than synthetic videos, as MLLM performance drops sharply under complex backgrounds and occlusions.

## Highlights & Insights
- **The evaluation dimensions of Dyn-Bench are systematically designed**: Covering a hierarchical progression from perception $\rightarrow$ tracking $\rightarrow$ reasoning, while evaluating both textual and visual output modalities. This makes it more comprehensive than existing video understanding benchmarks.
- **The finding that "existing models cannot perform both reasoning and localization simultaneously" is crucial**: It shines a light on a fundamental flaw in current MLLM architectural designs—the internal representations of textual reasoning and visual localization may be disjointed.
- **The design philosophy of ST-TCM is highly transferable**: Encoding any structured knowledge into a textual cognitive map to assist LLM reasoning is applicable not only to dynamic scenes, but also to graph-structured reasoning, multi-step planning, and other complex tasks.

## Limitations & Future Work
- Currently, Dyn-Bench scales to 1K videos / 7K VQA pairs / 3K localization pairs, which could be further expanded to serve as a larger benchmark.
- The data sources are predominantly secondary filterings of existing datasets, which might be inherently limited by the biases of the source datasets.
- Mask-Guided Fusion relies heavily on the quality of the segmentation model; inaccurate segmentation in densely occluded scenes will cause error propagation.
- The construction of textual cognitive maps in ST-TCM depends on accurate object detection and tracking results, exposing it to the risk of cascading errors.
- Integrating these enhancement methods directly into model training (e.g., using ST-TCM as a dataset for training-based augmentation) has not yet been explored.

## Related Work & Insights
- **vs. MVBench / Video-MME**: These video understanding benchmarks focus on high-level semantic understanding, whereas Dyn-Bench zeroes in on fine-grained spatiotemporal dynamics, which are better at exposing the deficiencies of MLLMs in dynamic perception.
- **vs. PointOdyssey / TAP-Vid**: These benchmarks evaluate point-level tracking capabilities but omit semantic reasoning. Dyn-Bench simultaneously covers both tracking and reasoning.
- **vs. SpatialBench**: SpatialBench evaluates static spatial understanding, while Dyn-Bench extends the assessment to dynamic understanding in the temporal dimension.

## Rating
- Novelty: ⭐⭐⭐⭐ The first benchmark to systematically evaluate the dynamic 4D understanding of MLLMs, offering a novel problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various types of MLLMs with comprehensive evaluation dimensions, accompanied by sufficient ablation studies on the enhancement strategies.
- Writing Quality: ⭐⭐⭐⭐ Structured concepts (perception-tracking-reasoning) are very clear, though the details of benchmark construction could be more transparent.
- Value: ⭐⭐⭐⭐ Exposes fundamental shortcomings of MLLMs in dynamic understanding, offering constructive guidance for the development of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](thinking_in_space_how_multimodal_large_language_models_see_remember_and_recall_s.md)
- [\[CVPR 2025\] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)
- [\[CVPR 2025\] Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models](coarse_correspondences_boost_spatial-temporal_reasoning_in_multimodal_language_m.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)
- [\[CVPR 2025\] SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](seqafford_sequential_3d_affordance_reasoning_via_multimodal_large_language_model.md)

</div>

<!-- RELATED:END -->
