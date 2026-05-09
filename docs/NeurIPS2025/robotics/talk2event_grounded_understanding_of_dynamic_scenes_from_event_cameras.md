---
title: >-
  [Paper Note] Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras
description: >-
  [NeurIPS 2025][Robotics][event camera] Talk2Event introduces the first large-scale visual grounding benchmark for event cameras (30,690 annotated referring expressions across four grounding attribute types), and proposes the EventRefer framework, which employs a Mixture of Event-Attribute Experts (MoEE) to dynamically fuse appearance, status, viewer-relation, and inter-object-relation features. EventRefer surpasses existing methods across all three evaluation settings: event-only, frame-only, and fusion.
tags:
  - NeurIPS 2025
  - Robotics
  - event camera
  - visual grounding
  - multimodal
  - mixture of experts
  - autonomous driving
  - benchmark
date: 2026-05-08
content_hash: 633f715162d1d43b
---

# Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras

**Conference**: NeurIPS 2025
**arXiv**: [2507.17664](https://arxiv.org/abs/2507.17664)
**Code**: None
**Area**: Robotics
**Keywords**: event camera, visual grounding, multimodal, mixture of experts, autonomous driving, benchmark

## TL;DR
Talk2Event introduces the first large-scale visual grounding benchmark for event cameras (30,690 annotated referring expressions across four grounding attribute types), and proposes the EventRefer framework, which employs a Mixture of Event-Attribute Experts (MoEE) to dynamically fuse appearance, status, viewer-relation, and inter-object-relation features. EventRefer surpasses existing methods across all three evaluation settings: event-only, frame-only, and fusion.

## Background & Motivation

1. **Event cameras lack language grounding**: Event cameras offer microsecond-level latency and robustness to motion blur, making them well-suited for dynamic scene perception; however, connecting asynchronous event streams with natural language remains an unexplored problem.
2. **No event-domain datasets for visual grounding**: Existing visual grounding benchmarks (RefCOCO, ScanRefer, etc.) are built on RGB frames or point clouds, covering only static scenes and lacking temporally dynamic information under high-speed or low-light conditions.
3. **Absence of attribute-level annotations**: Most existing benchmarks focus solely on appearance attributes and lack structured descriptions of object motion states, viewer-relative spatial relations, and inter-object relations, limiting interpretability.
4. **Difficulty of grounding in dynamic scenes**: Fast-moving objects (pedestrians, cyclists) are prone to blur in conventional frames; while event cameras capture motion effectively, they lack texture information, necessitating cross-modal fusion solutions.
5. **Event-based detection methods lack language capability**: Existing object detection methods for event cameras (RVT, LEOD, etc.) produce only category predictions and do not support free-text referring expression comprehension.
6. **Rigid fusion strategies**: Prior event-frame fusion methods lack adaptive mechanisms for exploiting multi-attribute information and cannot dynamically reallocate attention based on scene dynamics.

## Method

### Overall Architecture

- **Function**: Construct the Talk2Event benchmark and propose the EventRefer model to localize target objects from event streams (with optional RGB frame fusion) based on natural language descriptions.
- **Design Motivation**: Event cameras hold unique advantages in dynamic scenes, yet both a dataset and a method for language-guided grounding are absent; this work addresses both gaps simultaneously.
- **Mechanism**: A structured annotation pipeline is built upon the DSEC driving dataset, leveraging temporal context frames to generate rich referring expressions. An attribute-aware grounding framework is designed by introducing mixture-of-experts fusion on top of a DETR-style architecture.

### Key Designs

#### Four-Attribute Annotation Scheme
- **Function**: Each referring expression is annotated with four grounding attributes—Appearance, Status, Relation-to-Viewer, and Relation-to-Others.
- **Design Motivation**: Appearance descriptions alone are insufficient for precise grounding in dynamic scenes; motion state ("turning left"), egocentric spatial relations ("front-left"), and inter-object relations ("next to the bus") provide critical disambiguation cues.
- **Mechanism**: Two context frames within $t_0 \pm 200\text{ms}$ are used; Qwen2-VL generates three distinct referring expressions per instance (averaging 34.1 words each). Attribute labels are assigned via fuzzy matching combined with a language model and validated by human annotators.

#### Positive Word Matching (PWM)
- **Function**: Maps attribute-relevant text spans within referring expressions to token-level binary positive maps.
- **Design Motivation**: Manual annotation of token ranges is infeasible across four attribute types; PWM enables attribute-aware supervision at the token level during training.
- **Mechanism**: For each attribute's cue phrases (e.g., "moving left" for Status), a fuzzy matcher locates all matching positions within the expression, projects character-level spans to token indices, and constructs a softmax-normalized positive map $m_i$.

#### MoEE: Mixture of Event-Attribute Experts
- **Function**: Four attribute-specific experts independently extract attribute-aware features, which are then adaptively weighted and fused into a final representation.
- **Design Motivation**: The informativeness of each attribute varies across scenes (motion cues dominate at night; appearance is more useful in daylight), and static fusion cannot adapt to such variation.
- **Mechanism**: (1) Attribute-aware masking: binary masks $m_i^{\text{att}}$ are applied to encoder hidden states, retaining attribute-relevant tokens alongside shared context; (2) each attribute's features are projected via an FFN to produce expert features $H_i^{\text{exp}}$; (3) mean-pooled descriptors from all four experts are concatenated and passed through a learnable projection $W$ to generate gating weights $\lambda$; (4) Gaussian noise is injected to prevent expert collapse, and the final fused representation is $H^{\text{fuse}} = \sum \lambda_i \cdot H_i^{\text{exp}}$.

#### Multi-Attribute Fusion Training and Inference
- **Function**: Treats the four attributes as co-grounding pseudo-targets for multi-target matching during training; at inference, scores from all four attributes are aggregated to select the final predicted box.
- **Design Motivation**: This exploits supervision signals from all attributes without increasing decoder complexity, ensuring dense supervision at every attribute dimension.
- **Mechanism**: During training, the GT box is replicated four times (one per attribute) and assigned to queries via Hungarian matching. At inference, each query computes a softmax dot-product score against all four attribute maps, and the box with the highest score is selected as the prediction.

## Key Experimental Results

### Main Results (Val Set, mAcc%)

| Method | Modality | mAcc | Ped | Rider | Car | Bus | Truck | mIoU |
|--------|----------|------|-----|-------|-----|-----|-------|------|
| BUTD-DETR | Frame | 48.91 | 22.66 | 20.44 | 61.94 | 33.93 | 35.93 | 84.30 |
| GroundingDINO | Frame | 44.50 | 15.62 | 8.62 | 57.70 | 32.52 | 41.20 | 68.67 |
| **EventRefer** | **Frame** | **55.47** | **27.64** | **51.10** | **65.76** | **47.02** | 32.22 | **85.76** |
| EvRT-DETR | Event | 29.34 | 15.45 | 5.50 | 39.24 | 7.74 | 9.26 | 75.66 |
| **EventRefer** | **Event** | **31.96** | 12.09 | **25.00** | **40.83** | **15.48** | **16.30** | **76.46** |
| FlexEvent | Fusion | 59.40 | 30.39 | 33.50 | 71.34 | 47.85 | 38.58 | 86.83 |
| **EventRefer** | **Fusion** | **61.82** | 31.15 | 44.23 | **73.85** | 41.07 | 41.70 | **87.32** |

**Key Findings**: EventRefer achieves the best mAcc and mIoU across all three modality settings. In the frame-only setting, the Rider category shows the most significant improvement (+24.4%), indicating that attribute-level reasoning is particularly effective for small, dynamic objects. Event-only performance remains lower than frame-based methods (due to limited texture), but fusion achieves the highest overall mAcc of 61.82%.

### Ablation Study (Event-only, Val Set, mAcc%)

| Configuration | mAcc |
|--------------|------|
| Baseline (w/o PWM / MAF / MoEE) | 22.07 |
| + PWM | 26.38 (+4.31) |
| + MAF | 27.01 (+4.94) |
| + PWM + MAF | 29.66 (+7.59) |
| + PWM + MAF + MoEE | **31.96 (+9.89)** |

**Key Findings**: All three components contribute independently and complementarily—PWM provides token-level attribute supervision, MAF enables independent attribute reasoning, and MoEE contributes an additional +2.3% through adaptive fusion. Among fusion strategies, MoEE (31.96%) substantially outperforms attention-based fusion (29.66%), additive fusion (28.39%), and concatenation fusion (27.50%).

### Attribute Contribution Analysis

Single-attribute experiments show Status (28.90%) > Appearance (27.98%) > Viewer (27.03%) > Others (26.97%), while combining all four attributes yields the optimal 31.96%, confirming their complementarity. Visualization of MoEE gating weights reveals that Rider/Bike categories rely more heavily on Status cues, while Bus/Truck categories favor Appearance and Viewer Relation, demonstrating the adaptive nature of the mechanism.

## Highlights & Insights

- First visual grounding benchmark for event cameras: 5,567 scenes and 30,690 referring expressions averaging 34.1 words each, making it one of the most linguistically rich grounding datasets.
- The four-attribute annotation scheme provides structured and interpretable grounding dimensions covering spatiotemporal and relational reasoning.
- The adaptive gating mechanism in MoEE dynamically adjusts attribute weights according to scene dynamics and lighting conditions, offering both performance gains and interpretability.
- Support for three modality settings (event-only, frame-only, fusion) affords broad research flexibility.

## Limitations & Future Work

- The dataset is built exclusively on DSEC (urban driving in Switzerland), limiting scene diversity and excluding extreme weather, nighttime, and crowded scenarios.
- Event-only performance remains substantially below frame-based methods (31.96% vs. 55.47%), with insufficient semantic information in event streams being the fundamental bottleneck.
- Referring expressions are generated by Qwen2-VL and subsequently validated by human annotators, which may introduce generation bias.
- No direct comparison is conducted against recent large-scale vision-language models (e.g., GPT-4V, Qwen2.5-VL) on this task.

## Related Work & Insights

| Dimension | Talk2Event | RefCOCO / RefCOCOg |
|-----------|-----------|-------------------|
| Sensor | Event camera (optional RGB frame) | RGB frame |
| Scene Type | Dynamic driving scenes | Static images |
| Attribute Annotation | 4 types (Appearance + Status + Viewer + Others) | Appearance only |
| Expression Length | 34.1 words (rich) | 3.5–8.4 words (short) |
| Temporal Information | ✓ (motion, trajectory) | ✗ |

| Dimension | Talk2Event | ScanRefer |
|-----------|-----------|-----------|
| Sensor | Event camera + RGB | RGB-D point cloud |
| Scene Type | Dynamic outdoor driving | Static indoor |
| Attribute Annotation | 4 attribute types | Appearance + inter-object relation only |
| Dataset Scale | 30,690 expressions | 51,583 expressions |
| Dynamic Support | ✓ (motion state, temporal reasoning) | ✗ |

## Rating

- ⭐⭐⭐⭐⭐ **Novelty**: First introduction of language grounding to the event camera domain; the four-attribute design is highly original.
- ⭐⭐⭐⭐ **Technical Quality**: MoEE is well-motivated; ablation studies are thorough; evaluation spans three modality settings comprehensively.
- ⭐⭐⭐⭐ **Value**: Directly advances multimodal perception research in autonomous driving and robotics.
- ⭐⭐⭐⭐ **Writing Quality**: The paper is clearly structured with informative figures and complete mathematical derivations.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] SegMASt3R: Geometry Grounded Segment Matching](segmast3r_geometry_grounded_segment_matching.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[NeurIPS 2025\] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)

<!-- RELATED:END -->
