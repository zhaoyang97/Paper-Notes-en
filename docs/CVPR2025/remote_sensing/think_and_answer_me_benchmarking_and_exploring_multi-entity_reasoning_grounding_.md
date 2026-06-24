---
title: >-
  [Paper Note] Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing
description: >-
  [CVPR2025][Remote Sensing][Remote sensing visual grounding] Constructs the remote sensing multi-entity reasoning grounding benchmark ME-RSRG (the first remote sensing grounding dataset explicitly annotated with subject-object roles), and proposes the Entity-Aware Reasoning (EAR) framework. Combining SFT cold-start with entity-aware reward-driven GRPO optimization, it achieves structured reasoning chain outputs and joint subject-object localization…
tags:
  - "CVPR2025"
  - "Remote Sensing"
  - "Remote sensing visual grounding"
  - "multi-entity reasoning"
  - "reinforcement learning"
  - "GRPO"
  - "vision-language models"
date: 2026-05-08
content_hash: 59ea605a8e942327
---

# Think and Answer ME: Benchmarking and Exploring Multi-Entity Reasoning Grounding in Remote Sensing

**Conference**: CVPR2025  
**arXiv**: [2603.12788](https://arxiv.org/abs/2603.12788)  
**Code**: [github.com/CV-ShuchangLyu/ME-RSRG](https://github.com/CV-ShuchangLyu/ME-RSRG)  
**Area**: Remote Sensing  
**Keywords**: Remote sensing visual grounding, multi-entity reasoning, reinforcement learning, GRPO, vision-language models

## TL;DR

Constructs the remote sensing multi-entity reasoning grounding benchmark ME-RSRG (the first remote sensing grounding dataset explicitly annotated with subject-object roles), and proposes the Entity-Aware Reasoning (EAR) framework. Combining SFT cold-start with entity-aware reward-driven GRPO optimization, it achieves structured reasoning chain outputs and joint subject-object localization, with the Qwen2.5-VL series obtaining over 10% improvement in mAcc@0.5 after EAR optimization.

## Background & Motivation

- Reasoning language models (such as OpenAI-o1, DeepSeek-R1) demonstrate powerful multi-step reasoning capabilities, which are further enhanced when combined with reinforcement learning and verifiable rewards.
- Remote sensing visual grounding (RSVG) scene characteristics: large-scale spatial layout, dense target distribution, complex semantic relationships $\rightarrow$ naturally suitable for the reasoning paradigm.
- **Limitations of Prior Work**:
    - Data level: Existing remote sensing grounding datasets only annotate a **single entity**, lacking structured annotations for multiple entities and their relationships.
    - Method level: Existing methods are essentially **perception-level matching** (aligning text with visual regions), lacking explicit intermediate reasoning processes, entity role modeling, and reasoning about inter-entity relations.
- **Key Insight**: Target identity in remote sensing is typically determined by its relationships with other entities (e.g., "aircraft next to the apron") rather than isolated appearance.

## Method

### ME-RSRG Dataset Construction (5-Step Pipeline)

1. **Data Collection**: Integrate three datasets: RSVG-HR, DIOR-RSVG, and OPT-RSVG.
2. **Instance Filtering**: Use Gemini-2.5-Pro to filter, keeping only descriptions containing explicit subject-object structures.
3. **Manual Annotation**: Inheriting the original subject bounding boxes, annotators further label the object entity bounding boxes, supporting "single subject - single object" and "single subject - multiple objects".
4. **Expert Review**: Two experts sequentially review the box coordinates and entity role assignments for each instance.
5. **CoT Generation**: Use Gemini-2.5-Pro to generate \<think\> reasoning descriptions for approximately 20% of the training set.

**Data Scale**: 7,162 images, 12,091 image-text instances; 10,305 train / 1,786 test; 2,149 instances with CoT annotations.

### EAR Framework

**Overall Architecture**
- Policy Model (vision-language base model, e.g., Qwen2.5-VL): Input image + description, output \<think\> reasoning chain + \<answer\> subject/object bounding boxes.
- Reference Model (frozen copy of the policy model): Used for KL divergence regularization.
- $G$ candidate outputs are sampled for each input, which are scored by the entity-aware reward model.

**Two-Stage Optimization**

**Stage I: SFT Cold Start**
- Train the model to learn output structures (\<think\>+\<answer\>), basic entity recognition, and coarse localization using 2,149 CoT-annotated data samples.

**Stage II: Entity-Aware Reward-driven GRPO**
- Extended to all 10,305 training samples (with CoT annotations removed), optimized through three types of reward signals:

1. **Two-Level Format Reward $R_{fmt}$** (maximum 0.6)
    - Structural Tag Reward (0.3): Correct \<think\>…\</think\>\<answer\>…\</answer\> format.
    - Entity Format Reward (0.3): Correct entity format (subject/object + coordinates) inside \<answer\>.

2. **Entity-Aware Localization Accuracy Reward $R_{ent}$**
    - Role-specific weighting (subject: 1.5, object: 1.25).
    - Graded IoU scoring: $>0.75 \rightarrow 1.0$, $>0.5 \rightarrow 0.8$, $>0.25 \rightarrow 0.4$, otherwise $\rightarrow 0$.

3. **Relation Consistency Reward $R_{rel}$** (bonus mechanism)
    - Simultaneous correct matching of both subject and object $\rightarrow$ +0.3.
    - Multiple correct matching objects $\rightarrow$ additional +0.3.

## Key Experimental Results

### Main Results (ME-RSRG Dataset)

| Model | Baseline mAcc | +SFT mAcc | +EAR mAcc | Gain |
|------|--------------|-----------|-----------|------|
| Qwen2.5-VL-3B | 9.18% | 28.79% | **40.41%** | +31.23% |
| Qwen2.5-VL-7B | 10.35% | 33.34% | **46.72%** | +36.37% |
| Qwen3-VL-4B | 17.25% | 36.13% | **37.65%** | +20.40% |
| InternVL3.5-4B | 0.51% | 32.80% | **36.62%** | +36.11% |
| InternVL3.5-8B | 2.33% | 33.78% | **36.92%** | +34.59% |

- The Qwen2.5-VL series all obtain over 10% improvement in mAcc@0.5 after EAR optimization (SFT $\rightarrow$ EAR).
- Strongest result: Qwen2.5-VL-7B reaches 46.72% mAcc@0.5.

### Ablation Study
- GRPO-only (without SFT): Qwen2.5-VL-3B obtains only 15.49% mAcc $\rightarrow$ SFT is a necessary prerequisite for GRPO.
- Removing the first-level format reward: The reasoning chain easily collapses, leading to unstable training.
- Removing the relation consistency reward: The model tends to predict only a single object, and both subject and object accuracy decrease.

## Highlights & Insights

1. **First Remote Sensing Multi-Entity Reasoning Localization Benchmark**: Explicitly annotates subject-object roles and relationships, elevating remote sensing visual grounding from "matching" to "reasoning".
2. **Fine-grained Reward Design**: The three-layer reward (format/localization/relationship) covers multiple levels from structure to semantics, with graded IoU scoring providing dense feedback.
3. **Reasonable Two-Stage Strategy**: SFT resolves formatting and initialization issues, while GRPO addresses accuracy and reasoning quality. The ablation study proves both are indispensable.
4. **Extensive Benchmarking**: Involves multiple series of models such as Qwen-VL, InternVL, and LLaVA, providing a comprehensive baseline reference.

## Limitations & Future Work

1. The dataset currently only supports the "single subject - single/multiple objects" setting and does not cover "multi-subject" scenarios.
2. CoT annotation relies on Gemini-2.5-Pro generation (only 20% of the training set), which may introduce bias.
3. The best result (46.72%) remains relatively low, indicating that multi-entity reasoning visual grounding is still highly challenging.
4. User instance filtering relies on LLMs, which may miss some valid multi-entity samples or introduce errors.
5. Only evaluates the single IoU threshold of mAcc@0.5, lacking analysis under different thresholds.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (A new paradigm for multi-entity reasoning visual grounding, with complete contributions in the dataset, method, and benchmark)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multi-model benchmarking, detailed ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, but some notations are dense)
- Value: ⭐⭐⭐⭐⭐ (Drives the paradigm shift of remote sensing visual grounding from matching to reasoning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GeoViS: Geospatially Rewarded Visual Search for Remote Sensing Visual Grounding](../../CVPR2026/remote_sensing/geovis_geospatially_rewarded_visual_search_for_remote_sensing_visual_grounding.md)
- [\[ICLR 2026\] Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded Geospatial Chain-of-Thought for Vision-Language Models](../../ICLR2026/remote_sensing/towards_faithful_reasoning_in_remote_sensing_a_perceptually-grounded_geospatial_.md)
- [\[CVPR 2026\] GeoCoT: Towards Reliable Remote Sensing Reasoning with Manifold Perspective](../../CVPR2026/remote_sensing/geocot_towards_reliable_remote_sensing_reasoning_with_manifold_perspective.md)
- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/remote_sensing/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](../../ICCV2025/remote_sensing/smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)

</div>

<!-- RELATED:END -->
