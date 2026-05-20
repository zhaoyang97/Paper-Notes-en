---
title: >-
  [Paper Note] RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping
description: >-
  [ICCV 2025][Segmentation][affordance segmentation] This paper introduces RAGNet, the first large-scale reasoning-based affordance segmentation benchmark (273k images, 180 categories, 26k reasoning instructions)…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "affordance segmentation"
  - "robotic grasping"
  - "reasoning instructions"
  - "large-scale benchmark"
  - "vision-language model"
date: 2026-05-08
content_hash: a251524e8d9ae7ba
---

# RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping

**Conference**: ICCV 2025
**arXiv**: [2507.23734](https://arxiv.org/abs/2507.23734)  
**Code**: [GitHub](https://github.com/DongmingWu/RAGNet)  
**Area**: Image Segmentation
**Keywords**: affordance segmentation, robotic grasping, reasoning instructions, large-scale benchmark, vision-language model

## TL;DR

This paper introduces RAGNet, the first large-scale reasoning-based affordance segmentation benchmark (273k images, 180 categories, 26k reasoning instructions), and proposes the AffordanceNet framework, which integrates VLM-pretrained affordance prediction with grasp pose generation, demonstrating strong open-world generalization and reasoning capabilities.

## Background & Motivation

General-purpose robotic grasping systems must accurately perceive object affordance regions across diverse open-world scenarios in response to human instructions. Prior work suffers from two key bottlenecks:

**Insufficient data scale and diversity**: Existing affordance datasets are typically confined to specific domains (robotics, egocentric, or in-the-wild), with limited categories (UMD: 17, AGD20k: 50) and homogeneous image sources, resulting in poor open-world generalization.

**Lack of reasoning-based instructions**: Existing methods rely on fixed-template language prompts (e.g., "Please segment the affordance region of XX"), which cannot handle human-like high-level instructions such as "I need something that can cut bread," lacking complex reasoning capabilities.

**Key insight**: Instructions received by robots in real-world scenarios often do not directly specify the target object category but instead describe functional requirements. This demands that affordance prediction models possess a complete capability chain: from functional description reasoning to object identification to region segmentation.

## Method

### Overall Architecture

The paper makes two primary contributions:
1. **RAGNet Benchmark**: Large-scale multi-source data collection + multi-tool affordance annotation + three-tier reasoning instruction construction.
2. **AffordanceNet Model**: AffordanceVLM (affordance region prediction) + Pose Generator (grasp pose generation).

### Key Design 1: Multi-source Data Collection and Annotation

**Data sources spanning four domains**:
- **Wild**: HANDAL (17 categories of hardware/kitchen tools, real-world diverse scenes)
- **Robot**: Open-X (124 categories, robotic manipulation scenes), GraspNet (32 categories)
- **Ego-centric**: EgoObjects (74 indoor categories, first-person perspective)
- **Simulation**: RLBench (10 categories, simulated environments)

In total: **273k images across 180 categories**, far exceeding prior datasets.

**Five-tier affordance annotation pipeline**:
- ❶ **Raw masks**: Directly using existing fine-grained annotations (e.g., HANDAL)
- ❷ **SAM2**: For objects without handles, GT bounding boxes guide SAM2 to generate masks
- ❸ **Florence2 + SAM2**: Language instructions drive Florence2 to produce polygon proposals, refined by SAM2
- ❹ **VLPart + SAM2**: VLPart's part-level recognition (e.g., knife handle, cup handle) combined with SAM2 for affordance region segmentation
- ❺ **Manual annotation (+ SAM2)**: Fallback when the above tools fail

Annotation strategies adapt to object characteristics: soda cans require whole-object annotation (grasping the entire object), while woks require only handle annotation.

### Key Design 2: Three-tier Reasoning Instruction System

1. **Template-based instructions**: Fixed templates such as "Please segment the affordance map of \<category\> in this image," applicable to all data.
2. **Easy reasoning instructions**: Instructions containing the object name, e.g., "Can you find a mug for tea."
3. **Hard reasoning instructions**: Instructions describing only functional requirements without mentioning the object name, e.g., "I need something to drink coffee."

GPT-4 is used to generate reasoning instructions, producing 26k in total (HANDAL: 8.5k hard; EgoObjects: 12.7k easy + 4.7k hard).

### Key Design 3: AffordanceNet Model

**AffordanceVLM**: VLM-based affordance region prediction module
- Input: RGB image + human instruction (template-based or reasoning-based)
- Output: affordance segmentation mask
- Pretrained on RAGNet's large-scale data to learn mappings from diverse instructions to affordance regions

**Pose Generator**: Grasp pose generation module
- Input: 2D affordance mask + depth image
- Output: 3D gripper pose
- Converts VLM's 2D predictions into executable robot actions

### Validation Benchmark Design

Four validation splits are constructed:
- **Zero-shot category recognition**: Evaluates generalization to unseen categories
- **Cross-domain affordance prediction**: Evaluates prediction on unseen image domains
- **Reasoning instruction validation**: Tests reasoning capability using hard instructions without category names
- **Real-robot closed-loop grasping**: Fully cross-domain physical manipulation evaluation

## Key Experimental Results

### Dataset Scale Comparison

| Dataset | Images | Categories | Wild | Robot | Ego | Sim | Reasoning |
|--------|--------|------|------|-------|-----|-----|------|
| UMD (2015) | 10k | 17 | - | ✓ | - | - | - |
| AGD20k (2020) | 20k | 50 | ✓ | - | - | - | - |
| HANDAL (2023) | 200k | 17 | ✓ | - | - | - | - |
| ManipVQA (2024) | 84k | - | - | ✓ | ✓ | - | ✓ |
| **RAGNet** | **273k** | **180** | **✓** | **✓** | **✓** | **✓** | **✓** |

RAGNet is the first large-scale affordance benchmark to simultaneously cover all four domains and support reasoning instructions.

### Annotation Tool Combinations (by Data Source)

| Source | Domain | Annotation Tools | Reasoning Instructions | Categories |
|--------|------|----------|-----------|--------|
| HANDAL | Wild | ❶ | 8.5k (hard) | 17 |
| Open-X | Robot | ❸❹❺ | - | 124 |
| GraspNet | Robot | ❶❺ | - | 32 |
| EgoObjects | Ego | ❷❹❺ | 17.4k | 74 |
| RLBench | Sim | ❺ | - | 10 |

### Key Findings

- Pretraining on large-scale multi-source data significantly improves open-world generalization of affordance prediction.
- Hard reasoning instructions (without category names), while increasing task difficulty, effectively enhance the model's functional reasoning capability.
- Cross-domain validation demonstrates that the model can generalize from wild data to robot scenarios and vice versa.
- Real-robot closed-loop experiments validate the feasibility of the complete pipeline from VLM affordance prediction to grasp execution.
- The adaptive five-tier annotation pipeline substantially reduces manual annotation cost while maintaining annotation quality.

## Highlights & Insights

1. **Benchmark contribution outweighs methodological contribution**: The RAGNet dataset itself (273k images, 180 categories, 26k reasoning instructions, four-domain coverage) constitutes critical missing infrastructure for the field and will drive subsequent research.
2. **Tiered reasoning instruction design**: The three-tier system (template-based → easy reasoning → hard reasoning) precisely corresponds to different requirement levels ranging from academic research to real-world deployment.
3. **Engineering value of the annotation pipeline**: The five-tier adaptive annotation strategy (raw masks → SAM2 → Florence2+SAM2 → VLPart+SAM2 → manual) constitutes a reusable large-scale annotation methodology.
4. **Closed-loop validation approach**: The complete chain from VLM affordance prediction to depth-based pose generation to real-robot execution bridges the full perception–planning–execution pipeline.
5. **Clear data motivation**: By systematically comparing existing datasets across three dimensions—domain coverage, category breadth, and reasoning capability—the paper constructs a convincing argument for the proposed dataset.

## Limitations & Future Work

- The AffordanceNet model architecture is relatively straightforward, primarily consisting of VLM fine-tuning concatenated with pose generation, offering limited methodological novelty.
- Reasoning instructions are generated solely by GPT-4, without validation of the diversity and complexity of real user instructions.
- Annotations for some data sources rely on automated tools (SAM2/Florence2/VLPart), and annotation quality has not been systematically evaluated quantitatively.
- Simulation data (RLBench) covers only 10 categories, which is modest compared to other data sources.
- Real-robot experiments are limited in scene and object variety, and do not comprehensively cover grasping across all 180 categories.

## Related Work & Insights

- **Affordance datasets**: UMD pioneered affordance segmentation with 10k RGB-D images; AGD20k extended coverage to 36 affordance categories from an exocentric perspective; HANDAL provides fine-grained handle annotations; AED and 3DOI introduce more diverse scenes; ManipVQA first incorporated reasoning but remained limited in data scale.
- **Affordance algorithms**: Progress from supervised learning to transfer learning and self-supervised learning has gradually addressed cross-domain generalization; recent VLM-based methods (AffordanceLLM, ManipVQA) introduce language reasoning but are constrained by data scale.
- **Robotic grasping**: From classical geometric methods to learning-based approaches, affordance prediction plays an increasingly important role in grasp planning; this paper is the first to unify the complete pipeline from VLM affordance prediction to real-world grasping within a single framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The large-scale reasoning-based affordance benchmark fills a critical gap in the field; the hard reasoning instruction evaluation setting (without category names) is creative.
- **Technical Depth**: ⭐⭐⭐ — The model component (AffordanceNet) is relatively straightforward; the primary contribution lies in data and benchmark construction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four validation protocols (zero-shot, cross-domain, reasoning, real-robot) provide comprehensive coverage; some experimental details require consulting supplementary materials.
- **Writing Quality**: ⭐⭐⭐⭐ — The description of data construction is systematic and complete; comparison tables are clear.
- **Recommendation**: ⭐⭐⭐⭐ — The dataset resource holds significant value for the affordance and robotic grasping communities.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](../../CVPR2026/segmentation/realvlg-r1_a_large-scale_real-world_visual-language_grounding_benchmark_for_robo.md)
- [\[AAAI 2026\] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors](../../AAAI2026/segmentation/towards_affordance-aware_robotic_dexterous_grasping_with_human-like_priors.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICCV 2025\] Online Reasoning Video Segmentation with Just-in-Time Digital Twins](online_reasoning_video_segmentation_with_just-in-time_digital_twins.md)

</div>

<!-- RELATED:END -->
