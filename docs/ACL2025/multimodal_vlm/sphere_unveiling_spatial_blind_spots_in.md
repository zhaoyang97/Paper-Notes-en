---
title: >-
  [Paper Note] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation
description: >-
  [ACL 2025][Multimodal VLM][spatial reasoning] Introduces SPHERE, a three-tier hierarchical spatial reasoning evaluation framework (single-skill $\rightarrow$ multi-skill $\rightarrow$ reasoning). Based on 2,285 human-annotated QA pairs on MS COCO, it reveals a 25% performance gap between GPT-4o (67.9%) and humans (93.0%), with severe deficiencies particularly in distance judgment, perspective switching, and physical reasoning.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "spatial reasoning"
  - "VLM benchmark"
  - "hierarchical evaluation"
  - "egocentric/allocentric"
  - "physical reasoning"
date: 2026-05-08
content_hash: 8c38d02a18aa6b2c
---

# SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation

**Conference**: ACL 2025  
**arXiv**: [2412.12693](https://arxiv.org/abs/2412.12693)  
**Code**: Available  
**Area**: Multimodal VLMs  
**Keywords**: spatial reasoning, VLM benchmark, hierarchical evaluation, egocentric/allocentric, physical reasoning

## TL;DR
Introduces SPHERE, a three-tier hierarchical spatial reasoning evaluation framework (single-skill $\rightarrow$ multi-skill $\rightarrow$ reasoning). Based on 2,285 human-annotated QA pairs on MS COCO, it reveals a 25% performance gap between GPT-4o (67.9%) and humans (93.0%), with severe deficiencies particularly in distance judgment, perspective switching, and physical reasoning.

## Background & Motivation
**Background**: While VLMs demonstrate some capabilities in basic spatial directions (left/right/front/back), embodied AI and robotics require **multi-dimensional spatial reasoning**, including distance, proximity, egocentric/allocentric perspective switching, and physical constraint reasoning.

**Limitations of Prior Work**: Existing spatial benchmarks (such as EmbSpatial-Bench, VSR, and SpatialBench) only test isolated, simple spatial cues. They cannot decouple the intersecting effects of multiple spatial skills, and do not cover scenarios requiring reasoning, such as occlusion or manipulation.

**Key Challenge**: Models may perform acceptably on single spatial skills, but their performance drops drastically when combining multiple skills or performing reasoning. A hierarchical evaluation is needed to precisely pinpoint these bottlenecks.

**Goal**: To construct a systematic, hierarchical spatial reasoning benchmark spanning from single-skill to multi-skill and reasoning tasks, precisely diagnosing the spatial blind spots of VLMs.

**Key Insight**: The hierarchical model of spatial ability in cognitive science, which transitions from basic perception $\rightarrow$ skill combination $\rightarrow$ high-level reasoning.

**Core Idea**: A three-tier progressive evaluation using fine-grained annotations on real images to reveal systematic deficiencies of VLMs in distance estimation, perspective taking, and physical reasoning.

## Method

### Overall Architecture
Three-tier evaluation: Level 1 (4 types of single-skill) $\rightarrow$ Level 2 (3 types of multi-skill composition) $\rightarrow$ Level 3 (2 types of reasoning), totaling 2,285 QA pairs. All questions are manually annotated based on images from the MS COCO-2017 test set, with cross-verification by at least two authors.

### Key Designs

1. **Level 1 - Single-Skill Tasks (4 categories)**:

    - **Position** (172 egocentric + 185 allocentric): Judging the relative positions of objects, distinguishing between egocentric (relative to an entity) and allocentric (relative to the camera) perspectives.
    - **Counting** (201 questions): Includes trick questions to test robustness.
    - **Distance** (202 questions): Judging relative distance (near vs. far).
    - **Size** (198 questions): Judging relative size.

2. **Level 2 - Multi-Skill Composition Tasks (3 categories)**:

    - Position + Counting (169 questions): Counting objects at specific positions.
    - Distance + Counting (158 questions): Counting objects within specific distance ranges.
    - Distance + Size (199 questions): Requires understanding of **size constancy**—distant objects might appear smaller in the image, but are actually larger in reality.
    - **Design Motivation**: To test whether models can combine multiple spatial perceptions, rather than being limited to a single one.

3. **Level 3 - Reasoning Tasks (2 categories)**:

    - **Occlusion Reasoning** (202 intermediate + 200 final): Inferring the existence and attributes of occluded objects.
    - **Manipulation Reasoning** (199 intermediate + 200 final): Inferring the feasibility of object movement under physical constraints.
    - Each category includes intermediate comprehension questions + final reasoning questions, distinguishing whether the failure lies in "perception" or "reasoning".

### Loss & Training
A pure evaluation benchmark with no training involved.

## Key Experimental Results

### Main Results

| Level | Human | GPT-4o | Gemini 2.0 Flash | Qwen2.5-VL | Random |
|------|------|--------|-------------------|------------|------|
| Single-Skill | 95.4% | 77.3% | **78.2%** | 76.0% | 50.0% |
| Multi-Skill | 92.5% | **58.6%** | - | 57.9% | 44.3% |
| Reasoning | 89.0% | **64.7%** | - | - | 50.0% |
| **Overall** | **93.0%** | **67.9%** | - | - | 49.1% |

### Perspective Bias Analysis

| Model | Allocentric (%) | Egocentric (%) | Gap |
|------|----------------|----------------|------|
| Phi-3.5-Vision | 77.9 | 44.5 | **33.4%** |
| LLaVA-OneVision | 73.7 | 45.7 | 28.0% |
| GPT-4o | ~High | ~Low | Significant |

### Key Findings
- Performance drops drastically by 25-30% from single-skill to multi-skill tasks, indicating that combining spatial skills is extremely challenging.
- On the **Distance + Size** task, most models score below 50% (worse than random guessing), showing a lack of size constancy understanding.
- Egocentric reasoning is severely deficient: the gap for some models reaches up to 33%.
- Spatially specialized models (e.g., SpatialBot, SpaceMantis) surprisingly perform worse than the general-purpose model LLaVA-OneVision.
- Providing ground-truth intermediate answers improves reasoning performance by up to 21.9% (Qwen2-VL 72B), indicating that perceptual bottlenecks are the primary issue.
- The accuracy of intermediate reasoning questions (perception) is sometimes lower than that of the final questions (reasoning), suggesting that models may rely on shortcuts rather than genuine spatial understanding.

## Highlights & Insights
- **Hierarchical evaluation design** precisely pinpoints the bottleneck of "acceptable in single-skill $\rightarrow$ fails in composition," providing clear guidance for future improvements.
- **Egocentric vs. allocentric perspective analysis** reveals a deep issue: models primarily learn "directions relative to the camera" rather than "spatial relations between objects", which is a fatal flaw for embodied AI applications.
- The design of the **Distance + Size composition task** is ingenious: it requires size constancy (not being deceived by the visual size in the image), which is a core capability of 3D understanding.

## Limitations & Future Work
- The dataset size is relatively small (2,285 QA pairs), and human annotation is difficult to scale.
- It only uses static images and does not involve dynamic spatial reasoning (video scenarios).
- Being based on MS COCO images, the scene diversity is limited.
- No training data or improvement methods are provided; it only serves as a diagnostic tool.

## Related Work & Insights
- **vs EmbSpatial-Bench / VSR**: These only test basic orientations, whereas SPHERE introduces dimensions of distance, size, and reasoning.
- **vs SpatialBench**: The latter does not analyze skill composition, whereas SPHERE's hierarchical design can distinguish between "perception failure" and "reasoning failure."
- Implications for the embodied AI community: current VLMs are far from achieving reliable spatial reasoning, posing high risks if directly deployed for robot navigation and manipulation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The hierarchical spatial evaluation framework is novel, and the perspective analysis is highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 15+ models $\times$ 9 task types + human baseline + detailed analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear hierarchical structure and in-depth analysis.
- **Value**: ⭐⭐⭐⭐ Holds significant guiding importance for spatial reasoning research in VLMs.

## Highlights & Insights
- **Hierarchical decoupling** precisely pinpoints weaknesses—it is not a general "poor spatial capability," but rather deficiencies in "distance estimation, perspective taking, and physical reasoning."
- **Physical reasoning subtasks (occlusion + manipulation)** are the most discriminative—humans drop by only 6%, whereas models drop by over 13%.
- **Real image annotations (COCO)** better reflect practical requirements compared to synthetic data.

## Limitations & Future Work
- The dataset size is relatively small. Only static images are tested. Physical reasoning answers may carry subjectivity.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Hierarchical spatial evaluation and physical reasoning tasks are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple SOTA VLMs + human baseline + skill decomposition.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework diagrams and meticulous analysis of results.
- **Value**: ⭐⭐⭐⭐ Provides a targeted benchmark for spatial reasoning in embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[ACL 2025\] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](jailbreak_large_vision-language_models_through_multi-modal_linkage.md)
- [\[ACL 2025\] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](../../CVPR2026/multimodal_vlm/hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[ACL 2025\] Can Vision-Language Models Evaluate Handwritten Math?](can_vision-language_models_evaluate_handwritten_math.md)

</div>

<!-- RELATED:END -->
