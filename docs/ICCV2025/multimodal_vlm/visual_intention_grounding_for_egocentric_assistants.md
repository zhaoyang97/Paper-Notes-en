---
title: >-
  [Paper Note] Visual Intention Grounding for Egocentric Assistants
description: >-
  [ICCV 2025][Multimodal VLM][Visual Intention Grounding] Proposes the first task and dataset **EgoIntention** (26K images + 52K intention descriptions + 89K bounding boxes) for **egocentric visual intention grounding**, revealing significant deficiencies of existing MLLMs in implicit intention reasoning and first-person visual grounding. It further introduces the **Reason-to-Ground (RoG)** instruction tuning method, which significantly boosts performance by decoupling intentio…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Visual Intention Grounding"
  - "First-Person Perspective"
  - "Egocentric Vision"
  - "Object Functionality Reasoning"
  - "Instruction Tuning"
date: 2026-05-08
content_hash: 5fe6eb3d66ceb048
---

# Visual Intention Grounding for Egocentric Assistants

**Conference**: ICCV 2025  
**arXiv**: [2504.13621](https://arxiv.org/abs/2504.13621)  
**Code**: To be released  
**Area**: Multimodal VLMs  
**Keywords**: Visual Intention Grounding, First-Person Perspective, Egocentric Vision, Object Functionality Reasoning, Instruction Tuning

## TL;DR

Proposes the first task and dataset **EgoIntention** (26K images + 52K intention descriptions + 89K bounding boxes) for **egocentric visual intention grounding**, revealing significant deficiencies of existing MLLMs in implicit intention reasoning and first-person visual grounding. It further introduces the **Reason-to-Ground (RoG)** instruction tuning method, which significantly boosts performance by decoupling intention reasoning from object grounding.

## Background & Motivation

### Core Scenario

Imagine a wearable AI assistant: a user is looking for a place to sit and organize tools in a cluttered workshop, or a child wants to reach the kitchen sink. The assistant needs to locate the appropriate object (e.g., a chair) based on the contextual intention **without the user explicitly stating the object name**.

### Traditional Visual Grounding vs. Intention Grounding

| Dimension | Traditional Visual Grounding | Visual Intention Grounding |
|------|------------|------------|
| Perspective | Third-person | First-person (Egocentric) |
| Query | Explicit object description ("white chair") | Implicit intention ("pack up phone and luggage") |
| Reasoning | Direct word-to-object matching | Requires reasoning about object functionality/utility |
| Challenges | Object recognition | Occlusion, dynamic perspective + functional reasoning |

### Key Challenges

**Misleading explicit mentions**: Intention sentences may explicitly mention certain objects that are not the target—for example, "pack up my phone and luggage" mentions "phone", but the actual target is the **handbag** (used to hold those items).

**Unconventional object utility**: A chair is typically used for sitting, but in the context of "reaching the sink", it is used as a **step stool**—requiring an understanding of the object's **affordance**.

**First-person perspective difficulties**: Motion blur, small objects, perspective distortion, etc.

### Limitations of Prior Work

**Two-stage solutions** (GPT-4 reasoning + GroundingDINO detection): The two models operate in different feature spaces, leading to inconsistent modality alignment and potential hallucinations of non-existent objects.

**MLLMs**: Primarily designed for third-person visual grounding, lacking training data to bridge egocentric vision with intention sentences.

## Method

### Overall Architecture: EgoIntention Dataset

#### Data Sources

- Images are sourced from **Ego4D** (the largest real-world egocentric vision dataset).
- Bounding box annotations are inherited from **PACO-Ego4D** (an object part and attribute dataset).
- Building on this, meticulously curated human intention descriptions are added.

#### Three-Stage Data Construction Pipeline

**Stage 1: Intention Sentence Generation** (GPT-4)
- Two types of intention descriptions:
    - **Context-aware intentions**: Align with environmental expectations—e.g., "I noticed the desk leg is shaking and needs repair" $\rightarrow$ hammer. Pass rate: 97.2%.
    - **Uncommon intentions**: Atypical uses—e.g., "using a backpack temporarily to shield from rain". Pass rate: 74.1%.

**Stage 2: Human Verification** (Amazon MTurk)
- Evaluates semantic validity and real-world applicability.
- Complemented by a GPT-4 validator (92% agreement with human judgments).

**Stage 3: Alternative Object Annotation**
- Addressing the subjectivity of intentions, multiple objects that could satisfy the intention are annotated (e.g., "desk decoration" $\rightarrow$ flowerpot/bottle/cup are all acceptable).
- Complementary bounding boxes are added, also requiring double verification.

#### Dataset Statistics

| Split | Image Count | Context Bounding Boxes | Uncommon Bounding Boxes |
|------|-------|------------|------------|
| Train | 15,667 | 25,772 | 25,933 |
| Val | 825 | 1,402 | 1,366 |
| Test | 9,892 | 17,699 | 17,669 |
| **Total** | **26,384** | **44,873** | **44,968** |

### Key Designs: Reason-to-Ground (RoG) Instruction Tuning

#### Design Motivation

Experimental observations reveal:
- **Reasoning-then-Detection (R-D) significantly outperforms Detection-then-Reasoning (D-R)**: R-D achieves a P@0.5 of 46.6% vs. 21.1% for D-R (a 25% gap).
- Reason: Using GPT-4 first to narrow down the target to 1-2 categories allows GroundingDINO to detect it more accurately.
- However, the inconsistent feature space in two-stage solutions remains a bottleneck.

#### RoG Two-Stage Decoupling

Traditional methods directly input the intention sentence + `<ref>` token $\rightarrow$ output bounding box.

RoG decomposes the task into:
1. **Intention Reasoning**: `<reason>` token + implicit intention sentence $\rightarrow$ model outputs the **target object category**.
2. **Object Grounding**: `<ref>` token + explicit object description from the first stage $\rightarrow$ model outputs the **bounding box**.

Core Advantage: Prevents the model from directly mapping **explicitly mentioned but non-target objects** in the intention sentence to the bounding box.

#### Loss & Training

- Mixed training data: RefCOCO/+/g (traditional REC) + EgoIntention (intention grounding).
- Parameter-efficient fine-tuning using LoRA.
- Model-agnostic: Applicable to MiniGPTv2, Qwen-VL, etc.

## Key Experimental Results

### Main Results: Zero-Shot Evaluation (Table 3)

| Method | Context P@0.5 | Uncommon P@0.5 | Overall P@0.5 |
|------|-------------|-------------|-----------|
| D-R (GroundingDINO $\rightarrow$ GPT4) | 21.1 | 14.6 | 17.8 |
| R-D (GPT4 $\rightarrow$ GroundingDINO) | 46.6 | 23.6 | 35.1 |
| CogVLM-grounding | 3.4 | 2.4 | 2.9 |
| Groma | 4.8 | 4.3 | 4.5 |
| MiniGPT-v2 | 18.8 | 15.7 | 17.2 |
| Qwen-VL | 26.3 | 12.6 | 19.4 |

**Key Findings**: Grounding-specific MLLMs (CogVLM, Groma) fail almost completely—lacking intention reasoning capabilities.

### RoG Fine-Tuning Results (Table 4)

| Model | Method | RefCOCO val | RefCOCO+ val | EgoIntention Context | EgoIntention Uncommon | Overall |
|------|------|------------|-------------|-------------------|-------------------|------|
| MiniGPTv2 | Zero-shot | 87.37 | 79.00 | 18.73 | 15.72 | 17.22 |
| MiniGPTv2 | Naive SFT | 86.60 | 78.98 | 41.31 | 36.92 | 39.11 |
| MiniGPTv2 | **RoG SFT** | **87.83** | **79.76** | **45.06** | **40.21** | **42.64** |
| Qwen-VL | Zero-shot | 89.32 | 83.18 | 26.30 | 12.60 | 19.45 |
| Qwen-VL | **RoG SFT** | 89.26 | 83.29 | **38.25** | **31.56** | **34.91** |

**Core Results**: Compared to Naive SFT, RoG improves by 3.5 P@0.5 on EgoIntention and slightly enhances performance on the RefCOCO series, demonstrating that it does not damage traditional grounding capabilities.

### Ablation Study: Training Data Combinations (Table 5 & 6)

| Training Data | Method | RefCOCO val | EgoIntention Context | EgoIntention Uncommon |
|---------|------|------------|-------------------|-------------------|
| EgoInt. Only | Naive SFT | 66.53 (Catastrophic forgetting!) | 38.26 | 35.53 |
| RefCOCO/+/g + EgoInt. | Naive SFT | 87.48 | 41.07 | 36.91 |
| All Data | Naive SFT | 86.60 | 41.31 | 36.92 |
| All Data | **RoG SFT** | **87.83** | **45.06** | **40.21** |

**Key Findings**:
1. Fine-tuning solely on EgoInt. causes a performance plunge of over 20% on RefCOCO—a typical case of catastrophic forgetting.
2. Mixing RefCOCO data maintains traditional REC capabilities while enhancing intention grounding.
3. The RoG decoupling strategy outperforms Naive SFT across all training data combinations.
4. RoG also improves EgoIntention performance on explicit object queries (44.26% $\rightarrow$ 47.50%).

## Highlights & Insights

1. **Pioneering Task Definition**: Systematically defines and studies egocentric visual intention grounding for the first time, clearly distinguishing traditional object grounding from intention-driven grounding.
2. **Meticulously Designed Dataset**: Dual-type intentions (context-aware + uncommon) + alternative object annotations + three-stage validation completely capture the ambiguity of intentions.
3. **Simplicity and Effectiveness of RoG**: By simply introducing a `<reason>` token and a two-stage decoupling process, it significantly boosts intention grounding while maintaining or even improving traditional grounding capabilities.
4. **Diagnostic Insights**: Reasoning-then-detection vastly outperforms detection-then-reasoning, indicating that **narrowing down the search space** is key to visual intention understanding.

## Limitations & Future Work

1. **Dataset Scale**: 26K images is relatively small by deep learning standards, which may limit the learning of complex intention reasoning.
2. **GPT-4 Dependence**: Intention generation and verification rely on GPT-4, introducing bias and cost.
3. **Single Modality Input**: Only covers static images, whereas real-world egocentric scenes are naturally video streams.
4. **Unexplored Multi-turn Interactions**: A real-world AI assistant might require dialogues with the user to clarify intentions.
5. **Discrepancy between MiniGPTv2 and Qwen-VL**: RoG shows weaker gains on Qwen-VL, potentially due to its inferior instruction-following capability.

## Related Work & Insights

- **Difference from IntentionVG**: The first-person images in IntentionVG have a fixed perspective aligned with the objects, whereas EgoIntention utilizes real-world egocentric video frames from Ego4D (featuring motion blur, small objects, and distortions).
- **Difference from COCO-Tasks**: COCO-Tasks uses phrase-level queries, whereas EgoIntention employs free-form natural language and multi-intent annotations.
- **Insights**: The "reason-then-act" paradigm of RoG can be generalized to other visual tasks requiring implicit reasoning (e.g., understanding "find a safe place to park" in navigation).

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐⭐ — New task + new dataset + new method, establishing a highly complete framework.  
**Value**: ⭐⭐⭐⭐ — Directly addresses wearable AI assistant scenarios.  
**Experimental Thoroughness**: ⭐⭐⭐⭐ — Full coverage with zero-shot, fine-tuning, and ablation; evaluated on 6 models across 8 datasets.  
**Writing Quality**: ⭐⭐⭐⭐ — Vivid scenario introduction and clear methodological motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)
- [\[ICCV 2025\] ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition](viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[CVPR 2026\] HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding](../../CVPR2026/multimodal_vlm/hammer_harnessing_mllm_via_cross-modal_integration_for_intention-driven_3d_affor.md)

</div>

<!-- RELATED:END -->
