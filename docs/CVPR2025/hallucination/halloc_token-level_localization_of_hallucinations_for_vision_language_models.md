---
title: >-
  [Paper Note] HalLoc: Token-Level Localization of Hallucinations for Vision Language Models
description: >-
  [CVPR 2025][Hallucination Detection][Token-Level Localization] This work proposes HalLoc, a token-level hallucination annotation dataset with 155K samples covering three categories of tasks: VQA, instruction following, and image captioning. Based on this, a lightweight hallucination detection model named HalLocalizer is trained, which can be integrated into existing VLMs in a plug-and-play manner to achieve real-time probabilistic hallucination detection without sacrificing e…
tags:
  - "CVPR 2025"
  - "Hallucination Detection"
  - "Token-Level Localization"
  - "Probabilistic Detection"
  - "Plug-and-Play"
  - "VLM Reliability"
date: 2026-05-08
content_hash: 5f8f4c374bbfc453
---

# HalLoc: Token-Level Localization of Hallucinations for Vision Language Models

**Conference**: CVPR 2025  
**arXiv**: [2506.10286](https://arxiv.org/abs/2506.10286)  
**Code**: [GitHub](https://github.com/dbsltm/cvpr25_halloc)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Detection, Token-Level Localization, Probabilistic Detection, Plug-and-Play, VLM Reliability

## TL;DR

This work proposes HalLoc, a token-level hallucination annotation dataset with 155K samples covering three categories of tasks: VQA, instruction following, and image captioning. Based on this, a lightweight hallucination detection model named HalLocalizer is trained, which can be integrated into existing VLMs in a plug-and-play manner to achieve real-time probabilistic hallucination detection without sacrificing efficiency.

## Background & Motivation

**Background**:
Although Vision-Language Models (VLMs) have demonstrated excellent performance in various multimodal tasks, they are prone to generating descriptions inconsistent with visual content (i.e., hallucinations), including object, attribute, relation, and scene hallucinations.

**Limitations of Prior Work**:
1. Existing hallucination detection methods (e.g., GAVIE, FAITHScore, UNIHD) rely on large models like GPT-4 for cross-validation, resulting in high computational overhead and latency.
2. Current methods only provide binary (hallucinated/fact) decisions, failing to address real-world scenarios where the boundary between hallucinated and correct information is ambiguous.
3. Existing datasets (e.g., HaELM 56K, MHalDetect 16K) have coarse granularity (sentence- or paragraph-level) and lack hallucination category annotations.
4. Although MHaluBench offers category annotations, it contains only 400 samples, which is too small for model training.

**Key Challenge**:
Practical applications require efficient, real-time, fine-grained, and probabilistic hallucination detection. However, existing methods either impose prohibitive computational costs, operate at a coarse granularity, or suffer from insufficient data scale.

**Goal**:
Construct a large-scale, token-grained hallucination dataset with category annotations, and train a lightweight, plug-and-play hallucination detection module based on it.

**Key Insight**:
Design an HQA (Hallucinated Question-Answer) Injection pipeline to inject hallucinations into source text in a controllable manner, achieving large-scale, token-level hallucination annotation.

**Core Idea**:
First construct a large-scale hallucinated question-answer database, and then systematically inject these hallucinated answers into source texts across different tasks to obtain a multi-task hallucinated dataset with token-level annotations.

## Method

### Overall Architecture

The construction of HalLoc consists of two stages: (1) Hallucination Generation: constructing a large-scale hallucinated question-answering database (HQA Database, 160K entries) based on conceptual association bias and statistical bias; (2) Hallucination Injection: systematically injecting hallucinated answers from the HQA database into source texts of VQA, instruction following, and image captioning tasks, forming three subsets. HalLocalizer, trained on this dataset, uses a VisualBERT encoder combined with four linear classification heads to achieve token-level detection of four hallucination categories.

### Key Designs

1. **Four-Category Hallucination Taxonomy**:
    - **Function**: Provide fine-grained hallucination category classification.
    - **Mechanism**:
        - **Object Hallucination** `<obj>`: Referring to non-existent objects.
        - **Attribute Hallucination** `<attr><obj>`: Incorrect attributes (color, position, action, etc.) associated with a single object.
        - **Relation Hallucination** `<obj1><rel><obj2>`: Incorrect interaction relationships between two objects.
        - **Scene Hallucination** `<sce>`: Incorrect scene descriptions (weather, location, etc.).
    - **Design Motivation**: Different categories of hallucinations have distinct causes and detection difficulties; fine-grained classification enables targeted handling.

2. **HQA-Injection Pipeline**:
    - **Function**: Generate large-scale, token-level hallucination annotation data in a controllable and scalable manner.
    - **Mechanism**:
        - Select questions from the GQA dataset and classify them into categories of object, attribute, relation, and scene.
        - Construct hallucinated answers via conceptual association bias (borrowing attributes of other objects in the image) and statistical bias (incorrect attributes with high co-occurrence frequency).
        - Employ GPT-4 to inject the hallucinated answers into appropriate positions in the source text, replacing correct answers or inserting new sentences.
        - Apply auxiliary VLM validation (triple verification using InternVL, LLaVA, and InstructBLIP).
    - **Design Motivation**: It is difficult to control the category and position of hallucinations by directly prompting VLMs to generate hallucinated texts; the HQA-Injection pipeline achieves precise control.

3. **HalLocalizer Model Architecture**:
    - **Function**: Lightweight token-level hallucination detection.
    - **Mechanism**: Utilize a bidirectional VisualBERT encoder to process either the final hidden states of the VLM or directly process the textual responses, followed by four independent linear classification heads to predict the probability for each of the four hallucination categories.
    - **Design Motivation**: VisualBERT has a small parameter size and fast inference speed; four independent classification heads support concurrent multi-type detection.

### Loss & Training

- Standard binary cross entropy loss, with the four classification heads trained independently.
- AdamW optimizer with a learning rate of $1 \times 10^{-6}$ and cosine annealing scheduler.
- Trained for 25 epochs on 4×A6000 GPUs, taking approximately 10 hours.
- Sequence length of 512, batch size of 64.
- Thresholds for each classification head are independently tuned on the validation set.

## Key Experimental Results

### Main Results

**VQA Subset (Token-Level Hallucination Detection F1)**

| Method | Object F1 | Attribute F1 | Relation F1 | Scene F1 |
|------|---------|---------|---------|---------|
| CHAIR | 0.19 | - | - | - |
| Always 1 | 0.44 | 0.44 | 0.44 | 0.12 |
| HalLocalizer (InternVL) | **0.71** | **0.94** | **0.71** | **0.93** |

**Instruct Subset**

| Method | Object F1 | Attribute F1 | Relation F1 | Scene F1 |
|------|---------|---------|---------|---------|
| HalLocalizer (w/o Embed.) | **0.82** | **0.97** | 0.83 | **0.94** |
| HalLocalizer (InternVL) | 0.79 | 0.95 | **0.84** | 0.94 |

**Caption Subset**

| Method | Object F1 | Attribute F1 | Relation F1 | Scene F1 |
|------|---------|---------|---------|---------|
| HalLocalizer (w/o Embed.) | **0.68** | **0.64** | **0.71** | **0.76** |
| HalLocalizer (InternVL) | 0.58 | 0.37 | 0.46 | 0.25 |

### Ablation Study

**HalLocalizer vs. Token Log Probability (Overall Hallucination Detection F1)**

| Subset | HalLocalizer | LogProb (LLaVA) |
|------|-------------|-----------------|
| VQA | 0.95 | 0.95 |
| Instruct | **0.91** | 0.47 |
| Caption | **0.71** | 0.17 |

### Key Findings

1. **VLM embeddings do not always help**: On the Instruct and Caption subsets, the text-only mode without using VLM embeddings yields better performance, especially on the Caption subset where the difference is significant (F1: 0.68 vs 0.58).
2. **Log probabilities are far inferior to dedicated detectors**: In longer responses (Instruct/Caption), the hallucination detection accuracy of token log probability drops sharply (Caption F1 is only 0.17 vs. 0.71 for HalLocalizer).
3. **Attribute and scene hallucinations are easier to detect**: In the VQA and Instruct subsets, the F1 scores for attribute/scene exceed 0.90, while those for object and relation are around 0.70, indicating that the latter are more challenging.
4. **Caption subset is the most difficult**: All methods perform worst on Caption because of the long paragraph length (average 57.53 words) and low hallucination frequency (only 5%).
5. **Human evaluation validates high quality**: Hallucination generation accuracy is 91%, injection accuracy is 98%, and human annotators achieved perfect agreement.

## Highlights & Insights

1. **First large-scale token-level hallucination annotation dataset**: 155K samples covering three core VLM tasks (VQA, instruction following, image captioning) with token-level four-category annotations, far exceeding existing datasets.
2. **Probabilistic detection paradigm**: Unlike binary decisions of existing methods, HalLocalizer outputs hallucination probabilities, which is more suitable for real-world scenarios with ambiguous boundaries.
3. **Plug-and-play design**: HalLocalizer is integrated into VLMs as an auxiliary module without affecting the generation capability of the backbone model, enabling real-time concurrent hallucination detection.
4. **Ingenious design of the HQA-Injection pipeline**: Decomposing hallucinated data construction into two steps—constructing hallucinated answers and injecting them into the source text—achieves large-scale annotation with controllable categories and positions.

## Limitations & Future Work

1. The HQA database is constructed based on the GQA dataset, limiting the coverage of scene and object categories.
2. Scene hallucination samples are relatively scarce (only 4,084 entries) due to the limited number of images with explicit environmental settings.
3. The detection performance on the Caption subset still has significant room for improvement (best F1 scores are only 0.68/0.64/0.71/0.76).
4. The feedback mechanism of HalLocalizer on downstream VLM generation quality (such as rejection sampling or regeneration) remains unexplored.
5. Validation is limited to English data, leaving multilingual hallucination detection for future exploration.

## Related Work & Insights

- **CHAIR**: An early object hallucination detection method based on rule matching, which is coarse-grained and only supports the object category.
- **UNIHD**: Utilizes a chain of large language models for paragraph-level hallucination classification, which is computationally expensive.
- **MHalDetect**: Trains a dedicated detection model but operates at a sentence-level granularity with only 16K samples.
- Insight: The bottleneck of hallucination detection lies in training data rather than model architecture. The concept of HQA injection can be generalized to other NLP data augmentation scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] One Token, Two Fates: A Unified Framework via Vision Token Manipulation Against MLLMs Hallucination](one_token_two_fates_a_unified_framework_via_vision_token_manipulation_against_ml.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](../../ICLR2026/hallucination/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[CVPR 2025\] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)
- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](../../ICCV2025/hallucination/mitigating_object_hallucinations_via_sentence-level_early_intervention.md)

</div>

<!-- RELATED:END -->
