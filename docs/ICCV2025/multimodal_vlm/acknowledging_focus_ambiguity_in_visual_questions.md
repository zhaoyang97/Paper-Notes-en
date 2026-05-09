---
title: >-
  [Paper Note] Acknowledging Focus Ambiguity in Visual Questions
description: >-
  [ICCV 2025][Multimodal VLM][Visual Question Answering] This work is the first to formally define and systematically investigate **focus ambiguity** in visual question answering — the phenomenon arising when a linguistic expression in a question may plausibly refer to multiple regions in an image, a type of ambiguity entirely overlooked by existing VQA systems. The authors construct the VQ-FocusAmbiguity dataset (5,500 samples with 12,880 instance segmentation annotations) and demonstrate that modern models perform poorly at both recognizing and localizing focus ambiguity.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Visual Question Answering
  - Focus Ambiguity
  - Dataset
  - Visual Grounding
  - Multimodal Benchmark
date: 2026-05-08
content_hash: ea42210c475b3082
---

# Acknowledging Focus Ambiguity in Visual Questions

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)
**Code**: None (dataset available at [https://vizwiz.org/](https://vizwiz.org/))
**Area**: Multimodal VLM
**Keywords**: Visual Question Answering, Focus Ambiguity, Dataset, Visual Grounding, Multimodal Benchmark

## TL;DR

This work is the first to formally define and systematically investigate **focus ambiguity** in visual question answering — the phenomenon arising when a linguistic expression in a question may plausibly refer to multiple regions in an image, a type of ambiguity entirely overlooked by existing VQA systems. The authors construct the VQ-FocusAmbiguity dataset (5,500 samples with 12,880 instance segmentation annotations) and demonstrate that modern models perform poorly at both recognizing and localizing focus ambiguity.

## Background & Motivation

VQA research has addressed various sources of ambiguity — such as answer subjectivity and granularity differences — yet **no prior work** has considered the ambiguity arising from a question's linguistic expression pointing to multiple plausible spatial locations within an image.

**Why does this matter?** Consider a concrete example: a blind user photographs a scene and asks "What is the cleaning product?" If the image contains both dish soap and window cleaner, a VQA system should recognize the focus ambiguity rather than arbitrarily committing to a single answer. An incorrect response could have serious consequences — for instance, a blind user might accidentally use window cleaner to wash dishes.

**A critical distinction: question grounding vs. answer grounding.** The authors highlight a key insight: **the focus region of a question and the visual evidence for its answer can differ**. For example, when asked "What is above the mirror?", the question focuses on the mirror, whereas the answer evidence corresponds to the object above it. Analysis of 330 AnswerTherapy samples reveals that 79% of ambiguous questions and 36% of unambiguous questions exhibit a mismatch between question grounding and answer grounding, underscoring the necessity of treating question focus as an independent research object.

## Method

### Overall Architecture

The primary contributions of this paper are **dataset construction, task definition, and benchmarking**, rather than the proposal of a novel model. The work operates on three levels:

1. **VQ-FocusAmbiguity Dataset**: 5,500 visual questions, each accompanied by complete focus-region segmentation annotations.
2. **Two new tasks**: (a) determining whether a question exhibits focus ambiguity, and (b) localizing all plausible focus regions of a question.
3. **Benchmarking of modern models**: evaluating 4 foundation models on task (a) and 3 methods/pipelines on task (b).

### Key Designs

**Dataset Construction — Four Diverse Sources**:

Data are drawn from four distinct datasets, ensuring diversity in image content and question type:

| Source | Image Characteristics | Question Origin | Non-ambiguous Ratio |
|--------|----------------------|-----------------|---------------------|
| PACO (COCO 2017) | Complex scenes, multiple objects | Synthetic + annotators | 50% (2,272) |
| MSRA-B | Single foreground object | Synthetic ("What is this?" variants) | 100% (626) |
| AnswerTherapy-VQAv2 | COCO scenes | Manually created | 47% (82) |
| AnswerTherapy-VizWiz | Photographed by visually impaired users | Voice queries from blind users | 53% (83) |

**Why four sources?** To ensure diversity along the following dimensions: (1) single-object vs. complex scenes, (2) sighted vs. visually impaired photographers, (3) target objects of varying positions and sizes, and (4) typed questions vs. transcribed spoken queries.

**PACO Annotation Protocol** is carefully designed:
- The annotation interface presents the image along with all available segmentations.
- AI-generated candidate questions are first provided for selection (annotators may write from scratch, adopt directly, or modify).
- Annotators then select all segmentation regions that the question may plausibly refer to.
- For ambiguous questions, the proportion of annotator-authored questions is highest (55%), indicating that AI struggles to generate good ambiguous questions.

**Analysis of Ambiguity Sources**: Manual coding of 265 ambiguous samples identifies two primary causes:
- **Multiple instances of the same category** (61.5%): e.g., "What is next to the mirror?" when multiple mirrors are present.
- **Multiple instances of different categories** (31%): e.g., "What is this?" pointing to an indeterminate object, common in vague queries from visually impaired users.

**Dataset Splits** — supporting zero/few-shot learning: The training and validation sets each contain 70 samples (randomly sampled as 10 ambiguous + 10 non-ambiguous per source); the test set contains 5,360 samples. This reflects the trend wherein current state-of-the-art performance typically derives from zero/few-shot settings with foundation models.

### Loss & Training

This paper **does not train new models** but instead systematically evaluates existing models on two tasks:

**Task 1: Focus Ambiguity Recognition** (binary classification)
- Models: GPT-4o, InternVL2-76B, Qwen2.5-VL-7B, Molmo-7B
- 5 prompting strategies: ZS, ZS-CoT, ZS-ECoT (structured reasoning guidance), FS, FS-ECoT
- Evaluation metrics: Accuracy, Weighted F1, Positive Rate, Undecided Rate

**Task 2: Focus Region Localization** (instance segmentation)
- Pipelines: GLaMM (direct segmentation), GPT-4o+GLaMM (description followed by segmentation), Molmo+SAM (point localization followed by segmentation)
- Evaluation metrics: mAP, Union IoU, Max IoU

## Key Experimental Results

### Main Results

Focus ambiguity recognition results (all values in percentages):

| Model | Best Prompt | Accuracy | F1 | Positive Rate |
|-------|------------|----------|----|--------------|
| GPT-4o (>200B) | ZS-CoT | 69.6 | 69.8 | 53.3 |
| InternVL2 (76B) | ZS-CoT | 56.7 | 54.8 | 27.9 |
| Qwen2.5-VL (7B) | ZS-ECoT | 65.5 | 65.3 | 59.0 |
| Molmo (7B) | ZS-CoT | 56.9 | 57.1 | 48.1 |

Focus region localization results:

| Method | Best Prompt | mAP | Union IoU | Max IoU |
|--------|------------|-----|-----------|---------|
| GLaMM | ZS | 13.01 | 41.90 | 43.69 |
| GPT-4o+GLaMM | FS | 14.24 | 40.97 | 47.83 |
| Molmo+SAM | ZS-CoT | 24.3 | 36.2 | 45.4 |

### Ablation Study

Analysis of the relationship between question characteristics and ambiguity:

| Feature | Ambiguous Questions | Non-ambiguous Questions | Notes |
|---------|--------------------|-----------------------|-------|
| Average word count | Fewer | More | More words provide more disambiguating context |
| Proportion containing plural nouns | 4.7% | 23.8% | Plural forms naturally reduce ambiguity |
| Median number of segmentation regions | 3 | 1 (by definition) | Ambiguous questions average 4 focus regions |
| Question–answer grounding match | 21% matched | 64% matched | Ambiguous questions require independent question grounding |

Improvement from CoT prompting strategies:
- Molmo-7B: ZS-CoT outperforms ZS by **18.4 pp**
- Qwen2.5-VL: ZS-ECoT outperforms ZS by **8.3 pp**
- 7B models with CoT can match or surpass the 76B InternVL2

### Key Findings

1. **All models perform poorly**: The best accuracy is only 69.6% (GPT-4o), demonstrating that focus ambiguity remains an unsolved challenge.
2. **Training data matters**: Qwen2.5-VL and Molmo are trained on region-level counting and pointing tasks in the PixMo dataset, which are naturally relevant to ambiguity recognition (counting one region = unambiguous; multiple regions = ambiguous).
3. **InternVL2 (76B) performs worst despite its scale**: It lacks region-level training data and consistently biases toward predicting "non-ambiguous" (lowest Positive Rate of 27.9%).
4. **Molmo+SAM achieves the highest mAP in localization**: Because Molmo tends to point to multiple regions; however, its Union IoU is lower because SAM's point-prompted segmentation is insufficiently complete.
5. **Part-level segmentation is extremely difficult**: Models perform far worse at localizing object parts than complete objects in PACO data.

## Highlights & Insights

1. **First systematic study of questions themselves as a source of ambiguity**: This work extends VQA ambiguity research from "answer ambiguity" to "question focus ambiguity," representing an important conceptual distinction.
2. **High practical value**: The work directly addresses real-world use cases of visually impaired users; an ambiguity-aware VQA system could interactively guide users toward disambiguation.
3. **Sophisticated dataset design**: Four sources cover diverse scenarios with a near-balanced ambiguous/non-ambiguous distribution and support zero/few-shot evaluation paradigms.
4. **In-depth analysis of CoT**: Beyond demonstrating CoT effectiveness, the work reveals the complementary importance of training data and prompting strategies.

## Limitations & Future Work

1. **Limited dataset scale**: 5,500 samples may be insufficient for training domain-specific models, particularly given that the training set contains only 70 samples.
2. **English-only questions**: Focus ambiguity in cross-lingual settings may exhibit different characteristics.
3. **No solution model proposed**: The work only establishes benchmarks without designing a dedicated ambiguity-aware VQA model.
4. **Cascading errors in two-stage pipelines**: The GPT-4o+GLaMM and Molmo+SAM two-stage approaches suffer from error propagation.
5. **Extension to video and multimodal settings**: Focus ambiguity is equally present in video QA and multimodal dialogue, warranting further exploration.

## Related Work & Insights

- **VQA Therapy (ICCV 2023)**: Focuses on answer grounding; the present work extends this by introducing question grounding as a new dimension.
- **Molmo/PixMo**: Demonstrates that region-level training benefits ambiguity recognition, suggesting targeted training as a promising direction.
- **GLaMM**: Currently the best language-guided segmentation model, yet still generates only a single mask and cannot handle multi-focus scenarios.
- **SAM**: Point-prompted segmentation is powerful but incomplete, necessitating better prompt generation strategies.
- Inspiration: The work motivates building VQA systems oriented toward "understanding user intent" rather than "predicting a single answer."

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Iris: Breaking GUI Complexity with Adaptive Focus and Self-Refining](iris_breaking_gui_complexity_with_adaptive_focus_and_self-refining.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](../../NeurIPS2025/multimodal_vlm/focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ICCV 2025\] Mastering Collaborative Multi-modal Data Selection: A Focus on Informativeness, Uniqueness, and Representativeness](mastering_collaborative_multi-modal_data_selection_a_focus_on_informativeness_un.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)

<!-- RELATED:END -->
