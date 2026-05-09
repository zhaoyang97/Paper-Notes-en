---
title: >-
  [Paper Note] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading
description: >-
  [ICLR2026][Video Understanding][eye tracking] This paper introduces a novel task of decoding open-ended information seeking goals from eye movement trajectories during reading. Built upon the OneStop eye-tracking dataset (360 participants, 486 questions, 162 passages), the authors develop both discriminative and generative multimodal models. RoBERTEye-Fixations achieves 49.3% accuracy on three-way goal selection (random baseline: 33%) and 70.9% on different-critical-span conditions; DalEye-Llama/GPT also significantly outperforms eye-movement-free baselines on goal reconstruction.
tags:
  - ICLR2026
  - Video Understanding
  - eye tracking
  - reading comprehension
  - information seeking goal decoding
  - multimodal LLM
  - cognitive state decoding
date: 2026-05-08
content_hash: c3e73d0b58655575
---

# Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading

**Conference**: ICLR2026
**arXiv**: [2505.02872](https://arxiv.org/abs/2505.02872)
**Code**: To be confirmed
**Area**: Video Understanding
**Keywords**: eye tracking, reading comprehension, information seeking goal decoding, multimodal LLM, cognitive state decoding

## TL;DR
This paper introduces a novel task of decoding open-ended information seeking goals from eye movement trajectories during reading. Built upon the OneStop eye-tracking dataset (360 participants, 486 questions, 162 passages), the authors develop both discriminative and generative multimodal models. RoBERTEye-Fixations achieves 49.3% accuracy on three-way goal selection (random baseline: 33%) and 70.9% on different-critical-span conditions; DalEye-Llama/GPT also significantly outperforms eye-movement-free baselines on goal reconstruction.

## Background & Motivation

**Background**: Eye tracking is a core methodology for studying reading cognition; however, existing work primarily focuses on general "reading for comprehension" scenarios, neglecting the more prevalent information-seeking reading that occurs in everyday life.

**Limitations of Prior Work**: Existing cognitive state decoding work distinguishes only a small number of predefined reading modes (e.g., skimming vs. careful reading) and cannot handle open-ended, text-specific information seeking goals.

**Core Idea**: Given a passage and a reader's eye movement data, the system automatically decodes the specific question in the reader's mind—without any prior knowledge beyond the text itself, extracting goal signals solely from fixation durations, saccade sequences, and related eye movement features.

## Method

### Overall Architecture
The task is formulated in two variants: **goal selection** (identifying the reader's true question from 3 candidate questions) and **goal reconstruction** (generating the text of the question in the reader's mind). The OneStop information-seeking dataset is used: each passage has 3 associated questions, of which 2 share the same critical span and 1 has a different critical span.

### Key Design 1: Discriminative Models
- **Reading-Time Weighted Embedding Similarity**: Fixation durations are used to weight RoBERTa word embeddings, which are then compared to candidate questions via cosine similarity (baseline; performs near chance level).
- **RoBERTEye-Fixations**: Per-fixation eye movement features are integrated into RoBERTa; an attention mechanism jointly processes text tokens and fixation sequences using 10-fold cross-validation; supports generalization to new texts and new readers.

### Key Design 2: Generative Models
- **DalEye-Llama/GPT**: Task description, passage, and eye movement trajectories (fixated word indices + durations + saccade directions) are converted into text prompts; Llama 3.1 / GPT-4o-mini are fine-tuned for question reconstruction.
- **Gemini zero-shot/few-shot**: Using the same textualized eye movement representation, Gemini-3-Pro is prompted directly for zero-shot/few-shot generation.

### Key Design 3: Cognitive Interpretability Analysis
A linear mixed-effects model is used to analyze the relationship between RoBERTEye performance and 11 trial-level features. Longer reading time within the critical span and shorter reading time outside the span correlate with higher model accuracy ($p < 10^{-275}$). More goal-directed reading behavior leads to easier decoding.

## Key Experimental Results

### Goal Selection Accuracy

| Model | All (3-way) | Different Span (2-way) | Same Span (2-way) |
|-------|------------|------------------------|-------------------|
| Random Baseline | 33.0% | 55.3% | 49.9% |
| Haller RNN | 41.8% | 65.6% | 52.1% |
| **RoBERTEye-Fixations** | **49.3%** | **70.9%** | **57.3%** |

### Goal Reconstruction Comparison

| Model | Question Word Acc | BERTScore | QA Acc |
|-------|------------------|-----------|--------|
| Text-only Llama (no eye movements) | Baseline | Baseline | Baseline |
| DalEye-Llama | Significantly above baseline | Significantly above baseline | Significantly above baseline |
| DalEye-GPT | Significantly above baseline | Significantly above baseline | Significantly above baseline |
| Gemini few-shot | Best on new-reader condition | Significantly above baseline | Best on new-reader condition |

### Key Findings
- Even when two candidate questions focus on the same textual region (same span), RoBERTEye distinguishes them at 57.3% accuracy ($p < 0.001$), indicating that eye movements encode fine-grained cognitive information beyond mere "where one looks."
- Fixation order within the eye movement sequence is more informative than individual fixation features—ablation analysis shows that removing word embedding ordering leads to the largest performance drop.
- In the generation task, eye movement information continues to make a significant contribution in the new-text generalization setting.

## Highlights & Insights
- **Pioneering Task Formulation**: This is the first work to formally define "open-ended reading goal decoding" as both a selection and a reconstruction task, with an elegant difficulty stratification via same-span vs. different-span conditions.
- **Bidirectional Bridge Between Cognition and Computation**: Model performance can be explained through cognitive theory (goal-directed reading behavior → information filtering → stronger signal), and conversely, the model can serve as an analytical tool to validate cognitive hypotheses.
- The dataset is large-scale (over 1.05 million word-level eye movement records), and evaluation is comprehensive (three generalization settings: new readers, new texts, and new readers + texts).

## Limitations & Future Work
- Current accuracy (49.3%) remains far from practical deployment, especially in the same-span condition (57.3%), which only marginally exceeds chance.
- Experiments are conducted only in English; generalizability across languages and populations (e.g., L2 readers, dyslexic readers) remains unknown.
- Generative models show notable performance degradation on new-text scenarios, suggesting that better eye movement encoding strategies may be needed.

## Related Work & Insights
- Unlike traditional task-based reading research (covering a small number of predefined modes such as skimming and proofreading), this paper addresses hundreds of text-specific reading goals.
- The findings may inspire applications in educational systems (real-time detection of students' reading goals) and content personalization (adapting presentation based on users' information needs).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Novel task definition + elegant experimental design
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple models, baselines, and evaluation dimensions; in-depth cognitive interpretability analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem motivation; smooth narrative bridging cognitive science and NLP
- Value: ⭐⭐⭐⭐ High scientific value, though practical application requires higher accuracy

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](../../CVPR2026/video_understanding/movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[ACL 2026\] Probing for Reading Times](../../ACL2026/video_understanding/probing_for_reading_times.md)
- [\[ICCV 2025\] TOGA: Temporally Grounded Open-Ended Video QA with Weak Supervision](../../ICCV2025/video_understanding/toga_temporally_grounded_open-ended_video_qa_with_weak_supervision.md)
- [\[ICLR 2026\] Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs](map_the_flow_revealing_hidden_pathways_of_information_in_videollms.md)
- [\[CVPR 2026\] GoalForce: Teaching Video Models to Accomplish Physics-Conditioned Goals](../../CVPR2026/video_understanding/goal_force_teaching_video_models_to_accomplish_physics-conditioned_goals.md)

</div>

<!-- RELATED:END -->
