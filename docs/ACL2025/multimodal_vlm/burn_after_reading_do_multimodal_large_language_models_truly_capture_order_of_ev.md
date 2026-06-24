---
title: >-
  [Paper Note] Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?
description: >-
  [ACL 2025][Multimodal VLM][Temporal reasoning] The paper proposes the TempVS benchmark to systematically evaluate the grounding and reasoning capabilities of 38 MLLMs on multi-event temporal relationships in image sequences, revealing a substantial performance gap between state-of-the-art models and humans.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Temporal reasoning"
  - "multi-image understanding"
  - "event ordering"
  - "benchmark"
  - "multimodal large language models"
date: 2026-05-08
content_hash: 8c73cc6f6a67aa8e
---

# Burn After Reading: Do Multimodal Large Language Models Truly Capture Order of Events in Image Sequences?

**Conference**: ACL 2025  
**arXiv**: [2506.10415](https://arxiv.org/abs/2506.10415)  
**Code**: [GitHub](https://github.com/yjsong22/TempVS)  
**Area**: Multimodal VLM  
**Keywords**: Temporal reasoning, multi-image understanding, event ordering, benchmark, multimodal large language models

## TL;DR

The paper proposes the TempVS benchmark to systematically evaluate the grounding and reasoning capabilities of 38 MLLMs on multi-event temporal relationships in image sequences, revealing a substantial performance gap between state-of-the-art models and humans.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have demonstrated outstanding performance on single-image visual understanding tasks. However, the evaluation of temporal understanding and reasoning in multi-image scenarios remains relatively weak. Existing multi-image benchmarks primarily focus on描写 cross-image recognition and referencing, rarely addressing temporal relationships.

**Limitations of Prior Work**: Existing temporal evaluations suffer from three major issues: (a) some tasks can be answered using only a single image without needing to understand the sequence; (b) some tasks rely heavily on common sense and world knowledge (e.g., ordering cooking steps); (c) some benchmarks use distractor options that do not exist in the images, allowing models to infer answers based on object existence.

**Key Challenge**: Existing benchmarks cannot truly evaluate a model's understanding of multi-event temporal relations in visual storytelling—models might "pass" tests through shortcuts rather than actual temporal reasoning.

**Goal**: Construct a rigorous, cheat-proof temporal benchmark to answer the core question: "Do existing MLLMs truly understand the temporal order of events in image sequences?"

**Key Insight**: Start from visual storytelling and select image sequences where events are relatively independent and future events are difficult to predict from preceding ones, forcing models to integrate both visual and textual modalities to complete the task.

**Core Idea**: By designing three main tests (Event Relation Inference, Sentence Ordering, Image Ordering) and their corresponding grounding tests, eliminate single-modality shortcuts to atomically evaluate the temporal understanding capabilities of MLLMs.

## Method

### Overall Architecture

The TempVS benchmark contains **2,085 image sequences** (9,803 images) covering four data sources: cartoons (FlintstonesSV, PororoSV), movie scenes (VWP), and daily photo albums (VIST), generating a total of **15,192 multiple-choice questions**.

The benchmark is designed as a three-layer structure:
- **MT1: Event Relation Inference** — Determine whether statements describing the temporal relationship of two/three events are consistent with the image sequence (True/False).
- **MT2: Sentence Ordering** — Given an ordered image sequence and a shuffled set of sentences, select the correct sentence order from five options.
- **MT3: Image Ordering** — Given a text description and a shuffled set of images, select the correct image order.
- **GT: Grounding Test** — Given an event description and an image sequence, localize the corresponding image.

### Key Designs

**Data Formatting**: Each image sequence is represented as $\mathcal{S} = [(I_1, C_1, E_1), ..., (I_n, C_n, E_n)]$, where $I_i$ is the image, $C_i$ is the original caption, and $E_i$ is the extracted simplified event.

**Template-driven Positive/Negative Sample Construction**: MT1 uses 10 templates (5 for two-event + 4 for three-event) to generate positive and negative statements. Negative samples are constructed by swapping the positions of event clauses, maintaining the same temporal connectives while only changing the event order. For example:
- Positive: "$E_j$ after $E_i$" → Negative: "$E_i$ after $E_j$"
- Positive: "$E_i$. Then, $E_j$" → Negative: "$E_j$. Then, $E_i$"

**Multi-layer Filtering Mechanism**:
- Use Detectron2 to retain sequences where $\ge 60\%$ of images contain people.
- Remove sequences containing stative verbs (e.g., belong, love) to avoid temporal overlap.
- Remove overly similar descriptions and images using BERTScore and CLIP cosine similarity.
- Use CLIP cross-modal similarity to filter ambiguous pairs, ensuring $sim(I_i, E_i) > sim(I_i, E_j)$.

**Language Bias Prevention**: Filter using three text-only LLMs (Phi-3.5-mini [4B], Llama-3.1 [8B], and Qwen-2.5 [72B]), discarding samples in MT1 and MT2 that at least two LLMs can answer correctly using only text.

### Evaluation Metrics

- **MT Accuracy**: Standard multiple-choice accuracy.
- **GT_strict**: The proportion of image sequences where the model passes all corresponding grounding tests.
- **MT|GT_strict**: Main task accuracy calculated only under the condition of passing all grounding tests.

## Key Experimental Results

### Main Results

38 MLLMs (with parameter sizes ranging from 0.5B to 78B) were evaluated. The main results are as follows:

| Model | Params | MT1(2-Event) | MT1(3-Event) | MT2(Event) | MT2(Caption) | MT3 |
|------|------|------------|------------|----------|----------|-----|
| InternVL2.5-78B-MPO | 78B | 58.5 | 61.4 | **79.8**| **86.3**| **53.8**|
| InternVL2.5-26B-MPO | 26B | **60.3**| 62.1 | 69.9| 76.9| 34.4|
| GPT-4o | API | 58.3 | **64.5**| 53.4| 61.5| 22.6|
| LLaVA-OneVision-72B | 72B | 59.3 | 61.5 | 65.2| 75.1| 27.6|
| Random Baseline | - | 50   | 50   | 20  | 20  | 20  |
| **Human** | - | **82.5**| **80.0**| -   | -   | -   |

### Ablation Study

**Impact of Event Distance**: The further apart the events are, the higher the MT1 accuracy (models can more easily distinguish events that are temporally distant).

**Impact of Linguistic Structure**: Using original captions yields better results than using simplified event descriptions (captions provide extra context and temporal clues).

**Chain-of-Thought (CoT)**: Provides limited improvement for most models, and in some cases, even degrades performance.

**Relationship Between Grounding and Temporal Reasoning**:
- For InternVL2.5-78B-MPO in MT2, adding the GT_strict constraint increased accuracy from 79.8% to **96.6%** (Events) and from 86.3% to **96.4%** (Captions).
- This demonstrates that grounding capability is an important prerequisite for temporal reasoning.

### Key Findings

1. **MT1 is Close to Random**: Most models with $\le 7\text{B}$ parameters score around 50% (random level) on MT1 and around 20% on MT2/MT3.
2. **Image Ordering is the Hardest**: The best model scored only 53.8% on MT3, which is far lower than the 86.3% achieved on MT2.
3. **Uneven Performance of GPT-4o**: GPT-4o achieved the best grounding performance but lags significantly behind leading open-source models on ordering tasks.
4. **Dual Importance of Model Scale & Post-Training**: InternVL2.5-MPO systematically outperforms its non-MPO counterparts across all tasks.

## Highlights & Insights

- **Exquisite Anti-Cheating Design**: The three-layer filtering (visual similarity, text similarity, and text-only LLMs) effectively eliminates shortcuts, ensuring evaluation purity.
- **Decoupled Analysis of Grounding and Reasoning**: The GT + MT|GT_strict metrics clearly demonstrate that "recognizing images $\ne$ understanding temporal order".
- **Cross-Domain Data Sources**: Covering cartoons, movies, and daily photos enhances the generalization of the evaluation.
- **Template Diversity**: 328 prompt variants avoid specific-prompt bias.

## Limitations & Future Work

1. Only multiple-choice formats were used; open-ended generation capabilities were not evaluated.
2. Most image sequences consist of 5 images; temporal reasoning on longer sequences (e.g., 10+ images) was not tested.
3. Only linear temporal relationships of events were evaluated; complex temporal relationships such as parallel or overlapping events were not addressed.
4. During evaluation, multiple images are horizontally concatenated into a single image input, which may discard single-image details.
5. Further validation can be conducted on video understanding models.

## Related Work & Insights

- Multi-image benchmarks like **MIBench / MuirBench / MMIU** mainly focus on cross-image recognition; TempVS fills the gap in temporal reasoning evaluation.
- **Mementos** focuses on hallucination detection in sequential images but does not involve reverse reasoning (ordering images from text).
- Inundated insights: Design specialized temporal alignment pre-training objectives, or introduce event graph structures to enhance model temporal reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first systematic multi-event temporal grounding and reasoning benchmark with innovative anti-cheating designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated 38 models across 4 data sources with human baselines and multi-dimensional analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with precise task definitions.
- **Value**: ⭐⭐⭐⭐ — Provides a standardized tool for evaluating MLLM temporal capabilities, exposing crucial blind spots.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ACL 2025\] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)
- [\[CVPR 2026\] Personalized Image Descriptions from Attention Sequences](../../CVPR2026/multimodal_vlm/personalized_image_descriptions_from_attention_sequences.md)
- [\[CVPR 2026\] Do Vision-Language Models Measure Up? Benchmarking Visual Measurement Reading with MeasureBench](../../CVPR2026/multimodal_vlm/do_vision-language_models_measure_up_benchmarking_visual_measurement_reading_wit.md)
- [\[ACL 2025\] COLING-UniA at SciVQA 2025: Few-Shot Example Retrieval and Confidence-Informed Ensembling for Multimodal Large Language Models](coling-unia_at_scivqa_2025_few-shot_example_retrieval_and_confidence-informed_en.md)

</div>

<!-- RELATED:END -->
