---
title: >-
  [Paper Note] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models
description: >-
  [ICCV 2025][Multimodal VLM][Number sense] This paper proposes VisNumBench, a benchmark containing approximately 1,900 multiple-choice questions covering 7 visual numerical attributes and 4 types of visual numerical estimation tasks. It systematically evaluates the intuitive number sense of 17 MLLMs, revealing that even state-of-the-art models perform far below the human level.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "Number sense"
  - "visual numerical estimation"
  - "MLLM evaluation benchmark"
  - "quantity perception"
  - "multimodal reasoning"
date: 2026-05-08
content_hash: 3466be466b7f0ca6
---

# VisNumBench: Evaluating Number Sense of Multimodal Large Language Models

**Conference**: ICCV 2025  
**arXiv**: [2503.14939](https://arxiv.org/abs/2503.14939)  
**Code**: [https://wwwtttjjj.github.io/VisNumBench/](https://wwwtttjjj.github.io/VisNumBench/)  
**Area**: Multimodal VLMs  
**Keywords**: Number sense, visual numerical estimation, MLLM evaluation benchmark, quantity perception, multimodal reasoning

## TL;DR

This paper proposes VisNumBench, a benchmark containing approximately 1,900 multiple-choice questions covering 7 visual numerical attributes and 4 types of visual numerical estimation tasks. It systematically evaluates the intuitive number sense of 17 MLLMs, revealing that even state-of-the-art models perform far below the human level.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made significant progress on complex multimodal tasks. Existing evaluation benchmarks (e.g., MathVista, Math-Vision) mainly focus on symbolic mathematical reasoning and structured numerical computing.

**Limitations of Prior Work**: Existing benchmarks focus on abstract mathematical problem-solving, overlooking a core ability in human cognition—**intuitive number sense**. Humans can estimate angles, lengths, and quantities at a glance, but whether MLLMs possess similar capabilities remains completely unknown.

**Key Challenge**: While MLLMs can handle complex mathematical problems (relying on symbolic reasoning), they may perform poorly on simple numerical estimation tasks that require visual intuition. This reveals a deep-seated deficit in their "understanding" of numbers.

**Goal**: (a) To construct a benchmark specifically for evaluating visual number sense; (b) To systematically measure the gap in number sense between current MLLMs and humans.

**Key Insight**: Grounded in the cognitive science concept of the human Approximate Number System (ANS), the paper defines 7 visual numerical dimensions and 4 types of estimation tasks.

**Core Idea**: MLLMs' number sense is a foundational cognitive ability independent of mathematical reasoning, requiring dedicated evaluation and optimization.

## Method

### Overall Architecture

VisNumBench consists of two parts: VisNumBench-Synthetic (synthetic data, 1,011 questions) and VisNumBench-Real (real-world images, 902 questions). The input is an image combined with a multiple-choice question, and the output is the model's chosen answer. An automated pipeline is utilized for evaluation to directly compare model outputs with ground truth answers.

### Key Designs

1. **7 Visual Numerical Attributes**:

    - Angle, Length, Scale, Quantity, Depth, Area, Volume
    - Synthetic data covers the first 6 attributes, while real-world data covers 6 attributes excluding Area (incorporating Volume).
    - Design Motivation: To cover the primary dimensions of human number sense and avoid biased evaluations based on a single attribute.

2. **4 Visual Numerical Estimation Tasks**:

    - **Value Comparison**: Which of the two visual quantities is larger/longer?
    - **Value Estimation**: Estimating precise numerical values for a given figure (e.g., approximately how many degrees is the angle?).
    - **Range Estimation**: Which interval does the numerical value fall within?
    - **Multiplicative Estimation**: Approximately how many times larger is A than B?
    - Design Motivation: Hierarchical evaluation ranging from simple comparison to complex reasoning to align with different difficulty levels of human number sense.

3. **Data Construction Pipeline**:

    - **Synthetic Data**: Programmatic control is applied to precisely govern geometric parameters (angles, line segments, shape arrangements, etc.), ensuring unambiguous ground truths. Distractors are meticulously designed to avoid being excessively easy or difficult.
    - **Real-world Data**: Images are collected from real-world scenes (buildings, furniture, fruits, etc.), and quantities or depth relationships are manually annotated, ensuring that models' numerical intuition is tested in natural contexts.
    - Design Motivation: Synthetic data provides a controlled baseline, while real-world data evaluates generalization capabilities.

### Evaluation Protocol

- All questions are formatted as multiple-choice (3-5 options) to unify the evaluation standard.
- The human baseline is annotated by experts, achieving an average accuracy of approximately 95%.
- Zero-shot inference is used for each MLLM without providing any examples.

## Key Experimental Results

### Main Results

| Model | Angle | Length | Scale | Quantity | Depth | Area | Average |
|------|-------|--------|-------|----------|-------|------|------|
| Random | 24.4 | 25.4 | 25.0 | 25.0 | 25.0 | 23.7 | 24.8 |
| Qwen2.5-VL-72B | 37.1 | 59.7 | 65.0 | 57.7 | 61.5 | 70.4 | **58.5** |
| InternVL2.5-78B | 35.3 | 59.7 | 68.6 | 42.9 | 61.5 | 72.5 | 56.2 |
| Gemini 2.0 Flash | 31.2 | 57.5 | 81.4 | 55.1 | 51.1 | 70.9 | 57.6 |
| GPT-4o | 35.3 | 43.1 | 54.3 | 37.2 | 54.1 | 43.4 | 43.7 |
| LLaVA-v1.5-7B | 31.2 | 30.4 | 34.3 | 33.2 | 26.7 | 21.2 | 29.4 |
| **Human** | **90.0** | **96.0** | **100.0** | **96.0** | **98.0** | **92.0** | **95.3** |

### Real-world Data Comparison

| Model | Angle | Length | Scale | Quantity | Depth | Volume | Average |
|------|-------|--------|-------|----------|-------|--------|------|
| Qwen2.5-VL-72B | 34.2 | 50.6 | 43.4 | 80.3 | 52.6 | 59.2 | 53.3 |
| InternVL2.5-78B | 36.9 | 58.6 | 49.0 | 79.6 | 52.6 | 62.6 | 56.5 |
| GPT-4o | 27.5 | 30.3 | 37.1 | 60.5 | 35.7 | 47.6 | 39.6 |
| LLaVA-v1.6-34B | 28.9 | 54.9 | 23.1 | 68.0 | 63.6 | 63.3 | 50.6 |
| **Human** | ~ 95 | ~ 95 | ~ 95 | ~ 95 | ~ 95 | ~ 95 | **~ 95** |

### Key Findings
- **Angle estimation is the most prominent bottleneck for all models**: Even the best-performing model (InternVL2.5-78B) only achieves 36.9% accuracy, which is close to random guessing (25%), compared to 90.0% for humans. Angle perception may require spatial rotation reasoning capabilities.
- **Quantity perception performs better in real-world scenarios**: Qwen2.5-VL-72B reaches 80.3% accuracy in Quantity on real-world data but only 57.7% on synthetic data, indicating a higher prevalence of counting scenarios in pre-training data.
- **Model scaling yields limited improvements**: For models of the same family, scaling parameters by 24x (from 3B to 72B) only improves average accuracy by around 16 percentage points.
- **Multimodal math/CoT models show no significant advantage**: This indicates that number sense is an independent dimension of capability separate from symbolic reasoning.
- **Newer versions outperform older ones within the same family** (e.g., Qwen2.5-VL > Qwen2-VL), suggesting that iterative improvements in datasets and training methodologies slowly enhance number sense.

## Highlights & Insights
- **Comprehensive and theoretically grounded design of evaluation dimensions**: The cross-coverage of 7 numerical attributes × 4 estimation tasks far exceeds existing benchmarks. Grounded in the Approximate Number System from cognitive science, it possesses a solid theoretical foundation.
- **Unveiling an overlooked bottleneck in capability**: MLLMs' near-random performance on intuitive number sense could be an underlying cause of failures in many downstream tasks (e.g., chart understanding and spatial reasoning).
- **The complementary design of synthetic + real-world data is highly effective**: Synthetic data isolates pure numerical reasoning from the interference of scene complexity, while real-world data examines practical utility. The performance discrepancy between these two domains reveals interesting patterns.

## Limitations & Future Work
- The dataset size is relatively small (only ~1,900 questions), which may lead to statistical fluctuations.
- Evaluation is limited to zero-shot setups; the potential of few-shot or chain-of-thought prompting to improve number sense remains unexplored.
- No training data or fine-tuning schemes are provided to improve number sense, keeping the study restricted to the "diagnostic" level.
- Volume estimation only appears in the real-world dataset, lacking controlled experiments in synthetic data.
- Future research could explore improvement pathways by incorporating vision-spatial reasoning enhancement methods (e.g., spatial tokens, coordinate encoding).

## Related Work & Insights
- **vs. MathVista**: MathVista focuses on solving math problems (requiring symbolic computation), whereas VisNumBench measures intuitive number sense (without requiring computation). The two are complementary.
- **vs. PhysBench**: PhysBench evaluates physical reasoning, while VisNumBench evaluates numerical intuition. Both expose the limitations of MLLMs in "common-sense" domains.
- **vs. Ordinal Regression Research** (e.g., age estimation, image aesthetics assessment): While they test numerical estimation in specific domains, VisNumBench provides a unified cross-domain framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The first benchmark to systematically evaluate MLLM number sense, featuring a novel problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive testing on 17 models, though lacking mitigation strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive figures/tables.
- Value: ⭐⭐⭐⭐ Unravels overlooked cognitive deficits in MLLMs, offering inspiring insights for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](../../ACL2025/multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](../../NeurIPS2025/multimodal_vlm/evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)

</div>

<!-- RELATED:END -->
