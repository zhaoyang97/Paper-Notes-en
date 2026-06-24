---
title: >-
  [Paper Note] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video
description: >-
  [VLM Reasoning] This paper proposes RTV-Bench, a benchmark comprising 552 videos and 4,608 QA pairs, designed to systematically evaluate MLLMs' continuous analysis capabilities in real-time video streams through three core designs: **multi-timestamp QA** (the same question yields different correct answers at different timestamps), **hierarchical question structure**, and **multidimensional evaluation**. Key findings include that online models outperform offline models…
tags:
  - "VLM Reasoning"
date: 2026-05-08
content_hash: 0d40bcf8744f4ff4
---

# RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video

## TL;DR

This paper proposes RTV-Bench, a benchmark comprising 552 videos and 4,608 QA pairs, designed to systematically evaluate MLLMs' continuous analysis capabilities in real-time video streams through three core designs: **multi-timestamp QA** (the same question yields different correct answers at different timestamps), **hierarchical question structure**, and **multidimensional evaluation**. Key findings include that online models outperform offline models, and that simply scaling model size or increasing frame count yields limited gains.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have advanced rapidly in visual perception, understanding, and reasoning. Video-LLM research has expanded from short clips to long-form content, with growing integration of multimodal signals such as video, audio, and subtitles.

**Limitations of Prior Work**: Existing video benchmarks (e.g., Video-MME, MVBench) are primarily designed for offline evaluation using static QA pairs, and are thus unable to assess models' real-time responsiveness to continuous dynamic video streams. While VStream, StreamingBench, and OVOBench offer improvements, they remain insufficient in evaluating real-time responsiveness—neglecting models' ability to capture transitions and instantaneous details in sequentially arriving visual inputs.

**Key Challenge**: Real-time video scenarios require models to continuously maintain coherent understanding and update their internal states as visual scenes evolve, yet the prevailing "single-question, single-answer" static evaluation paradigm in existing benchmarks cannot effectively assess this continuous analysis capability.

**Goal**: To construct a fine-grained real-time video analysis benchmark that systematically evaluates MLLMs' continuous perception, understanding, and reasoning capabilities in dynamic video streams.

**Key Insight**: The paper approaches the problem from three dimensions: (1) multi-timestamp QA, where the same conceptual question has different correct answers at different time points; (2) a hierarchical question structure that progressively increases in difficulty; and (3) a multidimensional evaluation framework covering 8 dimensions for fine-grained diagnosis.

**Core Idea**: By designing questions whose correct answers change over time, the benchmark directly tests models' sensitivity to dynamic state transitions and their ability to continuously track evolving content.

## Method

### Overall Architecture

RTV-Bench is a fine-grained benchmark for real-time video analysis, comprising 552 diverse videos (total duration: 167.2 hours; average: 18.2 minutes per video) and 4,608 manually annotated QA pairs. The videos span three major domains (autonomous driving, sports events, egocentric perspective) across 16 subcategories. The evaluation framework includes two metrics: Accuracy and Score (a conditioned metric that counts higher-level question scores only when all basic questions are answered correctly). Online models are queried directly at the designated timestamps, while offline models are evaluated on video clips truncated at the query timestamp.

### Key Designs

1. **Multi-Timestamp QA**
    - **Function**: Within the same video, the same conceptual question yields different correct answers at different timestamps.
    - **Mechanism**: Unlike OVO-Bench, which introduces different questions at different timestamps, RTV-Bench revisits the same conceptual query (e.g., "What is A holding?"), with the correct answer changing as the scene unfolds. Annotators manually label the earliest valid timestamp for each answer option.
    - **Design Motivation**: To more rigorously test continuous analysis capability—requiring models to actively track temporal changes and continuously update their understanding, rather than merely locating relevant information.

2. **Hierarchical Question Structure**
    - **Function**: Each question group contains approximately three multiple-choice questions of increasing difficulty; the first two are basic perceptual questions, and the third is a high-difficulty integrative reasoning question.
    - **Mechanism**: Higher-order questions logically depend on mastery of basic perception and understanding. Combined with the Score metric—which only counts advanced question scores when all basic questions are answered correctly—this ensures that evaluation reflects genuine hierarchical reasoning capability.
    - **Design Motivation**: To prevent models from achieving spuriously high scores on complex questions through cognitive shortcuts, ensuring that advanced reasoning is grounded in solid foundational understanding.

3. **Multidimensional Evaluation**
    - **Function**: Fine-grained diagnosis across 8 dimensions—Temporal Perception (TP), Scene Perception (SP), Visual Perception (VP), Future Prediction (FP), Phenomenon Understanding (PU), Intention Analysis (IA), Global Understanding (GU), and Spatiotemporal Reasoning (SR).
    - **Mechanism**: Capability dimensions are organized into three categories—perception, understanding, and reasoning—each with 2–3 sub-dimensions, providing a detailed model capability profile beyond aggregate scores.
    - **Design Motivation**: To offer researchers an informative view of model capabilities and limitations across different aspects of dynamic scene comprehension, thereby guiding targeted improvements.

### Loss & Training

As a benchmark paper, no model training is involved. Regarding evaluation metric design:

- **Accuracy**: Directly computed as the proportion of correct answers.
- **Score (Conditioned Metric)**: $$\text{Score} = \frac{\sum_{i=1}^{N} B_i \cdot N_{q2,i}^{\text{correct}}}{\sum_{i=1}^{N} N_{q2,i}^{\text{total}}}$$ where $B_i$ is an indicator that equals 1 only when all basic questions are answered correctly. This metric ensures that advanced question scores are counted only when foundational questions are fully correct, reflecting model reliability and hierarchical reasoning consistency.

## Key Experimental Results

### Main Results

| Model | Scale | Perception Acc/Score | Understanding Acc/Score | Reasoning Acc/Score | FQA Acc | MTQA Acc | Overall Acc/Score |
|-------|-------|----------------------|-------------------------|---------------------|---------|----------|-------------------|
| GPT-4o | - | 51.61/21.90 | 49.31/20.76 | 48.71/23.95 | 56.53 | 44.73 | **50.02/22.10** |
| IXC2.5-OL | 7B | 47.21/15.87 | 48.22/15.23 | 46.18/14.45 | 59.05 | 38.21 | 47.33/15.40 |
| VITA-1.5 | 7B | 45.66/12.80 | 44.12/11.83 | 43.37/10.15 | 55.06 | 36.32 | 44.51/11.80 |
| VideoChat-Online | 4B | 46.86/12.30 | 46.34/12.80 | 43.53/11.00 | 55.16 | 38.21 | 45.83/12.10 |
| Qwen2.5-VL | 7B | 42.30/7.70 | 39.85/7.00 | 38.16/6.90 | 44.07 | 37.46 | 40.41/7.13 |
| VideoLLaMA2 | 7B | 40.62/8.67 | 39.85/7.77 | 37.49/6.75 | 45.77 | 34.95 | 39.55/7.90 |
| LLaVA-Video | 7B | 35.83/5.03 | 33.81/3.77 | 35.15/5.75 | 36.28 | 34.17 | 34.90/4.80 |

### Ablation Study

**Effect of Frame Count and Model Scale (Qwen2.5-VL)**

| Model Scale | 8 frames | 16 frames | 32 frames | 64 frames | Trend |
|-------------|----------|-----------|-----------|-----------|-------|
| 3B | ~37% | ~37% | ~38% | ~37% | Non-monotonic fluctuation |
| 7B | ~39% | ~40% | ~40% | ~40% | Marginal improvement |
| 32B | ~39% | ~39% | ~40% | ~40% | Slight improvement |
| 72B | ~40% | ~40% | ~40% | ~40.78% | Best overall, but diminishing returns |

### Key Findings

1. **Online models substantially outperform offline models**: IXC2.5-OL (47.33%) significantly surpasses the best offline model Qwen2.5-VL (40.41%); even the weakest online model, VITA-1.5, outperforms the offline representative VideoLLaMA2.
2. **Increasing frame count yields limited or even negative returns**: Increasing the number of sampled frames does not consistently improve performance and in some cases leads to degradation (e.g., IXC2.5-OL shows notable performance drops with more frames), suggesting that excessive temporal input may cause attention dilution.
3. **Model scale positively correlates with performance but with diminishing returns**: Accuracy improves by approximately 2–3 percentage points from 3B to 72B; larger models benefit more stably from additional frames, but the absolute gains remain limited.
4. **MTQA is the core bottleneck**: All models achieve substantially lower accuracy on multi-timestamp QA (33%–44%) compared to basic QA (35%–59%), indicating that continuous state tracking remains a fundamental challenge.
5. **Large gap between Score and Accuracy**: All models' Score values are far below their Accuracy values, revealing frequent failures on basic questions and raising serious concerns about the reliability of advanced reasoning.

## Highlights & Insights

- **The "same question, time-varying answers" design is particularly elegant**: It more rigorously tests continuous tracking capability than asking different questions at different timestamps, representing a qualitative advancement in the evaluation paradigm for real-time understanding.
- **The conditioned Score metric is conceptually illuminating**: By requiring all basic questions to be answered correctly before counting advanced question scores, it effectively identifies models that guess correctly on complex questions while exhibiting deficient foundational understanding. This form of hierarchical consistency evaluation merits broader adoption.
- **The counterintuitive findings carry important practical implications**: The finding that more frames do not necessarily help challenges the naive assumption that "more information is always better," pointing toward research directions in temporally selective modeling and adaptive frame utilization.
- **The systematic online vs. offline comparison provides clear architectural guidance**: The advantage of dedicated streaming architectures stems from continuous state updating, not merely offline preprocessing.

## Limitations & Future Work

1. **Limited to the visual modality**: Audio and other important modalities are not incorporated; in practice, multimodal signals are mutually complementary in real-time scenarios.
2. **Moderate evaluation scale**: The scale of 552 videos and 4,608 QA pairs is relatively modest, and scenario diversity warrants further expansion.
3. **Questionable fairness in offline model evaluation**: Truncating video clips for offline models alters the task setup and may underestimate the true capabilities of offline models.
4. **Lack of in-depth analysis of model internals**: While counterintuitive phenomena such as "frame count ineffectiveness" are observed, the underlying causes are not deeply analyzed, remaining at the level of empirical observation.
5. **Expandable model coverage**: Future editions could include additional recent streaming video models (e.g., the VideoLLM-Online series) as well as larger-scale closed-source models.

## Related Work & Insights

- **StreamingBench / OVOBench**: Preceding streaming video benchmarks that evaluate different questions at different timestamps; RTV-Bench's same-question design imposes a stricter test.
- **VStream**: An early attempt at real-time video evaluation, but focused on extending video duration rather than assessing continuous analysis.
- **IXC2.5-OmniLive**: Achieves streaming processing via modular parallelism and long-term memory; the best-performing open-source model on RTV-Bench.
- **VITA-1.5**: An online model trained with staged multimodal fusion, validating the necessity of streaming architecture design.
- **Insights**: Future real-time video understanding calls not for larger models or more frames, but for principled temporal modeling (temporally selective attention) and selective information aggregation strategies.

## Rating

- **Novelty** ⭐⭐⭐⭐ — The multi-timestamp QA design (same question with time-varying answers) and the conditioned Score metric are novel, filling a gap in the evaluation of continuous analysis in real-time video.
- **Practicality** ⭐⭐⭐⭐ — Provides systematic diagnosis of current MLLMs' real-time video capabilities; the 8-dimensional evaluation framework and hierarchical metric design offer valuable references for future benchmarks.
- **Reliability** ⭐⭐⭐⭐ — The scale of 552 videos and 4,608 QA pairs is adequate; manual annotation quality is high with rigorous multi-round review, though fairness concerns remain regarding the offline model evaluation protocol.
- **Overall** ⭐⭐⭐⭐ — A high-quality benchmark paper with sound and insightful core design principles, revealing important counterintuitive findings and providing valuable directional guidance for the development of real-time video MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Training-Free Personalization via Retrieval and Reasoning on Fingerprints](../../ICCV2025/vlm_reasoning/training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)
- [\[CVPR 2025\] Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents](../../CVPR2025/vlm_reasoning/document_haystacks_vision-language_reasoning_over_piles_of_1000_documents.md)
- [\[ACL 2025\] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](../../ACL2025/vlm_reasoning/mammoth_vl_multimodal_reasoning.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/vlm_reasoning/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](../../ICCV2025/vlm_reasoning/r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)

</div>

<!-- RELATED:END -->
