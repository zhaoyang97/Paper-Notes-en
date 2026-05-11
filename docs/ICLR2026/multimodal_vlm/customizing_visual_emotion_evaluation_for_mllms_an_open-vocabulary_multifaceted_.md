---
title: >-
  [Paper Note] Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach
description: >-
  [ICLR 2026][Multimodal VLM][Visual Emotion] This paper proposes the Emotion Statement Judgment (ESJ) task and the INSETS automatic annotation pipeline…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Visual Emotion"
  - "MLLM Evaluation"
  - "Open-vocabulary"
  - "ESJ"
  - "MVEI Benchmark"
date: 2026-05-08
content_hash: 03f9d613b21ba9ae
---

# Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach

**Conference**: ICLR 2026
**arXiv**: [2509.21950](https://arxiv.org/abs/2509.21950)
**Code**: [GitHub](https://github.com/wdqqdw/MVEI)
**Area**: Multimodal VLM
**Keywords**: Visual Emotion, MLLM Evaluation, Open-vocabulary, ESJ, MVEI Benchmark

## TL;DR

This paper proposes the Emotion Statement Judgment (ESJ) task and the INSETS automatic annotation pipeline, reformulating visual emotion evaluation from "open-ended classification" to "statement veracity judgment." The authors construct the MVEI benchmark (3,086 samples, 424 emotion labels, four cognitive dimensions) and systematically evaluate 19 MLLMs, finding that even GPT-4o lags behind humans (91.6%) by 13.3% in accuracy.

## Background & Motivation

**Background**: Affective image content analysis (AICA) is a critical direction in multimodal understanding. As MLLMs continue to advance on general visual tasks, their visual emotion perception capabilities have attracted increasing attention; however, research conclusions remain contradictory—some studies report that MLLMs have limited emotion recognition ability, while others successfully employ them as emotion annotators for data augmentation.

**Limitations of Prior Work**: The authors systematically attribute this contradiction to an incompatibility between traditional evaluation paradigms and MLLMs, manifesting in four aspects: (1) fixed label sets exclude other plausible answers—emotion perception is inherently subjective and the same image can evoke different responses; (2) emotion classification granularity is too coarse—mainstream benchmarks (FI, Artemis) contain only 8 emotion categories; (3) contextual factors are neglected—only intrinsic image attributes are considered, ignoring scene, viewer identity, and other external factors psychologically proven to influence emotion perception; (4) annotation costs are prohibitively high—the EMOTIC dataset required coordinating 23,788 crowdsourced annotators.

**Key Challenge**: Existing evaluations query MLLMs with open-ended questions (e.g., "What emotion does this image convey?"). On one hand, the open answer space renders evaluation criteria ambiguous; on the other hand, closed taxonomy systems fail to capture fine-grained emotional distinctions. A fundamental conflict therefore exists between evaluation precision and coverage.

**Goal**: (1) How to eliminate answer ambiguity in open-ended emotion evaluation? (2) How to cover fine-grained emotions while maintaining scalability? (3) How to incorporate scene context and subjectivity as evaluation dimensions? (4) How to construct large-scale evaluation data with minimal human effort?

**Key Insight**: Inspired by cognitive psychology, the paper reframes emotion evaluation from "generative answering" to "verificational judgment"—requiring models to assess whether an image matches a given emotion statement—while designing four complementary dimensions that span the complete capability spectrum from basic emotion recognition to understanding of subjectivity.

**Core Idea**: Replacing "answering what emotion is present" with "judging whether an emotion statement is correct" fundamentally eliminates the ambiguity of open-ended evaluation, while an automated pipeline enables open-vocabulary, multi-dimensional, and large-scale assessment.

## Method

### Overall Architecture

The framework comprises two core components: the ESJ task defines *how* to evaluate, and the INSETS pipeline determines *what* to evaluate. The pipeline proceeds as follows: INSETS first automatically extracts open-vocabulary emotion labels from 17,716 images in EmoSet via ensemble voting across 9 MLLMs; emotion statements (half correct, half incorrect) are then constructed across four dimensions from these labels; 462K annotated instances are automatically generated (INSETS-462k); and finally, human refinement yields 3,086 high-quality MVEI benchmark samples. During evaluation, MLLMs receive an image–statement pair and are required to output only "Correct" or "Incorrect."

### Key Designs

1. **Four-Dimensional Evaluation Framework (grounded in cognitive psychology)**

    - **Function**: Comprehensively measure MLLMs' visual emotion understanding across four complementary dimensions.
    - **Mechanism**: (a) **Sentiment Polarity**—judges the emotional valence (positive/negative/mixed); correctness is automatically determined by the spectrum membership of labels in the Parrott Ontological Model (POM), paired with three predefined polarity statements. (b) **Emotion Interpretation**—combines a prototypical interpretation with an emotional state; a match constitutes a correct statement, while mismatches form incorrect ones via cross-image distraction (replacing the interpretation with one from a visually similar but emotionally different image) or within-image distraction (swapping labels between opposing polarities within the same image). (c) **Scene Context**—pairs a prototypical scene background with an emotional conclusion; incorrect statements are constructed via polarity reversal (random sampling from opposing POM spectra) or within-image swapping of scene descriptions across opposite polarities. (d) **Perception Subjectivity**—pairs a prototypical viewer role with a preference orientation toward a candidate emotion; incorrect statements are constructed by reversing the preference ordering.
    - **Design Motivation**: Existing benchmarks cover only the first two dimensions (intrinsic image attributes), whereas psychological research demonstrates that external factors—scene and viewer identity—critically influence emotion perception. The four dimensions constitute a complete capability spectrum from "recognizing emotions" to "understanding how emotions vary across persons and contexts."

2. **INSETS Open-Vocabulary Emotion Labeling Pipeline**

    - **Function**: Assign open-vocabulary emotion labels to images with minimal human intervention and automatically construct multi-dimensional emotion statements.
    - **Mechanism**: The pipeline operates in two stages. **Stage 1 (Labeling)**: Nine MLLMs each extract candidate emotion words per image (averaging 8–13 words per model); these are pooled and filtered by GPT-4 to remove inappropriate terms; surviving words are mapped onto the Parrott Ontological Model (POM; 6 primary / 25 secondary / 113 tertiary categories, forming an extended POM); consensus labels are then selected via POM-guided ensemble majority voting (quota allocation at the secondary-category level, followed by intra-category top-k selection by frequency). **Stage 2 (Construction)**: For each label, the source MLLM generates prototypical interpretation, scene, and role statements, which are then combined into correct/incorrect statement pairs according to dimension-specific rules.
    - **Design Motivation**: Single-MLLM annotation is susceptible to hallucination and bias. Multi-model ensemble combined with hierarchical psychological model constraints simultaneously ensures annotation reliability (90.6% accuracy) and open-vocabulary flexibility (751 distinct emotion labels).

3. **MVEI Benchmark Construction (Human Refinement)**

    - **Function**: Refine a high-quality evaluation benchmark from INSETS-462k.
    - **Mechanism**: A sample of 3,164 instances is drawn from the corpus; five graduate students evaluate annotation accuracy following dimension-specific guidelines, with ≥4/5 consensus designating correct labels, ≤1/5 designating incorrect labels, and intermediate cases flagged as ambiguous. Correct instances are retained, incorrect ones revised, and ambiguous ones discarded, yielding the final 3,086 MVEI samples.
    - **Design Motivation**: While automatic annotation is efficient, residual errors are inevitable. Human refinement ensures gold-standard quality, and the approximately 100 person-hours required is far lower than the cost of conventional annotation from scratch.

## Key Experimental Results

### Main Results

| Model | Parameters | Sentiment Polarity | Emotion Interpretation | Scene Context | Perception Subjectivity | Overall Acc. |
|-------|------------|-------------------|----------------------|---------------|------------------------|-------------|
| GPT-4o | — | 72.5% | 84.3% | 81.6% | 69.2% | 78.3% |
| InternVL2.5 | 8.3B | 75.7% | 80.2% | 79.4% | 61.3% | 74.7% |
| mPLUG-Owl3 | 8.1B | 73.9% | 79.3% | 81.7% | 75.0% | 78.1% |
| Qwen2.5-VL | 8.3B | 63.2% | 81.5% | 83.9% | 66.3% | 75.9% |
| Qwen2-VL | 8.3B | 70.7% | 75.0% | 86.1% | 72.8% | 76.6% |
| LLaVa-1.6 | 7.6B | 66.4% | 69.7% | 55.3% | 49.7% | 60.2% |
| **Human Average** | — | **92.3%** | **90.1%** | **95.3%** | **89.6%** | **91.6%** |

### Ablation Study (Gains from MLLM Adaptation Strategies on Qwen2.5-VL)

| Adaptation Strategy | Sentiment Polarity | Emotion Interpretation | Scene Context | Perception Subjectivity | Overall Acc. |
|--------------------|--------------------|----------------------|---------------|------------------------|-------------|
| Direct Inference | 63.2% | 81.5% | 83.9% | 66.3% | 75.9% |
| Chain-of-Thought | 67.4 (+4.2) | 81.5 (+0.0) | 84.6 (+0.7) | 67.0 (+0.7) | 76.6 (+0.8) |
| ICL 8-shot | 70.1 (+6.9) | 81.7 (+0.2) | 84.9 (+1.0) | 67.0 (+0.7) | 77.3 (+1.4) |
| LoRA Fine-tuning | 78.6 (+15.4) | 84.7 (+3.2) | 86.3 (+2.4) | 70.3 (+4.0) | 80.7 (+4.8) |
| Full Fine-tuning | 84.3 (+21.1) | 84.8 (+3.3) | 87.0 (+3.1) | 71.1 (+4.8) | 81.9 (+6.0) |
| GRPO | 83.2 (+20.0) | 82.5 (+1.0) | 86.5 (+2.6) | 71.1 (+4.8) | 80.7 (+4.8) |

### Key Findings

- **Sentiment polarity is one of the greatest weaknesses**: MLLMs perform poorly at distinguishing positive/negative/mixed valence, yet performance improves substantially through fine-tuning (full fine-tuning: +21.1%), suggesting the underlying issue is category boundary confusion rather than an absence of capability.
- **Perception subjectivity is a fundamental challenge**: Even full fine-tuning yields only a +4.8% improvement; humans achieve 89.6% while the best MLLM reaches only 75.0%, indicating this limitation is tied to intrinsic model properties.
- **INSETS automatic annotation achieves 90.6% accuracy**: 89.7% for correct statements and 91.5% for incorrect statements, validating the reliability of the pipeline.
- **No single model dominates across all dimensions**: GPT-4o achieves the highest overall accuracy, yet underperforms mPLUG-Owl3 on perception subjectivity (69.2% vs. 75.0%).

## Highlights & Insights

- **Elegant ESJ task design**: Reformulating a subjective open-ended problem as an objective binary classification task preserves evaluation depth (four dimensions) while eliminating answer ambiguity. This "statement verification" paradigm is transferable to any evaluation task with high subjectivity, such as aesthetics, humor, or sarcasm understanding.
- **"Low-cost, high-quality" paradigm of INSETS**: By combining multi-MLLM ensemble with a psychologically grounded taxonomic model, the pipeline constructs 462K annotated instances in approximately 115 person-hours—orders of magnitude more efficient than EMOTIC's 23,788 annotators. This "AI pre-annotation + human refinement" approach has broad applicability.
- **Actionable insights from the four dimensions**: The results reveal a meaningful distinction between capabilities that are amenable to adaptation (polarity recognition) and those requiring fundamental model improvements (subjectivity understanding), providing concrete directions for MLLM development.

## Limitations & Future Work

- **Skewed data distribution**: Positive-emotion images constitute 65.2% of the dataset, inheriting a social media bias from EmoSet, which may reduce the reliability of evaluations on negative emotions.
- **Limited evaluation granularity**: ESJ operates as a binary correct/incorrect judgment and cannot assess MLLMs' continuous perception of emotional intensity.
- **Implicit bias in automated role generation**: Viewer roles in the perception subjectivity dimension may encode demographic stereotypes.
- **Dynamic emotion not covered**: The benchmark evaluates only static single images; temporal emotional dynamics in video and multimodal emotion (combined with text/audio) are not addressed.

## Related Work & Insights

- **vs. EmoSet/FI**: Conventional benchmarks perform closed-set classification over 8 fixed categories, whereas this work employs statement judgment over 751 open-vocabulary labels, representing a qualitative leap in evaluation flexibility and granularity.
- **vs. EmoBench-M/EEmo-Bench**: These works expand task coverage but retain open-ended questions, failing to resolve answer ambiguity at the task-formulation level; ESJ eliminates ambiguity through the task structure itself.
- **vs. FABA-Bench**: Focuses on facial expressions and actions while neglecting deeper dimensions such as scene context and perceptual subjectivity.

## Rating

- Novelty: ⭐⭐⭐⭐ The ESJ task design and four-dimensional evaluation framework are innovative, though the core contribution is an evaluation methodology rather than a model architecture breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 19 MLLMs, 5 adaptation strategies, and 25 human participants, with comprehensive and in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ The logic is clear, the integration of psychological theory with technical design is natural, and the ethical discussion is thorough.
- Value: ⭐⭐⭐⭐ Establishes a new paradigm for visual emotion evaluation; the MVEI benchmark and INSETS-462k corpus offer practical value for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model](../../AAAI2026/multimodal_vlm/o3slm_open_weight_open_data_and_open_vocabulary_sketch-language_model.md)
- [\[CVPR 2026\] EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](../../CVPR2026/multimodal_vlm/emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)
- [\[ICLR 2026\] K-Sort Eval: Efficient Preference Evaluation for Visual Generation via Corrected VLM-as-a-Judge](k-sort_eval_efficient_preference_evaluation_for_visual_generation_via_corrected_.md)
- [\[AAAI 2026\] Towards Scalable Web Accessibility Audit with MLLMs as Copilots](../../AAAI2026/multimodal_vlm/towards_scalable_web_accessibility_audit_with_mllms_as_copilots.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)

</div>

<!-- RELATED:END -->
