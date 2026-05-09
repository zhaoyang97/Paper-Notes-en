---
title: >-
  [Paper Note] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning
description: >-
  [ACL 2026][Multimodal VLM][ancient character evolution] This paper constructs a benchmark for ancient Chinese character evolution analysis comprising 11 tasks and 130,000+ instances, evaluates 19 MLLMs to reveal their limited capacity for glyph-level recognition and evolution reasoning, and proposes GEVO—a glyph-driven contrastive fine-tuning framework—that achieves consistent improvements across all tasks on a 2B-scale model.
tags:
  - ACL 2026
  - Multimodal VLM
  - ancient character evolution
  - multimodal large language models
  - glyph-driven contrastive fine-tuning
  - oracle bone script
  - curriculum learning
date: 2026-05-08
content_hash: d319c4104126fb7d
---

# Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning

**Conference**: ACL 2026
**arXiv**: [2604.11299](https://arxiv.org/abs/2604.11299)
**Code**: [https://github.com/songruiecho/GEVO](https://github.com/songruiecho/GEVO)
**Area**: Multimodal VLM / Digital Humanities
**Keywords**: ancient character evolution, multimodal large language models, glyph-driven contrastive fine-tuning, oracle bone script, curriculum learning

## TL;DR
This paper constructs a benchmark for ancient Chinese character evolution analysis comprising 11 tasks and 130,000+ instances, evaluates 19 MLLMs to reveal their limited capacity for glyph-level recognition and evolution reasoning, and proposes GEVO—a glyph-driven contrastive fine-tuning framework—that achieves consistent improvements across all tasks on a 2B-scale model.

## Background & Motivation

**Background**: With the rapid advancement of MLLMs, an increasing body of research has begun leveraging these models to analyze ancient scripts (e.g., oracle bone script, bronze inscriptions), demonstrating potential across tasks ranging from character recognition to artifact interpretation. The analysis of ancient Chinese character evolution—from oracle bone script to regular script (kaishu)—constitutes a foundational pathway for understanding cultural transformation and historical transmission.

**Limitations of Prior Work**: (1) No systematic benchmark exists for evaluating MLLMs on ancient character evolution analysis; (2) existing MLLMs perform poorly on cross-era font style recognition and ancient character identification; (3) while some studies have explored ancient scripts, how to systematically improve MLLMs on evolution analysis tasks remains an open problem.

**Key Challenge**: Ancient character evolution involves subtle glyph differences and cross-era structural changes. Existing MLLMs are predominantly trained on modern data and lack understanding of ancient glyph features. Nevertheless, limited fine-tuning can substantially improve era attribution performance, indicating that MLLMs have latent potential that requires targeted guidance.

**Goal**: (1) Construct a comprehensive benchmark for ancient Chinese character evolution analysis; (2) systematically evaluate the capability boundaries of existing MLLMs; (3) propose an effective fine-tuning method to enhance evolution analysis performance.

**Key Insight**: The observation that a small amount of fine-tuning can significantly improve era attribution motivates the design of a glyph-contrastive fine-tuning approach—training the model to distinguish subtle differences arising from both temporal periods and character identity within glyph variations.

**Core Idea**: Inspired by curriculum learning, the method constructs positive and negative glyph pairs and employs contrastive learning to guide the model in capturing glyph transformation patterns that underlie evolutionary consistency.

## Method

### Overall Architecture
The GEVO framework consists of two stages: (1) benchmark construction—covering 7,740 Chinese characters across five historical stages from oracle bone script to regular script, with nearly 30,000 facsimile images and 11 sub-tasks across three categories; (2) glyph-driven fine-tuning—constructing glyph contrastive data and applying curriculum learning to train the model to discriminate glyph differences from easy to hard.

### Key Designs

1. **Ancient Chinese Character Evolution Benchmark Construction**:

    - **Function**: Provides 11 sub-tasks and 130,000+ test instances for systematically evaluating MLLMs on evolution analysis.
    - **Mechanism**: The evolution process is divided into five stages—oracle bone script, bronze inscription, seal script, clerical script, and regular script. Three major task categories are defined: (T1) Basic Recognition—font style identification, era attribution; (T2) Glyph Understanding—image-level character recognition, structural analysis; (T3) Evolution Analysis—cross-era comparison, evolutionary path reasoning. All tasks are formulated as image-text mixed QA with text outputs.
    - **Design Motivation**: Existing ancient script benchmarks mostly focus on a single stage (e.g., oracle bone script) or a single task. This benchmark covers a complete evolutionary chain with multi-dimensional evaluation, enabling comprehensive identification of MLLM capability boundaries.

2. **Glyph-Driven Contrastive Fine-Tuning (GEVO)**:

    - **Function**: Guides the model via contrastive learning to capture evolutionary consistency and temporal differences within glyph transformations.
    - **Mechanism**: Positive glyph pairs are constructed from variants of the same character across different eras (capturing evolutionary consistency), while negative pairs consist of glyphs from different characters (learning to discriminate differences). A curriculum learning strategy is adopted, beginning with visually dissimilar easy pairs and progressively transitioning to harder pairs with subtle differences. The training objective combines glyph recognition and contrastive judgment tasks.
    - **Design Motivation**: Glyph changes in ancient character evolution are often subtle (e.g., stroke simplification, structural adjustment). Fine-tuning solely on recognition tasks may lead the model to learn surface-level features. Contrastive learning forces the model to attend to fine-grained differences, while curriculum learning prevents the training from being overwhelmed by difficult samples at the outset.

3. **Multi-Dimensional Evaluation Protocol**:

    - **Function**: Systematically assesses MLLM capabilities on evolution analysis across different granularities and dimensions.
    - **Mechanism**: The 11 sub-tasks span from single-image recognition to cross-era reasoning. Evaluation combines accuracy metrics with validation by domain experts. A total of 19 MLLMs are evaluated, ranging from 1B to 72B parameters, including closed-source models such as GPT-4o-mini and GPT-5-mini.
    - **Design Motivation**: Different tasks place distinct demands on models' visual understanding, knowledge reasoning, and cross-era association abilities. Multi-dimensional evaluation precisely identifies the strengths and weaknesses of MLLMs.

### Loss & Training
Standard cross-entropy loss is employed, combined with positive and negative sample construction for contrastive learning. Training data are ordered by the visual salience of glyph differences under the curriculum learning strategy. Fine-tuning is conducted on a 2B-scale model.

## Key Experimental Results

### Main Results (Evaluation of 19 MLLMs)

| Model | Avg. Score | Font Recognition (T1) | Character Recognition (T2) | Evolution Analysis (T3) |
|---|---|---|---|---|
| GPT-5-mini | 24.88 | Low | Near-zero (0.07) | Low |
| Gemini-3-Flash | 27.89 | Low | Near-zero | Low |
| Qwen2.5-VL-7B | **47.65** | Moderate | 23.51 | Moderate |
| Qwen2.5-VL-72B | 46.30+ | Moderate | 24.45 | Moderate |
| GEVO-2B | Across-the-board gains | Significant gain | Significant gain | Significant gain |

### Ablation Study

| Configuration | Effect | Note |
|---|---|---|
| GEVO (full) | Gains on all 11 tasks | Contrastive + curriculum learning |
| w/o curriculum learning | Partial gains diminished | Easy-to-hard ordering is beneficial |
| w/o contrastive learning | Limited gains | Recognition-only training is insufficient |
| Recognition fine-tuning only | Era attribution improves but reasoning is weak | Validates necessity of contrastive learning |

### Key Findings
- All existing MLLMs (including GPT-5-mini) perform poorly on ancient character evolution analysis, with average scores not exceeding 50.
- Character-level recognition (T2.1) is the biggest bottleneck for all models—almost all achieve near 0%.
- A surprising finding: limited fine-tuning significantly improves era attribution, whereas reasoning tasks require contrastive learning support.
- GEVO achieves consistent improvements across all 11 tasks on a 2B-scale model.
- Open-source 7B models (e.g., Qwen2.5-VL-7B) outperform larger closed-source models, potentially because safety restrictions in the latter impair performance on non-standard tasks.

## Highlights & Insights
- **Cultural value of the benchmark**: An AI evaluation benchmark covering the complete evolutionary chain from oracle bone script to regular script is itself a significant digital humanities contribution that can advance computational paleography.
- **Contrastive learning captures evolutionary consistency**: Using variants of the same character across different eras as positive pairs to learn evolutionary patterns is an approach generalizable to any visual task requiring cross-temporal or cross-style understanding.
- **Potential of small models**: A 2B model with targeted fine-tuning achieves improvements across all tasks, demonstrating that domain knowledge injection matters more than model scale.

## Limitations & Future Work
- The dataset covers only approximately 7,740 characters with documented evolutionary records; many characters have incomplete evolutionary paths.
- The absolute performance of the 2B model remains limited; validation on larger models is needed.
- The benchmark is primarily based on facsimile images rather than actual artifact photographs, which may create a gap with real-world ancient script recognition scenarios.
- Using evolutionary knowledge to assist in interpreting undeciphered characters has not been explored.

## Related Work & Insights
- **vs. TongGu-VL**: A VLM specifically designed for ancient scripts, but limited to 2B scale with weak evolution analysis capability. GEVO achieves more effective improvements through its fine-tuning strategy.
- **vs. traditional ancient script OCR**: CNN-based specialized recognition models lack reasoning and association capabilities. MLLMs possess this potential but require guided training.
- **vs. general VLM fine-tuning**: Standard SFT can improve recognition but is insufficient to support evolution reasoning. Contrastive learning provides an additional structured learning signal.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic MLLM benchmark for ancient character evolution; glyph-contrastive fine-tuning approach is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 19 models, 11 sub-tasks, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Benchmark construction pipeline is clearly described; evaluation results are analyzed in depth.
- Value: ⭐⭐⭐⭐ Unique contribution to digital humanities and ancient script research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](../../CVPR2026/multimodal_vlm/emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)

<!-- RELATED:END -->
