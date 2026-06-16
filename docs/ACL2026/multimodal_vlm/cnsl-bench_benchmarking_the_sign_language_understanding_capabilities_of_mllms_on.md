---
title: >-
  [Paper Note] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language
description: >-
  [ACL 2026][Multimodal VLM][manual articulation] CNSL-bench is the first authoritative evaluation benchmark for Chinese Sign Language (CSL) MLLMs based on the *National Common Sign Language Dictionary*. It covers 6,707 unique sign entries across text, image, and video modalities, combined with three types of manual articulation (air-writing, finger-spelling, and manu
tags:
  - ACL 2026
  - Multimodal VLM
  - manual articulation
date: 2026-05-08
content_hash: 29c518b7b83f0a79
---
# CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language

**Conference**: ACL 2026  
**arXiv**: [2604.22367](https://arxiv.org/abs/2604.22367)  
**Code**: https://github.com/rzhao-zhsq/CNSL-bench  
**Area**: Multimodal VLM / Sign Language Understanding  
**Keywords**: Chinese National Sign Language, Sign Language Benchmark, MLLM Evaluation, Modality Imbalance, manual articulation

## TL;DR
CNSL-bench is the first authoritative evaluation benchmark for Chinese Sign Language (CSL) MLLMs based on the *National Common Sign Language Dictionary*. It covers 6,707 unique sign entries across text, image, and video modalities, combined with three types of manual articulation (air-writing, finger-spelling, and manual-alphabet), totaling 20,121 four-way multiple-choice questions. Evaluations of 21 SOTA MLLMs reveal that while GPT-5 achieves 89.6% in text, 67.0% in image, and 56.7% in video, a substantial gap remains compared to the 97% human baseline, and CoT reasoning provides negligible improvements for video understanding.

## Background & Motivation

**Background**: LLMs have transitioned sign language research from traditional SLR/SLT pipelines into the LLM-as-decoder phase (e.g., Sign2GPT, SignLLM). Although recent MLLMs have demonstrated strong visual and video understanding capabilities, most existing works embed LLMs into specific downstream tasks (such as translation or recognition) as semantic enhancement modules.

**Limitations of Prior Work**: **The intrinsic sign language understanding capabilities of MLLMs have never been systematically evaluated**. Existing sign language datasets (WLASL, PHOENIX, CSL-Daily, How2Sign, etc.) are primarily designed for training specific tasks and lack aligned lexical-level evaluation across modalities. Furthermore, general MLLM benchmarks (MME, MMMU, Video-MME, etc.) do not include sign language content. Consequently, the specific strengths, weaknesses, and modality gaps of MLLMs in sign language remain unknown.

**Key Challenge**: Sign language is inherently a **multimodal language** (spatio-temporal dynamics + linguistic structure), requiring both visual perception from VLMs and semantic reasoning from LLMs. Current evaluations either focus exclusively on vision without semantic grounding or on language without visual input, failing to determine whether MLLMs truly understand linguistic structures or merely exploit surface-level visual correlations.

**Goal**: To construct the first sign language MLLM evaluation benchmark featuring **(1) authoritative lexical grounding, (2) multimodal alignment (text/image/video), and (3) articulation diversity (air-writing/finger-spelling/manual-alphabet)**, and to systematically evaluate 21 SOTA MLLMs.

**Key Insight**: By utilizing the *National Common Sign Language Dictionary* (jointly released by the Ministry of Education, State Language Commission, and China Association of the Deaf), the benchmark anchors to the only official sign language standard in China. This approach eliminates ambiguities from dialects or non-standard variants, providing **controllable, consistent, and reproducible** semantic references. Aligning each entry with the CNSL-DP dataset (Xiamen University 2025) ensures a perfectly aligned lexical-level benchmark across three modalities.

**Core Idea**: A 4-way multiple-choice benchmark is constructed based on the principles of "authoritative lexical grounding, multimodal alignment, and manual articulation categorization." This converts open-ended sign language understanding—currently unattainable for MLLMs—into a controlled closed-form evaluation. Comprehensive testing across 21 MLLMs, multiple modalities, and various frame rates reveals systematic failure patterns in sign language understanding.

## Method

### Overall Architecture
The construction of CNSL-bench follows two primary tracks:

- **Lexical grounding**: Starting from 8,214 glosses in the *National Common Sign Language Dictionary*, sign-level preprocessing is performed (merging different glosses with identical gestures, splitting polysemous entries with different gestures, and retaining stylistic variants), resulting in 6,707 unique sign entries.
- **Multimodal alignment**: Each sign entry is aligned across three modalities: (1) original text descriptions from the dictionary; (2) dictionary illustrations; (3) representative video segments from CNSL-DP (24 fps, 512×512 center-cropped, centered signer).
- **Articulation subsets**: A total of 407 air-writing (AW), 77 finger-spelling (FS), and 592 manual-alphabet (MA) entries (based on the *Chinese Finger Spelling Scheme*) were manually annotated for fine-grained analysis.
- **Task format**: A 4-way multiple-choice format is used. Each question provides one modal input, one correct answer, and three random distractors.
- **Scale**: 6,707 entries × 3 modalities = 20,121 questions.
- **Evaluation**: Zero-shot evaluations are conducted on 21 MLLMs (including LLaVA-NeXT, Qwen-VL, InternVL-3.5, GLM-4.1V, Gemini-2.5, and GPT-5) across all modalities and articulations. Fast and slow thinking modes are tested for reasoning-supported models.

### Key Designs

**1. Authoritative dictionary lexical grounding + sign-level deduplication alignment**: To ensure each question has a unique standard answer, the benchmark is anchored to China's national sign language standards. Preprocessing addresses three types of redundancy: merging different glosses sharing the same gesture, splitting identical glosses with multiple meanings/gestures, and retaining synonymous variants.

**2. Manual articulation classification (AW / FS / MA)**: Articulation is categorized into three types based on sign language linguistics. **Air-writing (AW)** involves drawing shapes or strokes in the air, testing spatial trajectory tracking. **Finger-spelling (FS)** depicts the shapes of Chinese characters, emphasizing graphic cues. **Manual-alphabet (MA)** maps gestures to Pinyin letters, requiring symbolic recognition. This categorization reveals that while models perform best on FS (which resembles OCR), they struggle with the spatial trajectories of AW and the symbolic combinations of MA.

**3. Multimodal alignment evaluation + modality gap quantification**: By evaluating the same sign entry across text, image, and video, the "modality dependency bias" can be quantified. Results show that GPT-5 drops from 89.6% (text) to 56.7% (video), a 33% decrease, while humans maintain near-constant performance (~97%) across all modalities. This indicates that MLLMs rely heavily on linguistic priors, with significant deficiencies in spatio-temporal visual understanding.

## Key Experimental Results

### Main Results (Accuracy % for 21 MLLMs across modalities and articulations)

| Model | Text-All | Image-All | Video 2fps-All | Video 10fps-All | FS-Text | FS-Video10 |
|-------|----------|-----------|-----------------|------------------|---------|------------|
| **GPT-5 (M, slow)** | **89.64** | **66.96** | **53.42** | **56.72** | **97.40** | **53.25** |
| Gemini-2.5-Pro (M, slow) | 84.79 | 61.13 | 48.32 | 48.35 | 94.81 | 35.06 |
| Gemini-2.5-Flash (slow) | 79.95 | 51.62 | 42.28 | 42.63 | 93.51 | 32.47 |
| Qwen3-VL-Plus (slow) | 76.22 | 42.41 | 35.34 | 36.92 | 89.61 | 18.18 |
| GPT-4o | 69.03 | 39.07 | 31.26 | 28.43 | 88.31 | 23.38 |
| InternVL-3.5-8B | 67.53 | 38.36 | 32.26 | 33.59 | 83.12 | 36.36 |
| Qwen3-VL-8B-Instruct | 67.06 | 38.39 | 30.94 | 33.89 | 79.22 | 24.68 |
| GLM-4.1V-9B (slow) | 68.24 | 39.62 | 28.03 | 29.75 | 84.42 | 24.68 |
| Qwen2.5-VL-3B | 60.07 | 34.26 | 28.36 | 30.34 | 72.73 | 28.57 |
| Qwen2-VL-2B | 43.36 | 30.62 | 27.23 | 27.58 | 51.95 | 18.18 |
| LLaVA-NeXT-Video-7B | 1.34 | 12.94 | 15.43 | 15.91 | 2.60 | 14.29 |
| **Random** | 25.23 | 24.73 | 25.03 | 25.04 | 27.27 | 24.67 |
| **Human** | **96.93** | **97.39** | **97.39** | **97.39** | **97.40** | **98.70** |

### Ablation Study (Test-time scaling, Reasoning effort L/M/H)

| Model | Text-All | Image-All | Video-All | Notes |
|-------|----------|-----------|-----------|-------|
| GPT-5 (L) | 88.94 | 66.77 | 51.89 | Low reasoning |
| GPT-5 (M) | **89.64** | 66.96 | **53.42** | Mid reasoning (Best) |
| GPT-5 (H) | 89.95 | **68.34** | 53.09 | High reasoning |
| Gemini-2.5-Pro (L) | 81.32 | 58.09 | 48.83 | |
| Gemini-2.5-Pro (M) | 84.79 | 61.13 | 48.32 | |
| Gemini-2.5-Pro (H) | 84.84 | 61.92 | 48.17 | High reasoning declines in video |
| Gemini-2.5-Flash fast | 73.04 | 43.57 | 36.62 | |
| Gemini-2.5-Flash slow | **79.95** | **51.62** | **42.28** | Max avg gain +6.45% |
| Qwen3-VL-Plus fast | 76.68 | 43.69 | 33.74 | |
| Qwen3-VL-Plus slow | 76.22 | 42.41 | 35.34 | **Slow thinking decreases performance** |

### Key Findings
- **Significant gap between MLLMs and humans**: The best-performing model, GPT-5, reaches only 56.7% in the video modality compared to 97.4% for humans, representing a 41-point gap.
- **Strong modality imbalance**: Performance follows a consistent text >> image > video trend across all models. This suggests that "multimodal alignment" in MLLMs is incomplete, with models relying heavily on linguistic priors.
- **Uneven articulation performance**: Finger-spelling (FS) is consistently the easiest category, while AW and MA are significantly harder, indicating that models struggle with continuous spatial trajectories and symbolic combinations.
- **CoT is largely ineffective for video**: For top-tier models like Gemini-2.5-Pro and GPT-5, increasing reasoning effort does not improve video accuracy. This suggests the bottleneck in sign language understanding lies in visual perception rather than reasoning.
- **Modality-biased reasoning tokens**: Models consume significantly more reasoning tokens for text than for images or videos, reflecting a "textual thinking" habit.

## Highlights & Insights
- **Formalization of sign language understanding**: By converting sign language comprehension into a controlled lexical-level MCQ benchmark, the study provides a reproducible paradigm for evaluating MLLMs.
- **Dimensional diagnosis**: The benchmark's multi-dimensional design allows for the isolation of "modality gaps," "articulation gaps," and "reasoning effects," providing clear directions for future MLLM improvements (specifically, prioritizing visual perception).
- **Authoritative grounding**: Using national standard dictionaries as the lexical truth serves as a model for evaluating MLLMs in other low-resource or professional domains.
- **Reasoning token perspective**: The quantification of token consumption across modalities reveals a strong textual bias, offering new dimensions for training multimodal alignment goals.

## Limitations & Future Work
- **Ours**: The study is limited to lexical-level evaluation, omitting sentence-level SLT. It covers only Chinese National Sign Language, excluding ASL or regional dialects.
- **Additional limitations**: Random distractors may simplify the task; the study did not systematically explore the full frame rate-accuracy curve or continuous sequences. The human baseline was limited to four participants.
- **Future Work**: Expansion to sentence-level evaluation, cross-linguistic comparisons (ASL, BSL), and improving the vision encoder (e.g., through hand-region attention or temporal motion encoders).

## Related Work & Insights
- **Comparison to SLR/SLT datasets**: Unlike WLASL or CSL-Daily, which are training-centric, CNSL-bench is a dedicated MLLM evaluation benchmark.
- **Comparison to general MLLM benchmarks**: CNSL-bench fills the gap left by benchmarks like MMMU or Video-MME, which lack professional sign language content.
- **Relationship to recent LLM-SLR works**: While works like Sign2GPT use LLMs as enhancers, CNSL-bench assesses the intrinsic capabilities of MLLMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)
- [\[ACL 2026\] GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs](grouptom-bench_benchmarking_group_theory_of_mind_and_nonlinear_social_emergence_.md)
- [\[ACL 2026\] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)
- [\[ACL 2026\] AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis](aica-bench_holistically_examining_the_capabilities_of_vlms_in_affective_image_co.md)
- [\[CVPR 2026\] IF-Bench: Benchmarking and Enhancing MLLMs for Infrared Images with Generative Visual Prompting](../../CVPR2026/multimodal_vlm/if-bench_benchmarking_and_enhancing_mllms_for_infrared_images_with_generative_vi.md)

</div>

<!-- RELATED:END -->
