---
title: >-
  [Paper Note] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language
description: >-
  [ACL 2026][Multimodal VLM][Chinese National Sign Language] CNSL-bench is the first authoritative MLLM evaluation benchmark for Chinese sign language based on the *National Common Sign Language Dictionary*. It covers 6…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Chinese National Sign Language"
  - "Sign Language Benchmark"
  - "MLLM Evaluation"
  - "Modality Imbalance"
  - "manual articulation"
date: 2026-05-08
content_hash: 387c199427fa3554
---

# CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language

**Conference**: ACL 2026  
**arXiv**: [2604.22367](https://arxiv.org/abs/2604.22367)  
**Code**: https://github.com/rzhao-zhsq/CNSL-bench  
**Area**: Multimodal VLM / Sign Language Understanding  
**Keywords**: Chinese National Sign Language, Sign Language Benchmark, MLLM Evaluation, Modality Imbalance, manual articulation

## TL;DR
CNSL-bench is the first authoritative MLLM evaluation benchmark for Chinese sign language based on the *National Common Sign Language Dictionary*. It covers 6,707 unique sign entries across three modalities (text/image/video) and three types of hand articulation (air-writing/finger-spelling/manual-alphabet), totaling 20,121 four-way multiple-choice questions. Evaluations of 21 SOTA MLLMs reveal: GPT-5 achieves 89.6% in text, 67.0% in images, and 56.7% in video, still showing a significant gap compared to human performance (97%), with CoT reasoning providing minimal assistance for video understanding.

## Background & Motivation

**Background**: LLMs have propelled sign language research from pure SLR/SLT pipelines into the LLM-as-decoder stage (e.g., Sign2GPT, SignLLM). Recently, MLLMs have significantly enhanced visual and video understanding capabilities, but nearly all work embeds LLMs into specific downstream tasks (translation, recognition) as semantic enhancement modules.

**Limitations of Prior Work**: **The intrinsic sign language understanding capabilities of MLLMs have never been systematically evaluated.** Existing sign language datasets (WLASL, PHOENIX, CSL-Daily, How2Sign, etc.) are designed for training specific tasks and lack cross-modal aligned lexical evaluation. General MLLM benchmarks (MME, MMMU, Video-MME, etc.) do not include sign language. Consequently, it remains unclear where MLLMs excel or fail in sign language and how large the modality gap is.

**Key Challenge**: Sign language is inherently a **multimodal language** (visual-spatial + temporal dynamics + linguistic structure), requiring both the visual perception of VLMs and the semantic understanding of LLMs. Existing evaluations either test vision alone (without semantic grounding) or language alone (without vision), failing to determine whether MLLMs truly understand linguistic structures or merely perform surface-level visual matching.

**Goal**: To construct the first sign language MLLM evaluation benchmark featuring **(1) authoritative lexical grounding, (2) multimodal alignment (text/image/video), and (3) articulation diversity (air-writing/finger-spelling/manual-alphabet)**, and to systematically evaluate 21 SOTA MLLMs.

**Key Insight**: Starting from an official national standard—the *National Common Sign Language Dictionary* (jointly published by the Ministry of Education, State Language Commission, and China Disabled Persons' Federation). Using this as an anchor eliminates ambiguity from dialects or non-standard variations, providing **controllable, consistent, and reproducible** semantic references. Each entry is then aligned with videos from CNSL-DP (a 2025 dual-view sign language video dataset from Xiamen University), resulting in a perfectly aligned lexical-level benchmark across text, image, and video modalities.

**Core Idea**: Construct a 4-way multiple-choice benchmark based on the principles of "authoritative dictionary lexical grounding + multimodal alignment + manual articulation subdivision." This transforms open-ended sign language understanding (which current MLLMs cannot perform) into a controlled, closed-form evaluation. Evaluating a dense grid of 21 MLLMs across 3 modalities, 3 articulations, and 2 video frame rates reveals systematic failure modes of MLLMs in sign language understanding.

## Method

### Overall Architecture
The construction of CNSL-bench follows two main threads:

- **Lexical grounding**: Starting from 8,214 glosses in the *National Common Sign Language Dictionary*, sign-level preprocessing is performed (merging different glosses with identical hand movements, splitting polysemous entries with same gloss but different meanings/movements, and preserving variations of the same meaning). This results in 6,707 unique sign entries.
- **Multimodal alignment**: Each sign entry is aligned across three modalities: (1) original text descriptions from the dictionary; (2) dictionary illustrations; (3) a representative video selected from CNSL-DP (24 fps, $512 \times 512$ center crop, signer centered).
- **Articulation subset**: Manual annotation identifies 407 air-writing, 77 finger-spelling, and 592 manual-alphabet entries (based on the *Chinese Finger Spelling Scheme*) for fine-grained analysis.
- **Task format**: 4-way multiple-choice. Each question provides one modal input, one correct answer, and three random distractors (semantic distractors were also compared; conclusions remained consistent, but random distractors are more reproducible).
- **Scale**: 6,707 entries $\times$ 3 modalities = 20,121 questions.
- **Evaluation**: Zero-shot evaluation of 21 MLLMs (including open-source LLaVA-NeXT, Qwen-VL, InternVL-3.5, GLM-4.1V and closed-source Qwen-Plus/Max, Gemini-2.5, GPT-4o/5) across text/image/video (2 fps & 10 fps) $\times$ AW/FS/MA/All. Fast/slow thinking is tested on supported reasoning models.

### Key Designs

1. **Authoritative dictionary lexical grounding + sign-level deduplication and alignment**:
    - **Function**: Eliminates ambiguity introduced by dialects, regional variations, or instructional examples, ensuring each question has a **unique standard answer**, which is fundamental to benchmark reproducibility.
    - **Mechanism**: The *Common Lexicon of National Common Sign Language* (2018) and *National Common Sign Language Dictionary* (2019) serve as lexical ground truths. Pruning handles three types of redundancy: (i) different glosses sharing the same gesture (merged); (ii) a single gloss representing polysemy (e.g., "seatbelt" for car vs. airplane) (split); (iii) a single meaning with multiple gestural variations (all preserved). The final set contains 6,707 unique entries. On the video side, representative recordings are selected from CNSL-DP for each sign, ensuring a one-to-one correspondence between video and dictionary.
    - **Design Motivation**: The primary source of noise in sign language evaluation is the existence of multiple regional/dialectal forms for the same meaning, causing models to be incorrectly penalized. Dictionary grounding coupled with strict multimodal alignment removes this lexical ambiguity, distinguishing CNSL-bench from non-standardized benchmarks like WLASL or How2Sign.

2. **Manual articulation three-category fine-grained analysis (AW / FS / MA)**:
    - **Function**: Decomposes sign language articulation into three linguistically distinct subcategories, allowing the benchmark to locate which movements MLLMs excel at or struggle with, rather than providing only a single overall accuracy.
    - **Mechanism**: Three categories are defined based on sign linguistics: (a) **Air-writing (AW)**: Drawing shapes or character strokes in the air (e.g., drawing a $\lightning$ for "lightning rod"), requiring spatial trajectory tracking; (b) **Finger-spelling (FS)**: Depicting Chinese character structures with one or both hands (e.g., crossing hands to form "North"), emphasizing graphic cues rather than letter-by-letter spelling; (c) **Manual-alphabet (MA)**: Mapping gestures to Pinyin letters according to the *Chinese Finger Spelling Scheme* (e.g., "CO2" using C+O+2 gestures), requiring symbol recognition and compositional understanding. Specific subsets are evaluated independently.
    - **Design Motivation**: Experiments show vast difficulty gaps—all models perform best on FS (e.g., GPT-5 text 97.4%, video 53.3%), while AW and MA are significantly worse. Reason: FS is more "discrete and character-like," resembling OCR tasks. AW involves continuous trajectories and MA involves symbol combinations, both requiring sign-specific reasoning currently lacking in MLLMs.

3. **Three-modal alignment evaluation + quantization of text/image/video modality gaps**:
    - **Function**: Labels the "modality dependence bias" of MLLMs by testing the same sign entry across three modalities to see how performance varies for identical semantic content.
    - **Mechanism**: Each question uses only one modality as input (4-way MCQ). Video is tested at 2 fps and 10 fps to study temporal density impacts. A human baseline is established with a professor and three sign language students (including one Deaf individual). The evaluation reveals a massive modality gap: GPT-5 drops from 89.6% (text) to 67.0% (image) to 56.7% (video), whereas the human baseline remains stable at ~97% across all modalities.
    - **Design Motivation**: This serves as a probe to diagnose whether MLLMs have achieved true "multimodal alignment." High performance in text but low performance in video indicates reliance on linguistic priors, suggesting that visual/temporal understanding is far from adequate.

### Loss & Training
This work presents an evaluation benchmark and does not involve training. Evaluation parameters are fixed at temperature=0 and max_tokens=2048. Reasoning models are tested across low/medium/high reasoning effort levels.

## Key Experimental Results

### Main Results (21 MLLMs across modalities and articulations, accuracy %)

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

### Ablation Study (test-time scaling, Reasoning effort L/M/H)

| Model | Text-All | Image-All | Video-All | Notes |
|-------|----------|-----------|-----------|-------|
| GPT-5 (L) | 88.94 | 66.77 | 51.89 | Low reasoning |
| GPT-5 (M) | **89.64** | 66.96 | **53.42** | Med reasoning (Optimal) |
| GPT-5 (H) | 89.95 | **68.34** | 53.09 | High reasoning (text up, video down) |
| Gemini-2.5-Pro (L) | 81.32 | 58.09 | 48.83 | |
| Gemini-2.5-Pro (M) | 84.79 | 61.13 | 48.32 | |
| Gemini-2.5-Pro (H) | 84.84 | 61.92 | 48.17 | Video drops at High effort |
| Gemini-2.5-Flash fast | 73.04 | 43.57 | 36.62 | |
| Gemini-2.5-Flash slow | **79.95** | **51.62** | **42.28** | Largest gain +6.45% |
| Qwen3-VL-Plus fast | 76.68 | 43.69 | 33.74 | |
| Qwen3-VL-Plus slow | 76.22 | 42.41 | 35.34 | **Slow thinking decreases accuracy** |

### Key Findings
- **MLLMs significantly trail humans**: The strongest model, GPT-5, reaches only 56.7% in the video modality compared to 97.4% for humans—a gap of approximately 41 percentage points.
- **Strong modality imbalance**: All models consistently show a performance drop following the pattern: text >> image > video. The gap between text and video reaches 33 points for GPT-5 and is even larger for open-source models, suggesting MLLMs rely heavily on language priors.
- **Articulation imbalance**: FS (finger-spelling) is consistently the easiest category, while AW and MA are significantly harder. This indicates MLLMs can handle "discrete, character-like" gestures but fail at continuous spatial trajectories or symbolic compositions.
- **CoT is largely ineffective for video, showing boundary effects for top-tier models**: For Gemini-2.5-Pro and GPT-5, increasing reasoning effort from low to high results in stagnant or decreasing video accuracy. This suggests the bottleneck in sign language video understanding is visual perception rather than reasoning.
- **Reasoning tokens are heavily modality-biased**: Models consume significantly more reasoning tokens for text than for images or video, reflecting a "habit" of thinking in text without internalizing reasoning for visual modalities.
- **Incorrect answers have longer reasoning chains**: GPT-5-M's ratio of reasoning length for incorrect vs. correct answers in text is 2.89, matching human behavior (thinking longer for difficult questions), but this extra time does not translate to video accuracy.
- **Open-source vs. closed-source gap is narrowing**: GLM-4.1V-9B, InternVL-3.5-8B, and Qwen3-VL-8B approach or surpass GPT-4o-mini and Gemini-2.5-Flash in several subsets.

## Highlights & Insights
- **Formalized sign language understanding as a controlled lexical-level MCQ**: Avoids the failure-prone area of open-ended generation, making evaluation reproducible and establishing a new paradigm for sign language MLLM assessment.
- **Multi-dimensional design (3 principles + 3 modalities + 3 articulations)**: Provides not just an overall score, but a diagnostic breakdown of modality gaps and articulation weaknesses, pinpointing visual perception as the key area for improvement.
- **National dictionary as lexical truth**: This paradigm of using authoritative official sources for lexical grounding provides a template for evaluating MLLMs in other low-resource languages or specialized domains (law, medicine).
- **CoT failure and boundary effects reveal new directions**: The ineffectiveness of test-time scaling for video implies current multimodal reasoning is a serial combination of text reasoning and visual perception. Improvements must come from vision encoders or training data rather than just increased thinking time.
- **Quantification of modality token consumption**: Provides a new dimension for understanding how models allocate "thinking capacity" across modalities.

## Limitations & Future Work
- **Limitations**: (1) Only lexical-level (single words) are tested, not sentence-level SLT; (2) Coverage is limited to Chinese National Sign Language, excluding ASL or other regional dialects; (3) MCQ format may not fully capture nuances of open-ended understanding.
- **Additional Constraints**: (1) Random distractors may lead to overestimation of models compared to semantic distractors; (2) Frame rate studies are limited to 2 and 10 fps; (3) Human baseline group size is relatively small (n=4).
- **Future Work**: (1) Expansion to sentence-level open-ended SLT; (2) Cross-linguistic comparisons (ASL, BSL, JSL); (3) Using CNSL-bench data for sign-language-specific instruction tuning; (4) Improving vision encoders with hand-region attention or temporal motion encoding.

## Related Work & Insights
- **vs. WLASL / PHOENIX / CSL-Daily**: These are training datasets for recognition/translation and lack MLLM-friendly aligned evaluation protocols; CNSL-bench is a dedicated MLLM benchmark.
- **vs. MME / MMMU / Video-MME**: General benchmarks focus on everyday scenes and lack sign language content; CNSL-bench fills this professional modality gap.
- **vs. Sign2GPT / SignLLM / FLa-LLM**: These use LLMs as enhancers in a pipeline; this work evaluates intrinsic MLLM capability, focusing on upstream assessment.
- **vs. CLARITY / PRACTIQ**: CNSL-bench follows the modern "authoritative source + automatic construction + fine-grained evaluation" paradigm for specialized benchmarks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First MLLM benchmark for Chinese National Sign Language with original three-principle design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 21 MLLMs across multiple modalities, frame rates, and reasoning efforts with human baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline, structured findings, and effective visualization.
- Value: ⭐⭐⭐⭐⭐ Significant push for MLLM training, sign language AI, and low-resource language assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)
- [\[ACL 2026\] AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis](aica-bench_holistically_examining_the_capabilities_of_vlms_in_affective_image_co.md)
- [\[ACL 2026\] GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs](grouptom-bench_benchmarking_group_theory_of_mind_and_nonlinear_social_emergence_.md)
- [\[ACL 2026\] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)
- [\[AAAI 2026\] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](../../AAAI2026/multimodal_vlm/vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)

</div>

<!-- RELATED:END -->
