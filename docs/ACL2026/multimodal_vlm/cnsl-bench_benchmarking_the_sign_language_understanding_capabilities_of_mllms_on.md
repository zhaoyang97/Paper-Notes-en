---
title: >-
  [Paper Note] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language
description: >-
  [ACL 2026][Multimodal VLM][manual articulation] CNSL-bench is the first authoritative benchmark for evaluating Chinese sign language in MLLMs based on the *National Common Sign Language Dictionary*. It covers 6,707 unique sign entries across text, image, and video modalities, featuring three types of hand articulation (air-writing, finger-spelling, and manual-alphab
tags:
  - ACL 2026
  - Multimodal VLM
  - manual articulation
date: 2026-05-08
content_hash: 0f7887dd67f05d04
---
# CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language

**Conference**: ACL 2026  
**arXiv**: [2604.22367](https://arxiv.org/abs/2604.22367)  
**Code**: https://github.com/rzhao-zhsq/CNSL-bench  
**Area**: Multimodal VLM / Sign Language Understanding  
**Keywords**: Chinese National Sign Language, Sign Language Benchmark, MLLM Evaluation, Modality Imbalance, manual articulation

## TL;DR
CNSL-bench is the first authoritative benchmark for evaluating Chinese sign language in MLLMs based on the *National Common Sign Language Dictionary*. It covers 6,707 unique sign entries across text, image, and video modalities, featuring three types of hand articulation (air-writing, finger-spelling, and manual-alphabet), totaling 20,121 four-way multiple-choice questions. Evaluations across 21 SOTA MLLMs reveal that while GPT-5 achieves 89.6% in text, it drops to 67.0% in image and 56.7% in video—a significant gap compared to the 97% human performance. Furthermore, CoT reasoning provides minimal benefit for video understanding.

## Background & Motivation

**Background**: LLMs have propelled sign language research from pure SLR/SLT pipelines into the LLM-as-decoder phase (e.g., Sign2GPT, SignLLM). Recent MLLMs have further enhanced visual and video understanding. However, most work embeds LLMs into specific downstream tasks (translation, recognition) as semantic enhancement modules.

**Limitations of Prior Work**: **The intrinsic sign language understanding capability of MLLMs has never been systematically evaluated**. Existing sign language datasets (WLASL, PHOENIX, CSL-Daily, How2Sign, etc.) are designed for training specific tasks and lack cross-modal aligned lexical-level evaluation. General MLLM benchmarks (MME, MMMU, Video-MME, etc.) do not include sign language. Consequently, it remains unclear where MLLMs excel or fail in sign language and how large the modality gap truly is.

**Key Challenge**: Sign language is inherently a **multimodal language** involving visuospatial dynamics and linguistic structures. It requires both the visual perception of VLMs and the semantic understanding of LLMs. Existing evaluations either test vision alone (without semantic grounding) or language alone (without vision), failing to determine if MLLMs truly understand linguistic structures or merely process surface-level visual cues.

**Goal**: To construct the first sign language MLLM benchmark featuring **(1) authoritative lexical grounding, (2) multimodal alignment (text/image/video), and (3) articulation diversity (air-writing/finger-spelling/manual-alphabet)**, and to systematically evaluate 21 SOTA MLLMs.

**Key Insight**: Utilizing the *National Common Sign Language Dictionary*—the only official standard in China—anchors the evaluation to eliminate ambiguities from dialects or non-standard variations. This ensures **controllable, consistent, and reproducible** semantic references. By aligning each entry with videos from CNSL-DP (a 2025 dual-view dataset), a lexical-level benchmark fully aligned across three modalities is achieved.

**Core Idea**: The benchmark is constructed as a 4-way multiple-choice task based on the three principles of "authoritative lexical grounding + multimodal alignment + manual articulation subdivision." This transforms open-ended sign language understanding (currently beyond MLLM capabilities) into a controlled closed-form evaluation. Dense testing of 21 MLLMs across 3 modalities, 3 articulations, and 2 video frame rates reveals systematic failure modes in MLLM sign language understanding.

## Method

### Overall Architecture
The construction of CNSL-bench follows two main tracks:

- **Lexical grounding**: Starting from 8,214 glosses in the *National Common Sign Language Dictionary*, sign-level preprocessing is performed (merging different glosses with identical hand movements, splitting polysemous entries, and retaining variations) to obtain 6,707 unique sign entries.
- **Multimodal alignment**: Each sign entry is aligned across three modalities: (1) the original text description from the dictionary; (2) dictionary illustrations; (3) a representative video from CNSL-DP (24 fps, 512×512 center-cropped, signer centered).
- **Articulation Subsets**: 407 air-writing, 77 finger-spelling, and 592 manual-alphabet entries (based on the *Chinese Finger Spelling Scheme*) were manually labeled as dedicated subsets for fine-grained analysis.
- **Task Format**: 4-way multiple-choice questions. Each question provides one modal input, one correct answer, and three random distractors (semantic distractors were also tested with consistent findings).
- **Scale**: 6,707 entries × 3 modalities = 20,121 questions.
- **Evaluation**: Zero-shot testing of 21 MLLMs (including open-source LLaVA-NeXT, Qwen-VL, InternVL-3.5, GLM-4.1V and closed-source Qwen-Plus/Max, Gemini-2.5, GPT-4o/5) across text/image/video (2 fps & 10 fps) × AW/FS/MA/All. Fast/slow thinking was tested for reasoning-capable models.

### Key Designs

**1. Authoritative lexical grounding + sign-level deduplication alignment: Ensuring a unique standard answer.**

The primary noise in sign language evaluation stems from regional variations for the same meaning, leading to irreproducible benchmarks. CNSL-bench anchors truth in China's national standard (2018/2019 authorities) to eliminate variation ambiguity. Sign-level preprocessing addresses three redundancies: merging different glosses sharing the same gesture; splitting glosses with multiple meanings (e.g., "seatbelt" in cars vs. planes); and retaining multiple gesture variants for a single meaning.

**2. Manual articulation classification (AW / FS / MA): Identifying structural strengths and weaknesses.**

To diagnose specific failure points, articulation is categorized based on linguistics: **Air-writing (AW)** involves drawing shapes in the air, testing spatial trajectory tracking. **Finger-spelling (FS)** depicts the glyphs of Chinese characters, emphasizing graphic cues. **Manual-alphabet (MA)** maps gestures to Pinyin letters, requiring symbol recognition and combination.

**3. Multimodal alignment evaluation + modality gap quantification.**

Testing the same sign across text, image, and video modalities quantifies "modality dependency bias." The protocol uses one modality as input for each 4-way MCQ. Video is tested at 2 fps and 10 fps. A human baseline (experts and native signers) provides a reference point.

### Loss & Training
This work is a benchmark; no training was performed. Evaluation settings: temperature=0, max_tokens=2048. Reasoning models were tested at low, medium, and high execution levels.

## Key Experimental Results

### Main Results (21 MLLM Accuracy %)

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
| **Random** | 25.23 | 24.73 | 25.03 | 25.04 | 27.27 | 24.67 |
| **Human** | **96.93** | **97.39** | **97.39** | **97.39** | **97.40** | **98.70** |

### Ablation Study (Test-time scaling, Reasoning effort L/M/H)

| Model | Text-All | Image-All | Video-All | Description |
|-------|----------|-----------|-----------|------|
| GPT-5 (L) | 88.94 | 66.77 | 51.89 | Low reasoning |
| GPT-5 (M) | **89.64** | 66.96 | **53.42** | Med reasoning (Strongest) |
| GPT-5 (H) | 89.95 | **68.34** | 53.09 | High reasoning |
| Qwen3-VL-Plus slow | 76.22 | 42.41 | 35.34 | Slow thinking drops performance |

### Key Findings
- **MLLMs significantly trail humans**: The strongest GPT-5 scores 56.7% in video, while humans achieve 97.4% (a 41-point gap).
- **Strong modality imbalance**: Performance consistently drops from text to image to video. MLLMs rely heavily on linguistic priors rather than visual/temporal understanding.
- **Articulation disparity**: FS is consistently the easiest (similar to character OCR), while AW and MA are significantly harder due to reliance on spatial tracking and complex symbol combination.
- **CoT is ineffective for video**: Reasoning from low to high effort barely improves video accuracy (and sometimes reduces it), suggesting the bottleneck is visual perception, not secondary reasoning.
- **The gap between open and closed source is narrowing**: Models like InternVL-3.5 and GLM-4.1V approach GPT-4o levels on specific subsets.

## Highlights & Insights
- **Standardized Evaluation**: Transforms sign language understanding into a reproducible lexical MCQ task, providing a new paradigm for MLLM assessment.
- **Multidimensional Design**: Effectively decomposes model performance into modality, articulation, and reasoning gaps, offering clear directions for improvement (focusing on visual perception).
- **Reasoning Token Bias**: Models consume significantly more tokens reasoning about text than video, highlighting a dependency on "thinking in text."

## Limitations & Future Work
- **Ours**: (1) Limited to lexical-level; (2) covers only National Sign Language (excluding ASL/dialects); (3) MCQ format may not capture full open-ended nuances.
- **Future Work**: Scaling to sentence-level SLT, expanding to international sign languages, and investigating specialized vision encoders (hand-region attention/motion encoders) to bridge the modality gap.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hybrid Autoregressive-Diffusion Model for Real-Time Sign Language Production](hybrid_autoregressive-diffusion_model_for_real-time_sign_language_production.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)
- [\[ACL 2026\] GroupToM-Bench: Benchmarking Group Theory of Mind and Nonlinear Social Emergence in MLLMs](grouptom-bench_benchmarking_group_theory_of_mind_and_nonlinear_social_emergence_.md)
- [\[ACL 2026\] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding](vulca-bench_a_multicultural_vision-language_benchmark_for_evaluating_cultural_un.md)
- [\[ACL 2026\] AICA-Bench: Holistically Examining the Capabilities of VLMs in Affective Image Content Analysis](aica-bench_holistically_examining_the_capabilities_of_vlms_in_affective_image_co.md)

</div>

<!-- RELATED:END -->
