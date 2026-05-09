---
title: >-
  [Paper Note] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Multi-image reasoning] This paper presents OMIBench — the first large-scale benchmark for olympiad-level multi-image reasoning, covering 1,000+ competition problems across biology, chemistry, mathematics, and physics. Even the strongest LVLM (Gemini-3-Pro) achieves only ~50% accuracy, a drop of more than 25% compared to single-image benchmarks.
tags:
  - ACL 2026
  - Multimodal VLM
  - Multi-image reasoning
  - Olympiad-level reasoning
  - Vision-language model benchmark
  - Cross-image association
  - Scientific reasoning
date: 2026-05-08
content_hash: 3d9f9567a9a2dc4a
---

# OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models

**Conference**: ACL 2026
**arXiv**: [2604.20806](https://arxiv.org/abs/2604.20806)
**Code**: [GitHub](https://github.com/LightChen233/OMIBench)
**Area**: Multimodal VLM / LLM Evaluation
**Keywords**: Multi-image reasoning, Olympiad-level reasoning, Vision-language model benchmark, Cross-image association, Scientific reasoning

## TL;DR
This paper presents OMIBench — the first large-scale benchmark for olympiad-level multi-image reasoning, covering 1,000+ competition problems across biology, chemistry, mathematics, and physics. Even the strongest LVLM (Gemini-3-Pro) achieves only ~50% accuracy, a drop of more than 25% compared to single-image benchmarks.

## Background & Motivation

**Background**: LVLMs have made significant progress on standard reasoning tasks, with chain-of-thought (CoT) prompting achieving major breakthroughs on single-image olympiad benchmarks. Existing benchmarks such as OlympiadBench are approaching saturation by top models.

**Limitations of Prior Work**: (1) Existing olympiad-level multimodal benchmarks are almost entirely limited to single-image settings, whereas real scientific competitions frequently feature problems that require reasoning across multiple interrelated figures and experimental diagrams. (2) Existing multi-image benchmarks (e.g., MuirBench, MMIU) focus on perception and cross-image reference but are of limited difficulty and lack strong semantic/quantitative cross-image dependencies, making them insufficient for evaluating olympiad-level reasoning. (3) Expert-annotated reasoning paths are absent, precluding fine-grained analysis of where model reasoning fails.

**Key Challenge**: Olympiad-level multi-image reasoning requires models not only to understand individual images, but also to (1) maintain coherent information flow across images and (2) perform deep cross-image, cross-modal reasoning — a qualitative leap from perception to integrated reasoning that existing benchmarks cannot effectively evaluate.

**Goal**: To construct an olympiad-level multi-image reasoning benchmark spanning four major scientific disciplines, with expert-annotated reasoning paths and multiple evaluation protocols, systematically exposing the reasoning deficiencies of LVLMs in multi-image settings.

**Key Insight**: Real competition problems requiring joint multi-image reasoning are collected from international and national subject olympiads, rather than synthetic or simplified multi-image tasks.

**Core Idea**: Extending olympiad-level reasoning evaluation from single-image to multi-image settings — when evidence is distributed across multiple images, the reasoning difficulty undergoes a qualitative rather than merely quantitative change.

## Method

### Overall Architecture
OMIBench contains 1,000+ olympiad-level multi-image reasoning problems, with an average of 3.07 images per problem. Both multiple-choice and open-ended answer formats are supported. Each problem is accompanied by an expert-verified reasoning rationale, and two evaluation modes are supported: exact match and semantic equivalence. The data construction pipeline consists of four stages: data collection and filtering → reasoning path annotation → quality control → category labeling.

### Key Designs

1. **Multi-Image Competition Problem Dataset Construction**:

    - **Function**: Provides authentic, olympiad-level evaluation data requiring cross-image reasoning.
    - **Mechanism**: PDF problems are collected from international olympiads (IPhO, IChO, etc.), national/regional competitions, and mixed-difficulty benchmarks, converted to Markdown via Mathpix OCR, and manually verified. Selection criteria require each problem to contain ≥2 images that jointly provide reasoning evidence. Multilingual problems are first translated using Google Translate and then verified by human annotators.
    - **Design Motivation**: To ensure problem difficulty genuinely reaches competition level and that non-trivial semantic/quantitative dependencies exist among multiple images.

2. **Two-Stage Expert Reasoning Path Annotation**:

    - **Function**: Provides reference reasoning paths for each problem that can be used to analyze model reasoning processes.
    - **Mechanism**: Gemini-2.5-pro-thinking is first used to generate up to 16 candidate solutions per problem; solutions with correct answers are retained (if all fail, the correct answer is provided for regeneration, reducing manual annotation effort by ~20%). Annotators with competition experience then verify and revise the solutions to ensure reasoning steps are correct, complete, and well-structured.
    - **Design Motivation**: Most competition datasets lack solution processes, yet reasoning paths are critical for diagnosing exactly where model reasoning fails.

3. **Dual Evaluation Protocol (Exact Match + GPTScore)**:

    - **Function**: Simultaneously evaluates strict answer correctness and semantic equivalence.
    - **Mechanism**: Exact match (ACC) requires complete answer agreement; GPTScore evaluates semantic equivalence of open-ended answers under multimodal context constraints, handling cases where expressions differ but meanings are equivalent.
    - **Design Motivation**: Open-ended answers can be expressed in multiple equivalent forms; relying solely on exact match underestimates true model capability.

### Loss & Training
This is a pure benchmarking work; no model training is involved.

## Key Experimental Results

### Main Results

| Model | Bio Score | Chem Score | Math Score | Phys Score | Overall Score |
|-------|-----------|------------|------------|------------|---------------|
| Gemini-3-Pro | 71.31 | 25.35 | 62.56 | 38.92 | **50.53** |
| GPT-5 | 62.55 | 29.03 | 56.51 | 40.80 | 48.11 |
| GPT-5-mini | 59.36 | 24.42 | 56.74 | 43.63 | 47.73 |
| Qwen3-VL-32B | 58.57 | 20.74 | 40.70 | 25.00 | 35.78 |
| InternVL3-78B | 46.61 | 20.74 | 17.21 | 18.63 | 23.83 |

### Comparison with Single-Image Benchmarks

| Analysis | Result |
|----------|--------|
| Gemini-3-Pro: OlympiadBench → OMIBench | 75.67% → 50.53% (↓25%+) |
| Model ranking correlation (Spearman ρ) | 0.614 < 0.7 (moderate correlation) |
| Manual review: error rate in o4-mini reasoning steps | 46% of key steps contain logical errors |

### Key Findings
- The strongest model, Gemini-3-Pro, reaches only 50.53%, demonstrating that multi-image olympiad reasoning remains an enormous challenge.
- Accuracy drops by more than 25% moving from single-image to multi-image settings, and model rankings shift substantially (ρ = 0.614), indicating that multi-image reasoning ability cannot be simply inferred from single-image performance.
- A significant gap exists between closed-source and open-source models — Gemini-3-Pro outperforms the best open-source model by ~15%, yet GPT-4o is on par with open-source models, suggesting that scale alone is not the decisive factor.
- Long CoT, test-time scaling, and ICL yield limited but consistent improvements; parameter scaling and think-with-image approaches offer negligible or even negative gains.
- Chemistry and physics are the most challenging disciplines (lowest scores), while biology is the most "accessible" — possibly because biology problems rely more on knowledge recall than multi-step reasoning.

## Highlights & Insights
- The claim of a **qualitative shift from single-image to multi-image reasoning** is strongly supported by experimental evidence — an absolute drop of 25%+ and rank reordering (ρ = 0.614) together demonstrate that this is not a simple additive difficulty increase.
- Manual review reveals that 46% of key reasoning steps contain logical errors — models can produce fluent reasoning chains that are nonetheless logically flawed, which serves as an important warning for CoT evaluation methodology.
- Coverage across four disciplines allows the benchmark to reveal imbalances in reasoning capability across subjects, which is informative for education and capability assessment.

## Limitations & Future Work
- The dataset contains ~1,000 problems; subsets for some disciplines may be too small for sufficient statistical power.
- Semantic evaluation relies on GPTScore; the reliability of LLM-as-judge for assessing equivalence of mathematical/scientific answers remains to be validated.
- The types of cross-image dependencies (complementary information, contradictory information, temporal change, etc.) are not categorized at a fine-grained level.
- Multimodal RAG or tool-augmented strategies have not been tested.
- Problem sources are biased toward international and Chinese competitions, which may introduce unfair bias for models from certain cultural backgrounds.

## Related Work & Insights
- **vs. OlympiadBench (He et al., 2024)**: Both are competition-level benchmarks, but fewer than 5% of OlympiadBench problems involve multiple images; OMIBench is entirely multi-image, exposing capability deficiencies previously obscured by single-image settings.
- **vs. MuirBench / MMIU**: These multi-image benchmarks are of low difficulty, lack competition-level reasoning, and provide no reasoning path annotations.
- **vs. ReMI (Kazemi et al., 2024)**: Covers mathematics and physics but at H/COL difficulty level, excludes biology and chemistry, and provides no reasoning annotations.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of multi-image and olympiad-level evaluation represents a novel assessment angle, though the benchmark construction methodology is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 30+ models, analysis of multiple augmentation strategies, and systematic comparison with single-image benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich data presentation.
- Value: ⭐⭐⭐⭐ Fills the gap in multi-image olympiad reasoning evaluation and provides meaningful reference for model capability analysis.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] Benchmarking Deflection and Hallucination in Large Vision-Language Models](benchmarking_deflection_and_hallucination_in_large_vision-language_models.md)
- [\[ACL 2026\] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](lami_augmenting_large_language_models_via_late_multi-image_fusion.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] Position: Multimodal Large Language Models Can Significantly Advance Scientific Reasoning](position_multimodal_large_language_models_can_significantly_advance_scientific_r.md)

<!-- RELATED:END -->
