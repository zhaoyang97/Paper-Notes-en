---
title: >-
  [Paper Note] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Multi-image reasoning] This paper introduces OMIBench, the first large-scale benchmark for Olympiad-level multi-image reasoning. It covers over 1,000 competition problems across Biology…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multi-image reasoning"
  - "Olympiad-level reasoning"
  - "Vision-language model benchmark"
  - "Cross-image association"
  - "Scientific reasoning"
date: 2026-05-08
content_hash: 54e81d37395145b4
---

# OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.20806](https://arxiv.org/abs/2604.20806)  
**Code**: [GitHub](https://github.com/LightChen233/OMIBench)  
**Area**: Multimodal VLM / LLM Evaluation  
**Keywords**: Multi-image reasoning, Olympiad-level reasoning, Vision-language model benchmark, Cross-image association, Scientific reasoning

## TL;DR
This paper introduces OMIBench, the first large-scale benchmark for Olympiad-level multi-image reasoning. It covers over 1,000 competition problems across Biology, Chemistry, Mathematics, and Physics. Findings indicate that even the strongest LVLM (Gemini-3-Pro) achieves only approximately 50% accuracy, representing a drop of over 25% compared to single-image benchmarks.

## Background & Motivation

**Background**: LVLMs have made significant progress in standard reasoning tasks, and Chain-of-Thought (CoT) prompting has achieved major breakthroughs on single-image Olympiad benchmarks. Existing benchmarks like OlympiadBench are approaching saturation for top-tier models.

**Limitations of Prior Work**: (1) Existing Olympiad-level multimodal benchmarks are almost entirely limited to single-image settings, whereas many problems in real scientific competitions rely on multiple interconnected charts and experimental diagrams; (2) Existing multi-image benchmarks (e.g., MuirBench, MMIU) focus on perception and cross-image referencing but lack high difficulty and strong semantic/quantitative cross-image associations, making them insufficient for evaluating Olympiad-level reasoning; (3) There is a lack of expert-annotated reasoning paths, preventing in-depth analysis of specific failure points in the reasoning process.

**Key Challenge**: Olympiad-level multi-image reasoning requires models not only to understand individual images but also to (1) maintain the coherence of cross-image information flow and (2) perform deep cross-image, cross-modal reasoning. This represents a qualitative leap from perception to integrated reasoning that existing benchmarks cannot effectively evaluate.

**Goal**: Construct an Olympiad-level multi-image reasoning benchmark covering four major scientific disciplines, including expert-annotated rationales and various evaluation protocols to systematically expose the reasoning shortcomings of LVLMs in multi-image scenarios.

**Key Insight**: Collect real competition problems requiring joint multi-image reasoning from international and national science competitions, rather than using synthetic or simplified multi-image tasks.

**Core Idea**: Extend Olympiad-level reasoning evaluation from single-image to multi-image settings. When evidence is scattered across multiple images, the reasoning difficulty undergoes a qualitative change rather than just a quantitative increase.

## Method

### Overall Architecture
OMIBench contains 1,000+ Olympiad-level multi-image reasoning problems, with an average of 3.07 images per problem. It supports both multiple-choice and open-ended response formats. Each problem is paired with an expert-verified reasoning path (rationale), supporting both exact match and semantic equivalence evaluation modes. The data construction pipeline includes four stages: Data Collection & Filtering $\rightarrow$ Rationale Annotation $\rightarrow$ Quality Control $\rightarrow$ Categorical Labeling.

### Key Designs

1.  **Multi-image Competition Dataset Construction**:
    *   Function: Provide authentic Olympiad-level evaluation data requiring cross-image reasoning.
    *   Mechanism: Collect PDF exam papers from international Olympiads (IPhO, IChO, etc.), national/regional competitions, and mixed-complexity benchmarks. Use Mathpix OCR to convert to Markdown followed by manual verification. Filter for problems containing $\ge 2$ images providing joint reasoning evidence. Multilingual problems are translated via Google Translate then manually verified.
    *   Design Motivation: Ensure problem difficulty reaches competition standards and that non-trivial semantic/quantitative dependencies exist between multiple images.

2.  **Two-Stage Expert Rationale Annotation**:
    *   Function: Provide reference reasoning paths for analyzing model reasoning processes.
    *   Mechanism: Initially use Gemini-2.5-pro-thinking to generate up to 16 candidate solutions per problem, retaining those with correct answers (if all fail, provide the correct answer for re-generation, reducing human workload by ~20%). Then, annotators with competition experience verify and refine common solutions to ensure correct, complete, and standardized reasoning steps.
    *   Design Motivation: Most competition datasets lack step-by-step solutions, which are critical for diagnosing exactly where a model fails.

3.  **Dual Evaluation Protocol (Exact Match + GPTScore)**:
    *   Function: Evaluate both strict answer correctness and semantic equivalence.
    *   Mechanism: Exact Match (ACC) requires identical answers; GPTScore evaluates semantic equivalence of open-ended answers under multimodal context constraints, handling variations in expression with identical meanings.
    *   Design Motivation: Open-ended answers have multiple equivalent forms; relying solely on exact matching would underestimate the true capabilities of the models.

### Loss & Training
Ours is a pure benchmark work and does not involve model training.

## Key Experimental Results

### Main Results

| Model | Biology Score | Chemistry Score | Maths Score | Physics Score | Overall Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini-3-Pro | 71.31 | 25.35 | 62.56 | 38.92 | **50.53** |
| GPT-5 | 62.55 | 29.03 | 56.51 | 40.80 | 48.11 |
| GPT-5-mini | 59.36 | 24.42 | 56.74 | 43.63 | 47.73 |
| Qwen3-VL-32B | 58.57 | 20.74 | 40.70 | 25.00 | 35.78 |
| InternVL3-78B | 46.61 | 20.74 | 17.21 | 18.63 | 23.83 |

### Comparison with Single-Image Benchmarks

| Analysis | Data |
| :--- | :--- |
| Gemini-3-Pro: OlympiadBench $\rightarrow$ OMIBench | 75.67% $\rightarrow$ 50.53% ($\downarrow$ 25%+) |
| Model Ranking Correlation (Spearman $\rho$) | 0.614 < 0.7 (Moderate Correlation) |
| Manual Audit of o4-mini Reasoning Error Rate | 46% of critical steps contain logical errors |

### Key Findings
*   The strongest model Gemini-3-Pro reached only 50.53%, indicating that multi-image Olympiad reasoning remains a significant challenge.
*   From single-image to multi-image, model accuracy dropped by over 25%, and model rankings changed significantly ($\rho = 0.614$), suggesting that multi-image reasoning capability cannot be simply inferred from single-image capability.
*   A significant gap exists between closed-source and open-source models—Gemini-3-Pro is ~15% higher than the best open-source model, while GPT-4o is only comparable to open-source models, indicating that scale is not the sole determinant.
*   Long CoT, test-time scaling, and ICL provide limited but consistent improvements; parameter scaling and think-with-image methods yield minimal or even negative gains.
*   Chemistry and Physics are the most difficult (lowest scores), while Biology is the "easiest"—possibly because Biology problems rely more on knowledge retrieval than multi-step reasoning.

## Highlights & Insights
*   The **"qualitative change" thesis from single-image to multi-image** is supported by solid experimental evidence—the $\ge 25\%$ absolute drop and the re-ranking ($\rho = 0.614$) together show this is not a simple cumulative difficulty.
*   Manual audits revealed logical errors in 46% of critical reasoning steps—models can generate fluent reasoning chains that are logically flawed, serving as a warning for CoT evaluation methodologies.
*   The coverage of four disciplines allows the benchmark to reveal imbalances in reasoning capabilities across subjects, providing a reference for education and capability assessment.

## Limitations & Future Work
*   The dataset size is approximately 1,000 problems; some subject subsets may be small, limiting statistical power.
*   Dependence on GPTScore for semantic evaluation; the reliability of LLM-as-judge for identifying equivalence in mathematical/scientific answers requires further validation.
*   Fine-grained classification of multi-image dependency types (supplementary information, contradictory information, temporal changes, etc.) has not been performed.
*   Multimodal RAG or tool-enhanced strategies were not tested.
*   Problem sources lean towards International and Chinese competitions, which may introduce unfair bias toward models from specific cultural backgrounds.

## Related Work & Insights
*   **vs OlympiadBench (He et al., 2024)**: Both are competition-level, but OlympiadBench contains $<5\%$ multi-image problems. OMIBench is entirely multi-image, exposing capability deficiencies previously masked by single-image settings.
*   **vs MuirBench / MMIU**: These multi-image benchmarks have lower difficulty, lack competition-level reasoning, and provide no reasoning paths.
*   **vs ReMI (Kazemi et al., 2024)**: Covers Math and Physics but at H/COL difficulty levels, excludes Biology/Chemistry, and lacks reasoning annotations.

## Rating
*   Novelty: ⭐⭐⭐⭐ The combination of multi-image and Olympiad-level tasks is a new evaluation perspective, though the benchmark construction methodology is relatively standard.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 30+ models, analysis of various enhancement strategies, and systematic comparison with single-image benchmarks.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure and rich data.
*   Value: ⭐⭐⭐⭐ Fills a gap in evaluating multi-image Olympiad reasoning and serves as a reference for model capability analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion](lami_augmenting_large_language_models_via_late_multi-image_fusion.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)
- [\[ACL 2026\] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)

</div>

<!-- RELATED:END -->
