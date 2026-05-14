---
title: >-
  [Paper Note] Style over Story: Measuring LLM Narrative Preferences via Structured Selection
description: >-
  [ACL 2026][Interpretability][narrative preference] This paper proposes a constrained-selection experimental paradigm to measure LLM narrative preferences. Using a library of 200 constraints constructed from narratologica…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "narrative preference"
  - "LLM bias"
  - "constrained selection"
  - "narratology"
  - "style preference"
date: 2026-05-08
content_hash: f144c0a0708bb815
---

# Style over Story: Measuring LLM Narrative Preferences via Structured Selection

**Conference**: ACL 2026
**arXiv**: [2510.02025](https://arxiv.org/abs/2510.02025)
**Code**: None
**Area**: Interpretability / Text Generation
**Keywords**: narrative preference, LLM bias, constrained selection, narratology, style preference

## TL;DR

This paper proposes a constrained-selection experimental paradigm to measure LLM narrative preferences. Using a library of 200 constraints constructed from narratological theory, six LLMs are evaluated across different instruction types, revealing that models systematically favor "Style" over content elements such as "Event," "Character," and "Setting."

## Background & Motivation

**Background**: Writers are increasingly exploring LLMs as creative aids; however, research suggests that LLM usage may reduce narrative plot diversity, collective creativity, and individual writing style. While existing studies on LLM preferences have identified political biases and personality traits, narrative preferences remain unexplored.

**Limitations of Prior Work**: (1) Existing narrative studies focus on analyzing generated outputs (e.g., plot coherence, linguistic complexity) and cannot directly characterize underlying narrative preferences. (2) Output analysis conflates preference with capability—a model's failure to produce a certain narrative type may reflect either dispreference or inability. (3) LLM-generated text exhibits notable stylistic uniformity, yet the underlying preference structure remains poorly understood.

**Key Challenge**: Without understanding LLMs' latent narrative preferences, it is impossible to distinguish "deliberate creative choices" from "systematic biases," which has significant implications for LLM-assisted writing practices.

**Goal**: To design a measurement method that isolates "preference" from "capability" and quantitatively characterizes the narrative preference structure of LLMs.

**Key Insight**: Having models select rather than generate—using structured selection tasks to isolate preferences, and employing narratological theory to construct an interpretable constraint library.

**Core Idea**: A constrained-selection paradigm—providing a narratologically grounded candidate constraint set, having models choose which constraints to apply, and using selection behavior as a proxy for preference.

## Method

### Overall Architecture

A library of 200 narrative constraints is constructed (4 elements × 5 categories × 10 constraints), with each constraint annotated with 1–3 axis attributes. Six LLMs perform selections under 3 instruction types (basic / quality-oriented / creativity-oriented) across 5 task conditions. Constraint order is randomized in each run to eliminate position effects. A total of 8,820 runs are conducted.

### Key Designs

1. **Narratology-Driven Constraint Library**:

    - Function: Provides a theoretically grounded, interpretable instrument for measuring narrative preferences.
    - Mechanism: Drawing on classical and contemporary narratological theory, narrative is decomposed into four core elements—Event (plot dynamics), Style (voice/tone/narration), Character (character agency), and Setting (space/context)—with 5 categories per element and 10 constraints per category. Constraints are standardized to 15–20 words, parallel grammatical structure, and matched conceptual granularity to reduce surface-level selection biases.
    - Design Motivation: Constraints must have a narratological foundation to yield interpretable preference structures; otherwise, selection behavior cannot be meaningfully analyzed.

2. **Multi-Condition Experimental Design**:

    - Function: Tests the stability and condition-sensitivity of preferences.
    - Mechanism: Five task conditions are employed—within-element free budget (1-1), within-element fixed budget (1-2), pooled unlabeled free (2-1), pooled unlabeled fixed (2-2, baseline), and element-blocked quota (3)—combined with 3 instruction types (basic / quality / creativity). The baseline condition is established through condition comparisons: pooled unlabeled fixed budget most closely approximates models' native preference structures.
    - Design Motivation: The multi-condition design disentangles preference from task-design artifacts—preferences that remain stable across conditions are more credible.

3. **Statistical Analysis Framework**:

    - Function: Rigorously quantifies and compares selection patterns.
    - Mechanism: Poisson GEE (with run-level clustering) is used to estimate element-level and category-level selection rate ratios (RR); $K$-weighted WLS is used to estimate condition contrasts. Axis richness is assessed via stratified permutation tests.
    - Design Motivation: Selection data have a count nature and clustered structure, making Poisson GEE the appropriate statistical model.

### Loss & Training

This is a pure inference experiment with no training involved. Six commercial LLMs are evaluated: GPT-4.1, GPT-5, o4-mini, Claude, Gemini, and Qwen.

## Key Experimental Results

### Main Results

**Element-Level Selection Rate Ratios (vs. Event baseline, Poisson GEE)**

| Element | RR [95% CI] | p |
|---------|-------------|---|
| Event (baseline) | 1.00 | — |
| **Style** | **1.78** [1.74, 1.82] | <.001 |
| Character | 0.98 [0.96, 1.01] | .160 |
| Setting | 1.28 [1.25, 1.31] | <.001 |

### Ablation Study

**Cross-Model Stability**

| Finding | Description |
|---------|-------------|
| Style preference | Consistently highest across all 6 models |
| GPT-4.1 specificity | Strongest Style preference; lowest rates for all other elements |
| Instruction sensitivity | Style is stable across instructions; content elements are affected by creativity-oriented instructions |

### Key Findings

- All LLMs systematically favor Style constraints, with a selection rate 78% higher than Event.
- Style preference is highly stable across models and instruction types, whereas content elements (Event/Character/Setting) exhibit greater cross-model variation and instruction sensitivity.
- GPT-4.1 acts as a "Style preference amplifier," occupying the extreme end across all comparisons.
- Creativity-oriented instructions alter axis-level distributions but not element-level rankings—Style consistently ranks first.
- Selection behavior is consistent with the stylistic uniformity observed in output analysis studies, confirming that LLMs hold a systematic preference for style.

## Highlights & Insights

- The paradigm innovation of "selection over generation" elegantly isolates preference from capability, filling a gap that output analysis cannot address.
- The "Style over Story" finding carries a practical warning for AI-assisted writing: if LLMs systematically favor style, AI-assisted literature may trend toward surface polish at the expense of narrative variety.
- The constraint library itself is a reusable research instrument that can be applied to evaluate the narrative preferences of any future LLM.

## Limitations & Future Work

- The relationship between selection preferences and actual generation behavior is not directly validated.
- Only commercial LLMs are evaluated; open-source models and cross-scale comparisons are not included.
- Although theoretically grounded, the constraint library remains a subjective design; alternative narratological frameworks may yield different categorizations.
- The sources of preference are not explored—it remains unclear whether the Style preference originates from training data bias or architectural properties.

## Related Work & Insights

- **vs. LLM preference measurement (Rozado 2024, political preferences)**: The latter operates in the political domain; this paper is the first to extend such measurement to the narrative domain.
- **vs. output analysis (Chakrabarty et al., 2024)**: The latter analyzes the quality of generated text, while this paper directly measures preferences through selection—the two approaches are complementary rather than mutually exclusive.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic measurement of LLM narrative preferences; both the paradigm and findings are original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 models × 3 instructions × 5 conditions × 8,820 runs, with rigorous statistical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The integration of narratological theory and computational experimentation is elegant.
- Value: ⭐⭐⭐⭐ Important implications for AI-assisted creative writing and LLM bias research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](../../ICLR2026/interpretability/semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)
- [\[NeurIPS 2025\] Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](../../NeurIPS2025/interpretability/deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)
- [\[ACL 2026\] A Structured Clustering Approach for Inducing Media Narratives](a_structured_clustering_approach_for_inducing_media_narratives.md)
- [\[ACL 2026\] SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings](soft_head_selection_for_injecting_icl-derived_task_embeddings.md)
- [\[ACL 2026\] LePREC: Reasoning as Classification over Structured Factors for Assessing Relevance of Legal Issues](leprec_reasoning_as_classification_over_structured_factors_for_assessing_relevan.md)

</div>

<!-- RELATED:END -->
