---
title: >-
  [Paper Note] Style over Story: Measuring LLM Narrative Preferences via Structured Selection
description: >-
  [ACL 2026][Interpretability][Narrative Preferences] This paper designs an experimental paradigm based on constrained selection to measure the narrative preferences of LLMs. Using a library of 200 constraints constructed…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Narrative Preferences"
  - "LLM Bias"
  - "Constrained Selection"
  - "Narratology"
  - "Style Preferences"
date: 2026-05-08
content_hash: 27e0d5fab665d25b
---

# Style over Story: Measuring LLM Narrative Preferences via Structured Selection

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.02025](https://arxiv.org/abs/2510.02025)  
**Code**: None  
**Area**: Interpretability / Text Generation  
**Keywords**: Narrative Preferences, LLM Bias, Constrained Selection, Narratology, Style Preferences

## TL;DR

This paper designs an experimental paradigm based on constrained selection to measure the narrative preferences of LLMs. Using a library of 200 constraints constructed from narratological theory, 6 LLMs were tested under various instruction types. The study finds that models systematically prioritize "Style" over content elements such as "Event," "Character," and "Setting."

## Background & Motivation

**Background**: Novelists have begun exploring the use of LLMs for collaborative writing. However, research suggests that LLM usage may reduce narrative plot diversity, collective creativity, and individual writing styles. While existing LLM preference research has identified political biases and personality traits, narrative preferences remain unexplored.

**Limitations of Prior Work**: (1) Existing narrative studies focus on analyzing generated outputs (e.g., plot coherence, linguistic complexity), which cannot directly characterize underlying narrative preferences; (2) Output analysis confuses preference with capability—a model might not generate a certain narrative because it lacks the ability, not necessarily because it lacks the preference; (3) LLM-generated texts exhibit significant stylistic uniformity, but the underlying preference structure is poorly understood.

**Key Challenge**: Without understanding the latent narrative preferences of LLMs, it is impossible to distinguish between "deliberate creative choices" and "systematic biases," which has significant implications for AI-assisted writing practices.

**Goal**: To design a measurement method that isolates "preference" from "capability" and quantitatively characterizes the structure of narrative preferences in LLMs.

**Key Insight**: Let models select rather than generate—isolate preferences through structured selection tasks and use narratological theory to construct an interpretable library of constraints.

**Core Idea**: The constrained selection paradigm—providing a candidate set of constraints driven by narratological theory and allowing the model to choose which constraints to apply, using selection behavior as a proxy for preference.

## Method

### Overall Architecture

A library of 200 narrative constraints was constructed (4 elements × 5 categories × 10 constraints), with each constraint labeled with 1-3 axial attributes. Six LLMs were tested across 3 instruction types (Base/Quality-oriented/Creativity-oriented) and 5 task conditions. Constraint order was randomized in each run to eliminate position effects, totaling 8,820 runs.

### Key Designs

1.  **Narratology-Driven Constraint Library**:
    *   **Function**: Provides a theoretically grounded, interpretable tool for measuring narrative preferences.
    *   **Mechanism**: Based on classical and contemporary narratological theories, narratives are divided into four core elements: Event (plot dynamics), Style (voice/tone/narration), Character (character agency), and Setting (space/context). Each element contains 5 categories with 10 constraints each. Constraints are standardized to 15-20 words with parallel syntax and matched conceptual granularity to reduce surface selection bias.
    *   **Design Motivation**: Constraints must have a basis in narratological theory to produce an interpretable preference structure; otherwise, selection behavior cannot be meaningfully analyzed.

2.  **Multi-Condition Experimental Design**:
    *   **Function**: Tests the stability and conditional sensitivity of preferences.
    *   **Mechanism**: Five task conditions—Free budget within elements (1-1), Fixed budget within elements (1-2), Pooled unlabeled free (2-1), Pooled unlabeled fixed (2-2, baseline), and Per-element quota (3). Three instruction types (Base/Quality/Creativity). The baseline preference was established by comparing conditions: Pooled unlabeled fixed budget most closely reflects the model's native preference structure.
    *   **Design Motivation**: Multi-condition design decouples preferences from task design artifacts—if preferences remain stable across conditions, they are more credible.

3.  **Statistical Analysis Framework**:
    *   **Function**: Rigorously quantifies and compares selection patterns.
    *   **Mechanism**: Poisson GEE (clustering by run) was used to estimate element-level and category-level Rate Ratios (RR). K-weighted WLS was used to estimate condition contrasts. Axial richness was evaluated using hierarchical permutation tests.
    *   **Design Motivation**: Selection data possess count properties and clustering structures; Poisson GEE is the appropriate statistical model for such data.

### Loss & Training

This study consists of pure inference experiments and does not involve training. Commercial LLMs including GPT-4.1, GPT-5, o4-mini, Claude, Gemini, and Qwen were evaluated.

## Key Experimental Results

### Main Results

**Element-level Rate Ratios (vs. Event baseline, Poisson GEE)**

| Element | RR [95% CI] | p |
| :--- | :--- | :--- |
| Event (Baseline) | 1.00 | — |
| **Style** | **1.78** [1.74, 1.82] | <.001 |
| Character | 0.98 [0.96, 1.01] | .160 |
| Setting | 1.28 [1.25, 1.31] | <.001 |

### Ablation Study

**Stability Across Models**

| Finding | Description |
| :--- | :--- |
| Style Preference | Consistently highest across all 6 models. |
| gpt4.1 Specificity | Shows the strongest Style preference and the lowest for all other elements. |
| Instruction Sensitivity | Style remains stable across instructions, while content elements are more affected by creativity prompts. |

### Key Findings

*   All LLMs systematically prioritize Style constraints, with a selection rate 78% higher than Event.
*   Style preference is highly stable across models and instruction types, whereas content elements (Event/Character/Setting) show greater cross-model variance and instruction sensitivity.
*   gpt4.1 serves as a "Style preference amplifier"—consistently occupying the extremes in all comparisons.
*   Creativity-oriented instructions shift axial distributions but do not change the element-level ranking—Style always ranks first.
*   Selection behavior aligns with the stylistic uniformity found in previous output analysis research—LLMs do indeed have a systematic preference for style.

## Highlights & Insights

*   The "selection over generation" paradigm innovation—ingeniously isolates preference from capability, filling the gap that output analysis cannot reach.
*   The "Style over Story" finding serves as a practical warning for AI-assisted writing—if LLMs systematically prefer style, AI-assisted literature may trend toward surface sophistication while remaining narratively monotonous.
*   The constraint library itself is a reusable research tool that can be applied to evaluate narrative preferences in any future LLM.

## Limitations & Future Work

*   The relationship between selection preference and actual generation behavior has not been directly verified.
*   Only commercial LLMs were evaluated; open-source models or comparisons across different model scales were not included.
*   While theory-driven, the constraint library remains a subjective design; different narratological frameworks might yield different classifications.
*   The source of these preferences—whether they stem from training data bias or architectural characteristics—remains unexplored.

## Related Work & Insights

*   **vs. LLM Preference Measurement (Rozado 2024, Political Bias)**: The latter focuses on the political domain; this paper expands the scope to the narrative domain for the first time.
*   **vs. Output Analysis (Chakrabarty et al., 2024)**: The latter analyzes the quality of generated text; this paper directly measures preferences through selection—complementary rather than competitive.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic measurement of LLM narrative preferences; both paradigm and findings are original.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 models × 3 instructions × 5 conditions × 8,820 runs + rigorous statistics.
*   Writing Quality: ⭐⭐⭐⭐⭐ Elegant integration of narratological theory with computational experiments.
*   Value: ⭐⭐⭐⭐ Provides significant insights for AI-assisted creation and LLM bias research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Interpreting Style Representations via Style-Eliciting Prompts](interpreting_style_representations_via_style-eliciting_prompts.md)
- [\[ICLR 2026\] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language](../../ICLR2026/interpretability/semantic_regexes_auto-interpreting_llm_features_with_a_structured_language_of_re.md)
- [\[ACL 2026\] A Structured Clustering Approach for Inducing Media Narratives](a_structured_clustering_approach_for_inducing_media_narratives.md)
- [\[NeurIPS 2025\] Deep Value Benchmark: Measuring Whether Models Generalize Deep Values or Shallow Preferences](../../NeurIPS2025/interpretability/deep_value_benchmark_measuring_whether_models_generalize_deep_values_or_shallow_.md)
- [\[ACL 2026\] SITE: Soft Head Selection for Injecting ICL-Derived Task Embeddings](soft_head_selection_for_injecting_icl-derived_task_embeddings.md)

</div>

<!-- RELATED:END -->
