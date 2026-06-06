---
title: >-
  [Paper Note] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language
description: >-
  [ACL 2026][LLM/NLP][repeated subsequences] This paper proposes an evaluation framework based on the distribution of repeated subsequences. By characterizing the entropy growth behavior of text through higher-order Rényi…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "repeated subsequences"
  - "Rényi entropy"
  - "LLM evaluation"
  - "long-range structure"
  - "entropy growth analysis"
date: 2026-05-08
content_hash: 7d09398664e6da14
---

# Repeated Sequences Reveal Gaps between Large Language Models and Natural Language

**Conference**: ACL 2026  
**arXiv**: [2605.24850](https://arxiv.org/abs/2605.24850)  
**Code**: None  
**Area**: llm_nlp  
**Keywords**: repeated subsequences, Rényi entropy, LLM evaluation, long-range structure, entropy growth analysis

## TL;DR
This paper proposes an evaluation framework based on the distribution of repeated subsequences. By characterizing the entropy growth behavior of text through higher-order Rényi entropy, it reveals that natural language exhibits a stable sub-linear entropy growth pattern, while the entropy exponents of GPT-generated text increase monotonically with model scale. This exposes systematic differences between LLMs and natural language in long-range statistical organization.

## Background & Motivation
**Background**: LLMs exhibit excellent performance on various task benchmarks. However, evaluation primarily relies on task performance or short-context behavior, lacking systematic analysis of the long-range statistical structure of generated text.

**Limitations of Prior Work**: Existing evaluation methods cannot determine whether LLMs truly capture the structural organization of natural language at a large scale—high benchmark scores do not imply that generated text possesses the long-range statistical properties of human text. Prior studies have identified issues such as excessive repetition and diversity degradation in LLMs.

**Key Challenge**: Expressions in natural language are not used in isolation; instead, they form reference structures across long distances through repeated citation and reorganization. It remains unclear whether LLMs can reproduce this structure under the next-token prediction objective.

**Goal**: To propose a quantitative diagnostic tool based on the distribution of repeated subsequences to distinguish differences in long-range organization between natural language and LLM outputs.

**Key Insight**: Treat repetition as a distributional property to be analyzed across scales, rather than focusing only on extreme repetition or generation degradation phenomena.

**Core Idea**: A deep connection exists between the number of repeated subsequences and higher-order Rényi entropy. Fitting power-law vs. log-power-law models to entropy growth can reveal the structural reuse characteristics of text.

## Method

### Overall Architecture
The method consists of three steps: (1) Count the number of repeated subsequences of length $m$, $D_m = T_m - K_m$ (total blocks minus distinct blocks); (2) Associate $D_m$ with higher-order Rényi entropy $H_\alpha(m)$ and derive its asymptotic expansion; (3) Fit power-law models ($\propto m^\beta$) and log-power-law models ($\propto (\log m)^\gamma$) to $H_\alpha(m)$, comparing the differences between natural language and GPT text.

### Key Designs
1. **Connection between Repeated Subsequences and Rényi Entropy**:
    - **Function**: Establish a bridge from observable repetition statistics to information-theoretic quantities.
    - **Mechanism**: The expectation of the number of repetitions $D_m$ for length $m$ can be expanded into a series of $\sum p_w^\alpha$ ($\alpha \geq 2$), which naturally corresponds to Rényi entropy $H_\alpha(m) = \frac{1}{1-\alpha}\log_2 \sum p_w^\alpha$.
    - **Design Motivation**: Direct analysis of $D_m$ is heavily influenced by document length; transforming it into Rényi entropy yields length-independent structural features.

2. **Two-stage Parameter Estimation (Finite-length Correction)**:
    - **Function**: Accurately estimate entropy growth exponents on finite-length text.
    - **Mechanism**: First, estimate $\lambda_m = T_m/S_m$ from the functional relationship of $D_m/T_m$, then fit $\log_2 S_m = H_\alpha(m) + \Delta_\alpha$, where $\Delta_\alpha$ is a finite-length correction term depending on $\lambda_m$.
    - **Design Motivation**: Estimating exponents directly from $K_m$ or $H_\alpha(m)$ is unstable due to finite-length effects; the two-stage method significantly improves fitting reliability.

3. **Power-law vs. Log-power-law Model Comparison**:
    - **Function**: Distinguish between two qualitatively different modes of information accumulation.
    - **Mechanism**: The power law $G(m) \propto m^\beta$ corresponds to the continuous expansion of structural degrees of freedom (constantly introducing new information), while the log-power law $G(m) \propto (\log m)^\gamma$ corresponds to strong structural reuse (sharing resources through reorganization and re-indexing).
    - **Design Motivation**: The entropy growth of natural language may lie at the boundary of these two mechanisms; distinguishing them helps understand the essence of language generation.

### Loss & Training
This is a purely analytical method with no training process. All analyses are performed at the character level to avoid tokenizer bias. Fitting quality and group differences are evaluated using the coefficient of determination $R^2$ and Welch's t-test.

## Key Experimental Results

### Dataset Scale
| Dataset | Quantity | Average Length (Characters) |
|--------|------|-----------------|
| gpt-3.5turbo | 100 | 35,045 ± 2,287 |
| gpt-4o-mini | 100 | 110,889 ± 23,379 |
| gpt-5-mini | 100 | 347,045 ± 19,793 |
| gpt-5 | 100 | 601,187 ± 24,973 |
| nl (matched to each GPT) | 100 each | Corresponding match |

### Core Statistical Test Results
| Comparison | $\beta$ Difference | $\gamma$ Difference | p-value |
|------|-------------|-------------|------|
| gpt-5 vs nl-5 | GPT significantly larger | GPT significantly larger | ≈0 |
| gpt-5-mini vs nl-5-mini | GPT significantly larger | GPT significantly larger | ≈0 |
| nl-5 vs nl-5-mini | No significant difference | No significant difference | β: 0.12, γ: 0.94 |

### Key Findings
- Entropy growth exponents $\beta$ and $\gamma$ for natural language remain stable across datasets of different lengths (weak universality), while exponents for GPT text increase monotonically with model scale.
- Log-power-law models generally outperform power-law models in long texts ($R^2 > 0.97$ vs 0.90-0.96), indicating that natural language is dominated by structural reuse.
- Short texts favor power-law fitting (continuous introduction of new information), while long texts favor log-power-law fitting (enhanced structural reuse).
- Traditional maximal repeated subsequence methods are almost indistinguishable between gpt-5 and natural language (mean $\eta$ values are close), but the proposed method still detects significant differences.

## Highlights & Insights
- Proposes a novel LLM evaluation dimension based on fundamental information-theoretic principles, independent of downstream tasks.
- The derivation from repeated subsequences to Rényi entropy is concise and elegant, with rigorous handling of finite-length corrections.
- Discovers "weak universality" in natural language—individual text differences are large, but overall exponents are stable, which is an interesting statistical law.
- Analysis of the Complete Works of Shakespeare ($n=5,442,126$ characters) demonstrates the significance of log-power-law behavior in extremely long texts.

## Limitations & Future Work
- Analyzes only GPT family models; applicability to other architectures (e.g., Llama/Claude) remains to be verified.
- Analysis is performed at the character level and is not directly linked to word-level or syntactic language structures.
- The method is descriptive and cannot identify the specific mechanisms causing the observed differences.
- Requires long texts (at least tens of thousands of characters) for reliable fitting, limiting its use in short-text scenarios.
- The entropy rate $h_\alpha$ is not directly estimated, making it impossible to determine if the entropy rate of natural language is zero.

## Related Work & Insights
- **Hilberg (1990)**: Proposed a sub-linear power-law growth conjecture for block entropy in natural language; this paper further distinguishes between power-law and log-power-law mechanisms.
- **Dębowski (2015)**: Based on maximal repeated subsequence analysis; this paper shows that distributional methods are more stable and discriminative than extreme statistics.
- **Holtzman et al. (2020)**: Focused on repetition degradation in LLMs; this paper repositions repetition from a "problem" to a "structural signal."
- **Insight**: Evaluating LLMs should go beyond task scores to verify whether their output possesses the intrinsic statistical structure of natural language.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ New evaluation perspective, deeply integrating information theory with LLM evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Reasonable dataset design (length matching) and rigorous statistical testing, though limited to the GPT family.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, smooth narrative, and excellent chart design.
- Value: ⭐⭐⭐⭐ Provides a fresh tool for LLM evaluation, though practical application scenarios need expansion.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Identifying the Periodicity of Information in Natural Language](identifying_the_periodicity_of_information_in_natural_language.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)
- [\[ACL 2026\] Why Did Apple Fall: Evaluating Curiosity in Large Language Models](why_did_apple_fall_evaluating_curiosity_in_large_language_models.md)
- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)

</div>

<!-- RELATED:END -->
