---
title: >-
  [Paper Note] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity
description: >-
  [ACL 2026][LLM Evaluation][Role Conflict] RoleConflictBench evaluates LLM contextual sensitivity by constructing 13,914 role conflict scenarios using situational urgency as an objective constraint…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Role Conflict"
  - "Contextual Sensitivity"
  - "Social Bias"
  - "Situational Urgency"
  - "Benchmark"
date: 2026-05-08
content_hash: ba0cca848c49fbdb
---

# RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity

**Conference**: ACL 2026  
**arXiv**: [2509.25897](https://arxiv.org/abs/2509.25897)  
**Code**: [https://github.com/ddindidu/RoleConflictBench](https://github.com/ddindidu/RoleConflictBench)  
**Area**: LLM Evaluation  
**Keywords**: Role Conflict, Contextual Sensitivity, Social Bias, Situational Urgency, Benchmark

## TL;DR

RoleConflictBench evaluates LLM contextual sensitivity by constructing 13,914 role conflict scenarios using situational urgency as an objective constraint, revealing a significant issue where model decisions are dominated by static role preferences instead of responding to dynamic situational cues.

## Background & Motivation

**Background**: LLMs are increasingly utilized in personalized advisor systems and social simulations, requiring them to handle complex social dilemmas. Existing assessments of LLM social capabilities primarily focus on normative paradigms such as social norms, moral reasoning, and social relationship understanding, which typically feature predetermined "correct answers."

**Limitations of Prior Work**: Role conflicts—situations where expectations of multiple social roles are contradictory and cannot be satisfied simultaneously—are common real-world social dilemmas but lack dedicated evaluation frameworks. Such problems have no unique correct answer; correct decisions depend on multiple contextual factors, which existing benchmarks cannot evaluate regarding LLM contextual sensitivity in these subjective domains.

**Key Challenge**: The lack of objective evaluation criteria in subjective social dilemmas—how can model behavior be quantitatively evaluated in scenarios with "no standard answer"?

**Goal**: (1) Design a benchmark capable of quantitatively evaluating LLM contextual sensitivity in role conflict scenarios; (2) Uncover behavioral patterns and inherent biases of LLMs facing role conflicts.

**Key Insight**: Introduce "situational urgency" as an objective control variable. While the "correct role" is debatable, there is a broad consensus that emergency situations must take priority over daily routines (98% agreement in human evaluation). This establishes a baseline: high urgency must take priority over low urgency.

**Core Idea**: Establish an objective baseline using differences in urgency, and quantify the degree to which model decisions deviate from this baseline as a sensitivity score, thereby achieving objective evaluation in a subjective domain.

## Method

### Overall Architecture

RoleConflictBench generates role conflict stories through a three-stage pipeline, queries models with binary choice questions, and analyzes model behavior using sensitivity scores and role priority indices. The process includes: Expectation Generation → Situation Instantiation → Story Synthesis → Model Querying → Behavior Analysis.

### Key Designs

1.  **Three-stage Story Generation Pipeline**:

    *   **Function**: Generates diverse and controlled role conflict scenarios.
    *   **Mechanism**: (a) Expectation Generation—Using LLMs to generate 3 concise social expectations for 65 social roles (covering family, occupational, social, interpersonal, and religious domains); (b) Situation Instantiation—Generating situations with three urgency levels for each expectation ($u \in \{1,2,3\}$, corresponding to Routine, Important but Deferrable, and Urgent), all expectations and situations are human-verified; (c) Story Synthesis—Sampling two roles from different domains, assigning an expectation and situation to each, and synthesizing a 100-200 word first-person narrative, covering all $3 \times 3 = 9$ combinations of urgency.
    *   **Design Motivation**: Systematically varying urgency levels ensures decisions are not driven by simple asymmetric scenarios; the $3 \times 3$ grid covers both symmetric and asymmetric conflicts.

2.  **Sensitivity Score ($S$)**:

    *   **Function**: Quantifies the degree of alignment between model decisions and situational urgency signals.
    *   **Mechanism**: For each pair of roles $(r_i, r_j)$, the empirical win probability $p_{ij,l}$ of role $r_i$ is calculated under three urgency relationships (Higher/Equal/Lower than the opponent). These are compared with the ideal strategy $p^*_l \in \{1, 0.5, 0\}$, measuring deviation using Mean Squared Error: $S = \sum_{l} \text{MSE}_l$. $S = 0$ represents perfect alignment, $S = 50$ is a random baseline, and $S = 225$ represents total reversal.
    *   **Design Motivation**: Provides a standardized metric scale to precisely quantify the competition between a model's inherent role preferences and external situational context.

3.  **Role Priority Index (RPI)**:

    *   **Function**: Quantifies the model's inherent priority preference for each role.
    *   **Mechanism**: Based on the Bradley-Terry model, priority parameters $\pi_i$ for each role are estimated from pairwise comparisons using iterative maximum likelihood estimation and normalized. Domain preference scores $P_d$ are derived from RPI to measure the model's preference for entire social domains (e.g., family, profession).
    *   **Design Motivation**: Provides an interpretable metric to reveal the hierarchy of the model's inherent social biases.

### Loss & Training

This work is an evaluation study and does not involve model training. The dataset was generated using GPT-4.1, and all generated content was human-verified.

## Key Experimental Results

### Main Results

**Sensitivity Score (Lower is better, Random Baseline = 50)**

| Model | Sensitivity Score $S$ |
| :--- | :--- |
| Gemini 2.5 Flash | 72.06 |
| GPT-4.1 | 73.26 |
| Qwen3-30B-Base | 75.24 |
| Gemini 2.5 Flash-Lite | 76.53 |
| OLMo2-32B-SFT | 78.39 |
| OLMo2-32B-Instruct | 79.27 |
| Qwen3-30B-SFT | 79.53 |
| GPT-4.1-mini | 80.41 |
| Qwen3-30B-Instruct | 82.82 |
| OLMo2-32B-Base | 85.63 |

### Demographic Bias Analysis

| User Identity | $S$ (↓) | Family Preference | Occupation Preference |
| :--- | :--- | :--- | :--- |
| Default | 73.26 | 16.3% | 70.3% |
| Man | 77.58 | 26.7% | 56.7% |
| Woman | 76.47 | 18.6% | 64.0% |
| Asian | 80.09 | 23.6% | 62.9% |
| Hispanic | 79.21 | 22.9% | 63.1% |

### Key Findings

*   **No models outperformed the random baseline**: Scores ranged from 72-86, whereas the random baseline is 50, indicating that model decisions deviate significantly from situational urgency constraints.
*   Models do process urgency signals ($p_{i,\text{high}} > p_{i,\text{equal}} > p_{i,\text{low}}$), but these signals are consistently suppressed by stronger static role preferences.
*   The effectiveness of post-training is inconsistent: Qwen3's sensitivity worsened after SFT and instruction tuning (75.24 → 82.82), while OLMo2 initially improved and then regressed.
*   GPT-4.1 significantly increased family role preference for male users (16.3% → 26.7%) and showed higher family preference for Asian and Hispanic users, exposing demographic-based biases.
*   Model reasoning processes over-rely on a few pro-social values (Benevolence, Universalism), while neglecting diverse values like Power or Stimulation.
*   GPT-4.1 exhibits clear gender and income biases: male roles have higher priority than female roles (53.8% vs 46.2%), and high-income roles have higher priority than low-income roles (57.9% vs 42.1%).

## Highlights & Insights

*   Methodological innovation in using urgency to establish an objective baseline for evaluating subjective decisions—transforming "no standard answer" problems into quantitatively evaluable ones; this approach is transferable to other subjective evaluation tasks.
*   The analytical framework combining the Bradley-Terry model with the Role Priority Index provides a tool to systematically reveal the hierarchy of LLMs' inherent biases.
*   Found that LLM "value reasoning" is actually a simplified heuristic: mapping social domains to a few fixed values rather than performing true context-dependent reasoning.

## Limitations & Future Work

*   The use of only a binary choice format limits the richness of responses—resolving real-life role conflicts often involves trade-offs and compromises.
*   Three levels of urgency may be too coarse; finer granularity might reveal more subtle behavioral patterns.
*   Only 10 models were evaluated, primarily in the 8B-32B scale, lacking evaluation of 70B+ models.
*   The stories were entirely LLM-generated, which may introduce generation bias; although human-verified, the verification coverage has limits.

## Related Work & Insights

*   **vs SOCIALBENCH/MoralChoice**: These benchmarks use normative paradigms with predetermined correct answers. RoleConflictBench is the first to achieve objective evaluation in a subjective domain.
*   **vs Bias Detection**: Traditional bias benchmarks measure stereotypes in outputs, whereas this work reveals the implicit social bias hierarchy at the decision-making level.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ Innovative problem definition and methodology for establishing objective baselines using urgency.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Deep and multi-angled analysis (sensitivity/preference/demographics/values), though model coverage could be broader.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, rigorous methodology, and strong presentation of findings.
*   Value: ⭐⭐⭐⭐⭐ Reveals deep flaws in LLM social reasoning, providing important insights for alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[AAAI 2026\] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](../../AAAI2026/llm_evaluation/coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[ACL 2026\] MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge](mm-judgebias_a_benchmark_for_evaluating_compositional_biases_in_mllm-as-a-judge.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)

</div>

<!-- RELATED:END -->
