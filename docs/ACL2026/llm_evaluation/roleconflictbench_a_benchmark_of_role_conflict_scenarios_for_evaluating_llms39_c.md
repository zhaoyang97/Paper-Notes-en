---
title: >-
  [Paper Note] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity
description: >-
  [ACL 2026][LLM Evaluation][role conflict] RoleConflictBench constructs 13,914 role conflict scenarios and leverages situational urgency as an objective constraint to evaluate LLMs' contextual sensitivity…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "role conflict"
  - "contextual sensitivity"
  - "social bias"
  - "situational urgency"
  - "benchmark"
date: 2026-05-08
content_hash: a567a64197ace769
---

# RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity

**Conference**: ACL 2026
**arXiv**: [2509.25897](https://arxiv.org/abs/2509.25897)
**Code**: [https://github.com/ddindidu/RoleConflictBench](https://github.com/ddindidu/RoleConflictBench)
**Area**: LLM Evaluation
**Keywords**: role conflict, contextual sensitivity, social bias, situational urgency, benchmark

## TL;DR

RoleConflictBench constructs 13,914 role conflict scenarios and leverages situational urgency as an objective constraint to evaluate LLMs' contextual sensitivity, revealing that model decisions are dominated by static role preferences rather than dynamic contextual cues.

## Background & Motivation

**Background**: LLMs are increasingly deployed in personalized advisory systems and social simulations, requiring them to navigate complex social dilemmas. Existing evaluations of LLMs' social capabilities focus primarily on social norm compliance, moral reasoning, and social relationship understanding, typically adopting normative paradigms with predetermined correct answers.

**Limitations of Prior Work**: Role conflict—situations in which the expectations of multiple social roles are mutually contradictory and cannot be simultaneously satisfied—is a common real-world social dilemma, yet no dedicated evaluation framework exists. Such problems have no single correct answer; the appropriate decision depends on multiple contextual factors, and existing benchmarks cannot assess LLMs' contextual sensitivity in such subjective domains.

**Key Challenge**: Subjective social dilemmas lack objective evaluation criteria—how can model behavior be quantitatively assessed in scenarios with no ground-truth answer?

**Goal**: (1) Design a benchmark that quantitatively evaluates LLMs' contextual sensitivity in role conflict scenarios; (2) reveal the behavioral patterns and inherent biases of LLMs when confronted with role conflicts.

**Key Insight**: Situational urgency is introduced as an objective control variable. While the "correct role" to prioritize may be debatable, the principle that emergencies must take precedence over routine matters commands broad consensus (98% human agreement). This establishes a baseline: high urgency must be prioritized over low urgency.

**Core Idea**: Urgency differentials are used to establish an objective baseline, and the degree to which model decisions deviate from this baseline is quantified as a sensitivity score, enabling objective evaluation within a subjective domain.

## Method

### Overall Architecture

RoleConflictBench employs a three-stage pipeline to generate role conflict stories, queries models with binary-choice questions, and analyzes model behavior via sensitivity scores and role priority indices. The pipeline proceeds as follows: expectation generation → scenario instantiation → story synthesis → model querying → behavioral analysis.

### Key Designs

1. **Three-Stage Story Generation Pipeline**:

    - Function: Generate diverse, controlled role conflict scenarios.
    - Mechanism: (a) *Expectation generation*—for 65 social roles spanning five domains (family, professional, social, interpersonal, and religious), an LLM generates three concise social expectations per role; (b) *Scenario instantiation*—three urgency levels ($u \in \{1,2,3\}$, corresponding to routine / important but deferrable / urgent) are instantiated for each expectation; all expectations and scenarios undergo human verification; (c) *Story synthesis*—two roles are sampled from different domains, each paired with one expectation and one scenario, and synthesized into a 100–200-word first-person narrative covering all $3 \times 3 = 9$ urgency combinations.
    - Design Motivation: Systematically varying urgency levels ensures that decisions are not driven by trivially asymmetric situations; the $3 \times 3$ grid covers both symmetric and asymmetric conflicts.

2. **Sensitivity Score**:

    - Function: Quantify the alignment between model decisions and situational urgency signals.
    - Mechanism: For each role pair $(r_i, r_j)$, the empirical win probability $p_{ij,l}$ of role $r_i$ under three urgency relationships (higher than / equal to / lower than the opponent) is computed and compared against the ideal strategy $p^*_l \in \{1, 0.5, 0\}$ using mean squared error: $S = \sum_{l} \text{MSE}_l$. A score of $S = 0$ indicates perfect alignment, $S = 50$ corresponds to a random baseline, and $S = 225$ indicates complete inversion.
    - Design Motivation: Provides a standardized metric that precisely quantifies the competition between a model's inherent role preferences and externally provided contextual cues.

3. **Role Priority Index (RPI)**:

    - Function: Quantify a model's intrinsic priority preference for each role.
    - Mechanism: Based on the Bradley-Terry model, priority parameters $\pi_i$ for each role are estimated from pairwise comparisons via iterative maximum likelihood estimation and normalized. Domain preference scores $P_d$ are derived from the RPI to measure the model's preference for entire social domains (e.g., family, professional).
    - Design Motivation: Provides interpretable metrics that reveal the hierarchical structure of a model's inherent social biases.

### Loss & Training

This paper presents an evaluation study and does not involve model training. The dataset was generated using GPT-4.1, and all generated content underwent human verification.

## Key Experimental Results

### Main Results

**Sensitivity Score (lower is better; random baseline = 50)**

| Model | Sensitivity Score S |
|-------|-------------------|
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

| User Identity | S (↓) | Family Preference | Professional Preference |
|--------------|-------|------------------|------------------------|
| Default | 73.26 | 16.3% | 70.3% |
| Man | 77.58 | 26.7% | 56.7% |
| Woman | 76.47 | 18.6% | 64.0% |
| Asian | 80.09 | 23.6% | 62.9% |
| Hispanic | 79.21 | 22.9% | 63.1% |

### Key Findings

- **No model surpasses the random baseline**: scores range from 72 to 86, while the random baseline is only 50, indicating that model decisions deviate substantially from situational urgency constraints.
- Models do process urgency signals ($p_{i,\text{high}} > p_{i,\text{equal}} > p_{i,\text{low}}$), but this signal is consistently overridden by stronger static role preferences.
- Post-training effects are inconsistent: sensitivity worsens for Qwen3 after SFT and instruction tuning (75.24→82.82), while OLMo2 first improves then degrades.
- GPT-4.1 significantly increases family role preference for male users (16.3%→26.7%) and shows higher family preference for Asian and Hispanic users, exposing demographically conditioned biases.
- Model reasoning over-relies on a small set of prosocial values (Benevolence, Universalism) and rarely invokes diverse values such as Power or Stimulation.
- GPT-4.1 exhibits clear gender bias: male roles receive higher priority than female roles (53.8% vs. 46.2%), and high-income roles receive higher priority than low-income roles (57.9% vs. 42.1%).

## Highlights & Insights

- The methodological contribution of using urgency to establish an objective baseline for evaluating subjective decisions is notable—transforming a "no ground-truth" problem into a quantitatively assessable one, an approach transferable to other subjective evaluation tasks.
- The analytical framework combining the Bradley-Terry model with the Role Priority Index offers a systematic tool for uncovering the hierarchical structure of LLMs' inherent biases.
- The finding that LLMs' "value reasoning" is in practice a simplified heuristic—mapping social domains onto a small set of fixed values rather than performing genuinely context-sensitive reasoning—is particularly insightful.

## Limitations & Future Work

- The exclusive use of a binary-choice format limits response richness; real-world role conflict resolution typically involves trade-offs and compromises.
- Three urgency levels may be too coarse; finer-grained gradations could reveal more nuanced behavioral patterns.
- Only 10 models are evaluated, predominantly in the 8B–32B scale range, with no coverage of models at 70B or larger.
- All stories are LLM-generated, which may introduce generation bias; although human verification was conducted, its coverage is limited.

## Related Work & Insights

- **vs. SOCIALBENCH / MoralChoice**: These benchmarks adopt normative paradigms with predetermined correct answers. RoleConflictBench is the first to enable objective evaluation in a subjective domain.
- **vs. bias detection work**: Conventional bias benchmarks measure stereotypes in model outputs; this paper instead reveals an implicit hierarchical structure of social bias at the decision-making level.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Novel problem formulation with a particularly innovative methodology of using urgency to establish an objective baseline.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Analysis is in-depth and multi-faceted (sensitivity / preference / demographics / values), though model coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem definition is clear, methodology is rigorous, and findings are compellingly articulated.
- Value: ⭐⭐⭐⭐⭐ — Exposes deep deficiencies in LLMs' social reasoning with important implications for alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[AAAI 2026\] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](../../AAAI2026/llm_evaluation/coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)

</div>

<!-- RELATED:END -->
