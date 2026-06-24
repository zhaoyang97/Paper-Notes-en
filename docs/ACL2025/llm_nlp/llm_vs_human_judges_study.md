---
title: >-
  [Paper Note] LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks
description: >-
  [ACL 2025][LLM (Other)][LLM-as-a-Judge] Establishes Judge-Bench with 20 datasets (70k+ instances) to systematically evaluate 11 LLM judges against human annotations. Findings reveal colossal performance variance across tasks, attributes, and user expertise, indicating that task-specific human validation remains critical before deploying LLM judges.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM-as-a-Judge"
  - "human evaluation"
  - "evaluation benchmark"
  - "NLP evaluation"
  - "Judge-Bench"
date: 2026-05-08
content_hash: 8c51cfdfb00e672a
---

tags:
  - ACL 2025
  - LLM (Other)
  - LLM-as-a-Judge
  - Judge-Bench
date: 2026-05-08
content_hash: 3670ded2ec82d258
---
# LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks

**Conference**: ACL 2025  
**arXiv**: [2406.18403](https://arxiv.org/abs/2406.18403)  
**Code**: [github.com/dmg-illc/JUDGE-BENCH](https://github.com/dmg-illc/JUDGE-BENCH)  
**Area**: LLM/NLP  
**Keywords**: LLM-as-a-Judge, human evaluation, evaluation benchmark, NLP evaluation, Judge-Bench  

## TL;DR

Establishes Judge-Bench with 20 datasets (70k+ instances) to systematically evaluate 11 LLM judges against human annotations. Findings reveal colossal performance variance across tasks, attributes, and user expertise, indicating that task-specific human validation remains critical before deploying LLM judges.

---

## Background & Motivation

**Rise of LLM-as-a-Judge**: Utilizing LLMs to evaluate NLP models is increasingly popular due to low costs, yet its empirical reliability lacks extensive and systematic verification.

**Conflicting Findings**: Prior publications present contradictory claims on whether LLM evaluations align with human grades, often suffering from constrained dataset and model selection.

**Reproducibility Threats**: Closed-source LLMs function as black-boxes that undergo silent updates, posing a severe threat to scientific reproducibility.

**Diverse Biases**: Models possess distinctive biases differing from human evaluators, such as self-model preference or over-sensitive safe refusals.

**Scanty Coverage**: Prior works mostly evaluate confined dimensions, leaving cross-analyses of expertise levels, task targets, and dataset properties largely unexplored.

**Core Problem**: Under what specific situations and boundaries can LLMs evaluate output text reliably in place of humans?

---

## Method

### Overall Architecture

This body of work constructs **Judge-Bench**, a standardized suite aggregating 20 NLP datasets featuring human assessments (70k+ instances) across categorical and graded structures. By formatting inputs to a unified schema, it benchmarks 11 LLM judges and audits their alignment against human references.

### Key Designs

#### Data Construction & Categorization
- **Source Dichotomy**: Classifies texts into human-authored versus machine-generated to systematically check whether LLMs are biased towards machine outputs.
- **Annotation Schemas**: Categorical targets utilize Cohen's $\kappa$ metric; graded evaluations use Spearman's $\rho$ correlation.
- **Covered Attributes**: Spans fluency, coherence, factual consistency, acceptability, verbosity, engagingness, toxicity, and safety labels.
- **Evaluator Expertise**: Differentiates annotations between expert panels and non-expert crowdworkers to analyze task-wise sensitivity.

#### Model Selection & Prompt Design
- **11 Benchmark Models**: Includes closed-source models (GPT-4o, Gemini-1.5) and open models (LLaMA-3.1-8B/70B, Mixtral-8x7B/8x22B, Command R/R+, OLMo, Starling-7B, Mistral).
- **Prompt Strategy**: Leverages the original annotation guidelines and appends formatting constraints like `"Answer with one of {}. Do not explain your answer."` Altered setups (CoT, few-shot) yielded inconsistent gains.

#### Evaluation Protocol
- **Refusal Handlers**: Safeguard triggers and empty outputs are filled using random draws, ensuring valid, matched-size samples for comparisons.
- **Target Comparisons**: Computes Cohen's $\kappa$ (categorical agreement) or Spearman's $\rho$ (graded agreement) against humans.
- **Human Upper Bound**: Estimated via bootstrap correlation between single annotators and aggregated consensus labels.

### Loss & Training

This paper performs evaluation only; **no model training or weight adjustments** are conducted. All LLMs perform inference in zero-shot or few-shot formats.

---

## Key Experimental Results

### Main Results

#### Table 1: Main Results on Categorical & Graded Annotations

| Model | Categorical Avg $\kappa$ | Graded Avg $\rho$ |
|------|----------------------|---------------------|
| GPT-4o | $0.28 \pm 0.32$ | $0.50 \pm 0.21$ |
| LLaMA-3.1-70B | $0.28 \pm 0.30$ | $0.43 \pm 0.22$ |
| Mixtral-8x22B | $0.24 \pm 0.30$ | $0.44 \pm 0.19$ |
| Gemini-1.5 | $0.22 \pm 0.28$ | $0.43 \pm 0.21$ |
| Mixtral-8x7B | $0.21 \pm 0.28$ | $0.38 \pm 0.22$ |
| Command R+ | $0.10 \pm 0.18$ | $0.30 \pm 0.17$ |

- GPT-4o secures head leadership overall, but top-tier open models like LLaMA-3.1-70B and Mixtral-8x22B sit within close range.
- Open-source models occasionally outperform closed models on specific tasks (such as CoLa syntax and SummEval).
- Massive standard deviations ($\sigma$ up to 0.23) underscore structural difficulty gaps across the evaluated tasks.

#### Table 2: Cross-Analysis of Key Dimensions

| Analytic Dimension | Key Finding |
|---------|---------|
| Expert vs Non-expert | Models exhibit higher correlation with **non-expert** evaluators, likely due to shared reliance on surface-level context cues. |
| Human vs Machine Text | LLMs correlate significantly better when evaluating **human-authored texts** compared to machine-generated alternatives. |
| Attribute Variation | Closed models lead on acceptability; Mixtral leads on coherence and consistency. Every model struggles on engagingness. |
| Safety & Toxicity | Achieves negative correlations on DICES and Medical-safety due to overactive guardrails causing invalid outputs. |

### Key Findings

1. **No Absolute Winner**: Attribute strengths vary across different model series, cautioning against defaulting exclusively to GPTs.
2. **CoT is Inconsistent**: Chain-of-Thought prompting fails to universally scale up judge performance across tasks.
3. **Safety Filters Impede Judges**: System guardrails trigger high refusal rates on sensitive datasets, rendering safety judgments inaccurate.
4. **Gap to Human Ceilings**: Performance largely stays far beneath human upper bounds, with sparse exceptions.
5. **Logic and Structure Remain Safest**: Evaluation tasks emphasizing mathematical rules and clear formatting constraints present highly reliable correlations.

---

## Highlights & Insights

- **Unmatched Benchmark Scale**: Combines evaluations over 20 tasks, 11 engines, and 70k+ instances for thorough, empirical evidence.
- **Multi-Dimensional Metrics**: Cross-references source types, human expertise levels, and task metrics to pinpoint exact alignment patterns.
- **Unified Testing Schema**: Judge-Bench provides a standardized data format, making it easier to integrate new domains.
- **Sobering Real-world Advice**: Dispenses detailed domain advice outlining when LLM judges are viable and when they require human checks.

## Limitations & Future Work

- **Assumes Perfect Agreement**: Relies on Correlation/Kappa values, which can mask instances of shared, systematic systematic bias.
- **Excludes Pairwise Setups**: Benchmarks primarily direct-scoring systems, omitting pairwise comparison modes (e.g., PairEval).
- **Language Limitation**: Focuses heavily on English tasks, leaving multilingual capacities untested.
- **Potential Data Leakage**: Open evaluation datasets might already have been crawled within proprietary training runs.
- **Post-GPT-4o Missing**: Does not feature the newest model iterations (such as Claude 3.5 or o1 series).

## Related Work & Insights

- **LLM Judges**: Explores G-Eval (Liu et al., 2023) and MT-Bench (Zheng et al., 2024) highlighting cost decreases.
- **Evaluation Biases**: Evaluates target model bias (Wang et al., 2024; Xu et al., 2024).
- **Alternative Setups**: Explores pairwise alignment models (Park et al., 2024; Kim et al., 2024).

---

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐ | Standard evaluation tooling; value stems from comprehensive analysis. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Sweepy testing across a variety of models, tasks, and attributes. |
| Practical Value | ⭐⭐⭐⭐ | Offers clear verification pipelines for building LLM judges. |
| Writing Quality | ⭐⭐⭐ | Clear structural writing with informative figures. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] An Empirical Study of Large Language Models for Automated Review Generation](an_empirical_study_of_large_language_models_for_automated_review_generation.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)
- [\[ACL 2025\] An Empirical Study of Iterative Refinements for Non-Autoregressive Translation](an_empirical_study_of_iterative_refinements_for_non-autoregressive_translation.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)
- [\[ACL 2025\] A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)

</div>

<!-- RELATED:END -->
