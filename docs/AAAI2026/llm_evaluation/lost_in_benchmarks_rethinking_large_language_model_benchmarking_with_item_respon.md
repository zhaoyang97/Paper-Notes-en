---
title: >-
  [Paper Note] Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory
description: >-
  [AAAI 2026][LLM Evaluation][IRT] This paper proposes PSN-IRT (Pseudo-Siamese Network for IRT), an enhanced Item Response Theory framework that jointly estimates LLM ability parameters and four-parameter item characterist…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "IRT"
  - "benchmark evaluation"
  - "condition number"
  - "item quality"
  - "PSN-IRT"
date: 2026-05-08
content_hash: ba081f0389fe7ecc
---

# Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory

**Conference**: AAAI 2026
**arXiv**: [2505.15055](https://arxiv.org/abs/2505.15055)  
**Code**: [https://github.com/Joe-Hall-Lee/PSN-IRT](https://github.com/Joe-Hall-Lee/PSN-IRT)  
**Area**: LLM Evaluation
**Keywords**: IRT, benchmark evaluation, condition number, item quality, PSN-IRT

## TL;DR
This paper proposes PSN-IRT (Pseudo-Siamese Network for IRT), an enhanced Item Response Theory framework that jointly estimates LLM ability parameters and four-parameter item characteristics (difficulty / discrimination / guessing / feasibility). Applied to 41,871 items across 11 benchmarks, the framework reveals systemic issues including widespread saturation, insufficient difficulty ceilings, and data contamination. Item subsets selected by PSN-IRT achieve a ranking consistency of Kendall $\tau = 1.00$.

## Background & Motivation

**Background**: LLM evaluation predominantly relies on average-score rankings on benchmarks (e.g., MMLU, HumanEval), yet different leaderboards produce inconsistent rankings, and performance gaps among top models are too small to be meaningfully distinguished.

**Limitations of Prior Work**:
- Different benchmarks can yield entirely divergent rankings for the same set of models — raising the question of whether such signals reflect genuine capability differences or merely noise.
- Current benchmarks treat all items with equal weight, ignoring item quality — easy and hard items contribute identically to the final score.
- No systematic tooling exists to diagnose the quality of benchmarks themselves (saturation, discrimination, contamination).

**Key Challenge**: Benchmarks are supposed to serve as objective measuring instruments, yet the quality of the instruments themselves has never been systematically audited.

**Goal**: To audit the item quality of LLM benchmarks using IRT, and to establish more reliable ability estimation and model ranking.

**Key Insight**: Deeply customizing Item Response Theory (IRT) from educational measurement for LLM evaluation — learning model ability and item parameters end-to-end via a deep pseudo-siamese network.

**Core Idea**: PSN-IRT = dual network (model ability + item parameters) $\times$ 4PL IRT formula $\rightarrow$ benchmark quality auditing + reliable ranking.

## Method

### Overall Architecture
Input: binary response matrices of 12 LLMs $\times$ 11 benchmarks $\rightarrow$ PSN-IRT dual-branch network $\rightarrow$ Output: per-model ability $\theta$ + four parameters per item (difficulty $b$, discrimination $a$, guessing $c$, feasibility upper bound $d$) $\rightarrow$ benchmark quality analysis + model ranking.

### Key Designs

1. **PSN-IRT Architecture**:

    - **Function**: End-to-end joint estimation of model ability and item parameters.
    - **Mechanism**: Two independent MLP branches — one estimates $\theta$ from model response patterns; the other estimates $(a, b, c, d)$ from item response patterns. Both are jointly optimized via the 4PL IRT formula $P(\theta) = c + \frac{d-c}{1+e^{-a(\theta-b)}}$.
    - **Design Motivation**: Traditional IRT relies on MLE or MCMC for iterative inference; PSN-IRT uses neural network end-to-end training for greater efficiency and scalability to large datasets.

2. **Four-Parameter IRT Model (4PL)**:

    - **Function**: More accurately models LLM response behavior compared to standard IRT.
    - **Four parameters**: difficulty $b$ (minimum $\theta$ required to answer correctly), discrimination $a$ (effectiveness of the item in differentiating high- vs. low-ability models), guessing $c$ (probability that a low-ability model answers correctly by chance), and feasibility upper bound $d$ (probability ceiling even for the strongest models).
    - **Design Motivation**: LLMs may answer easy items correctly by "guessing" ($c > 0$), and some items may be infeasible for all models ($d < 1$) — phenomena that standard IRT cannot capture.

3. **Benchmark Quality Diagnostics**:

    - **Function**: Diagnosing systemic benchmark issues via item parameters.
    - **Diagnostic dimensions**: Saturation (proportion of items with excessively low discrimination $a$); difficulty ceiling (whether the maximum $b$ suffices to differentiate top models); data contamination (anomalously high guessing rate $c$ may indicate answer leakage into training data).

### Loss & Training
- Binary cross-entropy loss (response prediction).
- Evaluated on 12 models (GPT-4, DeepSeek-V3, Qwen-Plus, etc.) $\times$ 11 benchmarks.

## Key Experimental Results

### Main Results

| Metric | PSN-IRT | Deep-IRT (1PL) | Traditional IRT (4PL MLE) |
|--------|---------|----------------|--------------------------|
| ACC | **0.7998** | 0.7974 | 0.7211 |
| F1 | **0.8538** | 0.8516 | 0.8034 |
| AUC | 0.8485 | **0.8519** | 0.7012 |
| Kendall $\tau$ | **1.0000** | 0.9697 | 0.9697 |

### Ablation Study: Benchmark Quality Diagnostics

| Benchmark | Primary Issue | Description |
|-----------|--------------|-------------|
| MMLU | High saturation | Most items fail to differentiate top-tier models |
| HumanEval+ | Insufficient difficulty ceiling | Hardest items are not challenging enough for GPT-4 |
| GSM8K | Suspected contamination | Some items exhibit anomalously high guessing rate $c$ |
| MATH | Good discrimination | The only benchmark performing well across multiple quality dimensions |

### Key Findings
- **No single benchmark excels across all quality dimensions** — every benchmark exhibits systemic weaknesses.
- **PSN-IRT rankings align with human preferences**: $\tau = 1.00$, substantially outperforming traditional methods ($0.97$).
- **Item subsets selected by PSN-IRT can substitute entire benchmarks** — a small set of high-quality items suffices for reliable ranking.
- **Model scale is not the sole determinant of capability** — IRT-estimated $\theta$ values sometimes diverge from rankings based on parameter count.

## Highlights & Insights
- **"Auditing the benchmark itself"** constitutes meta-evaluation — applying psychometric tools to assess the quality of measurement instruments, which is conceptually significant.
- **Using the 4PL guessing parameter $c$ as a contamination detector** is a clever application — if answers appear in training data, models can respond correctly even without genuine understanding.
- PSN-IRT can serve as a quality-gating tool for any AI benchmark.

## Limitations & Future Work
- Assumes binary responses (correct/incorrect), making it inapplicable to generative evaluation settings.
- 12 models may be insufficient to yield stable IRT parameter estimates.
- Item interdependencies are not modeled (standard IRT assumes local independence).

## Related Work & Insights
- **vs. Chatbot Arena (LMSYS)**: Human preference-based ranking. PSN-IRT ranks models via items; the two approaches are complementary.
- **vs. BenchmarkCards**: Descriptive diagnostics. PSN-IRT provides quantitative item-level parameters.
- **vs. DynaBench**: DynaBench addresses data leakage via dynamic datasets but does not resolve item quality issues; PSN-IRT quantifies each item's discriminative power from a statistical perspective.
- The application of IRT to AI evaluation is generalizable to domain-specific benchmarks in coding, reasoning, and beyond.
- Insight: New benchmarks should undergo IRT analysis prior to release to filter out low-discrimination items.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — A systematic framework for auditing LLM benchmarks with IRT, introducing mature psychometric tools into AI evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 11 benchmarks, 12 models, and 41K items; analysis is conducted at sufficient scale.
- **Writing Quality**: ⭐⭐⭐⭐ — Theory and empirical findings are well integrated; visualizations clearly illustrate item quality issues.
- **Value**: ⭐⭐⭐⭐⭐ — Makes an important foundational contribution to LLM evaluation methodology by exposing the pervasive presence of low-quality items in existing benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] How Reliable is Language Model Micro-Benchmarking?](../../ICLR2026/llm_evaluation/how_reliable_is_language_model_micro-benchmarking.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](../../ICCV2025/llm_evaluation/rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](../../NeurIPS2025/llm_evaluation/bayesian_evaluation_of_large_language_model_behavior.md)
- [\[ICML 2026\] BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback](../../ICML2026/llm_evaluation/bespoke_benchmark_for_search-augmented_large_language_model_personalization_via_.md)
- [\[AAAI 2026\] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)

</div>

<!-- RELATED:END -->
