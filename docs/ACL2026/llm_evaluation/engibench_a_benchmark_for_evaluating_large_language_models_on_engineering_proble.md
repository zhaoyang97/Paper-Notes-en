---
title: >-
  [Paper Note] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving
description: >-
  [ACL 2026][LLM Evaluation][Paper Note] This paper proposes EngiBench—the first multi-level LLM evaluation benchmark for real-world engineering problem solving. Tasks are organized into three difficulty levels (Basic Knowledge Retrieval → Contextual Reasoning → Open-ended Modeling) and accompanied by three controlled variants (Perturbation / Knowledge Enhanc
tags:
  - ACL 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: d36def6223514b1d
---
# EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving

**Conference**: ACL 2026 Findings  
**arXiv**: [2509.17677](https://arxiv.org/abs/2509.17677)  
**Code**: https://github.com/AI4Engi/EngiBench  
**Area**: LLM Evaluation / Engineering Problems / Reasoning Benchmark  
**Keywords**: Engineering Reasoning, Hierarchical Evaluation, Controlled Variants, Open-ended Modeling, Perturbation Testing

## TL;DR
This paper proposes EngiBench—the first multi-level LLM evaluation benchmark for real-world engineering problem solving. Tasks are organized into three difficulty levels (Basic Knowledge Retrieval → Contextual Reasoning → Open-ended Modeling) and accompanied by three controlled variants (Perturbation / Knowledge Enhancement / Math Abstraction). Covering 1,760 problems across three engineering sub-domains, it reveals that even GPT-4.1 and Claude 3.7 Sonnet lag significantly behind human experts on Level 3 open-ended engineering tasks.

## Background & Motivation

**Background**: LLMs have shown impressive performance on mathematical reasoning (GSM8K / MATH / Omni-MATH), and model designers frequently use these math benchmarks to gauge "reasoning ability." However, real-world applications increasingly involve engineering problems—such as power scheduling, bridge design, and chemical reactor selection—tasks that involve much more than symbolic calculation.

**Limitations of Prior Work**: (1) Mainstream general benchmarks (MMLU / MMLU-Pro / BIG-Math / SuperGPQA) contain sparse engineering content and mostly feature multiple-choice questions, failing to examine the core engineering ability to "make trade-offs under fuzzy constraints." (2) Existing specialized engineering benchmarks (EEE-Bench / ElecBench / FEABench / TransportBench) are mostly single-discipline with closed answers, lacking open-ended tasks and cross-domain coverage. (3) Data contamination is a severe issue; replication experiments on GSM1k show that some models drop 8% in accuracy on rewritten problems, suggesting that much of the measured "reasoning ability" is actually memorization.

**Key Challenge**: The essence of engineering problem solving is "finding feasible solutions under constraints, uncertainty, and multiple objectives," where no single closed-form answer exists. In contrast, existing evaluations force models toward "finding a single correct answer," failing to measure the critical dimensions of engineering competence.

**Goal**: (1) Provide a hierarchical benchmark that truly measures "engineering problem-solving ability." (2) Deconstruct model "capability deficits"—determining whether they stem from insufficient knowledge, mathematical weakness, or poor engineering context understanding. (3) Offer an evaluation protocol that accommodates open-ended modeling tasks.

**Key Insight**: Engineering problem-solving capability is decomposed into four dimensions: Information Extraction, Domain Reasoning, Multi-objective Decision Making, and Uncertainty Handling. Tasks are then categorized into three layers of cognitive complexity using Bloom’s Taxonomy. Finally, three "controlled variants" are derived for each problem to isolate the true sources of model success or failure.

**Core Idea**: Construct a contamination-aware engineering benchmark for LLMs that explains failure causes and covers open-ended modeling by utilizing a three-axis orthogonal structure: "Hierarchical Difficulty × Controlled Variants × Four Dimensions of Engineering Ability."

## Method

### Overall Architecture

EngiBench decomposes the ability to solve engineering problems into a three-axis orthogonal diagnostic network. Vertically, it categorizes difficulty into three stages based on cognitive complexity; horizontally, it derives controlled variants for each problem to locate failure sources; and internally, it organizes scoring based on four engineering capability dimensions. Specifically, 1,760 problems are distributed across three sub-domains: Systems & Control (939), Physical & Structural (354), and Chemical & Biological (467). Each problem is assigned to Level 1 (single-step formulas), Level 2 (multi-step reasoning with constraints), or Level 3 (open-ended modeling), with additional perturbed, knowledge-enhanced, and math-abstraction variants generated (Level 3 only includes perturbed). Models receive engineering problems and provide free-form responses, which are scored via specific protocols: Levels 1/2 use binary scoring with multi-model cross-validation and manual inspection, while Level 3 uses expert rubrics (scoring 1–10 across four dimensions), LLM-based grading, and manual calibration, compared against submissions from competition winners and top students to establish a human upper bound.

### Key Designs

**1. Hierarchical Difficulty: Separating heterogeneous abilities from average scores**

Engineering capability is a combination of disparate skills, and a single accuracy metric averages them into an undifferentiated mass. On Level 1, both GPT-4.1 and Qwen2.5-7B can achieve over 80%, masking their differences. EngiBench uses Bloom’s Taxonomy to segment tasks into three layers: Level 1 is self-contained and solvable via single-step formulas (e.g., direct application of $V=IR$); Level 2 requires multi-step reasoning with unit/physical constraints and variable coupling (e.g., calculating branch resistance before merging for total resistance); Level 3 consists of 43 problems derived from actual mathematical modeling competitions (CUMCM / MCM-ICM / APMCM), which are open-ended, lack unique solutions, and require trade-offs under uncertainty. Only at Level 3 does the real gap—"8.74 for top students vs. low 7s for SOTA models"—become apparent.

**2. Controlled Variants: Attributing failures to memory, knowledge, math, or engineering context**

Traditional benchmarks only report an accuracy rate, failing to determine if a model forgot a formula, misunderstood the engineering semantics, or simply made a calculation error. EngiBench derives three variants for each Level 1/2 problem for controlled comparison: **Perturbed** retains the structure but changes numbers and phrasing to detect contamination and robustness; **Knowledge-enhanced** provides necessary formulas, constants, and definitions to separate "knowledge gaps" from "reasoning failures"; **Math abstraction** strips away engineering context to leave only the mathematical skeleton, isolating pure mathematical ability. The scoring trajectory across these four versions explicitly points to the bottleneck layer, upgrading evaluation from "scoring" to "diagnosis."

**3. Rubric-based Evaluation Protocol: Benchmarking open-ended modeling against human experts**

Preference-based scoring (like MT-Bench) suffers from high subjective bias, and reference-based scoring is unsuitable for open-ended problems without unique solutions. Therefore, Level 3 adopts a rubric-based approach. Forty-three problems with official rubrics were selected from nearly a thousand competition problems. Twenty PhDs and engineering professionals decomposed these standards into four dimensions: Information Extraction, Domain Reasoning, Multi-objective Decision Making, and Uncertainty Handling. Scoring is performed by an LLM judge based on the rubric, then manually calibrated. Solutions from competition winners (original) and top students (perturbed versions) are collected as the human upper bound. By explicitly deconstructing "what makes a good answer" into scorable dimensions, the benchmark maintains open-endedness while ensuring a reproducible metric.

### Loss & Training

This work focuses purely on evaluation and does not involve training models. The only involvement of LLMs is in the data construction pipeline: Engineering Relevance Filtering → Discipline Classification → Difficulty Assignment → Variant Generation → Expert Validation. LLMs assist in screening, translation, and variant generation, but every critical decision is verified by human experts.

## Key Experimental Results

### Main Results
Representative results for 16 models (GPT-4.1 series, Claude 3.5/3.7 Sonnet, Gemini 2.0/2.5 Flash, GLM-4, Qwen2.5, Llama 4 & 3.3, DeepSeek-V3, DeepSeek-R1 7B, Mixtral) across the three levels:

| Model | Level 1 (Acc) | Level 2 (Acc) | Level 3 (Avg Score, 0–10) | vs. Human (8.74) |
|------|-------------|--------------|-------------------------|---------------|
| GPT-4.1 | 90%+ | >80% | ~7.0 (Elite) | -1.7 |
| Claude 3.7 Sonnet | 90%+ | >80% | >6 | -2.0+ |
| Gemini 2.5 Flash | High | 81 (80.0 post-perturb) | Mid-High | — |
| DeepSeek-V3 | High | High | Significantly above structured pred | — |
| Qwen2.5-7B | ~70% | ~50%; Perturb ↓11.4 | <4 | — |
| Mixtral-8x7B | 70s | ~50; Perturb ↓8.3 | <4 | — |
| **Human Expert** | — | — | **8.74** | baseline |

Key observations: (1) Performance layers become distinct as difficulty increases. (2) All LLMs significantly lag behind human experts on Level 3. (3) The gap between closed-source SOTA and small open-source models widens dramatically with difficulty (merging at 70–90% in Level 1, but separating by orders of magnitude at Level 3).

### Ablation Study
Comparison across four variants (Level 2, Perturbed vs. Knowledge-enhanced vs. Math Abstraction):

| Model | Perturbed Δ | Knowledge-enhanced Δ | Math Abstraction Δ | Interpretation |
|------|-----------|---------------------|--------------------|------|
| GPT-4.1 Nano | -9.3 | + | + | Sensitive to perturbation |
| Qwen2.5-7B | **-11.4** | **+16.6** | **+15.5** | Relies on surface patterns + knowledge gaps |
| Mixtral-8x7B | -8.3 | + | + | As above |
| Gemini 2.5 Flash | **-1.2** | +2.4 | +2.5 | Robust; balanced knowledge and reasoning |

Level 3 Dimensional Breakdown: All models perform reasonably well in "Information Extraction" and moderately well in "Multi-objective Decision Making," but are severely deficient in **Domain Reasoning** and **Uncertainty Handling**. Llama 4, for instance, scored 0 in multi-objective decision making for failing to perform trade-off analysis, while GPT-4.1 scored 7.5.

### Key Findings
- **Small models are extremely sensitive to input format**: Qwen2.5-7B dropped 11.4% on perturbed versions but improved 15.5% on math abstractions—indicating it relies on engineering context for pattern recognition but requires formula extraction to perform calculations. It relies on surface patterns rather than robust reasoning.
- **Knowledge enhancement vs. Math abstraction gains identify failure causes**: Many models improved significantly with provided formulas/constants, showing that many "reasoning errors" are actually instances of "not knowing which formula to apply." This diagnostic signal is unavailable in other benchmarks.
- **Structured ability ≠ Open-ended ability**: While Level 1/2 scores generally correlate with Level 3, GPT-4.1, Claude 3.7, and DeepSeek-V3 performed significantly better on Level 3 than structured predictions would suggest. Conversely, Llama 4 was strong on structured tasks but failed on Level 3, proving that engineering reasoning cannot be measured by math benchmarks alone.
- **Models generally perform best after math abstraction**: This proves that "translating natural language engineering descriptions into structured mathematical formulas" is the true bottleneck for current LLMs; their base computational capacity is often sufficient.
- **Closed vs. Open-source gap expands with difficulty**: Everyone performs well at Level 1, but at Level 3, closed-source SOTA models outperform small open-source models by more than double, suggesting the benchmark is far from saturated.

## Highlights & Insights
- The "Hierarchical Difficulty × Controlled Variants × Four Dimensions" framework is a clean tri-axial diagnostic system. It decouples engineering ability, general reasoning, and contamination risks into different slices, providing an order of magnitude more information than a single accuracy metric.
- Introducing the "Math Abstraction Variant" as a standalone dimension is ingenious—it allows the observation "Engineering problem = Math problem + Contextual understanding" to be quantitatively measured.
- Constructing Level 3 using 43 real competition problems with official rubrics and winner solutions is a rare and successful attempt to make "open-ended" evaluation comparable and reproducible.
- Using "accuracy drop on perturbed versions" as a contamination indicator is more persuasive than general contamination tests; it reflects a model's dependence on surface patterns.
- Identifying the gap between the 8.74 human expert score and the ~7.0 SOTA score provides a quantitative anchor for current reasoning research gaps, serving as a roadmap for achieving human-expert-level engineering AI.

## Limitations & Future Work
- The benchmark is currently text-only and does not support multi-modality. Many real engineering problems involve blueprints, circuit diagrams, and tables; cross-modal evaluation was excluded to avoid visual processing interference but limits coverage.
- Long-context engineering tasks were excluded due to varying context windows across LLMs, meaning high-value scenarios like large-scale system design or long specification parsing were not evaluated.
- Only three engineering sub-domains are covered, missing fields like software engineering or detailed civil engineering calculations; the 1,760 samples are relatively small compared to MMLU.
- Level 3 scoring involves LLM judges; despite manual calibration, biases toward certain expression styles may persist, and rubric design itself can be subject to different academic perspectives.
- No training data or SFT-friendly subsets for engineering tasks were provided, as the paper focuses purely on evaluation.

## Related Work & Insights
- **vs. MMLU-Pro / SuperGPQA**: These are broad multi-disciplinary benchmarks with small engineering components and multiple-choice formats. EngiBench is engineering-focused, utilizes free-form responses, and includes open-ended modeling.
- **vs. EEE-Bench / ElecBench / FEABench**: These are single-discipline with closed answers. EngiBench spans three sub-domains, three difficulty levels, and uses open-ended rubric scoring.
- **vs. MATH / GSM8K + GSM1k**: These are pure math evaluations. This paper demonstrates that LLMs are often strongest on math abstraction variants, suggesting the root of engineering difficulty is "translation" rather than "calculation."
- **vs. Prometheus (Kim 2024)**: This is a general rubric-based evaluation for general capabilities like context retention. EngiBench encodes four specific engineering dimensions into its rubrics, making it more professional and aligned with experts.

## Rating
- Novelty: ⭐⭐⭐⭐ The orthogonal design (Difficulty × Variants × Dimensions) is a clear innovation for benchmark papers; systematic rubric-based open-ended engineering evaluation is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ A complete evaluation matrix of 16 models across 1,760 problems and 4 variants, with high reproducibility through provided prompts and rubrics.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, with Figures 1/3/4 providing intuitive visualizations of the stratification phenomenon.
- Value: ⭐⭐⭐⭐⭐ Establishes a first-principles coordinate system for LLM engineering capability evaluation, likely to become a standard benchmark in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Fleet of Agents: Coordinated Problem Solving with Large Language Models](../../ICML2025/llm_evaluation/fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](../../NeurIPS2025/llm_evaluation/creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2026\] Identifying the Achilles' Heel: An Iterative Method for Dynamically Uncovering Factual Errors in Large Language Models](identifying_the_achilles_heel_an_iterative_method_for_dynamically_uncovering_fac.md)

</div>

<!-- RELATED:END -->
