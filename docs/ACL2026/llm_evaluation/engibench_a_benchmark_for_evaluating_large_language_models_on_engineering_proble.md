---
title: >-
  [Paper Note] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving
description: >-
  [ACL 2026][LLM Evaluation][Engineering Reasoning] This paper proposes EngiBench, the first multi-level evaluation benchmark for real-world engineering problem solving by LLMs. It organizes tasks into three levels of diff…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Engineering Reasoning"
  - "Hierarchical Evaluation"
  - "Controlled Variants"
  - "Open-ended Modeling"
  - "Perturbation Testing"
date: 2026-05-08
content_hash: a69d2160ce9dfefc
---

# EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving

**Conference**: ACL 2026  
**arXiv**: [2509.17677](https://arxiv.org/abs/2509.17677)  
**Code**: https://github.com/AI4Engi/EngiBench  
**Area**: LLM Evaluation / Engineering Problems / Reasoning Benchmark  
**Keywords**: Engineering Reasoning, Hierarchical Evaluation, Controlled Variants, Open-ended Modeling, Perturbation Testing

## TL;DR
This paper proposes EngiBench, the first multi-level evaluation benchmark for real-world engineering problem solving by LLMs. It organizes tasks into three levels of difficulty (Basic Knowledge Retrieval → Contextual Reasoning → Open-ended Modeling) and incorporates three controlled variants (Perturbed / Knowledge-enhanced / Math-abstraction). Covering 1,760 problems across three major engineering sub-domains, the study finds that even SOTA models like GPT-4.1 and Claude 3.7 Sonnet significantly lag behind human experts in Level 3 open-ended engineering tasks.

## Background & Motivation

**Background**: LLMs have demonstrated remarkable performance in mathematical reasoning ($GSM8K$ / $MATH$ / $Omni-MATH$). Model designers frequently utilize these mathematical benchmarks to measure "reasoning ability." However, an increasing number of real-world application scenarios involve engineering problems—such as power scheduling, bridge design, and chemical reactor selection—which require far more than symbolic calculation.

**Limitations of Prior Work**: (1) Mainstream general benchmarks ($MMLU$ / $MMLU-Pro$ / $BIG-Math$ / $SuperGPQA$) have sparse engineering content and consist mostly of multiple-choice questions, failing to examine the core engineering capability of making "trade-offs under fuzzy constraints." (2) Existing specialized engineering benchmarks ($EEE-Bench$ / $ElecBench$ / $FEABench$ / $TransportBench$) are mostly single-discipline with closed answers, lacking open-ended tasks and cross-domain coverage. (3) Data contamination is severe; replication experiments on $GSM1k$ show an 8% drop for some models on rewritten questions, suggesting that much "reasoning ability" is actually memorization.

**Key Challenge**: The essence of engineering problem solving is "finding feasible solutions under constraints, uncertainty, and multiple objectives," where no unique closed-form answer exists. Current evaluations force models toward "finding a single correct answer," failing to measure the critical dimensions of engineering capability.

**Goal**: (1) Provide a hierarchical benchmark that truly measures "engineering problem-solving capability." (2) Decompose "capability defects"—determining if they stem from insufficient knowledge, mathematical deficiencies, or poor understanding of engineering context. (3) Offer an evaluation protocol capable of accommodating open-ended modeling tasks.

**Key Insight**: Decompose engineering problem-solving capability into four dimensions: Information Extraction, Domain Reasoning, Multi-objective Decision-making, and Uncertainty Handling. Map tasks to three levels of cognitive complexity based on Bloom's Taxonomy. Finally, derive three "controlled variants" for each question to decompose the true source of model success or failure.

**Core Idea**: Construct a contamination-aware, diagnostic, and open-modeling LLM engineering benchmark using a three-axis orthogonal structure: "Hierarchical Difficulty × Controlled Variants × 4D Engineering Capabilities."

## Method

### Overall Architecture
EngiBench contains 1,760 questions distributed across three engineering sub-domains: Systems & Control (939), Physical & Structural (354), and Chemical & Biological (467). It features Level 1 (Single-step formula application), Level 2 (Multi-step reasoning with constraints), and Level 3 (Open-ended modeling). Each question also generates "perturbed," "knowledge-enhanced," and "math-abstraction" variants (Level 3 includes "perturbed" only). Evaluation Protocol: Binary scoring for Level 1/2 with multi-model cross-validation and manual audits; Level 3 uses expert-defined rubrics (1–10 scores across 4 dimensions) with LLM-as-a-judge, manual calibration, and comparison against competition winners/top student submissions.

### Key Designs

1.  **Hierarchical Difficulty (Cognitive Ladder)**:
    -   **Function**: Enables the benchmark to separately measure "knowledge retrieval," "reasoning under constraints," and "open-ended modeling," rather than conflating them into a single average score.
    -   **Mechanism**: Level 1 tasks are self-contained and solvable with single-step formulas (e.g., direct application of $V=IR$). Level 2 requires multi-step reasoning, units/physical constraints, and variable coupling (e.g., calculating total circuit resistance by merging branches). Level 3 is sampled from actual mathematical modeling competitions ($CUMCM$ / $MCM-ICM$ / $APMCM$), which are open-ended with no unique solution, requiring trade-offs under uncertainty and conflicting objectives; all 43 tasks include official rubrics.
    -   **Design Motivation**: A single metric (like accuracy) masks differences across capability dimensions—both GPT-4.1 and Qwen2.5-7B score 80%+ on Level 1, but at Level 3, human experts score 8.74 while SOTA models peak at just over 7. The gap only becomes visible at appropriate difficulty slices.

2.  **Controlled Variants (Failure Attribution)**:
    -   **Function**: Decomposes "model errors" into four potential causes: memory, knowledge, mathematics, or engineering context.
    -   **Mechanism**: Each Level 1/2 question derives three variants: (a) **Perturbed**: Retains the structure but changes numbers and contextual phrasing to detect contamination/robustness; (b) **Knowledge-enhanced**: Provides necessary formulas/constants/definitions within the prompt to separate "knowledge gaps" from "reasoning failures"; (c) **Math abstraction**: Removes engineering context, retaining only the mathematical structure to isolate pure mathematical ability. The score trajectory across these variants reveals the specific performance bottleneck.
    -   **Design Motivation**: Traditional benchmarks only report accuracy, failing to clarify if a failure is due to an unknown formula, misunderstood engineering semantics, or a calculation error. Controlled variants upgrade evaluation from "scoring" to "diagnosis."

3.  **Rubric-based Evaluation Protocol (Level 3 Rubric Scoring)**:
    -   **Function**: Allows LLMs to be fairly evaluated on open-ended engineering modeling tasks and compared with human experts on the same scale.
    -   **Mechanism**: 43 tasks with official rubrics were selected from nearly 1,000 competition problems. 20 PhDs and engineering practitioners decomposed the standards into four dimensions: Information Extraction, Domain Reasoning, Multi-objective Decision-making, and Uncertainty Handling. Scoring is performed by an LLM judge based on the rubric, followed by manual calibration. Submissions from competition winners (original) and top students (perturbed versions) serve as human upper bounds.
    -   **Design Motivation**: Preference-based evaluation ($MT-Bench$) suffers from subjective bias, while reference-based evaluation is unsuitable for open-ended problems. Rubric-based evaluation allows experts to explicitly define "what a good answer is" across gradable dimensions, maintaining openness while providing reproducible standards.

### Loss & Training
This paper does not train a model; it is fundamentally a benchmark paper. The "training" is reflected in the data construction pipeline where LLMs assist in filtering, translating, and generating variants ($Engineering Relevance Filtering \rightarrow Discipline Classification \rightarrow Difficulty Assignment \rightarrow Variant Generation \rightarrow Expert Validation$), with all key decisions overseen by human experts.

## Key Experimental Results

### Main Results
Representative results for 16 models (GPT-4.1 series, Claude 3.5/3.7 Sonnet, Gemini 2.0/2.5 Flash, GLM-4, Qwen2.5, Llama 4 & 3.3, DeepSeek-V3, DeepSeek-R1 7B, Mixtral) across the three difficulty levels:

| Model | Level 1 (Acc) | Level 2 (Acc) | Level 3 (avg score, 0–10) | vs Human (8.74) |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4.1 | 90%+ | >80% | ~7.0 (SOTA) | -1.7 |
| Claude 3.7 Sonnet | 90%+ | >80% | >6 | -2+ |
| Gemini 2.5 Flash | High | 81 (80.0 perturbed) | Mid-High | — |
| DeepSeek-V3 | High | High | Significantly higher than struct. pred. | — |
| Qwen2.5-7B | ~70% | ~50%; Perturbed ↓11.4 | <4 | — |
| Mixtral-8x7B | 70s | ~50; Perturbed ↓8.3 | <4 | — |
| **Human Expert** | — | — | **8.74** | Baseline |

Key Observations: (1) Performance stratification is clear as difficulty increases. (2) All LLMs significantly lag behind human experts on Level 3. (3) The performance gap between closed-source SOTA and small open-source models widens sharply with difficulty.

### Ablation Study
Comparison across four variants (Level 2: Perturbed vs. Knowledge-enhanced vs. Math-abstraction):

| Model | Perturbed $\Delta$ | Knowledge-enhanced $\Delta$ | Math Abstraction $\Delta$ | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4.1 Nano | -9.3 | + | + | Sensitive to perturbation |
| Qwen2.5-7B | **-11.4** | **+16.6** | **+15.5** | Heavy reliance on patterns + knowledge gaps |
| Mixtral-8x7B | -8.3 | + | + | Similar to above |
| Gemini 2.5 Flash | **-1.2** | +2.4 | +2.5 | Robust, balanced knowledge and reasoning |

Level 3 Dimension Analysis: All models perform adequately in "Information Extraction" and moderately in "Multi-objective Decision-making," but are severely deficient in **Domain Reasoning** and **Uncertainty Handling**. Llama 4 scored 0 on Multi-objective Decision-making due to a lack of trade-off analysis, while GPT-4.1 scored 7.5.

### Key Findings
-   **Small models are extremely sensitive to input format**: Qwen2.5-7B dropped 11.4% on perturbed versions but gained 15.5% on math abstraction, suggesting it relies on engineering context for pattern matching but only "calculates" once formulas are extracted.
-   **Knowledge Enhancement > Math Abstraction gains reveal failure causes**: Many models improve significantly when formulas are provided, indicating that "reasoning errors" are often just "not knowing which formula to use."
-   **Structural ability $\neq$ Open-ended ability**: While Level 1/2 scores correlate with Level 3, GPT-4.1/Claude 3.7/DeepSeek-V3 outperform structural predictions on Level 3, while Llama 4 excels at structural tasks but fails on Level 3.
-   **Models generally perform best after Math Abstraction**: This proves that translating natural language engineering descriptions into structured mathematical formulas is the primary bottleneck for current LLMs; pure calculation ability is relatively sufficient.
-   **The Closed vs. Open gap scales with difficulty**: Benchmarks are far from saturated at Level 3, where closed SOTA models outperform small open-source models by more than twofold.

## Highlights & Insights
-   The "Difficulty × Variants × 4D Capabilities" framework provides a clean 3-axis diagnostic, separating engineering ability, general reasoning, and contamination risk into different high-density slices.
-   Isolating "math abstraction" as a dimension quantifies the observation that "engineering problem = math problem + context understanding." It proves that the semantic-to-formal translation step is the true bottleneck.
-   Using 43 mathematical modeling competition tasks ($Level 3$) is a rare and successful attempt to make "open-ended" evaluation comparable and reproducible.
-   The "drop in perturbed score" serves as a convincing metric for contamination detection—reflections of structural reliance vs. robust reasoning.
-   The gap between human experts (8.74) and SOTA (~7.0) on Level 3 provides a quantitative anchor for the current reasoning research gap.

## Limitations & Future Work
-   The benchmark is text-only and lacks multi-modality (drawings, circuit diagrams, tables), which were excluded to avoid interference from vision-processing capabilities.
-   Long-context engineering tasks (system design, parsing long specification documents) were excluded due to varying model context windows.
-   Coverage is limited to three sub-domains; software engineering and detailed civil engineering calculations are not yet included.
-   Level 3 scoring involves LLM judges; despite manual calibration, biases toward specific styles remain a risk.
-   No SFT-friendly training subsets are provided; the benchmark is purely for evaluation.
-   The paper does not deeply analyze whether reasoning models (DeepSeek-R1 / o1) can narrow the gap with humans using longer CoT on Level 3.

## Related Work & Insights
-   **vs MMLU-Pro / SuperGPQA**: These are comprehensive multi-disciplinary benchmarks with small engineering footprints and multiple-choice formats; EngiBench is engineering-focused with free-form and open-ended modeling.
-   **vs EEE-Bench / FEABench**: These are single-discipline and closed-answer; EngiBench is cross-domain with hierarchical and open rubrics.
-   **vs MATH / GSM8K**: These are pure math tests; this paper proves LLMs are strongest in math abstraction, showing the difficulty in engineering is "translation," not "calculation."
-   **vs Prometheus (Kim 2024)**: While Prometheus uses generic rubrics for general capabilities, EngiBench encodes four specific engineering dimensions aligned with expert standards.

## Rating
-   Novelty: ⭐⭐⭐⭐ The orthogonal 3-axis design is a clear innovation for benchmarks; systematic open-ended engineering evaluation via rubrics is a first.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Full matrix evaluation of 16 models across 1,760 questions and 4 variants.
-   Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive figures (Fig 1/3/4) demonstrating stratification.
-   Value: ⭐⭐⭐⭐⭐ Establishes a first-principles coordinate system for LLM engineering evaluation; likely to become a standard benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](../../NeurIPS2025/llm_evaluation/creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)

</div>

<!-- RELATED:END -->
