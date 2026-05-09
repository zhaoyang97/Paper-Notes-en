---
title: >-
  [Paper Note] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics
description: >-
  [NeurIPS 2025][LLM Reasoning][mathematical reasoning] This paper introduces RealMath, a **continuously refreshable** benchmark that automatically extracts verifiable mathematics problems from arXiv papers and Math StackExchange, designed to evaluate LLMs on real-world research-level mathematical tasks.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - mathematical reasoning
  - research-level mathematics
  - benchmark
  - data contamination
  - automatic evaluation
date: 2026-05-08
content_hash: b28014ed11bacd57
---

# RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics

**Conference**: NeurIPS 2025
**arXiv**: [2505.12575](https://arxiv.org/abs/2505.12575)
**Code**: [GitHub](https://github.com/) / [HuggingFace](https://huggingface.co/)
**Area**: LLM Reasoning / Mathematical Benchmark Evaluation
**Keywords**: mathematical reasoning, research-level mathematics, benchmark, data contamination, automatic evaluation

## TL;DR
This paper introduces RealMath, a **continuously refreshable** benchmark that automatically extracts verifiable mathematics problems from arXiv papers and Math StackExchange, designed to evaluate LLMs on real-world research-level mathematical tasks.

## Background & Motivation
**State of the Field**: Existing mathematical reasoning benchmarks primarily draw from three categories: (1) curriculum/competition problems (GSM8K, MATH, AIME, IMO); (2) formal theorem proving (LeanDojo, MiniF2F); (3) expert-curated extremely difficult problems (FrontierMath, HLE).

**Limitations of Prior Work**: These benchmarks cover only a narrow slice of mathematical practice — competition mathematics ≠ research mathematics; formal proofs ≠ day-to-day mathematical research; extremely hard problems address only the frontier capabilities of human experts.

**Root Cause**: Problems encountered in real mathematical research differ fundamentally from competition problems in structure, topic, and difficulty, yet existing benchmarks fail to reflect the practical value of LLMs as assistants in authentic research settings.

**Paper Goals**: Construct a mathematical reasoning benchmark that reflects real research practice, supports automatic evaluation, and can be continuously refreshed to resist data contamination.

**Starting Point**: Automatically extract mathematical theorems with definitive answers from arXiv papers and StackExchange, and convert them into QA pairs.

**Core Idea**: An automated pipeline extracts "constructive theorems" (those with unique, exact answers) from academic papers, serving as a continuously refreshable benchmark for evaluating LLMs on research-level mathematics.

## Method

### Overall Architecture
A five-stage automated data collection pipeline: paper retrieval → LaTeX source extraction → constructive theorem identification → QA pair generation → easy-question filtering.

### Key Designs
1. **Paper Retrieval and Parsing**: Mathematics-related papers are retrieved in bulk from the arXiv API (approximately 4,000 papers over a five-month window), with LaTeX source downloaded and parsed to preserve mathematical notation.
2. **Constructive Theorem Identification**: An LLM (o3-mini) serves as a judge to screen extracted theorems for those possessing a **unique, exact answer**. Theorems involving inequalities, multiple solutions, or non-constructive proofs are excluded. Approximately 407 theorems are selected from ~14,747 candidates.
3. **QA Pair Generation**: Selected theorems are converted into question–answer pairs; the context preceding each theorem in the paper (from the introduction up to the theorem statement) is retained as contextual input.
4. **Quality Filtering**: An LLM review filters out samples whose answers are obvious or trivial, retaining approximately 280 high-quality QA pairs per five-month window.
5. **Continuous Refresh**: The pipeline runs continuously, generating 70+ new samples per month and automatically resisting data contamination.

### Design Criteria
- **Authentic application focus**: Sourced from real research papers rather than artificially constructed
- **Automatic verification**: Only problems with exact numerical or symbolic answers are retained
- **Continuous collection**: Synchronized with newly published papers to avoid contamination

### Dataset Scale and Sources

| Dataset | Time Span | QA Pairs |
|--------|---------|----------|
| Math.arXiv | 2022.05–2022.09 + 2024.12–2025.03 | 633 |
| CS.arXiv | 2022.05–2023.10 | 111 |
| Math.StackExchange | 2024.04–2025.03 | 542 |

### Pipeline Stage Statistics (Example: 4,000 Papers)

| Stage | Output |
|------|--------|
| Paper retrieval | 4,000 papers |
| LaTeX source extraction | 3,922 papers |
| All theorems extracted | 14,747 |
| Constructive theorems confirmed | 407 |
| QA pairs generated | 401 |
| After easy-question filtering | 280 |

## Key Experimental Results

### Main Results

| Model | Math.arXiv | CS.arXiv | Math.StackExchange |
|------|-----------|----------|-------------------|
| o3 | **49.1** | **44.1** | 70.7 |
| o4-mini | 43.4 | 42.3 | **70.8** |
| Gemini 2.5-pro | 32.5 | 25.2 | 60.9 |
| DeepSeek-R1 | 30.5 | 31.5 | 62.2 |
| Claude 3.7-Sonnet | 34.1 | 31.5 | 61.1 |
| Grok 3 | 29.5 | 25.2 | 54.8 |
| Claude 3.5-Sonnet | 18.3 | 16.2 | 37.6 |
| Llama 3.1-405B | 16.4 | 15.3 | 32.1 |
| GPT-4o-mini | 12.5 | 7.2 | 40.8 |

### Difficulty-Stratified Analysis

| Difficulty | o3 Accuracy |
|------|----------|
| Easy | 97.5% |
| Medium | 81.4% |
| Hard | 27.9% |

### Key Findings
1. **LLMs perform surprisingly well on research mathematics**: o3 achieves 49.1% on Math.arXiv, substantially higher than its performance on extreme benchmarks such as FrontierMath, suggesting LLMs may already serve as valuable assistants to mathematicians.
2. **Models exhibit distinct capability profiles**: o3 excels in highly theoretical areas such as representation theory and number theory, while Gemini 2.5-pro performs better on applied domains such as machine learning and optimization — the two models' strongest and weakest domains are nearly complementary.
3. **Context is not always necessary**: o4-mini achieves 21.6% on CS.arXiv without context (vs. 42.3% with context), indicating that models can infer some notation and concepts independently.
4. **No evidence of data contamination**: Models perform better on 2025 papers than on 2022 papers.
5. **Fine-tuning yields limited gains**: Fine-tuning GPT-4o-mini on 500 samples produces no accuracy improvement, suggesting the bottleneck lies in a lack of specialized mathematical knowledge and skills.
6. **Primary error types**: Reasoning errors > conceptual misunderstandings > failure to identify key insights.

## Highlights & Insights
1. **Paradigm innovation**: This is the first work to propose an automated pipeline for constructing a mathematics benchmark from real academic papers, departing from the paradigms of competition problems and expert-curated construction.
2. **Sustainability**: The continuously auto-refreshable design is more sustainable and reproducible than FrontierMath's private, fixed test set.
3. **Practical insight**: The results reveal that LLMs may already be practically useful for research-level mathematics (~50% accuracy), whereas prior benchmarks such as FrontierMath (near 0%) conveyed an overly pessimistic signal.
4. **Data quality**: A high-quality rate of 94% is achieved without manual annotation, yielding 633 high-quality samples from 9,000+ papers.
5. **Divergent capability profiles**: Model performance varies substantially and complementarily across mathematical subfields; o3 and Gemini 2.5-pro exhibit nearly opposite strongest and weakest domains.
6. **Theorem-to-QA conversion design**: Reformulating theorems as "context + QA" pairs preserves original mathematical notation and prerequisites, offering greater scalability than direct theorem-proving evaluation.

## Limitations & Future Work
1. **Constructive problems only**: Proof problems, inequalities, and open-ended questions — important components of mathematical research — are excluded.
2. **Dependence on paper correctness**: The pipeline assumes theorems in arXiv papers are correct, though arXiv papers are not peer-reviewed.
3. **Lower quality of StackExchange data**: User-submitted content frequently contains errors or ambiguous descriptions.
4. **Relatively high initial performance**: The overall accuracy is comparatively high (o3: 49%), which may pose a ceiling effect concern.
5. **Evaluation scope**: The benchmark focuses solely on theorem verification, without covering proof generation, mathematical modeling, or other capabilities.

## Related Work & Insights
- **vs. FrontierMath**: FrontierMath targets extremely hard problems with a private test set; RealMath spans the full difficulty range with a publicly refreshable collection.
- **vs. MathConstruct**: Both adopt a constructive problem approach, but RealMath extracts problems directly from papers rather than designing them manually.
- **vs. SWE-bench**: RealMath parallels the trend in software engineering of evaluating models on real-world tasks, constituting an "in the wild" evaluation for mathematics.

## Rating
- Novelty: ⭐⭐⭐⭐ (new paradigm for constructing benchmarks from real papers)
- Experimental Thoroughness: ⭐⭐⭐⭐ (covers 10+ models, multiple data sources, and multi-dimensional analysis)
- Writing Quality: ⭐⭐⭐⭐ (clear and fluent, with well-motivated arguments)
- Value: ⭐⭐⭐⭐ (fills a gap in research-level mathematical evaluation; pipeline is continuously reusable)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)
- [\[NeurIPS 2025\] Note 6: Self-Evaluating LLMs - Step-Level Confidence Estimation for Multi-Step Tasks](value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](../../ACL2026/llm_reasoning/challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[NeurIPS 2025\] The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness](the_hawthorne_effect_in_reasoning_models_evaluating_and_steering_test_awareness.md)

<!-- RELATED:END -->
