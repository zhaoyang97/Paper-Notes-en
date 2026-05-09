---
title: >-
  [Paper Note] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics
description: >-
  [ICLR 2026][LLM Reasoning][mathematical reasoning] This paper introduces the ContextMATH benchmark, which systematically converts abstract mathematical problems from AIME and MATH-500 into two contextual variants—Scenario Grounding (SG) and Complexity Scaling (CS)—to reveal substantial performance degradation in LLMs on contextual mathematical reasoning. Open-source models drop by 13% on average on SG and 34% on CS. Two complementary performance bottlenecks are identified: **problem formulation** and **reasoning execution**.
tags:
  - ICLR 2026
  - LLM Reasoning
  - mathematical reasoning
  - contextual reasoning
  - problem formulation
  - benchmark evaluation
  - LLM capability assessment
date: 2026-05-08
content_hash: 22ce33133c2190d6
---

# From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics

**Conference**: ICLR 2026
**arXiv**: [2601.23048](https://arxiv.org/abs/2601.23048)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: mathematical reasoning, contextual reasoning, problem formulation, benchmark evaluation, LLM capability assessment

## TL;DR

This paper introduces the ContextMATH benchmark, which systematically converts abstract mathematical problems from AIME and MATH-500 into two contextual variants—Scenario Grounding (SG) and Complexity Scaling (CS)—to reveal substantial performance degradation in LLMs on contextual mathematical reasoning. Open-source models drop by 13% on average on SG and 34% on CS. Two complementary performance bottlenecks are identified: **problem formulation** and **reasoning execution**.

## Background & Motivation

- **State of the Field**: LLMs have achieved near-perfect performance on mathematical benchmarks (approaching full scores on AIME and even reaching IMO gold-medal level), yet this success remains confined to well-defined abstract problems. In real-world applications, mathematical problems frequently appear embedded in narrative contexts—from financial analysis to scientific research—where the mathematical core must be extracted and modeled from concrete descriptions.
- **Limitations of Prior Work**: Existing mathematical benchmarks (e.g., GSM8K, MATH, AIME) almost exclusively target abstract problems. Even when simple narratives are present (e.g., "Jack has 8 pens"), the contextual information is shallow and limited. As a result, a critical capability—**contextual mathematical reasoning**—has remained substantially underexplored.
- **Paper Goals**: To answer the following question: when the mathematical core is embedded within a narrative scenario, how do LLMs perform?

## Method

### Overall Architecture

ContextMATH is constructed from AIME 2024, AIME 2025, and MATH-500 (difficulty ≥ 3). Each original problem is converted into two contextual variants, and systematic evaluation is conducted across 61 models (46 open-source + 15 proprietary).

### Key Designs

**Scenario Grounding (SG)**: Abstract mathematical structures are embedded into narrative scenarios with realistic entities and interactions, while the mathematical reasoning core remains unchanged. For example, variables $(a, b, c)$ are mapped to system components. SG primarily tests whether models can correctly comprehend and apply their mathematical knowledge in the presence of contextual noise.

**Complexity Scaling (CS)**: Explicit conditions are concealed within sub-problems. For instance, a direct statement such as "25 indicator lights" is replaced by "the number of unique pairings of indicator lights is exactly 300." CS not only introduces contextualization but also requires the model to first solve sub-problems to recover the original conditions, simulating real-world scenarios where constraints are given indirectly and reducing reliance on surface-level pattern matching.

**Problem Formulation Evaluation**: To analyze failure causes in depth, three dedicated metrics are designed to assess a model's ability to extract mathematical formulations from context:
- **Formulation Accuracy**: The proportion of cases in which the model correctly translates the context into a mathematical problem.
- **Formulation Necessity**: $P(\text{formulation correct} \mid \text{answer correct})$, measuring the degree to which correct formulation is necessary for correct solution.
- **Formulation Sufficiency**: $P(\text{answer correct} \mid \text{formulation correct})$, measuring whether correct formulation reliably leads to a correct answer.

**Benchmark Construction**: Contextual variants are generated using o1-mini via multi-step prompting, then independently reviewed by three experts holding advanced degrees in computer science with competitive mathematics backgrounds, ensuring mathematical equivalence, narrative plausibility, and unambiguity. SG variants average 133 words and CS variants average 176 words.

### Loss & Training

In the training experiments, two strategies are explored:

**End-to-End Fine-Tuning**: Based on the DeepMath-103K dataset, Qwen3-Base models are fine-tuned via SFT under three data configurations:
- SFT_Ori: original mathematical problems only (50k)
- SFT_Syn: synthetic contextual problems only (50k)
- SFT_Mix: mixture of both (100k)

**Dedicated Formulation Model Training**: Context–original problem pairs are used as training data to train a specialized formulation model, which is then pipelined with a solver. Both untuned and fine-tuned settings are evaluated.

## Key Experimental Results

### Main Results

Performance degradation of proprietary models on ContextMATH:

| Model | AIME24-Ori | AIME24-SG | AIME24-CS | AIME25-Ori | AIME25-SG | AIME25-CS |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| GPT-5 | 90.0% | 83.3% (-7%) | 80.0% (-11%) | 90.0% | 80.0% (-11%) | 66.7% (-26%) |
| DeepSeek-R1 | 93.3% | 70.0% (-25%) | 66.7% (-29%) | 86.7% | 73.3% (-15%) | 53.3% (-38%) |
| Gemini 2.5 Pro | 83.3% | 73.3% (-12%) | 76.7% (-8%) | 83.3% | 56.7% (-32%) | 50.0% (-40%) |
| o3 | 83.3% | 70.0% (-16%) | 66.7% (-20%) | 76.7% | 70.0% (-9%) | 60.0% (-22%) |
| Qwen3-32B | 81.2% | 67.9% (-16%) | 57.1% (-30%) | 70.0% | 54.4% (-22%) | 45.0% (-36%) |

Key finding: even GPT-5 drops by 26% on AIME25-CS; DeepSeek-R1 drops by 38% on AIME25-CS.

### Ablation Study

| Dimension | Key Finding |
|-----------|------------|
| Formulation analysis (GPT-5) | Average formulation accuracy 81.4%, necessity 85.6%, sufficiency 82.7% |
| Model scale effect | Qwen3-0.6B formulation accuracy 42.8% → Qwen3-32B 75.0%; scale helps but does not resolve the issue |
| Error type analysis | Formulation errors account for ~80%, far exceeding computational, logical, and other errors |
| SFT effect (14B) | SFT_Mix improves AIME24-SG from 11.0% → 52.5%, with an average gain of 31.9% |
| SFT_Ori vs. SFT_Syn | Contextual data (SFT_Syn) is more effective on contextual problems; mixed data (SFT_Mix) is globally optimal |
| Dedicated formulation model | Direct solving (57.7%) > untuned formulation + solving (56.2%) > fine-tuned formulation + solving (24.6%) |
| Generalization | SFT_Mix also improves performance on AMC23 and Math-Perturb without degrading abstract reasoning |

### Key Findings

1. **Contextualization is a universal bottleneck**: Performance degradation from abstract to contextual settings is consistent across both open-source and proprietary models; CS yields more severe drops than SG.
2. **Formulation is the primary bottleneck**: Samples solved correctly exhibit substantially higher formulation accuracy than the average (high necessity); however, correct formulation does not always lead to correct answers (insufficient sufficiency).
3. **Two complementary bottlenecks**: Formulation ability and reasoning ability represent two independent yet complementary bottlenecks; both improve with model scale but are not eliminated.
4. **Risk of over-specialization**: Further SFT/RL may overfit to standard formats, resulting in greater performance drops on SG/CS.
5. **Formulation cannot be trained in isolation**: Pipelining a fine-tuned dedicated formulation model with a solver causes performance collapse (from 57.7% to ~22%), indicating that formulation ability is difficult to acquire independently through paired supervision.

## Highlights & Insights

- **Precise problem definition**: The paper clearly identifies "contextual mathematical reasoning" as a distinct and underexplored capability dimension, separate from simple mathematical word problems.
- **Elegant experimental design**: SG preserves reasoning difficulty while adding only contextual framing; CS additionally introduces the cognitive burden of condition recovery. The two variants complement each other to form a comprehensive evaluation spectrum.
- **In-depth bottleneck analysis**: A probabilistic framework of necessity and sufficiency precisely characterizes the dependency between formulation and reasoning.
- **Large-scale evaluation**: Evaluation across 61 models spanning from 0.6B to GPT-5 provides a complete scale spectrum; within-family comparisons (base → SFT → RL) offer systematic evidence on training strategies.

## Limitations & Future Work

1. Contextual variants are generated by LLMs and then reviewed by human experts, which may introduce distributional bias; future work could explore more diverse context generation approaches.
2. CS variants are not constructed for MATH-500 (some problems are too simple), leaving room for broader coverage.
3. The analysis of why dedicated formulation model training fails is insufficiently deep; alternative training strategies (e.g., RL, multi-task learning) remain unexplored.
4. Evaluation is limited to final answer correctness, without analyzing the quality of intermediate reasoning chains.

## Related Work & Insights

This paper complements traditional mathematical benchmarks such as GSM8K and MATH, advancing evaluation from "can the model solve equations" to "can the model derive equations from narratives." An interesting contrast arises with Cheng et al. (2025): the latter finds that formulation is not the primary bottleneck on simple benchmarks, whereas this paper demonstrates that formulation is the foremost bottleneck in complex contextual settings. This suggests that LLM mathematical capability evaluation must distinguish between "solving formatted problems" and "understanding problems."

## Rating

- Novelty: ⭐⭐⭐⭐ — The problem is clearly defined and significant; the SG/CS dual-variant design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large-scale evaluation across 61 models, with multi-dimensional analysis and training experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorously structured, with analysis that builds progressively and highly informative figures.
- Value: ⭐⭐⭐⭐⭐ — Reveals a core and previously overlooked weakness in LLM mathematical reasoning.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[ICLR 2026\] Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)

<!-- RELATED:END -->
