---
title: >-
  [Paper Note] Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs
description: >-
  [ICML2025][Reasoning][Mathematical Reasoning Benchmark] This work introduces Putnam-AXIOM, a benchmark comprising 522 university-level Putnam competition math problems and 100 programmatic functional variants, which reveals memorization reliance in LLM mathematical reasoning and introduces Teacher-Forced Accuracy (TFA) as an evaluation metric for reasoning quality beyond final answers.
tags:
  - "ICML2025"
  - "Reasoning"
  - "Mathematical Reasoning Benchmark"
  - "Data Contamination"
  - "Functional Variants"
  - "Teacher-Forced Accuracy"
  - "Putnam Competition"
date: 2026-05-08
content_hash: 659dda4f9ff35b6d
---

# Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs

**Conference**: ICML2025  
**arXiv**: [2508.08292](https://arxiv.org/abs/2508.08292)  
**Code**: [brando90/putnam-axiom](https://github.com/brando90/putnam-axiom)  
**Area**: LLM Evaluation / Mathematical Reasoning  
**Keywords**: Mathematical Reasoning Benchmark, Data Contamination, Functional Variants, Teacher-Forced Accuracy, Putnam Competition

## TL;DR

This work introduces Putnam-AXIOM, a benchmark comprising 522 university-level Putnam competition math problems and 100 programmatic functional variants, which reveals memorization reliance in LLM mathematical reasoning and introduces Teacher-Forced Accuracy (TFA) as an evaluation metric for reasoning quality beyond final answers.

## Background & Motivation

**Saturation of Existing Benchmarks**: GPT-4 has achieved 97.1% on GSM8K and 87.92% on the MATH dataset, failing to distinguish reasoning capability differences among frontier models.

**Severe Data Contamination**: Problems in benchmarks like MATH, AGIEval, and OlympiadBench widely exist on the internet and are highly likely to have been included in pre-training data. Models obtain high scores by "memorizing answers" rather than "reasoning".

**Single Evaluation Dimension**: Current mainstream "boxed answer" evaluation only assesses whether the final answer is correct, completely ignoring the quality of the reasoning process—models may happen to get the correct answer by guessing or incorrect derivation.

**Limitations of Existing Hard Benchmarks**: ARB and OlympiadBench contain many proof problems that cannot be automatically graded, requiring expensive human evaluation; PutnamBench focuses on formal theorem proving (Lean/Isabelle/Coq), which has a high barrier to entry.

Core Motivation: A mathematical reasoning benchmark is needed that is **sufficiently difficult, contamination-resistant, automatically-gradeable, and capable of evaluating the reasoning process**.

## Method

### 3.1 Putnam-AXIOM Original Dataset

- **Source**: William Lowell Putnam Mathematical Competition problems from 1938 to 2023.
- **Scale**: 522 problems, covering 11 mathematical areas (geometry, algebra, trigonometry, calculus, linear algebra, combinatorics, probability, number theory, complex numbers, differential equations, and analysis).
- **Difficulty Grading**: Retains the original exam IDs (session A/B + number 1-6, where 6 is the hardest).
- **Modified Boxing**: For 221 (42.3%) problems that originally required proof or multiple answers, a simple computational step is added to produce a unique boxed answer, enabling automatic grading. For example, a problem asking for all values of $n$ satisfying a condition is modified to "find the sum of the first $k$ values of $n$ satisfying the condition".
- **Grading Mechanism**: Uses equivalence functions to convert TeX into SymPy objects to verify mathematical equivalence (e.g., $0.5 = 1/2 = \frac{1}{2}$).

### 3.2 Putnam-AXIOM Variation Dataset

To counter data contamination, programmatic functional variants are created for 100 problems, categorized into two types:

| Variant Type | Quantity | Operation | Answer Change |
|---------|------|------|---------|
| Variable Change | 63 | Replace variable names (e.g., $x \to w$, $y \to v$) | Unchanged |
| Constant Change | 37 | Modify numerical constants + replace variable names | Changed |

Key Design: Each variant is generated via a Python script, capable of producing **infinitely many** new instances of equivalent difficulty. During evaluation, 5 snapshots are randomly sampled, and the mean and 95% confidence intervals are reported.

### 3.3 Teacher-Forced Accuracy (TFA)

TFA directly evaluates the ground-truth reasoning process "understanding" of the model. Given problem $q$ and ground-truth solution token sequence $s_1, s_2, \dots, s_N$, let $\hat{s}_i$ be the model's prediction under teacher forcing:

$$\text{TFA} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{s}_i = s_i]$$

Auxiliary metrics also include:

- **TFCE (Teacher-Forced Cross Entropy)**: $\text{TFCE} = -\frac{1}{N} \sum_{i=1}^{N} \log \mathbb{P}(\hat{s}_i = s_i \mid q, s_1, \dots, s_{i-1})$
- **Perplexity**: $\exp(\text{TFCE})$
- **BPC (Bits Per Character)**: Cross-entropy normalized by character instead of token.

Advantages of TFA: Requires no additional annotated data, no training of a reward model, and only a single forward pass; correlates with final answer accuracy while punishing cases that are "correct by chance but with faulty reasoning".

### 3.4 Data Contamination Simulation Experiment

A LoRA fine-tuned model is used to simulate contamination scenarios:

| Stage | Original Accuracy | Variation Accuracy |
|------|-------------------|------------------|
| Before Fine-Tuning | 23% | 12% |
| After Fine-Tuning | **80%** | 33% |

After fine-tuning, the original accuracy climbs to 80%, while the variant accuracy only rises to 33% $\to$ the model is "memorizing" the original answers rather than learning reasoning capabilities.

## Experimental Setup & Main Results

### Experimental Setup

- Uses the LM Harness evaluation framework with standardized prompt templates (demanding step-by-step reasoning + boxed answers).
- Evaluates 18 models: 7 Base, 7 Instruct/RL, and 4 closed-source models.
- Variation Evaluation: Averages over 5 random snapshots, computing 95% CI.

### Original Dataset Results (Selected from Table 1)

| Model | Correct/Total | Accuracy | TFA |
|------|----------|--------|-----|
| o1-preview | 219/522 | **41.94%** | - |
| GPT-4o | 101/522 | 19.35% | - |
| Claude-3.5 Sonnet | 83/522 | 15.96% | - |
| Qwen2-Math-7B-Instruct | 60/522 | 11.49% | 0.758 |
| GPT-4 | 59/522 | 11.30% | - |
| NuminaMath-7B-Base | 54/522 | 10.34% | 0.742 |
| Gemma-2B-Instruct | 5/522 | 0.95% | 0.634 |

The strongest model, o1-preview, only achieves less than 42%, while most models fall below 10%, indicating that the benchmark has abundant difficulty.

### Variation Dataset Results (Drop in Accuracy)

| Model | Original (100 Problems) | Variation | Relative Drop |
|------|-------------------|-----------|---------|
| o1-preview | 46.8% $\to$ | Decreased by 19.6pp | 46.8% relative drop |
| DeepSeek-R1-Qwen-32B | - | - | **37.5%** maximum drop |
| GPT-4o | - | - | 36% drop |

The confidence intervals of Original and Variation for 10 models do not overlap $\to$ the differences are statistically significant, demonstrating that model performance on original problems is artificially inflated by memorization.

### TFA Metric Analysis

- TFA is highly correlated with boxed accuracy on the MATH dataset, serving as an effective proxy metric for the reasoning process.
- The 18 metrics from ROSCOE have poor comparability across different models; most embedding-based metrics fail to effectively distinguish reasoning quality.
- TFA requires only a forward pass, without extra models or annotations, thus incurring extremely low costs.

### Error Analysis

Although the solutions generated by o1-preview generally progress along a correct logical path, they **lack mathematical rigor**—frequently invoking unproven claims to advance the proof, which would receive very few marks under manual grading. This reveals fundamental deficiencies of LLMs in formal reasoning.

## Limitations & Future Work

1. **TFA relies on reference solutions**: Models might adopt completely different but valid solutions, which TFA would penalize; this is particularly unfair for models fine-tuned to specific styles (like code generation).
2. **Limited variant coverage**: Only 100/522 problems have functional variants (19.2%); some problems cannot generate variants due to a lack of variable constants or non-boxable answers.
3. **Modified Boxing may change problem nature**: 42.3% of the problems were modified for automatic grading, which may introduce bias into the translation process.
4. **Absence of latest models**: Due to the evaluation timeframe, newer models like GPT-4o-mini, Claude-3.5-Sonnet-V2, and DeepSeek-V3 are not included.
5. **English-only evaluation**: The Putnam competition itself is in English, and multilingual mathematical reasoning is not evaluated.
6. **Closed-source models cannot compute TFA**: Lack of access to log probabilities limits the applicability of TFA.

## Related Work & Insights

- **MATH / GSM8K**: Classic but saturated mathematics benchmarks. Putnam-AXIOM surpasses them in both difficulty and contamination resistance.
- **FrontierMath (Srivastava et al., 2024)**: Pioneered the idea of functional variants; this work scales it from high school to university competition level.
- **PutnamBench**: Originates from the same competition but focuses on formal theorem proving (Lean/Isabelle/Coq), being complementary rather than competitive.
- **ROSCOE**: A suite of 18 reasoning metrics, which suffer from poor comparability across models; TFA offers a simpler and more practical alternative.
- **Process Supervision (PRM)**: Requires extensive step-level annotations and auxiliary reward models, whereas TFA serves as a zero-training-cost, lightweight alternative.

## Personal Commentary

**Strengths**:
- Precise problem formulation: Simultaneously addresses the twin pain points of benchmark saturation and data contamination.
- Clever functional variant design: Programmatically generates infinite new instances; simple in principle but highly effective (the 19.6pp drop clearly demonstrates memorization dependency).
- Practical TFA metric: Requires only a single forward pass, needs no annotation, and can be automatically computed, providing a low-cost solution for reasoning process evaluation.
- Rigorous experimental design: LoRA fine-tuning experiments precisely simulate contamination scenarios, and the 5-snapshot + CI evaluation methodology is statistically reliable.

**Weaknesses**:
- Variants cover only 19.2% of the problems, which limits the persuasiveness of the findings.
- The paper classification under model_compression is clearly inappropriate; it actually belongs to the domain of LLMs evaluation/mathematical reasoning.
- Lacks direct comparison with contemporaneous hard benchmarks such as FrontierMath, Minerva, and MathOdyssey.
- The correlation of TFA with accuracy is only validated on the MATH dataset, and has not been verified on Putnam-AXIOM itself.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of functional variants and TFA holds innovative value, though each component is not entirely novel on its own.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 18 models evaluated with fine-tuning contamination simulations and statistical testing are quite comprehensive, though newer models and cross-benchmark comparisons are missing.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, explicit motivation, and informative charts/tables.
- Value: ⭐⭐⭐⭐ — Makes a practical contribution to the field of LLM mathematical reasoning evaluation; GitHub code is open-source and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](../../NeurIPS2025/llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[NeurIPS 2025\] CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](../../NeurIPS2025/llm_reasoning/core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)
- [\[ACL 2025\] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](../../ACL2025/llm_reasoning/enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)
- [\[ICLR 2026\] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](../../ICLR2026/llm_reasoning/the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)

</div>

<!-- RELATED:END -->
