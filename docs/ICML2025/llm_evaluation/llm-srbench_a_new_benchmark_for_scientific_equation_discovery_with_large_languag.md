---
title: >-
  [Paper Note] LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs
description: >-
  [ICML2025 Oral][LLM Evaluation][Symbolic Regression] Proposes the LLM-SRBench benchmark (239 problems across 4 scientific domains) that prevents LLM memorization through equation transformation (LSR-Transform) and synthetic problems (LSR-Synth). The current best method achieves only 31.5% symbolic accuracy.
tags:
  - "ICML2025 Oral"
  - "LLM Evaluation"
  - "Symbolic Regression"
  - "Scientific Equation Discovery"
  - "LLM Benchmark"
  - "Memorization Prevention"
  - "Equation Transformation"
date: 2026-05-08
content_hash: 062e14a3e26c91a8
---

# LLM-SRBench: A New Benchmark for Scientific Equation Discovery with LLMs

**Conference**: ICML2025 Oral  
**arXiv**: [2504.10415](https://arxiv.org/abs/2504.10415)  
**Code**: [GitHub - LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench)  
**Area**: LLM Evaluation  
**Keywords**: Symbolic Regression, Scientific Equation Discovery, LLM Benchmark, Memorization Prevention, Equation Transformation

## TL;DR
Proposes the LLM-SRBench benchmark (239 problems across 4 scientific domains) that prevents LLM memorization through equation transformation (LSR-Transform) and synthetic problems (LSR-Synth). The current best method achieves only 31.5% symbolic accuracy.

## Background & Motivation

### Key Challenge

**Key Challenge**: LLMs embed a vast amount of scientific knowledge that can assist in hypothesis generation. However, existing benchmarks (such as Feynman/SRBench) use textbook equations, which LLMs might simply "memorize" rather than "discover."

### Limitations of Prior Work

**Limitations of Prior Work**: Experiments show that the numerical error curve of LLMs on Feynman problems drops sharply (a signal of memorization) rather than improving gradually (a signal of genuine discovery).

### Goal

To design a memorization-resistant test suite while leveraging the scientific priors of LLMs to simulate real-world discovery scenarios.

## Method

### LSR-Transform
Transforms known physical equations into uncommon mathematical representations:
- Symbolically alters input-output mappings.
- Generates rare mathematical formulations of the same physical problem.
- Challenges the reasoning capability of LLMs beyond mere memorization.

### LSR-Synth
Introduces synthetic discovery-driven problems:
- Requires data-driven reasoning.
- Does not rely on known equations.
- Evaluates genuine discovery capabilities.

### Evaluation System
- Data fidelity (numerical fitting)
- Symbolic accuracy (exact matching)
- Computational efficiency

## Key Experimental Results

### Current Method Performance


### Main Results

| Method | Feynman Accuracy | LSR-Transform | LSR-Synth |
|------|-------------|-------------|----------|
| Llama-3.1-8B Direct Sampling | High (Memorization) | Low | Low |
| Best LLM Method | ~80% | ~35% | ~28% |
| **Best Overall** | - | - | **31.5%** |

### Differentiation of Memorization vs. Discovery


### Ablation Study

| Feature | Feynman (Old) | LLM-SRBench (New) |
|------|-----------|---------------|
| Numerical Error Curve | Sharp drop | Gradual improvement |
| Symbolic Error | Extremely low | Significantly high |
| Is Memorized | Highly likely | Unlikely |

### Key Findings
1. The current best method achieves only 31.5% on LLM-SRBench—far lower than on Feynman.
2. Open-source LLMs perform better on LSR-Transform than on Synth (some priors remain useful).
3. Closed-source LLMs (GPT-4) are slightly superior, but the gap remains limited.
4. Equation transformations successfully prevent simple memorization.

## Highlights & Insights

1. Accurately punctures the illusion of LLM success in equation discovery—revealing that most achievements stem from memorization.
2. Clever design of LSR-Transform: distinct mathematical formulations representing the same physical problem.
3. The scale of 239 problems across 4 domains significantly exceeds prior 5-problem custom sets.
4. Numerical error curve analysis serves as a simple yet effective tool to distinguish memorization from genuine discovery.
5. Provides a standardized evaluation platform for the LLM+SR community.

## Limitations & Future Work

1. The performance upper bound of 31.5% indicates that the problems are extremely challenging, or the benchmark might be overly difficult.
2. Transformation methods may introduce unnatural mathematical forms.
3. Validated only across 4 scientific domains.
4. Lacks baseline comparisons with human scientists.
5. Systematic evaluation of multi-step reasoning and tool use is currently insufficient.

## Related Work & Insights

- Relationship with SRBench/SRSD: LLM-SRBench serves as their LLM-aware counterpart.
- Complementarity with PAN+SR: PAN addresses high dimensions, while LLM-SRBench addresses memorization.
- Insights: Memorization issues should also be seriously considered in other LLM evaluation tasks.

## Rating
- Novelty: 5.0/5 — First memorization-resistant LLM equation discovery benchmark.
- Experimental Thoroughness: 4.5/5 — Evaluates multiple methods and LLMs.
- Writing Quality: 4.5/5
- Value: 5.0/5 — Significant contribution to LLM evaluation methodology.

## Supplementary Analysis

### Memorization Detection Metrics
The shape of the numerical error curve is a key differentiator: a sharp drop indicates memorization, while a gradual improvement signifies genuine discovery. This analysis tool itself is a key contribution.

### Design Details of LSR-Transform
Symbolically transforms the input-output mappings of known equations to generate uncommon mathematical formulations of the same physical problem. For example, transforming $F=ma$ into logarithmic or integral forms.

### Complementarity with SRBench
SRBench evaluates traditional symbolic regression methods, while LLM-SRBench specifically evaluates LLM-based methods—the two complement each other.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](../../ACL2026/llm_evaluation/researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ACL 2025\] MisMatched: A Benchmark for Scientific Natural Language Inference](../../ACL2025/llm_evaluation/a_mismatched_benchmark_for_scientific_natural_language_inference.md)
- [\[ACL 2025\] YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering](../../ACL2025/llm_evaluation/yescieval_llm_judge_science.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](../../ACL2025/llm_evaluation/elaboration_competitive_programming.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)

</div>

<!-- RELATED:END -->
