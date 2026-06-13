---
title: >-
  [Paper Note] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff
description: >-
  [ACL 2026][LLM Evaluation][Overthinking] This paper proposes LLMThinkBench, a benchmark for systematically evaluating the efficiency of LLMs in basic math reasoning. It introduces the Overthinking Score (a harmonic mean…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Overthinking"
  - "Basic Math Reasoning"
  - "Accuracy-Efficiency Tradeoff"
  - "Reasoning tokens"
  - "Benchmark"
date: 2026-05-08
content_hash: ab9596d46c8a5cd2
---

# Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff

**Conference**: ACL 2026  
**arXiv**: [2507.04023](https://arxiv.org/abs/2507.04023)  
**Code**: [GitHub](https://github.com/ctrl-gaurav/LLMThinkBench)  
**Area**: LLM Evaluation  
**Keywords**: Overthinking, Basic Math Reasoning, Accuracy-Efficiency Tradeoff, Reasoning tokens, Benchmark

## TL;DR

This paper proposes LLMThinkBench, a benchmark for systematically evaluating the efficiency of LLMs in basic math reasoning. It introduces the Overthinking Score (a harmonic mean of accuracy and token efficiency) and evaluates 53 LLMs using 14 dynamically generated deterministic math tasks. The study reveals that reasoning models generate approximately $18\times$ more tokens on average but sometimes achieve lower accuracy, with diminishing returns for scaling reasoning budgets.

## Background & Motivation

**Background**: LLMs perform exceptionally well on complex math benchmarks (GSM8K, MATH), and reasoning models further enhance performance through inference-time scaling (chain-of-thought). However, their performance and efficiency on basic mathematical operations have not been systematically evaluated.

**Limitations of Prior Work**: (1) Models scoring 90%+ on complex benchmarks may score below 40% on basic addition—complex benchmark performance does not transfer to basic operations; (2) Reasoning models generate excessively long reasoning chains for simple problems (e.g., hundreds of tokens to explain carrying in $234+567$), wasting computational resources and sometimes reducing accuracy; (3) Existing evaluations focus solely on accuracy, ignoring computational waste; (4) Static benchmarks risk data contamination; (5) Lack of metrics that jointly measure accuracy and efficiency.

**Key Challenge**: Reasoning models are trained to "think more" to improve performance, but more thinking is harmful on basic tasks—models confuse explanation with understanding, producing long texts that appear to reason but lack problem-solving capability.

**Goal**: (1) Formalize the accuracy-redundancy tradeoff; (2) Propose the Overthinking Score metric; (3) Establish a dynamically generated evaluation protocol; (4) Conduct a large-scale empirical study on the reasoning efficiency of 53 LLMs.

**Key Insight**: Focus on 14 deterministic basic math tasks (sorting, summation, multiplication, finding maximums, etc.) that have unique correct answers and known computational complexity, allowing for the precise measurement of the relationship between accuracy and redundancy.

**Core Idea**: More reasoning tokens $\neq$ better math reasoning—on basic tasks, redundant generation in reasoning models not only wastes computation but may also reduce accuracy due to error accumulation and self-contradiction.

## Method

### Overall Architecture

The LLMThinkBench framework consists of four core components: (1) A task space containing 14 deterministic basic math tasks; (2) Formalization of the accuracy-redundancy 2D space; (3) The Overthinking Score metric; (4) An installable open-source tool (PyPI: llmthinkbench) supporting dynamic test generation, multi-backend inference, hierarchical answer extraction, and report generation.

### Key Designs

1. **Overthinking Score**:

    - Function: Unified measurement of accuracy and token efficiency as a single metric.
    - Mechanism: Define Token efficiency $E_{t,i} = 1 - \frac{\bar{T}_i - T_{min}}{T_{max} - T_{min}}$, then combine accuracy and efficiency using the harmonic mean $\mathcal{O}_i = \frac{2 \cdot A_i \cdot E_{t,i}}{A_i + E_{t,i}}$. The harmonic mean severely penalizes imbalance—90% accuracy + 10% efficiency yields only 0.18, mientras 60%+60% yields 0.60.
    - Design Motivation: The arithmetic mean fails to sufficiently penalize extreme imbalance (90% accuracy + 10% efficiency still gives 0.55). The harmonic mean penalizes imbalance the most among all symmetric homogeneous means.

2. **Dynamic Test Generation Protocol**:

    - Function: Eliminate data contamination risks and ensure evaluation fairness.
    - Mechanism: Dynamically generate test instances based on reproducible seeds. List lengths are sampled from $\{8, 16, 32, 64\}$, values from $\text{Uniform}[-1000, 1000]$, with 1000 samples per fold and 3-fold cross-validation (open-source models), or 100 samples (closed-source models due to cost). Each model is tested on 42,000 unique problems.
    - Design Motivation: Static benchmarks are prone to training data contamination; dynamic generation ensures fresh data for every evaluation.

3. **Hierarchical Answer Extraction System**:

    - Function: Reliably extract answers from diverse model outputs.
    - Mechanism: A four-level extraction strategy—(1) Prioritize content within \boxed{}; (2) Parse explicit answer markers ("The answer is..."); (3) Extract from code blocks or Markdown; (4) Task-specific heuristics as a fallback. Validated on 5000+ responses with a 98.7% success rate.
    - Design Motivation: Output formats vary significantly across models; reliable extraction is a prerequisite for fair evaluation.

### Loss & Training

Does not involve model training. Evaluates existing models using public weights or APIs. Evaluation covers 53 models, including base, instruction-tuned, reasoning, and quantized variants.

## Key Experimental Results

### Main Results

**Overthinking Score Comparison of Representative Models**

| Model | Parameters | Accuracy | Overthinking Score | Avg Output Tokens |
|------|------|--------|-------------------|---------------|
| Phi-4 | 14B | 78.92% | **0.863** | 378.6 |
| Phi-4-reasoning-plus | 14B | 69.54% | 0.234 | 6,780.7 |
| Qwen3-14B | 14B | 86.52% | 0.727 | 3,607.6 |
| Qwen3-0.6B | 0.6B | 49.99% | 0.545 | 3,162.8 |

### Ablation Study

**Reasoning Budget Constraint Experiment (Qwen3 Reasoning Model)**

| Configuration | Accuracy |
|------|--------|
| Full Budget | 72% |
| 1024 Token Limit | 44% (-28%) |
| Reasoning Budget low→medium→high (GPT-5/o series) | Accuracy gain ≈ 0 |

**Quantization Experiment (Qwen2.5 Family)**

| Configuration | Accuracy Change |
|------|-----------|
| FP16 → 8-bit | Large models almost unchanged |
| FP16 → 4-bit | Slight drop for large models, significant drop for small models |

### Key Findings

- Basic Math Paradox: Models scoring 95%+ on GSM8K perform below 75% on these tasks—complex benchmark performance does not represent basic math capability.
- Reasoning models generate 6,780 tokens on average vs 378 tokens for standard models ($18\times$), yet achieve lower accuracy (Phi-4-reasoning-plus 69.54% vs Phi-4 78.92%).
- Overthinking Score reveals efficiency traps hidden by accuracy: Phi-4 (0.863) far exceeds Phi-4-reasoning-plus (0.234).
- Reasoning models suffer "catastrophic collapse" under token constraints—dropping from 72% to 44%, indicating that reasoning ability is deeply tied to long-chain inference.
- Scaling reasoning budgets yields diminishing returns—the accuracy gain for GPT-5/o series from low to high effort is near zero.
- Quantization preserves basic reasoning ability, suggesting overthinking stems from training strategies rather than hardware limitations.

## Highlights & Insights

- Overthinking Score is an elegant and informative metric—the strict penalty of the harmonic mean distinguishes "efficiently correct" from "redundantly correct."
- The "basic math paradox" is a significant finding—it challenges the assumption that "high complex benchmark scores = strong mathematical capability."
- Dynamic test generation combined with open-source tools (PyPI package and leaderboard) makes results reproducible and easily extensible.

## Limitations & Future Work

- Currently covers 14 deterministic math tasks; does not cover more complex symbolic reasoning or non-mathematical domains.
- Token efficiency normalization depends on global maximum/minimum values within the evaluation set, which may be affected by extreme outliers.
- Specific patterns of overthinking (e.g., the proportion of error accumulation vs. self-contradiction) have not yet been analyzed.
- Did not explore how to train reasoning models that are both accurate and efficient.

## Related Work & Insights

- **vs ThoughtTerminator/Self-Braking**: These works propose strategies to mitigate overthinking; this paper provides the metric to quantify it—measurement is a prerequisite for intervention.
- **vs GSM8K/MATH Benchmarks**: These benchmarks focus primarily on accuracy; this paper supplements them with an efficiency dimension.
- **vs Graph of Thoughts/LogicPuzzleRL**: These methods enhance complex reasoning but do not address overthinking in basic operations.

## Rating

- Novelty: ⭐⭐⭐⭐ Overthinking Score is a novel and useful metric; the basic math paradox is a significant discovery.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 53 models, quantization analysis, budget constraints, and dynamic generation make it comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are rigorous and experimental narratives are clear.
- Value: ⭐⭐⭐⭐⭐ Provides a standardized tool and deep insights into the efficiency evaluation of reasoning models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[AAAI 2026\] Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy](../../AAAI2026/llm_evaluation/do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)
- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](../../ICLR2026/llm_evaluation/benchmarking_overton_pluralism_in_llms.md)

</div>

<!-- RELATED:END -->
