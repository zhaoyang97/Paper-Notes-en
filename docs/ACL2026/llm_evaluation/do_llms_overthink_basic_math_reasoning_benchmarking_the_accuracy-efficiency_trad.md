---
title: >-
  [Paper Note] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff
description: >-
  [ACL 2026][LLM Evaluation][overthinking] This paper presents LLMThinkBench, a benchmark for systematically evaluating the efficiency of LLMs on basic mathematical reasoning. It introduces the Overthinking Score — a harmo…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "overthinking"
  - "basic math reasoning"
  - "accuracy-efficiency tradeoff"
  - "reasoning tokens"
  - "benchmarking"
date: 2026-05-08
content_hash: 1a96ccb21dba9b7b
---

# Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff

**Conference**: ACL 2026
**arXiv**: [2507.04023](https://arxiv.org/abs/2507.04023)
**Code**: [GitHub](https://github.com/ctrl-gaurav/LLMThinkBench)
**Area**: LLM Evaluation
**Keywords**: overthinking, basic math reasoning, accuracy-efficiency tradeoff, reasoning tokens, benchmarking

## TL;DR

This paper presents LLMThinkBench, a benchmark for systematically evaluating the efficiency of LLMs on basic mathematical reasoning. It introduces the Overthinking Score — a harmonic mean of accuracy and token efficiency — and evaluates 53 LLMs across 14 deterministically generated math tasks. Results show that reasoning models generate on average ~18× more tokens yet sometimes achieve lower accuracy, and that scaling inference budgets yields diminishing returns.

## Background & Motivation

**Background**: LLMs achieve strong performance on complex math benchmarks (GSM8K, MATH), and reasoning models further boost performance through inference-time scaling (chain-of-thought). However, the performance and efficiency of these models on basic mathematical operations have not been systematically evaluated.

**Limitations of Prior Work**: (1) Models scoring 90%+ on complex benchmarks may fall below 40% on basic addition — strong performance on complex benchmarks does not transfer to elementary operations. (2) Reasoning models generate excessively long reasoning chains for simple problems (e.g., generating hundreds of tokens to explain the carry principle for 234+567), wasting compute and sometimes reducing accuracy. (3) Existing evaluations focus solely on accuracy while ignoring computational waste. (4) Static benchmarks are susceptible to data contamination. (5) No metric jointly captures accuracy and efficiency.

**Key Challenge**: Reasoning models are trained to "think more" to improve performance, but on basic tasks, more thinking is counterproductive — models conflate explanation with understanding, producing lengthy outputs that superficially resemble reasoning yet lack genuine problem-solving capability.

**Goal**: (1) Formalize the accuracy-verbosity tradeoff; (2) propose the Overthinking Score metric; (3) establish a dynamically generated evaluation protocol; (4) conduct large-scale empirical study of reasoning efficiency across 53 LLMs.

**Key Insight**: The work focuses on 14 deterministic basic math tasks (sorting, summation, multiplication, argmax, etc.) with unique correct answers and known computational complexity, enabling precise measurement of the accuracy-verbosity relationship.

**Core Idea**: More reasoning tokens ≠ better mathematical reasoning — on basic tasks, the verbose generation of reasoning models not only wastes compute but can reduce accuracy through error accumulation and self-contradiction.

## Method

### Overall Architecture

The LLMThinkBench framework comprises four core components: (1) a task space of 14 deterministic basic math tasks; (2) a formal two-dimensional accuracy-verbosity space; (3) the Overthinking Score metric; and (4) an installable open-source tool (PyPI: llmthinkbench) supporting dynamic test generation, multi-backend inference, hierarchical answer extraction, and report generation.

### Key Designs

1. **Overthinking Score**:

    - **Function**: Unifies accuracy and token efficiency into a single metric.
    - **Mechanism**: Token efficiency is defined as $E_{t,i} = 1 - \frac{\bar{T}_i - T_{min}}{T_{max} - T_{min}}$, then combined with accuracy via harmonic mean: $\mathcal{O}_i = \frac{2 \cdot A_i \cdot E_{t,i}}{A_i + E_{t,i}}$. The harmonic mean heavily penalizes imbalance — 90% accuracy + 10% efficiency yields only 0.18, whereas 60% + 60% yields 0.60.
    - **Design Motivation**: The arithmetic mean insufficiently penalizes extreme imbalance (90% accuracy + 10% efficiency still yields 0.55). Among all symmetric homogeneous means, the harmonic mean maximally penalizes imbalance.

2. **Dynamic Test Generation Protocol**:

    - **Function**: Eliminates data contamination risk and ensures evaluation fairness.
    - **Mechanism**: Test instances are dynamically generated from reproducible seeds. List lengths are sampled from {8, 16, 32, 64}, values from Uniform[−1000, 1000], with 1,000 samples per fold and 3-fold cross-validation (open-source models) or 100 samples (closed-source models, cost-constrained). Each model is evaluated on 42,000 unique problems.
    - **Design Motivation**: Static benchmarks are prone to training data contamination; dynamic generation ensures fresh data for each evaluation.

3. **Hierarchical Answer Extraction System**:

    - **Function**: Reliably extracts answers from diverse model outputs.
    - **Mechanism**: A four-level extraction strategy — (1) prioritize content within \boxed{}; (2) parse explicit answer markers ("The answer is..."); (3) extract from code blocks or Markdown formatting; (4) task-specific heuristics as fallback. Validated on 5,000+ responses with a 98.7% success rate.
    - **Design Motivation**: Model output formats vary substantially; reliable answer extraction is a prerequisite for fair evaluation.

### Loss & Training

No model training is involved. Evaluation uses publicly available weights or APIs of existing models. The evaluation covers 53 models, including base, instruction-tuned, reasoning, and quantized variants.

## Key Experimental Results

### Main Results

**Overthinking Score Comparison for Selected Representative Models**

| Model | Params | Accuracy | Overthinking Score | Avg. Output Tokens |
|-------|--------|----------|-------------------|-------------------|
| Phi-4 | 14B | 78.92% | **0.863** | 378.6 |
| Phi-4-reasoning-plus | 14B | 69.54% | 0.234 | 6,780.7 |
| Qwen3-14B | 14B | 86.52% | 0.727 | 3,607.6 |
| Qwen3-0.6B | 0.6B | 49.99% | 0.545 | 3,162.8 |

### Ablation Study

**Inference Budget Constraint Experiments (Qwen3 Reasoning Models)**

| Configuration | Accuracy |
|---------------|----------|
| Full budget | 72% |
| 1,024 token limit | 44% (−28%) |
| Reasoning budget low→medium→high (GPT-5/o series) | Accuracy gain ≈ 0 |

**Quantization Experiments (Qwen2.5 Family)**

| Configuration | Accuracy Change |
|---------------|----------------|
| FP16 → 8-bit | Negligible for large models |
| FP16 → 4-bit | Slight degradation for large models; significant for small models |

### Key Findings

- **Basic math paradox**: Models achieving 95%+ on GSM8K score below 75% on the proposed tasks — strong complex benchmark performance does not reflect basic math ability.
- Reasoning models generate on average 6,780 tokens vs. 378 tokens for standard models (18×), yet achieve lower accuracy (Phi-4-reasoning-plus: 69.54% vs. Phi-4: 78.92%).
- The Overthinking Score exposes efficiency pitfalls masked by accuracy-only metrics: Phi-4 scores 0.863, far exceeding Phi-4-reasoning-plus at 0.234.
- Under token constraints, reasoning models exhibit catastrophic collapse — accuracy drops from 72% to 44%, indicating that reasoning capability is tightly coupled to long-chain inference depth.
- Scaling inference budgets yields diminishing returns — accuracy gains across low-to-high reasoning effort in the GPT-5/o series approach zero.
- Quantization preserves basic reasoning capability, indicating that overthinking originates from training rather than hardware constraints.

## Highlights & Insights

- The Overthinking Score is an elegant and informative metric — the harmonic mean's strict penalization distinguishes "efficiently correct" from "verbosely correct" models.
- The "basic math paradox" is a significant finding, challenging the assumption that high scores on complex benchmarks imply strong mathematical ability.
- Dynamic test generation combined with open-source tooling (PyPI package + leaderboard) makes results reproducible and extensible.

## Limitations & Future Work

- Coverage is limited to 14 deterministic math tasks; more complex mathematical reasoning and non-mathematical domains are not included.
- Token efficiency normalization depends on global maximum/minimum values within the evaluation set, which may be influenced by outliers.
- Specific patterns of overthinking (e.g., proportions of error accumulation vs. self-contradiction) are not analyzed.
- Training reasoning models that are both accurate and efficient remains unexplored.

## Related Work & Insights

- **vs. ThoughtTerminator/Self-Braking**: These works propose strategies to mitigate overthinking; this paper provides a metric to quantify it — measurement is a prerequisite for intervention.
- **vs. GSM8K/MATH benchmarks**: These benchmarks emphasize accuracy; this paper adds the efficiency dimension.
- **vs. Graph of Thoughts/LogicPuzzleRL**: These methods enhance complex reasoning but do not address overthinking on basic operations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The Overthinking Score is a novel and useful metric; the basic math paradox is an important finding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 53 models, quantization analysis, budget constraints, and dynamic generation — large-scale and comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Formal definitions are rigorous; experimental narrative is clear.
- **Value**: ⭐⭐⭐⭐⭐ Provides standardized tooling and deep insights for evaluating the efficiency of reasoning models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](../../ICLR2026/llm_evaluation/benchmarking_overton_pluralism_in_llms.md)
- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](../../NeurIPS2025/llm_evaluation/evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)

</div>

<!-- RELATED:END -->
