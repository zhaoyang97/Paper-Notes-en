---
title: >-
  [Paper Note] Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models
description: >-
  [ACL2025][LLM Evaluation][numerical reasoning] This paper proposes NumericBench, a comprehensive benchmark to evaluate 6 fundamental numerical abilities (number recognition, arithmetic, retrieval, comparison, aggregation, and logical reasoning) across 6 datasets. It reveals that SOTA models, including GPT-4o and DeepSeek-V3, still perform poorly on simple numerical tasks, and provides an in-depth analysis of 5 root causes.
tags:
  - "ACL2025"
  - "LLM Evaluation"
  - "numerical reasoning"
  - "benchmark"
  - "arithmetic"
  - "number understanding"
  - "tokenizer"
date: 2026-05-08
content_hash: b6a7d8564a65b9e0
---

# Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models

**Conference**: ACL2025  
**arXiv**: [2502.11075](https://arxiv.org/abs/2502.11075)  
**Code**: [TreeAI-Lab/NumericBench](https://github.com/TreeAI-Lab/NumericBench)  
**Area**: LLM Evaluation  
**Keywords**: numerical reasoning, benchmark, arithmetic, number understanding, LLM evaluation, tokenizer

## TL;DR
This paper proposes NumericBench, a comprehensive benchmark to evaluate 6 fundamental numerical abilities (number recognition, arithmetic, retrieval, comparison, aggregation, and logical reasoning) across 6 datasets. It reveals that SOTA models, including GPT-4o and DeepSeek-V3, still perform poorly on simple numerical tasks, and provides an in-depth analysis of 5 root causes.

## Background & Motivation

**Background**: LLMs perform exceptionally well in text generation, semantic understanding, and even Olympiad-level mathematics, yet surprisingly fail at simple numerical tasks (e.g., basic multiplication, number comparison, and number retrieval).

**Limitations of Prior Work**: Semantic benchmarks (e.g., GLUE, SuperGLUE) only measure language capabilities, while mathematical benchmarks (e.g., GSM8K, MathBench) focus on structured algebraic problems. Both ignore the fundamental numerical reasoning demands within unstructured data in real-world scenarios.

**Key Challenge**: LLMs rely on surface statistical patterns and treat numbers as discrete tokens rather than continuous magnitudes, leading to a fundamental flaw in numerical semantic understanding.

**Goal**: To construct a systematic benchmark for comprehensively evaluating LLMs' performance across 6 fundamental numerical abilities, exposing their numerical reasoning bottlenecks.

**Key Insight**: To integrate synthetic data (number lists, arithmetic operations) and real-world scraped data (stocks, weather) to cover the entire spectrum of numerical abilities from simple recognition to multi-step reasoning.

**Core Idea**: LLMs' systematic failures on simple numerical tasks reveal fundamental architecture-level limitations, such as tokenization, training paradigms, and positional encodings.

## Method

### Overall Architecture
6 datasets (number lists, stocks, weather, sequences, arithmetic operations, mixed strings) $\times$ 6 abilities (number recognition, arithmetic, retrieval, comparison, aggregation, logical reasoning) $\rightarrow$ evaluation of 12+ SOTA LLMs $\rightarrow$ 5 root-cause analyses (tokenizer / training corpus / training paradigm / positional encoding / Transformer architecture).

### Key Designs

#### Key Design 1: Systematic Definition of Six Fundamental Numerical Abilities
- **Function**: Disassemble LLM numerical abilities into 6 complementary dimensions for independent testing.
- **Mechanism**: Identify numbers (counting numbers in mixed strings), arithmetic operations (addition, subtraction, multiplication, and division with decimals), contextual retrieval (locating specific values from long sequences/structured data), comparison (determining relative size), aggregation (calculating averages/statistical trends), and logical reasoning (identifying sequence patterns and predicting the next value).
- **Design Motivation**: Existing benchmarks are either too simple or too complex, lacking fine-grained diagnosis of basic numerical operations. These 6 abilities span the complete pipeline from perception to reasoning.

#### Key Design 2: Introduction of Noise and Long Context in Real-World Datasets
- **Function**: Crawl Eastmoney stock data (18 attributes) and Open-Meteo weather data to construct noisy, long-context evaluations.
- **Mechanism**: The average token length reaches 27K+; noise scenarios are simulated by adding $k\in\{2,4,6\}$ irrelevant attributes; both short and long contexts are evaluated to measure long-distance dependency capabilities.
- **Design Motivation**: In real-world scenarios, numerical values are often embedded within massive amounts of irrelevant information, which pure synthetic data fails to reflect.

#### Key Design 3: Dual Questioning and Increasing Number of Digits in Arithmetic Operations
- **Function**: Design two questioning methods, $Q_{oper}$ (symbolic expression, e.g., $a+b$) and $Q_{context}$ (natural language, e.g., "add $a$ and $b$"), with digits increasing from 1 to 6.
- **Mechanism**: $12,000$ pairs of numbers $\times$ 4 operations $\times$ 2 querying formats are used to systematically test the decay of arithmetic accuracy as digits increase.
- **Design Motivation**: Expose the fundamental conflict between the "most-to-least significant digit" generation pattern of LLMs and the actual "least-to-most significant digit" carry logic in arithmetic.

### Loss & Training
This paper presents an evaluation benchmark and does not involve training loss functions. The evaluation metric is consistently **Accuracy**. For the arithmetic dataset, answers are kept to two decimal places, while other datasets use multiple-choice questions (1-of-8, random baseline is $12.5\%$).

## Key Experimental Results

### Main Results (Retrieval/Comparison/Aggregation/Logical Reasoning, Table 2 Selected)

| Model | Number List - Retrieval | Stocks - Retrieval | Weather - Comparison | Sequence - Logical Reasoning |
|------|-------------|----------|----------|-------------|
| Random | 12.5 | 12.5 | 12.5 | 12.5 |
| Llama-3.1-8B | 22.8 | 14.4 | 13.7 | 18.2 |
| Llama-3.3-70B | 44.4 | 19.4 | 35.8 | 18.6 |
| DeepSeek-V3 | 47.2 | **47.5** | 35.8 | 15.8 |
| GPT-4o | 41.7 | 37.5 | **64.2** | 14.6 |
| o3-mini | **96.8** | 68.6 | 83.9 | **66.4** |
| DeepSeek-R1 | 73.6 | **81.3** | **98.8** | 65.4 |
| Human | 100 | 100 | 100 | 52.6 |

### Ablation Study (Impact of CoT on Llama-3.3-70B, Table 4)

| Method | Number List - Retrieval | Number List - Comparison | Stocks - Retrieval | Stocks - Comparison |
|------|-------------|-------------|----------|----------|
| Base | 44.4 | 31.5 | 19.4 | 13.8 |
| Plain-CoT | 65.2 | 39.4 | 24.8 | 27.7 |
| PS-CoT | 65.4 | 40.0 | 24.3 | 16.7 |
| Table-CoT | 65.8 | 38.4 | 27.6 | 29.1 |

### Number Recognition in Mixed Strings (Table 3)

| Model | 50 Chars | 100 Chars | 150 Chars | 200 Chars |
|------|--------|---------|---------|---------|
| GPT-4o | 18.2 | 6.4 | 4.0 | 4.2 |
| DeepSeek-V3 | 13.2 | 4.0 | 3.2 | 2.0 |
| Human | 100 | 100 | 100 | 100 |

### Key Findings
1. **Pervasive Failures**: Even GPT-4o performs far below human levels on most numerical tasks (retrieval: 41.7 vs. 100, aggregation: 11.6 vs. 100).
2. **Reasoning Models Lead Significantly**: o3-mini and DeepSeek-R1 significantly outperform standard LLMs, reaching 73-97% on retrieval tasks, showing that CoT reasoning is crucial for numerical tasks.
3. **Multiplication is the Hardest Operation**: LLMs perform acceptably in addition, subtraction, and division, but maintain extremely low accuracy in multiplication.
4. **Sharp Degradation in Long Contexts**: Almost all models show significantly worse performance in long contexts compared to short contexts.
5. **Significant Interference from Noise**: Models' performance steadily declines with the addition of irrelevant attributes.
6. **Limited Help from CoT**: CoT yields minor improvements on simple tasks but even introduces noise in complex stock aggregation tasks.
7. **SFT is Only Effective on Simple Data**: QLoRA fine-tuning shows substantial improvements on number lists (retrieval: 24.4 $\rightarrow$ 62.8) but fails to improve on complex data such as stocks or weather.

## Highlights & Insights
1. **Outstanding Diagnostic Depth**: Instead of simply reporting that "LLMs perform poorly on numbers," the paper systematically attributes this to 5 dimensions (tokenizer, training paradigm, positional encoding, and architecture), providing concrete mechanical explanations for each.
2. **Surprising Findings on Human Baseline**: Humans scored only 52.6% on the logical reasoning task, indicating that some sequence pattern recognitions are inherently difficult, and thus LLMs should not be judged too harshly.
3. **Huge Gap Between Reasoning and Standard Models**: The performance of o3-mini/DeepSeek-R1 proves that explicit reasoning chains represent an effective path to compensate for numerical ability limits.
4. **Practically-Oriented Dataset Design**: Stock and weather data are directly fetched from real APIs, keeping them close to actual production scenarios.

## Limitations & Future Work
1. Covers only 6 types of numerical tasks, lacking more real-world domains like transportation and healthcare.
2. Did not evaluate the Claude series and o1 models (due to an API cost of approximately $15,000).
3. The maximum digit length in the arithmetic dataset is 6, leaving larger numerical ranges unexplored.
4. The human baseline of only 52.6% on sequence pattern tasks suggests that the dataset design might be overly difficult or the labeling ambiguous.
5. Lacks analysis of models' internal representations (such as probing experiments); thus, the root-cause analysis remains largely speculative.

## Related Work & Insights

### vs GSM8K / MathBench
GSM8K tests multi-step math word problems, while MathBench covers structured problems like algebra and geometry. NumericBench focuses on **more fundamental numerical operations** (retrieval, comparison, recognition), revealing that LLMs have systematic flaws even in primary-school-level operations—which are prerequisites for high-order mathematical reasoning.

### vs MATH-Perturb / GSM-Symbolic
These studies prove that LLMs rely on pattern matching rather than genuine understanding by perturbing numbers in math questions. NumericBench further extends this discovery to **non-mathematical scenarios** (stock analysis, weather forecasts), proving that numerical understanding flaws are general rather than unique to math.

### Insights
- Tokenizer improvements (such as number-aware tokenization) might be the most cost-effective path to enhance numerical capabilities.
- Reverse generation (from least-to-most significant digit) is worth exploring to resolve arithmetic conflicts.

## Rating
- Novelty: ⭐⭐⭐⭐ — The first comprehensive benchmark to systematically evaluate LLMs' basic numerical abilities; the design combining 6 abilities $\times$ 6 datasets is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations across 12+ models, multiple scenarios (noise/long context/CoT/SFT); lacking models like Claude is a minor drawback.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, precise problem definitions, and well-substantiated root-cause analysis.
- Value: ⭐⭐⭐⭐⭐ — Unveils a severely underestimated fundamental bottleneck in LLMs, providing crucial guidance for improving tokenizers, architectures, and training paradigms.

| Logical Reasoning | ~40% | ~35% | ~25% |

### Root-Cause Analysis

| Cause | Impact Level | Description |
|------|---------|------|
| Tokenization | High | Numbers are split into multiple tokens |
| Training Data | High | Insufficient numerical training data |
| Positional Encoding | Medium | Positional bias in long sequences |
| Architecture | Medium | Attention mechanism is unsuited for precise calculation |

### Key Findings
- **All models fall far short of perfection across all tasks**
- **Aggregation and logical reasoning are the greatest weaknesses**
- **Real-world data is harder than synthetic data** (due to noise and contextual interference)
- **GPT-4o still suffers from an ~40% error rate on simple 3-digit multiplication**

## Highlights & Insights
- **The systematic classification of 6 abilities** provides a comprehensive framework for numerical understanding research
- **The root-cause analysis** links surface-level symptoms (task failures) to underlying causes (tokenization/architecture)

## Limitations & Future Work
- Does not test the mitigation effect of using calculator tools
- Directions for improvement: Number-aware tokenization, neuro-symbolic hybrid approaches

## Related Work & Insights
- **vs GSM8K**: GSM8K tests mathematical reasoning, while NumericBench tests more fundamental numerical operations
- **vs CUTE/EXECUTE**: They evaluate character understanding, whereas NumericBench evaluates numerical understanding—representing complementary directions

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic evaluation of six abilities + root-cause analysis
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models $\times$ 6 datasets $\times$ real + synthetic
- Writing Quality: ⭐⭐⭐⭐ Clear taxonomy
- Value: ⭐⭐⭐⭐ Provides direct guidance for developing number-aware LLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](../../NeurIPS2025/llm_evaluation/creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)

</div>

<!-- RELATED:END -->
