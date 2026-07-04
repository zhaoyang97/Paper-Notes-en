---
title: >-
  [Paper Note] EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning
description: >-
  [ICML2025][Code Intelligence][Code Generation] EffiCoder constructs an "accurate and efficient" instruction tuning dataset named EffiInstruct. It enables code LLMs to significantly reduce execution time and total memory overhead while improving the pass@1 rate, demonstrating that "efficiency can be learned through data design."
tags:
  - "ICML2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Efficiency-Aware Fine-tuning"
  - "Instruction Tuning"
  - "Execution Time"
  - "Memory Overhead"
date: 2026-05-08
content_hash: f3091ec77d3c9bff
---

# EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning

**Conference**: ICML2025  
**arXiv**: [2410.10209](https://arxiv.org/abs/2410.10209)  
**Code**: [GitHub - EffiCoder](https://github.com/huangd1999/EffiCoder)  
**Area**: Code Intelligence  
**Keywords**: Code Generation, Efficiency-Aware Fine-tuning, Instruction Tuning, Execution Time, Memory Overhead

## TL;DR
EffiCoder constructs an "accurate and efficient" instruction tuning dataset named EffiInstruct. It enables code LLMs to significantly reduce execution time and total memory overhead while improving the pass@1 rate, demonstrating that "efficiency can be learned through data design."

## Background & Motivation

### Why Correctness Alone Is Insufficient
Current code LLMs are mostly optimized for accuracy (e.g., pass@k, unit test pass rate). However, in real-world software engineering, code must also satisfy efficiency constraints, such as running faster (lower execution time) and consuming less memory (lower memory usage), allowing deployment on resource-constrained devices. The paper points out that even when mainstream models generate functionally correct code, their efficiency is often inferior to human canonical solutions. This leads to higher computational and energy costs, limiting model applicability in edge scenarios.

### Authors' Key Observation
The authors conducted a preliminary experiment by fine-tuning the same model using training datasets with different "efficiency levels" and evaluating the efficiency of the generated code. The results showed a high correlation between the efficiency of the training data and that of the model's output:
- ET correlation coefficient: 0.972
- MU correlation coefficient: 0.950
- TMU correlation coefficient: 0.986

This confirms that "the model learns whichever style is present in the training set." Therefore, the problem is formulated as: how to construct a large-scale, executably validated, and cross-lingual "efficient code instruction set."

## Method

### Overall Architecture
The core of EffiCoder is not altering the model architecture, but rather modifying the training data design:
1. Aggregate multi-source open-source code instruction data.
2. Prompt multiple LLMs to generate candidate solutions for each task.
3. Execute candidate solutions locally to measure execution time and memory usage.
4. Select the "correct and more efficient" solutions as supervised labels.
5. Construct EffiInstruct to perform SFT on target code LLMs.

### Step 1: Candidate Task Pool Construction
The paper aggregates tasks from 9 public datasets (such as SelfCodeAlign, CodeFeed, APPS, etc.), cleaning and filtering the original ~790k candidates to obtain approximately 65.7k tasks. These tasks cover 5 programming languages: Python, C++, Java, Rust, and Go, ensuring cross-lingual generalization.

### Step 2: Multi-Model Candidate Solution Generation
For each task, instead of relying on a single model, multiple LLMs generate multiple sets of candidate code. This aims to:
- Increase diversity in the solution space.
- Improve the probability of obtaining highly efficient implementations.
- Avoid the style bias of any single model.

### Step 3: Local Executable Efficiency Evaluation
The authors execute candidate code locally and record three classes of metrics:
- ET: Execution Time
- MU: Max Memory Usage
- TMU: Total Memory Usage (with normalized metrics provided in the paper)

The candidates with superior execution time and memory performance are selected as final supervision targets.

### Step 4: Construction of EffiInstruct and Fine-Tuning
The final EffiInstruct dataset contains:
- Task descriptions.
- High-quality target code filtered based on efficiency.
- Meta-data for analytical purposes.

For training, a standard supervised fine-tuning (SFT) pipeline (implemented via LLaMA-Factory) is employed:
- max length: 2048
- batch size: 128
- lr: 5e-6
- scheduler: cosine
- warmup ratio: 0.03
- epoch: 4

### Mechanism
The core idea of EffiCoder can be summarized as:
- Rather than instructing the model on "how to optimize algorithms,"
- The "efficiency preference" is distilled into the model via output distribution shaping (data distribution).

This approach is highly compatible across different base models and languages, keeping engineering migration costs low.

## Key Experimental Results

### Dataset Scale and Language Distribution (EffiInstruct)

| Language | Task Count |
|---|---|
| Python | 33,489 |
| Java | 14,726 |
| C++ | 11,547 |
| Rust | 4,270 |
| Go | 1,678 |
| Total | 65,710 |

This indicates that the dataset is not a small, single-language sample, but rather a cross-lingual resource capable of supporting SFT for medium-to-large models.

### Main Results (EffiBench)

| Model | Pass@1 (Original) | Pass@1 (+EffiInstruct) | ET Improvement |
|------|---------------|------------------------|---------|
| Qwen2.5-Coder-7B-Instruct | 44.8 | 57.7 | 0.31s -> 0.16s (-48.4%) |
| Qwen2.5-Coder-14B | 57.5 | 63.6 | 0.36s -> 0.15s (-58.3%) |
| Qwen2.5-Coder-7B | 50.1 | 57.3 | 0.26s -> 0.17s (-34.6%) |
| DeepSeek-Coder-6.7B-Instruct | 44.4 | 51.7 | 0.34s -> 0.22s (-35.3%) |
| CodeLlama-7B | 15.0 | 17.6 | 0.24s -> 0.20s (-16.7%) |

Two key observations can be made:
- Accuracy increases (overall improvement in pass@1).
- Efficiency metrics decrease (significant reduction in ET/TMU).

This directly validates the paper's thesis that correctness and efficiency can be optimized simultaneously.

### Additional Comparisons (Representative Conclusions from the Paper)
- On HumanEvalPlus, Qwen2.5-Coder-7B-Instruct's pass@1 improved from 76.2 to 78.0.
- Compared with PIE, CodeLlama-7B's pass@1 on HumanEvalPlus improved to 31.1, and the reduction in execution time was more pronounced (the paper indicates that EffiCoder achieves a sharper reduction in latency compared to PIE).

### Key Findings
1. Strong correlation between training data efficiency and generation efficiency, validating the core motivation.
2. Benefits are observable across small to large models, demonstrating scalability and transferability.
3. While some models show marginal improvements in MU, they still consistently benefit in terms of ET, TMU, and pass@1.
4. Data engineering (constructing highly efficient supervision targets) can provide performance gains close to algorithmic improvements.

## Highlights & Insights

1. Highly practical topic.  
   Transitioning code generation from "runnable" to "efficient" aligns with real-world industry demands.

2. Simple yet effective methodology.  
   Without altering the network architecture or introducing complex RL, significant performance improvements are achieved through dataset construction and standard SFT.

3. Strong scalability.  
   The framework seamlessly supports more programming languages, pre-trained base models, and candidate generators.

4. Closed-loop evaluation design.  
   Candidate solutions are evaluated through actual local execution rather than relying solely on heuristic scoring.

5. Strong evidence for the "data is algorithms" paradigm.  
   For coding tasks, high-quality target distributions act as strong optimization signals.

## Limitations & Future Work

1. High evaluation costs.  
   Large-scale local execution, cross-language environment overheads, and sandbox security constraints increase data construction costs.

2. Unidimensional definition of efficiency.  
   The main focus is on ET, MU, and TMU, lacking a systematic approach to encompass other engineering metrics such as readability, robustness, and concurrency.

3. Candidate space relies heavily on generator capabilities.  
   If the candidate models cannot produce highly efficient implementations, the performance upper bound will be limited.

4. Lack of granular analysis across task difficulties.  
   The benefits might vary across different algorithm categories (such as DP, graph theory, or string manipulation), which warrants more detailed reporting.

5. Safety and potential side effects require further investigation.  
   "Efficient code" might occasionally involve aggressive or unsafe optimizations, requiring extra verification for edge cases and error handling.

## Related Work & Insights

- Compared to conventional instruction tuning pipelines (such as Self-Instruct, Evol-Instruct, and OSS-Instruct), EffiCoder explicitly incorporates "efficiency preferences" rather than focusing solely on task coverage and functional correctness.

- Unlike inference-time optimization strategies, EffiCoder bakes efficiency attributes directly into parameters, eliminating extra runtime overhead.

- Insights for future work:
  1. "Efficiency" could be extended to address metrics like energy consumption/carbon footprint.
  2. Multi-objective data selection (joint scoring of accuracy, latency, memory, and readability) can be investigated.
  3. Test-case generation and formal verification could be incorporated to reduce "efficient but fragile" generations.

## Rating
- Novelty: ⭐⭐⭐⭐☆ (4.0/5)
- Experimental Thoroughness: ⭐⭐⭐⭐☆ (4.5/5)
- Writing Quality: ⭐⭐⭐⭐☆ (4.0/5)
- Value: ⭐⭐⭐⭐⭐ (5.0/5)

Overall Evaluation: This work systematizes efficiency-aware dataset building and exhibits stable performance gains on several mainstream code LLMs, making it a highly practical academic contribution with great engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](../../ACL2025/code_intelligence/gift_gibbs_fine_tuning_code_gen.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](../../ACL2025/code_intelligence/personality_guided_code_gen.md)
- [\[ICML 2025\] SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity](sparselora_accelerating_llm_fine-tuning_with_contextual_sparsity.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](../../ACL2025/code_intelligence/codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)

</div>

<!-- RELATED:END -->
