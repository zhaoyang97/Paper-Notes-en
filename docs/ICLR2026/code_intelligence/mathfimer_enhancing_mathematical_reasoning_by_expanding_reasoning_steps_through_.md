---
title: >-
  [Paper Note] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task
description: >-
  [ICLR 2026][Mathematical Reasoning] Drawing inspiration from the Fill-in-the-Middle (FIM) paradigm in code completion, this work trains a dedicated step-expansion model, MathFimer-7B, to insert finer-grained intermediate reasoning steps into existing mathematical solution chains, thereby systematically improving the mathematical reasoning capability of downstream models.
tags:
  - ICLR 2026
  - Mathematical Reasoning
  - Fill-in-the-Middle
  - Reasoning Step Expansion
  - Chain-of-Thought
  - Data Augmentation
date: 2026-05-08
content_hash: 1752ee7ce63b464b
---

# MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task

**Conference**: ICLR 2026
**arXiv**: [2502.11684](https://arxiv.org/abs/2502.11684)
**Code**: None
**Area**: Code Intelligence
**Keywords**: Mathematical Reasoning, Fill-in-the-Middle, Reasoning Step Expansion, Chain-of-Thought, Data Augmentation

## TL;DR

Drawing inspiration from the Fill-in-the-Middle (FIM) paradigm in code completion, this work trains a dedicated step-expansion model, MathFimer-7B, to insert finer-grained intermediate reasoning steps into existing mathematical solution chains, thereby systematically improving the mathematical reasoning capability of downstream models.

## Background & Motivation

Chain-of-Thought (CoT) has become the dominant paradigm for mathematical reasoning in LLMs, yet the quality and granularity of reasoning steps in training data directly constrain model performance. Prior work (Jin et al., 2024) has demonstrated that more detailed intermediate steps can significantly improve reasoning accuracy. However, existing step-expansion methods suffer from three major limitations:

**Dependence on stronger models**: Generating higher-quality steps requires larger external models, creating a circular dependency of "bigger is better."

**High computational cost**: Search-based strategies such as MCTS demand substantial computational resources when exploring reasoning paths.

**Insufficient reliability**: These methods typically generate entirely new reasoning chains rather than augmenting existing human-verified steps, potentially introducing new errors.

Core research question: Can a more efficient and reliable method be developed to expand reasoning steps while preserving the validity of existing human-generated solutions?

## Method

### Overall Architecture

The MathFimer framework consists of two stages: (1) training a FIM model; and (2) using the trained model to expand reasoning steps in existing datasets. The approach is inspired by the Fill-in-the-Middle task in code completion — given a prefix and suffix of code, the model completes the missing middle portion.

### Key Designs

1. **NuminaMath-FIM Dataset Construction**:

    - Based on NuminaMath-CoT (853K math QA pairs) with step decomposition.
    - For each sample with solution $Y = \{y_1, y_2, ..., y_n\}$, a step $y_i$ is randomly selected; $y_1...y_{i-1}$ serves as prefix P, $y_{i+1}...y_n$ as suffix S, and $y_i$ as the middle step M to be predicted.
    - Each sample undergoes 3 rounds of random sampling, yielding **2.5M** training instances in total.
    - The PSM (Prefix-Suffix-Middle) sequence format is adopted, using three special tokens: `<|fim_prefix|>`, `<|fim_suffix|>`, and `<|fim_middle|>`.

2. **MathFimer-7B Training**:

    - Base model: Qwen2.5-Math-7B (a math-specialized model).
    - Training approach: SFT, with loss computed only on tokens following `<|fim_middle|>`.
    - Core formulation: $\text{FIM}(Q, P, S) \Rightarrow M$

3. **Step Expansion Strategy**:

    - FIM inference is applied between each pair of consecutive steps in the original solution: $\hat{y}_i = \text{FIM}(Q, y_1...y_{i-1}, y_i...y_n)$
    - **Similarity filtering**: The sequence similarity between the generated step $\hat{y}_i$ and the original next step $y_i$ is computed; steps with similarity above threshold $\eta = 0.8$ are marked as invalid (indicating the original step is already sufficiently detailed).
    - Only newly generated steps with similarity below the threshold are inserted.

### Loss & Training

- Standard cross-entropy loss applied exclusively to tokens in the middle segment.
- SFT conducted using the Megatron-LM framework.
- max_length = 8192, global batch size = 128.
- Learning rate: 1e-5.
- All training samples are packed to accelerate training.
- Training performed on 64 Ascend H910B-64G accelerators.

## Key Experimental Results

### Main Results

Comprehensive experiments were conducted across 4 base models × 5 datasets:

| Base Model | Dataset | GSM8K | MATH | Odyssey | OB-EN | Avg. Gain |
|---------|--------|-------|------|---------|-------|---------|
| Llama-3.1-8B | MathInstruct-CoT | 67.78→75.21 | 18.74→22.90 | 22.11→24.42 | 2.37→3.56 | **+3.77** |
| Llama-3.1-70B | MathInstruct-CoT | 89.31→90.98 | 41.96→44.72 | 36.50→39.33 | 9.19→12.15 | **+2.56** |
| Qwen2.5-Math-7B | MetaMathQA | 93.18→93.10 | 70.22→79.08 | 49.10→52.70 | 34.81→41.04 | **+4.65** |
| Qwen2.5-Math-72B | MetaMathQA | 90.22→92.95 | 57.68→63.40 | 42.93→47.30 | 20.00→24.89 | **+4.43** |

### Ablation Study

| Configuration | Key Metric | Remarks |
|------|---------|------|
| Distilled data vs. original data | G+M original +5.61 vs. distilled +1.13 (GSM8K) | Part of the gain stems from model distillation, but FIM step expansion itself provides additional improvement. |
| Iterative expansion (1→3 rounds) | MI-CoT: 67.78→75.21→80.21→83.32 (GSM8K) | Iterative expansion yields continuous performance gains; 3 rounds achieve +15.54%. |
| Model scale (7B vs. 72B) | 7B: +7.43 vs. 72B: +6.14 (GSM8K, MI-CoT) | The performance gap between 7B and 72B is marginal, indicating that step expansion does not depend on large model capacity. |

### Key Findings

1. **Strong generalizability**: MathFimer proves effective on both general-purpose models (Llama) and math-specialized models (Qwen-Math), with consistent improvements observed across 4 base models and 5 datasets.
2. **Iterative scalability**: Multi-round step expansion yields continuously accumulating gains; 3 rounds of iteration on GSM8K achieve +15.54%.
3. **Small models suffice**: MathFimer-7B and 72B deliver nearly identical expansion performance, demonstrating that the step-expansion task does not require large model capacity.
4. **Greater benefit for math-specialized models**: The Qwen2.5-Math series achieves the largest improvements on MetaMathQA and ScaleQuest (MATH +8.86%).

## Highlights & Insights

- **Elegant analogical transfer**: Adapting the well-established FIM paradigm from code completion to mathematical reasoning step expansion is both novel and effective.
- **Preservation of original structure**: Unlike methods that regenerate entire reasoning chains, MathFimer inserts new steps while retaining the original ones, making the process more reliable.
- **Similarity filtering mechanism**: Automatically determines whether expansion is needed at a given position, preventing redundant insertions.
- **Strong practicality**: No external, stronger models are required; a single 7B model suffices for expansion.
- **Data contribution**: Both the NuminaMath-FIM dataset (2.5M samples) and the MathFimer-7B model are open-sourced.

## Limitations & Future Work

1. **Unknown domain generalizability**: Validation has been limited to mathematical reasoning; applicability to code reasoning, logical inference, commonsense reasoning, and other domains remains unexplored.
2. **Generation reliability**: No automatic mechanism is provided to verify the logical consistency and mathematical correctness of inserted steps; iterative expansion may lead to error accumulation.
3. **Methodological scope**: The approach primarily expands upon existing solution patterns and offers limited exploration of novel solution strategies for highly complex or unconventional problems.
4. **Lack of adaptive expansion**: The current method uniformly applies expansion to all step pairs, without intelligently identifying which steps genuinely require elaboration.

## Related Work & Insights

- CoT Prompting (Wei et al., 2023) and Self-Consistency (Wang et al., 2023) form the foundation of reasoning enhancement.
- Jin et al. (2024) establish the importance of reasoning step length for LLM performance.
- MCTS-based methods (Chen et al., 2024; Guan et al., 2025) pursue the direction of searching for optimal reasoning paths.
- Unlike OpenMathInstruct (Toshniwal et al., 2024), MathFimer does not rely on a stronger model to generate new data.
- **Insight**: The FIM paradigm may generalize to any sequential task requiring "intermediate step completion."

## Rating
- Novelty: ⭐⭐⭐⭐ (Adapting FIM to mathematical reasoning is novel, though it is fundamentally a data augmentation approach.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 base models × 5 datasets × 4 benchmarks; highly comprehensive.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and rigorous experimental presentation.)
- Value: ⭐⭐⭐⭐ (The method is simple and practical, and can be directly applied to augment existing datasets.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](../../NeurIPS2025/code_intelligence/fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [\[ICLR 2026\] The Limits of Long-Context Reasoning in Automated Bug Fixing](the_limits_of_long-context_reasoning_in_automated_bug_fixing.md)
- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)
- [\[ICLR 2026\] Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning](supervised_reinforcement_learning_from_expert_trajectories_to_step-wise_reasonin.md)
- [\[NeurIPS 2025\] CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](../../NeurIPS2025/code_intelligence/core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)

</div>

<!-- RELATED:END -->
