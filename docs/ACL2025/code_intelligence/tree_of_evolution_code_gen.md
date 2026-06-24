---
title: >-
  [Paper Note] Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models
description: >-
  [ACL 2025 (Long Paper)][Code Intelligence][Code Instruction Synthesis] Proposes Tree-of-Evolution (ToE), a tree-structured code instruction synthesis framework. By leveraging multi-path evolution and quality-driven optimization, ToE overcomes the limitations of unidirectional synthesis and random generation in Code Evol-Instruct and OSS-Instruct. Fine-tuning a base model with only 75K of its synthesized data achieves or exceeds the performance of Qwen2.5-Coder-Instruct (fine-…
tags:
  - "ACL 2025 (Long Paper)"
  - "Code Intelligence"
  - "Code Instruction Synthesis"
  - "Tree-Structured Evolution"
  - "Optimization-Driven Generation"
  - "Code LLM"
  - "Data Quality"
date: 2026-05-08
content_hash: 3348eb049c05f414
---

# Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models

**Conference**: ACL 2025 (Long Paper)  
**Code**: None  
**Area**: Code Generation / Data Synthesis / Instruction Tuning  
**Keywords**: Code Instruction Synthesis, Tree-Structured Evolution, Optimization-Driven Generation, Code LLM, Data Quality  

## TL;DR
Proposes Tree-of-Evolution (ToE), a tree-structured code instruction synthesis framework. By leveraging multi-path evolution and quality-driven optimization, ToE overcomes the limitations of unidirectional synthesis and random generation in Code Evol-Instruct and OSS-Instruct. Fine-tuning a base model with only 75K of its synthesized data achieves or exceeds the performance of Qwen2.5-Coder-Instruct (fine-tuned on millions of samples).

## Background & Motivation

**Data Bottleneck**: Manual annotation of high-quality code instruction data is extremely costly (averaging 15–30 minutes of expert review per sample), making LLM-driven automated synthesis the mainstream solution.

**Limitations of Prior Work**: (1) **Unidirectional Synthesis**: Methods like Code Evol-Instruct and OSS-Instruct generate only a single evolutionary chain (A->B->C) for each seed instruction, limiting the diversity of data and the exploration of the evolutionary space; (2) **Randomly Driven**: The evolutionary direction is selected randomly (e.g., "increase complexity", "change algorithm") without considering the generation quality of the previous step, resulting in a large number of low-quality dead ends along the evolutionary path.

**Key Challenge**: Under a fixed synthesis budget (e.g., 75K items), how can both the **quality** and **diversity** of code instructions be maximized simultaneously?

## Method

### Overall Architecture
Code instruction synthesis is modeled as a **tree search optimization problem**: each seed instruction serves as the root node of a tree, branching along multiple evolutionary paths (tree width), where each path can iterate over multiple steps (tree depth). Through pruning and expansion strategies driven by quality evaluation, the evolutionary space is optimally explored under a limited budget.

### Key Designs

1. **Tree-Structured Multi-Path Evolution**

    - Traditional methods model instruction evolution as a linear chain (A->B->C), whereas ToE models it as a multi-way tree (A->{B1,B2,B3}->{C1,...,C9}->...)
    - Each node can explore multiple evolutionary directions: increasing algorithmic complexity, changing programming paradigms, adding external boundary constraints, introducing new data structures, etc.
    - The branching factor and maximum depth of the tree are adjustable to balance the breadth and depth of exploration.
    - Different branches from the same root node naturally guarantee diversity under theme relevance (e.g., the same sorting problem can branch into merge sort optimization, parallel sorting, external sorting, etc.).

2. **Optimization-Driven Evolutionary Strategy**

    - Immediately after each evolutionary step, quality evaluation is performed: code executability checks, test case pass rates, code complexity analysis, etc.
    - Next-round evolutionary strategies are dynamically adjusted based on quality feedback from the previous round: high-quality nodes receive more branching budget, while low-quality nodes are pruned.
    - Similar to the UCB guidance concept in Monte Carlo Tree Search (MCTS), but replacing reward signals with code quality signals.
    - Effectively avoids resource waste caused by random evolution—in traditional methods, 30–50% of synthesized data is discarded due to failing to meet quality standards.

3. **Multi-Dimensional Quality Evaluation**

    - Progressive filtering: Syntactic correctness -> Executability -> Functional correctness (test cases) -> Code quality (complexity, readability).
    - Each level of evaluation filters out sub-standard evolutionary branches, focusing resources on high-potential directions.
    - The optimal nodes in the evolutionary paths (not necessarily the leaf nodes) are preserved to construct the final training dataset.

### Loss & Training
- Standard Supervised Fine-Tuning (SFT) is performed on the base model with the 75K synthesized data, without requiring additional RLHF or DPO training.
- The synthesized data format consists of standard (instruction, reference code) pairs.

## Key Experimental Results

### Main Results: Comparison with SOTA Code LLMs

| Model | HumanEval | MBPP | EvalPlus | LiveCodeBench | BigCodeBench |
|------|-----------|------|----------|---------------|-------------|
| Qwen2.5-Coder-Instruct (Millions) | Baseline | Baseline | Baseline | Baseline | Baseline |
| **ToE (75K)** | **>=Baseline** | **>=Baseline** | **~=Baseline** | **~=/>=Baseline** | **~=Baseline** |
| Code Evol-Instruct (75K) | <Baseline | <Baseline | <Baseline | <Baseline | <Baseline |
| OSS-Instruct (75K) | <Baseline | <Baseline | <Baseline | <Baseline | <Baseline |

Core Highlight: Achieving or exceeding the SOTA performance across 5 mainstream code benchmarks with only about **1/15 to 1/50 of the data scale** (75K vs. Millions).

### Ablation Study

| Ablation Dimension | Conclusion |
|----------|------|
| Tree-structured vs. Chain Evolution | Tree structure is significantly superior in data diversity (unique solutions +23%) and downstream performance. |
| Optimization-driven vs. Random Evolution | Optimization-driven increases the synthesized data test pass rate by 15–20%, directly translating to downstream improvement. |
| Branching Factor = 2 vs. 4 vs. 8 | A branching factor of 4 is the optimal balance point; excessively large branching leads to exponential growth in computing cost with diminishing returns. |
| Tree Depth = 2 vs. 3 vs. 5 | A depth of 3–4 is optimal; deeper levels of evolutionary data are too difficult and show degraded quality. |
| Quality Filtering Threshold | Strict filtering (top 30%) is slightly better than loose filtering (top 70%), indicating that data quality is more critical than quantity. |

### Key Findings
- The diversity advantage of the tree structure is more prominent on hard benchmarks (such as LiveCodeBench)—hard problems require more diverse code patterns.
- Proportion of "dead ends" during evolution: about 35% for random evolution, compared to only about 12% for optimization-driven evolution.
- The 75K data generated by ToE is close to a manually curated 500K dataset in terms of code pattern coverage.

## Highlights
- **Highly Data-Efficient**: Fine-tuning with 75K data achieves millions-scale SOTA performance, making it highly practical for resource-constrained researchers.
- **Natural and Elegant Tree-Structured Idea**: Instruction evolution naturally branches across multiple paths, and the tree structure captures this characteristic much better than linear chains.
- **Avoidance of Resource Waste via Optimization-Driven Process**: Real-time quality feedback at each step avoids the subsequent overhead of filtering massive amounts of low-quality data.
- **Generic Framework**: The tree structure + optimization-driven approach can be generalized to data synthesis in other domains (e.g., mathematical reasoning, dialogue systems).

## Limitations
- The computational cost of tree search exploration scales exponentially with the branching factor, requiring rational pruning strategies to control the overall budget.
- Quality evaluation relies heavily on code executability tests, requiring new sources of quality signals for non-code domains.
- The performance scaling behavior when synthesizing larger scales (500K+) of data has not been explored.
- The selection strategy of seed instructions is not discussed in detail; different seeds may result in varying qualities of evolutionary trees.
- Validated only on Qwen series base models; cross-model generalizability remains to be confirmed.

## Related Work & Insights
- **vs Code Evol-Instruct**: Chain-based unidirectional evolution + random direction choice, whereas ToE switches to tree-structured + quality-guided evolution, yielding significantly higher data efficiency.
- **vs OSS-Instruct**: Unidirectional evolution after extracting seeds from open-source code, whereas ToE's tree structure can generate richer variations from a small number of seeds.
- **vs Self-Instruct**: A general instruction synthesis framework, without specialization for quality evaluation in the coding domain.
- **vs Qwen2.5-Coder-Instruct**: Requires millions of data samples + multi-stage training, whereas ToE achieves comparable performance with 1/15 of the data scale.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of tree-structured evolution and optimization-guided generation is novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on 5 mainstream benchmarks, with ablation studies covering key design choices.
- Writing Quality: ⭐⭐⭐⭐ The progression logic from unidirectional chains to multi-path trees is natural and clear.
- Value: ⭐⭐⭐⭐ The generic concept of data-efficient synthesis can be transferred to other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)

</div>

<!-- RELATED:END -->
