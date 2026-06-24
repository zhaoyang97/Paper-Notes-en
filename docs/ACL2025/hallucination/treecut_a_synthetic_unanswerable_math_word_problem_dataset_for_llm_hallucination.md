---
title: >-
  [Paper Note] TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation
description: >-
  [ACL2025][Hallucination Detection][Unanswerable math word problems] This paper proposes TreeCut, a tree-structure-based synthetic dataset generation method. By systematically removing essential condition edges along tree paths, it generates an infinite number of unanswerable mathematical word problems to evaluate the hallucinating behavior of LLMs when facing unsolvable tasks.
tags:
  - "ACL2025"
  - "Hallucination Detection"
  - "Unanswerable math word problems"
  - "LLM hallucination"
  - "synthetic dataset"
  - "tree structure"
  - "math reasoning evaluation"
date: 2026-05-08
content_hash: 1964208013462d5b
---

<!-- Generated automatically by src/gen_stubs.py -->
# TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation

**Conference**: ACL2025  
**arXiv**: [2502.13442](https://arxiv.org/abs/2502.13442)  
**Code**: [j-bagel/treecut-math](https://github.com/j-bagel/treecut-math)  
**Area**: Hallucination Detection  
**Keywords**: Unanswerable math word problems, LLM hallucination, synthetic dataset, tree structure, math reasoning evaluation

## TL;DR
This paper proposes TreeCut, a tree-structure-based synthetic dataset generation method. By systematically removing essential condition edges along tree paths, it generates an infinite number of unanswerable mathematical word problems to evaluate the hallucinating behavior of LLMs when facing unsolvable tasks.

## Background & Motivation

1. **LLM math reasoning capabilities are overestimated**: Models like GPT-4o have achieved human-level performance on GSM8K (>90% accuracy). However, whether this truly reflects reasoning ability remains controversial, as it might just be pattern matching.
2. **LLMs hallucinate on unanswerable questions**: Prior studies show that when faced with unanswerable math problems, LLMs tend to confidently provide incorrect answers instead of identifying that the question is unsolvable.
3. **Existing unanswerable datasets depend on preset problem repositories**: Approaches like Li et al. (2024) and Zhou et al. (2024a) make modifications based on existing datasets such as GSM8K, which faces the risk of training data contamination and yields limited scale.
4. **Lack of structurally controllable generation mechanisms**: Existing methods cannot precisely control the complexity (depth, number of variables, cut position, etc.) of the problems, restricting fine-grained analysis of hallucination causes.
5. **High manual annotation costs**: Sun et al. (2024) rely on manual annotation to create unanswerable questions, producing only 2,600 pairs, which is small-scale and difficult to scale.
6. **Need for answerable/unanswerable pairs**: Ideal evaluations require comparative analysis of answerable and unanswerable versions of the same question. Existing methods struggle to guarantee pair quality.

## Method

### Tree Structure Representation
TreeCut models each mathematical word problem as a tree:

- **Nodes**: Each non-root node represents a variable (e.g., the price of food). The root node is a reserved special node.
- **Edges**: Edges from the root node to child nodes assign initial values to variables. Edges between non-root nodes represent linear relationships between two variables.
- **Solution Path**: For any variable, the unique path from the root to that node allows stepping through basic arithmetic operations to solve it, without needing to solve systems of linear equations.

### Controllable Parameters
The generation process precisely controls the problem structure through four key parameters:

| Parameter | Meaning |
|------|------|
| `numVars` | Total number of variables (number of nodes) |
| `ansDepth` | Distance from the root node to the queried variable (path depth) |
| `compositeName` | Whether to use composite names (e.g., "Burger at Bistro Nice" vs "Burger") |
| `cutDepth` | In the unanswerable version, the distance from the cut edge to the queried variable |

### Unanswerable Question Generation (TreeCut Operation)
Starting from an answerable math problem, one edge along the path from the root to the queried variable is removed (cut), making the queried variable un-determinable. For example, given the price of $x_1$, and a linear relationship between $x_2$ and $x_1$, and another between $x_3$ and $x_2$. If the edge $x_1-x_2$ is removed, neither $x_2$ nor $x_3$ can be solved—two unknowns with only one equation.

### Numerical Constraints
- Each food unit price is restricted to an integer from 5–15.
- Coefficients of linear equations are non-zero integers between -3 and 3.
- Variables are randomly mapped to food names, and formulas are translated into natural language through templates.

### Dataset Scale
Each configuration of parameters generates 500 questions, theoretically allowing the infinite generation of mutually distinct questions.

## Key Experimental Results

### Experiment 1: Hallucination Rates of Various LLMs on Unanswerable Questions (Zero-Shot)

| ansDepth | Llama-8B | Llama-70B | Qwen-7B | Qwen-72B | GPT-4o | o3-mini |
|----------|----------|-----------|---------|----------|--------|---------|
| 2 | 80.2% | 24.6% | 84.6% | 59.8% | 12.0% | 44.0% |
| 4 | 86.2% | 40.2% | 90.4% | 82.8% | 18.0% | 25.2% |
| 6 | 86.0% | 63.4% | 95.6% | 88.4% | 47.4% | 19.2% |
| 8 | 84.2% | 65.0% | 93.4% | 85.2% | 64.0% | 25.6% |

**Key Findings**: All models fail to identify unanswerable questions satisfactorily. Llama-8B and Qwen-7B/72B almost completely fail; GPT-4o's hallucination rate reaches up to 64% on deep questions (`ansDepth=8`). o3-mini performs best on deep questions but conversely exhibits a hallucination rate up to 44% at the simplest `ansDepth=2`, presenting an anomalous deviation pattern.

### Experiment 2: Accuracy Comparison on Answerable Questions (Zero-Shot)

| ansDepth | Llama-8B | GPT-4o | o3-mini |
|----------|----------|--------|---------|
| 2 | 68% (14%) | 99% (1%) | 100% (0%) |
| 8 | 5% (12%) | 84% (2%) | 100% (0%) |

Numbers in parentheses indicate the ratio of answerable questions incorrectly judged as unanswerable. At `ansDepth=8`, GPT-4o correctly solves 84% of answerable questions but only identifies 36% of unanswerable ones, showing that **there is a huge gap between mathematical computation capability and unsolvability judgment capability**.

### Experiment 3: Structural Factor Analysis of GPT-4o Hallucination Rate

| Factor | Impact |
|------|------|
| Deeper tree structure (`ansDepth`↑) | Hallucination rate monotonically increases |
| More complex tree structure (`numVars=ansDepth+2`) | Hallucination rate is higher than a simple path across all depths |
| Composite item names | Consistently increases the hallucination rate |
| `cutDepth` in the middle of the path (3–6) | Highest hallucination rate (>60%, >70% at `cutDepth=5`) |
| `cutDepth` at both ends (1,2,7) | Hallucination rate <50%, easier for models to identify |

## Highlights & Insights

- **Infinitely Scalable**: Based on parameterized generation, it can theoretically produce infinitely many non-repetitive questions, completely avoiding data contamination.
- **Precisely Controllable Analysis**: Four independent parameters allow fine-grained study of the influence of various factors on hallucinations, such as first revealing that "a cut position in the middle of the path is most likely to trigger hallucinations".
- **Answerable/Unanswerable Pairing**: Generating both versions from the same tree ensures a fair comparison.
- **Broad Model Coverage**: Evaluates the latest reasoning models including o3-mini, revealing their specific hallucination bias patterns.
- **Concise and Elegant Formalization**: Unifies question generation and unsolvability verification using a tree + edge-cutting mathematical framework.

## Limitations & Future Work

1. **Only covers linear arithmetic word problems**: Does not involve broader mathematical fields like geometry, probability, algebraic equations, etc.; generated question types are homogeneous.
2. **Only evaluates zero-shot and few-shot CoT**: Does not explore more advanced reasoning frameworks such as self-consistency or tool-augmented reasoning.
3. **Relatively artificial question structure**: Natural language generated based on templates may have a gap in complexity compared to real-world math questions.
4. **Lack of exploration on mitigation strategies**: Identifies the problem but does not propose improvement plans to reduce LLM hallucinations.
5. **No evaluation of fine-tuned performance**: It remains unclear whether fine-tuning on TreeCut data can improve the model's capability to recognize unsolvable questions.

## Related Work & Insights

### vs GSM-Plus (Li et al., 2024)
GSM-Plus generates unanswerable questions by using GPT-4 prompt modifications based on GSM8K, which requires manual review, and its scale is limited by the size of the original dataset. TreeCut is completely synthetically generated, has no upper limit on scale, and provides precise structural control parameters. However, GSM-Plus's questions are closer to the style of real math word problems.

### vs MathGAP (Opedal et al., 2024)
MathGAP also uses tree structures to generate synthetic math word problems, but the nodes in its tree represent logical propositions, the root represents the answer, and it **does not involve unanswerable questions**. In TreeCut, nodes represent variables, leaves represent queried variables, and it focuses on creating unsolvability by cutting edges. Though both use trees, they are fundamentally different.

### vs Sun et al. (2024)
Sun et al. rely on manual annotation to modify existing questions into unanswerable ones, producing only 2,600 pairs, which is costly and hard to scale. TreeCut automates generation, can produce any number of paired questions, and additionally provides structural parameters for analyzing the causes of hallucinations.

## Rating
- Novelty: ⭐⭐⭐⭐ (The formalized framework of trees + edge-cutting is novel and intuitive, systematizing the generation of unanswerable questions)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers 6 models × multiple parameter configurations, zero-shot/few-shot comparisons, and comprehensive structural factor analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear formalization, intuitive illustrations, though the paper is brief, and some analyses could be deeper)
- Value: ⭐⭐⭐⭐ (Provides an extensible and systematic tool for LLM hallucination evaluation, holding high significance for understanding reasoning flaws)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset](../../CVPR2025/hallucination/phd_a_chatgpt-prompted_visual_hallucination_evaluation_dataset.md)
- [\[ACL 2025\] HalluLens: LLM Hallucination Benchmark](hallulens_llm_hallucination_benchmark.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[ACL 2026\] Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](../../ACL2026/hallucination/rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)
- [\[ACL 2025\] FIHA: Autonomous Fine-grained Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs](fiha_autonomous_hallucination_evaluation_in_vision-language_models_with_davidson.md)

</div>

<!-- RELATED:END -->
