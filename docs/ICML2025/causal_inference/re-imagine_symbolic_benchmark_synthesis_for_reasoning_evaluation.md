---
title: >-
  [Paper Note] RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation
description: >-
  [ICML 2025 (Poster)][Causal Inference][Reasoning Evaluation] Inspired by Pearl's Ladder of Causation, this work proposes the RE-IMAGINE framework. By translating questions into intermediate symbolic representations (code) and executing multi-level mutations on the computation graph, the framework generates benchmark variants that cannot be solved by memorization, systematically evaluating the genuine reasoning capabilities of LLMs.
tags:
  - "ICML 2025 (Poster)"
  - "Causal Inference"
  - "Reasoning Evaluation"
  - "Ladder of Causation"
  - "Symbolic Representation"
  - "Benchmark Mutation"
  - "Memorization Detection"
  - "Computation Graph"
date: 2026-05-08
content_hash: 5edf80e5f499c6ba
---

# RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation

**Conference**: ICML 2025 (Poster)

**arXiv**: [2506.15455](https://arxiv.org/abs/2506.15455)

**Authors**: Xinnuo Xu, Rachel Lawrence, Kshitij Dubey, Atharva Pandey, Risa Ueno, Fabian Falck, Aditya V. Nori, Rahul Sharma, Amit Sharma, Javier Gonzalez

**Area**: Causal Reasoning / LLM Reasoning Evaluation

**Keywords**: Reasoning Evaluation, Ladder of Causation, Symbolic Representation, Benchmark Mutation, Memorization Detection, Computation Graph

## TL;DR

Inspired by Pearl's Ladder of Causation, this work proposes the RE-IMAGINE framework. By translating questions into intermediate symbolic representations (code) and executing multi-level mutations on the computation graph, the framework generates benchmark variants that cannot be solved by memorization, systematically evaluating the genuine reasoning capabilities of LLMs.

## Background & Motivation

In recent years, the accuracy of LLMs on reasoning benchmarks (such as GSM8K) has continued to rise, but a core question remains unresolved:

> **Does the high performance of models stem from genuine reasoning capabilities, or from statistical memorization of the training set?**

Existing evaluation methods suffer from the following issues:

**Static Benchmark Leakage**: Once a fixed benchmark is published, it is likely to be incorporated into subsequent pre-training data.

**Fragmented Mutation Methods**: Prior work (such as GSM-Symbolic) only explored limited types of mutations and lacked a unified hierarchical framework.

**Entanglement of Difficulty and Memorization**: Performance degradation may stem from increased question difficulty rather than a failure of memorization.

The core idea of RE-IMAGINE draws inspiration from Judea Pearl's **Ladder of Causation**, dividing reasoning capability into three progressive levels and constructing an extensible automated mutation pipeline.

## Method

### Overall Architecture

The RE-IMAGINE pipeline consists of four steps:

```text
Natural Language Question → Symbolic Representation (Code) → AST Mutation → Natural Language Variant Question
```

1. **NL-to-Symbolic**: Translate natural language questions into executable Python code (computation graphs).
2. **Symbolic Mutation**: Execute deterministic mutation operations on the Abstract Syntax Tree (AST).
3. **Mutation-to-NL**: Translate the mutated code back into natural language questions.
4. **Answer Verification**: Obtain deterministic answers by executing the mutated code.

### Reasoning Hierarchy (Mapping to Ladder of Causation)

Inspired by Pearl's Ladder of Causation (Association $\rightarrow$ Intervention $\rightarrow$ Counterfactual), three reasoning levels are defined:

| Level | Pearl Equivalent | Mutation Type | Core Capability Tested |
|:---|:---|:---|:---|
| Level-1 | Association | Original Question | Baseline Performance |
| Level-2 | Intervention | SampleValues, UselessInfo, OverWriteValue | Generalization with an unchanged reasoning path |
| Level-3 | Counterfactual | AddDependence, InsertConditional, CounterFactual | Generalization requiring modifications to the reasoning path |

### Key Mutation Operations

**Level-2 Mutations** (preserving the original reasoning path):

- **SampleValues**: Replace numerical values of leaf nodes in the computation graph ($x_i \to x_i + \delta$, $\delta \in [-10, 10]$).
- **UselessInfo**: Add redundant nodes to the computation graph that do not affect the final answer.
- **OverWriteValue**: Overwrite intermediate variable values.

**Level-3 Mutations** (modifying the reasoning path):

- **AddDependence**: Add new dependent nodes, increasing the reasoning steps by one.
- **InsertConditional**: Insert conditional branches (if-else), changing the computational logic.
- **CounterFactual**: Assume a condition contrary to facts and re-derive the answer.

### Loss Function & Quality Assurance

The correctness of the mutation process is guaranteed through the following mechanisms:

1. **NL-to-Code Verification**: Ensure that the constants in the code correspond one-to-one with the values in the question.
2. **Code-to-NL Back-translation Verification**: Use GPT-4o to back-translate and execute it, verifying answer consistency.
3. **Manual Sampling Inspection**: Report the error rate of each mutation.

Core Model Selection:
- GSM8K NL→Code: Mixtral-8x7B
- GSM8K Code→NL: GPT-4o
- CLadder: Original causal engine + Meta-Llama-70B-Instruct
- Loop/CruxEval: Directly start from symbolic representations, no LLM required.

## Key Experimental Results

### Main Results: Mutation Performance of Various Models on GSM8K

| Model | Raw | SampleValues (L2) | OverWriteValue (L2) | UselessInfo (L2) | AddDependence (L3) | InsertConditional (L3) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| QwQ-32B | 100 | 98.2 | 92.6 | 99.1 | **59.6** | 95.6 |
| R1-Distill-Qwen-32B | 98.3 | 93.5 | 85.9 | 97.6 | **63.2** | 85.0 |
| GPT-o3-mini | 97.4 | 90.3 | 84.0 | 93.5 | **77.3** | 91.6 |
| GPT-4.5 | 97.5 | 89.8 | 81.3 | 95.5 | **61.5** | 89.3 |

### Ablation Study: Performance Degradation by Computation Steps

Average accuracy after controlling for question difficulty (number of computational steps), demonstrating that the performance drop is not solely due to increased difficulty:

| Mutation Type | 2 Steps | 3 Steps | 4 Steps | 5 Steps | 6 Steps |
|:---|:---:|:---:|:---:|:---:|:---:|
| Raw | 0.95 | 0.94 | 0.84 | 0.91 | 0.83 |
| SampleValues (L2) | 0.87 | 0.84 | 0.75 | 0.74 | 0.80 |
| UselessInfo (L2) | 0.91 | 0.90 | 0.90 | 0.81 | 0.88 |
| CounterFactual (L3) | 0.74 | 0.71 | 0.75 | 0.62 | 0.67 |
| InsertConditional (L3) | 0.62 | 0.68 | 0.65 | 0.61 | 0.57 |
| AddDependence (L3) | 0.57 | 0.47 | 0.46 | 0.45 | 0.42 |

### CruxEval Code Benchmark Results

| Model | Raw | Mutate String (L2) | Mutate Value (L2) | Redefine Function (L2) | Replace Operator (L2) | Swap Conditional (L2) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4.5 | 45.5 | 23.6 | 32.5 | 29.8 | 29.1 | 32.1 |
| GPT-o3-mini | 56.9 | 36.0 | 59.8 | 56.4 | 58.5 | 50.0 |

### Key Findings

1. **Level-3 mutations cause the largest performance drops**: AddDependence causes QwQ-32B to plummet from 100% to 59.6%, indicating that even the strongest models struggle with mutations that require updating the reasoning path.
2. **Significant drops persist under the same-step setting**: Under the same number of computational steps, the 2-step accuracy of Level-3 mutations (0.57) is lower than the 6-step accuracy on raw questions (0.83), demonstrating that the performance degradation is not driven by increased difficulty.
3. **Pervasive impact of SampleValues**: Merely changing numerical values leads to a 5-10% performance drop, suggesting that models tend to memorize specific values.
4. **Code reasoning is more fragile**: GPT-4.5 retains only 23.6% accuracy on Mutate String within CruxEval, suffering from nearly half of its performance.

## Highlights & Insights

- **Novel Mapping of the Ladder of Causation**: Introducing Pearl's causal framework into LLM reasoning evaluation provides a theoretical foundation rather than an ad-hoc definition of difficulty.
- **Deterministic Answer Guarantee**: Mutated question answers are obtained deterministically through symbolic representations (executable code), avoiding noise from hand-labeling or LLM tagging.
- **Cross-Domain Unified Framework**: The same framework covers math (GSM8K), code (CruxEval/Loop), and logic (CLadder), spanning three core reasoning domains.
- **Modular Design**: Each step of the pipeline can be substituted independently. Adapting to a new benchmark only requires writing about 150 lines of code (50 lines of mutation definitions + 100 lines for Code→NL prompts).
- **Dynamic Benchmark Paradigm**: Generating different variants for each evaluation, fundamentally resolving the benchmark leakage issue.

## Limitations & Future Work

1. **NL→Code step relies on LLMs**: For complex reasoning tasks, the NL→Code translation might introduce errors and requires custom prompts for each benchmark.
2. **Entanglement of mutation and difficulty**: Although the authors conducted a same-step analysis, Level-3 mutations inherently add reasoning steps, making it difficult to completely isolate difficulty effects.
3. **Looseness in Ladder of Causation mapping**: Reviewers pointed out a gap between Level-3 mutations and strict definitions of counterfactuals, which resemble "intervention" rather than true "counterfactuals".
4. **Inability to detect non-memorization-related reasoning failures**: If a model lack both memorization and reasoning capability, the performance drop cannot be solely attributed to memorization.
5. **Only covering zero/few-shot settings**: The performance of fine-tuned models on mutated benchmarks remains unexplored.

## Related Work & Insights

- **GSM-Symbolic** (Mirzadeh et al., 2024): Only explored Level-2 mutations like SampleValues; RE-IMAGINE extends this to Level-3.
- **GSM-IC** (Shi et al., 2023): Investigated the interference of irrelevant information on reasoning, corresponding to the UselessInfo of RE-IMAGINE.
- **iGSM** (Ye et al., 2024): Uses symbolic structures to generate synthetic pipelines, but lacks a unified multi-level framework.
- **Pearl's Ladder of Causation** (2009): Provides a theoretical foundation for stratifying reasoning capabilities.
- **Reasoning Elicitation via Counterfactual Feedback** (Hüyük et al., 2024): The inspiration source for Level-3 Bi-Counterfactual.

**Insight**: Incorporating causal reasoning theory into LLM evaluation is a promising direction. Future directions could explore: (1) whether incorporating mutated data in training improves true reasoning capabilities; (2) how models handle different levels of mutations from a mechanistic interpretability perspective.

## Rating

| Metric | Score (1-10) | Comments |
|:---|:---:|:---|
| Novelty | 7 | The mapping to the Ladder of Causation and the unified mutation framework are novel, but Level-2 mutations overlap with prior work. |
| Technical Depth | 7 | The pipeline design is comprehensive and the verification mechanism is rigorous, though the theoretical analysis could be deeper. |
| Experimental Thoroughness | 8 | Four benchmarks, multiple model families, and comprehensive ablation analyses; reviewers evaluated it as "extremely thorough". |
| Writing Quality | 7 | The framework is clearly described, but reviewers pointed out layout issues such as duplicate paragraphs. |
| Value | 8 | Dynamic benchmark generation and modular design possess high practical value. |
| Overall Recommendation | 7.5 | A solid, systematic study connecting causal theory to LLM evaluation, highly worthy of attention. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reasoning is All You Need for Video Generalization: A Counterfactual Benchmark with Sub-question Evaluation](../../ACL2025/causal_inference/reasoning_is_all_you_need_for_video_generalization_a_counterfactual_benchmark_wi.md)
- [\[ACL 2025\] CoA-Reasoning: Explorations on Counterfactual Analysis in Physical Reasoning of LVLMs](../../ACL2025/causal_inference/coa-reasoning_explorations_on_counterfactual_analysis_in_physical_reasoning_of_l.md)
- [\[ACL 2025\] Causal Graph based Event Reasoning using Semantic Relation Experts](../../ACL2025/causal_inference/causal_graph_based_event_reasoning_using_semantic_relation_experts.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)
- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)

</div>

<!-- RELATED:END -->
