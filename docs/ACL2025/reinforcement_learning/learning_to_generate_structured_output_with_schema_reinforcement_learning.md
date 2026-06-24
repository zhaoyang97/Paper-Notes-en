---
title: >-
  [Paper Note] Learning to Generate Structured Output with Schema Reinforcement Learning
description: >-
  [ACL 2025][Reinforcement Learning][structured generation] Proposes SchemaBench, a benchmark containing approximately 40,000 JSON schemas, and Schema Reinforcement Learning (SRL), a training framework. By utilizing a fine-grained schema validator to provide dense reward signals combined with a Thoughts of Structure (ToS) reasoning mechanism, SRL improves LLM accuracy in complex JSON generation by up to 16% without compromising general reasoning abilities.
tags:
  - "ACL 2025"
  - "Reinforcement Learning"
  - "structured generation"
  - "JSON schema"
  - "reinforcement-learning"
  - "LLM"
  - "benchmark"
date: 2026-05-08
content_hash: 812ab2f968da0497
---

# Learning to Generate Structured Output with Schema Reinforcement Learning

**Conference**: ACL 2025  
**arXiv**: [2502.18878](https://arxiv.org/abs/2502.18878)  
**Code**: [https://github.com/thunlp/SchemaReinforcementLearning](https://github.com/thunlp/SchemaReinforcementLearning)  
**Area**: Reinforcement Learning  
**Keywords**: structured generation, JSON schema, reinforcement-learning, LLM, benchmark

## TL;DR

Proposes SchemaBench, a benchmark containing approximately 40,000 JSON schemas, and Schema Reinforcement Learning (SRL), a training framework. By utilizing a fine-grained schema validator to provide dense reward signals combined with a Thoughts of Structure (ToS) reasoning mechanism, SRL improves LLM accuracy in complex JSON generation by up to 16% without compromising general reasoning abilities.

## Background & Motivation

Large language models increasingly need to generate structured outputs (particularly in JSON format) in real-world applications to interface with automated systems and APIs. Several mainstream approaches currently exist:

**Prompting**: Direct prompting for generation, which is effective for simple schemas but prone to errors under complex logic.

**Tool Calls**: Converting model outputs to JSON, but often missing schema-specific syntax.

**Constrained Decoding** (e.g., Outlines, SGLang): Ensuring valid JSON by restricting the decoding space, which may, however, degrade output quality.

**Limitations of Prior Work**:

- Lack of comprehensive benchmarks to evaluate model generation capabilities under complex JSON schemas.
- Existing models (including GPT-4o) achieve an overall accuracy of only about 61% under complex schemas.
- SFT methods face a shortage of high-quality training data—both automated generators and model prompting struggle to generate compliant JSON at scale for complex schemas.
- Even after SFT, models still fail to learn basic JSON syntax in certain scenarios.

## Method

### 1. SchemaBench Benchmark Construction

A total of 108,528 schema files were scraped from the JSON Schema Store and GitHub. After filtering out external URIs (retaining 46,280) and validating syntax compliance (removing 5,574), approximately 40,706 valid schemas were obtained (36,960 for the training set and 3,746 for the test set). The average character length is about 35,754, with an average nesting depth of 16.7 layers.

**Task 1: Schema-only Generation** (Three sub-tasks):

- **Complex Schema**: Evaluates the model's ability to generate valid JSON under complex references (\) and logical combinations (anyOf/oneOf).
- **Custom Formats**: Tests the model's understanding of natural language descriptions for custom format constraints (phone numbers, file paths, RGB colors, base64 encoding, etc.), validated through const/pattern fields.
- **Escape Translation**: Evaluates the model's capability to correctly handle escape characters (\", \\, \n, etc.), where any escape error renders the entire JSON invalid.

**Task 2: Schema-constrained Reasoning**: Requires models to generate answers under schema constraints on GSM8K, MATH-500, MMLU, and ARC-Challenge, evaluating both reasoning capability and JSON compliance.

### 2. Schema Reinforcement Learning (SRL)

An online reinforcement learning method based on the PRIME framework, executed iteratively in three stages:

**Sampling Stage**: Generates K candidate responses for each schema task using the current policy model. Introduces the **Thoughts of Structure (ToS)** mechanism—inspired by CoT, the model performs structural reasoning in the form of JSON5 comments before generating the JSON, articulating the rationale for each key-value pair. Comments are ignored during validation, and only the final JSON is verified.

**Reward Stage**:

- **Fine-grained Schema Validator**: Instead of using a simple binary reward (pass/fail), it calculates the correctness ratio (number of correct tokens / total number of tokens). For partially valid JSON, it computes the correctness ratio of the valid portion; for JSON that fails to parse, it truncates at the error location and inserts control characters to continue validating the remainder. This significantly alleviates the sparse reward issue.
- **Advantage Estimation** employs a leave-one-out approach: subtracting the average reward of the remaining K-1 responses from the reward of the current response.

**Update Stage**: Updates the reward model using Cross Entropy loss, and updates the policy model using the PPO algorithm (with clipping).

### 3. Key Design Choices

- Training utilizes 37K schemas from the SchemaBench training set with a batch size of 32 and a learning rate of 5e-7.
- Supplementary training data (Collected JSON) includes UltraChat 6K, UltraInteract 6K, xLAM 20K, Glaive 20K, and ToolACE 10K.
- Tool definitions in tool-calling datasets are converted into JSON schema format.

## Key Experimental Results

### Table 1: SchemaBench Main Results (%)

| Model | Complex | Custom | Escape | Overall | GSM8K | MATH-500 |
|------|---------|--------|--------|---------|-------|----------|
| GPT-4o | 84.47 | 61.56 | 37.14 | 61.06 | 97.80 | 41.40 |
| Qwen-2.5 7B | 72.42 | 43.60 | 11.11 | 42.38 | 94.54 | 38.60 |
| LLaMA-3.1 8B | 64.26 | 33.07 | 12.02 | 36.45 | 95.91 | 85.60 |
| LLaMA-3.1 8B SFT | 74.56 | 46.64 | 60.58 | 60.59 | 89.46 | 63.80 |
| LLaMA-3.1 8B **SRL** | **90.48** | **78.67** | **69.86** | **79.67** | 90.90 | 88.00 |
| LLaMA-3.2 3B | 49.84 | 27.31 | 8.37 | 28.51 | 80.97 | 35.40 |
| LLaMA-3.2 3B SFT | 71.71 | 45.52 | 52.21 | 56.48 | 82.94 | 44.40 |
| LLaMA-3.2 3B **SRL** | **82.25** | **66.13** | **69.10** | **72.50** | 84.23 | 43.20 |

**Key Findings**: SRL substantially outperforms SFT on all schema-only sub-tasks. LLaMA-3.1 8B SRL achieves 79.67%, significantly exceeding GPT-4o's 61.06%.

### Table 2: BFCL-Live Downstream Task Results (%)

| Model | Simple | Multiple | Parallel | Multi-Para. | Overall |
|------|--------|----------|----------|-------------|---------|
| GPT-4o Tool | 36.43 | 37.22 | 18.75 | 41.67 | 59.13 |
| LLaMA-3.1 8B | 0.39 | 0.00 | 0.00 | 0.00 | 24.08 |
| LLaMA-3.1 8B SFT | 72.09 | 68.76 | 50.00 | 66.67 | 52.69 |
| LLaMA-3.1 8B **SRL** | 72.09 | 73.10 | **75.00** | 50.00 | **70.10** |
| LLaMA-3.2 3B | 4.26 | 13.11 | 0.00 | 0.00 | 35.72 |
| LLaMA-3.2 3B SFT | **74.03** | **74.64** | **68.75** | **58.33** | **64.10** |
| LLaMA-3.2 3B SRL | 65.50 | 64.22 | 50.00 | 29.17 | 57.00 |

LLaMA-3.1 8B SRL achieves 70.10% in the BFCL downstream function calling task, outperforming all baseline methods.

### Ablation Study

| Setting | Schema (%) | MATH-500 | ARC-C |
|------|-----------|----------|-------|
| LLaMA-3.2 3B baseline | 28.51 | 35.40 | 79.27 |
| + ORM | 31.15 | 39.40 | 78.92 |
| + ToS | **44.89** | 36.60 | 80.38 |
| + Fine-grained Validator | 35.59 | 35.60 | 79.10 |

ToS contributes the most (+13.74%), and the fine-grained validator also shows significant improvement (+4.44%), with the two being complementary.

## Highlights & Insights

- **SchemaBench is the first large-scale, high-complexity benchmark for JSON schema generation**, covering three types of challenging tasks, with an average schema nesting depth of 16.7, which aligns closely with real-world application scenarios.
- **Fine-grained Schema Validator** addresses the sparse reward problem in structured generation by providing effective gradient signals even for partially correct JSON, significantly outperforming simple binary rewards.
- **Thoughts of Structure (ToS)** mechanism is novel; it prompts the model to reason about structure before generating JSON and maintains format compatibility using JSON5 comments, which contributes the most in the ablation study.
- **SRL not only substantially improves structured generation capabilities but also largely preserves general reasoning abilities** (with no significant degradation in MATH-500 and ARC-C), in contrast to the performance drops typically seen with SFT.
- RL training is highly efficient, surpassing the SFT baseline after approximately half of the training process.

## Limitations & Future Work

1. **Format Limitations**: Currently focuses only on JSON schema and has not been extended to other structured formats such as YAML, XML, or TOML.
2. **Efficiency Bottlenecks**: The online sampling stage of SRL incurs high computational overhead, requiring repeated generation and validation.
3. **Limited Generalization in Small Models**: LLaMA-3.2 3B SRL performs worse than SFT on downstream BFCL tasks (57.00% vs 64.10%), suggesting that the transferability of SRL in extremely small models needs improvement.
4. **Lack of Broader Comparisons**: Stronger models like Claude or Gemini have not been evaluated, and comparisons with other alignment methods like GRPO or DPO are missing.
5. **Quality of ToS Comments**: The actual content quality of ToS comments has not been analyzed, and the relationship between reasoning length and performance remains unexplored.

## Related Work & Insights

| Method Category | Representative Work | Relationship to Ours |
|---------|---------|-----------|
| Prompting Methods | OpenAI Structured Outputs | Our benchmark proves that prompting is insufficient under complex schemas |
| Constrained Decoding | Outlines, SGLang, XGrammar | Can be complementary to SRL, but may degrade output quality |
| Tool Call Methods | ToolLLM, Toolformer | Relies on post-processing, making it difficult to align with standard schemas |
| Structured Generation Benchmarks | BFCL, StructuredBench | SchemaBench is larger and more complex (40K schemas) |
| RL for LLM | PRIME, PPO, RLHF | SRL introduces a schema validator and ToS on top of PRIME |

## Rating

- Novelty: ⭐⭐⭐⭐ — SchemaBench fills the gap in complex JSON generation evaluation; ToS and the fine-grained validator are meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison across multiple models and tasks with clear ablation studies, but lacks comparisons with stronger closed-source models and other RL methods.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich charts/tables, and well-articulated motivation.
- Value: ⭐⭐⭐⭐⭐ — JSON structured generation is a core challenge in LLM applications; both the benchmark and the method possess high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](../../ICLR2026/reinforcement_learning/learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ACL 2025\] Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering](prompt-based_personality_profiling_reinforcement_learning_for_relevance_filterin.md)
- [\[ICML 2025\] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](../../ICML2025/reinforcement_learning/robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)
- [\[NeurIPS 2025\] Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/learning_to_focus_prioritizing_informative_histories_with_structured_attention_m.md)

</div>

<!-- RELATED:END -->
