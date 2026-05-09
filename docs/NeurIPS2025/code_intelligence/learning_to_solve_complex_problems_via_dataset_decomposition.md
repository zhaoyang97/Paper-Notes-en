---
title: >-
  [Paper Note] Learning to Solve Complex Problems via Dataset Decomposition
description: >-
  [NeurIPS 2025][dataset decomposition] This paper proposes Decomp, a method that employs a teacher model (GPT-4o) to recursively decompose complex math problems into simpler subproblems along reasoning steps, constructs a concept dependency graph to quantify difficulty, and trains student models following an easy-to-hard curriculum. Qwen2.5-1.5B achieves 51.6% on MATH-500 (surpassing MuggleMath's 50.4% with 147K samples), while Qwen3-4B reaches 16.7% on AIME2025 using only 385 samples (surpassing Qwen2.5-72B's 15.0%).
tags:
  - NeurIPS 2025
  - dataset decomposition
  - curriculum learning
  - mathematical reasoning
  - subproblem generation
  - skill graph
date: 2026-05-08
content_hash: f108d12bc040bd17
---

# Learning to Solve Complex Problems via Dataset Decomposition

**Conference**: NeurIPS 2025
**arXiv**: [2602.20296](https://arxiv.org/abs/2602.20296)
**Code**: None
**Area**: Code Intelligence
**Keywords**: dataset decomposition, curriculum learning, mathematical reasoning, subproblem generation, skill graph

## TL;DR
This paper proposes Decomp, a method that employs a teacher model (GPT-4o) to recursively decompose complex math problems into simpler subproblems along reasoning steps, constructs a concept dependency graph to quantify difficulty, and trains student models following an easy-to-hard curriculum. Qwen2.5-1.5B achieves 51.6% on MATH-500 (surpassing MuggleMath's 50.4% with 147K samples), while Qwen3-4B reaches 16.7% on AIME2025 using only 385 samples (surpassing Qwen2.5-72B's 15.0%).

## Background & Motivation

**State of the Field**: A common paradigm for training small models on mathematical reasoning is SFT on datasets of (problem, full solution) pairs. Data augmentation methods such as MetaMath (problem rephrasing) and MuggleMath (multi-solution sampling) can expand dataset size.

**Limitations of Prior Work**: (a) Direct SFT does not simulate the human learning process of "master fundamentals before advanced topics," leading to overfitting on small datasets (experiments show that SFT on AIME2024 even degrades AIME2025 performance); (b) conventional curriculum learning only reorders examples within a dataset by difficulty, yet the dataset itself may lack simple examples covering foundational skills; (c) data augmentation typically produces same-difficulty paraphrases rather than substantially simpler subproblems.

**Root Cause**: Complex reasoning requires composing multiple foundational skills, yet training data typically contains only complete, complex problems, preventing models from independently acquiring each underlying skill.

**Paper Goals**: How to automatically generate subproblems spanning all difficulty levels from complex problems, and quantify difficulty to construct an effective curriculum?

**Starting Point**: Exploit the structure of multi-step reasoning solutions — each individual reasoning step defines a simpler subproblem and can be recursively decomposed.

**Core Idea**: Recursively decompose reasoning steps → generate independent subproblems for each step → construct a concept dependency graph to quantify difficulty → train in easy-to-hard order.

## Method

### Overall Architecture
**Input**: A mathematical problem dataset $\mathcal{D}_{raw} = \{(q, cot, a)\}$. Decomp proceeds in three stages: (1) a teacher model (GPT-4o) recursively decomposes each problem into a subproblem tree; (2) concept labels are extracted to build a cross-dataset concept dependency graph, quantifying the difficulty of each subproblem; (3) problems are bucketed by difficulty and used for curriculum-based SFT. **Output**: A student model trained via the curriculum.

### Key Designs

1. **Recursive Problem Decomposition**

   - **Function**: Transforms each reasoning step of a complex problem into an independent subproblem; applicable recursively.
   - **Mechanism**: Given a solution $cot$, the teacher model segments it into $\leq k$ reasoning steps $[s_1, \ldots, s_k]$. For each step $s_i$, a concept label $t_i$ (e.g., "GCD," "square root") is extracted, and an independent subproblem $q_i$ is generated with its solution $cot_i$ and answer $a_i$ provided by the teacher. Verification is performed by having the teacher re-solve the subproblem without the original context, and a symbolic verifier checks answer consistency. Upon passing verification, $cot_i$ is recursively decomposed until the maximum depth $D$ is reached.
   - **Design Motivation**: Each reasoning step intrinsically defines a simpler problem. Recursive decomposition ensures full coverage of the skill spectrum, from the most elementary (e.g., "What is $3^3$?") to the original difficulty level. The verification step ensures that generated subproblems have correct answers.

2. **Concept Dependency Graph and Difficulty Quantification**

   - **Function**: Constructs a graph of prerequisite relations between concepts across the dataset and quantifies the difficulty of each subproblem.
   - **Mechanism**: All concept labels from subproblems are organized into a directed acyclic graph (DAG); an edge $t_p \rightarrow t_c$ is added if concept $t_c$ is the label of a subproblem derived from a step labeled $t_p$. Synonym labels (e.g., "GCD" = "greatest common divisor") are merged via embedding-based clustering. The difficulty score is defined as $\alpha_1 \cdot SC(s)$ (structural complexity: number of subproblems) $+ \alpha_2 \cdot CD(s)$ (concept depth: depth of the label in the DAG).
   - **Design Motivation**: Reasoning step count alone is insufficient (e.g., "$1+1+\cdots+1$" has many steps but trivial concepts), and concept depth alone is insufficient (the same concept can appear in applications of varying complexity). The two-dimensional combination yields more accurate difficulty estimates.

3. **Curriculum Learning Training**

   - **Function**: Partitions data into difficulty buckets and trains sequentially from easiest to hardest.
   - **Mechanism**: The decomposed dataset is divided into $K$ buckets $\mathcal{D}'_1, \ldots, \mathcal{D}'_K$ by difficulty quantile. The model is trained sequentially from $\mathcal{D}'_1$ to $\mathcal{D}'_K$ with early stopping at each stage. Total training budget is compute-matched to baselines.
   - **Design Motivation**: With identical data, curriculum ordering yields a 25.6% relative improvement over random ordering on AIME, demonstrating that training order is critical for learning efficiency on small datasets.

### Loss & Training
- Student models: Qwen2.5-1.5B / Qwen3-4B-Base
- 5 epochs, batch size 16, lr $10^{-5}$, cosine annealing
- bfloat16 precision, AdamW optimizer
- Evaluation temperature 0 (greedy decoding)

## Key Experimental Results

### Main Results

| Model / Method | Training Size | MATH-500 | AIME2025 |
|---|---|---|---|
| Qwen2.5-1.5B Base | — | 47.2 | — |
| SFT (MATH full) | 7500 | 47.6 | — |
| SFT-MetaMATH-Aug | 2638 | 37.2 | — |
| SFT-MuggleMath | 147787 | 50.4 | — |
| **Decomp (Ours)** | **4500** | **50.8** | — |
| **Decomp + Curriculum** | **4500** | **51.6** | — |
| Qwen3-4B SFT (AIME24) | 30 | — | 3.3 |
| SFT-MetaMATH-Aug | 114 | — | 4.4 |
| **Decomp + Curriculum** | **385** | — | **16.7** |
| Qwen2.5-72B-Instruct | — | — | 15.0 |

### Code Generation (HumanEval)

| Method | pass@1 |
|---|---|
| Qwen2.5-1.5B-Instruct Base | 34.15 |
| SFT | 35.37 |
| **Decomp (Ours)** | **42.68** |

### Key Findings
- **Decomposition substantially outperforms data augmentation at the same scale**: Decomp with 4,500 samples surpasses MuggleMath with 147K samples (50.8 vs. 50.4 on MATH-500), demonstrating that "skill decomposition" is more effective than "quantity scaling."
- **Curriculum ordering has a significant impact**: With identical decomposed data, curriculum ordering yields a 3.4% absolute improvement (25.6% relative) on AIME and 0.8% on MATH compared to random ordering.
- **Small models can outperform large models**: Qwen3-4B + Decomp (16.7%) surpasses Qwen2.5-72B-Instruct (15.0%) on AIME2025 using only 385 training samples.
- **Direct SFT can degrade performance**: SFT on AIME2024 yields only 3.3% on AIME2025, below the base model's 10.0% — a clear case of small-data overfitting.
- **The method transfers to code generation**: Decomp achieves a 7.3% absolute improvement on HumanEval.

## Highlights & Insights
- **Reasoning step = subproblem**: A simple yet profound insight — each step in a multi-step reasoning solution intrinsically defines a simpler, self-contained problem. Recursive application yields a complete skill gradient.
- **Incidental value of the concept dependency graph**: The constructed DAG serves not only for difficulty quantification but also as a dataset "map" — revealing which skills are covered, which are absent, and how different datasets relate to one another.
- **Extreme sample efficiency**: 385 decomposed samples on AIME2025 outperform a 72B model, providing strong evidence for "quality over quantity" and "structured training over brute-force SFT."

## Limitations & Future Work
- Relies on a powerful teacher model (GPT-4o); decomposition quality is bounded by teacher capability.
- The verification step checks only final answer consistency, without validating the correctness of intermediate reasoning.
- Concept label quality and the clustering parameter $\delta$ require tuning.
- Validation is limited to mathematics and code; applicability to domains such as natural language reasoning remains untested.
- The number of curriculum stages $K$ is a hyperparameter without thorough ablation.

## Related Work & Insights
- **vs. MetaMath**: MetaMath performs same-difficulty paraphrasing (rephrasing, backward reasoning), whereas Decomp performs cross-difficulty decomposition; the performance gap is substantial (37.2 vs. 50.8 on MATH-500).
- **vs. S1**: S1 achieves strong results on a 32B model with 1,000 curated samples but does not generalize well to AIME2025 (13.3); Decomp achieves 16.7 with a 4B model and 385 samples.
- **vs. TinyStories**: The conceptual motivation is similar — synthesizing simple data to teach small models foundational skills — but Decomp automatically decomposes real complex data rather than generating data entirely from scratch.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of recursive decomposition and concept graph is novel, though curriculum learning itself is not new
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-domain validation on math and code, though dataset and model scale are limited
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with intuitive case studies
- Value: ⭐⭐⭐⭐⭐ Exceptional sample efficiency carries significant practical value for small model training

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[NeurIPS 2025\] Co-Evolving LLM Coder and Unit Tester via Reinforcement Learning](co-evolving_llm_coder_and_unit_tester_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)
- [\[NeurIPS 2025\] Learning From Design Procedure To Generate CAD Programs for Data Augmentation](learning_from_design_procedure_to_generate_cad_programs_for_data_augmentation.md)
- [\[ICLR 2026\] Learning to Reason without External Rewards](../../ICLR2026/code_intelligence/learning_to_reason_without_external_rewards.md)

<!-- RELATED:END -->
