---
title: >-
  [Paper Note] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding
description: >-
  [ACL 2025][Synthetic Dataset] KodCode proposes a three-stage synthetic data pipeline (synthesizing programming problems $\rightarrow$ solution + unit test self-verification $\rightarrow$ post-training data synthesis) to construct 447K verified programming question-solution-test triplets. The fine-tuned models outperform Qwen2.5-Coder-32B-Instruct and DeepSeek-R1-Distill-Llama-70B on benchmarks such as HumanEval, MBPP, BigCodeBench, and LiveCodeBench.
tags:
  - "ACL 2025"
  - "Synthetic Dataset"
  - "Code Generation"
  - "Self-Verification"
  - "Reinforcement Learning"
  - "Reasoning Model"
date: 2026-05-08
content_hash: 9b54d1b68ba81ab2
---

# KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding

**Conference**: ACL 2025  
**arXiv**: [2503.02951](https://arxiv.org/abs/2503.02951)  
**Code**: [Yes](https://kodcode-ai.github.io) / [Models](https://huggingface.co/KodCode)  
**Area**: Other  
**Keywords**: Synthetic Dataset, Code Generation, Self-Verification, Reinforcement Learning, Reasoning Model

## TL;DR

KodCode proposes a three-stage synthetic data pipeline (synthesizing programming problems $\rightarrow$ solution + unit test self-verification $\rightarrow$ post-training data synthesis) to construct 447K verified programming question-solution-test triplets. The fine-tuned models outperform Qwen2.5-Coder-32B-Instruct and DeepSeek-R1-Distill-Llama-70B on benchmarks such as HumanEval, MBPP, BigCodeBench, and LiveCodeBench.

## Background & Motivation

Training high-performance code LLMs requires high-quality, verifiable training data, but existing resources suffer from three major limitations:

**Limited scale of human-annotated datasets**: Although TACO (26K), APPS (10K), and CodeContests (13K) are of high quality, their scales are limited.

**Insufficient quality of synthetic datasets**:
   - Code Alpaca (20K): Low diversity, low difficulty, no unit tests.
   - Evol Instruct (111K): Low diversity, no test verification.
   - OSS Instruct (75K): Moderate diversity, no test verification.

**Lack of a unified large-scale, multi-difficulty, verifiable dataset**: No existing dataset simultaneously satisfies high diversity, mixed difficulty, and validation via unit tests.

The core goal of KodCode is to construct a **large-scale** (447K), **diverse** (12 subsets), **challenging** (from easy to competitive level), and **verifiable** (comes with unit tests) synthetic coding dataset.

## Method

### Overall Architecture

A three-stage pipeline: Step 1 Synthesizing diverse programming problems $\rightarrow$ Step 2 Generating solutions and unit tests with self-verification $\rightarrow$ Step 3 Post-training data synthesis (style conversion + CoT response generation by reasoning models).

### Key Designs

1. **Step 1: Programming Problem Synthesis (12 subsets, 5 methods)**:

    - **Magpie-Prefill**: Leverages the prefilled suffix ("Write a Python function that") + Qwen2.5-Coder-7B completion to efficiently generate simple problems.
    - **Evaluation Task Expansion**: Uses GPT-4o as a teacher LLM to generate new problems after analyzing seed problem structures (seeds sourced from LeetCode, Codeforces, APPS, TACO, and CodeContests).
    - **DSA Knowledge to Problems**: Extracts from Python DSA code snippets to generate data structure and algorithm problems.
    - **Technical Documentation to Problems**: Converts documentation from libraries such as Flask, Pandas, and PyTorch into programming tasks with built-in quality control.
    - **Additional Problems**: Generated using Magpie + 7 open-source LLMs, with high-quality problems retained through filtering with an LLM classifier.
    - **Deduplication**: Uses `all-mpnet-base-v2` embeddings + FAISS nearest neighbor distance filtering.
    - Design Motivation: Multi-source and multi-method approach ensures the diversity and difficulty coverage of the problems.

2. **Step 2: Solution and Test Generation (Self-Verification Mechanism)**:

    - Uses GPT-4o to simultaneously generate both the solution and unit tests.
    - Executes unit tests to verify the correctness of the solution.
    - Uses pytest-cov to conduct branch coverage analysis, retaining only triplets with 100% branch coverage.
    - **Key Innovation — Allocating Extra Attempts for Hard Problems**:
        - Up to $n=10$ attempts per problem.
        - Generates the solution and tests from scratch in each attempt (to avoid cascading failures caused by initial faulty tests).
        - Retains new versions only if their branch coverage is no lower than prior attempts.
        - Problems that still fail after $n$ attempts are discarded (as they may be inherently flawed).
        - Naturally generates difficulty labels: categorized into easy/medium/hard based on pass rates.
    - Yields 279K verified triplets in the end.
    - Design Motivation: Prevents easy-problem bias introduced by simply discarding hard problems.

3. **Step 3: Post-Training Data Synthesis**:

    - **Style Converter**: Rewrites natural language problems into Python function signature format, paired with the solution and test inputs.
    - Generates an additional 168K triplets (totaling 447K), which are directly applicable for RL training.
    - **SFT Data Generation**: Uses DeepSeek R1 to generate CoT responses with 3 attempts per problem + test-based reject sampling.
    - Design Motivation: Bridges the gap between programming problem formats and training data formats.

### Loss & Training

- **SFT**: Qwen2.5-Coder-32B-Instruct, learning rate 1e-5, maximum sequence length 16384.
- **RL (GRPO)**: Qwen2.5-7B-Instruct-1M / Qwen2.5-Coder-7B-Instruct, 256 steps, 16 rollouts per problem, binary reward (1 if all tests pass, 0 otherwise).

## Key Experimental Results

### Main Results

| Model | HumanEval | MBPP | BCB-C Full | BCB-C Hard | BCB-I Full | BCB-I Hard | Average |
|------|-----------|------|-----------|-----------|-----------|-----------|------|
| Qwen2.5-Coder-32B-Inst | 90.9 | 90.2 | 57.6 | 31.1 | 49.4 | 25.7 | 59.25 |
| DeepSeek-R1-Distill-70B | 89.0 | 81.7 | 53.5 | 25.7 | 43.9 | 25.7 | 57.79 |
| Bespoke-Stratos-32B | 88.4 | 88.1 | 56.2 | 33.1 | 47.3 | 27.0 | 59.64 |
| **KodCode-32B-SFT-50K** | **92.7** | 89.9 | **59.8** | **37.8** | **51.1** | **32.4** | 61.22 |
| **KodCode-32B-SFT-Hard-18K** | 90.9 | 89.2 | 59.7 | 37.2 | 50.5 | 31.1 | **61.26** |

### Ablation Study

| Data Selection | BCB-C Hard | BCB-I Hard | LCB Hard |
|---------|-----------|-----------|---------|
| KodCode-SFT-Hard-10K | **39.9** | **31.8** | **6.3** |
| KodCode-SFT-10K (Random) | 38.5 | 27.7 | 4.8 |
| KodCode-SFT-NoConvert-10K | 35.1 | 28.4 | 5.6 |

### RL Experiments

| Model | Steps | BCB-C Full | BCB-I Full | HumanEval | Average |
|------|------|-----------|-----------|-----------|------|
| Qwen2.5-Coder-7B-Inst (Baseline) | - | 52.0 | 41.8 | 91.5 | 52.32 |
| + GRPO KodCode | 128 | 52.5 | 42.2 | 90.9 | 53.56 |
| + GRPO KodCode | 256 | **53.7** | **42.9** | 90.2 | **53.99** |

### Key Pipeline Analysis Metrics

| Validation Metric | MBPP | LiveCodeBench-V5 |
|---------|------|-----------------|
| Self-verification pass rate | 88.9% (80/90) | 49.9% (190/381) |
| Pass rate on human tests | **97.5%** (78/80) | **99.47%** (189/190) |
| Pass@1 $\rightarrow$ Pass@5 | Average gain of 20%+ | - |
| Potential contamination count | 94/447K (0.02%) | - |

### Key Findings

1. **KodCode SFT model achieves comprehensive SOTA**: Outperforms the strongest baseline on BigCodeBench Hard by 4.7% (Complete) and 5.4% (Instruct).
2. **Key value of hard problems**: Hard-10K performs 4.1% higher than random 10K on BCB-I Hard (31.8 vs 27.7).
3. **Style converter effectiveness**: Removing the style converter drops BCB-C Hard performance from 38.5 to 35.1 (-3.4%).
4. **Value of extra attempts**: Pass@1 $\rightarrow$ Pass@5 improves by 20%+, and Pass@5 $\rightarrow$ Pass@10 further gains 4%.
5. **Reliable self-verification**: The retained solutions achieve a pass rate of 97.5%-99.5% on human-annotated tests.
6. **Reinforcement learning is effective**: GRPO consistently boosts performance, and more steps lead to further improvements.

## Highlights & Insights

- **Core innovation lies in pipeline design rather than model architecture**: Each step of the three-stage pipeline features clear design considerations.
- **Clever strategy for handling difficult problems**: Utilizing extra attempts instead of discarding them both retains hard problems and naturally generates difficulty labels.
- **Self-verification + reject sampling** achieves low-cost, highly reliable data quality assurance.
- **Dual applicability of the dataset for SFT and RL**: The solution-test pairs are naturally suited for designing RL reward signals.
- t-SNE visualization clearly demonstrates the diversity advantage of KodCode: covering the entire space rather than clustering in a single corner.

## Limitations & Future Work

1. **Limited performance on LiveCodeBench-Hard**: Competitive-level programming problems remain a weakness, requiring more high-difficulty problems.
2. Data synthesis relies on GPT-4o and DeepSeek R1, incurring high costs.
3. Limited to the Python language, lacking multilingual support.
4. Optimal active selection strategies for post-training data remain unexplored.
5. Lack of synthetic data for repository-level code.

## Related Work & Insights

- Shares a similar concept of teacher-model test generation with OpenCoder (Huang et al., 2024), but KodCode incorporates hard problem retention and style conversion.
- The mutation-based test expansion of EvalPlus (Liu et al., 2024) can complement the self-verification of KodCode.
- The pipeline design can be generalized to other domains that require verifiable data, such as mathematical reasoning.
- Substantiates the perspective that synthetic data can enable smaller models to outperform larger ones.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The pipeline design is novel (particularly the hard problem retention strategy), but the core components are based on existing technologies.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Extremely comprehensive pipeline analysis (self-verification, Pass@k, contamination, diversity) + performance evaluations (SFT/RL/ablation).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logic, rich visualizations (t-SNE, Sankey diagram, difficulty distribution), and highly detailed pipeline descriptions.
- **Value**: ⭐⭐⭐⭐⭐ — Open-sourced dataset, open-sourced models, and reproducible methods; delivers significant infrastructural value to the code LLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] SynDaCaTE: A Synthetic Dataset for Evaluating Part-Whole Hierarchical Inference](../../ICML2025/others/syndacate_a_synthetic_dataset_for_evaluating_part-whole_hierarchical_inference.md)
- [\[CVPR 2026\] What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](../../CVPR2026/others/what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)
- [\[CVPR 2025\] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](../../CVPR2025/others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)

</div>

<!-- RELATED:END -->
