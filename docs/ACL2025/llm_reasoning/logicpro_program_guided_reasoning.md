---
title: >-
  [Paper Note] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning
description: >-
  [ACL 2025][Reasoning][Data Synthesis] This paper proposes LogicPro, a data synthesis method that leverages LeetCode algorithmic problems and Python code solutions as logic sources. Through a three-step pipeline ("problem generation $\rightarrow$ code intermediate variable extraction $\rightarrow$ program-guided reasoning generation"), it synthesizes 540K high-quality textual reasoning data from 2,360 algorithmic problems, significantly outperforming existing reasoning dataset…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Data Synthesis"
  - "Logical Reasoning"
  - "Program-Guided"
  - "LeetCode"
  - "Intermediate Variables"
date: 2026-05-08
content_hash: 4d22227261674b4d
---

# LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning

**Conference**: ACL 2025  
**arXiv**: [2409.12929](https://arxiv.org/abs/2409.12929)  
**Code**: [https://github.com/jiangjin1999/LogicPro](https://github.com/jiangjin1999/LogicPro)  
**Area**: LLM Reasoning  
**Keywords**: Data Synthesis, Logical Reasoning, Program-Guided, LeetCode, Intermediate Variables  

## TL;DR
This paper proposes LogicPro, a data synthesis method that leverages LeetCode algorithmic problems and Python code solutions as logic sources. Through a three-step pipeline ("problem generation $\rightarrow$ code intermediate variable extraction $\rightarrow$ program-guided reasoning generation"), it synthesizes 540K high-quality textual reasoning data from 2,360 algorithmic problems, significantly outperforming existing reasoning datasets on multiple OOD benchmarks such as BBH27, LogicBench, and DROP.

## Background & Motivation
**Background**: Synthetic data is widely used to improve LLM reasoning capabilities. Mature methods such as evol-instruct and back-translation exist in the math and code domains, but data synthesis in the complex logical reasoning domain is relatively lacking.

**Limitations of Prior Work**: Existing logical reasoning data synthesis mainly relies on propositional logic (e.g., RuleTakers, ProofWriter) or formal languages (e.g., FLD). The logical patterns are simplistic (e.g., modus ponens), lacking diversity, and poorly connected to real-world task scenarios.

**Key Challenge**: Algorithmic problems can be easily solved using code, but LLMs often fail when the same problems are converted into natural language reasoning problems. This gap suggests a new source of training data: guiding textual reasoning learning using program logic.

**Goal**: How to efficiently synthesize high-quality, diverse logical reasoning training data with standard answers and reasoning paths using rich algorithmic problems and their code solutions?

**Key Insight**: The intermediate variable outputs of code naturally correspond to the "intermediate steps" of human reasoning—for instance, the step-by-step computation of the Fibonacci sequence corresponds to the step-by-step reasoning in the climbing stairs problem.

**Core Idea**: Algorithmic problems + Python code = natural logical reasoning seeds; intermediate variables of code = the backbone of the reasoning process. Based on this, high-quality training data with accurate answers and reasoning paths can be synthesized at scale.

## Method

### Overall Architecture
Three-step pipeline:
1. **Problem Construction**: LeetCode algorithmic problem + specific test case $\rightarrow$ textual reasoning problem
2. **Intermediate Variable Extraction**: Python code + test case $\rightarrow$ run code $\rightarrow$ obtain final answer + key intermediate variable values
3. **Program-Guided Reasoning Generation**: Textual problem + intermediate variable outputs $\rightarrow$ LLM generates complete reasoning process

### Key Designs

1. **Data Collection and Test Case Construction**:

    - **Function**: Collects 2,360 official LeetCode problems along with their standard Python solutions, and uses GPT-4 to generate 150 test cases for each problem (deduplicated and filtered after sampling 3 times).
    - **Mechanism**: Each problem ultimately has ~300 test cases, and each test case can generate a different reasoning problem instance, achieving high scalability.
    - **Design Motivation**: Algorithmic problems involve diverse logical patterns such as recursion, iteration, and data structure operations, and are naturally connected to real-world scenarios (e.g., path planning, resource allocation).

2. **Step 1: Constructing Textual Reasoning Problems**:

    - **Function**: Combines the algorithmic problem with a specific test case, which is then translated by an LLM into a natural language reasoning problem with randomized background information.
    - **Example**: LeetCode-70 "Climbing Stairs" + test case n=17 $\rightarrow$ "A person wants to climb 17 stairs. They can climb 1 or 2 stairs at a time. How many distinct ways are there?"
    - **Quality Control**: Consistency checks (whether the textual problem corresponds to the code) + solvability checks (filtering out meaningless problems) reduced the data from 699K to 595K.

3. **Step 2: Code Intermediate Variable Extraction**:

    - **Function**: Modifies the Python code to insert print statements for key intermediate variables, and executes the code to capture the full execution trace.
    - **Mechanism**: For the climbing stairs problem, print the value of each step in the Fibonacci sequence: `dp[1]=1, dp[2]=2, ..., dp[17]=1597`.
    - **Design Motivation**: Intermediate variables naturally correspond to the intermediate steps of reasoning, providing a "backbone" for subsequent reasoning generation.
    - **Quality Control**: Execution checks—filter out code that fails to run or throws exceptions (50K filtered).

4. **Step 3: Program-Guided Reasoning Generation**:

    - **Function**: Feeds the textual problem and the intermediate variable outputs to LLaMA-3.1-70B to generate the complete textual reasoning process.
    - **Mechanism**: Intermediate variables serve as "anchors" to ensure each step of the reasoning process is supported by the code execution results.
    - **Design Motivation**: Prevents the LLM from generating inaccurate reasoning steps by utilizing the program execution output as ground truth guidance.

### Loss & Training
- **Final Dataset**: 540K question-answer pairs
- **Training Mixture**: 100K general data (OpenHermes-2.5) + LogicPro data
- **SFT Training**: Megatron-LM, lr=1e-5, cosine schedule, 3 epochs, 32×A100

## Key Experimental Results

### Main Results (Qwen2-7B, Baseline Comparison)

| Synthetic Data | BBH27 | LogicBench | DROP | AR-LSAT | BoardgameQA | FOLIO | GSM8K | Average |
|----------|-------|-----------|------|---------|-------------|-------|-------|------|
| RuleTakers | 45.4 | 59.1 | 65.7 | 16.5 | 42.2 | 44.6 | 80.9 | 45.8 |
| LogicNLI | 43.3 | 71.3 | 67.4 | 17.8 | 45.3 | 41.7 | 81.6 | 45.0 |
| LogicBench | 44.7 | *95.9 | 67.4 | 17.8 | 41.4 | 38.7 | 82.1 | 46.6 |
| FLD | 42.0 | 69.5 | 68.3 | 14.8 | 34.9 | 45.6 | 80.0 | 43.8 |
| **LogicPro** | **50.9** | 73.5 | **68.3** | **19.1** | **48.1** | **46.1** | 81.5 | **51.2** |

### LLM Experiments (Qwen2-72B)

| Synthetic Data | BBH27 | LogicBench | DROP | AR-LSAT | Average |
|----------|-------|-----------|------|---------|------|
| CLUTRR | 68.1 | 79.0 | 78.4 | 24.4 | 66.7 |
| LogicBench | 67.1 | *97.0 | 77.9 | 24.8 | 67.2 |
| **LogicPro** | **72.4** | 81.7 | **79.6** | **27.4** | **70.4** |

### Key Findings
- **Comprehensive OOD Superiority**: LogicPro achieves state-of-the-art results across 7 out of 10 metrics across all benchmarks. Crucially, all benchmarks are OOD (out-of-distribution) for LogicPro (note that LogicBench's score on LogicBench is in-domain at *95.9, which is an unfair comparison for others).
- **Largest Improvement on BBH27**: This is the core benchmark for complex logical reasoning. LogicPro improves the Qwen2-7B performance from the highest baseline of 46.2 to 50.9 (+4.7).
- **Scaling Effects**: The advantages of LogicPro are even more pronounced on the 72B model (average of 70.4 vs. 67.2 for the runner-up).
- **Data Efficiency**: Synthesized 540K instances from only 2,360 algorithmic problems, demonstrating continuous scalability as more problem seeds are collected.

## Highlights & Insights
- **Bridging Code and Textual Reasoning is Ingenious**: Code intermediate variables naturally correspond to intermediate reasoning steps. This insight effectively injects the precision of code execution into textual reasoning data synthesis.
- **LeetCode as a Logic Source is a Fresh Perspective**: Prior logical reasoning data synthesis focused heavily on propositional logic or formal languages. LogicPro introduces algorithmic logic (recursion, data structures, search, etc.), significantly expanding the diversity of logical patterns.
- **Triple Quality Control** (consistency check + solvability check + execution check) ensures high-quality synthesized data.

## Limitations & Future Work
- High dependency on GPT-4 for test case generation and LLaMA-70B for reasoning generation, resulting in non-negligible synthesis costs.
- The number of LeetCode problems is limited (~2,360). Although scalable, it requires sourcing more algorithmic problems.
- The selection of intermediate variables depends on LLM judgment, which may occasionally omit critical reasoning steps.
- Lack of comparison with the latest reasoning models (such as o1/R1) on synthetic data.

## Related Work & Insights
- **vs FLD (Morishita et al. 2023)**: FLD synthesizes based on formal languages, resulting in relatively simplistic logical patterns. LogicPro is based on algorithmic problems, offering more diverse logical patterns.
- **vs ProofWriter (Tafjord et al. 2021)**: ProofWriter is based on propositional logic derivation rules; LogicPro draws its logic from practical algorithms.
- **vs evol-instruct**: evol-instruct synthesizes math problems by progressively scaling up complexity. LogicPro achieves complexity and diversity via different algorithmic seeds paired with various test cases.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Synthesizing reasoning data from algorithmic problems and code execution results is highly novel. The insight of "intermediate variables in code = reasoning step backbone" is brilliant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluations across 4 models (7B to 72B), 10 benchmarks, and multiple baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear pipeline description and complete quality control modules.
- **Value**: ⭐⭐⭐⭐⭐ Provides an extremely practical, highly scalable, and high-quality method for logical reasoning data synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ClozeMath: Improving Mathematical Reasoning in Language Models by Learning to Fill Equations](clozemath_improving_mathematical_reasoning_in_language_models_by_learning_to_fil.md)
- [\[ACL 2025\] Complex Reasoning with Natural Language Contexts and Background Knowledge](complex_reasoning_with_natural_language_contexts_and_background_knowledge.md)
- [\[ICML 2025\] MARGE: Improving Math Reasoning for LLMs with Guided Exploration](../../ICML2025/llm_reasoning/marge_improving_math_reasoning_for_llms_with_guided_exploration.md)
- [\[ACL 2025\] Enhancing Retrieval Systems with Inference-Time Logical Reasoning](enhancing_retrieval_systems_with_inference-time_logical_reasoning.md)
- [\[ACL 2025\] Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework](aristotle_logical_reasoning.md)

</div>

<!-- RELATED:END -->
