---
title: >-
  [Paper Note] TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure
description: >-
  [ACL 2025][LLM (Other)][Test case generation] This work proposes TestCase-Eval, a benchmark containing 500 Codeforces competitive programming problems and 100,000 human code submissions. Through two tasks, Fault Coverage and Fault Exposure, this benchmark systematically evaluates the ability of 19 LLMs in test case generation for algorithmic problems. The findings reveal that the strongest model, Qwen3-32B, achieves an exposure rate of only 43.8%…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Test case generation"
  - "LLM evaluation"
  - "fault coverage"
  - "fault exposure"
  - "competitive programming"
date: 2026-05-08
content_hash: 07c027c3d06bf66a
---

# TestCase-Eval: A Systematic Evaluation of Fault Coverage and Exposure

**Conference**: ACL 2025  
**arXiv**: [2506.12278](https://arxiv.org/abs/2506.12278)  
**Code**: [FlowRays/TestCase-Eval](https://github.com/FlowRays/TestCase-Eval)  
**Area**: LLM/NLP  
**Keywords**: Test case generation, LLM evaluation, fault coverage, fault exposure, competitive programming  

## TL;DR

This work proposes TestCase-Eval, a benchmark containing 500 Codeforces competitive programming problems and 100,000 human code submissions. Through two tasks, Fault Coverage and Fault Exposure, this benchmark systematically evaluates the ability of 19 LLMs in test case generation for algorithmic problems. The findings reveal that the strongest model, Qwen3-32B, achieves an exposure rate of only 43.8%, which is far below the 93.3% of human experts.

## Background & Motivation

**Background**: LLMs have made significant progress in code generation, but the generation of high-quality test cases—a critical aspect of software quality assurance—has not yet been systematically evaluated. Existing benchmarks such as TestEval, which are based on LeetCode and use traditional line/branch coverage for evaluation, have seen top models approach nearly 100%, indicating a lack of discriminative power.

**Limitations of Prior Work**: (1) Existing code evaluation benchmarks primarily focus on code generation capabilities, neglecting test case generation as an independent ability; (2) LeetCode problems lack sufficient difficulty, allowing a 6.7B parameter model to achieve 90%+ coverage; (3) Traditional line/branch coverage metrics are not precise enough in algorithmic competition scenarios, failing to distinguish between errors at different levels.

**Key Challenge**: While LLMs continue to advance in code generation, there is a lack of sufficiently challenging and fine-grained benchmarks to evaluate whether they truly understand program logic and can design test cases that expose specific faults.

**Goal**: To build a systematic and challenging benchmark to evaluate LLMs' capabilities in testcase generation for algorithmic problems, covering both the broad coverage of multiple faults and the precise exposure of specific faults.

**Key Insight**: Leveraging the Codeforces competitive programming platform, evaluation data is constructed using real human incorrect submissions (rather than synthetic erroneous code). Two complementary tasks are designed: Fault Coverage to measure the breadth of test coverage, and Fault Exposure to evaluate the ability to precisely expose specific vulnerabilities.

**Core Idea**: To construct a benchmark using real incorrect code from Codeforces and evaluate LLM test case generation capabilities through the dual dimensions of coverage and exposure rates.

## Method

### Overall Architecture

TestCase-Eval includes two core evaluation tasks: (1) **Fault Coverage**: Given a problem description, the LLM generates N test cases to cover as many types of incorrect submissions as possible, using the evaluation metric $\text{Cov}@N = \frac{|\bigcup_{i=1}^{N} \mathcal{F}(t_i)|}{|\mathcal{F}_{\text{total}}|}$; (2) **Fault Exposure**: Given a problem description and a specific incorrect code submission, the LLM generates a single test case to precisely expose the fault in that code, inspired by the "hack" phase in Codeforces.

### Key Designs

1. **Real Erroneous Code Data Collection**
    - **Function**: Provides high-quality and diverse erroneous code samples.
    - **Mechanism**: Collects 500 problems from 2024 Codeforces contests, with 200 incorrect submissions per problem (totaling 100,000), spanning C++, Python, and Java. Problems requiring a special judge are excluded to ensure deterministic evaluation.
    - **Design Motivation**: Utilizing real human errors instead of synthetic errors ensures that fault patterns reflect the actual distribution of real-world bugs in programming.

2. **Fault Difficulty Stratification (Easy/Medium/Hard)**
    - **Function**: Evaluates the differences in LLM performance across different levels of fault difficulty.
    - **Mechanism**: Stratifies faults based on the index of the first failed test case of the incorrect code (provided by the Codeforces platform)—early failures are classified as Easy, and late failures are classified as Hard, as the latter typically involve more subtle logical errors.
    - **Design Motivation**: To provide a more fine-grained analysis, helping to understand the performance differences of LLMs between handling simple boundary conditions and complex logical errors.

3. **Fault Type Analysis Framework**
    - **Function**: Decomposes and analyzes errors into four types: WA, RE, TLE, and MLE.
    - **Mechanism**: Utilizes fault type labels provided by the Codeforces platform (Wrong Answer, Runtime Error, Time Limit Exceeded, Memory Limit Exceeded) to analyze LLMs' detection capabilities for different error mechanisms.
    - **Design Motivation**: Program logic constraints (WA) and resource efficiency issues (TLE/MLE) require distinct testing strategies. Categorized analysis can reveal the performance boundaries of LLMs.

### Loss & Training

This work is an evaluation benchmark and does not involve training. Evaluation is conducted via the ExecEval sandbox environment for code execution and test input evaluation. Both Direct Output and Chain-of-Thought prompting strategies are evaluated.

## Key Experimental Results

### Main Results

Performance of 19 LLMs on TestCase-Eval (CoT prompting):

| Model | Cov@1 | Cov@5 | Cov@20 | Fault Exposure (Overall) |
|------|-------|-------|--------|-------------------------|
| Human Expert | 56.2 | 85.7 | 97.2 | **93.3** |
| Qwen3-32B | **50.8** | **82.3** | **95.7** | **43.8** |
| Qwen3-8B | 46.2 | 78.5 | 92.1 | 41.3 |
| R1-Distill-Qwen-32B | 31.9 | 65.3 | 82.6 | 41.6 |
| GPT-4.1 | 45.3 | 67.5 | 80.0 | 36.5 |
| Llama-3.1-70B | 47.8 | 75.4 | 90.9 | 34.3 |
| Qwen2.5-72B | 38.2 | 57.8 | 73.1 | 29.0 |

### Ablation Study

Breakdown of Fault Exposure rates by error types (Top models):

| Model | WA | RE | TLE | MLE | Overall |
|------|-----|-----|------|------|---------|
| Qwen3-32B | 52.2 | 38.7 | 21.2 | 22.3 | 43.8 |
| R1-Distill-Qwen-32B | 48.0 | 37.8 | 23.9 | 30.3 | 41.6 |
| GPT-4.1 | 42.0 | 35.4 | 20.9 | 25.1 | 36.5 |
| Qwen3-8B | 48.0 | 39.0 | 22.8 | 26.9 | 41.3 |
| Gemma-3-12B | 35.8 | 35.1 | 27.7 | 30.9 | 33.8 |

### Key Findings

1. **Huge Human-Machine Gap**: The strongest LLM (Qwen3-32B) achieves a Fault Exposure rate of only 43.8%, whereas human experts achieve 93.3%, pointing to more than a two-fold performance gap.
2. **Pronounced Advantage of Reasoning Models**: Reasoning-oriented models such as the Qwen3 series and R1-Distill lead on both tasks, indicating that testcase generation heavily relies on logical reasoning capabilities.
3. **Open-Source Models Outperform Closed-Source Ones**: Qwen3-32B outperforms GPT-4.1 in terms of coverage (Cov@20: 95.7 vs 80.0).
4. **Logical Faults vs. Resource Faults**: All models perform significantly better in detecting WA and RE than TLE and MLE, indicating that LLMs are more adept at logical reasoning than efficiency analysis.
5. **CoT is Highly Effective**: Chain-of-Thought prompting significantly outperforms direct output prompting on both tasks.
6. **Python Code is Easier to Exploit**: Due to dynamic typing and flexible syntax, the fault exposure rate for Python code is higher than those for C++ and Java.

## Highlights & Insights

- Identifies "test case generation" as an overlooked yet crucial capability of LLMs, which is not completely positively correlated with code generation performance.
- Uses real incorrect code from Codeforces instead of synthetic data, ensuring the ecological validity of the benchmark.
- The Fault Exposure task is highly challenging—it requires simultaneously understanding the problem description and analyzing incorrect code, serving as a rigorous test for program semantics comprehension.
- The advantage of reasoning-oriented models primarily stems from stronger WA detection capabilities, implying that logical reasoning is the core driving factor.

## Limitations & Future Work

- Evaluates only quantitative metrics without analyzing the specific failure modes of LLMs when generating test cases.
- The difficulty stratification is a heuristic method based on testcase indices, rather than an explicit classification of fault types.
- The detection capability for resource performance bottlenecks (TLE/MLE) has not been systematically evaluated.
- The framework can be extended to more comprehensive debugging tasks, such as bug localization and root cause analysis.

## Related Work & Insights

- TestEval: A test generation benchmark based on LeetCode, which lacks sufficient discriminative power.
- LiveCodeBench: A contamination-free code evaluation benchmark, whose approach to data timeframe selection was referenced in this work.
- The evaluation design as it naturally aligns with Codeforces' "hack" mechanism is highly elegant.

## Rating

- **Novelty**: 4/5 — Fills the gap in evaluating test case generation.
- **Technical Depth**: 3/5 — Primarily focused on benchmark construction and experimental analysis.
- **Experimental Thoroughness**: 5/5 — Covers 19 models, multi-dimensional analysis, and a human baseline.
- **Utility**: 4/5 — Serves as a vital reference for understanding and improving LLMs' code capabilities.
- **Overall Rating**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers](can_llms_identify_critical_limitations_within_scientific_research_a_systematic_e.md)
- [\[ACL 2025\] SCULPT: Systematic Tuning of Long Prompts](sculpt_systematic_tuning_of_long_prompts.md)
- [\[CVPR 2025\] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](../../CVPR2025/llm_nlp/exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)
- [\[ACL 2025\] A Systematic Study of Compositional Syntactic Transformer Language Models](a_systematic_study_of_compositional_syntactic_transformer_language_models.md)
- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)

</div>

<!-- RELATED:END -->
