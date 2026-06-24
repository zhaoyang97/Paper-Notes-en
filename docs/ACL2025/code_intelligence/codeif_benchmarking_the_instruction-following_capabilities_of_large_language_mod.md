---
title: >-
  [Paper Note] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation
description: >-
  [ACL 2025][Code Intelligence][Code Generation] CodeIF is proposed as the first systematic benchmark to evaluate the instruction-following capabilities of LLMs in code generation. It includes 50 fine-grained constraint instructions across 8 major categories, introduces 4 new evaluation metrics, and comprehensively evaluates 35 SOTA models.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Instruction Following"
  - "Evaluation Benchmark"
  - "Constraint Satisfaction"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: 7e3a216a15e47c06
---

# CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation

**Conference**: ACL 2025  
**arXiv**: [2502.19166](https://arxiv.org/abs/2502.19166)  
**Code**: [https://github.com/lin-rany/codeIF](https://github.com/lin-rany/codeIF)  
**Area**: Code Intelligence  
**Keywords**: Code Generation, Instruction Following, Evaluation Benchmark, Constraint Satisfaction, LLM Evaluation

## TL;DR

CodeIF is proposed as the first systematic benchmark to evaluate the instruction-following capabilities of LLMs in code generation. It includes 50 fine-grained constraint instructions across 8 major categories, introduces 4 new evaluation metrics, and comprehensively evaluates 35 SOTA models.

## Background & Motivation

**Background**: LLMs have made significant progress in code generation, but their ability to understand and execute complex instructions remains a key challenge. Existing evaluation frameworks (such as HumanEval and MBPP) mainly focus on functional correctness, lacking a systematic evaluation of instruction following.

**Limitations of Prior Work**: (1) Existing code benchmarks do not assess whether models adhere to constraints such as global formatting, naming conventions, and structural control; (2) There is a lack of evaluation metrics for multi-constraint problems, making it impossible to distinguish model performance across different constraint types; (3) Instruction-following evaluation benchmarks (such as FollowBench and InfoBench), although rich, are not tailored for code generation scenarios.

**Key Challenge**: Code generation in practice requires simultaneously satisfying functional requirements and coding style constraints. However, existing evaluations only focus on whether the code runs successfully, ignoring whether it is well-written and complies with constraints.

**Goal**: To build a multilingual, multi-difficulty, and multi-constraint code generation instruction-following benchmark, and to propose quantifiable metrics for assessing instruction-following capabilities.

**Key Insight**: Starting from coding constraints, decompose constraints into indivisible atomic instructions and model the dependencies between them to achieve binary objective evaluation.

**Core Idea**: Decompose code instruction following into 50 atomic constraints across 8 categories, paired with 4 hierarchical evaluation metrics to systematically characterize LLMs' capabilities in constraint satisfaction, logical consistency, and dependency handling.

## Method

### Overall Architecture

The construction of CodeIF consists of three steps: (1) Designing a constraint instruction taxonomy (50 fine-grained sub-instructions under 8 major categories); (2) Generating constraint instruction lists based on benchmarks like McEval and FullStackBench using GPT-4, covering four programming languages: Java, Python, Go, and C++; (3) Utilizing LLMs to model the dependency relationships among atomic constraints.

### Key Designs

1. **Constraint Instruction Taxonomy**: Eight categories cover different abstraction levels — Global, Structural Control, Variable, Interface, Function, Class, File, and Combination. Each atomic instruction is designed as a binary evaluation (Yes/No) to avoid subjective judgment.

2. **Multilingual and Multi-Difficulty Design**: The dataset spans Java (353), Python (348), C++ (269), Go (230), and is divided into two difficulty levels, Easy (averaging 11.99 instructions/task) and Hard (averaging 13.80 instructions/task), totaling 1,200 tasks.

3. **Instruction Dependency Modeling**: LLMs are used to automatically label prerequisite dependencies among atomic constraints. For example, "creating a function body" depends on "naming the function" and "defining parameter types". This dependency structure supports the more rigorous evaluation metric, RSR.

4. **Automated Generation Pipeline**: Using GPT-4 based on 20 detailed examples, task-level constraint instruction lists are generated automatically, with manual reviews conducted to guarantee quality.

### Evaluation Metrics

This work proposes four complementary evaluation metrics. For a dataset containing $m$ problems with $n_i$ constraints for each problem:

1. **CSR (Completely Satisfaction Rate)**: $\text{CSR} = \frac{1}{m}\sum_{i=1}^{m}\prod_{j=1}^{n_i} r_{i,j}$, where a score is given only if all constraints are fully satisfied. This is the most stringent metric.

2. **SSR (Soft Satisfaction Rate)**: $\text{SSR} = \frac{1}{m}\sum_{i=1}^{m}\frac{\sum_{j=1}^{n_i} r_{i,j}}{n_i}$, which calculates the average constraint satisfaction ratio per problem. This is a softer metric.

3. **RSR (Rigorous Satisfaction Rate)**: Building on SSR, this metric introduces dependencies where constraint $j$ is satisfied only if all its prerequisite dependencies $D_{i,j}$ are also satisfied: $\text{RSR} = \frac{1}{m}\sum_{i=1}^{m}\frac{\sum_{j=1}^{n_i}[r_{i,j}\cdot\prod_{k\in D_{i,j}}r_{i,k}]}{n_i}$.

4. **CCSR (Consistent Continuity Satisfaction Rate)**: $\text{CCSR} = \frac{1}{m}\sum_{i=1}^{m}\frac{L_i}{n_i}$, where $L_i$ represents the length of the longest consecutively satisfied sequence of constraints. This measures sustained following capabilities.

## Key Experimental Results

### Main Results

| Model | CSR(Full) | SSR(Full) | RSR(Full) | CCSR(Full) |
|------|-----------|-----------|-----------|------------|
| DeepSeek-V3 | **0.414** | **0.821** | **0.764** | **0.712** |
| Claude-3-5-Sonnet | 0.444 | 0.727 | 0.692 | 0.652 |
| GPT-4o | 0.383 | 0.748 | 0.689 | 0.650 |
| Gemini-Exp-1206 | 0.357 | 0.744 | 0.685 | 0.636 |
| Qwen2.5-Coder-32B | 0.365 | 0.736 | 0.679 | 0.634 |
| Llama-3.3-70B | 0.307 | 0.698 | 0.632 | 0.589 |
| Llama-3.2-1B | 0.034 | 0.218 | 0.182 | 0.152 |

### Ablation Study

| Difficulty | GPT-4o CSR | DeepSeek-V3 CSR | Claude CSR |
|------|-----------|----------------|-----------|
| Easy | 0.441 | 0.468 | 0.525 |
| Hard | 0.325 | 0.359 | 0.362 |

The best CSR on Hard tasks is only 0.362 (Claude), indicating that strict constraint satisfaction is highly challenging.

### Key Findings

- **DeepSeek-V3 is the overall strongest**: It leads across SSR, RSR, and CCSR, particularly reaching an SSR of 0.831 on compositional constraint tasks.
- **Claude-3-5-Sonnet achieves the highest CSR**: It reaches a CSR of 0.504 on Java, but its SSR on Python is only 0.703.
- **Model size is positively correlated but not absolutely**: The Qwen2.5 family displays clear scaling effects, whereas the Llama3 family shows inconsistency.
- **Closed-source > Open-source**: GPT-4o and Claude perform significantly better than open-source models of similar scale on complex constraints.
- **C++ is the most difficult language**: Complex template metaprogramming leads to the poorest performance across all models.
- **Common deviation types**: Naming conventions (e.g., camelCase requested but snake_case outputted), and ignoring negative constraints (e.g., using "if" statements when instructed not to).

## Highlights & Insights

- The first dedicated benchmark targeting instruction following in code generation, filling an important gap.
- The 4-metric system is elegantly hierarchical: CSR evaluates full satisfaction, SSR evaluates average, RSR evaluates dependency chains, and CCSR evaluates continuity.
- The binary design of atomic instructions avoids subjective evaluation; the Pearson correlation between GPT-4 automatic evaluation and human annotation reaches 0.87.
- Designing instruction dependencies is a unique aspect, with the RSR metric successfully capturing cascading failures.

## Limitations & Future Work

1. **Inadequate language coverage**: It only includes Java, Python, Go, and C++, lacking other popular languages like JavaScript, Rust, and Swift.
2. **Mainly static evaluation**: Focuses primarily on code structure and naming conventions, neglecting runtime behavior, performance, and debugging capabilities.
3. **Uniform metric weighting**: All constraints are weighted equally, whereas in practice, syntactic correctness is far more important than naming conventions.
4. **Automated evaluation is dependent on GPT-4**: Binary judgments rely on GPT-4-1106-Preview, which incurs high costs and may introduce biases.

## Related Work & Insights

- **Code Benchmarks**: HumanEval and MBPP focus on functional correctness; EvalPlus enhances test cases, and BigCodeBench extends function calls. None of these involve instruction following.
- **Instruction-Following Benchmarks**: InfoBench, FollowBench, and CFBench evaluate instruction following in text generation. This work extends this to the programming domain.
- **Insights**: Constraint-following capabilities can be evaluated jointly with code functional correctness. The constraint dependency graph can be used to diagnose points where the LLM's reasoning chain breaks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first code instruction-following benchmark, precisely positioned.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 35 models, 4 languages, 2 difficulty levels, multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and rich tables.
- **Value**: ⭐⭐⭐⭐ — Provides a new dimension for code generation evaluation, guiding future model improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](tree_of_evolution_code_gen.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)
- [\[ACL 2025\] LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](benchmarking_long-context_language_models_on_long_code_understanding.md)

</div>

<!-- RELATED:END -->
