---
title: >-
  [Paper Note] Revisit Self-Debugging with Self-Generated Tests for Code Generation
description: >-
  [ACL 2025][Code Intelligence][code generation] This paper systematically investigates the effectiveness of self-debugging with self-generated tests using LLMs. It finds that post-execution-based self-debugging degrades performance on basic programming problems due to self-generated test bias. Conversely, in-execution self-debugging successfully avoids this bias, achieving consistent improvements on both basic and competitive programming tasks.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "code generation"
  - "self-debugging"
  - "self-generated tests"
  - "execution feedback"
  - "LLM"
date: 2026-05-08
content_hash: 1211db01e0c8b5c8
---

# Revisit Self-Debugging with Self-Generated Tests for Code Generation

**Conference**: ACL 2025  
**Code**: -  
**Area**: Code Intelligence  
**Keywords**: code generation, self-debugging, self-generated tests, execution feedback, LLM  

## TL;DR

This paper systematically investigates the effectiveness of self-debugging with self-generated tests using LLMs. It finds that post-execution-based self-debugging degrades performance on basic programming problems due to self-generated test bias. Conversely, in-execution self-debugging successfully avoids this bias, achieving consistent improvements on both basic and competitive programming tasks.

## Background & Motivation

- **Self-debugging** has become a popular method to improve LLM code generation quality in recent years: generate code $\rightarrow$ execute tests $\rightarrow$ obtain feedback $\rightarrow$ repair code.
- However, existing methods (e.g., Self-Debugging, Reflexion, AlphaCodium) largely depend on **predefined oracle tests**, whereas high-quality test cases are often unavailable in practice.
- **Dilemma of Self-Generated Tests**: Utilizing model-generated tests for self-debugging is a natural solution, but its effectiveness remains under-explored.
    - Though Reflexion utilizes self-generated tests for feedback, it evaluates pre-repair code using oracle tests.
    - AlphaCodium iterates on oracle tests before iterating on self-generated tests.
- **Core Problem**: The quality of self-generated tests is limited (test output accuracy is only ~85%). What impact does this have on self-debugging?

## Method

### Overall Architecture

A unified framework for self-debugging is proposed, distinguishing between two paradigms:

1. **Post-Execution Self-Debugging**
    - Compares actual execution outputs with expected outputs after running the code.
    - If a mismatch occurs, the failed test cases, actual outputs, and error messages are provided as feedback for the model to fix the program.
    - Two feedback granularities: label (only correct/incorrect status) and detail (includes test inputs, expected outputs, and actual outputs).

2. **In-Execution Self-Debugging**
    - Decomposes the program into basic blocks based on the Control Flow Graph (CFG).
    - Collects intermediate variable states (execution trace) before and after the execution of each basic block.
    - The model assesses program correctness and debugs purely based on the test inputs and intermediate states.
    - **Does not utilize post-execution information** (unaware of whether the final output matches the expected output).

### Key Differences

- Post-execution relies on the **output labels** of self-generated tests, which can be erroneous.
- In-execution focuses solely on the **intermediate states of execution**, thereby bypassing the label bias issue.

### Bias Analysis Framework

Four cases are defined:
- **True Positive (TP)**: Correct program passes the test.
- **True Negative (TN)**: Incorrect program fails the test.
- **False Positive (FP)**: Incorrect program passes a buggy test.
- **False Negative (FN)**: Correct program fails due to an erroneous test.

## Experiments

### Experimental Setup

- **Models**: GPT-4o, Claude-3.5-Sonnet, LLaMA-3-70B-Instruct, Qwen2.5-Coder-7B-Instruct
- **Benchmarks**: HumanEval(+), MBPP(+) (basic), LiveCodeBench (competition-level, 450 problems)
- Greedy decoding, with 10 test cases generated per problem.
- Iterations: 1-2 rounds

### Post-Execution Self-Debugging + Oracle Tests (Control Baseline)

Self-debugging brings consistent improvements when using oracle tests, for example:
- GPT-4o on HumanEval: 92.1 $\rightarrow$ 95.1 (+3.0)
- Claude-3.5 on MBPP+: 77.0 $\rightarrow$ 86.0 (+9.0)

### Post-Execution Self-Debugging + Self-Generated Tests

**Performs poorly on basic problems**:
- Claude-3.5 on HumanEval: 94.5 $\rightarrow$ 87.2 (**-7.3**)
- LLaMA-3-70B on HumanEval: 79.9 $\rightarrow$ 73.8 (**-6.1**)
- All models show performance drops on HumanEval.

**Shows potential on competitive programming tasks**:
- GPT-4o on LiveCodeBench: 46.0 $\rightarrow$ 49.3 (+3.3) (with label feedback)
- However, detail feedback leads to performance drops on easy problems.

### Self-Generated Test Quality Analysis

| Model | Test Input Accuracy | Test Output Accuracy | Test Suite Effectiveness |
|------|--------------|--------------|--------------|
| GPT-4o | 97.63% | 89.77% | 59.15% |
| Claude-3.5 | 97.68% | 89.14% | 56.71% |
| LLaMA-3-70B | 94.53% | 84.69% | 49.39% |
| Qwen2.5-Coder-7B | 97.19% | 84.85% | 44.50% |

- Generating test inputs is relatively easy, while generating correct outputs is difficult (~85%).
- The effectiveness of the complete test suite is only ~50-60%.

### Key Findings from Bias Analysis

- On HumanEval/MBPP, **False Negatives outnumber True Negatives**, meaning correct programs are labeled as failed by erroneous tests, leading to unnecessary modifications that introduce bugs instead.
- On LiveCodeBench, **True Negatives have a higher proportion** because competitive programming tasks have lower base pass rates, making self-generated test labels more likely to be correct.

### In-Execution Self-Debugging

**Demonstrates positive performance on basic problems**:
- GPT-4o on HumanEval/MBPP+: 87.8/76.5 $\rightarrow$ 89.0/79.1 (2 iteration rounds)
- Qwen2.5-Coder-7B on MBPP+: 70.6 $\rightarrow$ 72.0
- Most models maintain or improve performance.

**On competitive programming tasks**:
- GPT-4o on LiveCodeBench: 46.0 $\rightarrow$ 47.6 (+1.6)

### Synthesis of Comparison

| Paradigm | Basic Problems | Competitive Problems |
|------|--------|--------|
| Post-execution + self-tests | ❌ Decreases in general | ⚠️ Label improves performance |
| In-execution + self-tests | ✅ Mild gain | ✅ Consistent gain |

## Highlights & Insights

1. **First to systematically reveal the failure mode of self-debugging with self-generated tests**: Post-execution self-debugging is counterproductive on basic problems, which is an important and counterintuitive finding.
2. **The bias analysis framework (TP/TN/FP/FN)** precisely explains the root cause of the inconsistency: high FN rates on basic problems versus more reasonable TN rates on competitive problems.
3. **The In-execution paradigm serves as a practical solution**: By analyzing intermediate execution states rather than relying on unreliable test labels, it effectively circumvents test bias.
4. **Insight**: The effectiveness of self-debugging depends on not only the code-repair capability but also the **ability to recognize erroneous feedback**.
5. **Comprehensive experimental coverage**: 4 models $\times$ 3 benchmarks $\times$ 2 paradigms $\times$ 2 feedback granularities.

## Limitations & Future Work

- Restricted to Python programming tasks; multilingual scenarios are not validated.
- In-execution self-debugging requires full execution traces. For complex programs (deep loops/recursions), this may yield excessively long traces, and truncation or compression strategies are not discussed.
- Self-generated tests are limited to 10 instances; whether more tests can alleviate bias remains under-explored.
- Hybrid approaches combining oracle tests and self-generated tests are not explored.
- Evaluated strictly using greedy decoding, without considering candidate sampling strategies.

## Related Work & Insights

- **Code Generation**: CodeT (Chen et al., 2023) utilizes dual execution consistency; Self-Edit (Zhang et al., 2023a) incorporates sample tests as execution feedback.
- **Self-debugging**: Self-Debugging (Chen et al., 2024b) focuses on iterative debugging; LDB (Zhong et al., 2024) leverages runtime trace information; Reflexion (Shinn et al., 2023) uses self-generated test feedback but evaluates with oracle tests.
- **Code Evaluation**: EvalPlus (Liu et al., 2023) scales up evaluation test suites; LiveCodeBench (Jain et al., 2024) continuously collects competitive coding questions.

## Rating ⭐⭐⭐⭐

Clear research problem, deep analysis (bias framework), valuable findings (discovering post-execution self-debugging is harmful on basic questions), and proposes a viable alternative (in-execution self-debugging). However, the absolute gains of in-execution self-debugging are somewhat limited, requiring further verification of practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](codedpo_code_alignment.md)
- [\[ACL 2025\] STaR-SQL: Self-Taught Reasoner for Text-to-SQL](star-sql_self-taught_reasoner_for_text-to-sql.md)
- [\[NeurIPS 2025\] A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [\[ACL 2025\] MLDebugging: Towards Benchmarking Code Debugging Across Multi-Library Scenarios](mldebugging_towards_benchmarking_code_debugging_across_multi-library_scenarios.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)

</div>

<!-- RELATED:END -->
