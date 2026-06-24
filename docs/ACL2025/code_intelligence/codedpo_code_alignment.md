---
title: >-
  [Paper Note] CodeDPO: Aligning Code Models with Self Generated and Verified Source Code
description: >-
  [ACL 2025][Code Intelligence][code generation] Proposes CodeDPO, which constructs high-quality preference pairs (93K correctness + 21K efficiency) from self-generated code via a PageRank-inspired self-validation scoring mechanism. After DPO training, it achieves an average improvement of over 10 points on HumanEval across 8 code models, while accelerating code execution efficiency by 1.25-1.45×.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "code generation"
  - "DPO"
  - "self-validation"
  - "PageRank"
  - "code efficiency"
date: 2026-05-08
content_hash: c7ed7ce95ec23f60
---

# CodeDPO: Aligning Code Models with Self Generated and Verified Source Code

**Conference**: ACL 2025  
**arXiv**: [2410.05605](https://arxiv.org/abs/2410.05605)  
**Code**: [https://anonymous.4open.science/r/CodeDPO/](https://anonymous.4open.science/r/CodeDPO/)  
**Area**: LLM Alignment / Code Generation  
**Keywords**: code generation, DPO, self-validation, PageRank, code efficiency

## TL;DR
Proposes CodeDPO, which constructs high-quality preference pairs (93K correctness + 21K efficiency) from self-generated code via a PageRank-inspired self-validation scoring mechanism. After DPO training, it achieves an average improvement of over 10 points on HumanEval across 8 code models, while accelerating code execution efficiency by 1.25-1.45×.

## Background & Motivation
**Background**: Code generation models are primarily enhanced through SFT. However, SFT increases the probability of generating incorrect code while raising the probability of correct code, leading to performance saturation. DPO is effective in reasoning tasks but remains under-explored in code generation.

**Limitations of Prior Work**: Code DPO requires positive/negative preference pairs, but constructing reliable pairs faces two challenges: (1) the quality of self-generated test cases is mixed—low-quality tests misjudge code correctness; (2) optimizing correctness alone is insufficient, as code efficiency is equally critical.

**Key Challenge**: Simply using "pass test = positive / fail test = negative" to construct preference pairs is prone to noise from low-quality test cases. A mechanism is needed to simultaneously evaluate the reliability of both test cases and code.

**Goal**: How to construct high-quality preference pairs from self-generated code and tests, while optimizing both correctness and efficiency?

**Key Insight**: Leverage the mutual reference idea of PageRank—code passed by more tests is more trustworthy, and tests passed by more code are more reliable.

**Core Idea**: Replace simple pass/fail judgments with PageRank-style iterative self-validation scoring to construct high-quality preference pairs for DPO.

## Method

### Overall Architecture
Four steps: (1) Extract programming concepts from open-source code repositories to generate diverse problems; (2) Generate 15 code solutions + 15 test suites for each problem, rank them using PageRank-style mutual validation scoring, and select extreme pairs as correctness preference pairs; (3) Measure execution time on trusted test sets to pair fast/slow versions as efficiency preference pairs; (4) Train using the RPO format (weighted SFT + DPO).

### Key Designs

1. **PageRank-style Self-Validation Score (Self-Validation Score)**:

    - **Function**: For each problem's 15 code solutions and 15 test suites, a bipartite graph is constructed where edges between code nodes and test nodes represent "pass" relationships.
    - **Mechanism**: Iterative update (T=10 rounds, damping factor d=0.85): $Score_t(c_i) = (1-d) \cdot Score_{t-1}(c_i) + d \cdot \sum_j Score_{t-1}(t_j) \cdot Link(t_j, c_i)$
    - **Design Motivation**: Tests passed by multiple high-scoring code solutions are more trustworthy $\to$ code passing more high-scoring tests is more correct, forming a virtuous cycle. This achieves a higher correlation (Spearman r=0.86) than simply counting "how many tests were passed" (r=0.77).

2. **Dual-Dimension Preference Pair Construction**:

    - **Correctness Preference** (93K pairs): Pairs are created by ranking solutions according to self-validation scores and selecting the highest- and lowest-scoring code.
    - **Efficiency Preference** (21K pairs): Within high-scoring code solutions (where correctness is guaranteed), fast and slow versions are paired based on execution time.
    - **Design Motivation**: Correctness and efficiency are two orthogonal dimensions of code quality; constructing separate preference pairs allows for their individualized optimization.

3. **RPO Training Strategy**:

    - **Function**: Uses the RPO format loss = weighted SFT loss + original DPO loss.
    - **Mechanism**: SFT loss ensures that the model does not drift from correct generation patterns, while DPO loss widens the gap between positive and negative samples.
    - **Design Motivation**: Pure DPO can lead to unstable model behavior; RPO enhances stability.

### Loss & Training
RPO format loss, 10 epochs, lr=5e-6, linear schedule with warmup, 16 × A100 GPUs, data generated by DeepSeekCoder-v2 at temperature 1.5. The entire dataset construction cost is approximately $80.

## Key Experimental Results

### Main Results (HumanEval / HumanEval+)

| Model | Baseline | + CodeDPO | Gain |
|------|------|----------|------|
| MagiCoder-S-DS-6.7B | 73.17 / 68.29 | **83.54 / 76.22** | +10.37 / +7.93 |
| DeepSeekCoder-6.7B | 47.60 / 39.60 | **59.75 / 51.83** | +12.15 / +12.23 |
| StarCoder2-7B | 35.40 / 29.90 | **48.17 / 34.15** | +12.77 / +4.25 |
| Phi-2-2.7B | 48.78 / 46.34 | **57.32 / 51.83** | +8.54 / +5.49 |
| DeepSeekCoder-1.3B | 31.53 / 28.65 | **42.07 / 38.04** | +10.54 / +9.39 |

### Ablation Study (DeepSeekCoder-1.3B)

| Preference Pair Construction Method | HumanEval | HumanEval+ |
|--------------|-----------|------------|
| Random selection | 21.34 | 18.29 |
| Sorted by number of passed tests | 37.19 | 31.09 |
| Filtered by all tests | 34.75 | 29.89 |
| **CodeDPO (Self-Validation Score)** | **42.07** | **38.04** |

### Key Findings
- The Spearman correlation coefficient between self-validation scores and ground-truth correctness is 0.86, significantly higher than simple counting (0.77) and filtering by all tests (0.61).
- Code efficiency achieves a 1.25-1.45× speedup, with 20-45% of the generated code being at least 10% faster.
- Effective on LiveCodeBench as well (Easy: +12%, Medium: +5.6%), demonstrating generalization.
- DPO is more effective than KTO (42.07 vs 40.85 on HumanEval) because the preference pairs constructed by CodeDPO are naturally balanced.
- Performance scales continuously from 25% to 100% data volume, but the growth tends to plateau.
- Data contamination check: The average similarity with HumanEval is only 0.109, which is lower than Self-Instruct (0.169).

## Highlights & Insights
- **PageRank self-validation** is the most central design: modeling code-test mutual validation as iterative reputation propagation on a graph reflects code quality more accurately than simple pass-rate counting. This idea is transferable to any scenario requiring automated grading.
- **Dual-dimension optimization of correctness and efficiency** is highly practical: real-world programming requires code to not only run, but run fast. Separating preference pair construction for individual optimization is an elegant decoupling strategy.
- **Extremely low data construction cost** (~$80) demonstrates the cost-effectiveness of the self-generation + self-verification paradigm.

## Limitations & Future Work
- It is not compared with RL methods (such as GRPO, Reinforcement++), which may be stronger than offline DPO.
- Test-case-based correctness evaluation inherently suffers from edge-case omissions—passing all tests does not equal 100% correctness.
- The dimensions of code readability and security are not optimized.
- Self-validation scoring relies on having a sufficient exploration space of code and tests; it may fail when problems are too simple or too difficult.

## Related Work & Insights
- **vs AceCoder**: AceCoder focuses on building a dedicated RM + RL, while CodeDPO focuses on offline DPO + self-validation scoring. The two are complementary: the self-validation score of CodeDPO can serve as a preference pair construction method for AceCoder.
- **vs RLTF (Liu et al. 2023)**: RLTF uses compiler/compiler execution feedback for RL but does not perform scoring—only pass/fail. The PageRank score of CodeDPO provides finer-grained quality ranking.
- **vs CodeRL**: CodeRL uses a critic model for reward, requiring additional training. CodeDPO is entirely self-contained and does not require an external RM.

## Rating
- Novelty: ⭐⭐⭐⭐ PageRank self-validation scoring + efficiency optimization dimension are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 models × 4 benchmarks + efficiency evaluation + detailed ablation + data scaling analysis.
- Writing Quality: ⭐⭐⭐⭐ The clear methodology explanation and well-designed ablation studies.
- Value: ⭐⭐⭐⭐ A cost-effective and efficient code alignment method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](galla_graph_aligned_large_language_models.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)

</div>

<!-- RELATED:END -->
