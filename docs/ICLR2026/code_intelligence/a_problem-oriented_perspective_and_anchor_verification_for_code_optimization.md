---
title: >-
  [Paper Note] A Problem-Oriented Perspective and Anchor Verification for Code Optimization
description: >-
  [ICLR 2026][Code Intelligence][Code Optimization] This paper proposes a problem-oriented (rather than user-oriented) approach to constructing optimization pairs that integrates the strategic diversity of multiple program…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "Code Optimization"
  - "LLM"
  - "Problem-Oriented"
  - "Anchor Verification"
  - "Program Performance"
date: 2026-05-08
content_hash: a2f8bc89bbcbecf1
---

# A Problem-Oriented Perspective and Anchor Verification for Code Optimization

**Conference**: ICLR 2026
**arXiv**: [2406.11935](https://arxiv.org/abs/2406.11935)
**Code**: None
**Area**: Code Intelligence
**Keywords**: Code Optimization, LLM, Problem-Oriented, Anchor Verification, Program Performance

## TL;DR
This paper proposes a problem-oriented (rather than user-oriented) approach to constructing optimization pairs that integrates the strategic diversity of multiple programmers, and designs an anchor verification framework that leverages "slow but correct code" to generate test cases for mitigating the "optimization tax" (correctness loss), improving the optimization rate from 31.24% to 71.06% and the speedup ratio from 2.95x to 6.08x.

## Background & Motivation

**Background**: LLMs have demonstrated strong performance on code generation tasks, yet their potential for code performance optimization (minimizing execution time) remains insufficiently explored.

**Limitations of Prior Work**: Existing methods (PIE) construct optimization pairs from iterative submissions by the same user, constraining LLMs to local incremental improvements and neglecting global algorithmic innovation. Furthermore, code optimization suffers from an "optimization tax"—LLM-optimized code frequently exhibits correctness issues.

**Key Challenge**: Code optimization is inherently a dual-objective problem (efficiency + correctness), and the two objectives often conflict. User-oriented optimization pairs reflect only the cognitive patterns of individual programmers and lack strategic diversity.

**Goal**: (a) How to construct richer and more diverse optimization pairs to elicit global optimization capabilities in LLMs; (b) How to mitigate correctness degradation in code optimization.

**Key Insight**: Shifting from a user-oriented to a problem-oriented perspective—sorting all user submissions for the same problem by runtime to form new optimization trajectories; leveraging the "slow but correct" property of pre-optimization code to generate test cases for verification.

**Core Idea**: Problem-oriented optimization pairs drawn from a multi-user perspective stimulate global algorithmic innovation, while the anchor verification framework uses the original slow code as a correctness anchor to repair the optimization tax.

## Method

### Overall Architecture
The method consists of two components: (1) Problem-oriented Construction of optimization pairs (PCO)—breaking user boundaries by sorting all submissions to the same programming problem by runtime; (2) Anchor Verification Framework—using an LLM to generate test inputs, executing the original slow code to obtain exact outputs, and forming complete test cases for iterative repair of optimized code.

### Key Designs

1. **Problem-oriented Construction of Optimization Pairs (PCO)**:

    - **Function**: Sorts all user submissions to the same problem by runtime and constructs optimization pairs across user boundaries.
    - **Mechanism**: For problem $\mathcal{P}$, all correct submissions from all users are collected and sorted by runtime as $[A_1, C_1, B_1, A_2, ...]$, with optimization pairs constructed along the trajectory. The number of optimization pairs grows from $\sum C_{n_u}^2$ to $C_{\sum n_u}^2$, increasing by an order of magnitude when the number of users reaches 10.
    - **Design Motivation**: User-oriented optimization pairs are constrained by the cognitive patterns of individual programmers, predominantly yielding local improvements (small GED, clustered semantic embeddings); the problem-oriented approach integrates diverse algorithmic strategies and encompasses more global algorithmic transformations (large GED, dispersed semantic embeddings).

2. **Anchor Verification Framework**:

    - **Function**: Uses the original slow code as a correctness anchor to generate test cases for iteratively repairing correctness issues in optimized code.
    - **Mechanism**: (a) An LLM interprets the slow code and generates test inputs; (b) the slow code is actually executed (rather than predicted by the LLM) to obtain exact outputs; (c) input–output pairs are assembled into verification test cases; (d) these test cases are used to iteratively repair errors in the optimized code.
    - **Design Motivation**: Unlike code generation, code optimization has a unique structural advantage—the pre-optimization code, though slow, is guaranteed to be correct and can serve as a test oracle. This eliminates the need to synthesize test cases from scratch, which may themselves be incorrect.

### Loss & Training
Supervised fine-tuning (SFT) is applied to code LLMs (DeepSeek-Coder, Qwen2.5-Coder, etc.) on PCO optimization pairs. During inference, a best@k strategy is employed (sampling $k$ candidates and selecting the fastest correct one). Evaluation is conducted using the gem5 CPU simulator.

## Key Experimental Results

### Main Results

**PIE vs. PCO Fine-tuning Results (best@8):**

| Model | Training Data | %Opt↑ | Speedup↑ | Correct↑ |
|-------|--------------|-------|----------|----------|
| DS-Coder 6.7B | PIE | 31.24% | 2.95x | 61.14% |
| DS-Coder 6.7B | **PCO** | **58.90%** | **5.22x** | 61.55% |
| DS-Coder 6.7B | PCO + Anchor Verification | **71.06%** | **6.08x** | **74.54%** |

### Ablation Study

| Configuration | %Opt | Speedup | Correct |
|--------------|------|---------|---------|
| PIE (User-Oriented) | 31.24% | 2.95x | 61.14% |
| PCO (Problem-Oriented) | 58.90% | 5.22x | 61.55% |
| + Anchor Verification | 71.06% | 6.08x | 74.54% |

### Key Findings
- The problem-oriented perspective nearly doubles the optimization rate (+27.66 pp) and substantially improves the speedup ratio (2.95x→5.22x), highlighting the importance of cross-user strategy integration.
- Manual analysis reveals that global algorithmic optimizations predominate in PCO pairs, whereas local optimizations are more common in PIE pairs.
- The anchor verification framework further improves correctness (+12.99 pp) while also increasing the optimization rate (+12.16 pp), demonstrating that correctness and efficiency are not entirely in conflict.
- The problem-oriented approach not only improves performance but also substantially alleviates data scarcity by increasing the number of optimization pairs by an order of magnitude.

## Highlights & Insights
- **The Power of Perspective Shift**: The transition from "user iteration" to "problem aggregation" appears straightforward yet yields substantial gains—the key insight is that code optimization requires algorithm-level innovation (e.g., from $O(n^2)$ to $O(n \log n)$), which is unlikely to emerge from a single programmer's iterative refinement.
- **Slow Code as Oracle**: The structural distinction between code optimization and code generation—the pre-optimization code serves as a natural test oracle—is a clever insight unique to the code optimization setting.
- **Data Scale Effect**: The combinatorial explosion inherent to the problem-oriented approach ($C_{\sum n_u}^2 \gg \sum C_{n_u}^2$) naturally resolves the data scarcity problem.

## Limitations & Future Work
- The work focuses exclusively on execution time optimization for C++ code; generalization to other languages and optimization dimensions (memory, energy consumption) remains unexplored.
- The anchor verification framework relies on the LLM to generate valid test inputs—if the generated inputs lack sufficient diversity, not all bugs may be detected.
- Evaluation is conducted using the gem5 simulator rather than real hardware, so runtime measurements may diverge from actual deployment performance.
- A gap exists between competitive programming-style code optimization and optimization in real-world engineering projects.

## Related Work & Insights
- **vs. PIE (Shypula et al.)**: PIE pioneered the concept of code optimization pairs but adopted only a user-oriented perspective; PCO comprehensively surpasses it through the problem-oriented perspective and anchor verification.
- **vs. Code Generation Verification**: Test case synthesis methods for code generation (CodeT) require bidirectional execution filtering; anchor verification is more reliable by using the original code as an oracle.
- **vs. Compiler Optimization**: Compilers perform hardware-level optimization (-O3), while this work targets algorithm-level optimization; the two approaches are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the perspective shift and anchor verification are innovative, though the core ideas are relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional analysis (structural/semantic/manual), multi-model comparisons, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with well-articulated motivation.
- Value: ⭐⭐⭐⭐ Provides effective data construction and verification methodology for LLM-based code optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models](dro-instructzero_distributionally_robust_prompt_optimization_for_large_language_.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](../../ACL2026/code_intelligence/qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](../../ACL2026/code_intelligence/socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)
- [\[NeurIPS 2025\] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](../../NeurIPS2025/code_intelligence/preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)

</div>

<!-- RELATED:END -->
