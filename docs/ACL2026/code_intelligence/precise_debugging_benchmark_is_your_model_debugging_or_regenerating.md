---
title: >-
  [Paper Note] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?
description: >-
  [ACL 2026][Code Intelligence][Code Debugging] This paper reveals the "regeneration" tendency of frontier LLMs in debugging tasks. By introducing the PDB framework alongside edit-level precision and bug-level recall metri…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code Debugging"
  - "LLM Programming"
  - "Precise Editing"
  - "Benchmarking"
  - "Code Regeneration"
date: 2026-05-08
content_hash: d2557878f2d32f76
---

# Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?

**Conference**: ACL 2026  
**arXiv**: [2604.17338](https://arxiv.org/abs/2604.17338)  
**Code**: [GitHub](https://github.com)  
**Area**: Code Intelligence / Debugging Evaluation  
**Keywords**: Code Debugging, LLM Programming, Precise Editing, Benchmarking, Code Regeneration

## TL;DR

This paper reveals the "regeneration" tendency of frontier LLMs in debugging tasks. By introducing the PDB framework alongside edit-level precision and bug-level recall metrics, it is discovered that while models such as GPT-5.1-Codex can pass over 76% of unit tests, their edit precision is below 45%. Furthermore, iterative and agent-based debugging strategies fail to significantly improve precision.

## Background & Motivation

**Background**: LLMs have achieved significant success in code generation, synthesizing complex algorithms from natural language descriptions. However, the primary effort in real-world software development is not generation from scratch, but rather debugging and maintenance.

**Limitations of Prior Work**: (1) When provided with buggy code, models often rewrite large portions or even the entire code to "fix" it—while this may pass tests, it is costly, risky, and difficult to review in realistic codebases; (2) Existing debugging benchmarks rely solely on unit test pass rates for evaluation, failing to distinguish between precise fixes and large-scale rewrites—rewriting an entire function receives the same score as fixing a single-line bug; (3) For programs with multiple bugs, models that fix only a portion of the bugs receive the same zero score as models that fix none.

**Key Challenge**: There is a negative correlation between unit test pass rates and debugging precision—the more aggressively a model rewrites code, the more likely it is to pass tests (functional correctness), but the lower its edit precision becomes. Existing evaluation systems reward regeneration behavior and fail to incentivize precise debugging.

**Goal**: (1) Design an evaluation framework capable of distinguishing "precise debugging" from "code regeneration"; (2) Quantify the gap between current frontier models and precise debugging; (3) Evaluate whether iterative and agentic debugging strategies improve precision.

**Key Insight**: Define two new metrics: "edit-level precision" and "bug-level recall." Precision measures the proportion of necessary modifications in the model's output, while recall measures how many bugs are correctly fixed. A debugging benchmark with ground-truth edit scripts is constructed by automatically injecting verified atomic bugs and combining them into multi-bug programs.

**Core Idea**: Shift debugging evaluation from the program level (pass/fail) to the edit level (identifying which modifications are necessary versus redundant). Build a precise evaluation benchmark through atomic bug synthesis and independence verification.

## Method

### Overall Architecture

The PDB framework consists of two stages: THE **Generation Phase** starts from existing programming datasets, using LLMs to synthesize verified atomic bugs and combine them into multi-bug programs; the **Evaluation Phase** requires debugging systems to fix the buggy programs, which are then assessed using edit-level precision and bug-level recall.

### Key Designs

1.  **Atomic Bug Synthesis and Combination**:
    - **Function**: Generate buggy programs with ground-truth edit scripts, supporting both single-line and multi-line bugs.
    - **Mechanism**: For each ground-truth program, based on the five categories of ODC (Orthogonal Defect Classification: Assignment, Checking, Algorithm, Build/Package, Timing), an operation type (Insert/Delete/Replace) and editable lines are randomly selected. LLMs are used to inject single-line bugs. Bug validity is verified via unit tests (which must fail). Multi-bug programs are constructed by combining multiple independent atomic bugs, requiring a minimum distance (stride) between bugs and satisfying independence constraints.
    - **Design Motivation**: Ensure atomicity (the fix cannot be completed by modifying only a subset of the bug) and independence (fixing one bug does not affect the fix of another), which are prerequisites for precisely defining edit-level precision and bug-level recall.

2.  **Edit-Level Precision**:
    - **Function**: Measure the proportion of necessary modifications in model outputs.
    - **Mechanism**: $\text{precision}_\epsilon = \frac{1}{|\hat{E}|} \sum_{i=1}^k F_\mathcal{U}(\hat{C}_i) \cdot (|\hat{E}_i|)_\epsilon$. A map function aligns ground-truth edits with predicted edits, while an essential function searches for the minimum necessary subset of edits, introducing a tolerance $\epsilon$ to allow for minor edit redundancy.
    - **Design Motivation**: Traditional unit test pass rates fail to penalize redundant modifications. Precision metrics bring evaluation down to the line level, directly measuring "is this modification necessary?"

3.  **Bug-Level Recall**:
    - **Function**: Measure how many bugs were correctly fixed.
    - **Mechanism**: $\text{recall} = \frac{1}{k} \sum_{i=1}^k F_\mathcal{U}(\hat{C}_i)$. For each bug $i$, a pseudo-fix version is constructed—retaining ground-truth fixes for all other bugs and using only the model's modification for bug $i$ to check if it passes unit tests.
    - **Design Motivation**: In multi-bug scenarios, fixing part of the bugs should receive partial credit rather than an all-or-nothing score.

### Loss & Training

PDB does not involve model training. Evaluation uses PDB-Single-Hard (5,751 single-line bug samples) and PDB-Multi (256 multi-line bug samples), constructed from BigCodeBench and LiveCodeBench. The bug generator pool includes GPT-5.1-Codex, Claude-4.5-Sonnet, and Gemini-2.5-Pro.

## Key Experimental Results

### Main Results

| Model | Precision | Recall | Unit Test (%) |
|-------|-----------|--------|---------------|
| Claude-Sonnet-4.5 | **71.8** | **81.4** | 75.7 |
| Gemini-2.5-Pro | 71.4 | 83.5 | 78.1 |
| Qwen3-Coder-480B | 65.8 | 77.2 | 70.3 |
| DeepSeek-V3.2 | 48.4 | 70.0 | 71.4 |
| DeepSeek-V3.2-Thinking | 45.0 | 71.2 | **79.0** |
| GPT-5.1-Codex | 39.7 | 71.7 | 76.1 |

### Ablation Study

| Analytic Dimension | Result |
|--------------------|--------|
| Free Prompting vs. Minimal Edit Prompting | Precision plummeted for all models under free prompting; Gemini dropped by 40 absolute points. |
| Iterative Debugging (3 rounds) | Improved pass rates and recall, but precision remained stagnant or decreased. |
| Agent Debugging (with test feedback) | Claude-Code precision remained at only 50%; additional feedback exacerbated regeneration. |
| Impact of Bug Count | More bugs led to lower precision (more redundant edits); recall depended on the dataset. |

### Key Findings

- **Ranking Inversion**: GPT-5.1-Codex ranked high in unit test pass rate (76.1%) but last in precision (39.7%)—it is the most severe "regenerator."
- Although Qwen3-Coder-480B had a lower pass rate (70.3%), its precision reached 65.8%—characterizing it as "weak but precise."
- Model debugging behaviors can be classified into four types: Precise-Pass, Weak-but-Precise, Weak-but-Locating, and Pass-Oriented (Regeneration).
- Iterative and agentic strategies improve functional correctness but do not improve precision—current methods fix bugs by expanding the scope of modifications rather than precise localization.
- Approximately 1.65% of cases exhibited bug interaction; PDB's independence assumption holds in the vast majority of cases.

## Highlights & Insights

- The question "Debugging or regenerating?" targets the core pain point of current code LLMs—revealing the fundamental flaw of unit-test-only evaluation.
- The definitions of edit-level precision and bug-level recall are precise and practically significant—they can be directly used to improve post-training pipelines.
- The discovery that GPT-5.1-Codex precision is only 39.7% is striking—the strongest models are often the least precise, suggesting that post-training processes may be reinforcing regeneration behavior.

## Limitations & Future Work

- The assumption of bug independence often fails in real-world software—interacting bugs are the true difficulty in debugging.
- Only Python is evaluated; the applicability to other languages requires verification.
- Fixes that are semantically equivalent but different in form may be incorrectly penalized.
- How to improve the post-training process to enhance precision was not explored—this represents the most valuable direction for future work.

## Related Work & Insights

- **vs. DebugBench**: DebugBench mines bugs from historical commits but relies only on unit tests, failing to measure precision; PDB fills this gap with edit-level evaluation.
- **vs. SWE-bench**: SWE-bench focuses on repo-level bug fixing involving complex localization but lacks precision metrics; the two are complementary.
- **vs. APR (Automated Program Repair)**: Traditional APR focuses on minimal fixes; PDB introduces this philosophy into LLM evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Proposes a paradigm shift in debugging evaluation—from program-level to edit-level—with highly impactful findings.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Analyzes 9 frontier models, iterative/agent/multi-line/classification settings, and includes manual verification of metric accuracy.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Precise problem definitions, rigorous formalization, and in-depth experimental analysis.
- **Value**: ⭐⭐⭐⭐⭐ Directly reveals the fundamental problems in code LLM post-training, providing important insights for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)
- [\[ICML 2026\] AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms](../../ICML2026/code_intelligence/algoveri_an_aligned_benchmark_for_verified_code_generation_on_classical_algorith.md)
- [\[NeurIPS 2025\] AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy](../../NeurIPS2025/code_intelligence/astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)

</div>

<!-- RELATED:END -->
