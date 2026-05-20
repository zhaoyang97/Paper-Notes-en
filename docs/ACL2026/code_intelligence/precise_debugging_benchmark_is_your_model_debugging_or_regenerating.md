---
title: >-
  [Paper Note] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?
description: >-
  [ACL 2026][Code Intelligence][Code Debugging] This paper reveals the "regeneration" tendency of frontier LLMs on debugging tasks. By introducing the PDB framework along with edit-level precision and bug-level recall metr…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Code Debugging"
  - "LLM Programming"
  - "Precise Editing"
  - "Benchmark"
  - "Code Regeneration"
date: 2026-05-08
content_hash: b137a1293591351a
---

# Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?

**Conference**: ACL 2026
**arXiv**: [2604.17338](https://arxiv.org/abs/2604.17338)  
**Code**: [GitHub](https://github.com)  
**Area**: Code Intelligence / Debugging Evaluation
**Keywords**: Code Debugging, LLM Programming, Precise Editing, Benchmark, Code Regeneration

## TL;DR

This paper reveals the "regeneration" tendency of frontier LLMs on debugging tasks. By introducing the PDB framework along with edit-level precision and bug-level recall metrics, the authors find that models such as GPT-5.1-Codex pass over 76% of unit tests yet achieve edit precision below 45%, and that iterative and agent-based debugging strategies fail to substantially improve precision.

## Background & Motivation

**Background**: LLMs have achieved remarkable success in code generation, synthesizing complex algorithms from natural language descriptions. However, the dominant activity in real-world software development is not generation from scratch but rather debugging and maintenance.

**Limitations of Prior Work**: (1) When presented with buggy code, LLMs tend to rewrite most or all of the program to "fix" it — a practice that may pass tests but is costly, risky, and difficult to review in production codebases. (2) Existing debugging benchmarks rely solely on unit test pass rate, which cannot distinguish precise repairs from wholesale rewrites — rewriting an entire function and fixing a single buggy line receive identical scores. (3) For programs containing multiple bugs, a model that fixes only a subset of bugs receives the same score of zero as one that fixes nothing.

**Key Challenge**: There is a negative correlation between unit test pass rate and debugging precision — the more aggressively a model rewrites code, the more likely it is to pass tests (functional correctness), yet the lower its edit precision. The existing evaluation paradigm rewards regeneration and provides no incentive for precise debugging.

**Goal**: (1) Design an evaluation framework capable of distinguishing "precise debugging" from "code regeneration." (2) Quantify how far current frontier models are from precise debugging. (3) Assess whether iterative and agent-based debugging strategies improve precision.

**Key Insight**: Two new metrics are defined — *edit-level precision*, measuring what proportion of the model's modifications are necessary, and *bug-level recall*, measuring what proportion of bugs are correctly fixed. A debugging benchmark with ground-truth edit scripts is constructed by automatically injecting validated atomic bugs and composing them into multi-bug programs.

**Core Idea**: Shift debugging evaluation from the program level (pass/fail) to the edit level (which modifications are necessary and which are superfluous), and construct a precise evaluation benchmark through atomic bug synthesis and independence validation.

## Method

### Overall Architecture

The PDB framework operates in two phases. In the **generation phase**, atomic bugs are synthesized and validated by LLMs starting from existing programming datasets, and then composed into multi-bug programs. In the **evaluation phase**, debugging systems repair the buggy programs, and their outputs are assessed using edit-level precision and bug-level recall.

### Key Designs

1. **Atomic Bug Synthesis and Composition**:

    - Function: Generate buggy programs with ground-truth edit scripts, supporting both single-line and multi-line bugs.
    - Mechanism: For each ground-truth program, a bug is injected into a single line by randomly selecting an operation type (insert / delete / replace) and an editable line according to five ODC (Orthogonal Defect Classification) categories (assignment, checking, algorithm, build/package, timing), with an LLM performing the injection. Bug validity is verified by unit tests (the injected bug must cause at least one test to fail). Multi-bug programs are constructed by composing multiple independent atomic bugs, subject to a minimum inter-bug distance (stride) and independence constraints.
    - Design Motivation: Ensuring atomicity (a bug cannot be fixed by modifying only a subset of the affected edits) and independence (fixing one bug does not interfere with fixing others) is a prerequisite for precisely defining edit-level precision and bug-level recall.

2. **Edit-Level Precision**:

    - Function: Measure what proportion of the model's modifications are necessary.
    - Mechanism: $\text{precision}_\epsilon = \frac{1}{|\hat{E}|} \sum_{i=1}^k F_\mathcal{U}(\hat{C}_i) \cdot (|\hat{E}_i|)_\epsilon$. A map function aligns ground-truth edits with predicted edits; an essential function searches for the minimal necessary edit subset; a tolerance $\epsilon$ is introduced to allow a degree of edit redundancy.
    - Design Motivation: Traditional unit test pass rate cannot penalize superfluous modifications. The precision metric moves evaluation to the line level, directly asking "is this modification necessary?"

3. **Bug-Level Recall**:

    - Function: Measure what proportion of bugs are correctly fixed.
    - Mechanism: $\text{recall} = \frac{1}{k} \sum_{i=1}^k F_\mathcal{U}(\hat{C}_i)$. For each bug $i$, a pseudo-corrected version is constructed by applying the ground-truth fixes for all other bugs while using only the model's modifications for bug $i$, then checking whether unit tests pass.
    - Design Motivation: In multi-bug scenarios, partially fixing bugs should yield partial credit rather than an all-or-nothing score.

### Loss & Training

PDB does not involve model training. Evaluation is conducted on PDB-Single-Hard (5,751 single-line bug instances) and PDB-Multi (256 multi-line bug instances), both constructed from BigCodeBench and LiveCodeBench. The bug generator pool consists of GPT-5.1-Codex, Claude-4.5-Sonnet, and Gemini-2.5-Pro.

## Key Experimental Results

### Main Results

| Model | Precision | Recall | Unit Tests (%) |
|-------|-----------|--------|----------------|
| Claude-Sonnet-4.5 | **71.8** | **81.4** | 75.7 |
| Gemini-2.5-Pro | 71.4 | 83.5 | 78.1 |
| Qwen3-Coder-480B | 65.8 | 77.2 | 70.3 |
| DeepSeek-V3.2 | 48.4 | 70.0 | 71.4 |
| DeepSeek-V3.2-Thinking | 45.0 | 71.2 | **79.0** |
| GPT-5.1-Codex | 39.7 | 71.7 | 76.1 |

### Ablation Study

| Analysis Dimension | Result |
|--------------------|--------|
| Free prompt vs. minimal-edit prompt | Precision drops sharply for all models under the free prompt; Gemini decreases by 40 absolute points |
| Iterative debugging (3 rounds) | Improves test pass rate and recall, but precision remains unchanged or decreases |
| Agent debugging (with test feedback) | Claude-Code precision remains only 50%; additional feedback further exacerbates regeneration |
| Effect of bug count | More bugs correlate with lower precision (more superfluous edits); recall varies by dataset |

### Key Findings

- **Ranking inversion**: GPT-5.1-Codex ranks near the top by unit test pass rate (76.1%) yet ranks last by precision (39.7%) — it is the most severe "regenerator."
- Qwen3-Coder-480B achieves lower pass rate (70.3%) but the highest precision (65.8%) — a "weaker but precise" profile.
- Model debugging behaviors can be categorized into four types: precise-and-passing, weak-but-precise, weak-but-localizing, and pass-oriented (regenerating).
- Iterative and agent strategies improve functional correctness but not precision — current methods fix bugs by broadening the scope of modifications rather than by precise localization.
- Approximately 1.65% of cases exhibit bug interactions; the independence assumption of PDB holds in the vast majority of instances.

## Highlights & Insights

- The question "debugging or regenerating?" strikes at a core weakness of current code LLMs, exposing a fundamental flaw in unit-test-based evaluation.
- The definitions of edit-level precision and bug-level recall are both rigorous and practically meaningful — they can be directly incorporated into post-training pipelines.
- The finding that GPT-5.1-Codex achieves only 39.7% precision is striking — the strongest model by functional metrics is the least precise, suggesting that post-training procedures may be reinforcing regeneration behavior.

## Limitations & Future Work

- The assumption of bug independence frequently does not hold in real-world software — interacting bugs represent the true challenge of debugging.
- Evaluation is limited to Python; applicability to other languages remains to be verified.
- Semantically equivalent but syntactically different repairs may be incorrectly penalized.
- The paper does not explore how to improve post-training pipelines to enhance precision — this is the most valuable direction for future work.

## Related Work & Insights

- **vs. DebugBench**: DebugBench mines bugs from historical commits but evaluates solely with unit tests, providing no measure of precision; PDB fills this gap with edit-level evaluation.
- **vs. SWE-bench**: SWE-bench targets real repository-level bug fixing involving more complex localization, yet equally lacks precision evaluation; the two benchmarks are complementary.
- **vs. APR (Automated Program Repair)**: Traditional APR emphasizes minimal repairs; PDB transplants this principle into LLM evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Proposes a paradigm shift in debugging evaluation — from program level to edit level — with highly impactful findings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9 frontier models with iterative, agent, multi-line, and categorical analyses, and includes manual validation of metric accuracy.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is precise, formalization is rigorous, and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Directly exposes a fundamental issue in the post-training of code LLMs, with important implications for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)
- [\[ACL 2026\] River-LLM: Large Language Model Seamless Exit Based on KV Share](river-llm_large_language_model_seamless_exit_based_on_kv_share.md)
- [\[NeurIPS 2025\] AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy](../../NeurIPS2025/code_intelligence/astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)
- [\[ACL 2026\] StoryCoder: Narrative Reformulation for Structured Reasoning in LLM Code Generation](storycoder_narrative_reformulation_for_structured_reasoning_in_llm_code_generati.md)

</div>

<!-- RELATED:END -->
