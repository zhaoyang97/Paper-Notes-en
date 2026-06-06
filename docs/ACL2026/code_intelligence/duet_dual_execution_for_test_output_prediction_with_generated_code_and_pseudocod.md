---
title: >-
  [Paper Note] DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode
description: >-
  [ACL 2026][Code Intelligence][Test output prediction] This paper proposes DUET, a dual-path framework combining direct code execution and LLM-based pseudocode execution. By fusing two complementary execution paths via fu…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Test output prediction"
  - "Pseudocode execution"
  - "Dual-path execution"
  - "Code generation"
  - "Functional majority voting"
date: 2026-05-08
content_hash: dd88cdb9a8182bf2
---

# DUET: Dual Execution for Test Output Prediction with Generated Code and Pseudocode

**Conference**: ACL 2026  
**arXiv**: [2604.11514](https://arxiv.org/abs/2604.11514)  
**Code**: [GitHub](https://github.com/ldilab/DuET)  
**Area**: LLM Safety  
**Keywords**: Test output prediction, Pseudocode execution, Dual-path execution, Code generation, Functional majority voting

## TL;DR

This paper proposes DUET, a dual-path framework combining direct code execution and LLM-based pseudocode execution. By fusing two complementary execution paths via functional majority voting—where the former is reliable given correct implementation but susceptible to implementation errors, and the latter bypasses implementation details but may produce execution hallucinations—DUET improves Pass@1 by 13.6 percentage points on LiveCodeBench test output prediction.

## Background & Motivation

**Background**: Test case generation is a critical stage in code generation pipelines. Within this, test output prediction (predicting the correct output given a problem description and test input) is a challenging task requiring precise program reasoning. Existing methods like TestChain perform prediction by generating code follow by direct execution.

**Limitations of Prior Work**: (1) Fatal issue of direct code execution—models may understand the correct algorithmic logic (generating correct pseudocode) but introduce subtle errors in executable code implementation (e.g., using `len(nums)` instead of `i+1` for cumulative average), leading to execution failures or incorrect outputs; (2) In end-to-end code generation, using execution results of generated code to filter candidate programs faces the "zero-advantage problem"—if the generated test code is also incorrect, filtering becomes ineffective.

**Key Challenge**: Direct code execution depends on code correctness (deterministic but brittle), while LLM reasoning is independent of code but prone to execution hallucinations (flexible but non-deterministic). Their failure modes are complementary.

**Goal**: (1) Propose LLM pseudocode execution to decouple correct logic from implementation errors; (2) Design the DUET framework to fuse the complementary strengths of both execution paths.

**Key Insight**: Decouple "logical correctness in code generation" from "implementational correctness." Pseudocode captures algorithmic intent without being constrained by syntax details, allowing LLM simulation to bypass implementation errors.

**Core Idea**: The two paths for test output prediction—direct execution (reliable when code is correct) and pseudocode simulation (bypasses implementation errors but may hallucinate)—are naturally complementary. Functional majority voting utilizes the strengths of both.

## Method

### Overall Architecture

Given a problem description and test input: Path 1 generates executable code for direct execution; Path 2 generates pseudocode for LLM-based simulation (step-by-step reasoning). Multiple samples are taken from each path, and functional majority voting selects the final predicted output.

### Key Designs

1. **LLM Pseudocode Execution**:

    - Function: Predicts test outputs without relying on code implementation correctness.
    - Mechanism: The LLM first generates a pseudocode description (high-level algorithmic intent), followed by step-by-step simulation—tracking variable states to derive the final output. Pseudocode provides more precise algorithmic descriptions than natural language while being more fault-tolerant than executable code.
    - Design Motivation: Implementation errors (e.g., loop boundaries, variable name confusion) are primary causes of direct execution failure; pseudocode bypasses these details through higher-level abstraction.

2. **Functional Majority Voting**:

    - Function: Combines the complementary advantages of both execution paths.
    - Mechanism: Samples each path $N$ times (e.g., $N=5$), collecting $2N$ outputs. If an output appears in both paths (or achieves a majority in total votes), it is more likely correct. Voting is based on functional equivalence of outputs rather than literal string matching.
    - Design Motivation: Failure modes differ—code execution fails on implementation errors (Figure 2a), while pseudocode execution hallucinates in deep nested loops (Figure 2b). Complementary fusion significantly enhances reliability.

3. **Solving the Zero-Advantage Problem**:

    - Function: Addresses the failure of test filtering in end-to-end code generation pipelines.
    - Mechanism: In CODET pipelines using predicted outputs to filter candidate code, if outputs come from direct execution of generated test code, filtering fails when the test code is wrong (zero-advantage). Since the pseudocode path is decoupled from code correctness, it provides a valid filtering signal even if the candidate code is flawed.
    - Design Motivation: TestChain decreased performance in the CODET pipeline (-5.6pp) due to the zero-advantage problem; DUET avoids this via the pseudocode path.

### Loss & Training

No model training involved. Uses existing LLMs (e.g., Llama-3.1-8B-Instruct) for inference. Each path is sampled 5 times (total 10 calls), with the final output selected via functional majority voting.

## Key Experimental Results

### Main Results

**Test Output Prediction on LiveCodeBench**

| Method | Pass@1 | Gain |
| :--- | :--- | :--- |
| Direct (Code Exec Only) | Baseline | - |
| TestChain | +5.6pp | - |
| Pseudocode Exec | Competitive | - |
| **DUET** | **+13.6pp** | **SOTA** |

**End-to-End Code Generation (CODET Pipeline + Llama-3.1-8B-Instruct)**

| Prediction Method | Pass@1 Change |
| :--- | :--- |
| No Filtering | Baseline |
| TestChain | -5.6pp (Zero-advantage problem) |
| **DUET** | **+3.2pp** |

### Ablation Study

**End-to-End Code Generation on LCB-Easy/BigCodeBench-Hard/DevEval/HumanEval(+)**

| Method | LCB-Easy | BCB-Hard | DevEval | HumanEval(+) |
| :--- | :--- | :--- | :--- | :--- |
| DUET | **Best** | **Best** | **Best** | **Best** |

### Key Findings

- DUET improves test output prediction by 13.6pp, significantly outperforming single-path methods, validating the complementarity of the two paths.
- TestChain reduced performance by 5.6pp in the CODET pipeline, whereas DUET improved it by 3.2pp, highlighting the zero-advantage problem as a critical barrier in deployment.
- Error distributions are complementary: complex logic problems favor implementation errors (pseudocode path is more reliable), while deep nested loops favor hallucinations (code execution is more reliable).
- The abstraction level of pseudocode makes it naturally more fault-tolerant than executable code—even with the same logic, its "implementation" is less error-prone.

## Highlights & Insights

- Decoupling "correct logic" from "correct implementation" is an elegant analysis—pseudocode execution is a natural byproduct of this decoupling.
- The discovery and solution of the zero-advantage problem are significant for the practical deployment of code generation pipelines.
- Functional majority voting is a general fusion strategy extensible to additional execution paths.

## Limitations & Future Work

- Pseudocode execution requires additional LLM calls ($2N$ vs. $N$), doubling computational costs.
- Both pseudocode and code may fail simultaneously on extremely complex algorithmic problems.
- Quality of pseudocode generation depends on the LLM's meta-language capabilities.
- Specialized training for pseudocode execution to reduce hallucinations remains unexplored.

## Related Work & Insights

- **vs. TestChain**: TestChain relies solely on direct code execution and suffers from the zero-advantage problem; DUET's dual-path design resolves this.
- **vs. AlphaCode**: AlphaCode focuses on test input generation, while DUET addresses output prediction.
- **vs. CODET**: The CODET framework relies on accurate test output filtering; DUET provides more reliable outputs for this purpose.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of pseudocode execution and dual-path fusion is novel; discovery of the zero-advantage problem is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple benchmarks for both output prediction and end-to-end generation.
- Writing Quality: ⭐⭐⭐⭐⭐ Deep and clear problem analysis with intuitive graphical explanations.
- Value: ⭐⭐⭐⭐ Directly practical for testing and filtering within code generation pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] PaT: Planning-after-Trial for Efficient Test-Time Code Generation](pat_planning-after-trial_for_efficient_test-time_code_generation.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ICML 2026\] BoostAPR: Boosting Automated Program Repair via Execution-Grounded Reinforcement Learning with Dual Reward Models](../../ICML2026/code_intelligence/boostapr_boosting_automated_program_repair_via_execution-grounded_reinforcement_.md)
- [\[ACL 2026\] To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing](to_diff_or_not_to_diff_structure-aware_and_adaptive_output_formats_for_efficient.md)

</div>

<!-- RELATED:END -->
