---
title: >-
  [Paper Note] Planning-Driven Programming: A Large Language Model Programming Workflow
description: >-
  [ACL 2025][LLM (Other)][Code Generation] This work proposes LPW (LLM Programming Workflow), a two-phase workflow integrating "solution generation -> plan verification -> code implementation -> precise debugging based on plan verification." LPW significantly improves LLM code generation accuracy, achieving new SOTA results on GPT-4o with 98.2% on HumanEval, 84.8% on MBPP, and 59.3% on LiveCode.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Code Generation"
  - "Plan Verification"
  - "LLM Workflow"
  - "Program Repair"
  - "Test-Driven Development"
date: 2026-05-08
content_hash: 67e4c3bdc5082498
---

# Planning-Driven Programming: A Large Language Model Programming Workflow

**Conference**: ACL 2025  
**arXiv**: [2411.14503](https://arxiv.org/abs/2411.14503)  
**Code**: [github](https://github.com/you68681/lpw)  
**Area**: LLM/NLP  
**Keywords**: Code Generation, Plan Verification, LLM Workflow, Program Repair, Test-Driven Development

## TL;DR

This work proposes LPW (LLM Programming Workflow), a two-phase workflow integrating "solution generation -> plan verification -> code implementation -> precise debugging based on plan verification." LPW significantly improves LLM code generation accuracy, achieving new SOTA results on GPT-4o with 98.2% on HumanEval, 84.8% on MBPP, and 59.3% on LiveCode.

## Background & Motivation

Although LLM-based code generation has demonstrated exceptional performance, it still faces several core challenges:

**Deviation in Debugging Directions**: Existing methods (such as Self-Debugging) perform code repair based on execution results and error explanations. However, the feedback lacks precise correction instructions, often causing the debugging process to deviate from the expected solution.

**Unreliable Plans and Tests**: Multi-agent collaborative approaches (such as MapCoder) introduce extra public tests and solution plans, but they lack a methodology to verify the correctness of the generated plans and tests. Incorrect plans can easily mislead subsequent code generation.

**Limited Reasoning Capabilities**: Generating code under strict lexical, syntactic, and semantic constraints remains challenging for LLMs. Furthermore, repairing programs that deviate significantly from the problem description remains an open problem.

**High Resource Consumption**: Multi-agent collaboration requires substantial token resources for communication, leading to low efficiency.

Key Insight: Human programmers verify their problem-solving logic before writing code (analogous to Test-Driven Development, or TDD), whereas existing LLM-based approaches skip this critical step.

## Method

### Overall Architecture

LPW consists of two phases:

**Phase 1: Solution Generation**
- Plan Creation -> Plan Verification -> Verification Review -> Iterative Correction

**Phase 2: Code Implementation**
- Initial Code Generation -> Test Execution -> Error Analysis (Comparing plan verification against execution trace) -> Code Repair

### Key Designs

1. **Solution Plan**: Employs Self-Planning to decompose the problem description into several manageable sub-problems (intermediate steps), providing structured guidance for code generation.

2. **Plan Verification**: The core innovation of LPW. For each visible test case, the LLM performs step-by-step analysis based on the plan to derive the expected output of each intermediate step as well as the final output, comparing it against the test truth values. This process validates the solution plan at the natural language level, encompassing the full set of conditions and logical constraints required to solve the problem.

3. **Verification Review**: Even if the final output matches, the LLM reviews all intermediate step outcomes to detect contextual inconsistencies, mathematical miscalculations, or logical flaws, ensuring the accuracy of intermediate results (which are crucial as subsequent debugging relies on them).

4. **Debugging Based on Plan Verification**: When the code fails on visible tests, LPW compares the code's execution trace (obtained via automatically inserted print statements) against the expected intermediate outputs recorded during plan verification. This allows the model to precisely pinpoint bug locations and generate detailed repair suggestions (Error Analysis). It then leverages code explanations as feedback to repair the code.

5. **Iterative Update Mechanism**:

    - Solution Generation Phase: If the verified output does not match the test truth values, the plan is automatically revised; if the intermediate results are incorrect, the verification is regenerated.
    - Code Implementation Phase: In case of execution failure, the original code is replaced with the repaired version, iterating until it passes or reaches the maximum number of iterations.

### Loss & Training

LPW is a test-time/inference-only method and requires no additional training:
- Uses 2-shot prompting.
- Maximum iterations: 12 iterations each for solution generation and code implementation phases.
- All components are autonomously generated by the LLM via few-shot prompting.
- Relies solely on runtime execution information and LLM-generated output, requiring no annotated corpora.

## Key Experimental Results

### Main Results

**GPT-3.5 backbone**:

| Method | HumanEval | HumanEval-ET | MBPP | MBPP-ET |
|------|-----------|-------------|------|---------|
| Baseline | 74.4 | 66.5 | 67.4 | 52.8 |
| Self-Planning | 77.4 | 69.5 | 69.2 | 52.4 |
| MapCoder | 77.4 | 66.5 | 72.0 | 56.6 |
| Self-Debugging | 81.1 | 72.0 | 71.2 | 56.0 |
| LDB | 82.9 | 72.6 | 72.4 | 55.6 |
| **LPW** | **89.0** | **77.4** | **76.0** | **57.6** |

**GPT-4o backbone (New SOTA)**:

| Method | HumanEval | MBPP | LiveCode | APPS | CodeContests |
|------|-----------|------|----------|------|-------------|
| Baseline | 91.5 | 78.4 | 45.7 | 41.7 | 28.0 |
| LDB | 92.1 | 82.4 | 54.3 | 53.2 | 29.3 |
| **LPW** | **98.2** | **84.8** | **59.3** | **62.6** | **34.7** |

**Llama-3 backbone achieves the largest gain**:

| Benchmark | Baseline | LDB | LPW | LPW vs LDB |
|------|---------|-----|-----|-----------|
| HumanEval | 73.2 | 84.1 | 88.4 | +4.3 |
| MBPP | 44.0 | 57.2 | **73.6** | **+16.4** |

### Ablation Study

| Configuration | HumanEval | MBPP | Description |
|------|-----------|------|------|
| LPW (Full) | 89.0 | 76.0 | - |
| LPW-V (w/o Plan Verification) | 86.0 (-3.0) | 73.2 (-2.8) | Plan verification is critical for both phases |
| LPW-S (w/o Solution Gen Phase) | 86.0 (-3.0) | 73.0 (-3.0) | Directly debug Baseline code |
| LPW-C (w/o Code Repair) | 79.9 (-9.1) | 72.2 (-3.8) | Only generate code based on plans |
| More visible tests (MBPP-ET → MBPP-ET-3) | - | +4.4 | LPW exhibits the highest efficiency in leveraging extra test cases |

### Key Findings

1. **Plan Verification is Crucial**: The performance drop of LPW-V demonstrates the dual value of plan verification in both initial code generation and debugging; unverified plans yield limited performance (LPW-V performs similarly to LPW-S).
2. **Both Phases are Indispensable**: Removing either phase leads to degraded performance, but removing code repair has a more severe impact (-9.1% on HumanEval).
3. **Greater Advantages on Challenging Benchmarks**: Shows prominent advantages on challenging benchmarks such as LiveCode (+5%), APPS (+10%), and CodeContests (+5%).
4. **High Iterative Efficiency**: LPW outperforms the best-performing configurations of LDB/SD with just a single iteration.
5. **Higher Initial Code Quality**: LPW's initial code (at 0 iterations) already achieves 79.9% (vs. Baseline's 74.4%), showing that plan verification directly improves the quality of initial code generation.

## Highlights & Insights

- **Automation of "Think Before You Code"**: This marks the first complete automation of the human developer workflow (verifying reasoning -> coding -> contrastive analysis), translating TDD principles to the LLM era elegantly.
- **Natural Language as an Intermediate Debugging Representation**: Plan verification provides a higher-level "specification of expected behavior" compared to raw code, shifting the paradigm from "trial-and-error debugging" to "discrepancy comparison." This effectively transforms debugging into a text alignment task.
- **Model Agnosticism**: Remains robustly effective across four different models (GPT-3.5, Llama-3, Phi-3, and GPT-4o), proving the generalizability of the workflow design.
- **Highly Efficient Utilization of Visible Tests via Plan Verification**: MBPP-ET-3 experiments demonstrate that LPW utilizes additional test information most efficiently (+4.4% gain vs. LDB's +2.0%).

## Limitations & Future Work

1. **High Token Consumption**: Generating plans and verifications consumes a large volume of tokens, which may represent over-engineering for simpler problems.
2. **Plan-to-Code Translation Bottleneck**: The pass rate of the code remains lower than the accuracy of plan verification, indicating room for improvement in translating from natural-language solutions to concrete code implementations.
3. **Dependence on Visible Test Cases**: LPW requires visible unit tests to verify plans, limiting its direct applicability in scenarios without immediate test cases.
4. **LLM Reasoning Constraints**: The underlying mechanism is ultimately bound by the core logical reasoning limits of LLMs, struggling with complex logic that exceeds the model's base capacity.
5. **XML Format Compatibility**: Prior failures with MapCoder on Phi-3 suggest that rigid formatting constraints may impact small language model performance; although LPW has not explicitly encountered this, it remains an aspect to watch.

## Related Work & Insights

- **Self-Planning (Jiang et al., 2023)**: Generates plans without verifying them, showing limited gains (+1-3% over baseline).
- **Self-Debugging (Chen et al., 2023)**: Rubber-duck-style debugging, which lacks explicit, directed repair instructions.
- **LDB (Zhong et al., 2024)**: Light debugging based on control flow graphs, offering runtime execution traces but coarse feedback.
- **MapCoder (Islam et al., 2024)**: Multi-agent generation of multiple plans without explicit verification.
- **Insight**: In LLM-based code generation, verifying plans at the natural-language level might be more efficient than debugging directly at the code level, as LLMs excel much more at natural language reasoning than tedious code execution tracing.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel design of plan-verification mechanism and verification-guided precise debugging strategy, showcasing an elegant adaptation of TDD principles.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 7 benchmarks, 4 models, full ablation, cost analysis, and case studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear workflow illustrations, comprehensive case descriptions, and rigorous mathematical/formalized problem definitions.
- Value: ⭐⭐⭐⭐⭐ Achieves a new SOTA. The method is practical, model-agnostic, and significantly pushes the boundaries of LLM-driven code generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] APPL: A Prompt Programming Language for Harmonious Integration of Programs and Large Language Model Prompts](appl_a_prompt_programming_language_for_harmonious_integration_of_programs_and_la.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[ACL 2025\] Achieving Certification-by-Design Through Model-Driven Development](achieving_certification-by-design_through_model-driven_development.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] On the Limit of Language Models as Planning Formalizers](limit_llm_planning_formalizer.md)

</div>

<!-- RELATED:END -->
