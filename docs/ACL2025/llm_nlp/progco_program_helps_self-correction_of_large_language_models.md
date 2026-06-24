---
title: >-
  [Paper Note] ProgCo: Program Helps Self-Correction of Large Language Models
description: >-
  [ACL 2025 Main][LLM (Other)][Self-correction] ProgCo proposes using LLMs to automatically generate and execute verification pseudoprograms (ProgVe) to check the correctness of their own answers, and then utilizes a dual reflection and correction mechanism (ProgRe) on both the answers and the verification programs to achieve reliable self-correction. This significantly improves correction success rates on instruction-following and mathematical reasoning tasks.
tags:
  - "ACL 2025 Main"
  - "LLM (Other)"
  - "Self-correction"
  - "program-driven verification"
  - "pseudoprogram"
  - "dual reflection"
  - "LLM reasoning"
date: 2026-05-08
content_hash: ea544ef2efd964a3
---

# ProgCo: Program Helps Self-Correction of Large Language Models

**Conference**: ACL 2025 Main  
**arXiv**: [2501.01264](https://arxiv.org/abs/2501.01264)  
**Code**: [https://github.com/songxiaoshuai/progco](https://github.com/songxiaoshuai/progco)  
**Area**: LLM/NLP  
**Keywords**: Self-correction, program-driven verification, pseudoprogram, dual reflection, LLM reasoning

## TL;DR

ProgCo proposes using LLMs to automatically generate and execute verification pseudoprograms (ProgVe) to check the correctness of their own answers, and then utilizes a dual reflection and correction mechanism (ProgRe) on both the answers and the verification programs to achieve reliable self-correction. This significantly improves correction success rates on instruction-following and mathematical reasoning tasks.

## Background & Motivation

**Background**: Self-correction is an important direction in LLM research, aimed at enabling models to verify and correct their initial answers on their own without relying on external feedback.

**Limitations of Prior Work**: Existing research indicates that LLMs are often unreliable during the self-verification phase—they struggle to accurately judge the correctness of their own answers, especially in complex reasoning tasks. Incorrect verification feedback further misleads the correction phase, leading to a vicious cycle of "correcting into worse answers." Extensive experiments show that pure natural language self-verification easily misses key compliance checks when constraints are complex.

**Key Challenge**: The reliability of self-verification is the critical bottleneck determining the success of self-correction. Natural language verification lacks structured logic and struggles to cover all constraints; if the verification itself goes wrong, correction will instead introduce new errors.

**Goal**: Design a more reliable verification mechanism and a more robust correction strategy to enable LLMs to achieve genuinely effective self-correction in complex reasoning scenarios.

**Key Insight**: The authors observe that LLMs possess strong capabilities in code understanding and execution, and programmatic logic is naturally structured, executable, and debuggable. Therefore, they propose using "pseudoprograms" to replace natural language for verification—encoding verification logic into executable code snippets.

**Core Idea**: Program the verification process (ProgVe), utilizing LLMs to automatically generate and symbolically execute verification pseudoprograms; during the correction phase (ProgRe), reflect simultaneously on both the answer and the verification program itself to avoid being misled by erroneous feedback.

## Method

### Overall Architecture

ProgCo's overall pipeline is divided into three phases: (1) Initial answer generation by LLM; (2) ProgVe phase—the LLM automatically generates a verification pseudoprogram based on the question and the answer, simulating execution to judge whether the answer is correct; (3) ProgRe phase—if verification identifies issues, the LLM performs dual reflection and correction on both the answer and the verification program. These three phases can be iteratively executed for multiple turns, gradually improving answer quality.

### Key Designs

1. **Program-Driven Verification (ProgVe)**:

    - **Function**: Encode the verification logic of the answer into a structured pseudoprogram and simulate execution to determine whether the answer satisfies all constraints.
    - **Mechanism**: Based on the question requirements and the initial answer, the LLM automatically generates a Python-style verification pseudoprogram. The program contains checking logic for each constraint (such as formatting requirements in instruction-following tasks or reverse checks in mathematical tasks). The LLM then simulates the execution of this program and outputs the pass/fail status of each checklist item. Compared with natural language verification, programmatic verification systematically covers all constraints, has clearer logic, and is less prone to omissions.
    - **Design Motivation**: Natural language verification is prone to ad-hoc checking, failing to comprehensively cover complex constraints. Pseudoprograms naturally require structured enumeration of all check items, and the execution process is traceable, making it easy to locate errors when they occur.

2. **Program-Driven Correction (ProgRe)**:

    - **Function**: Based on the verification feedback from ProgVe, simultaneously correct the answer and the verification program.
    - **Mechanism**: ProgRe introduces a "dual reflection" mechanism—when the verification program reports that the answer is incorrect, the LLM reflects not only on how to improve the answer but also on whether the verification program itself has a bug. Specifically, the LLM first checks whether the logic of the verification program is correct (i.e., whether there is a false positive). If the verification program is buggy, it prioritizes correcting the program; if the verification program is correct, it corrects the answer accordingly. This dual-channel reflection effectively avoids mis-correction caused by verification errors.
    - **Design Motivation**: Traditional methods only correct answers and fully trust verification feedback, which amplifies errors when verification is unreliable. Dual reflection endows the system with "self-doubt" capabilities.

3. **Tool-Augmented ProgCo**:

    - **Function**: Delegate the numerical calculation parts of the pseudoprogram to a physical Python interpreter for execution.
    - **Mechanism**: Identify parts of the verification pseudoprogram that require precise numerical calculations (e.g., mathematical calculations, statistical computing), run these snippets in a real Python environment via API calls, and return the results to the LLM. This overcomes the LLM's inherent weakness in complex numerical reasoning.
    - **Design Motivation**: When LLMs simulate the execution of pseudoprograms, numerical calculation is error-prone (such as large number multiplication or floating-point operations), whereas a physical program executor guarantees precision.

### Loss & Training

ProgCo is a test-time/inference-time framework that does not involve model fine-tuning or extra training. All functions are achieved through carefully designed prompts, which can be directly applied to any LLM with strong instruction-following capabilities (such as GPT-4o, Claude, etc.). The iteration turn count (`max_cur_turn`) is a key hyperparameter, set to 3 in the paper's experiments.

## Key Experimental Results

### Main Results

The paper evaluates on three benchmarks: IFEval (instruction following), GSM8K (basic math reasoning), and MATH (complex math reasoning).

| Method | IFEval Prompt-Strict | IFEval Inst-Strict | GSM8K Acc | MATH Acc |
|------|---------------------|-------------------|-----------|----------|
| Initial (GPT-4o) | 76.7 | 83.5 | 95.1 | 76.4 |
| Self-Refine | 75.6 (-1.1) | 82.7 (-0.8) | 94.5 (-0.6) | 74.8 (-1.6) |
| Self-Verify | 77.3 (+0.6) | 84.1 (+0.6) | 95.3 (+0.2) | 76.8 (+0.4) |
| ProgCo (ours) | **80.4 (+3.7)** | **86.3 (+2.8)** | **96.2 (+1.1)** | **78.6 (+2.2)** |
| ProgCo + Tool | **81.1 (+4.4)** | **86.9 (+3.4)** | **96.8 (+1.7)** | **80.1 (+3.7)** |

### Ablation Study

| Configuration | IFEval Prompt-Strict | MATH Acc | Description |
|------|---------------------|----------|------|
| ProgCo Full | 80.4 | 78.6 | Full model |
| w/o ProgVe (using NL verification) | 77.8 (-2.6) | 77.1 (-1.5) | Remove programmatic verification, switch to natural language verification |
| w/o ProgRe (single-channel correction) | 78.5 (-1.9) | 77.6 (-1.0) | Remove dual reflection, only correct the answer |
| w/o Iteration (single turn) | 78.1 (-2.3) | 77.3 (-1.3) | Perform only one turn of correction |
| ProgVe Only (no correction) | 77.9 (-2.5) | 77.0 (-1.6) | Verify only without correction |

### Key Findings

- ProgVe contributes the most; replacing natural language verification with pseudoprograms brings about a 2.6% improvement, confirming that structured verification is key.
- ProgRe's dual reflection mechanism provides an additional improvement of approximately 1.9%, effectively mitigating mis-correction caused by verification errors.
- The traditional Self-Refine method showed performance drops across all tasks, validating the argument that "unreliable verification is worse than none."
- Multi-turn iterations bring consistent improvements to final performance, typically converging after 2-3 turns.
- Combining physical Python tools (ProgCo + Tool) brings an additional 1.5% gain on mathematical tasks, validating the complementarity between symbolic tools and pseudoprogram verification.

## Highlights & Insights

- **The Ingenuity of Pseudoprogram Verification**: Instead of running real code, it leverages the LLM's code comprehension capability to "simulate execution"—obtaining the structural advantages of programmatic logic without relying on an external execution environment. This design makes the framework widely applicable.
- **Dual Reflection is the Core Competency**: The design of simultaneously doubting both the answer and the verification program breaks the traditional assumption of "fully trusting the verifier," making it much more robust in realistic scenarios where verifiers are imperfect.
- **Transferability of Programmatic Verification to Code Generation Tasks**: For code generation, LLMs can generate test cases (pseudoprograms) to verify their own generated code, forming a similar self-correction feedback loop.

## Limitations & Future Work

- The framework's performance heavily depends on the LLM's code comprehension ability, which might be less effective on weaker models.
- Multi-turn iterations increase inference latency and costs (each turn requires additional verification and correction calls), requiring a trade-off in latency-sensitive scenarios.
- The paper primarily evaluates on English tasks, leaving its effectiveness on other languages like Chinese unknown.
- The quality of the pseudoprogram depends entirely on the LLM's prompt-following ability, lacking formal guarantees.
- Future work can explore training specialized verification program generators or combining this with formal verification methods.

## Related Work & Insights

- **vs Self-Refine**: Self-Refine uses natural language for verification and correction, which often fails or even degrades performance on complex tasks. ProgCo drastically improves verification reliability through programmatic verification.
- **vs Self-Consistency**: Self-Consistency improves accuracy by sampling multiple reasoning paths and voting, but does not perform correction. ProgCo can complement Self-Consistency.
- **vs Program-Aided Language models (PAL/PoT)**: PAL-like methods translate the reasoning process into code to execute. Unlike them, ProgCo programs the "verification process" instead of the "reasoning process", and does not strictly require physical execution of the code.

## Rating

- Novelty: ⭐⭐⭐⭐ Programming the verification process is a simple and effective innovation, though the overall framework is still an improvement of the verify-correct paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers instruction-following and mathematical reasoning tasks, with a complete ablation study, but lacks comparisons across more models.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and the framework diagram is intuitive.
- Value: ⭐⭐⭐⭐ Provides a general test-time self-correction strategy with high utility, directly applicable to existing LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Understanding the Dark Side of LLMs' Intrinsic Self-Correction](understanding_the_dark_side_of_llms_intrinsic_self-correction.md)
- [\[ACL 2025\] Self-Training Elicits Concise Reasoning in Large Language Models](self-training_elicits_concise_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Multi-Prompting Decoder Helps Better Language Understanding](multi-prompting_decoder_helps_better_language_understanding.md)
- [\[ACL 2025\] Conformity in Large Language Models](conformity_in_large_language_models.md)
- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](classifying_unreliable_narrators.md)

</div>

<!-- RELATED:END -->
