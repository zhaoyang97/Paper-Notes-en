---
title: >-
  [Paper Note] CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning
description: >-
  [ICLR2026][Code Intelligence][Code Reasoning] CodeSense is the first fine-grained code semantic reasoning benchmark oriented toward real-world software engineering. The authors performed testing and captured execution traces across 744 Python/C/Java GitHub projects to automatically construct ground truth for execution values and program properties (loops, pointer aliasing, branches) at statement, block, and function levels. Evaluating 14 SOTA LLMs across 4…
tags:
  - "ICLR2026"
  - "Code Intelligence"
  - "Code Reasoning"
  - "Real-World Code"
  - "Execution Tracing"
  - "Fine-grained Semantics"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: b9b690499c94b9eb
---

# CodeSense: a Real-World Benchmark and Dataset for Code Semantic Reasoning

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=ehXVDJm0PS](https://openreview.net/forum?id=ehXVDJm0PS)  
**Code**: https://codesense-bench.github.io/  
**Area**: Code Intelligence / Benchmark / Code Semantic Reasoning  
**Keywords**: Code Reasoning, Real-World Code, Execution Tracing, Fine-grained Semantics, LLM Evaluation

## TL;DR
CodeSense is the first fine-grained code semantic reasoning benchmark oriented toward real-world software engineering. The authors performed testing and captured execution traces across 744 Python/C/Java GitHub projects to automatically construct ground truth for execution values and program properties (loops, pointer aliasing, branches) at statement, block, and function levels. Evaluating 14 SOTA LLMs across 4,483 samples reveals that they frequently fail to correctly calculate arithmetic and API calls even for individual real-world statements.

## Background & Motivation
**Background**: Numerous benchmarks for code LLMs exist, generally falling into two categories. One category is code generation (HumanEval+, LiveCodeBench, BigCodeBench, CodeBenchGen), where data predominantly comes from competition problems or synthetic snippets. The other is reasoning (CruxEval, REval, CodeMind), which primarily focuses on function-level input/output prediction for short, synthetic code. Others like SWE-Bench and KGym use real-world code but only measure the success of end-to-end tasks (e.g., generating patches for GitHub issues).

**Limitations of Prior Work**: Code generation benchmarks do not measure whether a model truly "understands" code behavior. Coarse-grained reasoning tasks like input/output prediction provide only a final conclusion, failing to locate exactly where the model erred—be it a specific statement, API, or loop. Task-level evaluations like SWE-Bench cannot distinguish whether a model succeeds through genuine semantic understanding or mere pattern matching. In short, no benchmark provides **fine-grained** probes into the semantic reasoning capabilities of models using **real-world code**.

**Key Challenge**: The fundamental requirement underlying software engineering tasks (test input generation, vulnerability detection, fault localization, program repair) is fine-grained semantic understanding of program execution behavior. For instance, to trigger a specific piece of hazardous code, one must understand the semantics of the arithmetic operation $n = input \times 23$ and the branch condition $3465 \ge n \ge 2287$ to deduce $input=120$. Existing evaluations are neither fine-grained nor real-world, making it impossible to measure this capability.

**Goal**: Construct a multi-language, fine-grained semantic reasoning benchmark covering real projects across three levels (statement, block, and function) and three types of program properties (loops, pointers, and branches), accompanied by an execution tracing framework for automated ground truth extraction.

**Key Insight**: Code semantics are formally defined in programming language theory—operational semantics describe step-by-step execution, while axiomatic semantics use logical assertions to describe properties. The authors operationalize these formal definitions into a series of automatically verifiable prediction tasks: asking models to predict execution values for statements/blocks, loop iteration counts, pointer aliasing, or branch outcomes. Ground truth is not manually labeled but derived from **actual program execution and trace recording**.

**Core Idea**: Use dynamic execution traces of real-world projects as ground truth to decompose "code understanding" into a spectrum of fine-grained, automatically verifiable semantic prediction tasks, thereby precisely exposing the weaknesses of LLMs in code reasoning.

## Method

### Overall Architecture
CodeSense is not a model but a benchmark construction pipeline: (1) Data collection from 744 real Python/C/Java projects on GitHub, using language-specific toolchains to **build, run tests, and record execution traces** (variable values, data types, function names, and memory addresses per statement); (2) Automated **ground truth extraction** from traces using static and dynamic analysis, filtering out insignificant functions to obtain 2,125 Python, 876 C, and 875 Java unique functions, curating 4,483 samples with standard answers; (3) Evaluation of 14 SOTA LLMs using natural language prompts, scored by exact-match accuracy across six research questions.

### Key Designs

**1. Fine-grained Semantic Reasoning Task Hierarchy: Decomposing "Code Understanding" into Verifiable Sub-problems**

To address the limitation where coarse-grained I/O prediction cannot locate errors, the authors designed four categories of tasks with explicit inputs and verifiable answers:

- **Task 1: Block-level Semantics**: Given a block of statements (starting from function entry and incrementally increasing), predict output given input, or predict input given output. Function-level I/O prediction is a special case of this.
- **Task 2: Statement-level Semantics**: Statements are categorized into five types—arithmetic, boolean expressions, API/function calls, variable assignments, and constant assignments. A statement is randomly sampled, and the model predicts its output given the input.
- **Task 3: Intra-procedural Program Properties**: Focuses on three critical properties for SE—3-1 Loops (predicting iterations, intra-loop variable values, post-loop values); 3-2 Pointers (predicting if two pointers alias to the same memory address, performed for C); 3-3 Branches (predicting whether a branch is evaluated as true or false).
- **Task 4: Semantic Approximation**: Many SE tasks require approximate rather than exact values (e.g., identifying that $input \in [100, 150]$ triggers a bug). The authors define abstract values mapping concrete values to interval labels to evaluate whether models can predict these abstractions.

**2. Multi-language Real-world Project Tracing Framework: Ground Truth from Execution**

To solve the difficulty of constructing fine-grained ground truth, specialized toolchains were built for each language:

- **Python**: 544 repositories were selected from PyPIbugs; dependencies were installed, tests run via `pytest`, and traces captured using `PySnooper`.
- **C**: 100 projects from OSS-Fuzz were built in Docker using fuzzing harnesses; the tracing framework was built on the GNU Debugger (GDB).
- **Java**: 100 projects from the SF110 dataset used EvoSuite to generate and run tests, with tracing built on the Java Debugger.

**3. Data Filtering and Atomic Type Constraints**

From the full traces, unique functions were extracted and filtered if they contained only comments, were too long for context windows, or had no functional body (e.g., only `return 0`). To ensure clean exact-match scoring, the authors restricted ground truth to **atomic data types** (int, float, str, bool, list, pointer, etc.).

## Key Experimental Results

### Main Results
14 SOTA LLMs (8 reasoning, 6 non-reasoning) were evaluated. The table below summarizes sample sizes and representative findings:

| Task | Sample Size (Py/C/Java) | Key Findings |
|------|------|------|
| Task 1 Block I/O | 1860 / 731 / – | Claude 3.5 and GPT-4o-mini accuracy <30% on single C blocks; no model exceeded 50% for Python. |
| Task 1 Function I/O | 308 / 94 / 74 | Accuracy drops as blocks grow from 1 to 3 statements; Claude 3.5 is ~20% for 3-statement Python blocks. |
| Task 2/4 Statement | 545 / 485 / – | Arithmetic and API calls are the most difficult; Boolean and constant assignments are handled better. |
| Task 3-1/4 Loops | 105 / – / – | Post-loop variable values are the hardest to predict; iterations are the easiest. |
| Task 3-2 Pointer Alias | – / 49 / – | Binary classification; some open-source models score <50% (worse than random guessing). |
| Task 3-3 Branching | 232 / – / – | Binary classification; pointer aliasing is better predicted than branch execution. |
| **Total** | **4483** | Claude 3.5 performed best overall; input prediction (inverse semantics) is the hardest task. |

### Ablation Study
- **Increasing Block Size (RQ1)**: Accuracy consistently decreases as block size increases, showing difficulty in tracking variable states across statements.
- **Statement Types (RQ2)**: Arithmetic and API calls are the hardest; providing API definitions in prompts offered limited improvement.
- **Program Properties (RQ3)**: Post-loop values are the most difficult; some models underperform random chance on binary pointer tasks.
- **Prompting Strategy (RQ4)**: More shots improve performance; RAG-style relevant examples are effective; CoT provides limited help for simple statement prediction.

### Key Findings
- **Pattern Matching Over Semantic Reasoning**: Models often correctly answer "do p and q alias" when code explicitly shows `p = q`, or predict iterations for `for i in range(100):`. This suggests reliance on surface patterns; accuracy collapses when no such explicit patterns exist.
- **Difficulty of Inverse Semantics**: Input prediction is significantly harder than output prediction, revealing that LLMs struggle with reverse operational semantics.

## Highlights & Insights
- **Ground truth from execution, not labeling**: Automated tracing makes it possible to scale fine-grained benchmarks to real-world code while mitigating data leakage.
- **Fine-grained Diagnostic Power**: Unlike SWE-Bench, CodeSense identifies whether a model fails at arithmetic, API calls, or loop state tracking, which is valuable for guiding post-training.
- **Pragmatic Approximation**: The abstract value tasks acknowledge the difficulty of exact value prediction and evaluate approximate semantics, which is often sufficient for practical software engineering.

## Limitations & Future Work
- **Atomic Type Focus**: Currently limited to simple types; complex objects and custom types are not yet evaluated.
- **Language Imbalance**: Java sample sizes are small (74) compared to Python and C.
- **Exact-Match Rigidity**: Strict matching may penalize semantically correct but differently formatted responses (e.g., floats).
- **Test Coverage Dependency**: Ground truth is limited to code paths covered by existing or generated tests.

## Related Work & Insights
- **vs. CruxEval/CruxEval-X**: These perform function-level I/O prediction on synthetic Python; CodeSense uses real-world code and refines the granularity to statements and specific program properties.
- **vs. SWE-Bench/KGym**: While these use real code, they evaluate end-to-end task success; CodeSense decomposes the underlying semantic understanding required for those tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits](edit-bench_evaluating_llm_abilities_to_perform_real-world_instructed_code_edits.md)
- [\[ACL 2026\] ReFEree: Reference-Free and Fine-Grained Method for Evaluating Factual Consistency in Real-World Code Summarization](../../ACL2026/code_intelligence/referee_reference-free_and_fine-grained_method_for_evaluating_factual_consistenc.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](../../ACL2026/code_intelligence/logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ICLR 2026\] Code World Models for General Game Playing](code_world_models_for_general_game_playing.md)
- [\[ICLR 2026\] Code2Bench: Scaling Source and Rigor for Dynamic Benchmark Construction](code2bench_scaling_source_and_rigor_for_dynamic_benchmark_construction.md)

</div>

<!-- RELATED:END -->
