---
title: >-
  [Paper Note] UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench
description: >-
  [ACL 2025][Code Intelligence][SWE-Bench] This paper proposes the UTBoost framework, which enhances test case coverage of SWE-Bench through an LLM-based test case generator (UTGenerator) and an improved parser. It identifies 36 inadequately tested instances and 345 patches falsely flagged as passed, leading to ranking changes of 40.9% on SWE-Bench Lite and 24.4% on SWE-Bench Verified.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "SWE-Bench"
  - "Test Case Augmentation"
  - "Code Generation Evaluation"
  - "Metamorphic Testing"
  - "LLM Coding Agents"
date: 2026-05-08
content_hash: dc88739cf49deea0
---

# UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench

**Conference**: ACL 2025  
**arXiv**: [2506.09289](https://arxiv.org/abs/2506.09289)  
**Code**: [github.com/CUHK-Shenzhen-SE/UTBoost](https://github.com/CUHK-Shenzhen-SE/UTBoost)  
**Area**: Other (Software Engineering/Code Generation Evaluation)  
**Keywords**: SWE-Bench, Test Case Augmentation, Code Generation Evaluation, Metamorphic Testing, LLM Coding Agents

## TL;DR

This paper proposes the UTBoost framework, which enhances test case coverage of SWE-Bench through an LLM-based test case generator (UTGenerator) and an improved parser. It identifies 36 inadequately tested instances and 345 patches falsely flagged as passed, leading to ranking changes of 40.9% on SWE-Bench Lite and 24.4% on SWE-Bench Verified.

## Background & Motivation

SWE-Bench is a standard benchmark for evaluating the capabilities of code generation agents on real-world Python projects. It is constructed based on GitHub issues and corresponding pull requests, using human-written test cases to verify whether the generated patches resolve the issues.

However, **human-written test cases are often inadequate**: generated patches may pass the tests but fail to truly resolve the issues. For instance, in the `mwaskom__seaborn-3010` instance, the issue requires the `PolyFit` function to handle missing data, but the original test only considers cases where both x and y are missing, failing to cover the boundary condition where only x is missing. The patch by IBM SWE-1.0 passes the original test but errors out when only x is missing.

Furthermore, the **test log parser of SWE-Bench is flawed**: when using regular expressions to extract test results, it fails to handle test case logs that span multiple lines, leading to incorrect test annotations. For example, in `django__django-13710`, the test name `test_immutable_content_type` spans two lines, and the parser mistakenly extracts "Regression for #9362" as the test name.

## Method

### Overall Architecture

The UTBoost system consists of three key components:

1. **UTGenerator**: An LLM-based test case generator
2. **Intramorphic Testing**: Constructing test oracles
3. **Improved Parser**: Fixing flaws in the original SWE-Bench parser

Workflow:
- Step 1: Filter generated patches that pass the original tests (satisfying $P(T_{orig}) = P'(T_{orig})$)
- Step 2: Generate augmented test cases $T_{aug}$ with UTGenerator
- Step 3: Check whether the metamorphic relation $P(T_{aug}) = P'(T_{aug})$ holds on the augmented tests
- Step 4: If it does not hold, mark the instance as suspicious

### Key Designs

**Intramorphic Testing**:
- A white-box automated testing technique that establishes a test oracle by comparing the outputs of the original system and the modified system on the same input.
- Standard gold patch is applied to program $P$, and the generated patch is applied to program $P'$.
- Oracle relation: If the two patches resolve the issue equivalently, then $P(T) = P'(T)$ should hold.
- If the augmented test breaks this relation, it indicates that the original test is inadequate.

**UTGenerator: Three-Stage Localization + Test Generation**

1. **File-level Localization**:
    - Build a tree-structure representation of the codebase.
    - The LLM receives the issue description, original test patches, and the tree structure.
    - Outputs the Top-$N$ files most likely requiring test addition.

2. **Function/Class-level Localization**:
    - Compress code files, retaining only class and function headers.
    - The LLM analyzes the compressed format to locate the functions or classes most likely to receive the tests.

3. **Line-level Localization**:
    - Extract specific code snippets.
    - The LLM identifies the exact line range for adding augmented test cases.

4. **Test Case Generation**:
    - Extend the localized lines using a context window of $x$ lines.
    - The LLM generates the augmented test cases and their dependencies.

**Improved Parser**:
- Use a queue to track adjacent log data.
- Match test case names precisely using regular expressions.
- Iteratively search until the correct test name is found when tests span multiple lines.
- Fixes various edge cases that the original parser failed to handle.

### Loss & Training

UTGenerator uses GPT-4o as the LLM backend. A multi-temperature sampling strategy is adopted to increase the diversity of test cases:
- Temperature 0: 1 deterministic patch
- Temperature 0.8: 20 patches
- Temperature 0.9: 20 patches
- Temperature 0.99: 20 patches

File-level localization uses Top-3, the localization stage uses a temperature of 0.8, and the context window is 10 lines of code. The average API cost per SWE-Bench instance is \$1.6.

## Key Experimental Results

### Main Results

**Inadequately Tested Instances Found**:
- SWE-Bench Lite: 23 inadequately tested instances (out of 300)
- SWE-Bench Verified: 26 inadequately tested instances (out of 500)
- 36 distinct instances in total (with overlap between the two sets)

**Incorrect Patches Identified**:
- SWE-Bench Lite: 170 out of 599 patches (28.4%) that passed the original tests were actually incorrect
- SWE-Bench Verified: 92 out of 584 passed patches (15.7%) were actually incorrect

**Impact of Parser Flaws**:
- SWE-Bench Lite: 54.7% (164/300) of instance annotations were affected
- SWE-Bench Verified: 54.2% (271/500) of instance annotations were affected
- After correction, an additional 64 (Lite) and 79 (Verified) incorrect patches were discovered

**Total Incorrect Patches Identified**:
- SWE-Bench Lite: 176 (augmented tests + improved parser)
- SWE-Bench Verified: 169

### Ablation Study

**Project Distribution Analysis**:
- django and sympy account for the majority of errors: 84.1% of incorrect patches in SWE-Bench Lite, and 82.6% in Verified
- Inadequately tested instances are distributed across 9 out of 12 projects

**Leaderboard Impact**:
- SWE-Bench Lite: 40.9% (18/44) of ranks changed
- SWE-Bench Verified: 24.4% (11/45) of ranks changed
- Typical case: Amazon-Q-Developer-Agent dropped from 1st place (55%) to tied 1st with devlo (53.6%) due to having 7 incorrect patches

### Key Findings

1. **Even manual review by 93 professional developers missed inadequate test issues**: UTBoost identified 26 problematic instances in SWE-Bench Verified.
2. **Parser flaws have an extremely broad impact**: over 54% of instance annotations contained errors.
3. **False pass rate caused by inadequate testing is as high as 28.4%**: nearly one-third of the "passed" patches are actually incorrect.
4. **django and sympy are the weakest projects**: concentrating the vast majority of errors.

## Highlights & Insights

- **First systematic resolution of inadequate testing in SWE-Bench**: prior works (Aleithan et al., Chen and Jiang) only manually identified issues, whereas UTBoost provides an automated solution.
- **First application of metamorphic testing to evaluate open-source software systems**: elegantly leveraging the equivalence of the Gold Patch and generated patches to establish test oracles.
- **Three-stage localization strategy effectively handles large-scale codebases**: progressively narrowing down from file $\to$ function/class $\to$ line.
- **Improved parser fixes a long-neglected infrastructure flaw**: affecting more than half of the instances.

## Limitations & Future Work

- **Dependency on at least one resolved instance by an agent**: UTBoost requires cross-validation of Gold Patch and generated patches, and cannot process instances that have not been resolved by any agent (currently covering 74.6% of Lite and 81.6% of Verified).
- **Reliance on GPT-4o only**: integrating other LLMs might increase test diversity.
- **Simplified architecture**: based on a simplified version of Agentless; employing more complex coding agent frameworks could generate more diverse tests.
- **Average cost of \$1.6 per instance**: making large-scale usage relatively expensive.
- **Covers only Python projects**: has not yet been extended to other programming languages.

## Related Work & Insights

- Comparison with EvalPlus (Liu et al., 2024): EvalPlus adds tests to HumanEval/MBPP via type-aware mutation, but cannot handle the multi-file/multi-dependency complexities of SWE-Bench.
- Insights for code generation benchmark design: **test coverage and parser correctness are fundamental to benchmark credibility** and cannot rely solely on manual review.
- Provides plug-and-play augmented test cases for future SWE-Bench submissions.

## Rating

- **Novelty**: ★★★★☆ (novel metamorphic testing application, but the overall approach is somewhat engineering-oriented)
- **Experimental Thoroughness**: ★★★★★ (comprehensive coverage of both benchmarks, with results confirmed by manual review)
- **Value**: ★★★★★ (directly impacts the SWE-Bench leaderboard, with code and data released)
- **Writing Quality**: ★★★★☆ (clear structure, detailed cases, but some content is repetitive)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents](../../NeurIPS2025/code_intelligence/swe-rebench_an_automated_pipeline_for_task_collection_and_decontaminated_evaluat.md)
- [\[ACL 2025\] CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation](coco-bench_a_comprehensive_code_benchmark_for_multi-task_large_language_model_ev.md)
- [\[ICML 2025\] Training Software Engineering Agents and Verifiers with SWE-Gym](../../ICML2025/code_intelligence/training_software_engineering_agents_and_verifiers_with_swe-gym.md)
- [\[ACL 2025\] DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal](dars_dynamic_action_re-sampling_to_enhance_coding_agent_performance_by_adaptive_.md)
- [\[ICML 2026\] SWE-IF: Aligning Code Evaluation with Human Preference](../../ICML2026/code_intelligence/swe-if_aligning_code_evaluation_with_human_preference.md)

</div>

<!-- RELATED:END -->
