---
title: >-
  [Paper Note] CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks
description: >-
  [NeurIPS 2025][Code Intelligence][code reasoning] This paper introduces CoRe, a high-quality benchmark comprising 12,553 manually validated task instances. Through three categories of fundamental static analysis tasks—da…
tags:
  - "NeurIPS 2025"
  - "Code Intelligence"
  - "code reasoning"
  - "static analysis"
  - "benchmark"
  - "data dependency"
  - "control dependency"
  - "information flow"
date: 2026-05-08
content_hash: e020b57ec4e00970
---

# CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks

**Conference**: NeurIPS 2025  
**arXiv**: [2507.05269](https://arxiv.org/abs/2507.05269)  
**Code**: [corebench.github.io](https://corebench.github.io/) (Apache-2.0)  
**Area**: Code Intelligence  
**Keywords**: code reasoning, static analysis, benchmark, data dependency, control dependency, information flow

## TL;DR

This paper introduces CoRe, a high-quality benchmark comprising 12,553 manually validated task instances. Through three categories of fundamental static analysis tasks—data dependency, control dependency, and information flow—CoRe directly evaluates the code semantic reasoning capabilities of LLMs, revealing that current models remain severely deficient on tasks requiring multi-step reasoning, such as trace generation and source enumeration.

## Background & Motivation

**Background**: LLMs have been widely applied to software engineering tasks including code generation (HumanEval), program repair (SWE-Bench), and vulnerability detection. Success on these tasks implicitly relies on deep understanding of program semantics, such as value propagation, control flow, and inter-element dependency relationships. Recent work has also explored using LLMs directly as static analyzers (e.g., LLMDFA, IRIS).

**Limitations of Prior Work**: Existing benchmarks primarily evaluate final output correctness in an end-to-end manner—whether code is repaired or whether generated code passes tests—without directly assessing a model's ability to reason about program semantics. Dynamic trace prediction work (e.g., CRUXEval) focuses only on runtime behavior under specific inputs, neglecting unexecuted branches and static semantic properties.

**Key Challenge**: Although LLMs exhibit acceptable performance on downstream tasks, whether their underlying program semantic reasoning capabilities sufficiently support these tasks remains unknown. There is a lack of targeted, fine-grained evaluation methods to diagnose models' reasoning weaknesses.

**Goal**: To design a benchmark that directly evaluates the core code reasoning capabilities of LLMs, covering three fundamental static analysis tasks, and providing finer-grained diagnostic information than end-to-end evaluations.

**Key Insight**: The paper is grounded in three fundamental concepts from program analysis—data dependency (def-use chains), control dependency (execution path guards), and information flow (explicit and implicit propagation)—to construct a multi-task benchmark spanning C/C++, Java, and Python.

**Core Idea**: Rather than indirectly inferring capabilities from end-to-end results, CoRe directly examines whether LLMs have mastered the fundamental skills of code semantic reasoning—dependency relation reasoning in static analysis.

## Method

### Overall Architecture

The CoRe benchmark construction consists of four stages: **program sampling** → **semi-automatic annotation** → **task design** → **semantics-aware diverse sampling**, ultimately producing 12,553 task instances (including 1,584 instances in the CoRe Lite subset).

### Key Designs

#### 1. Multi-Language Program Sampling
- **Data Sources**: Programs are sampled from two sources: CodeNet (a large-scale competitive programming dataset) and Google Code Jam (GCJ, expert-level algorithmic problem-solving code)
- **Scale**: 180 programs total, with 60 each in C/C++, Java, and Python
- **Complexity Balancing**: Programs are divided into four buckets by lines of code (21–40 / 41–60 / 61–80 / 81–100 LoC), with uniform sampling from each bucket
- **Analysis Scope**: Focus is restricted to intra-procedural analysis; functions invoking non-library functions are excluded to avoid incomplete prompt information caused by cross-function reasoning

#### 2. Semi-Automatic Annotation Pipeline
- **Initial Annotation**: Custom static analysis tools based on tree-sitter are developed to automatically generate initial annotations for data and control dependencies
- **Manual Verification**: Two authors with over five years of program analysis experience independently verify annotations, with conflicts resolved by a third party
- **Quality Assurance**: Inter-annotator agreement reaches 87.5%, yielding 6,306 annotated variables and 48,050 lines of annotation data
- **Necessity**: The authors demonstrate that existing program analysis tools cannot automatically extract comprehensive semantic properties accurately across languages, making human involvement indispensable

#### 3. Three Core Task Categories
- **Data Dependency**: Determines whether the value of variable $b$ depends on variable $a$ through assignment chains, corresponding to def-use chains in program analysis. Application example: taint analysis of CVE-2017-5638, where an untrusted HTTP header propagates to a sensitive operation via data dependency
- **Control Dependency**: Determines whether the execution of statement $\ell_2$ is governed by conditional statement $\ell_1$ (i.e., one branch guarantees execution of $\ell_2$ while another may not). Application example: fuzzing path reachability analysis of CVE-2022-26129
- **Information Flow**: Determines whether a variable's value influences another variable through explicit assignment (data dependency) or implicit control flow (control dependency). This is the most challenging task, requiring simultaneous reasoning over both dependency types. Application example: program slicing and fault localization for null pointer exceptions

#### 4. Two Query Types
- **Pairwise Query**: Given two program elements, determines whether a specific dependency relation exists; if so, requires generating a complete trace from source to target (a transitive dependency sequence)
- **Target-Centric Query**: Given a target element, enumerates all source elements within the function that have a specified dependency relation with the target (returning a complete, unordered set)

#### 5. Semantics-Aware Diverse Sampling
- At most 5 targets are sampled per program, with at most 5 positive and 5 negative examples constructed per target
- **Diversity Guarantee**: Avoids overlapping traces or repeated contexts to ensure structural coverage
- **Reasoning Complexity Guarantee**: Targets with non-trivial dependency structures are selected
- **Negative Sample Design**: Structurally "plausible" element pairs (e.g., within the same code block or syntactically proximate) that have no actual dependency are selected, preventing models from exploiting positional heuristics

### Evaluation Design
- **Prompt**: Includes detailed definitions, output format specifications, and 5–7 synthetic small-program examples with step-by-step reasoning explanations
- **Evaluation Metrics**: Dependency classification (Precision/Recall/F1), trace quality (Correct Trace Rate), and source enumeration (Exact Match)

## Key Experimental Results

### Main Results: Overall Performance of 10 Models on CoRe Lite

| Model | Reasoning | Classification F1 (%) | Trace CT (%) | Enumeration EM (%) |
|---|---|---|---|---|
| Gemini 2.5 Pro | ✓ | **91.74** | **84.02** | **50.25** |
| GPT o3 | ✓ | 92.56 | 72.80 | 42.61 |
| GPT o4-mini | ✓ | 86.74 | 60.43 | 32.89 |
| DeepSeek R1 | ✓ | 86.18 | 58.36 | 31.82 |
| Claude 3.7 | ✓ | 82.07 | 57.13 | 25.82 |
| Qwen3 235B | ✓ | 80.31 | 52.26 | 23.30 |
| Claude 3.5 | ✗ | 77.27 | 49.43 | 19.13 |
| DeepSeek V3 | ✗ | 75.80 | 41.08 | 13.38 |
| GPT 4o | ✗ | 74.16 | 43.56 | 14.52 |
| Llama 3.1 405B | ✗ | 68.93 | 28.48 | 3.28 |

### Per-Task Detailed Performance (Gemini 2.5 Pro / GPT o3 / DeepSeek R1)

| Task Type | Metric | Gemini 2.5 Pro | GPT o3 | DeepSeek R1 |
|---|---|---|---|---|
| Data Dependency | Classification F1 | 88.53 | 93.24 | 83.29 |
| Data Dependency | Trace CT | 90.38 | 86.23 | 67.31 |
| Data Dependency | Enumeration EM | 49.43 | 41.89 | 38.88 |
| Control Dependency | Classification F1 | 92.49 | 92.11 | 92.28 |
| Control Dependency | Trace CT | 92.26 | 77.52 | 66.62 |
| Control Dependency | Enumeration EM | 75.66 | 70.90 | 48.37 |
| Information Flow | Classification F1 | 94.79 | 92.13 | 83.59 |
| Information Flow | Trace CT | 68.66 | 52.37 | 39.58 |
| Information Flow | Enumeration EM | 26.73 | 15.61 | 7.12 |

### Factor Analysis

| Factor | Performance Change | Notes |
|---|---|---|
| Function length 21–40 → 81–100 LoC | Trace CT drops 8.9–15.8% | Longer functions introduce more complex dependency chains and more tokens |
| Increasing number of conditionals | Trace CT drops 24.4–44.6% | Complex control flow substantially increases reasoning difficulty |
| Presence of early exits (return/break) | Enumeration EM drops up to 21.7% | Non-linear control flow disrupts models' sequential reasoning |
| Reverse dependency (source after target) | Classification F1 drops up to 49.5% | Exposes LLMs' left-to-right reasoning bias |
| Trimmed context (target function only) | Performance improves up to 5.7% | Models struggle to focus in the presence of redundant context |
| Joint classification + trace generation | Precision decreases, Recall increases | Models tend to predict "dependency exists," increasing false positives |
| Few-shot retrieval augmentation | No improvement or degradation | Standard retrievers fail to capture structural/semantic similarity |

### Key Findings

1. Reasoning models consistently outperform non-reasoning models, with performance gaps of 5.2–31.5%; Gemini 2.5 Pro achieves the best overall performance.
2. Dependency classification is relatively easier (F1 up to 92.56%), whereas trace generation and source enumeration are highly challenging—even the best model achieves only 50% on enumeration.
3. **Information flow is the most difficult task**: even with a classification F1 as high as 94.79%, the trace correct rate for Gemini is only 68.66%, and lower for other models.
4. Control dependency enumeration is relatively easier (an average of only 4.0 line-level sources), whereas data dependency and information flow average 7.7 and 17.0 variable-level sources respectively, making enumeration substantially harder.
5. Reverse dependencies expose a critical weakness of LLMs: Claude 3.7's F1 drops by as much as 49.5% on reverse dependency instances.

## Highlights & Insights

- **Fills an Evaluation Gap**: CoRe is the first benchmark to systematically evaluate LLMs on core static analysis capabilities, shifting evaluation granularity from "whether code is fixed" to "whether dependency relations are understood."
- **High-Quality Manually Validated Data**: An inter-annotator agreement of 87.5% and a semi-automatic pipeline (tree-sitter + dual cross-validation + arbitration) ensure the soundness and completeness of annotations.
- **Sophisticated Semantics-Aware Sampling Strategy**: Negative samples are selected as structurally "plausible" element pairs, effectively preventing models from exploiting positional heuristics as shortcuts.
- **Important Finding on Reverse Dependencies**: The results reveal that LLMs exhibit a strong left-to-right sequential reasoning bias and are unable to construct abstract dependency graphs for bidirectional analysis. This finding has significant implications for understanding the nature of model reasoning.
- **Longer Context Hurts Performance**: Providing complete files results in lower performance than providing only the target function (up to 5.7% difference), indicating that models struggle to focus on relevant information in long contexts—in contrast to traditional analysis tools, which naturally isolate scopes.
- **Clear Difficulty Gradient Across Tasks**: Classification > Trace > Enumeration, providing a well-defined difficulty ladder to guide future improvement.

## Limitations & Future Work

1. **Intra-procedural Analysis Only**: Cross-function call chain reasoning is excluded, yet dependencies in real software frequently span function boundaries; future work should extend to inter-procedural analysis.
2. **Narrow Program Sources**: Both CodeNet and GCJ consist of competitive programming code, whose style and complexity differ significantly from real industrial projects involving class inheritance, design patterns, and large codebases.
3. **Absence of Complex Memory Models**: Pointer, alias, and indirect memory access semantics—particularly critical for C/C++—are not evaluated; these represent the most challenging aspects of static analysis.
4. **Room for Improvement on Information Flow**: Correct rates are generally below 50%; it is worth exploring whether step-wise reasoning frameworks for LLMs (e.g., first reasoning about data dependency, then control dependency, and finally combining both) would be beneficial.
5. **Failure of Retrieval Augmentation**: Standard sparse/dense retrievers fail to capture structural similarity between tasks; program-structure-aware retrieval methods are needed.
6. **Limited Scale**: With 180 programs and 12K task instances, the benchmark is of moderate scale; future work could expand to larger and more diverse codebases.

## Related Work & Insights

- **CRUXEval** evaluates only input–output behavior in Python (essentially dynamic analysis); CoRe targets static analysis across three languages, making the two approaches complementary.
- **CRQBench** automatically generates C++ code reasoning questions from GitHub PR comments, but automatic generation is error-prone and relatively coarse-grained; CoRe's manually validated data offers higher quality.
- **LLMDFA / IRIS**: Practical systems that employ LLMs as static analyzers—CoRe can serve as a litmus test for evaluating the foundational capabilities of such approaches.
- **REval**: Evaluates runtime behaviors such as code coverage prediction; while its focus differs from CoRe's static analysis orientation, the methodological approach is similar.
- **Implications for the LLM-as-Analyzer Paradigm**: Models achieve extremely low EM on enumeration tasks (at best 50%), indicating that using LLMs directly for complete dependency analysis remains impractical and that integration with traditional tools is necessary.

## Rating

- **Novelty**: ⭐⭐⭐⭐ CoRe is the first benchmark to evaluate LLMs' static analysis reasoning capabilities at fine granularity, filling an important gap; however, the ceiling for methodological innovation in benchmark papers is inherently limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 10 models (including 6 reasoning models), 3 task categories, 3 programming languages, multi-dimensional factor analysis, and ablation studies—an exceptionally comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly established through real-world CVE case studies; task definitions are mathematically rigorous; qualitative analysis is thorough and practically meaningful.
- **Value**: ⭐⭐⭐⭐ The benchmark provides substantial reference value for understanding and improving LLMs' code reasoning capabilities; findings such as the reverse dependency bias are insightful. However, the gap between the benchmark and practical applications remains considerable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)
- [\[NeurIPS 2025\] CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning](codecrash_exposing_llm_fragility_to_misleading_natural_language_in_code_reasonin.md)
- [\[NeurIPS 2025\] FractalBench: Diagnosing Visual-Mathematical Reasoning Through Recursive Program Synthesis](fractalbench_diagnosing_visual-mathematical_reasoning_through_recursive_program_.md)
- [\[NeurIPS 2025\] Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward](principled_fine-tuning_of_llms_from_user-edits_a_medley_of_preference_supervisio.md)
- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)

</div>

<!-- RELATED:END -->
