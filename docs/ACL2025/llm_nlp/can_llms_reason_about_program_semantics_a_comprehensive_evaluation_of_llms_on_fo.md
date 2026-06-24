---
title: >-
  [Paper Note] Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference
description: >-
  [ACL 2025][LLM (Other)][Program Semantic Reasoning] This paper proposes the FormalBench benchmark to systematically evaluate the program semantics reasoning capabilities of LLMs through formal specification inference tasks. It finds that while LLMs perform well on simple control flows, they struggle with complex structures like loops. Additionally, self-repair prompts are designed to improve the success rate by 25%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Program Semantic Reasoning"
  - "Formal Verification"
  - "Specification Inference"
  - "Code Understanding"
  - "Self-Repair Prompting"
date: 2026-05-08
content_hash: 6ca85a8feaf1cdf1
---

# Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference

**Conference**: ACL 2025  
**arXiv**: [2503.04779](https://arxiv.org/abs/2503.04779)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Program Semantic Reasoning, Formal Verification, Specification Inference, Code Understanding, Self-Repair Prompting

## TL;DR

This paper proposes the FormalBench benchmark to systematically evaluate the program semantics reasoning capabilities of LLMs through formal specification inference tasks. It finds that while LLMs perform well on simple control flows, they struggle with complex structures like loops. Additionally, self-repair prompts are designed to improve the success rate by 25%.

## Background & Motivation

**Background**: Large language models (LLMs) are increasingly applied to automate programming tasks, including code generation, code completion, and bug fixing. However, most evaluations focus on the ability of LLMs to generate "correct-looking" code, and there is insufficient research on whether models truly understand program semantics—the program's behavior across all possible execution paths.

**Limitations of Prior Work**: Existing code capability evaluations (such as HumanEval and MBPP) mainly verify the correctness of generated code by running test cases, which presents two fundamental issues: (1) test cases only cover limited inputs and cannot guarantee program correctness under all circumstances; (2) passing test cases does not imply that the LLM truly understands the semantics of the program, as the model might simply perform pattern matching on correct code templates. Formal verification requires program specifications, such as preconditions, postconditions, and loop invariants, which precisely describe the expected behavior of programs.

**Key Challenge**: Code evaluation methods based on test cases cannot accurately measure the depth of LLMs' understanding of program semantics. A model that passes all test cases may have zero understanding of the actual behavior of the program, relying solely on superficial pattern matching.

**Goal**: Design a benchmark specifically to evaluate the program semantics reasoning capability of LLMs by testing their understanding of program behavior through the generation of formal program specifications.

**Key Insight**: The authors select "formal specification inference" as the evaluation task. This task requires models to both comprehensively reason about all possible execution paths of a program and generate precise expressions conforming to formal syntax and semantics, representing a highly demanding program understanding task.

**Core Idea**: Use formal specification inference as a proxy task to evaluate the program semantics reasoning capabilities of LLMs, and construct FormalBench, a benchmark covering program structures of varying complexities.

## Method

### Overall Architecture

The design and evaluation of the FormalBench benchmark consist of three parts: (1) **Benchmark Construction**: Collect C programs covering different control flow complexities and manually write correct formal specifications as ground truth for each program; (2) **Evaluation Setup**: Design multiple prompting strategies to evaluate the specification inference capabilities of LLMs; (3) **In-depth Analysis**: Analyze failure modes, robustness, and the effectiveness of self-repair.

### Key Designs

1. **FormalBench Benchmark Construction**:

    - **Function**: Provide a benchmark for formal specification inference that covers different program complexities.
    - **Mechanism**: The programs in the benchmark cover four complexity levels: (a) **pure sequential structure**: no branches or loops, only assignment statements; (b) **conditional branching**: contains conditional statements like if-else; (c) **simple loops**: contains single-level loops with relatively simple loop bodies; (d) **complex loops**: nested loops, loop conditions depending on multiple variables, etc. For each program, the inferred specifications include preconditions, postconditions, and loop invariants. All specifications are written using ACSL (ANSI/ISO C Specification Language) syntax.
    - **Design Motivation**: Different program structures impose different demands on reasoning capabilities—sequential structures only require linear reasoning, whereas loops require inductive reasoning to identify invariants, which is a fundamental challenge.

2. **Multi-Strategy Prompting Evaluation**:

    - **Function**: Systematically evaluate the impact of different prompting strategies on the specification inference capability of LLMs.
    - **Mechanism**: Multiple prompting strategies are designed: (a) **Zero-shot**: directly provide the program and request the generation of specifications; (b) **Few-shot**: provide several examples of "program -> specification"; (c) **Chain-of-Thought**: require the model to analyze the program's execution flow first, then generate specifications; (d) **Hoare logic-based prompting**: introduce the concept of Hoare triples in the prompt to guide the model to think in a formal reasoning style. Each strategy is evaluated across multiple mainstream LLMs, including GPT-4, Claude, LLaMA, CodeLlama, etc.
    - **Design Motivation**: Different prompting strategies correspond to different ways of guiding reasoning. Systematic comparison can reveal the reasoning bottlenecks of LLMs.

3. **Self-Repair Prompting**:

    - **Function**: Enable LLMs to repair their generated specifications after a failed initial attempt.
    - **Mechanism**: When the specification generated by the LLM fails the verification checks of a formal verification tool (such as Frama-C), the error message returned by the verifier (e.g., "loop invariant does not hold after the second iteration") is fed back to the LLM to guide revision. This process can be iterated over multiple rounds—re-verifying after each repair, and continuing if it fails. Usually, a maximum of 3 rounds of repair are allowed in the experiments.
    - **Design Motivation**: Formal verification tools provide precise feedback signals, unlike general code generation which only has coarse-grained pass/fail signals. Leveraging this precise feedback to guide repair is a natural approach.

### Evaluation Metrics

Two main metrics are utilized: (1) **Consistency**: whether the generated specifications can pass the checks of formal verification tools, meaning the specifications are consistent with the program behavior; (2) **Completeness**: whether the generated specifications are strong enough to completely describe the program behavior, rather than being a trivially true specification.

## Key Experimental Results

### Main Results

Consistent pass rates of different LLMs across various program complexities:

| Model | Sequential Structure | Conditional Branching | Simple Loops | Complex Loops | Average |
|------|---------|---------|---------|---------|------|
| GPT-4 | 92.3% | 85.7% | 61.2% | 38.5% | 69.4% |
| Claude-3 | 89.1% | 82.3% | 57.8% | 34.2% | 65.9% |
| GPT-3.5-Turbo | 78.5% | 68.2% | 42.1% | 22.3% | 52.8% |
| CodeLlama-34B | 75.3% | 64.8% | 38.5% | 18.7% | 49.3% |
| LLaMA-3-70B | 82.1% | 73.5% | 48.2% | 28.1% | 58.0% |
| DeepSeek-Coder-33B | 80.7% | 71.2% | 46.5% | 26.8% | 56.3% |

### Ablation Study

| Prompt Strategy | GPT-4 Average Pass Rate | Relative Gain over Zero-shot | Description |
|---------|----------------|------------------|------|
| Zero-shot | 60.2% | — | Baseline |
| Few-shot (3 examples) | 65.8% | +5.6% | Examples help understand the format |
| Chain-of-Thought | 67.3% | +7.1% | Step-by-step reasoning is effective |
| Hoare Logic Prompting | 69.4% | +9.2% | Formal thinking guidance is most effective |
| Self-repair (1 round) | 78.1% | +17.9% | Verification feedback is highly effective |
| Self-repair (3 rounds) | 85.2% | +25.0% | Multi-round repair continues to improve |

### Key Findings

- **Loops are the fundamental bottleneck**: LLMs perform excellently on sequential structures (>90%), but their performance drops sharply on complex loops (<40%). Loops require inductive reasoning to discover invariants, which represents a fundamental challenge for LLMs.
- **Self-repair prompting is highly effective**: Utilizing the precise feedback from formal verifiers boosts the success rate by 25%. This indicates that the "first attempt" of LLMs might be close to correct, and verification feedback helps them fix detailed errors.
- **Lack of robustness**: After applying semantic-preserving transformations to programs (such as variable renaming or equivalent code rewriting), the outputs of LLMs can change completely. This suggests that the models might perform superficial pattern matching rather than genuine semantic reasoning.
- **Common failure modes**: (a) Under-strength loop invariants—generating invariants like `true`, which are consistent but useless; (b) omission of variable relationships—ignoring relationships among multiple variables; (c) syntax errors—generating ACSL expressions that do not conform to grammar rules.
- **Model scaling helps but does not resolve the core issue**: GPT-4 still achieves only 38.5% on complex loops, showing that scaling is not a silver bullet.

## Highlights & Insights

- Selecting formal specification inference as the evaluation task is ingenious—it requires deep program understanding whilst offering clean correctness standards (verifiable via formal verifiers), avoiding the inherent unreliability of alternative appraisal metrics.
- The self-repair mechanism exploits the unique advantage of formal verification: precise error feedback. This is rarely leveraged in the field of code generation.
- Analysis across different complexity levels reveals that the "cliff-like drop" in LLM reasoning capability occurs at loop structures, offering critical clues for mapping the boundary of LLMs' reasoning.

## Limitations & Future Work

- The benchmark size is relatively small, with programs being primarily textbook-style simple programs, which has a gap compared to industry-grade code.
- Only C-language ACSL specifications were tested, without involving other formal specification languages (e.g., JML, Dafny).
- The number of self-repair rounds is limited (3 rounds); whether more rounds can continue to yield improvements remains unclear.
- Future exploration directions: framing formal reasoning capability as a training objective to reinforce LLMs, or taking hybrid approaches combining symbolic reasoners.

## Related Work & Insights

- Complementary to code generation benchmarks like HumanEval and MBPP—while they evaluate the capability of "generating correct code", FormalBench evaluates the capability of "understanding program behavior".
- Related to formal reasoning efforts like Lemur, though FormalBench focuses more on evaluation rather than methodological improvements.
- The success of the self-repair mechanism inspires the possibility of leveraging tool feedback to augment LLMs in broader formal verification scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Systematically evaluates LLMs on formal specification inference for the first time, with a clever and profound task design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation covering multi-model and multi-strategy setups, with in-depth ablation and error analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly defined problems and reasonable experimental design.
- **Value**: ⭐⭐⭐⭐⭐ — Exposes the fundamental limitations of LLMs in program semantics reasoning, and its acceptance to the main conference of ACL 2025 demonstrates its significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Language Models Reason about Individualistic Human Values and Preferences?](can_language_models_reason_about_individualistic_human_values_and_preferences.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs](can_llms_help_uncover_insights_about_llms_a_large-scale_evolving_literature_anal.md)
- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)

</div>

<!-- RELATED:END -->
