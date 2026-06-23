---
title: >-
  [Paper Note] AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms
description: >-
  [ICML 2026][Code Intelligence][verified code generation] AlgoVeri constructs a strictly aligned benchmark for verified code generation of classical algorithms across Dafny, Verus, and Lean. It demonstrates that current LLMs still face significant gaps in handling complex global invariants, system-level constraints, and explicit proof search, with success rates in Lean and Ve
tags:
  - ICML 2026
  - Code Intelligence
  - verified code generation
  - Dafny
  - Verus
  - Lean
date: 2026-05-08
content_hash: 3349e29652d5393c
---
# AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms

**Conference**: ICML 2026  
**arXiv**: [2602.09464](https://arxiv.org/abs/2602.09464)  
**Code**: https://github.com/haoyuzhao123/algoveri  
**Area**: Code Intelligence / Formal Verification / Verified Code Generation  
**Keywords**: verified code generation, formal verification, Dafny, Verus, Lean, algorithm benchmark  

## TL;DR
AlgoVeri constructs a strictly aligned benchmark for verified code generation of classical algorithms across Dafny, Verus, and Lean. It demonstrates that current LLMs still face significant gaps in handling complex global invariants, system-level constraints, and explicit proof search, with success rates in Lean and Verus being substantially lower than those in Dafny.

## Background & Motivation
**Background**: While LLMs can generate code from natural language, passing unit tests does not guarantee program correctness for all inputs. Formal verification requires models to generate implementations, specifications, and proof artifacts, allowing a verifier to prove that the code satisfies the contract. This is more rigorous than test-driven benchmarks like HumanEval or MBPP.

**Limitations of Prior Work**: Existing vericoding benchmarks mostly cover a single tool or combine different tasks across different languages. This makes scores incomparable: success in Dafny might only involve array bounds checks, whereas success in Lean might require a complete proof of functional correctness. Without aligned specifications, it is impossible to determine whether performance differences stem from model capability or task difficulty.

**Key Challenge**: The community needs to know if LLMs can truly verify the global properties of "classical algorithms." However, current benchmarks are either too simple or unfair across tools. Complex algorithms like Red-Black Trees, Maximum Flow, or Tarjan's strongly connected components require maintaining global invariants, ghost states, and termination proofs, which are precisely where current models expose reasoning deficiencies.

**Goal**: The authors aim to decouple problem difficulty, verification language features, and model reasoning ability for evaluation. The approach involves collecting 77 textbook-style classical algorithm tasks and writing semantically aligned specifications in Dafny, Verus, and Lean, then evaluating closed-source and open-source models using a consistent iterative repair protocol.

**Key Insight**: Instead of pursuing an ultra-large-scale problem set, the paper emphasizes "alignment" and "algorithmic depth." Every task requires the same algorithm semantics to be expressed in three verification systems. Finally, a semantic validator is used to exclude cases where models pass via `assume`, `sorry`, or degenerate algorithms.

**Core Idea**: Use a three-language classical algorithm benchmark with strictly parallel specifications to advance LLM verified code generation from "local safety checks" to "global algorithm correctness" evaluation.

## Method
AlgoVeri is essentially a benchmark construction and evaluation pipeline. Its key is not proposing a new generative model, but rather fixing the tasks, specifications, verification systems, iterative repair, and semantic filtering so that performance differences between models and languages are interpretable.

### Overall Architecture
The input is a classical algorithm problem with its natural language description. The authors provide function signatures, preconditions, postconditions, and necessary helper definitions for Dafny, Verus, and Lean, requiring specifications in all three languages to express the same set of semantic constraints. During evaluation, the model receives the problem description and a formal specification in a specific language to generate implementation and proof artifacts. If the verifier reports an error, the model can read the error message and perform repairs over multiple rounds. A solution must first be accepted by the compiler/verifier and then pass an LLM semantic filter to ensure it actually implements the specified algorithm rather than bypassing the specification.

### Key Designs

**1. Specification Alignment Across Verification Systems: Making Cross-Language Scores Comparable**

A fundamental issue with existing vericoding benchmarks is that scores across tools are incomparable. AlgoVeri addresses this by manually writing and cross-reviewing function signatures, preconditions, postconditions, and helper definitions for the same algorithm task in Dafny, Verus, and Lean, forcing the three systems to express the same set of semantic constraints. For example, for the Longest Increasing Subsequence (LIS) task, it is not enough for the result to "fall within the array range"; instead, it is uniformly required that "there exists an increasing subsequence of the corresponding length, and no longer valid subsequence exists." By fixing the problem difficulty, observed cross-language gaps can be cleanly attributed to toolchain abstraction levels, model proof search capabilities, and linguistic syntax barriers.

**2. Global Invariant Tasks at the Classical Algorithm Level: Pushing Evaluation to Global Reasoning**

Benchmarks like HumanEval/MBPP only test single-step or local properties, which overestimates model capabilities. AlgoVeri uses 77 textbook-level classical algorithms—covering Heaps, Segment Trees, Red-Black Trees, Bellman-Ford, Edmonds-Karp, Gaussian Elimination, etc.—requiring models to generate loop invariants, ghost states, auxiliary lemmas, termination proofs, or Lean tactic scripts. The correctness of these problems depends on global properties across loops and data structures, targeting the reasoning weaknesses of current models in graph reachability invariants and complex state updates.

**3. Semantic Filtering After Verification: Distinguishing "True Verification" from "Exploiting Loopholes"**

Formal verification only guarantees that the "implementation satisfies the given specification." If the specification has holes, the model can use vacuous proofs, statements like `assume`/`sorry`, or implement a simpler degenerate algorithm to trick the verifier. AlgoVeri adds an LLM judge semantic check after the verifier passes, defining two metrics: *Compiler Verified* (accepted by the verifier) and *Full Mark* (accepted by the verifier and confirmed by the judge as the correct algorithm). The difference is defined as the *algorithmic fidelity gap*. This filter ensures the benchmark evaluates both "verification correctness" and "faithful implementation."

### Loss & Training
The paper does not train models but uses a unified inference-time refinement protocol. In the main experiment, most models use 15 repair rounds, while GPT-5.3 Codex uses 8 rounds; open-source models are additionally evaluated with a multi-sample budget of $10\times15$. In each round, the model modifies the previous code based on compiler/verifier errors. The authors also compare depth and width: *depth* involves iterative repair along the same chain, while *width* involves sampling multiple independent candidates in parallel, comparing gains under equal compute budgets.

## Key Experimental Results

### Main Results
The main table reports Compiler Verified and Full Mark (after semantic filtering). The following shows Full Mark results and the drop from Verified to Full Mark.

| Model / Budget | Dafny Verified / Full Mark | Verus Verified / Full Mark | Lean Verified / Full Mark | Main Implication |
|-------------|----------------------------|-----------------------------|----------------------------|----------|
| GPT-5.3 Codex, $1\times8$ | 49.35 / 42.86 | 14.29 / 11.69 | 23.38 / 11.69 | Codex is strongest in Dafny, but has a large semantic gap in Lean |
| Gemini-3 Flash, $1\times15$ | 55.84 / 40.26 | 25.97 / 24.68 | 9.09 / 7.79 | Frontier models can utilize repair rounds, but Lean remains extremely difficult |
| GPT-5 mini, $1\times15$ | 41.56 / 30.47 | 7.79 / 6.49 | 5.19 / 5.19 | Small models quickly lose ground in Verus/Lean |
| GPT-OSS-120B, $1\times15$ | 21.04 / 13.51 | 7.66 / 7.01 | 12.60 / 7.01 | OS model base capabilities are significantly lower than frontier models |
| GPT-OSS-120B, $10\times15$ | 44.16 / 28.57 | 12.99 / 10.39 | 25.97 / 14.29 | Increasing parallel budget helps, but still falls short of Gemini's Dafny performance |
| Qwen3-235B, $10\times15$ | 32.47 / 29.87 | 12.99 / 12.99 | 6.49 / 6.49 | Dafny improves with budget; Lean remains low |
| Devstral-2-123B, $10\times15$ | 33.77 / 18.18 | 14.29 / 12.99 | 6.49 / 6.49 | Semantic filtering reveals a large fidelity gap in Dafny |

### Ablation Study
The analysis section focuses on comparing deep repair and wide sampling for test-time compute and breaks down failure modes.

| Analysis Item | Observation | Explanation |
|--------|----------|------|
| Gemini-3 Flash Repair Depth | Pass rate for Dafny and Verus increases over 15 repair rounds; Dafny increases nearly 3x | Frontier models treat verifier feedback as useful reasoning signals |
| GPT-OSS-120B Repair Depth | Saturates after 3 to 4 rounds with minimal further gains | Current OS models behave more like "resampling with error context," lacking multi-round proof repair capability |
| GPT-OSS Equal Compute | Repair curve does not significantly exceed the parallel sampling baseline; deep repair might be worse | For OS models, width is more cost-effective than depth |
| Dafny Error Trajectory | Syntax/type errors drop quickly; verification failures remain the primary cause | High-level abstraction and SMT automation allow models to focus on logic |
| Verus Error Trajectory | Syntax/type errors persist throughout the 15 rounds | Rust macros, ownership, and integer type casting form syntax/type barriers |
| Lean Error Trajectory | Hallucinations and verification errors dominate | Models struggle with both finding the right lemma/tactic and constructing correct proofs |

### Key Findings
- "Verified" does not equal "implemented the specified algorithm." Gemini-3 Flash drops from 55.84% Verified to 40.26% Full Mark in Dafny, indicating that algorithmic fidelity must be evaluated separately.
- Language hierarchy is evident: Dafny's automation and mathematical abstraction are most friendly; Verus is closer to system code but has heavy syntax/type burdens; Lean has a small trusted kernel but the largest proof search space.
- Difficulties are concentrated in advanced data structures and graph algorithms. Tasks requiring ghost states, global reachability, balance invariants, or max-flow optimality proofs remain major bottlenecks.

## Highlights & Insights
- The value of this paper lies in making the benchmark "fair." By porting the same algorithm specifications to three verification ecosystems, cross-language performance differences have an interpretable basis.
- The Full Mark metric is critical. While many code generation benchmarks only look at pass rates, AlgoVeri shows that formal verification can be contaminated by degenerate implementations or specification loopholes, requiring a separation of "proof success" and "algorithmic fidelity."
- The depth vs. width analysis provides insights for agentic coding: not all models are suitable for multi-round repair. For models with insufficient repair capability, blindly lengthening the feedback chain may be more wasteful than parallel sampling.

## Limitations & Future Work
- Semantic filtering relies on an LLM judge. While it catches obvious degenerate algorithms and cheating, misjudgments can occur; designing mechanically checkable algorithmic fidelity conditions for more tasks would be ideal.
- Specifications are manually written and reviewed by experts, ensuring high quality but at great cost. Expanding to thousands of tasks will require semi-automated specification migration and formal well-formedness checking tools.
- AlgoVeri currently evaluates generation capability and does not systematically compare different agent architectures like specialized training, retrieval-augmented proof, or search tree planning. It serves as a reliable measuring tool rather than a complete solution.
- Low scores in Verus and Lean reflect both model deficiencies and scarce training data/tooling experience. Future improvements require joint progress in language ecosystems, error message design, and model training.

## Related Work & Insights
- **vs VeriCoding**: VeriCoding is larger in scale but lacks task and specification alignment across languages; AlgoVeri is smaller but provides a fairer comparison of the real difficulty between Dafny, Verus, and Lean.
- **vs CLEVER / Verina**: CLEVER and Verina focus on Lean and emphasize anti-cheating or proof-gap diagnostics; AlgoVeri emphasizes cross-system alignment and complex classical algorithms.
- **vs DafnyBench**: DafnyBench leans toward annotation/hint generation; AlgoVeri requires generating complete implementation and proof artifacts from specifications, making it closer to end-to-end verified code generation.
- **vs HumanEval / MBPP**: These only test limited cases; AlgoVeri requires the verifier to prove correctness for all inputs, making it better for evaluating the limits and shortcomings of reliable code generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The strictly aligned benchmark across three verification systems for classical algorithms is highly distinctive and solves the problem of incomparable cross-language scores.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers closed/open-source models, three verification ecosystems, repair rounds, equal-budget analysis, and failure mode breakdown.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and strong motivation; some model naming and versioning settings require contextual attention.
- Value: ⭐⭐⭐⭐⭐ Useful for research in code intelligence, formal verification, and agentic repair, distinguishing between true reasoning progress and superficial verification pass rates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](../../ACL2025/code_intelligence/codedpo_code_alignment.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](../../ACL2025/code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] GALLa: Graph Aligned Large Language Models for Improved Source Code Understanding](../../ACL2025/code_intelligence/galla_graph_aligned_large_language_models.md)
- [\[ACL 2025\] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](../../ACL2025/code_intelligence/feabench_repo_code_gen.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](../../ACL2025/code_intelligence/dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)

</div>

<!-- RELATED:END -->
