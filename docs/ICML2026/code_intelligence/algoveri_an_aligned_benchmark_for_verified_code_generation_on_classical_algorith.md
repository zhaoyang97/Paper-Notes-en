---
title: >-
  [Paper Note] AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms
description: >-
  [ICML 2026][Code Intelligence][verified code generation] AlgoVeri constructs a benchmark for verified code generation of classical algorithms strictly aligned across Dafny, Verus…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "verified code generation"
  - "formal verification"
  - "Dafny"
  - "Verus"
  - "Lean"
  - "algorithm benchmark"
date: 2026-05-08
content_hash: ad5a20616bc98c6e
---

# AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms

**Conference**: ICML 2026  
**arXiv**: [2602.09464](https://arxiv.org/abs/2602.09464)  
**Code**: https://github.com/haoyuzhao123/algoveri  
**Area**: Code Intelligence / Formal Verification / Verified Code Generation  
**Keywords**: verified code generation, formal verification, Dafny, Verus, Lean, algorithm benchmark  

## TL;DR
AlgoVeri constructs a benchmark for verified code generation of classical algorithms strictly aligned across Dafny, Verus, and Lean. It demonstrates that current LLMs still face significant gaps in identifying complex global invariants, system-level constraints, and explicit proof searches, with success rates in Lean and Verus being substantially lower than in Dafny.

## Background & Motivation
**Background**: LLMs have achieved natural language-to-code generation, but passing unit tests is not equivalent to program correctness across all inputs. Formal verification requires models to generate implementations, specifications, and proof artifacts, allowing a verifier to prove that the code satisfies a contract. This makes it more rigorous than test-driven benchmarks like HumanEval or MBPP.

**Limitations of Prior Work**: Existing vericoding benchmarks often cover only a single tool or combine disparate tasks across different languages. This results in incomparable scores: success in Dafny might only imply safety checks for array bounds, whereas success in Lean might require a full proof of functional correctness. Without aligned specifications, it is impossible to determine whether cross-language performance differences stem from model capability or task difficulty.

**Key Challenge**: The community needs to understand whether LLMs can truly verify the global properties of "classical algorithms." However, existing benchmarks are either too simple or unfair across tools. Complex algorithms such as Red-Black Trees, Max-Flow, and Tarjan’s Strongly Connected Components require maintaining global invariants, ghost states, and termination proofs, which are precisely the areas where current model reasoning falls short.

**Goal**: The authors aim to decouple evaluation across task difficulty, verification language features, and model reasoning capabilities. Specifically, they collect 77 textbook-style classical algorithm tasks, write semantically aligned specifications in Dafny, Verus, and Lean, and evaluate both closed-source and open-source models using the same iterative refinement protocol.

**Key Insight**: Rather than pursuing a massive task library, the paper emphasizes "alignment" and "algorithmic depth." Each task requires the same algorithmic semantics to be expressed in three verification systems. Finally, a semantic validator is used to prevent models from bypassing requirements through `assume`, `sorry`, or degenerate algorithms.

**Core Idea**: Push LLM verified code generation from "local safety checks" to "global algorithmic correctness" using a three-language classical algorithm benchmark with strictly parallel specifications.

## Method
AlgoVeri is essentially a benchmark construction and evaluation pipeline. Rather than proposing a new generative model, it fixes the tasks, specifications, verification systems, iterative refinement, and semantic filtering to make performance differences across models and languages interpretable.

### Overall Architecture
The input consists of a classical algorithm problem and its natural language description. The authors provide function signatures, preconditions, postconditions, and necessary helper definitions for Dafny, Verus, and Lean, ensuring the specifications in all three languages express the same set of semantic constraints. During evaluation, the model receives the description and many-sorted formal specifications to generate implementation and proof artifacts. If the verifier fails, the model can read error messages and fix the code over multiple rounds. A solution must first be accepted by the compiler/verifier and then undergo LLM semantic filtering to confirm it implements the specified algorithm rather than bypassing the specification.

### Key Designs
1.  **Specification Alignment Across Verification Systems**:
    - **Function**: Ensures the same task in Dafny, Verus, and Lean examines the same algorithmic properties rather than varying in difficulty levels.
    - **Mechanism**: The authors manually authored and reviewed helper definitions, preconditions, and postconditions to ensure semantic consistency across systems. For example, the LIS (Longest Increasing Subsequence) task does not just check if the result is within array bounds; it uniformly requires proving the existence of an increasing subsequence of a certain length and the non-existence of a longer valid subsequence.
    - **Design Motivation**: Without alignment, cross-language scores cannot be explained. AlgoVeri fixes "task difficulty variance," ensuring that observed gaps reflect toolchain abstractions, model proof search, and linguistic syntax barriers.

2.  **Global Invariant Tasks at the Classical Algorithm Level**:
    - **Function**: Shifts the evaluation focus from simple functions and local properties to data structures, graph algorithms, dynamic programming, greedy algorithms, and mathematical algorithms requiring global reasoning.
    - **Mechanism**: 77 tasks cover heaps, segment trees, red-black trees, Bellman-Ford, Edmonds-Karp, Gaussian elimination, etc., requiring the model to generate loop invariants, ghost states, auxiliary lemmas, termination proofs, or Lean tactic scripts.
    - **Design Motivation**: Real-world software correctness often depends on global properties across loops and data structures. Evaluating only single-step operations overestimates model capability and masks failures in handling graph invariants and complex state updates.

3.  **Semantic Filtering After Verification Success**:
    - **Function**: Prevents models from passing formal verification via vacuous proofs, cheating statements, or simplified algorithms.
    - **Mechanism**: The benchmark reports two metrics: **Compiler Verified** (accepted by the verifier) and **Full Mark** (accepted by the verifier and judged by an LLM to follow the specified algorithmic semantics). The difference between these is termed the *algorithmic fidelity gap*.
    - **Design Motivation**: Formal verification only guarantees that the "implementation satisfies the specification." If the specification is underspecified or the model exploits loopholes, the target algorithm may still not be realized. Semantic filtering ensures the benchmark evaluates both verification correctness and instruction fidelity.

### Loss & Training
The paper does not train models but uses a unified inference-time refinement protocol. In the main experiments, most models use 15 repair rounds, while GPT-5.3 Codex uses 8 rounds. Open-source models are additionally evaluated with a $10 \times 15$ multi-sample budget. In each round, the model modifies the previous code version based on compiler/verifier errors. The authors also compare depth versus width: depth involves iterative repair along a single chain, while width involves parallel sampling of independent candidates, comparing gains under equal computation budgets.

## Key Experimental Results

### Main Results
The main table reports **Compiler Verified** and the semantically filtered **Full Mark** across three systems. The Full Mark results, which are more indicative of true performance, are highlighted below along with the drop from Verified to Full Mark.

| Model / Budget | Dafny Verified / Full Mark | Verus Verified / Full Mark | Lean Verified / Full Mark | Main Implications |
| :--- | :--- | :--- | :--- | :--- |
| GPT-5.3 Codex, $1 \times 8$ | 49.35 / 42.86 | 14.29 / 11.69 | 23.38 / 11.69 | Codex is strongest in Dafny; Lean shows a large semantic gap. |
| Gemini-3 Flash, $1 \times 15$ | 55.84 / 40.26 | 25.97 / 24.68 | 9.09 / 7.79 | Frontier models leverage repair rounds, but Lean remains extremely difficult. |
| GPT-5 mini, $1 \times 15$ | 41.56 / 30.47 | 7.79 / 6.49 | 5.19 / 5.19 | Small models rapidly fall behind in Verus/Lean. |
| GPT-OSS-120B, $1 \times 15$ | 21.04 / 13.51 | 7.66 / 7.01 | 12.60 / 7.01 | Open-source capabilities are significantly lower than frontier models. |
| GPT-OSS-120B, $10 \times 15$ | 44.16 / 28.57 | 12.99 / 10.39 | 25.97 / 14.29 | Increasing parallel budget helps but still lags behind Gemini's Dafny performance. |
| Qwen3-235B, $10 \times 15$ | 32.47 / 29.87 | 12.99 / 12.99 | 6.49 / 6.49 | Dafny improves with expanded budget; Lean remains low. |
| Devstral-2-123B, $10 \times 15$ | 33.77 / 18.18 | 14.29 / 12.99 | 6.49 / 6.49 | Semantic filtering reveals a large fidelity gap in Dafny. |

### Ablation Study
The analysis section focuses on comparing test-time compute (depth repair vs. width sampling) and decomposing failure modes for different languages.

| Analysis Item | Observation | Explanation |
| :--- | :--- | :--- |
| Gemini-3 Flash Repair Depth | Pass rates for Dafny and Verus increase continuously over 15 repair rounds; Dafny improves nearly 3x from the first to the 15th round. | Frontier models treat verifier feedback as effective reasoning feedback, similar to training signals. |
| GPT-OSS-120B Repair Depth | Gains saturate around the 3rd or 4th round; further repair provides minimal benefit. | Current open-source models behave more like "resampling with error context" and lack multi-round proof repair capability. |
| GPT-OSS Budget | The repair curve does not significantly exceed the parallel sampling baseline; deep repair may even perform worse. | For open-source models, width is more cost-effective than depth. |
| Dafny Error Trajectories | Syntax/type errors drop quickly; verification failures remain the primary bottleneck. | High-level abstraction and SMT automation allow the model to focus on repairing logic. |
| Verus Error Trajectories | Syntax/type errors persist throughout the 15 rounds. | Rust macros, ownership, and integer type conversions create syntax and type barriers. |
| Lean Error Trajectories | Hallucinations and verification errors dominate. | Models must find the correct lemma/tactic and construct correct proofs, facing both search and reasoning difficulties. |

### Key Findings
- **"Verified" does not mean "implemented the specified algorithm."** Gemini-3 Flash drops from 55.84% Verified to 40.26% Full Mark in Dafny, indicating that algorithmic fidelity must be evaluated separately.
- **Language hierarchy is evident**: Dafny is the most friendly due to its automation and mathematical abstraction, Verus is closer to system code but has a heavy syntax/type burden, and Lean has a small trusted kernel but the largest proof search space.
- **Difficult tasks are concentrated in advanced data structures and graph algorithms.** Tasks requiring ghost states, global reachability, balance invariants, or max-flow optimality proofs remain bottlenecks for current models.

## Highlights & Insights
- The value of this paper lies in its "fair" benchmark design. It is not just a collection of problems but a translation of the same algorithmic specification into three verification ecosystems, providing an interpretable basis for cross-language performance gaps.
- The **Full Mark** metric is crucial. Many code generation benchmarks only look at pass rates, but AlgoVeri shows that formal verification can be "polluted" by degenerate implementations or specification loopholes. "Proof success" and "algorithmic fidelity" must be decoupled.
- The **depth vs. width** analysis offers insights for agentic coding: not all models are suited for multi-round repair. For models with insufficient repair capability, blindly lengthening the feedback chain may be more wasteful than parallel sampling.

## Limitations & Future Work
- Semantic filtering relies on an LLM judge. While it catches obvious degenerate algorithms and cheating, there may still be misjudgments; designing mechanically checkable algorithmic fidelity conditions for more tasks would be ideal.
- Specifications were manually written and reviewed by experts, ensuring high quality but at high cost. Future expansion to thousands of tasks would require semi-automated specification migration and formal well-formedness checking tools.
- AlgoVeri currently focuses on generation capability and does not systematically compare different agent architectures like specialized training, retrieval-augmented proofs, or search tree planning. It serves as a reliable measuring stick rather than a complete solution.
- Low scores in Verus and Lean reflect both model deficiencies and the scarcity of training data/tooling issues. Future improvements will likely require advancements in language ecosystems, error message design, and model training.

## Related Work & Insights
- **vs. VeriCoding**: VeriCoding is larger in scale but lacks task and specification alignment across languages; AlgoVeri is smaller but provides a fairer comparison of the inherent difficulty in Dafny, Verus, and Lean.
- **vs. CLEVER / Verina**: CLEVER and Verina focus on Lean and emphasize anti-cheating or proof-gap diagnosis; AlgoVeri extends this to cross-verification system alignment and complex classical algorithms.
- **vs. DafnyBench**: DafnyBench focuses more on annotation/hint generation; AlgoVeri requires generating complete implementations and proof artifacts from specifications, making it a more end-to-end evaluation.
- **vs. HumanEval / MBPP**: The latter only test limited test cases; AlgoVeri requires the verifier to prove correctness for all inputs, making it better suited for assessing the upper limits and shortcomings of reliable code generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The strictly aligned classical algorithm vericoding benchmark across three verification systems is highly distinctive and resolves previous across-language incomparability issues.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers closed/open-source models, three verification ecosystems, repair rounds, budget analysis, and failure mode decomposition with a complete chain of evidence.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and strong motivation; readers should note the timeline regarding model naming and future versions.
- **Value**: ⭐⭐⭐⭐⭐ Highly useful for research in code intelligence, formal verification, and agentic repair, distinguishing true reasoning progress from superficial verification pass rates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligned Multi-View Scripts for Universal Chart-to-Code Generation](../../ACL2026/code_intelligence/aligned_multi-view_scripts_for_universal_chart-to-code_generation.md)
- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](../../ACL2026/code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[ACL 2026\] Precise Debugging Benchmark: Is Your Model Debugging or Regenerating?](../../ACL2026/code_intelligence/precise_debugging_benchmark_is_your_model_debugging_or_regenerating.md)
- [\[NeurIPS 2025\] AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy](../../NeurIPS2025/code_intelligence/astrovisbench_a_code_benchmark_for_scientific_computing_and_visualization_in_ast.md)
- [\[ICML 2026\] UniRTL: Unified Code and Graph for Robust RTL Representation Learning](unirtl_unifying_code_and_graph_for_robust_rtl_representation_learning.md)

</div>

<!-- RELATED:END -->
