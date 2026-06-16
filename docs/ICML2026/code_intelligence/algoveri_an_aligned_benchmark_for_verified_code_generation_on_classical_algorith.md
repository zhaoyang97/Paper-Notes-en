---
title: >-
  [Paper Note] AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms
description: >-
  [ICML 2026][Code Intelligence][verified code generation] AlgoVeri constructs a verified code generation benchmark strictly aligned across Dafny, Verus, and Lean for classical algorithms. It demonstrates that current LLMs still face a massive gap in handling complex global invariants, system-level constraints, and explicit proof searches, with success rates in Lean and Verus
tags:
  - ICML 2026
  - Code Intelligence
  - verified code generation
  - Dafny
  - Verus
  - Lean
date: 2026-05-08
content_hash: a456e1d5e207ea5c
---
# AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms

**Conference**: ICML 2026  
**arXiv**: [2602.09464](https://arxiv.org/abs/2602.09464)  
**Code**: https://github.com/haoyuzhao123/algoveri  
**Area**: Code Intelligence / Formal Verification / Verified Code Generation  
**Keywords**: verified code generation, formal verification, Dafny, Verus, Lean, algorithm benchmark  

## TL;DR
AlgoVeri constructs a verified code generation benchmark strictly aligned across Dafny, Verus, and Lean for classical algorithms. It demonstrates that current LLMs still face a massive gap in handling complex global invariants, system-level constraints, and explicit proof searches, with success rates in Lean and Verus being significantly lower than in Dafny.

## Background & Motivation
**Background**: LLMs are already capable of generating code from natural language, but "passing unit tests" does not equate to a program being correct for all possible inputs. Formal verification requires models to generate implementations, specifications, and proof artifacts, allowing a verifier to prove that the code satisfies its contract. This is much stricter than test-driven benchmarks like HumanEval/MBPP.

**Limitations of Prior Work**: Most existing vericoding benchmarks cover only a single tool or combine disparate tasks across different languages. This makes scores incomparable: success in Dafny might only involve array bounds checks, while success in Lean might require a complete proof of functional correctness. If specifications are not aligned in multi-language comparisons, it is impossible to determine whether performance differences stem from model capability or task difficulty.

**Key Challenge**: The community needs to evaluate whether LLMs can truly verify the global properties of "classical algorithms," yet existing benchmarks are either too simple or unfair across tools. Complex algorithms such as Red-Black Trees, Maximum Flow, or Tarjan's Strongly Connected Components require maintaining global invariants, ghost states, and termination proofs—precisely where current models most easily expose their reasoning flaws.

**Goal**: The authors aim to decouple problem difficulty, verification language characteristics, and model reasoning capabilities for evaluation. Specifically, they collected 77 textbook-style classical algorithm tasks and wrote semantically aligned specifications in Dafny, Verus, and Lean, then evaluated closed-source and open-source models using the same iterative refinement protocol.

**Key Insight**: Instead of pursuing an ultra-large-scale problem set, the paper emphasizes "alignment" and "algorithmic depth." Each task requires the same algorithmic semantics to be expressed across three verification systems. Finally, a semantic validator is used to exclude cases where models exploit loopholes via `assume`, `sorry`, or degenerate algorithms.

**Core Idea**: Push verified code generation from "local safety checks" to "global algorithmic correctness" evaluation using a three-language classical algorithm benchmark with strictly parallel specifications.

## Method
AlgoVeri is essentially a benchmark construction and evaluation pipeline. Its key is not proposing a new generative model, but rather fixing the tasks, specifications, verification systems, iterative refinement, and semantic filtering to make performance differences between models and languages interpretable.

### Overall Architecture
The input is a classical algorithm problem along with its natural language description. The authors provide function signatures, pre-conditions, post-conditions, and necessary helper definitions for Dafny, Verus, and Lean, requiring specifications in all three languages to express the same set of semantic constraints. During evaluation, the model receives the problem description and the formal specification for a specific language to generate implementation and proof artifacts. If the verifier returns an error, the model can read the error message and perform multi-turn repairs. A solution must first be accepted by the compiler/verifier and then pass through an LLM semantic filter to ensure it actually implements the specified algorithm rather than bypassing the specification.

### Key Designs

**1. Specification Alignment across Verification Systems: Making Cross-Language Scores Comparable**

A fundamental problem with existing vericoding benchmarks is that scores across tools are incomparable—"success" in Dafny might be a simple array bounds check, while "success" in Lean requires a full correctness proof. Without alignment, one cannot judge if the gap comes from model capability or task difficulty. AlgoVeri's approach is to manually write and cross-review function signatures, pre-conditions, post-conditions, and helper definitions for the same algorithm task across Dafny, Verus, and Lean, forcing the three systems to express the same semantic constraints. For instance, the Longest Increasing Subsequence (LIS) task is not satisfied by merely "the result being within the array range," but uniformly requires "proving the existence of an increasing subsequence of the given length and the non-existence of a longer valid subsequence." By fixing the problem difficulty, observed cross-language gaps can be cleanly attributed to toolchain abstraction levels, the model's proof search capabilities, and linguistic syntax barriers.

**2. Global Invariant Tasks at the Classical Algorithm Level: Pushing Evaluation to Global Reasoning**

Benchmarks like HumanEval/MBPP only test single-step or local properties, which tends to overestimate model capabilities—for example, proving that a single BST rotation maintains local order is easy, but proving the full insertion correctness of a Red-Black Tree (maintaining global black-height invariants) is much harder. Consequently, AlgoVeri uses 77 textbook-level classical algorithms as its library, covering Heaps, Segment Trees, Red-Black Trees, Bellman-Ford, Edmonds-Karp, Gaussian Elimination, etc. It requires models to generate loop invariants, ghost states, auxiliary lemmas, termination proofs, or Lean tactic scripts. The correctness of these tasks depends on global properties across loops and data structures, hitting the reasoning weaknesses of current models regarding graph reachability invariants and complex state updates.

**3. Semantic Filtering after Verification: Distinguishing "True Verification" from "Exploiting Loopholes"**

Formal verification only guarantees that "the implementation satisfies the given specification." However, if the specification has loopholes, models can use vacuous proofs, cheating statements like `assume`/`sorry`, or simply implement a simpler degenerate algorithm to trick the verifier. To address this, AlgoVeri adds an LLM judge for semantic validation after the verifier passes, splitting results into two metrics: "Compiler Verified" (accepted by the verifier) and "Full Mark" (accepted by the verifier and confirmed as the intended algorithm by the judge). The difference between the two is defined as the "algorithmic fidelity gap." This layer of filtering allows the benchmark to simultaneously examine "verification correctness" and "faithful implementation of the target algorithm," avoiding the misclassification of loophole exploits as genuine capability.

### Loss & Training
The paper does not train models but uses a unified inference-time refinement protocol. In the main experiments, most models use 15 repair rounds, while GPT-5.3 Codex uses 8 rounds; open-source models are additionally evaluated with a $10\times15$ multi-sample budget. In each round, the model modifies the previous version based on compiler/verifier errors. The authors also compare depth and width: depth is iterative repair along the same chain, while width involves sampling multiple independent candidates in parallel, comparing gains under the same computational budget.

## Key Experimental Results

### Main Results
The main table reports Compiler Verified and Full Mark (post-semantic filtering) across the three verification systems. Below are the Full Mark results, which are more illustrative, alongside the drop from Verified to Full Mark.

| Model / Budget | Dafny Verified / Full Mark | Verus Verified / Full Mark | Lean Verified / Full Mark | Main Meaning |
|:---|:---|:---|:---|:---|
| GPT-5.3 Codex, $1\times8$ | 49.35 / 42.86 | 14.29 / 11.69 | 23.38 / 11.69 | Codex is strongest in Dafny, but Lean has a large semantic gap |
| Gemini-3 Flash, $1\times15$ | 55.84 / 40.26 | 25.97 / 24.68 | 9.09 / 7.79 | Frontier models leverage repair rounds, but Lean remains extremely difficult |
| GPT-5 mini, $1\times15$ | 41.56 / 30.47 | 7.79 / 6.49 | 5.19 / 5.19 | Small models fall behind quickly on Verus/Lean |
| GPT-OSS-120B, $1\times15$ | 21.04 / 13.51 | 7.66 / 7.01 | 12.60 / 7.01 | Open-source base capability is significantly lower than frontier models |
| GPT-OSS-120B, $10\times15$ | 44.16 / 28.57 | 12.99 / 10.39 | 25.97 / 14.29 | Increasing parallel budget helps, but still trails Gemini's Dafny performance |
| Qwen3-235B, $10\times15$ | 32.47 / 29.87 | 12.99 / 12.99 | 6.49 / 6.49 | Dafny improves with expanded budget; Lean remains low |
| Devstral-2-123B, $10\times15$ | 33.77 / 18.18 | 14.29 / 12.99 | 6.49 / 6.49 | Semantic filtering reveals a large fidelity gap in Dafny |

### Ablation Study
The analysis section focuses on comparing test-time compute between deep refinement and wide sampling, and breaks down failure modes across languages.

| Analysis Item | Observation | Explanation |
|:---|:---|:---|
| Gemini-3 Flash Repair Depth | Pass rate for Dafny and Verus increases continuously over 15 repair rounds; Dafny increases nearly 3x from the first to the 15th round. | Frontier models can treat verifier feedback as effective inference feedback, similar to training signals. |
| GPT-OSS-120B Repair Depth | Saturates after roughly 3 to 4 rounds, with very little gain from further repairs. | Current open-source models behave more like "resampling with error context," lacking multi-round proof repair capability. |
| GPT-OSS Equal Budget | The repair curve does not significantly exceed the parallel sampling baseline; deep repair might even be worse. | For open-source models, width is more cost-effective than depth. |
| Dafny Error Trajectory | Early syntax/type errors drop quickly; subsequent errors are primarily verification failures. | High-level abstraction and SMT automation allow the model to focus on fixing logic. |
| Verus Error Trajectory | Syntax/type errors persist throughout the 15 rounds. | Rust macros, ownership, and integer type conversions form a significant syntax and type barrier. |
| Lean Error Trajectory | Hallucinations and verification errors dominate. | Models struggle with both finding the right lemma/tactic and constructing the correct proof—a dual search and reasoning difficulty. |

### Key Findings
- "Verified" does not mean "implemented the specified algorithm." Gemini-3 Flash dropped from 55.84% Verified to 40.26% Full Mark on Dafny, indicating that algorithmic fidelity must be evaluated separately.
- Language hierarchy is evident: Dafny’s automation and mathematical abstraction are the most user-friendly; Verus is closer to systems code but carries a heavy syntax/type burden; Lean has a small trusted kernel but the largest proof search space.
- Difficult tasks are concentrated in advanced data structures and graph algorithms. Tasks requiring ghost states, global reachability, balance invariants, or max-flow optimality proofs remain a bottleneck for current models.

## Highlights & Insights
- The value of this paper lies in making the benchmark design sufficiently "fair." It is not just a collection of problems, but a translation of the same algorithm specifications into three verification ecosystems, providing an interpretable basis for cross-language performance differences.
- The Full Mark metric is crucial. While many code generation benchmarks only look at pass rates, AlgoVeri shows that formal verification can be contaminated by degenerate implementations or specification loopholes, and thus "proof success" and "algorithmic fidelity" must be decoupled.
- The depth vs. width analysis is very insightful for agentic coding: not all models are suitable for multi-round repair. For models with insufficient repair capabilities, blindly lengthening the feedback chain may be more wasteful than parallel sampling.

## Limitations & Future Work
- Semantic filtering uses an LLM judge; while it catches obvious degenerate algorithms and cheating, misjudgments are still possible. Designing mechanically checkable algorithmic fidelity conditions for more tasks would be ideal.
- Specifications were manually written and reviewed by experts, ensuring high quality but at great cost. Expanding to thousands of tasks in the future will require semi-automated specification migration and formal well-formedness checking tools.
- AlgoVeri currently evaluates generation capability and does not systematically compare different agent architectures such as specialized training, retrieval-augmented proof, or search-tree planning. It serves more as a reliable yardstick than a complete solution.
- Low scores for Verus and Lean reflect both model deficiencies and the scarcity of training data and tooling friction. Subsequent improvements will likely require advancements in language ecosystems, error message design, and model training.

## Related Work & Insights
- **vs VeriCoding**: VeriCoding is larger in scale but lacks task and specification alignment across languages; AlgoVeri is smaller but provides a fairer comparison of the true difficulty of Dafny, Verus, and Lean.
- **vs CLEVER / Verina**: CLEVER and Verina focus on Lean and emphasize anti-cheating or proof-gap diagnostics; AlgoVeri further emphasizes alignment across verification systems and complex classical algorithms.
- **vs DafnyBench**: DafnyBench leans more towards annotation/hint generation; AlgoVeri requires generating complete implementations and proof artifacts from specifications, making the task closer to end-to-end verified code generation.
- **vs HumanEval / MBPP**: The latter only test limited test cases, whereas AlgoVeri requires the verifier to prove correctness for all inputs, making it more suitable for evaluating the upper bounds and weaknesses of reliable code generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The strictly aligned classical algorithm vericoding benchmark across three verification systems is highly distinctive and resolves the previous issue of incomparable cross-language scores.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers closed/open-source models, three verification ecosystems, repair rounds, equal-budget analysis, and failure mode breakdown, providing a complete chain of evidence.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, and the benchmark motivation is strong; some model naming and future versioning require the reader to be mindful of the temporal context.
- Value: ⭐⭐⭐⭐⭐ Highly useful for research in code intelligence, formal verification, and agentic repair, distinguishing real reasoning progress from shallow verification pass rates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Aligned Multi-View Scripts for Universal Chart-to-Code Generation](../../ACL2026/code_intelligence/aligned_multi-view_scripts_for_universal_chart-to-code_generation.md)
- [\[ACL 2025\] CodeDPO: Aligning Code Models with Self Generated and Verified Source Code](../../ACL2025/code_intelligence/codedpo_code_alignment.md)
- [\[ACL 2026\] From Charts to Code: A Hierarchical Benchmark for Multimodal Models](../../ACL2026/code_intelligence/from_charts_to_code_a_hierarchical_benchmark_for_multimodal_models.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](../../ACL2025/code_intelligence/texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](../../ACL2025/code_intelligence/feabench_repo_code_gen.md)

</div>

<!-- RELATED:END -->
