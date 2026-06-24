---
title: >-
  [Paper Note] Benchmarking Testing in Automated Theorem Proving
description: >-
  [ACL 2026][Code Intelligence][Automated Theorem Proving] Drawing inspiration from the concept of "integration testing" in software engineering, the semantic correctness of a generated theorem is determined by whether "all successor theorems depending on it still compile." This work constructs T2, a Lean 4 benchmark with 2206 problems, revealing a significant gap where mainstream LLMs achieve a 80%+ compilation rate but a semantic accuracy of only ~39%.
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Automated Theorem Proving"
  - "Lean 4"
  - "Semantic Correctness"
  - "Integration Testing"
  - "Cut Elimination"
date: 2026-05-08
content_hash: e39665358caec3a5
---

# Benchmarking Testing in Automated Theorem Proving

**Conference**: ACL 2026  
**arXiv**: [2604.23698](https://arxiv.org/abs/2604.23698)  
**Code**: <https://github.com/ldilab/T2>  
**Area**: Code Intelligence / Theorem Proving / LLM Evaluation  
**Keywords**: Automated Theorem Proving, Lean 4, Semantic Correctness, Integration Testing, Cut Elimination

## TL;DR
Drawing inspiration from the concept of "integration testing" in software engineering, the semantic correctness of a generated theorem is determined by whether "all successor theorems depending on it still compile." This work constructs T2, a Lean 4 benchmark with 2206 problems, revealing a significant gap where mainstream LLMs achieve a 80%+ compilation rate but a semantic accuracy of only ~39%.

## Background & Motivation

**Background**: LLM-based Automated Theorem Proving (ATP) has progressed rapidly. Benchmarks such as MiniF2F, ProofNet, and PutnamBench use "compilation pass" as the sole indicator of correctness. Specialized provers like DeepSeek-Prover, Kimina, and Goedel are optimized specifically for this metric.

**Limitations of Prior Work**: Compilation success does not equal semantic correctness. A tautologically trivial theorem that formalizes $a+b=b+a$ as $a+b=a+b$ can compile in Lean but fails to capture the intent of commutativity. Existing remedies either rely on surface-level similarity like BLEU/NLI (unreliable), prover-based equivalence checks (requiring ground-truth references), or manual inspection (unscalable).

**Key Challenge**: An ideal semantic evaluation should be reference-free, automated, and directly verify semantics. Existing methods satisfy only one or two of these requirements.

**Goal**: To automatically determine the semantic correctness of generated theorems without relying on manual annotations or reference answers, and to ensure this determination can distinguish the true capability gap between specialized provers and general LLMs.

**Key Insight**: The Curry-Howard isomorphism dictates that "proof = program, theorem = type." In code generation, unit testing has long replaced BLEU—if all downstream tests calling a function pass, its behavior is considered correct. This idea is transferred to theorem proving: treatment of the theorem under evaluation $t_{fl}$ as a "function" and all successor theorems $\mathcal{T}_{succ}$ depending on it as "integration test cases." If the entire dependency chain still compiles after replacement, there is strong evidence that $t_{fl}$ is semantically correct.

**Core Idea**: Replace "Compilation Accuracy" with "Testing Accuracy (TA)"—substituting the generated theorem into a real Lean repository and verifying if all successor theorems still pass `lake build`.

## Method

### Overall Architecture
T2 is a "reference-free, automated, and direct semantic verification" ATP evaluation protocol consisting of two parts: metric definition and data construction. The core action involves replacing the original theorem $t_{GT}$ with the LLM-generated formalization $t_{fl}$ in a real Lean 4 repository while keeping the name unchanged, then re-compiling the dependency chain via `lake build`. Semantic correctness is only granted if all successor theorems $t_{succ}^{(i)}$ compile successfully. For the data, 2206 target theorems were automatically extracted from 5 high-quality Lean 4 repositories, with each theorem having an average of 41 successors acting as "integration test cases." Evaluation is conducted zero-shot (temperature=0.6, top_p=0.95), with a 600s compilation timeout per problem, and provides the NL proof alongside successor theorems in the context.

### Key Designs

**1. Formalizing Integration Testing as Cut Elimination: A Proof-Theoretic Basis for Semantic Correctness**

The analogy of "downstream tests passing implies correct function behavior" is grounded in proof theory. In sequent calculus, a proof $X \to W$ can form a cut chain $X \to Y \to Z \to W$. The cut elimination theorem guarantees that if the entire chain holds, intermediate lemmas can be eliminated to recover a direct proof of $X \to W$. By mapping $t_{fl}$ to $X \to Y$ and $\mathcal{T}_{succ}$ to $Y \to Z \to W$, the "successor chain compiles" condition is equivalent to "the cut is eliminable," which corresponds to the semantic correctness of $t_{fl}$. This provides a theoretical explanation for why deeper chains and more successors approximate a true semantic guarantee.

**2. Testing Accuracy (TA) Metric: Mapping Logic to a Binary Pass/Fail**

For each problem $P = (t_{nl}, \mathcal{T}_{pred}, \mathcal{T}_{succ})$, TA is defined as $\text{TA} = \mathbb{E}_{P \sim \mathcal{D}}\left[\bigwedge_{i=1}^{k}\text{compiles}(t_{succ}^{(i)} \mid t_{fl})\right]$, where 1 is recorded if all $k$ successors compile after replacing $t_{GT}$ with $t_{fl}$, and 0 otherwise. To ensure meaningful coverage, only theorems with $|\mathcal{T}_{succ}| \geq 2$ are included. TA directly aligns with the unit-test pass rate in software engineering as a hard metric for cross-model semantic comparison.

**3. Dependency-Aware Dataset Construction Pipeline: Extracting Theorems from Real Repositories**

Existing benchmarks are mostly isolated problems with no successors for TA evaluation. The pipeline uses Lean-Dojo to parse the global dependency graph $G=(V,E)$ and filters target theorems based on non-triviality (distance to successor > 1) and successor coverage ($|\mathcal{T}_{succ}| \geq 2$). Natural language descriptions $t_{nl}$ are generated by Claude Sonnet 4.5. A "Hard" subset of 389 problems requires generating both propositions and proofs for Prop types. These 2206 target theorems have an average depth of 7 and an average of 1.6k successors, imposing much stricter constraints than isolated problems.

## Key Experimental Results

### Main Results
Comparison of compilation rate vs. TA across 18 models:

| Model | T2 Compile | T2 TA | T2 Hard Compile | T2 Hard TA |
|------|-----------|-------|-----------------|-----------|
| Claude-Sonnet-4.5 | 80.3 | **38.9** | 46.0 | **4.5** |
| GPT-5 | 85.7 | 37.7 | 68.3 | 3.4 |
| GPT-5-nano | 88.7 | 36.6 | 75.6 | 1.5 |
| DeepSeek-Prover-v2-7B | 62.2 | 30.0 | 35.5 | 3.2 |
| Kimina-Autoformalizer-7B | 21.9 | 20.0 | 4.3 | 1.5 |

Claude-Sonnet-4.5 achieves an 80.3% compilation rate on the Full set but only 38.9% TA (a 2× gap). GPT-5-nano on the Hard set achieves 75.6% compilation but only 1.5% TA (a 50× gap).

### Ablation Study (Influence of Input Context, T2 Full)

| Model | NL✗ ST✗ | NL✓ ST✗ | NL✗ ST✓ | NL✓ ST✓ |
|------|---------|---------|---------|---------|
| Claude-Sonnet-4.5 | 34.0 | 33.0 | 32.9 | **38.9** |
| Claude-4-Sonnet | 30.0 | 27.7 | 32.3 | **36.0** |
| Llama-3.1-70B | 28.5 | 28.9 | 29.1 | **37.0** |

NL proof and successor theorem (ST) **must be provided together** for stable performance gains. Providing the successor theorem alone may even decrease performance, suggesting models need NL for reasoning context and successors for semantic constraints.

### Key Findings
- **Compilation rate is a poor proxy for semantic correctness**: The precision of using compilation success to predict semantic correctness is only 6.89%; over 70% of samples with high BLEU (90-100) are still semantically incorrect.
- **Specialized provers perform worse on the semantic dimension**: Models like Goedel, Kimina, and DeepSeek-Prover show improved compilation rates but lower TA compared to general LLMs, indicating that domain fine-tuning teaches "syntactic fluency" rather than "semantic alignment."
- **TA becomes stricter with more successors**: With only 2 successors, Claude-4-Sonnet reaches ~70% TA, but this drops to ~0% with 5 successors. This benchmark provides a rigorous evaluation with an average depth of 7 and 1.6k successors.
- **Few-shot and iterative refinement fails to close the semantic gap**: 2-shot performance sometimes drops on the Hard set; iterative refinement only improved Hard TA from 3.2% to 5.0%.

## Highlights & Insights
- By linking the Curry-Howard isomorphism with software integration testing, this work provides a proof-theoretic explanation (cut elimination) for semantic evaluation, moving beyond mere engineering analogies.
- The 50× gap (80% compilation / 4% TA) reveals that ATP benchmarks have been significantly over-claimed in recent years—similar to how checking only if code "compiles" would overestimate model capabilities in code generation.
- Dataset construction is entirely automated: mining targets and successors from public Lean repositories and using LLMs for NL descriptions cost only ~$100. This "real repository + dependency structure" paradigm can migrate to Coq, Isabelle, and other proof assistants.

## Limitations & Future Work
- TA is binary and cannot distinguish between "9 successors pass, 1 fails" and "all fail." Graded TA is suggested for future work.
- It is not applicable to isolated theorems (approx. 1.4%) located at the frontier of a repository, where one must revert to BLEU or manual check.
- Currently supports only Lean 4, using data from 5 repositories. NL labels from Claude Sonnet 4.5 might introduce bias toward similar provers.
- Constructing counter-examples for semantic errors is difficult: since names and type signatures are fixed, models usually fail to compile if they are wrong. TA primarily captures fine-grained "correct type but wrong content" errors.

## Related Work & Insights
- **vs MiniF2F / ProofNet / PutnamBench**: These check only compilation. T2 additionally checks successor chains. While others focus on isolated problems, T2 filters high-successor theorems from dependency graphs.
- **vs ProofNetVerif / Con-NF**: Those methods require ground-truth references for equivalence checks. T2 is reference-free.
- **vs HumanEval / MBPP**: T2 introduces the testing-based functional correctness paradigm from code generation into formal theorem proving, backed by the theory of the Curry-Howard isomorphism.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bringing the testing-based evaluation paradigm from code generation to ATP with a rigorous cut elimination explanation is a highly effective paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 18 models across 4 context settings and 3 subsets, compared against BLEU/BEq and compilation rates.
- Writing Quality: ⭐⭐⭐⭐ Clear conceptual framework; the analogy in Figure 1 and the cut elimination explanation make the core idea intuitive.
- Value: ⭐⭐⭐⭐⭐ Challenges the optimistic narrative of specialized provers in ATP; TA is likely to become an essential evaluation metric for future ATP research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)
- [\[ACL 2026\] CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges](creativebench_benchmarking_and_enhancing_machine_creativity_via_self-evolving_ch.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ICLR 2026\] VERINA: Benchmarking Verifiable Code Generation](../../ICLR2026/code_intelligence/verina_benchmarking_verifiable_code_generation.md)

</div>

<!-- RELATED:END -->
