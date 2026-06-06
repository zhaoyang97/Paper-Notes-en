---
title: >-
  [Paper Note] Benchmarking Testing in Automated Theorem Proving
description: >-
  [ACL 2026][Code Intelligence][Automated Theorem Proving] Drawing on the "integration testing" concept from software engineering, the semantic correctness of a generated theorem is determined by whether "all successor the…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Automated Theorem Proving"
  - "Lean 4"
  - "Semantic Correctness"
  - "Integration Testing"
  - "Cut Elimination"
date: 2026-05-08
content_hash: c71c566bd28a0fd7
---

# Benchmarking Testing in Automated Theorem Proving

**Conference**: ACL 2026  
**arXiv**: [2604.23698](https://arxiv.org/abs/2604.23698)  
**Code**: <https://github.com/ldilab/T2>  
**Area**: Code Intelligence / Theorem Proving / LLM Evaluation  
**Keywords**: Automated Theorem Proving, Lean 4, Semantic Correctness, Integration Testing, Cut Elimination

## TL;DR
Drawing on the "integration testing" concept from software engineering, the semantic correctness of a generated theorem is determined by whether "all successor theorems depending on it can still compile." This work constructs T2, a Lean 4 benchmark with 2206 problems, revealing a massive gap between the 80%+ compilation rate and the ~39% semantic correctness of mainstream LLMs.

## Background & Motivation

**Background**: LLM-based Automated Theorem Proving (ATP) has progressed rapidly in recent years. Benchmarks such as MiniF2F, ProofNet, and PutnamBench use "compilation success" as the sole metric for correctness. Specialized provers like DeepSeek-Prover, Kimina, and Goedel are optimized specifically for this metric.

**Limitations of Prior Work**: Compilation success $\neq$ semantic correctness. A tautological theorem that formalizes $a+b=b+a$ as $a+b=a+b$ can compile in Lean but fails to capture the actual intent of the commutative law. Existing remedies rely on surface similarity metrics like BLEU/NLI (unreliable), prover-based equivalence checks (require ground-truth references), or human inspection (not scalable).

**Key Challenge**: An ideal semantic evaluation should be reference-free, automatic, and direct. However, existing methods satisfy only one or two of these criteria.

**Goal**: To automatically determine the semantic correctness of generated theorems without relying on human annotations or reference answers, and to ensure this determination distinguishes the true capability gap between specialized provers and general LLM capabilities.

**Key Insight**: The Curry-Howard isomorphism suggests "proofs = programs, theorems = types." Code generation has long replaced BLEU with unit testing—if all downstream tests calling a function pass, its behavior is considered correct. This study applies this logic to theorem proving: by treating the theorem under evaluation $t_{fl}$ as a "function" and all successor theorems $\mathcal{T}_{succ}$ that depend on it as "integration test cases," successful compilation of the entire dependency chain after replacement provides strong evidence of the semantic correctness of $t_{fl}$.

**Core Idea**: Replace "Compilation Accuracy" with "Testing Accuracy (TA)"—substituting the generated theorem into a real Lean repository to verify if all successor theorems that call it can still pass `lake build`.

## Method

### Overall Architecture
The proposed method consists of two parts: (1) Metric definition—Testing Accuracy; (2) Dataset construction—automatically mining 2206 target theorems and an average of 41 successor theorems from five high-quality Lean 4 repositories. During evaluation, the generated formal theorem $t_{fl}$ replaces the original $t_{GT}$ (retaining the same name) in the repository. A "test pass" is recorded only if the entire dependency chain of $t_{succ}^{(i)}$ compiles successfully.

### Key Designs

1.  **Formalizing Integration Testing as Cut Elimination**:
    *   **Function**: Provides a rigorous logical foundation for "integration testing" using the cut elimination rule from proof theory, showing that this judgment method reflects semantic correctness in principle.
    *   **Mechanism**: In sequent calculus, proving $X \to W$ can be achieved by introducing intermediate lemmas $Y, Z$ to form a cut chain $X \to Y \to Z \to W$. The cut elimination theorem states that if the entire chain holds, the intermediate lemmas can be eliminated to obtain a direct proof of $X \to W$. By viewing the theorem under evaluation $t_{fl}$ as $X \to Y$ and the successor theorems $\mathcal{T}_{succ}$ as $Y \to Z \to W$, the compilation of the successor chain is equivalent to cut eliminability, corresponding to the semantic correctness of $t_{fl}$.
    *   **Design Motivation**: This legitimizes the "successor compilation = semantic correctness" heuristic at the proof-theory level, moving beyond a simple engineering analogy. It also explains why TA approaches a true semantic guarantee as the number of successors and the depth of the chain increase.

2.  **Testing Accuracy (TA) Metric**:
    *   **Function**: Implements the aforementioned idea as a computable metric, defining a binary pass/fail for each problem $P = (t_{nl}, \mathcal{T}_{pred}, \mathcal{T}_{succ})$.
    *   **Mechanism**: TA is defined as $\texttt{TA} = \mathbb{E}_{P \sim \mathcal{D}}\left[\bigwedge_{i=1}^{k}\texttt{compiles}(t_{succ}^{(i)} \mid t_{fl})\right]$. This denotes a score of 1 if all $k$ successor theorems compile after $t_{fl}$ replaces $t_{GT}$, and 0 otherwise. To ensure meaningful coverage, only target theorems with $|\mathcal{T}_{succ}| \geq 2$ are included. The evaluation runs the full dependency chain using the native Lean version of each repo via `lake build`.
    *   **Design Motivation**: While BLEU provides a continuous score (0-100) weakly related to correctness, TA provides a binary "test pass" result isomorphic to unit-test pass rates in software engineering, allowing for a direct comparison of model capabilities.

3.  **Dependency-Aware Dataset Construction Pipeline**:
    *   **Function**: Automatically extracts 2206 high-quality target theorems from real Lean 4 repositories, each associated with predictors $\mathcal{T}_{pred}$ (necessary preceding definitions) and successors $\mathcal{T}_{succ}$ (theorems that depend on it).
    *   **Mechanism**: The global dependency graph $G=(V,E)$ is parsed using Lean-Dojo. Target theorems are filtered based on two rules: (1) Non-triviality: distance to successor > 1 (excluding isolated theorems); (2) Successor coverage: $|\mathcal{T}_{succ}| \geq 2$. Claude Sonnet 4.5 generates a natural language description $t_{nl}$ for each target (verified correct in 10/10 manual samples). A "Hard" subset of 389 problems was further identified where the target is of type Prop, requiring the simultaneous generation of both the proposition and the proof.
    *   **Design Motivation**: Current benchmarks (MiniF2F, PutnamBench, etc.) mostly consist of isolated Olympiad problems with almost no successor theorems for TA evaluation (only 12 problems satisfy requirements). Target theorems from real repositories are naturally embedded in rich dependency structures, with an average depth of 7 and an average of 1.6k successors, providing multiple semantic constraints.

### Loss & Training
This work is an evaluation study; no training was performed. Evaluation protocol: all models were tested zero-shot with temperature=0.6 and top_p=0.95. One completion per problem was generated. Compilation timeout was set to 600s. The NL proof and successor theorems were provided as context to the models.

## Key Experimental Results

### Main Results
Compilation rate and TA were compared across 18 models (Claude, GPT, Llama, DeepSeek, Kimina, Goedel, etc.):

| Model | T2 Compile | T2 TA | T2 Hard Compile | T2 Hard TA |
| :--- | :--- | :--- | :--- | :--- |
| Claude-Sonnet-4.5 | 80.3 | **38.9** | 46.0 | **4.5** |
| GPT-5 | 85.7 | 37.7 | 68.3 | 3.4 |
| GPT-5-nano | 88.7 | 36.6 | 75.6 | 1.5 |
| DeepSeek-Prover-v2-7B | 62.2 | 30.0 | 35.5 | 3.2 |
| Kimina-Autoformalizer-7B | 21.9 | 20.0 | 4.3 | 1.5 |

The strongest model, Claude-Sonnet-4.5, achieved an 80.3% compilation rate on the Full set but only 38.9% TA (a 2× gap). GPT-5-nano achieved 75.6% compilation on the Hard set but only 1.5% TA (a 50× gap).

### Ablation Study (Impact of input context, T2 Full)

| Model | NL ✗ ST ✗ | NL ✓ ST ✗ | NL ✗ ST ✓ | NL ✓ ST ✓ |
| :--- | :--- | :--- | :--- | :--- |
| Claude-Sonnet-4.5 | 34.0 | 33.0 | 32.9 | **38.9** |
| Claude-4-Sonnet | 30.0 | 27.7 | 32.3 | **36.0** |
| Llama-3.1-70B | 28.5 | 28.9 | 29.1 | **37.0** |

NL proof and successor theorems **must both be provided** to see consistent gains. Providing the successor theorem alone may even decrease scores, suggesting models require NL for reasoning context and successors for semantic constraints.

### Key Findings
- **Compilation rate is a poor proxy for semantic correctness**: The precision of predicting semantic correctness through compilation success is only 6.89%. Among samples with high BLEU (90-100 range), over 70% remain semantically incorrect.
- **Specialized provers are inferior in the semantic dimension**: Specialized models like Goedel, Kimina, and DeepSeek-Prover show improved compilation rates compared to general LLMs but lower TA, indicating that domain fine-tuning teaches "syntactic fluency" rather than "semantic alignment."
- **TA becomes stricter with more successors**: With only 2 successors, Claude-4-Sonnet achieves ~70% TA, which drops to ~0% when 5 successors are used, as each successor injects an independent semantic constraint. This benchmark provides strict evaluation with an average depth of 7 and 1.6k successors.
- **Few-shot and iterative refinement do not significantly close the semantic gap**: 2-shot performance occasionally dropped on the Hard set. Refinement via compilation feedback (Hilbert) only pushed Hard TA from 3.2% to 5.0%.

## Highlights & Insights
- By linking the Curry-Howard isomorphism with software integration testing, the authors provide a clear proof-theory explanation (cut elimination) for "testing as semantic evaluation," grounding the method in theory rather than just an engineering metaphor.
- The 50× gap between an 80% compilation rate and a 4% TA is striking, revealing that ATP benchmarks over the past years have been significantly over-claimed—much like overestimating model capability by only looking at compilation in code generation.
- The dataset construction is fully automated: mining targets and successors from public Lean repositories and using LLMs for NL descriptions. The cost for 2206 problems was only ~$100. This "real-world repository + dependency structure" paradigm can be seamlessly migrated to other proof assistants like Coq and Isabelle.

## Limitations & Future Work
- TA is binary and does not distinguish between "passing 9 out of 10 successors" and "failing all of them." Graded TA is suggested for future work.
- It is not applicable to isolated theorems (approximately 1.4%) that have no successors, such as newly added theorems at the edge of a repository; these must still rely on BLEU or human inspection.
- The benchmark only supports Lean 4, and data is sourced from 5 repositories (primarily research-level mathematics). NL labels were generated by Claude Sonnet 4.5, which may introduce biases shared with the evaluated provers.
- Constructing counterexamples for semantic errors is difficult because once the name and type signature are given, incorrect logic usually leads directly to compilation failure. TA primarily captures fine-grained "type-correct but content-wrong" errors.

## Related Work & Insights
- **vs MiniF2F / ProofNet / PutnamBench**: These benchmarks only check if compilation passes, whereas this work checks the successor chain. Previous benchmarks consist of isolated problems with almost no testable successors; T2 selects high-successor theorems directly from dependency graphs.
- **vs ProofNetVerif / Con-NF**: Those methods use prover-based equivalence checks or BEq, which require ground-truth references. T2 is entirely reference-free, relying only on the integrity of the dependency chain.
- **vs HumanEval / MBPP in Code Generation**: While HumanEval uses unit tests for functional correctness, this work systematically introduces testing-based evaluation to ATP. The Curry-Howard isomorphism provides a natural theoretical foundation for this transfer.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Adapting the testing-based evaluation paradigm from code generation to ATP, grounded in a rigorous cut elimination theoretical explanation, is an elegant paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Testing 18 models across 4 context settings and 3 sets (Full/Hard/Existing), combined with BLEU/BEq/Compilation Rate comparisons, provides very solid evidence.
- Writing Quality: ⭐⭐⭐⭐ The conceptual framework is clear. The analogy in Figure 1 and the cut elimination explanation make the core idea intuitive. The appendix provides complete lists of models, prompts, and costs, ensuring high reproducibility.
- Value: ⭐⭐⭐⭐⭐ This work challenges the optimistic narrative that specialized provers are becoming increasingly powerful in ATP. TA is likely to become a standard evaluation metric for future ATP research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](../../ICLR2026/code_intelligence/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)
- [\[ICML 2026\] CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding](../../ICML2026/code_intelligence/centaureval_benchmarking_human-in-the-loop_value_in_agentic_coding.md)

</div>

<!-- RELATED:END -->
