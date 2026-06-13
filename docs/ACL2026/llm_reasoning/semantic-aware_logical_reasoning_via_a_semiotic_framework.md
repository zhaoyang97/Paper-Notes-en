---
title: >-
  [Paper Note] Semantic-Aware Logical Reasoning via a Semiotic Framework
description: >-
  [ACL 2026][LLM Reasoning][Semiotic Reasoning] Ours proposes LogicAgent, a logical reasoning framework based on the Greimas Semiotic Square. It achieves SOTA logical reasoning performance under the dual challenges of sema…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Semiotic Reasoning"
  - "Greimas Semiotic Square"
  - "Logical Reasoning"
  - "Semantic Complexity"
  - "Multi-perspective Reasoning"
date: 2026-05-08
content_hash: 2b2768fcba59690b
---

# Semantic-Aware Logical Reasoning via a Semiotic Framework

**Conference**: ACL 2026  
**arXiv**: [2509.24765](https://arxiv.org/abs/2509.24765)  
**Code**: [GitHub](https://github.com/AI4SS/Logic-Agent)  
**Area**: LLM Reasoning / Logical Reasoning  
**Keywords**: Semiotic Reasoning, Greimas Semiotic Square, Logical Reasoning, Semantic Complexity, Multi-perspective Reasoning

## TL;DR

Ours proposes LogicAgent, a logical reasoning framework based on the Greimas Semiotic Square. It achieves SOTA logical reasoning performance under the dual challenges of semantic and logical complexity through multi-perspective semantic analysis and reflective verification.

## Background & Motivation

**Background**: The logical reasoning ability of LLMs is one of their core capabilities. Existing methods are mainly divided into three categories: linear reasoning (CoT), aggregate reasoning (multi-trajectory aggregation like ToT/CR), and symbolic reasoning (combining FOL solvers like Logic-LM). These methods perform well on benchmarks with clear logical structures.

**Limitations of Prior Work**: Almost all existing methods focus on **logical complexity** (reasoning depth, number of steps) while ignoring **semantic complexity** (abstract propositions, ambiguous contexts, opposing stances). In real-world reasoning, semantic ambiguity and abstraction are often intertwined with logical complexity—for instance, philosophical propositions like "Is justice always beneficial?" require not only deep reasoning but also multi-angle interpretations of abstract concepts.

**Key Challenge**: Most existing benchmarks (ProntoQA, ProofWriter, etc.) are generated based on templates with clear and unambiguous propositions, failing to test the model's reasoning robustness in semantically complex scenarios. In the real world, the coupling of semantic complexity and logical complexity is the true difficulty of reasoning.

**Goal**: To construct a reasoning framework that simultaneously addresses semantic and logical complexity, and to provide a benchmark capable of evaluating this coupled challenge.

**Core Idea**: Drawing inspiration from the Greimas Semiotic Square in structuralist semantics, the proposition is expanded into a quaternary structure (original proposition $S_1$, contradiction $\lnot S_1$, contrary $S_2$, and contradiction of the contrary $\lnot S_2$). Reasoning and cross-verification are performed from multiple perspectives to enhance reasoning robustness under semantic ambiguity.

## Method

### Overall Architecture

LogicAgent is divided into three stages: (1) **Semantic Structuring**: Expanding the proposition into four associated propositions of the semiotic square and verifying FOL consistency; (2) **Logical Reasoning**: Converting premises into FOL, planning the reasoning path, and executing symbolic deduction; (3) **Reflective Verification**: Ensuring conclusion consistency through a three-layer reflection mechanism (Direct Parsing, Quick Reflection, and Deep Reflection).

### Key Designs

1.  **Semantic Structuring Stage**:
    - **Function**: Expands a single propositional system into a semantically associated quaternary structure to expose potential interpretative ambiguities.
    - **Mechanism**: Given proposition $S_1$, it constructs its contradictory proposition $\lnot S_1$ (strict negation), its contrary proposition $S_2$ (cannot both be true but can both be false), and $\lnot S_2$. Construction uses a unified rule table (6 rules covering universal/existential/implication/conjunction/disjunction/biconditional). It introduces an Existential Import Check (EIC) to ensure the logical correctness of contrary relations, avoiding vacuous truth in empty domains. All candidate propositions undergo triple filtering via truth table verification, CFG syntax checks, and LLM semantic validation.
    - **Design Motivation**: Natural language propositions often imply multiple interpretations. Expanding propositions into a structured semantic space allows subsequent reasoning to proceed from multiple perspectives rather than locking into a single interpretation prematurely.

2.  **Logical Reasoning Stage**:
    - **Function**: Executes formalized symbolic deduction on the propositions within the semiotic square.
    - **Mechanism**: Contains three functional units—the Translator converts natural language premises into FOL (using unified mapping specifications: entities $\rightarrow$ unary predicates, actions $\rightarrow$ binary predicates, evaluative properties $\rightarrow$ predicates); the Planner builds a reasoning blueprint (setting goals, selecting premises, identifying inference rules like Modus Ponens/Tollens); the Solver executes step-by-step deduction according to the plan, outputting transparent reasoning trajectories and determination results (True/False/Uncertain).
    - **Design Motivation**: By combining the natural language understanding of LLMs with the rigorous deduction of symbolic logic, it compensates for the unreliability of pure LLM reasoning.

3.  **Reflective Verification Stage**:
    - **Function**: Cross-verifies conclusions through the structural relationships of the semiotic square to ensure consistency.
    - **Mechanism**: A three-layer progressive mechanism—(a) **Direct Parsing**: Adopted directly when $S_1$ and $\lnot S_1$ provide complementary determinations; (b) **Quick Reflection**: When one side is Uncertain, the LLM analyzes the internal consistency of the reasoning trajectory; (c) **Deep Reflection**: When $S_1$ and $\lnot S_1$ yield the same determination (contradiction), it utilizes entailment relations $S_1 \Rightarrow \lnot S_2$ and $S_2 \Rightarrow \lnot S_1$ to introduce reasoning results from $S_2$ and $\lnot S_2$ for arbitration.
    - **Design Motivation**: The structural relationships of the semiotic square (contradiction, contrariety, entailment) provide a natural cross-verification framework that effectively detects and corrects reasoning errors.

### Loss & Training

LogicAgent is a training-free reasoning framework implemented via prompt engineering based on existing LLMs (Qwen2.5-32B, GPT-4o). CFG syntax checks use the `nltk` library, and the decoding temperature is set to 0.

## Key Experimental Results

### Main Results

| Benchmark | LogicAgent | Best Baseline | Gain |
| :--- | :--- | :--- | :--- |
| RepublicQA (Qwen2.5) | 82.50 | 76.00 (SymbCoT) | +6.50 |
| RepublicQA (GPT-4o) | 87.00 | 82.50 (Aristotle) | +4.50 |
| ProntoQA | 97.80 | 95.20 (SymbCoT) | +2.60 |
| ProofWriter | 71.95 | 64.67 (SymbCoT) | +7.28 |
| FOLIO | 79.90 | 72.54 (ToT) | +7.97 |
| ProverQA | 68.60 | 62.40 (Logic-LM) | +6.20 |
| **Average** | **79.56** | - | **+7.05** |

### Ablation Study

| Configuration | Avg | Description |
| :--- | :--- | :--- |
| Full LogicAgent | 76.36 | Complete model |
| w/o Square | 67.58 | Largest drop; multi-perspective reasoning is crucial |
| w/o Plan | 69.70 | Planning significantly aids complex reasoning |
| w/o Reflect | - | Reflective verification further improves reliability |

### Key Findings
- The semantic complexity metrics of RepublicQA comprehensively exceed existing benchmarks (FKGL=11.94 at college level; contrary construction rate of 0.70 far exceeds 0-0.30 in other benchmarks).
- Logic-LM's performance on RepublicQA is close to the naive baseline, indicating that pure symbolic enhancement fails under semantic ambiguity.
- The semiotic square contributes the most (an average drop of approximately 8.8 points when removed), validating the core value of multi-perspective reasoning.
- LogicAgent shows consistent improvements across both simple (ProntoQA) and complex (ProverQA) benchmarks, demonstrating good generalization.

## Highlights & Insights
- **Interdisciplinary Fusion of Linguistics and AI Reasoning**: Migrating the Greimas Semiotic Square from structuralist semantics to computational logical reasoning is both theoretically deep and practically effective.
- **First Systematization of Semantic Complexity**: Defines multi-dimensional semantic complexity metrics and constructs a dedicated benchmark, filling an important gap.
- **Progressive Design of the Three-Layer Reflection Mechanism**: From direct parsing to quick reflection and then deep reflection, it precisely matches different inconsistency patterns.
- **Rigor of the Existential Import Check (EIC)**: Ensures the logical correctness of contrary relations within the FOL framework, avoiding logical loopholes in empty domains.

## Limitations & Future Work
- RepublicQA focuses on philosophical/ethical domains, with limited coverage of scientific and commonsense reasoning.
- The framework depends on the LLM's ability to correctly execute FOL translation and semiotic square construction; weak models may generate low-quality intermediate results.
- Deep reflection introduces additional reasoning overhead (requiring complete reasoning for $S_2$ and $\lnot S_2$).
- The three-valued logic setting (True/False/Uncertain) might not be flexible enough; future work could explore probabilistic reasoning.
- LogicAgent could be combined with test-time compute in the future.

## Related Work & Insights
- **vs SymbCoT**: SymbCoT combines CoT and symbolic reasoning but lacks multi-perspective verification; LogicAgent significantly improves via semiotic square cross-verification.
- **vs Logic-LM**: Logic-LM directly calls FOL solvers, but its effectiveness is limited under semantic ambiguity; LogicAgent first addresses ambiguity through semantic structuring.
- **vs Aristotle**: Aristotle combines aggregation and symbolic reasoning but lacks a systematic reflection mechanism; LogicAgent's three-layer reflection is more effective in contradiction detection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The application of the Greimas Semiotic Square in AI reasoning is highly original, and the RepublicQA benchmark is a unique contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 benchmarks, multiple baselines, including ablation analysis, though model coverage (only 2 LLMs) is slightly limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, and definitions/theorems are clearly stated, though the heavy use of symbols results in a high barrier to entry.
- Value: ⭐⭐⭐⭐ Introduces the dimension of semantic complexity to logical reasoning; both the framework and benchmark provide independent contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[ACL 2026\] Calibration-Aware Policy Optimization for Reasoning LLMs](calibration-aware_policy_optimization_for_reasoning_llms.md)
- [\[ICLR 2026\] ActivationReasoning: Logical Reasoning in Latent Activation Spaces](../../ICLR2026/llm_reasoning/activationreasoning_logical_reasoning_in_latent_activation_spaces.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)

</div>

<!-- RELATED:END -->
