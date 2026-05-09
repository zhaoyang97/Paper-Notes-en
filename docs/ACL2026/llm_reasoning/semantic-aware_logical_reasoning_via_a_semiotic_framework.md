---
title: >-
  [Paper Note] Semantic-Aware Logical Reasoning via a Semiotic Framework
description: >-
  [ACL 2026][LLM Reasoning][symbolic reasoning] This paper proposes LogicAgent, a logical reasoning framework grounded in the Greimas Semiotic Square. By performing multi-perspective semantic analysis and reflective verification, LogicAgent achieves state-of-the-art logical reasoning performance under the dual challenges of semantic and logical complexity.
tags:
  - ACL 2026
  - LLM Reasoning
  - symbolic reasoning
  - Greimas semiotic square
  - logical reasoning
  - semantic complexity
  - multi-perspective reasoning
date: 2026-05-08
content_hash: e156995162aaf7a9
---

# Semantic-Aware Logical Reasoning via a Semiotic Framework

**Conference**: ACL 2026
**arXiv**: [2509.24765](https://arxiv.org/abs/2509.24765)
**Code**: [GitHub](https://github.com/AI4SS/Logic-Agent)
**Area**: LLM Reasoning / Logical Reasoning
**Keywords**: symbolic reasoning, Greimas semiotic square, logical reasoning, semantic complexity, multi-perspective reasoning

## TL;DR

This paper proposes LogicAgent, a logical reasoning framework grounded in the Greimas Semiotic Square. By performing multi-perspective semantic analysis and reflective verification, LogicAgent achieves state-of-the-art logical reasoning performance under the dual challenges of semantic and logical complexity.

## Background & Motivation

**State of the Field**: Logical reasoning is a core capability of LLMs. Existing approaches fall into three main categories: linear reasoning (CoT), aggregation-based reasoning (ToT/CR and other multi-trajectory methods), and symbolic reasoning (e.g., Logic-LM combined with FOL solvers). These methods perform well on benchmarks with clear logical structure.

**Limitations of Prior Work**: Existing methods focus almost exclusively on **logical complexity** (reasoning depth, number of steps) while neglecting **semantic complexity** (abstract propositions, ambiguous contexts, opposing stances). In real-world reasoning, semantic ambiguity and abstraction are often intertwined with logical complexity—for instance, a philosophical proposition such as "Is justice always beneficial?" demands not only deep reasoning but also multi-perspective interpretation of abstract concepts.

**Root Cause**: Most existing benchmarks (ProntoQA, ProofWriter, etc.) are template-generated with clear, unambiguous propositions, and thus fail to assess model robustness under semantically complex scenarios. In practice, the coupling of semantic and logical complexity constitutes the true challenge of reasoning.

**Paper Goals**: To construct a reasoning framework that addresses both semantic and logical complexity simultaneously, and to provide a benchmark capable of evaluating this coupled challenge.

**Core Idea**: Drawing on the Greimas Semiotic Square from structuralist semantics, the framework expands a proposition into a four-element structure—original proposition $S_1$, contradiction $\lnot S_1$, contrary $S_2$, and contrary of the contradiction $\lnot S_2$—enabling multi-perspective reasoning and cross-validation to enhance robustness under semantic ambiguity.

## Method

### Overall Architecture

LogicAgent proceeds in three stages: (1) **Semantic Structuring**: expanding propositions into four interrelated propositions of the semiotic square and verifying FOL consistency; (2) **Logical Reasoning**: translating premises into FOL, planning reasoning paths, and executing symbolic deduction; (3) **Reflective Verification**: ensuring consistency of conclusions through a three-tier reflection mechanism (direct resolution, quick reflection, and deep reflection).

### Key Designs

1. **Semantic Structuring Stage**:

    - Function: Expands a single proposition system into a semantically interrelated four-element structure, surfacing potential interpretive ambiguities.
    - Mechanism: Given proposition $S_1$, the framework constructs its contradictory $\lnot S_1$ (strict negation), its contrary $S_2$ (cannot both be true but can both be false), and $\lnot S_2$. Construction follows a unified rule table covering six forms: universal, existential, implication, conjunction, disjunction, and biconditional. An Existential Import Check (EIC) is introduced to ensure logical correctness of contrary relations and avoid vacuous truth in empty domains. All candidate propositions pass through three-stage filtering: truth-table verification, CFG syntactic checking, and LLM semantic validation.
    - Design Motivation: Natural language propositions frequently admit multiple interpretations. Expanding a proposition into a structured semantic space enables subsequent reasoning to proceed across multiple perspectives rather than prematurely committing to a single interpretation.

2. **Logical Reasoning Stage**:

    - Function: Executes formal symbolic deduction over the propositions within the semiotic square.
    - Mechanism: Three functional units are employed—a **Translator** that converts natural language premises into FOL (using a unified mapping scheme: entities → unary predicates, actions → binary predicates, evaluative properties → predicates); a **Planner** that constructs a reasoning blueprint (setting goals, selecting premises, identifying inference rules such as Modus Ponens and Modus Tollens); and a **Solver** that executes stepwise deduction according to the plan, producing transparent reasoning traces and verdicts (True/False/Uncertain).
    - Design Motivation: Combining LLMs' natural language understanding with the rigorous deduction of symbolic logic compensates for the unreliability of purely neural reasoning.

3. **Reflective Verification Stage**:

    - Function: Cross-validates conclusions via the structural relations of the semiotic square to ensure consistency.
    - Mechanism: A three-tier progressive mechanism is applied—(a) **Direct Resolution**: when $S_1$ and $\lnot S_1$ yield complementary verdicts, the result is adopted directly; (b) **Quick Reflection**: when one party returns Uncertain, the LLM analyzes the internal consistency of the reasoning trace; (c) **Deep Reflection**: when $S_1$ and $\lnot S_1$ yield identical verdicts (a contradiction), the implication relations $S_1 \Rightarrow \lnot S_2$ and $S_2 \Rightarrow \lnot S_1$ are exploited to introduce the reasoning results for $S_2$ and $\lnot S_2$ as arbitration evidence.
    - Design Motivation: The structural relations of the semiotic square (contradiction, contrariety, implication) provide a natural cross-validation framework that effectively detects and corrects reasoning errors.

### Loss & Training

LogicAgent is a training-free reasoning framework implemented via prompt engineering on top of existing LLMs (Qwen2.5-32B, GPT-4o). CFG syntactic checking employs the `nltk` library, and the decoding temperature is set to 0.

## Key Experimental Results

### Main Results

| Benchmark | LogicAgent | Best Baseline | Gain |
|-----------|-----------|---------------|------|
| RepublicQA (Qwen2.5) | 82.50 | 76.00 (SymbCoT) | +6.50 |
| RepublicQA (GPT-4o) | 87.00 | 82.50 (Aristotle) | +4.50 |
| ProntoQA | 97.80 | 95.20 (SymbCoT) | +2.60 |
| ProofWriter | 71.95 | 64.67 (SymbCoT) | +7.28 |
| FOLIO | 79.90 | 72.54 (ToT) | +7.97 |
| ProverQA | 68.60 | 62.40 (Logic-LM) | +6.20 |
| **Average** | **79.56** | - | **+7.05** |

### Ablation Study

| Configuration | Avg | Notes |
|---------------|-----|-------|
| Full LogicAgent | 76.36 | Complete model |
| w/o Square (remove semiotic square) | 67.58 | Largest drop; multi-perspective reasoning is critical |
| w/o Plan (remove reasoning planner) | 69.70 | Planning substantially aids complex reasoning |
| w/o Reflect (remove reflection) | - | Reflective verification further improves reliability |

### Key Findings
- RepublicQA's semantic complexity metrics comprehensively surpass existing benchmarks (FKGL = 11.94, college level; contrary construction rate 0.70, far exceeding the 0–0.30 range of other benchmarks).
- Logic-LM performs close to the naive baseline on RepublicQA, demonstrating that purely symbolic augmentation fails under semantic ambiguity.
- The semiotic square contributes most to performance (removal causes an average drop of approximately 8.8 points), validating the central value of multi-perspective reasoning.
- LogicAgent consistently improves performance on both simple benchmarks (ProntoQA) and complex benchmarks (ProverQA), demonstrating strong generalization.

## Highlights & Insights
- **Interdisciplinary integration of linguistic theory and AI reasoning**: Migrating the Greimas Semiotic Square from structuralist semantics to computational logical reasoning yields both theoretical depth and practical effectiveness.
- **First systematic treatment of semantic complexity**: The paper defines multi-dimensional semantic complexity metrics and constructs a dedicated benchmark, filling an important gap in the field.
- **Progressive design of the three-tier reflection mechanism**: The escalation from direct resolution to quick reflection to deep reflection precisely targets distinct patterns of inconsistency.
- **Rigor of the Existential Import Check (EIC)**: Ensures logical correctness of contrary relations within the FOL framework, preventing logical loopholes in empty domains.

## Limitations & Future Work
- RepublicQA focuses on philosophical and ethical domains, with limited coverage of scientific and commonsense reasoning.
- The framework relies on LLMs to correctly perform FOL translation and semiotic square construction; weaker models may produce low-quality intermediate outputs.
- Deep reflection introduces additional inference overhead (requiring complete reasoning over $S_2$ and $\lnot S_2$).
- The three-valued logic setting (True/False/Uncertain) may lack flexibility; probabilistic reasoning could be explored in future work.
- Future work may integrate the semiotic square with test-time compute strategies.

## Related Work & Insights
- **vs. SymbCoT**: SymbCoT combines CoT with symbolic reasoning but lacks multi-perspective verification; LogicAgent achieves significant improvements through cross-validation via the semiotic square.
- **vs. Logic-LM**: Logic-LM directly invokes FOL solvers and is constrained under semantic ambiguity; LogicAgent addresses ambiguity upstream through semantic structuring.
- **vs. Aristotle**: Aristotle combines aggregation and symbolic reasoning but lacks a systematic reflection mechanism; LogicAgent's three-tier reflection is more effective at detecting contradictions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Application of the Greimas Semiotic Square to AI reasoning is highly original; the RepublicQA benchmark is also a distinctive contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five benchmarks, multiple baselines, and ablation analysis are included, though model coverage (only two LLMs) is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and definitions and theorems are clearly stated, though the dense notation raises the reading barrier.
- Value: ⭐⭐⭐⭐ — Introduces the semantic complexity dimension to logical reasoning; both the framework and the benchmark constitute independent contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning](logical_phase_transitions_understanding_collapse_in_llm_logical_reasoning.md)
- [\[ACL 2026\] Discovering a Shared Logical Subspace: Steering LLM Logical Reasoning via Alignment of Natural-Language and Symbolic Views](discovering_a_shared_logical_subspace_steering_llm_logical_reasoning_via_alignme.md)
- [\[ACL 2026\] OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning](octotools_an_agentic_framework_with_extensible_tools_for_complex_reasoning.md)
- [\[ACL 2026\] Budget-Aware Anytime Reasoning with LLM-Synthesized Preference Data](budget-aware_anytime_reasoning_with_llm-synthesized_preference_data.md)
- [\[ICLR 2026\] Agentified Assessment of Logical Reasoning Agents](../../ICLR2026/llm_reasoning/agentified_assessment_of_logical_reasoning_agents.md)

</div>

<!-- RELATED:END -->
