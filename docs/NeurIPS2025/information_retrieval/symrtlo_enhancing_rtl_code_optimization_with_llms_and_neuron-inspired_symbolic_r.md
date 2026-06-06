---
title: >-
  [Paper Note] SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning
description: >-
  [NeurIPS 2025][Information Retrieval & RAG][RTL code optimization] This paper proposes SymRTLO, the first neurosymbolic framework integrating LLMs with symbolic reasoning for RTL code optimization. By combining retrieval…
tags:
  - "NeurIPS 2025"
  - "Information Retrieval & RAG"
  - "RTL code optimization"
  - "neurosymbolic reasoning"
  - "LLM"
  - "finite state machine"
  - "hardware design automation"
date: 2026-05-08
content_hash: e4527b514c68942d
---

# SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2504.10369](https://arxiv.org/abs/2504.10369)  
**Code**: Not available  
**Area**: Information Retrieval
**Keywords**: RTL code optimization, neurosymbolic reasoning, LLM, finite state machine, hardware design automation

## TL;DR

This paper proposes SymRTLO, the first neurosymbolic framework integrating LLMs with symbolic reasoning for RTL code optimization. By combining retrieval-augmented optimization rules, AST template-guided code generation, and an FSM symbolic system, SymRTLO achieves improvements of up to 43.9%, 62.5%, and 51.1% in power, performance, and area (PPA), respectively.

## Background & Motivation

Register Transfer Level (RTL) optimization is a cornerstone of modern chip design flows, as decisions made at the RTL stage directly affect downstream processes such as synthesis and place-and-route. Nevertheless, RTL optimization remains highly labor-intensive—engineers must iterate through multiple rounds of synthesis and layout feedback, with a single synthesis run potentially taking hours or even days.

Limitations of existing approaches:
- **Compiler-based methods** (e.g., Synopsys DC): Rely on predefined heuristic rules, struggle with unconventional design patterns and complex constraints, and face inherent conflicts among optimization objectives (power vs. latency vs. area).
- **LLM-based methods** (e.g., RTLRewriter): Leverage LLMs to automatically rewrite RTL code, but suffer from an **alignment problem**—generated code frequently deviates from optimization objectives, producing incomplete or incorrect results. These methods still depend on multi-round synthesis feedback and do not address the long design cycle issue.

The authors substantiate the limitations of LLMs through a motivating experiment: when tasked with optimizing an 11-state FSM design (with a detailed state-reduction algorithm provided), GPT-O1 reduced it to only 10 states with negligible PPA improvement, whereas algorithm-driven optimization reduced it to 4 states with substantial PPA gains. This demonstrates that LLMs are insufficient for complex logical reasoning and code alignment.

## Method

### Overall Architecture

SymRTLO takes a Verilog RTL module and a user-specified optimization objective (e.g., low power) as input. An **LLM scheduler** analyzes circuit characteristics and decides whether to apply dataflow optimization, control flow optimization, or both. Dataflow optimization extracts optimization rules via a search engine with RAG and constructs AST templates; control flow optimization handles FSM transformations using an LLM-driven symbolic system. The results of both paths are combined, and a verification system ensures functional correctness.

### Key Designs

1. **Optimization Rule Search Engine and RAG System**

   RTL optimization knowledge scattered across textbooks, lecture notes, and design manuals is consolidated into a unified knowledge base. An LLM structures the raw data into an optimization rule library, where each rule contains a description, applicable objectives (area/power/timing), category, and AST template construction instructions.

   **Conflict resolution**: Different optimization patterns may conflict in objectives (e.g., pipelining reduces latency but increases area; resource sharing reduces area but increases latency). The Elbow Method is applied to analyze similarity scores between the query and candidate rules, identifying a natural cutoff point:

   $i^* = \arg\max_{1\leq i < M} (s_i - s_{i+1})$

   Only rules whose cosine similarity exceeds the threshold $\tau_{\text{elbow}}$ are selected:

   $\text{sim}(\mathbf{e}_{\text{query}}, \mathbf{e}_{\text{rule}}) = \frac{\mathbf{e}_{\text{query}} \cdot \mathbf{e}_{\text{rule}}}{|\mathbf{e}_{\text{query}}||\mathbf{e}_{\text{rule}}|} \geq \tau_{\text{elbow}}$

   This automatically balances comprehensiveness and precision, avoiding the introduction of conflicting rules.

2. **AST Template Construction and Application**

   For rules with detailed template instructions, the LLM generates AST-based templates that formalize optimization operations as AST node matching and transformation:
   - A matching condition $\Phi: \mathcal{A} \rightarrow \{\text{true}, \text{false}\}$ determines whether a node requires optimization.
   - A transformation rule $\tau: \{a \in \mathcal{A} | \Phi(a) = \text{true}\} \rightarrow \mathcal{A}$ replaces the node with an optimized AST subtree.

   Advantages of AST templates: (1) precise structural representation; (2) each template targets a single objective, making it concise and easy to generate; (3) modular selection to balance conflicting objectives. The LLM selects the application order of templates based on design requirements (not a fixed sequence), and a feedback loop allows re-selection upon failure.

3. **FSM Control Flow Optimization (Symbolic System)**

   Minimizing a finite state machine $M = (Q, \Sigma, \delta, q_0, F)$ under partial specification is NP-complete ($O(2^{|Q|})$), making general-purpose AST scripts insufficient. Therefore:
   - The LLM converts the circuit into a symbolic representation focused solely on FSM components (isolating states, transitions, and outputs).
   - The LLM dynamically generates minimization scripts customized for the specific FSM structure and constraints, rather than applying a one-size-fits-all approach.
   - Data-path constraints $\phi: Q \times D \rightarrow B$ are handled to prevent pure FSM optimization from ignoring data-path side effects.

### Loss & Training

SymRTLO is not an end-to-end trained system. It uses GPT-4o as the reasoning engine, Pyverilog for AST extraction, Yosys+ABC for logic equivalence checking, and Synopsys Formality for timing equivalence verification. The verification pipeline adopts a two-stage strategy: LLM-generated testbenches perform rapid functional correctness screening, followed by formal equivalence checking for passing candidates.

## Key Experimental Results

### Main Results

PPA comparison on FSM designs (5 test cases, Synopsys DC):

| Method | Power↓ | Time↓ | Area↓ |
|--------|:---:|:---:|:---:|
| Original | baseline | baseline | baseline |
| GPT-O1 | negligible | negligible | partial degradation |
| GPT-4o | negligible | negligible | partial improvement |
| RTLRewriter | partial improvement | partial degradation | partial improvement |
| **SymRTLO** | **↓30.95~57.14%** | **0~48.00%** | **↓50.67~54.46%** |

Functional correctness (Pass Rate):

| Method | Pass@1 | Pass@5 | Pass@10 |
|--------|:---:|:---:|:---:|
| **SymRTLO** | **97.5** | **100.0** | **100.0** |
| GPT-4o | 45.9 | 60.0 | 72.7 |
| GPT-4-Turbo | 42.9 | 62.7 | 81.8 |
| RTL-Coder DeepSeek | 8.8 | 18.2 | 27.3 |
| Verigen-16B | 0.0 | 0.0 | 0.0 |

### Ablation Study

Wire/Cell ratio comparison (11 benchmarks, Yosys):

| Method | Wire Ratio | Cell Ratio | Note |
|--------|:---:|:---:|------|
| Yosys baseline | 1.00 | 1.00 | reference |
| GPT-4-Turbo | 0.87 | 1.10 | unstable |
| RTLRewriter† | 0.69 | 0.77 | published SOTA |
| **SymRTLO** | **0.63** | **0.67** | new SOTA |

Component ablation:

| Configuration | Effect |
|---------------|--------|
| Full SymRTLO | optimal PPA |
| w/o AST module | significant performance drop |
| w/o FSM symbolic system | significant performance drop |
| w/o objective-driven search engine | significant performance drop |
| GPT-4o only | negligible improvement |

### Key Findings

- SymRTLO achieves 97.5% Pass@1, far surpassing GPT-4o's 45.9%, demonstrating that symbolic constraints substantially improve functional correctness.
- In FSM optimization, GPT-O1 fails to correctly execute the algorithm even when provided with detailed instructions—symbolic execution is indispensable.
- All three components (AST templates, FSM symbolic system, objective-driven search engine) are necessary, each contributing significantly.
- When combined with compiler-based optimization, SymRTLO still delivers an additional 36.2% and 35.66% improvement in power and area, respectively.
- Overall average PPA improvements: 40.96% in power, 17.02% in delay, and 38.05% in area.

## Highlights & Insights

- SymRTLO is the first neurosymbolic RTL optimization framework, embodying a clear division of labor: LLMs for reasoning and decision-making, symbolic systems for precise execution.
- It addresses the most critical alignment problem in hardware code generation with LLMs—AST templates provide structural guarantees.
- The objective-driven conflict resolution mechanism has practical engineering value, as the power–latency–area trade-off is a central challenge in chip design.
- A 97.5% first-pass rate substantially reduces synthesis iterations, directly shortening the design cycle.

## Limitations & Future Work

- Dependence on the GPT-4o API introduces non-trivial cost and latency.
- The verification pipeline may require stronger tool support for complex scenarios such as asynchronous resets and CDC paths.
- The generalizability of the optimization rule library is limited by the knowledge sources collected.
- Scalability to very large-scale chip designs remains to be validated.
- Integration with reinforcement learning to automatically discover new optimization patterns is a promising direction.

## Related Work & Insights

- Compared with LLM-based code generation methods such as ChipNeMo and VeriGen—these lack verification, rule alignment, and conflict resolution capabilities.
- Compared with RTLRewriter—although it employs RAG, it still relies on multi-round synthesis feedback and cannot resolve the alignment problem.
- The neurosymbolic integration paradigm is generalizable to other code generation scenarios requiring precise logical reasoning, such as formal verification and FPGA synthesis.
- The concept of using AST templates as a "structural cage" to constrain LLM outputs offers broadly applicable reference value.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First neurosymbolic RTL optimization framework; the problem insight is deep and the design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers both functional correctness and PPA evaluation with complete ablations, though some comparisons with RTLRewriter are not directly apples-to-apples.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with thorough method descriptions.
- Value: ⭐⭐⭐⭐⭐ Addresses a practical pain point in the EDA domain with direct engineering value for chip design automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](../../ACL2026/information_retrieval/an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[NeurIPS 2025\] Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs](is_prm_necessary_problem-solving_rl_implicitly_induces_prm_capability_in_llms.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](../../ACL2026/information_retrieval/enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[AAAI 2026\] ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](../../AAAI2026/information_retrieval/comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)

</div>

<!-- RELATED:END -->
