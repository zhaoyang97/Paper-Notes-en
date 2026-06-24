---
title: >-
  [Paper Note] Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework
description: >-
  [ACL 2025][Reasoning][Logical Reasoning] This paper proposes Aristotle, a logical reasoning framework that fully integrates symbolic expressions and logical rules into every stage of the Decompose-Search-Resolve process. Utilizing three core components—a logical decomposer, a search router, and a resolver—it achieves logic-complete reasoning, outperforming SOTA on several logical reasoning benchmarks with an average improvement of 4.5% on GPT-4 and 5.4% on GPT-4o.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Logical Reasoning"
  - "Symbolic Reasoning"
  - "Proof by Contradiction"
  - "Decompose-Search-Resolve"
  - "Symbolic Expressions"
date: 2026-05-08
content_hash: 442e25fbeb49dd98
---

# Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework

**Conference**: ACL 2025  
**arXiv**: [2412.16953](https://arxiv.org/abs/2412.16953)  
**Code**: [http://llm-symbol.github.io/Aristotle](http://llm-symbol.github.io/Aristotle)  
**Area**: LLM Reasoning  
**Keywords**: Logical Reasoning, Symbolic Reasoning, Proof by Contradiction, Decompose-Search-Resolve, Symbolic Expressions  

## TL;DR

This paper proposes Aristotle, a logical reasoning framework that fully integrates symbolic expressions and logical rules into every stage of the Decompose-Search-Resolve process. Utilizing three core components—a logical decomposer, a search router, and a resolver—it achieves logic-complete reasoning, outperforming SOTA on several logical reasoning benchmarks with an average improvement of 4.5% on GPT-4 and 5.4% on GPT-4o.

## Background & Motivation

**Background**:
   - Reasoning methods such as CoT, ToT, and GoT have made significant progress in general reasoning tasks.
   - SymbCoT introduced symbolic expressions into CoT reasoning, achieving SOTA in logical reasoning.
   - However, logical reasoning requires the most rigorous evidence and logical strictness, making it one of the most challenging tasks in LLM reasoning.

**Limitations of Prior Work**:
   - **Decomposition Stage**: LLMs rely on language token relationships rather than logical structures to decompose problems, leading to disjointed sub-problems.
   - **Search Stage**: Existing path search relies on unreliable evaluators to select nodes, leading to error propagation (search error rate of 15.0%).
   - **Resolution Stage**: Using simple text prompts to guide LLMs in solving sub-problems frequently produces logical errors (reasoning error rate of 28.4%).
   - **Efficiency Issues**: Erroneous nodes waste computational resources, and unreliable evaluators lead to unnecessary node exploration.

**Key Challenge**:
   - General reasoning methods (e.g., ToT) ignore the intrinsic structure of logical tasks and fail to effectively integrate logical rules into the Decompose-Search-Resolve framework.
   - Consequently, they are neither efficient nor accurate.

**Goal**:
   - How to fully utilize logical structures and symbolic rules in every stage of logical reasoning (decomposition, search, resolution).
   - How to reduce search errors and reasoning errors while improving efficiency.

**Key Insight**:
   - Formal logical methodology: translating using Prolog syntax, decomposing with Conjunctive Normal Form (CNF), searching with proof by contradiction, and resolving with the resolution principle.
   - Systematically embedding the entire methodology of classical logical reasoning into the LLM reasoning framework.

**Core Idea**:
   - Fully integrating symbolic logical expressions and rules (translation $\rightarrow$ CNF decomposition $\rightarrow$ proof by contradiction search $\rightarrow$ resolution principle) into every stage of LLM reasoning.

## Method

### Overall Architecture

Aristotle consists of four modules:
1. **Translator**: Translates natural language premises and questions into a Prolog symbolic format.
2. **Decomposer**: Normalizes symbolic expressions into CNF (Conjunctive Normal Form), eliminating quantifiers.
3. **Search Router**: Searches complement clauses based on proof by contradiction.
4. **Resolver**: Eliminates complementary terms based on the resolution principle and generates new clauses.

The reasoning process consists of three steps: initialization $\rightarrow$ iterative search & resolution $\rightarrow$ drawing conclusions.

### Key Designs

1. **Logical Decomposer**:

    - **Function**: Converts premises and queries into CNF through Normalization and Skolemization.
    - **Mechanism**: For example, $\forall x (P(x) \rightarrow Q(x))$ is decomposed into $\neg P(x) \vee Q(x)$.
    - **Design Motivation**: CNF simplifies the reasoning process, making it easier to apply formal resolution rules.

2. **Logical Search Router**:

    - **Function**: Adopts proof by contradiction to search for premise clauses complementary to the current clause.
    - **Mechanism**: If the current clause contains $P(x, \text{True})$, it searches for a premise clause containing $P(x, \text{False})$.
    - **Design Motivation**: Proof by contradiction directly searches for logical conflicts, avoiding search errors that stem from relying on unreliable evaluators; this is a **rule-based** module rather than an LLM-based one.

3. **Logical Resolver**:

    - **Function**: Eliminates complementary terms based on Robinson's resolution principle and connects the remaining terms to generate new clauses.
    - **Mechanism**: $C_{\text{current}} = P(x,T) \vee A$, $C_{\text{complement}} = P(x,F) \vee B$ $\rightarrow$ $C_{\text{resolved}} = A \vee B$.
    - **Design Motivation**: The resolution principle provides clear and concise instructions, minimizing reasoning errors.

4. **Dual-Path Reasoning**:

    - **Function**: Simultaneously performs proof by contradiction reasoning on both paths: $S_n$ and $\neg S_n$.
    - **Mechanism**: Learns the answer based on four combinations: True/False/Unknown/Self-Contradictory.
    - **Design Motivation**: A single path cannot distinguish between all four logical states.

### Loss & Training

- This work is a prompting method and **does not require training**.
- The Search Router is a rule-based module (non-LLM), while other modules are implemented via LLM in-context learning.
- The maximum number of iterations $I_{\text{max}}$ controls the reasoning depth.

## Key Experimental Results

### Main Results

Tested on three logical reasoning datasets, using GPT-4 and GPT-4o:

| Method | ProntoQA | ProofWriter | LogicNLI | Average (GPT-4) |
|------|----------|-------------|----------|-------------|
| CoT | 98.9 | 68.1 | 51.0 | 72.6 |
| CoT-SC | 93.4 | 69.3 | 57.3 | 73.3 |
| ToT | 97.6 | 70.3 | 52.7 | 73.5 |
| SymbCoT | 99.6 | 82.5 | 59.0 | 80.4 |
| **Aristotle** | **99.6** | **86.8** | **68.3** | **84.9** |

GPT-4o Results:

| Method | ProntoQA | ProofWriter | LogicNLI | Average |
|------|----------|-------------|----------|------|
| SymbCoT | 99.4 | 82.3 | 58.7 | 80.1 |
| **Aristotle** | **99.6** | **88.5** | **70.7** | **86.3** |

- On GPT-4, the average gain is 4.5% (vs. SymbCoT), and on GPT-4o, the gain is 5.4% (+6.2 on ProofWriter, +12.0 on LogicNLI).

Cross-model generalization (Claude-3.5-Sonnet / Llama-3.1-405B):

| Model | ProntoQA | ProofWriter | LogicNLI | Average |
|------|----------|-------------|----------|------|
| Claude + Aristotle | 99.0 | 86.5 | 61.3 | 82.3 (+5.3) |
| Llama + Aristotle | 98.4 | 89.5 | 69.0 | 85.6 (+12.1) |

- Llama-3.1-405B achieves a gain of **20.0%** on ProofWriter and **8.7%** on LogicNLI.

### Key Findings

1. **More complex logical scenarios show larger gains**: There is almost no gain on simple ProntoQA scenarios, while the gain is most significant on complex LogicNLI scenarios (6.3–12.0%).
2. **Search Router is the most critical**: In the ablation study, removing the Search Router leads to a drop of 50.8% on ProofWriter and 31.6% on LogicNLI.
3. **Decomposer is more important in complex scenarios**: LogicNLI contains complex logical structures like "either...or...", relying heavily on the decomposer.
4. **Cross-model framework generalization**: It is effective across GPT-4, GPT-4o, Claude-3.5, and Llama-405B.

## Highlights & Insights

- **Modern revival of classical logical methodology**: Combines classical formal logic methods such as resolution principle, CNF conversion, and proof by contradiction with LLMs.
- **Near logic-complete**: Covers all four answer types: True/False/Unknown/Self-Contradictory.
- **Rule-based Search Router**: Non-LLM generated, which avoids the unreliability of LLM searches.
- **Ingenious dual-path proof-by-contradiction design**: A single path cannot distinguish between all four logical states, which is elegantly solved by the dual-path + Equation (1).
- **In-depth error type analysis**: The quantitative analysis of 28.4% reasoning errors + 15.0% search errors is highly persuasive.

## Limitations & Future Work

1. Relying on LLMs for translation/decomposition/resolution can still introduce errors.
2. It only handles a subset of propositional and first-order logic, and its applicability to more complex logical forms (e.g., modal logic, temporal logic) remains unknown.
3. The maximum number of iterations $I_{\text{max}}$ for proof by contradiction needs to be preset, which may affect the coverage of complex reasoning chains.
4. The Prolog translation step requires LLMs with strong symbolic understanding capability, and its effectiveness has not been verified on smaller models.
5. Computational costs are not detailed; dual-path reasoning might double the number of API calls.

## Related Work & Insights

- **SymbCoT (Xu et al., 2024)**: Delegates symbolic translation and logical resolution to the LLM itself; this work systemizes it further.
- **ToT (Yao et al., 2023)**: A tree-search general reasoning method, but inferior to specialized designs on logical reasoning.
- **Logic-LM (Pan et al., 2023)**: Employs an external logic solver, which can lead to information loss.
- **Insight**: Task-specific structured reasoning frameworks (e.g., resolution principles in logical reasoning) often outperform general search frameworks.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 8 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Practical Value | 7 |
| **Total Score** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Semantic-Aware Logical Reasoning via a Semiotic Framework](../../ACL2026/llm_reasoning/semantic-aware_logical_reasoning_via_a_semiotic_framework.md)
- [\[ACL 2025\] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](logicpro_program_guided_reasoning.md)
- [\[ACL 2025\] Enhancing Retrieval Systems with Inference-Time Logical Reasoning](enhancing_retrieval_systems_with_inference-time_logical_reasoning.md)
- [\[ACL 2025\] BPP-Search: Enhancing Tree of Thought Reasoning for Mathematical Modeling Problem Solving](bpp-search_enhancing_tree_of_thought_reasoning_for_mathematical_modeling_problem.md)
- [\[ACL 2025\] Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](self-correction_is_more_than_refinement_a_learning_framework_for_visual_and_lang.md)

</div>

<!-- RELATED:END -->
