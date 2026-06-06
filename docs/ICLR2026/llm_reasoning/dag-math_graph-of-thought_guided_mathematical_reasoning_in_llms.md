---
title: >-
  [Paper Note] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs
description: >-
  [ICLR 2026][LLM Reasoning][mathematical reasoning] This work formalizes LLM chain-of-thought reasoning as a rule-based stochastic process over DAGs…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "mathematical reasoning"
  - "DAG"
  - "chain-of-thought"
  - "logical closeness"
  - "evaluation metric"
date: 2026-05-08
content_hash: 836b60ca0360139d
---

# DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs

**Conference**: ICLR 2026  
**arXiv**: [2510.19842](https://arxiv.org/abs/2510.19842)  
**Code**: [https://github.com/YuanheZ/DAG-MATH](https://github.com/YuanheZ/DAG-MATH)  
**Area**: LLM Reasoning  
**Keywords**: mathematical reasoning, DAG, chain-of-thought, logical closeness, evaluation metric

## TL;DR
This work formalizes LLM chain-of-thought reasoning as a rule-based stochastic process over DAGs, proposes *logical closeness* as a metric to assess whether a model arrives at an answer through search or rigorous logical deduction, constructs a gold-standard DAG-MATH benchmark of 2,894 instances, and demonstrates that models with similar PASS@k scores can differ substantially in reasoning faithfulness.

## Background & Motivation
**Background**: LLMs exhibit strong mathematical reasoning performance (o1, R1, Gemini-2.5), with chain-of-thought (CoT) serving as the core strategy. However, the black-box nature of CoT makes it difficult to determine whether a model reaches a correct answer via logical reasoning or via search/heuristics.

**Limitations of Prior Work**: (1) PASS@k only evaluates final answers and ignores the logical consistency of the reasoning process—exploratory search can also yield correct answers. (2) LEAN-based formal verification is rigorous but requires substantial expert effort to formalize problems in advance. (3) Existing graph models (Dziri 2023) rely on deterministic subgraph matching, neglecting diverse sampling and long-range dependencies.

**Key Challenge**: PASS@k may overestimate reasoning ability—models may stumble upon correct answers through exploratory branching rather than strict logical derivation. An evaluation framework situated between free-form CoT and formal proof is needed.

**Goal**: How can the mathematical reasoning capability of LLMs be rigorously defined and evaluated—distinguishing answers obtained via search from those obtained via logical reasoning?

**Key Insight**: CoT is modeled as a stochastic process over a DAG, where nodes represent conclusions of derivation steps and edges represent applications of inference rules. *Logical closure* requires that every intermediate node be consumed by at least one subsequent node, with no idle exploratory branches.

**Core Idea**: Logical closeness over DAGs serves as a measure of whether an LLM is performing genuine logical reasoning rather than mere search.

## Method

### Overall Architecture
A two-phase framework: Phase 1 constructs task-specific DAGs (source nodes = premises, sink nodes = answers, intermediate nodes = derivation steps); Phase 2 has the LLM generate CoT trajectories over the DAG as a rule-based stochastic process, in which a node can only be visited after all its parent nodes have been generated. Three categories of CoT output are defined: perfect reasoning (containing only ancestors of the correct answer), imperfect reasoning (containing irrelevant nodes but ultimately reaching the correct answer), and erroneous reasoning.

### Key Designs

1. **Logical Closeness**:

    - Function: Evaluates whether every node in a CoT trajectory is consumed by subsequent reasoning steps (out-degree ≥ 1).
    - Mechanism: The presence of "dead-end" nodes in a trajectory—nodes that are generated but never referenced by subsequent steps—indicates search/exploration rather than strict reasoning. Perfect reasoning = logical closure + correct final answer.
    - Design Motivation: PASS@k ignores the reasoning process. A toy example shows that in a two-chain DAG, PRR decays exponentially with depth as $(1/2)^{L-1}$, even when final accuracy remains stable at 50%.

2. **Perfect Reasoning Rate (PRR)**:

    - Function: Provides a formal measure of mathematical reasoning capability.
    - Mechanism: $\text{PRR}(\mathbf{x}) = \mathbb{E}[\delta_{\text{close}} \times \delta_{\text{final}}]$—requiring both logical closure and a correct final answer. An AUC variant provides a continuous score by relaxing the closure threshold from 0% to 100%.
    - Distinction from PASS@k: PASS@k requires only $\delta_{\text{final}}=1$; PRR additionally requires $\delta_{\text{close}}=1$.

3. **DAG-MATH Benchmark Construction**:

    - Function: Constructs 2,894 gold-standard CoT instances in DAG format.
    - Mechanism: A three-stage prompting pipeline—(1) prompting the LLM to generate structured CoT with explicit parent-node dependencies annotated at each step; (2) GPT-4-class model validation and correction of DAG structure; (3) human review.
    - Statistical Findings: Harder problems correspond to larger, sparser DAGs with more complex branching structures.

### Loss & Training
N/A (evaluation framework + benchmark; no model training). Few-shot prompting is used to elicit DAG-MATH-formatted CoT from LLMs.

## Key Experimental Results

### Main Results (Cross-Model PRR vs. PASS@1 Comparison)

| Finding | Description |
|---------|-------------|
| Search inflates PASS@1 | Exploratory branching can raise PASS@1 by accidentally discovering correct paths, while PRR remains unchanged. |
| Comparable PRR across models | Even when PASS@k scores differ substantially, the perfect reasoning capability of models within different families is actually quite similar. |
| Perfect reasoning correlates with easy problems | High PRR instances tend to be low-difficulty—indicating that current models still struggle with rigorous logical reasoning on hard problems. |
| Erroneous CoT originates beyond model capacity | Difficulty stems from branching (merging of multiple derivation paths) rather than from linear chain length. |

### Graph-Level Statistical Analysis

| Difficulty | Avg. Node Count | Avg. Edge Count | Branching Factor |
|------------|----------------|----------------|-----------------|
| Easy | Low | Low | Low |
| Hard | High | High but sparse | High |

### Key Findings
- Across MATH-500, GSM8K, AIME24/25, and other benchmarks, PASS@1 gaps among the Gemini, GPT, and Qwen3 families can exceed 10%, while PRR differences are considerably smaller—suggesting that high PASS@1 is driven primarily by search rather than reasoning.
- DAG structural analysis reveals that LLMs fail most often on problems requiring multi-step convergence (merging multiple intermediate conclusions into a single step).
- The gap between PRR and PASS@1 quantifies a model's "exploratory overhead"—a larger gap indicates greater reliance on search over reasoning.

## Highlights & Insights
- **First formal definition of LLM mathematical reasoning capability**: Translates the intuitive notion of "whether the model is truly reasoning" into a computable metric (PRR), analogous to the formalization of memorization/generalization in learning theory.
- **"Goldilocks Principle"**: The framework occupies a practical middle ground between free-form CoT (too permissive) and LEAN formal verification (too strict).
- **Diagnostic value of DAG complexity analysis**: Analyzing the topological structure of DAGs (depth vs. branching) enables precise identification of model weaknesses—whether the deficiency lies in linear reasoning or in multi-path integration.

## Limitations & Future Work
- DAG construction relies on LLMs plus human annotation and is not fully automated.
- Node/edge decomposition may not be unique due to semantic variation; the paper discusses but does not fully resolve this issue.
- Validation is currently limited to mathematical reasoning; extension to other domains (e.g., legal or scientific reasoning) requires adaptation.
- The applicability of PRR to non-deterministic problems (e.g., open-ended reasoning) is limited.

## Related Work & Insights
- **vs. PASS@k**: PRR strictly strengthens PASS@k by requiring not only a correct answer but also a logically closed reasoning process.
- **vs. Dziri et al. (2023) graph matching**: Their approach uses deterministic subgraph matching and does not support multi-path reasoning or probabilistic sampling; DAG-Math employs a stochastic process model.
- **vs. LEAN formal verification**: LEAN is overly strict and requires prior formalization; DAG-Math uses natural language with structural constraints, making it more practical.
- **vs. Kim et al. (2025) Markov chain**: Their approach uses reversible Markov chains and does not support directed acyclic structure or absorbing states.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formal definition of LLM reasoning capability; the DAG + logical closeness framework is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-family model comparison, graph-level statistical analysis, and evaluation across multiple benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clearly defined concepts, and intuitive toy examples.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction for reasoning evaluation and may influence evaluation standards in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Verifying Chain-of-Thought Reasoning via Its Computational Graph](verifying_chain-of-thought_reasoning_via_its_computational_graph.md)
- [\[ICLR 2026\] Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?](are_reasoning_llms_robust_to_interventions_on_their_chain-of-thought.md)
- [\[ICML 2026\] Clustering as Reasoning: A $k$-Means Interpretation of Chain-of-Thought Graph Learning](../../ICML2026/llm_reasoning/clustering_as_reasoning_a_k-means_interpretation_of_chain-of-thought_graph_learn.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)

</div>

<!-- RELATED:END -->
