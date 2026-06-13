---
title: >-
  [Paper Note] Are Language Models Efficient Reasoners? A Perspective from Logic Programming
description: >-
  [NeurIPS 2025][LLM/NLP][reasoning efficiency] This paper proposes a framework for evaluating LLM reasoning *efficiency* (rather than correctness alone) from a logic programming perspective. By mapping natural language pr…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "reasoning efficiency"
  - "logic programming"
  - "irrelevant information"
  - "proof length"
  - "deductive reasoning"
date: 2026-05-08
content_hash: 0f3d254834ce5b57
---

# Are Language Models Efficient Reasoners? A Perspective from Logic Programming

**Conference**: NeurIPS 2025
**arXiv**: [2510.25626](https://arxiv.org/abs/2510.25626)  
**Code**: [GitHub](https://github.com/rycolab/reasoning-efficiency)  
**Area**: LLM/NLP
**Keywords**: reasoning efficiency, logic programming, irrelevant information, proof length, deductive reasoning

## TL;DR
This paper proposes a framework for evaluating LLM reasoning *efficiency* (rather than correctness alone) from a logic programming perspective. By mapping natural language proofs to logic program proofs via verbalized logic programs, the authors find that current LLMs not only suffer accuracy degradation on math problems containing irrelevant axioms, but also exhibit severely inefficient reasoning—more than half of all reasoning steps are unnecessary.

## Background & Motivation
**Background**: LLM reasoning evaluation predominantly focuses on final-answer correctness, neglecting the efficiency of the reasoning process. Recent reasoning models (e.g., o1, DeepSeek-R1) frequently generate far more reasoning steps than necessary.

**Limitations of Prior Work**: (1) Using token count as an efficiency metric conflates two distinct sources: "unnecessary reasoning steps" and "verbose natural language expression"; (2) no formal framework exists to define "efficient reasoning"; (3) existing datasets (e.g., GSM8k-IC) add only a single piece of irrelevant information, precluding systematic evaluation.

**Key Challenge**: Real-world reasoning involves large amounts of irrelevant available information; efficient reasoning requires identifying and ignoring distractors. However, LLMs lack the systematic ability to distinguish relevant from irrelevant information.

**Goal**: (1) Define and measure reasoning efficiency; (2) systematically evaluate LLM accuracy and efficiency in the presence of irrelevant information.

**Key Insight**: Logic programming provides a clean formal framework—each reasoning step corresponds to a hyperedge, and the shortest proof constitutes the most efficient derivation. Verbalized logic programs enable mapping natural language proofs back to formal derivation steps.

**Core Idea**: Use verbalized logic programs as a bridge between natural language reasoning and formal derivation, measuring LLM reasoning efficiency relative to the shortest proof as a reference.

## Method

### Overall Architecture
(1) Construct verbalized GSM logic programs: generate logic programs from GSM-style math problems, associating each theorem with a natural language description; (2) inject irrelevant axioms: introduce irrelevant information with varying degrees of semantic overlap with the target theorem; (3) evaluate: map LLM-generated CoT proofs back to logic program theorems and compare against the shortest proof.

### Key Designs

1. **Verbalized Logic Program**:

    - **Function**: Associates each theorem in a logic program with a set of natural language strings, enabling bidirectional mapping between formal and natural language reasoning.
    - **Mechanism**: Constructs a typed logic program comprising axioms (known facts), inference rules (e.g., addition, multiplication), and a goal theorem, with each element linked to a corresponding natural language template.
    - **Design Motivation**: To disentangle "unnecessary reasoning steps" from "verbose expression"—measuring only the former.

2. **Systematic Injection of Irrelevant Information**:

    - **Function**: Adds a controlled number of irrelevant axioms to GSM problems while controlling the degree of semantic overlap between irrelevant axioms and the target problem.
    - **Mechanism**: Three overlap types—(1) subject overlap (involving the same entity, e.g., "Ryan"); (2) object overlap (involving the same item, e.g., "cats"); (3) dual overlap (involving both the same entity and item).
    - **Design Motivation**: Existing datasets add only one irrelevant item; this framework supports arbitrary numbers of irrelevant axioms with controllable overlap.

3. **Efficiency Metric**:

    - **Function**: Maps LLM CoT steps to logic program theorems and computes the proportion of unnecessary theorems.
    - **Mechanism**: For correctly answered problems, counts how many theorems in the LLM-generated proof do not lie on the shortest proof path.
    - **Design Motivation**: The shortest proof serves as the theoretical lower bound for efficiency; the deviation directly quantifies the degree of inefficiency.

## Key Experimental Results

### Accuracy Drop (with Irrelevant Axioms)

| Model | No Irrelevant | 1 Irrelevant | 3 Irrelevant | 5 Irrelevant |
|-------|--------------|-------------|-------------|-------------|
| GPT-4o | ~95% | ~88% | ~78% | ~70% |
| o1-mini | ~97% | ~92% | ~85% | ~78% |
| Llama-3.1-70B | ~85% | ~75% | ~60% | ~50% |

### Reasoning Efficiency (Proportion of Unnecessary Theorems in Correct Answers)

| Condition | Proportion of Unnecessary Theorems |
|-----------|----------------------------------|
| 50% irrelevant axioms | >50% of LLM reasoning steps are also unnecessary |
| High semantic overlap | Higher unnecessary proportion (LLM attracted by similar information) |
| No irrelevant axioms | ~5–10% (minor redundant reasoning) |

### Key Findings
- Adding even a single in-domain irrelevant axiom causes a notable accuracy drop across all LLMs.
- Accuracy degradation cannot be attributed solely to increased input length—control groups of equal length but without distractors perform significantly better.
- LLMs frequently incorporate irrelevant information in their reasoning even when answering correctly, indicating that inefficiency is more pervasive than accuracy degradation alone.
- Higher semantic overlap leads LLMs to rely more heavily on irrelevant information, suggesting that LLM search processes depend on a form of **semantic similarity heuristic**.

## Highlights & Insights
- **Efficiency vs. Correctness**: This work is the first to disentangle *reasoning efficiency* from *reasoning correctness* as an independent evaluation dimension.
- **Bridging Role of Logic Programming**: Verbalized logic programs elegantly connect natural language reasoning and formal derivation, enabling automated evaluation.
- **"Semantic Attraction" Phenomenon**: LLMs tend to reason with semantically similar information even when it is irrelevant—revealing an inherent limitation of attention-based information retrieval mechanisms.

## Limitations & Future Work
- **Restricted to Math Word Problems**: GSM-style problems have relatively simple logical structure; more complex reasoning (e.g., multi-hop, counterfactual) is not covered.
- **Accuracy of Natural Language → Logic Mapping**: The mapping process may introduce noise.
- **Latest Reasoning Models Not Evaluated** (e.g., o3).

## Related Work & Insights
- **vs. Shi et al. (2023) GSM-IC**: Adds only one irrelevant item; this paper systematically injects multiple items with controllable semantic overlap.
- **vs. Reasoning Efficiency Optimization (Han et al., 2025)**: Those works use token count as the efficiency metric; this paper uses formally defined unnecessary reasoning steps.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First framework to evaluate LLM reasoning efficiency from a logic programming perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models + systematic irrelevant-information variants + dual evaluation of efficiency and accuracy.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous formal definitions and carefully designed experiments.
- Value: ⭐⭐⭐⭐⭐ Significant academic value for understanding and improving LLM reasoning quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networ.md)
- [\[NeurIPS 2025\] EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)
- [\[ICML 2026\] Rethinking LLM Ensembling from the Perspective of Mixture Models](../../ICML2026/llm_nlp/rethinking_llm_ensembling_from_the_perspective_of_mixture_models.md)
- [\[NeurIPS 2025\] Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers](dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](../../AAAI2026/llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)

</div>

<!-- RELATED:END -->
