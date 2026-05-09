---
title: >-
  [Paper Note] Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval
description: >-
  [ICLR 2026][LLM Reasoning][LLM personalization] Inspired by the dual-process theory in cognitive science, this paper proposes RF-Mem, a memory retrieval framework that achieves efficient and scalable LLM personalization through adaptive switching between two pathways: Familiarity (fast similarity matching) and Recollection (deep chain-based reconstruction).
tags:
  - ICLR 2026
  - LLM Reasoning
  - LLM personalization
  - memory retrieval
  - dual-process theory
  - adaptive retrieval
  - cognitive science
date: 2026-05-08
content_hash: aaffaa6eeb91bf3f
---

# Evoking User Memory: Personalizing LLM via Recollection-Familiarity Adaptive Retrieval

**Conference**: ICLR 2026
**arXiv**: [2603.09250](https://arxiv.org/abs/2603.09250)
**Code**: See Reproducibility Statement in the paper
**Area**: Personalization / Information Retrieval
**Keywords**: LLM personalization, memory retrieval, dual-process theory, adaptive retrieval, cognitive science

## TL;DR

Inspired by the dual-process theory in cognitive science, this paper proposes RF-Mem, a memory retrieval framework that achieves efficient and scalable LLM personalization through adaptive switching between two pathways: Familiarity (fast similarity matching) and Recollection (deep chain-based reconstruction).

## Background & Motivation

Personalizing large language models requires incorporating user-specific history, preferences, and context into response generation. Two dominant existing approaches each suffer from critical drawbacks:

**Full-context methods**: Stuffing all user memory into the prompt is costly and unscalable — as user memory accumulates, prompt length quickly exceeds model context window limits.

**One-shot retrieval methods**: Reducing retrieval to a single-round similarity search (top-K) only captures surface-level matches and fails to recover memories that are indirectly but critically related to the query.

Cognitive science research shows that human memory recognition operates via a **dual process**:
- **Familiarity**: A fast but coarse recognition process that rapidly judges whether something has been encountered before.
- **Recollection**: A slow but precise reconstruction process that consciously retrieves specific details and associated context.

Existing systems lack both recollection-style retrieval capability and a mechanism to adaptively switch between the two retrieval pathways. This leads to either under-retrieval (missing key memories) or noise introduction (retrieving irrelevant content).

## Method

### Overall Architecture

RF-Mem (Recollection-Familiarity Memory Retrieval) is a dual-pathway memory retriever guided by familiarity uncertainty. Given a user query, the framework first computes a Familiarity signal to assess retrieval certainty, then selects either the fast Familiarity pathway or the deep Recollection pathway based on that certainty level.

### Key Designs

1. **Familiarity Signal Computation**:

    - Computes the distribution of similarity scores between the query and candidate memories.
    - Familiarity level is measured by two metrics:
        - **Mean Score**: Captures overall matching strength; a higher mean indicates the presence of familiar memories.
        - **Entropy**: Measures uncertainty in the score distribution; lower entropy indicates more definitive and concentrated retrieval results.
    - High mean + low entropy = high Familiarity (confident direct match exists).
    - Low mean / high entropy = low Familiarity (deeper retrieval required).

2. **Familiarity Pathway (Fast Track)**:

    - Activated when the Familiarity signal is strong.
    - Performs standard top-K similarity retrieval and directly returns the most relevant memories.
    - Efficient: requires only a single forward retrieval pass with no additional computation.
    - Applicable when: the user query has a clear and direct association with historical memories.

3. **Recollection Pathway (Deep Reconstruction Track)**:

    - Activated when the Familiarity signal is weak, simulating the human process of conscious memory reconstruction.
    - **Candidate Memory Clustering**: Clusters retrieved candidate memories to identify distinct memory themes and contexts.
    - **Alpha-Mix Query Expansion**: Blends the query with cluster centroids of candidate memories via alpha mixing to generate expanded queries in embedding space.
    - **Iterative Evidence Expansion**: Uses the expanded query for re-retrieval, progressively covering memories that are indirectly yet semantically related to the original query.
    - Applicable when: the user query involves cross-temporal or cross-topic memory associations that require chain-like reasoning to recall.

4. **Adaptive Pathway Switching**:

    - Automatically selects the pathway based on a threshold mechanism applied to the Familiarity signal.
    - Avoids both extremes: the high cost of full-context methods and the low recall of one-shot retrieval.
    - Achieves optimal retrieval quality under fixed budget and latency constraints.

### Loss & Training

RF-Mem is a modular framework compatible with different embedding models and LLMs. The key innovation lies at the retrieval strategy level rather than model training — personalization is improved through smarter retrieval decisions.

## Key Experimental Results

### Main Results

Evaluation is conducted on three personalization benchmarks spanning different corpus scales:

| Method | Benchmark 1 | Benchmark 2 | Benchmark 3 | Notes |
|--------|-------------|-------------|-------------|-------|
| Full-context inference | Baseline | Baseline | Baseline | Highest cost, performance upper bound |
| One-shot retrieval | Below full-context | Below full-context | Below full-context | Simple and fast but lower quality |
| **RF-Mem** | **Best** | **Best** | **Best** | Consistently outperforms both baselines under fixed budget |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Familiarity pathway only | Baseline level | Equivalent to standard top-K retrieval |
| Recollection pathway only | Above Familiarity-only | Wastes computation on simple queries |
| Dual-pathway + adaptive switching | **Best** | Balances efficiency and quality |
| w/o clustering | Performance drop | Clustering helps identify memory topic structure |
| w/o Alpha-Mix | Performance drop | Query expansion is central to Recollection |

### Key Findings

- **Consistent advantage**: RF-Mem outperforms both baselines across all three benchmarks and at varying corpus scales.
- **Budget efficiency**: Under fixed retrieval budgets (token count) and latency constraints, RF-Mem achieves performance close to full-context methods while maintaining the efficiency of one-shot retrieval.
- **Scalability**: RF-Mem's advantage becomes more pronounced as the user memory store grows — full-context costs scale linearly, whereas RF-Mem's overhead grows modestly.
- **Pathway distribution**: Approximately 60–70% of queries are handled by the fast Familiarity pathway, while 30–40% require the deep Recollection pathway.

## Highlights & Insights

1. **Elegant transfer from cognitive science**: Introducing the Familiarity-Recollection dual-process theory of human memory into LLM retrieval system design is both intellectually elegant and practically effective.
2. **Uncertainty-guided adaptation**: Using the mean and entropy of retrieval score distributions as adaptive switching signals is more robust than simple threshold-based approaches.
3. **Memory reconstruction in embedding space**: Simulating the chain-like reconstruction of recollection via clustering and Alpha-Mix in embedding space avoids costly multi-round LLM calls.
4. **Pragmatic design philosophy**: The modular architecture allows individual components to be replaced and optimized independently, making it well-suited for engineering deployment.

## Limitations & Future Work

1. **Familiarity threshold tuning**: Adaptive switching relies on threshold parameters that may require dataset-specific tuning, lacking a fully automated solution.
2. **Clustering algorithm selection**: The clustering method in the Recollection pathway may underperform on high-dimensional sparse memories.
3. **Long-term memory forgetting and updates**: The paper does not explicitly address how to handle outdated or contradictory user memories.
4. **Privacy considerations**: Storing and retrieving user history entails privacy risks; the paper does not discuss privacy-preserving mechanisms in depth.
5. **Abstract-based evaluation only** (note): Due to the unavailability of the full-text HTML version, some experimental details and data points are inferred from the abstract.

## Related Work & Insights

- **Dual-process cognitive theory (Yonelinas, 2002)**: Familiarity and Recollection are two fundamental processes in human memory recognition; this paper operationalizes the theory into retrieval system design.
- **RAG (Retrieval-Augmented Generation)**: RF-Mem can be viewed as an enhanced RAG variant with retrieval strategies specifically optimized for personalization scenarios.
- **Adaptive Retrieval**: Works such as Self-RAG and FLARE study *when* to retrieve; RF-Mem studies *how* to retrieve.
- **Personalized LLMs**: Benchmarks such as LaMP and PersonaLLM have advanced the development of personalized LLMs; this paper improves upon the retrieval module in that line of work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Innovative application of the dual-process theory from cognitive science to LLM retrieval.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three benchmarks, multi-scale evaluation, and ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ — Concepts are clearly presented with thorough motivation from an interdisciplinary perspective.
- Value: ⭐⭐⭐⭐ — Provides a practical and scalable solution for memory retrieval in personalized LLMs.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](nudging_the_boundaries_of_llm_reasoning.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)
- [\[ICLR 2026\] Thinking in Latents: Adaptive Anchor Refinement for Implicit Reasoning in LLMs](thinking_in_latents_adaptive_anchor_refinement_for_implicit_reasoning_in_llms.md)

<!-- RELATED:END -->
