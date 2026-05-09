---
title: >-
  [Paper Note] Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents
description: >-
  [ICLR 2026 Workshop][LLM Efficiency][memory-augmented agents] This paper formalizes multi-store retrieval in memory-augmented agents as a cost-sensitive store routing problem, demonstrates that selective retrieval can reduce context tokens by 62% while improving QA accuracy (86% vs. 81%) over exhaustive retrieval, and proposes a semantics-based heuristic routing baseline.
tags:
  - ICLR 2026 Workshop
  - LLM Efficiency
  - memory-augmented agents
  - store routing
  - cost-sensitive retrieval
  - RAG
  - memory architecture
date: 2026-05-08
content_hash: a3c6c71821bd9591
---

# Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents

**Conference**: ICLR 2026 Workshop
**arXiv**: [2603.15658](https://arxiv.org/abs/2603.15658)
**Code**: None
**Area**: LLM Efficiency
**Keywords**: memory-augmented agents, store routing, cost-sensitive retrieval, RAG, memory architecture

## TL;DR
This paper formalizes multi-store retrieval in memory-augmented agents as a cost-sensitive store routing problem, demonstrates that selective retrieval can reduce context tokens by 62% while improving QA accuracy (86% vs. 81%) over exhaustive retrieval, and proposes a semantics-based heuristic routing baseline.

## Background & Motivation
**Background**: Memory-augmented agents (e.g., MemGPT) typically maintain multiple specialized stores — short-term memory (STM, current conversation), a summary store (Summary, compressed user facts), long-term memory (LTM, historical conversation summaries), and episodic memory (Episodic, raw transcripts). However, most systems retrieve from all stores for every query.

**Limitations of Prior Work**: Exhaustive retrieval incurs two costs: ① computational waste (querying stores that cannot contain the answer); and ② accuracy degradation (irrelevant/noisy context reduces the signal-to-noise ratio, forcing the model to locate the answer within large amounts of irrelevant text, especially in long-context settings).

**Key Challenge**: More context ≠ better performance. In long-context settings, interference from irrelevant stores actively misleads the model — for instance, outdated information in LTM may conflict with up-to-date information in Summary, and the model may erroneously select the stale version.

**Goal**: To determine *which stores to search* before retrieval, decouple store selection from intra-store ranking, and make the accuracy–cost tradeoff explicit.

**Key Insight**: Inspired by federated search in information retrieval and the episodic-vs-semantic memory taxonomy in cognitive science, the paper routes queries to stores according to their distinct semantic roles.

**Core Idea**: The routing decision is a first-class component of memory-augmented agent design, not an afterthought. It is formalized as $\pi^*(q) = \arg\max_{G \subseteq \mathcal{S}} [\mathbb{E}[\text{Acc}(q,G)] - \lambda \sum_{s \in G} c_s]$.

## Method

### Overall Architecture
Four memory stores form the store set $\mathcal{S} = \{\text{STM}, \text{Sum}, \text{LTM}, \text{Epi}\}$. Given a query $q$, a routing policy $\pi$ selects a subset $\hat{G} = \pi(q) \subseteq \mathcal{S}$; the system retrieves content exclusively from the selected stores, concatenates it, and feeds it to an LLM for answer generation. The framework is evaluated in two stages: ① synthetic routing evaluation (verifying store selection quality) → ② LLM QA evaluation (verifying downstream task performance).

### Key Designs

1. **Routing Evaluation Metric Suite**:

    - Function: Quantify the quality of store selection.
    - Coverage = $\frac{1}{N}\sum_i \mathbf{1}[G_i \subseteq \hat{G}_i]$: Whether all necessary stores are included (missed store = unanswerable).
    - Exact Match = $\frac{1}{N}\sum_i \mathbf{1}[G_i = \hat{G}_i]$: Whether exactly the necessary stores are selected.
    - Waste = $\frac{1}{N}\sum_i |\hat{G}_i \setminus G_i|$: Number of unnecessary stores retrieved in excess.
    - Design Motivation: Separating coverage (no misses) from precision (no excess) makes the accuracy–cost tradeoff measurable. Coverage is a hard constraint; Waste is a soft cost.

2. **Hybrid Heuristic Router**:

    - Function: Select target stores based on semantic signals in the query.
    - Core rules: Enumeration signal ("list all") → {LTM, Epi}; temporal signal ("before", "changed") → {LTM, Epi}; multi-hop signal ("compare", "relate") → {Sum, LTM}; current-session signal ("just said", "today") → {STM}; fact-lookup signal ("what is my") → {Sum}.
    - Falls back to {Sum, LTM} when no signal matches (highest coverage of 89% among the six two-store combinations).
    - Query–store embedding similarity is used as a tiebreaker, contributing +4% coverage.
    - Design principle: Prioritize coverage (a missed store renders the query unanswerable); narrow the routing scope only when a reliable signal is present.

3. **Cost-Sensitive Decision-Theoretic Framework**:

    - Function: Provide a mathematical foundation for store routing.
    - Core formula: $\pi^*(q) = \arg\max_{G \subseteq \mathcal{S}} [\mathbb{E}[\text{Acc}(q,G)] - \lambda \sum_{s \in G} c_s]$
    - Setting $\lambda = 0$ degenerates to exhaustive retrieval (Uniform); Oracle routing approximates the upper bound of $\pi^*$.
    - Explanatory power: Retrieving irrelevant stores increases effective retrieval cost while potentially reducing the probability of correct extraction (due to contextual noise), so selective retrieval yields gains on both dimensions.
    - Distinction from retriever routing: Store routing is a memory-architecture-level decision; the semantic roles of stores differ substantially (STM vs. LTM vs. Summary), making it coarser-grained than passage-level routing.

### Routing Strategy Spectrum
Ordered from simplest to strongest: Uniform (exhaustive, $\lambda=0$) → Fixed Subset (e.g., STM+Sum+LTM) → Hybrid Heuristic (rules + fallback) → Oracle (theoretical upper bound). The paper evaluates a full spectrum of 12 strategies.

## Key Experimental Results

### Synthetic Routing Evaluation (1,000 queries, 7 query types)

| Strategy | Coverage | Exact Match | Waste |
|----------|----------|-------------|-------|
| Uniform | 100% | 8% | 2.9 |
| Rule-based (linguistics only) | 57% | 35% | 0.5 |
| **Hybrid (Ours)** | **94%** | **58%** | **1.2** |
| Oracle | 100% | 100% | 0.0 |

### LLM QA Evaluation (150 questions)

| Model | Strategy | Overall Acc. | Short | Long | Tokens |
|-------|----------|-------------|-------|------|--------|
| GPT-4o-mini | Oracle | **86.7%** | 94% | **72%** | **299** |
| GPT-4o-mini | STM+Sum+LTM | 84.7% | 92% | 70% | 591 |
| GPT-4o-mini | Uniform | 81.3% | 92% | 60% | 787 |
| GPT-4o-mini | Hybrid | 70.7% | 80% | 52% | 379 |
| GPT-3.5 | Oracle | 85.3% | 93% | 70% | 299 |
| GPT-3.5 | Uniform | 83.3% | 91% | 68% | 787 |

### Ablation Study

| Feature Group | Coverage | Δ |
|---------------|----------|---|
| Linguistic features (pronouns, tense) | 57% | baseline |
| + Semantic signals (enumeration, temporal, multi-hop) | 90% | +33% |
| + Embedding similarity | 94% | +4% |

### Key Findings
- **Oracle achieves higher accuracy with 62% fewer tokens** (86.7% vs. 81.3%), providing strong evidence that "more context ≠ better."
- **Long-context settings amplify the penalty of over-retrieval**: In the Long setting, Oracle achieves 72% vs. Uniform's 60%, with the gap widening from 2% in the Short setting to 12%.
- **The fixed strategy STM+Sum+LTM closely approaches Oracle** (84.7% vs. 86.7%), making it a practical deployable solution.
- **Coverage–Accuracy Gap**: Hybrid achieves 94% coverage but only 70% QA accuracy. Of the errors, 12% stem from routing failures (missed stores) and 18% from extraction failures (correct store selected but model fails to extract the answer).

## Why Does Exhaustive Retrieval Underperform?
Two mechanisms: ① **Needle-in-a-haystack effect**: Locating sparse relevant information within 787 tokens yields a low signal-to-noise ratio. ② **Information conflicts**: Different stores contain outdated or conflicting information. A representative case: "Who is my current manager?" — the Summary store contains the correct answer "Jennifer Williams," but LTM records "Before the reorg...reported to Michael Torres." Under exhaustive retrieval, the model sometimes erroneously extracts the more detailed but stale information.

## Highlights & Insights
- **Rigorous empirical evidence for "more is not better"** — a cautionary finding for all RAG and memory systems: blindly increasing context length may be counterproductive.
- **Conceptual distinction between store routing and retriever routing**: Store routing is an architecture-level decision (stores with distinct semantic roles) with greater impact than passage-level retrieval routing, yet it has been largely overlooked.
- **Two-stage evaluation design**: Synthetic labels first validate routing quality; real LLM evaluation then validates downstream performance, effectively decoupling routing decisions from model capability.
- **Decomposed analysis of the Coverage–Accuracy Gap**: Distinguishing routing errors (12%) from extraction errors (18%) clearly identifies directions for improvement.

## Limitations & Future Work
- Labels are derived from query classification rules rather than human annotation, which may not fully reflect real-world store access requirements.
- The heuristic router trails Oracle by 16 points (70% vs. 86%), motivating end-to-end learned routing strategies (e.g., RL-based optimization of the $\lambda$-tradeoff).
- Only GPT-3.5 and GPT-4o-mini are evaluated; models with different long-context processing strategies may respond differently.
- Full store content concatenation is used rather than top-$k$ retrieval, which diverges from production system settings — the interaction between routing and intra-store retrieval remains unexplored.
- The test set of only 150 questions provides limited statistical power.

## Related Work & Insights
- **vs. Self-RAG / FLARE**: Those works decide *whether* to retrieve (when); this paper decides *from which store* to retrieve (where) — the two dimensions are complementary.
- **vs. MemGPT**: MemGPT focuses on memory organization and management operations (read/write/consolidation); this paper focuses on routing decisions for memory access — the two are combinable.
- **vs. ExpertRAG / RAP-RAG**: ExpertRAG uses MoE-style routing for context selection; RAP-RAG plans multi-hop retrieval sequences. Store routing in this paper operates at a coarser granularity (store-level vs. passage-level).
- **vs. federated retrieval literature**: Resource selection algorithms in information retrieval, which estimate the relevance distribution of each collection, are directly transferable to agent memory routing.
- Practical implication for multi-store RAG system design: route-then-retrieve outperforms retrieve-all-then-filter-by-LLM.

## Rating
- Novelty: ⭐⭐⭐ — The problem formulation is clear (store routing as a first-class citizen), but the technical solution (rules + fallback) is relatively simple.
- Experimental Thoroughness: ⭐⭐⭐ — The two-stage evaluation design is sound, but the sample size is small (150 questions).
- Writing Quality: ⭐⭐⭐⭐ — The argumentation is logically coherent, the failure case analysis is thorough, and the decision-theoretic framework is elegant.
- Value: ⭐⭐⭐⭐ — The "routing as a first-class citizen" perspective offers practical guidance for memory-augmented systems, and the 16-point Oracle gap motivates future research.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Universe Routing: Why Self-Evolving Agents Need Epistemic Control](universe_routing_why_self-evolving_agents_need_epistemic_control.md)
- [\[ICLR 2026\] IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling](iterresearch_rethinking_long-horizon_agents_with_interaction_scaling.md)
- [\[ICLR 2026\] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection](tokenseek_memory_efficient_fine_tuning_via_instance-aware_token_selection.md)
- [\[NeurIPS 2025\] Tensor Product Attention Is All You Need](../../NeurIPS2025/llm_efficiency/tensor_product_attention_is_all_you_need.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)

<!-- RELATED:END -->
