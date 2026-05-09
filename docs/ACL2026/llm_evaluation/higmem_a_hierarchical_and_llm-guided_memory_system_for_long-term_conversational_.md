---
title: >-
  [Paper Note] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents
description: >-
  [ACL 2026][LLM Evaluation][long-term conversational memory] This paper proposes HiGMem, a two-level event-turn memory system that enables an LLM to first browse event summaries and then predict which fine-grained conversation turns are worth reading, achieving the best F1 on four out of five question categories on the LoCoMo10 benchmark while retrieving an order of magnitude fewer turns.
tags:
  - ACL 2026
  - LLM Evaluation
  - long-term conversational memory
  - hierarchical memory system
  - LLM-guided retrieval
  - evidence distillation
  - event-turn architecture
date: 2026-05-08
content_hash: 44fd1429f3406341
---

# HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents

**Conference**: ACL 2026
**arXiv**: [2604.18349](https://arxiv.org/abs/2604.18349)
**Code**: [https://github.com/ZeroLoss-Lab/HiGMem](https://github.com/ZeroLoss-Lab/HiGMem)
**Area**: LLM Evaluation
**Keywords**: long-term conversational memory, hierarchical memory system, LLM-guided retrieval, evidence distillation, event-turn architecture

## TL;DR
This paper proposes HiGMem, a two-level event-turn memory system that enables an LLM to first browse event summaries and then predict which fine-grained conversation turns are worth reading, achieving the best F1 on four out of five question categories on the LoCoMo10 benchmark while retrieving an order of magnitude fewer turns.

## Background & Motivation

**Background**: LLM agents in long-term conversations require memory systems to retrieve relevant evidence from historical interactions. Existing systems such as MemGPT and A-Mem extend long-range interaction capabilities through external memory stores and vector-similarity-based retrieval.

**Limitations of Prior Work**: Existing memory systems, including hierarchical ones, still rely primarily on vector similarity for retrieval. This approach tends to produce "bloated evidence sets"—once the most relevant memories are recalled, adding further surface-similar fragments yields diminishing recall gains while continuously eroding retrieval precision, inflating the context fed to downstream answer generation, and making evidence sets difficult to inspect and manage.

**Key Challenge**: Vector similarity alone cannot determine whether a memory is "truly worth reading"; it lacks the ability to reason across different levels of abstraction and cannot proactively assess which fine-grained details actually contribute to answering a query.

**Goal**: To develop a retrieval strategy that simultaneously maintains high recall, high precision, and controllable token overhead, delivering compact and high-precision evidence sets.

**Key Insight**: Mimicking the human approach to information processing—first skimming high-level summaries to identify relevant topics, then diving into the details of relevant content. The LLM acts as an "information gatekeeper," using event summaries as semantic anchors to reason about which underlying conversation turns merit closer reading.

**Core Idea**: Memory is organized in a two-level event-turn architecture. The LLM first retrieves event summaries as semantic anchors, then predicts which associated turns are worth reading, thereby replacing brute-force vector retrieval with reasoning to obtain a compact and reliable evidence set.

## Method

### Overall Architecture
HiGMem comprises a two-level memory architecture and an LLM-guided retrieval strategy. The lower level is the Turn Layer, which stores each conversation turn along with LLM-generated metadata (keywords, tags, timestamps, and context). The upper level is the Event Layer, which groups related turns into coherent narrative units containing summaries and structured fact tables. Bidirectional links between the two levels ensure full traceability. During retrieval, the system retrieves candidates from both levels and uses event summaries to guide the LLM in selecting the genuinely useful turns.

### Key Designs

1. **Turn Analysis & Event Affiliation**:

    - **Function**: Incorporates new conversation turns into the memory system in real time, automatically extracting metadata and assigning turns to relevant events.
    - **Mechanism**: A new turn $D_t$ is processed by the LLM within a sliding window $\mathcal{W}_t$ to extract metadata such as keywords and tags, forming a Turn node. Its embedding is then compared via cosine similarity against all Event nodes; the top-$k_{\text{event}}$ candidates are selected and the LLM decides on affiliation. Depending on the size of the Event node (whether the number of linked turns exceeds threshold $\tau=10$), either a full refresh or an append-only update of the event summary is performed.
    - **Design Motivation**: Real-time updates ensure the memory system always reflects the latest conversational state; the adaptive update strategy maintains summary quality for small events while controlling computational overhead for large ones.

2. **LLM-Guided Hierarchical Retrieval**:

    - **Function**: Precisely locates evidence turns relevant to the query from a large memory store.
    - **Mechanism**: Given query $Q$, keywords $q_{\text{kw}}$ are generated; $k_{\text{turn}}=10$ semantically relevant Turn nodes $\mathcal{T}_{\text{semantic}}$ are retrieved from the Turn Layer, and $k_{\text{event}}=10$ relevant Event nodes $\mathcal{E}_{\text{semantic}}$ are retrieved from the Event Layer. Using the Event nodes as semantic anchors, the LLM evaluates the Turn nodes linked to each event and predicts which are worth reading, yielding $\mathcal{T}_{\text{pred}}$. The LLM then filters the union $\mathcal{T}_{\text{semantic}} \cup \mathcal{T}_{\text{pred}}$ to obtain the final evidence set $\mathcal{T}_{\text{final}}$.
    - **Design Motivation**: Pure vector retrieval returns "surface-similar" results and cannot determine whether a turn "truly answers the question." Event summaries provide a low-cost semantic overview, enabling the LLM to make precise selections without reading every memory individually.

3. **Bidirectional Links and Traceability**:

    - **Function**: Ensures data provenance between the Event Layer and the Turn Layer.
    - **Mechanism**: Each Event node maintains a link set $\mathcal{L}_E$ recording the indices of all affiliated Turn nodes. When a new Turn node is affiliated, the link set is updated as $\mathcal{L}_E = \mathcal{L}_E \cup \{l\}$.
    - **Design Motivation**: Bidirectional links enable both top-down traversal from events to turns and bottom-up traversal from turns to events, forming the structural foundation of hierarchical retrieval.

### Loss & Training
HiGMem does not involve end-to-end training. All LLM calls use GPT-4o-mini, and embeddings use all-MiniLM-L6-v2. Each module's reasoning is implemented via prompt engineering.

## Key Experimental Results

### Main Results
F1 scores across five question categories on the LoCoMo10 benchmark (average 587 turns/conversation):

| Method | Multi-Hop | Temporal | Open-Domain | Single-Hop | Adversarial | Avg. Rank |
|--------|-----------|----------|-------------|------------|-------------|-----------|
| Base LLM | 0.25 | 0.39 | 0.12 | 0.44 | 0.30 | 2.2 |
| A-Mem | 0.27 | **0.39** | 0.10 | 0.42 | 0.54 | 2.2 |
| **HiGMem** | **0.31** | 0.34 | **0.15** | **0.49** | **0.78** | **1.2** |

### Ablation Study

| Configuration | F1 | Recall@K |
|---------------|----|----------|
| w/o Hierarchy (Event Layer removed) | 0.39 | 0.55 |
| HiGMem (full) | **0.49** | **0.72** |

Retrieval efficiency comparison:

| Method | Avg. Retrieved Turns | Precision@K | Recall@K |
|--------|----------------------|-------------|----------|
| A-Mem | 99.84 | 0.0101 | 0.7502 |
| HiGMem | **8.09** | **0.1909** | 0.7241 |

### Key Findings
- HiGMem retrieves only 1/12 as many turns as A-Mem, yet achieves nearly equivalent recall (0.72 vs. 0.75) with a ~19× improvement in precision.
- F1 on adversarial questions improves from 0.54 to 0.78, demonstrating that a compact evidence set effectively reduces interference from misleading information.
- Slightly lower performance on temporal questions suggests that event-level abstraction may weaken certain fine-grained temporal cues.
- In a hybrid deployment scenario (GPT-4o-mini for memory + GPT-5 for answer generation), total cost decreases from \$17.43 to \$6.43, a reduction of approximately 2.7×.

## Highlights & Insights
- The hierarchical retrieval paradigm of "skim summaries first, then decide whether to read details" closely mirrors human information-processing intuition, and its effectiveness is validated experimentally. This approach is transferable to RAG, document QA, and any scenario requiring evidence selection from large candidate sets.
- A 19× improvement in retrieval precision with almost no drop in recall demonstrates that "less is more"—a large number of low-relevance memories is not merely unhelpful but actively harmful.
- The substantial gain on adversarial questions shows that a compact evidence set effectively helps LLMs resist distracting information.

## Limitations & Future Work
- The memory construction phase requires additional LLM calls, increasing time and token overhead (15.59s per turn vs. 6.38s for A-Mem).
- The system's effectiveness depends on the LLM's ability to infer relevance from event summaries and candidate turns.
- Underperformance on temporal questions suggests that event-level abstraction may discard fine-grained temporal information.
- Validation is currently limited to the single LoCoMo10 benchmark; robustness in multi-party conversations, noisy dialogues, and other settings remains to be evaluated.

## Related Work & Insights
- **vs. A-Mem**: A-Mem retrieves approximately 100 turns via vector search at extremely low precision (1%); HiGMem reduces retrieval to 8 turns via LLM guidance, raising precision to 19% while achieving superior F1.
- **vs. RAPTOR**: RAPTOR employs recursive summarization for multi-granularity retrieval but lacks an LLM reasoning step for active filtering; HiGMem's Event Layer is analogous to RAPTOR's summary layer, with the critical addition of LLM prediction of "whether a turn is worth reading."
- **vs. MemGPT**: MemGPT treats the LLM as an operating system for memory management but still relies on vector retrieval; HiGMem uses a hierarchical structure to explicitly guide the LLM toward precise selection.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hierarchy and LLM guidance is a natural yet effective innovation; the two-level event-turn design is clean and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation covers five question types, retrieval efficiency, cost analysis, and ablation studies—fairly comprehensive, though limited to a single benchmark.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and the method description is intuitive, though the overall length is relatively short.
- Value: ⭐⭐⭐⭐ The "compact evidence set" paradigm has broad implications for RAG and dialogue systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction](../../AAAI2026/llm_evaluation/bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[NeurIPS 2025\] OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling](../../NeurIPS2025/llm_evaluation/optitree_hierarchical_thoughts_generation_with_tree_search_for_llm_optimization_.md)
- [\[ICLR 2026\] MOSIV: Multi-Object System Identification from Videos](../../ICLR2026/llm_evaluation/mosiv_multi-object_system_identification_from_videos.md)

</div>

<!-- RELATED:END -->
