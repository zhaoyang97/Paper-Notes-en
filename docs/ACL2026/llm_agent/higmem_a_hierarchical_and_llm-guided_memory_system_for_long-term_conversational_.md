---
title: >-
  [Paper Note] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents
description: >-
  [ACL 2026][LLM Agent][Long-term conversational memory] This paper proposes HiGMem, a two-layer event-turn memory system. By having the LLM first browse event summaries and then predict which fine-grained dialogue turns a…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Long-term conversational memory"
  - "Hierarchical memory system"
  - "LLM-guided retrieval"
  - "Evidence streamlining"
  - "Event-turn architecture"
date: 2026-05-08
content_hash: cc875f2f29f20e4a
---

# HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents

**Conference**: ACL 2026  
**arXiv**: [2604.18349](https://arxiv.org/abs/2604.18349)  
**Code**: [https://github.com/ZeroLoss-Lab/HiGMem](https://github.com/ZeroLoss-Lab/HiGMem)  
**Area**: LLM Evaluation  
**Keywords**: Long-term conversational memory, Hierarchical memory system, LLM-guided retrieval, Evidence streamlining, Event-turn architecture

## TL;DR
This paper proposes HiGMem, a two-layer event-turn memory system. By having the LLM first browse event summaries and then predict which fine-grained dialogue turns are worth reading, it achieves the best F1 scores in four out of five categories on the LoCoMo10 benchmark with one order of magnitude fewer retrievals.

## Background & Motivation

**Background**: LLM agents require memory systems to recover relevant evidence from historical interactions in long-term dialogues. Existing systems like MemGPT and A-Mem extend long-range interaction capabilities through external memory storage and vector similarity retrieval.

**Limitations of Prior Work**: Existing memory systems (including hierarchical ones) still primarily rely on vector similarity for retrieval. This approach easily produces "inflated evidence sets"—once the most relevant memories are recalled, continuing to add surface-level similar fragments yields diminishing returns for recall while continuously eroding retrieval precision, inflating the downstream context for answer generation, and making evidence sets difficult to inspect and manage.

**Key Challenge**: Vector similarity itself cannot judge whether a memory is "truly worth reading." it lacks the ability to reason across different levels of abstraction and cannot actively evaluate which fine-grained details actually contribute to answering a query.

**Goal**: To develop a retrieval strategy that simultaneously maintains high recall, high precision, and controllable token overhead, delivering compact and high-precision evidence sets.

**Key Insight**: Mimicking the way humans process information—scanning high-level summaries first to determine thematic relevance before delving into pertinent details. The LLM acts as an "information gatekeeper," using event summaries as semantic anchors to reason about which underlying dialogue turns deserve detailed reading.

**Core Idea**: Organize memory using a two-layer event-turn architecture. The LLM first retrieves event summaries as semantic anchors and then predicts which associated dialogue turns are worth reading. This replaces brute-force vector retrieval with reasoning to obtain condensed and reliable evidence sets.

## Method

### Overall Architecture
HiGMem consists of a two-layer memory architecture and an LLM-guided retrieval strategy. The bottom layer is the Turn Layer, storing each dialogue turn and LLM-generated metadata (keywords, labels, timestamps, context); the top layer is the Event Layer, grouping related dialogue turns into coherent narrative units containing summaries and structured fact tables. Bidirectional links between the layers ensure traceability. During retrieval, the system retrieves candidates from both layers simultaneously and uses event summaries to guide the LLM in filtering truly useful dialogue turns.

### Key Designs

1. **Turn Analysis & Event Affiliation**:

    - **Function**: To incorporate new dialogue turns into the memory system in real-time by automatically analyzing metadata and assigning them to relevant events.
    - **Mechanism**: Metadata such as keywords and labels are extracted from the new dialogue turn $D_t$ within the context of a sliding window $\mathcal{W}_t$ to form a Turn node. Its embedding cosine similarity with all event nodes is calculated, and the top-$k_{\text{event}}$ candidates are selected for the LLM to decide affiliation. Depending on the size of the event node (whether the number of linked turns exceeds the threshold $\tau=10$), the system chooses to refresh the event summary entirely or perform an append-only update.
    - **Design Motivation**: Real-time updates ensure the memory system always reflects the latest dialogue state; the adaptive update strategy maintains summary quality for small events while controlling computational overhead for larger ones.

2. **LLM-Guided Hierarchical Retrieval**:

    - **Function**: To accurately locate evidence dialogue turns relevant to a query from a large volume of memory.
    - **Mechanism**: Keywords $q_{\text{kw}}$ are generated for query $Q$. Simultaneously, $k_{\text{turn}}=10$ semantically related Turn nodes $\mathcal{T}_{\text{semantic}}$ are retrieved from the turn layer, and $k_{\text{event}}=10$ relevant event nodes $\mathcal{E}_{\text{semantic}}$ are retrieved from the event layer. Using the event nodes as semantic anchors, the LLM evaluates the Turn nodes under each event to reason and predict which are worth reading, resulting in $\mathcal{T}_{\text{pred}}$. Finally, the LLM filters the union $\mathcal{T}_{\text{semantic}} \cup \mathcal{T}_{\text{pred}}$ to obtain the final evidence set $\mathcal{T}_{\text{final}}$.
    - **Design Motivation**: Pure vector retrieval returns "surface-level similar" results and cannot determine if a turn "truly answers the question." By providing a low-cost semantic overview through event summaries, the LLM can make precise selections without reading every individual memory.

3. **Bidirectional Linking and Traceability**:

    - **Function**: To ensure data provenance between the event layer and the dialogue turn layer.
    - **Mechanism**: Each event node maintains a link set $\mathcal{L}_E$ that records indices of all affiliated Turn nodes. The link set is updated when a new Turn node is assigned: $\mathcal{L}_E = \mathcal{L}_E \cup \{l\}$.
    - **Design Motivation**: Bidirectional links allow for both drilling down from events to turns and tracing back from turns to events, forming the structural basis for hierarchical retrieval.

### Loss & Training
HiGMem does not involve end-to-end training. All LLM calls use GPT-4o-mini, and embeddings use all-MiniLM-L6-v2. The system implements LLM reasoning at each stage through prompt engineering.

## Key Experimental Results

### Main Results
F1 scores for five categories of questions on the LoCoMo10 benchmark (averaging 587 turns per dialogue):

| Method | Multi-Hop | Temporal | Open-Domain | Single-Hop | Adversarial | Average Rank |
|------|-----------|----------|-------------|------------|-------------|---------|
| Base LLM | 0.25 | 0.39 | 0.12 | 0.44 | 0.30 | 2.2 |
| A-Mem | 0.27 | **0.39** | 0.10 | 0.42 | 0.54 | 2.2 |
| **Ours (HiGMem)** | **0.31** | 0.34 | **0.15** | **0.49** | **0.78** | **1.2** |

### Ablation Study

| Configuration | F1 | Recall@K |
|------|-----|----------|
| w/o Hierarchy (Removed Event Layer) | 0.39 | 0.55 |
| HiGMem (Full) | **0.49** | **0.72** |

Retrieval efficiency comparison:

| Method | Avg. Retrieved Turns | Precision@K | Recall@K |
|------|------------|-------------|----------|
| A-Mem | 99.84 | 0.0101 | 0.7502 |
| HiGMem | **8.09** | **0.1909** | 0.7241 |

### Key Findings
- HiGMem's retrieval volume is only 1/12th that of A-Mem, yet recall remains nearly equivalent (0.72 vs 0.75), and precision improves by nearly 19 times.
- F1 scores for adversarial questions increased from 0.54 to 0.78, indicating that a streamlined evidence set effectively reduces interference from misleading information.
- Performance on temporal questions is slightly inferior to A-Mem, suggesting that current event-level abstraction might weaken some fine-grained temporal cues.
- In a hybrid deployment scenario (GPT-4o-mini for memory + GPT-5 for answers), total costs dropped from $17.43 to $6.43, a reduction of approximately 2.7 times.

## Highlights & Insights
- The hierarchical retrieval paradigm of "scanning summaries before deciding to view details" aligns closely with human information processing and is validated by experiments. This approach can be transferred to RAG, document QA, and any scenario requiring evidence filtering from large candidate sets.
- Improving retrieval precision by 19 times with almost no drop in recall proves that "less is more"—a large number of low-relevance memories is not only unhelpful but harmful.
- The significant improvement in adversarial questions shows that concise evidence sets effectively help LLMs resist distracting information.

## Limitations & Future Work
- The memory construction phase requires additional LLM calls, increasing time and token overhead (15.59s per turn vs 6.38s for A-Mem).
- The system's effectiveness depends on the LLM's ability to infer relevance from event summaries and candidate turns.
- Performance on temporal questions is insufficient, indicating that event-level abstraction may lose fine-grained temporal information.
- Currently only validated on the LoCoMo10 benchmark; robustness needs evaluation in more scenarios such as multi-party or noisy dialogues.

## Related Work & Insights
- **vs A-Mem**: A-Mem uses vector retrieval returning ~100 turns with extremely low precision (1%); HiGMem reduces the retrieval to 8 turns via LLM guidance, increasing precision to 19% with better F1.
- **vs RAPTOR**: RAPTOR uses recursive summarization for multi-granularity retrieval but lacks an active LLM reasoning/filtering step; HiGMem's event layer is similar to RAPTOR's summary layer but adds the critical step of LLM prediction.
- **vs MemGPT**: MemGPT manages memory like an operating system but still relies on vector retrieval; HiGMem uses a hierarchical structure to explicitly guide the LLM for precise filtering.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hierarchical structure and LLM guidance is a natural yet effective innovation; the event-turn design is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across five question types, efficiency analysis, cost analysis, and ablation studies are comprehensive, though limited to a single benchmark.
- Writing Quality: ⭐⭐⭐⭐ Problem definitions are clear and the method description is intuitive, though the overall length is short.
- Value: ⭐⭐⭐⭐ The "streamlined evidence set" concept offers universal inspiration for RAG and dialogue systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](timem_temporal-hierarchical_memory_consolidation_for_long-horizon_conversational.md)
- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)
- [\[ACL 2026\] Why LLM Web Agents Fail: A Hierarchical Planning Perspective](why_do_llm-based_web_agents_fail_a_hierarchical_planning_perspective.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)

</div>

<!-- RELATED:END -->
