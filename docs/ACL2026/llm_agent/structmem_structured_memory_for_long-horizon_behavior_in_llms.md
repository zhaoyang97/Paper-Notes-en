---
title: >-
  [Paper Note] StructMem: Structured Memory for Long-Horizon Behavior in LLMs
description: >-
  [ACL 2026][LLM Agent][Long-term Memory] StructMem proposes a structure-enhanced hierarchical memory framework. By employing event-level dual-perspective extraction and cross-event semantic consolidation…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Long-term Memory"
  - "Event-level Binding"
  - "Cross-event Consolidation"
  - "Hierarchical Memory"
  - "Multi-hop Reasoning"
date: 2026-05-08
content_hash: 0562c769efdacb39
---

# StructMem: Structured Memory for Long-Horizon Behavior in LLMs

**Conference**: ACL 2026  
**arXiv**: [2604.21748](https://arxiv.org/abs/2604.21748)  
**Code**: [https://github.com/zjunlp/LightMem](https://github.com/zjunlp/LightMem)  
**Area**: LLM Agent / Dialogue Systems  
**Keywords**: Long-term Memory, Event-level Binding, Cross-event Consolidation, Hierarchical Memory, Multi-hop Reasoning

## TL;DR

StructMem proposes a structure-enhanced hierarchical memory framework. By employing event-level dual-perspective extraction and cross-event semantic consolidation, it achieves SOTA performance (76.82%) on the LoCoMo long-dialogue benchmark while significantly reducing token consumption (1.94M vs. 35.8M for graph memory) and the number of API calls.

## Background & Motivation

**Background**: Persistent memory systems are essential for LLM agents to maintain coherence during long-term interactions. Existing memory systems generally follow two paradigms: flat memory, which stores facts or summaries as independent units in vector databases for similarity retrieval; and graph memory, which constructs knowledge graphs via entity-relation extraction to support structured reasoning.

**Limitations of Prior Work**: Flat memory is efficient but lacks the ability to model cross-event relationships—retrieval often degrades into shallow similarity matching, failing at temporal reasoning and multi-hop QA. Graph memory can restore relational structures but is extremely costly, requiring cascaded LLM operations (extraction, deduplication, and updating). It is also fragile, as noisy extractions lead to propagated structural errors. For instance, Mem0^g consumes 35.8M tokens, necessitates 53,514 API calls, and requires 115,670 seconds of execution time.

**Key Challenge**: There is a fundamental trade-off between efficiency and structured reasoning. Flat methods are fast but shallow, while graph methods are deep but slow. The root cause is the selection of inappropriate memory units: isolated facts lose context, while triplets impose overly rigid schemas.

**Goal**: To design a memory unit that preserves the causal and interpersonal context of events without requiring explicit schema design, entity resolution, or symbolic graph traversal.

**Key Insight**: The fundamental unit of conversational memory should not be an isolated fact or a triplet, but rather a "temporally anchored relational event"—preserving "what happened" and "how events interrelate across subjects and time."

**Core Idea**: Use event-level binding (dual-perspective extraction + temporal anchoring) to preserve local structures and cross-event consolidation (semantic retrieval + batch synthesis) to construct global connections. This enables structured reasoning without building an explicit graph.

## Method

### Overall Architecture

StructMem operates across two layers: the event-level structure (§3.1) preserves relational bindings within individual utterances through dual-perspective extraction and temporal anchoring; the cross-event structure (§3.2) connects information across temporal boundaries via periodic semantic consolidation. The input is a dialogue stream, and the output is a hierarchically organized memory bank that supports downstream RAG-style questioning.

### Key Designs

1.  **Dual-Perspective Extraction**:
    - **Function**: Simultaneously extract factual content and relational context from each utterance.
    - **Mechanism**: For each utterance $m_i$, the LLM is invoked with two different prompts: $\Phi_i = \mathcal{L}(P_{fact} \| m_i)$ extracts factual entries (descriptions of event content), and $\Psi_i = \mathcal{L}(P_{rel} \| m_i)$ extracts relational entries (interpersonal dynamics, causal impacts, and temporal dependencies). All entries use natural language instead of triplets to avoid entity resolution overhead.
    - **Design Motivation**: Single-perspective extraction yields either only facts (flat memory) or only relations (triplets). Dual-perspective extraction ensures the preservation of contextual nuances necessary for episodic grounding.

2.  **Temporal Anchoring**:
    - **Function**: Bind factual and relational entries to original timestamps to form event-level units.
    - **Mechanism**: All entries are anchored to their original timestamp $\tau_i$, forming $\mathcal{M} \leftarrow \bigcup_{i=1}^{N} \{ \langle x, \mathbf{e}_x, \tau_i \rangle \mid x \in \Phi_i \cup \Psi_i \}$, where $\mathbf{e}_x$ is the entry embedding. During retrieval, the full fact-relation event can be reconstructed via the timestamp.
    - **Design Motivation**: Without temporal anchoring, factual and relational information becomes fragmented, making temporal reasoning impossible. Temporal coupling is critical for restoring event integrity from flat retrieval results.

3.  **Cross-Event Semantic Consolidation**:
    - **Function**: Periodically synthesize semantically related events to build high-level relational hypotheses across temporal boundaries.
    - **Mechanism**: Triggered when accumulated events exceed a temporal threshold. Unconsolidated entries in the buffer are first sorted chronologically and encoded as aggregate queries to retrieve the top-K semantically similar entries from history as seeds. For each seed, its full event context is reconstructed as $E_\tau(x^*) = \{x' \in \mathcal{M} \mid \tau(x') = \tau(x^*)\}$. Reconstructed and buffered events are merged, and the LLM synthesizes cross-event relational hypotheses. This is not lossy compression but the creation of new information not explicitly present in single memory entries.
    - **Design Motivation**: By leveraging temporal locality—the tendency for semantically related events to cluster within short windows—Ours reduces per-event operations to periodic batch processing, drastically cutting API calls and token consumption.

### Loss & Training
This framework operates at inference time and does not involve model training. All methods utilize gpt-4o-mini as the backbone and text-embedding-3-small for embeddings.

## Key Experimental Results

### Main Results (LoCoMo Benchmark)

| Method | Overall | Multi-hop | Temporal | Token (M) | API Calls | Time (s) |
|------|---------|-----------|----------|-----------|-----------|----------|
| FullContext | 73.83 | 68.79 | 50.16 | – | – | – |
| Mem0 | 66.88 | 67.13 | 59.19 | 12.196 | 9181 | 30057 |
| Mem0^g (Graph) | 68.44 | 65.71 | 58.13 | 35.825 | 53514 | 115670 |
| Zep | 75.14 | 74.11 | 67.71 | – | – | – |
| Memobase | 75.78 | 70.92 | 85.05 | – | – | – |
| **StructMem** | **76.82** | **68.77** | **81.62** | **1.937** | **1056** | **22854** |

### Ablation Study

| Configuration | Multi-hop | Temporal |
|------|-----------|----------|
| Flat Memory (Baseline) | 66.31 | 78.50 |
| Graph Memory | 66.67 | 76.64 |
| w/o Cross-Event | 66.31 | 79.44 |
| StructMem (Full) | 68.77 | 81.62 |

### Key Findings
- **StructMem achieves a SOTA Overall score of 76.82%**, outperforming Memobase (75.78%) and Zep (75.14%). Its temporal reasoning (81.62%) is second only to Memobase (85.05%).
- **Efficiency gains are highly significant**: Token consumption is only 1.94M, which is 1/18th of Mem0^g (35.8M); API calls are at 1,056, which is 1/50th of Mem0^g (53,514).
- Ablation reveals that event-level structures primarily improve temporal reasoning (78.50→79.44), while cross-event consolidation further pushes this to 81.62%.
- Graph Memory actually performs worse than flat memory in temporal reasoning (76.64 vs 78.50), indicating that rigid triplet structures can be detrimental to temporal modeling.
- Flat retrieval performance peaks and plateaus at 60 entries, suggesting that the bottleneck lies in knowledge reasoning rather than coverage.

## Highlights & Insights
- The insight that **"memory units should be temporally anchored relational events"** is very precise, effectively identifying a "third way" between flat and graph approaches. The design combining natural language representation with temporal coupling is simple but effective.
- The utilization of the **temporal locality hypothesis** is clever: semantically relevant events cluster within short time windows, making periodic consolidation more efficient than per-event graph updates. This assumption is highly valid in dialogue scenarios.
- Cross-event consolidation generates "relational hypotheses" rather than just compressed summaries. This is a creative enhancement that injects reasoning chains into the memory that were not directly present in the raw data.

## Limitations & Future Work
- The quality of dual-perspective extraction depends heavily on prompt design; suboptimal prompts may lead to incomplete or inaccurate relational information.
- There is a lack of an explicit conflict resolution and memory update mechanism—user preferences may evolve over time, leading to inconsistencies between historical summaries and new information.
- Evaluation was limited to the LoCoMo benchmark; verification on other long-dialogue benchmarks (such as LongMemEval) is still needed.
- While the temporal locality hypothesis holds for dialogues, it may not apply to other scenarios, such as work logs spanning many disparate days.

## Related Work & Insights
- **vs Mem0^g**: A graph memory method requiring entity-relation extraction and graph maintenance. StructMem replaces triplets with natural language events, improving efficiency 18-fold.
- **vs HiMem**: Organizes hierarchical text segments using physical session boundaries. StructMem does not rely on session boundaries but rather performs cross-event connections based on semantic similarity.
- **vs TiMem**: Introduces turn-by-turn Chain-of-Thought reflection to deepen single-turn understanding, but incurs overhead at every turn. StructMem’s batch consolidation strategy is more cost-effective.
- **vs EMem**: Preserves raw episodes to prioritize retrieval faithfulness. StructMem actively synthesizes cross-event relationships while maintaining original memories.

## Rating
- Novelty: ⭐⭐⭐⭐ The hierarchical design of dual-perspective + temporal anchoring + semantic consolidation is innovative, though individual components are not entirely unprecedented.
- Experimental Thoroughness: ⭐⭐⭐ Only one benchmark (LoCoMo) was used, and the ablation study could be deeper; however, the efficiency comparison is comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The comparison diagram of the three paradigms is intuitive, and the methodology descriptions are clear, though the Related Work section is somewhat long.
- Value: ⭐⭐⭐⭐ The efficiency improvements are extremely significant (1/18 tokens, 1/50 APIs), which is of great importance for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)
- [\[ACL 2026\] TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](timem_temporal-hierarchical_memory_consolidation_for_long-horizon_conversational.md)
- [\[ACL 2026\] OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation](opera_a_dataset_of_observation_persona_rationale_and_action_for_evaluating_llms_.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL](memsearcher_training_llms_to_reason_search_and_manage_memory_via_end-to-end_rein.md)

</div>

<!-- RELATED:END -->
