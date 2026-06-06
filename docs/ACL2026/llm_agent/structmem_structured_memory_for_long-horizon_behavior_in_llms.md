---
title: >-
  [Paper Note] StructMem: Structured Memory for Long-Horizon Behavior in LLMs
description: >-
  [ACL 2026][LLM Agent][long-term memory] StructMem proposes a structure-enhanced hierarchical memory framework that achieves state-of-the-art performance on the LoCoMo long-conversation benchmark (76.82%) through dual-per…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "long-term memory"
  - "event-level binding"
  - "cross-event consolidation"
  - "hierarchical memory"
  - "multi-hop reasoning"
date: 2026-05-08
content_hash: 988090ecb0dd3299
---

# StructMem: Structured Memory for Long-Horizon Behavior in LLMs

**Conference**: ACL 2026
**arXiv**: [2604.21748](https://arxiv.org/abs/2604.21748)  
**Code**: [https://github.com/zjunlp/LightMem](https://github.com/zjunlp/LightMem)  
**Area**: LLM Agent / Dialogue Systems
**Keywords**: long-term memory, event-level binding, cross-event consolidation, hierarchical memory, multi-hop reasoning

## TL;DR

StructMem proposes a structure-enhanced hierarchical memory framework that achieves state-of-the-art performance on the LoCoMo long-conversation benchmark (76.82%) through dual-perspective event-level extraction and cross-event semantic consolidation, while substantially reducing token consumption (1.94M vs. 35.8M for graph memory) and API call counts.

## Background & Motivation

**Background**: Persistent memory systems are essential for LLM agents to maintain coherence over long-term conversations. Existing memory systems fall into two paradigms: flat memory stores facts or summaries as independent units and performs similarity-based retrieval via vector databases; graph memory constructs knowledge graphs through entity-relation extraction to support structured reasoning.

**Limitations of Prior Work**: Flat memory is efficient but cannot model cross-event relationships—retrieval degenerates into shallow similarity matching, precluding temporal reasoning and multi-hop QA. Graph memory recovers relational structure but at prohibitive cost, requiring cascaded LLM operations (entity extraction, relation extraction, deduplication, and updates), and is fragile—noisy extraction propagates structural noise. Mem0^g alone consumes 35.8M tokens, 53,514 API calls, and 115,670 seconds of runtime.

**Key Challenge**: A fundamental trade-off between efficiency and structured reasoning. Flat approaches are fast but shallow; graph approaches are deep but slow. The root issue lies in inappropriate choice of memory unit: isolated facts lose context, while triples impose rigid schemas.

**Goal**: Design a memory unit that preserves causal and interpersonal relational context of events without requiring explicit schema design, entity resolution, or symbolic graph traversal.

**Key Insight**: The fundamental unit of conversational memory should not be an isolated fact or triple, but a "temporally anchored relational event" that retains both what happened and how events are interrelated across participants and time.

**Core Idea**: Use event-level binding (dual-perspective extraction + temporal anchoring) to preserve local structure, and cross-event consolidation (semantic retrieval + batch synthesis) to build global connections, achieving structured reasoning without constructing an explicit graph.

## Method

### Overall Architecture

StructMem operates at two levels: event-level structure (§3.1) preserves relational binding within individual utterances via dual-perspective extraction and temporal anchoring; cross-event structure (§3.2) connects information across temporal boundaries through periodic semantic consolidation. The input is a dialogue stream; the output is a hierarchically organized memory store supporting downstream RAG-style question answering.

### Key Designs

1. **Dual-Perspective Extraction**:

    - Function: Simultaneously extract factual content and relational context from each utterance.
    - Mechanism: For each utterance $m_i$, two distinct prompts are used to invoke the LLM: $\Phi_i = \mathcal{L}(P_{fact} \| m_i)$ extracts factual entries (event content descriptions), and $\Psi_i = \mathcal{L}(P_{rel} \| m_i)$ extracts relational entries (interpersonal dynamics, causal influences, temporal dependencies). All entries are represented in natural language rather than triples, avoiding the overhead of entity resolution.
    - Design Motivation: Single-perspective extraction captures either facts (flat memory) or relations (triples) but not both. The dual-perspective design ensures that contextual nuances required for episodic grounding are preserved.

2. **Temporal Anchoring**:

    - Function: Bind factual and relational entries to their original timestamps, forming event-level units.
    - Mechanism: All entries are anchored to their original timestamp $\tau_i$, yielding $\mathcal{M} \leftarrow \bigcup_{i=1}^{N} \{ \langle x, \mathbf{e}_x, \tau_i \rangle \mid x \in \Phi_i \cup \Psi_i \}$, where $\mathbf{e}_x$ is the entry embedding. During retrieval, timestamps enable reconstruction of the complete fact-relation event.
    - Design Motivation: Without temporal anchoring, factual and relational information become scattered, precluding temporal reasoning. Temporal coupling is the key to recovering event integrity beyond flat retrieval.

3. **Cross-Event Semantic Consolidation**:

    - Function: Periodically synthesize semantically related events to construct higher-order relational hypotheses across temporal boundaries.
    - Mechanism: Triggered when accumulated events exceed a temporal threshold. Unconsolidated entries in the buffer are sorted chronologically and encoded as an aggregated query to retrieve the top-$K$ semantically most similar historical entries as seeds. For each seed entry, the complete event context is reconstructed via its timestamp: $E_\tau(x^*) = \{x' \in \mathcal{M} \mid \tau(x') = \tau(x^*)\}$. The reconstructed events and buffered events are merged, and an LLM synthesizes cross-event relational hypotheses—this is not lossy compression but the creation of new information absent from any individual memory entry.
    - Design Motivation: Exploits temporal locality—semantically related events naturally cluster within short time windows—reducing per-event operations to periodic batch processing, substantially cutting API calls and token consumption.

### Loss & Training
StructMem is an inference-time framework and involves no model training. All methods use gpt-4o-mini as the backbone and text-embedding-3-small for embeddings.

## Key Experimental Results

### Main Results (LoCoMo Benchmark)

| Method | Overall | Multi-hop | Temporal | Token (M) | API Calls | Time (s) |
|--------|---------|-----------|----------|-----------|-----------|----------|
| FullContext | 73.83 | 68.79 | 50.16 | – | – | – |
| Mem0 | 66.88 | 67.13 | 59.19 | 12.196 | 9181 | 30057 |
| Mem0^g (Graph) | 68.44 | 65.71 | 58.13 | 35.825 | 53514 | 115670 |
| Zep | 75.14 | 74.11 | 67.71 | – | – | – |
| Memobase | 75.78 | 70.92 | 85.05 | – | – | – |
| **StructMem** | **76.82** | **68.77** | **81.62** | **1.937** | **1056** | **22854** |

### Ablation Study

| Configuration | Multi-hop | Temporal |
|---------------|-----------|----------|
| Flat Memory (baseline) | 66.31 | 78.50 |
| Graph Memory | 66.67 | 76.64 |
| w/o Cross-Event | 66.31 | 79.44 |
| StructMem (Full) | 68.77 | 81.62 |

### Key Findings
- **StructMem achieves SOTA overall accuracy of 76.82%**, surpassing Memobase (75.78%) and Zep (75.14%), with temporal reasoning at 81.62% second only to Memobase (85.05%).
- **Efficiency gains are striking**: token consumption of 1.94M is 1/18 that of Mem0^g (35.8M); API calls of 1,056 are 1/50 that of Mem0^g (53,514).
- Ablations show that event-level structure primarily improves temporal reasoning (78.50→79.44), with cross-event consolidation providing further gains to 81.62%.
- Graph memory yields worse temporal reasoning than flat memory (76.64 vs. 78.50), indicating that rigid triple structures are harmful to temporal modeling.
- Flat retrieval performance peaks at 60 entries and plateaus, suggesting that the bottleneck lies in knowledge reasoning rather than coverage.

## Highlights & Insights
- The insight that **"the fundamental memory unit should be a temporally anchored relational event"** is highly precise, identifying a third path between flat and graph paradigms. The design of natural language representation combined with temporal coupling is simple yet effective.
- The exploitation of the **temporal locality assumption** is particularly elegant: semantically related events cluster within short time windows, making periodic consolidation more efficient than per-event graph updates. This assumption holds strongly in conversational settings.
- Cross-event consolidation produces "relational hypotheses" rather than compressed summaries—a form of creative augmentation that injects reasoning chains not directly present in the raw data.

## Limitations & Future Work
- The quality of dual-perspective extraction is highly dependent on prompt design; suboptimal prompts may yield incomplete or inaccurate relational information.
- The framework lacks explicit conflict resolution and memory update mechanisms—user preferences may evolve over time, and historical summaries may become inconsistent with new information.
- Evaluation is conducted on a single benchmark (LoCoMo); generalization to other long-conversation benchmarks (e.g., LongMemEval) remains unvalidated.
- The temporal locality assumption holds in conversational settings but may not generalize to other scenarios (e.g., work logs spanning multiple days).

## Related Work & Insights
- **vs. Mem0^g**: A graph memory approach requiring entity-relation extraction and graph maintenance. StructMem replaces triples with natural language events, achieving an 18× efficiency improvement.
- **vs. HiMem**: Organizes hierarchical text segments using physical session boundaries. StructMem does not rely on session boundaries, instead building cross-event connections based on semantic similarity.
- **vs. TiMem**: Introduces per-turn reflective chain-of-thought to deepen single-turn understanding, but incurs per-turn overhead. StructMem's batch consolidation strategy is more cost-efficient.
- **vs. EMem**: Retains original episodes to prioritize retrieval faithfulness. StructMem preserves original memories while actively synthesizing cross-event relational information.

## Rating
- Novelty: ⭐⭐⭐⭐ The hierarchical design combining dual-perspective extraction, temporal anchoring, and semantic consolidation is innovative, though each individual component is not entirely novel in isolation.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation is limited to a single benchmark (LoCoMo) and ablations are not sufficiently in-depth; however, efficiency comparisons are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The three-paradigm comparison figure is intuitive and method descriptions are clear; the Related Work section is overly lengthy.
- Value: ⭐⭐⭐⭐ Efficiency gains are remarkable (1/18 tokens, 1/50 API calls), carrying significant implications for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](../../ICLR2026/llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/harnessing_uncertainty_entropy-modulated_policy_gradients_for_long-horizon_llm_a.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](../../CVPR2026/llm_agent/carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)

</div>

<!-- RELATED:END -->
