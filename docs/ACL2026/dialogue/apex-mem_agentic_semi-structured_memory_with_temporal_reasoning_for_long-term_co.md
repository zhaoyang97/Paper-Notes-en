---
title: >-
  [Paper Note] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI
description: >-
  [ACL 2026][Dialogue Systems][To be supplemented] To be supplemented after thorough reading.
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "To be supplemented"
date: 2026-05-08
content_hash: 2a504f4e21c62b8d
---

# APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI

**Conference**: ACL 2026
**arXiv**: [2604.14362](https://arxiv.org/abs/2604.14362)  
**Code**: None  
**Area**: Agent / Dialogue Systems
**Keywords**: Long-term memory, property graph, temporal reasoning, conversational AI, multi-tool retrieval

## TL;DR
APEX-MEM proposes a conversational memory system based on a Property Graph, append-only event storage, and a multi-tool retrieval agent. Through a domain-agnostic ontology and retrieval-time temporal reasoning, it achieves 88.88% and 86.2% accuracy on LOCOMO and LongMemEval respectively, significantly outperforming existing structured memory approaches.

## Background & Motivation

**Background**: LLMs remain weak at long-term conversational memory. Simply expanding the context window introduces noise and hallucinations, while RAG-based retrieval methods treat memory as unstructured text and cannot maintain normalized entities, track fact evolution, or distinguish persistent information from transient conversational content.

**Limitations of Prior Work**: Existing structured memory systems such as Mem0 introduce entity-centric graphs but with limited entity categories; information is stored primarily as inter-entity relations, making it difficult to capture fine-grained attributes and the temporal evolution of facts. Furthermore, most systems merge or overwrite prior information at write time, risking the loss of contextual details needed for temporal reasoning.

**Key Challenge**: Memory systems must balance "preserving complete history" against "reducing retrieval noise"—premature merging loses information, while full retention increases noise.

**Goal**: To construct a memory framework that both preserves the complete evolutionary history of conversations and dynamically resolves conflicts and outdated information at retrieval time.

**Key Insight**: Drawing on event ontology ideas from knowledge graphs, the paper anchors conversational facts to timestamped events rather than attaching them directly to entities, thereby enabling an append-write + retrieval-time resolution design paradigm.

**Core Idea**: Replace the traditional "merge-at-write" memory model with a unified architecture of "property graph + append-only event storage + retrieval-time temporal resolution."

## Method

### Overall Architecture
The system comprises two major components: Graph Construction and Graph QA Agent. The input is multi-turn conversational text; the output is memory-aware answers to user queries. Conversational text undergoes fact extraction and entity resolution before being written into the property graph. At query time, a ReAct-style agent invokes multiple tools for retrieval and reasoning.

### Key Designs

1. **Domain-Agnostic Temporal Event Ontology**:

    - Function: Provides a unified structured representation for conversational memory.
    - Mechanism: Defines 35 entity types (Person, Organization, Product, etc.). Facts are represented as temporally anchored triples $f=(s,p,v,\delta,[t_{from},t_{to}],c,\mathcal{E})$, each anchored to a timestamped conversational event. Events are modeled as first-class citizens, comprising type, time, location, participants, and associated facts.
    - Design Motivation: Unlike Mem0 and similar systems that model only entity relations, the event ontology captures both entity attributes and temporal evolution, supporting scenarios where the same entity holds different attribute values at different times.

2. **Append-Only Event Storage**:

    - Function: Preserves the complete temporal evolutionary history of information.
    - Mechanism: New facts do not overwrite old ones; instead, they are appended to the graph as new event nodes. Each fact carries a temporal validity interval $[t_{from}, t_{to}]$ and a confidence score $c \in [0,1]$. Conflicting and revised information coexist in the graph.
    - Design Motivation: Avoids information loss caused by premature merging at write time; defers conflict resolution to retrieval time, where the validity of information is determined dynamically based on query context.

3. **Multi-Tool Retrieval Agent**:

    - Function: Retrieves and reasons over the property graph at query time to produce answers.
    - Mechanism: A ReAct-style agent is equipped with four tools—SchemaViewer (meta-level planning guidance), EntityLookup (entity retrieval and normalization), GraphSQL (SQL-based structured graph traversal supporting temporal ordering, interval computation, and multi-hop reasoning), and Search (hybrid semantic + lexical search). The agent alternates between reasoning and tool invocation to incrementally construct answers.
    - Design Motivation: Different question types require different retrieval strategies—EntityLookup for entity queries, GraphSQL for temporal computation, Search for open-domain questions, and SchemaViewer for meta-planning.

### Loss & Training
The graph construction stage uses few-shot prompted LLMs for fact extraction; entity resolution employs a pipeline of dense semantic retrieval followed by structured LLM reasoning. In online construction mode, for very long conversations (>1,000 documents), semantically and lexically relevant document subsets are first retrieved before constructing a local graph.

## Key Experimental Results

### Main Results

| Dataset | Method | Overall Accuracy | Single-hop | Multi-hop | Temporal | Open-domain |
|--------|------|-----------|------|------|------|--------|
| LOCOMO | APEX-MEM (GPT5) | **88.88%** | 89.88% | 86.29% | 90.63% | 91.68% |
| LOCOMO | MIRIX | 85.38% | 85.11% | 83.70% | 65.62% | 88.39% |
| LOCOMO | Mem0 | 68.44% | 65.71% | 47.19% | 75.71% | 58.13% |
| LOCOMO | Full Context GPT4o | 87.52% | 88.53% | 77.70% | 71.88% | 92.70% |

### Ablation Study

| Configuration | Key Performance | Notes |
|------|---------|------|
| Full APEX-MEM | 88.88% | All four tools in combination |
| SimpleSearch only | 77.90% | Large drop after removing structured graph |
| EntityLookup only | — | Entity-centric retrieval; suitable for single-hop |
| GraphSQL only | — | Structured queries; suitable for temporal reasoning |

### Key Findings
- APEX-MEM shows the largest advantage on temporal reasoning tasks (90.63% vs. MIRIX 65.62%), demonstrating the value of the event ontology and append-only storage for temporal reasoning.
- Even with a weaker LLM (Claude 4.5 Haiku), APEX-MEM achieves 84.25%, indicating that the architectural design contributes more than the underlying LLM capability.
- On adversarial questions, APEX-MEM reaches 86.77%, far surpassing full-context methods, confirming that structured memory effectively suppresses noise interference.

## Highlights & Insights
- The **append-only + retrieval-time resolution** design paradigm is elegant: no decisions are made at write time, and all intelligence resides on the read side—an approach analogous in spirit to the write-ahead log in database systems.
- Using SQLite as the graph storage backend allows the GraphSQL tool to perform temporal computations directly in SQL, which is more general than custom graph query languages.
- Although the 35-class entity ontology appears complex, its domain-agnostic design achieves cross-domain applicability, avoiding the need to redesign schemas for each new setting.

## Limitations & Future Work
- Graph construction relies on LLM-based fact extraction and entity resolution, so extraction quality is bounded by LLM capability.
- Append-only storage causes the graph to grow continuously; storage and retrieval efficiency under long-term operation remains to be verified.
- Evaluation is conducted only on LOCOMO and LongMemEval; validation in real-world deployment scenarios is lacking.

## Related Work & Insights
- **vs. Mem0/Mem0g**: Mem0 has limited entity categories and stores relations directly on entities; APEX-MEM introduces an event ontology and append-only storage, yielding substantially stronger temporal reasoning.
- **vs. MIRIX**: MIRIX uses six specialized memory stores with multi-agent routing to reach 85.4%, but its eager update strategy may lose temporal information; APEX-MEM achieves higher accuracy with a simpler architecture.
- **vs. Zep**: Zep constructs temporally aware knowledge graphs but relies heavily on text retrieval; APEX-MEM's GraphSQL tool provides more precise structured query capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of event ontology + append-only storage + retrieval-time resolution is genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset comparisons are comprehensive, though ablations could be more fine-grained.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and mathematical formalization is complete.
- Value: ⭐⭐⭐⭐⭐ Provides a practical and effective solution for long-term conversational memory.
**Code**: To be confirmed  
**Area**: dialogue
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ReIn: Conversational Error Recovery with Reasoning Inception](../../ICLR2026/dialogue/rein_conversational_error_recovery_with_reasoning_inception.md)
- [\[ICLR 2026\] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](../../ICLR2026/dialogue/think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio.md)
- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)
- [\[AAAI 2026\] TalkSketch: Multimodal Generative AI for Real-time Sketch Ideation with Speech](../../AAAI2026/dialogue/talksketch_multimodal_generative_ai_for_real-time_sketch_ideation_with_speech.md)
- [\[CVPR 2026\] Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition](../../CVPR2026/dialogue/evolutionary_multimodal_reasoning_via_hierarchical_semantic_representation_for_i.md)

</div>

<!-- RELATED:END -->
