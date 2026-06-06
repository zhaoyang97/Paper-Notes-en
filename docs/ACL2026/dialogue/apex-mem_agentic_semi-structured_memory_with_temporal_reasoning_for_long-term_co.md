---
title: >-
  [Paper Note] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI
description: >-
  [ACL 2026][Dialogue Systems][Property Graph] Constructs long-term dialogue memory as a triplet of "Property Graph supported by domain-agnostic ontology + Append-only event storage + ReAct multi-tool retrieval agent"—neve…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Property Graph"
  - "Append-Only Storage"
  - "Temporal Reasoning"
  - "ReAct Agent"
  - "GraphSQL"
date: 2026-05-08
content_hash: addd3e4ec7e34efe
---

# APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI

**Conference**: ACL 2026  
**arXiv**: [2604.14362](https://arxiv.org/abs/2604.14362)  
**Code**: None  
**Area**: Dialogue Systems / Long-term Memory / Agents  
**Keywords**: Property Graph, Append-Only Storage, Temporal Reasoning, ReAct Agent, GraphSQL

## TL;DR
Constructs long-term dialogue memory as a triplet of "Property Graph supported by domain-agnostic ontology + Append-only event storage + ReAct multi-tool retrieval agent"—never overwriting during construction and resolving temporal conflicts at retrieval. It achieved 88.88% on LOCOMO (3.5 points higher than MIRIX) and 86.2% on LongMemEval (13.7 points higher than the strongest RAG baseline).

## Background & Motivation

**Background**: LLMs in long-term multi-turn dialogues need to "remember previous sessions, accumulate knowledge across sessions, and update facts as context evolves." Common solutions fall into two categories: ① Lengthening context windows (128K only yields 51.6% F1); ② RAG retrieving text/summary segments (noise remains heavy). Recent structured memory systems like Mem0 / A-MEM / Zep / MIRIX organize memory as graphs but are far from "glass-box" standards.

**Limitations of Prior Work**: Current structured memory systems remain weak in two aspects:
① Entity-centric graphs like Mem0 cram all information into entity-relation triples, with limited entity categories and an inability to capture the temporal evolution of properties.
② Almost all systems perform **merging/overwriting** (eager update / state merge) during writing, discarding historical versions early and failing to answer temporal questions like "When did X change?" or "What was Y like before?".

**Key Challenge**: There is a natural conflict between "retaining complete history" (beneficial for temporal reasoning) and "reducing retrieval noise" (beneficial for answering accuracy)—early merging loses information, while complete retention increases the retrieval burden.

**Goal**: Postpone conflict resolution from writing time to retrieval time, allowing the memory system to retain a complete timeline while dynamically selecting the correct version based on context during queries.

**Key Insight**: Drawing inspiration from the entity type hierarchy of YAGO 4.5 in knowledge graphs and the append-only log concept in databases, dialogue facts are **anchored to timestamped events** rather than directly to entities.

**Core Idea**: A trinity of "Property Graph + Append-only event storage + Multi-tool temporal reasoning at retrieval"—applying subtraction at the writing end (no merging) and addition at the reading end (multi-tool ReAct).

## Method

### Overall Architecture
The system consists of two stages. **Construction phase**: Each source document $d_i$ is extracted into a subgraph $g_i$, which is incrementally merged into the global property graph $G^{(t+1)}\leftarrow\mathrm{Merge}(G^{(t)},g_t)$ via soft-canonicalization. Formally, $G=(V,E,\Pi,\Lambda)$, where $\Pi$ is the key-value property map and $\Lambda$ denotes ontology types for nodes/edges. **Query phase**: The ReAct agent $\pi_\theta$ generates a reasoning trace $r_t$ and action $a_t$ at each step $t$. Actions are either tool calls $(T_t,z_t)$ (where $T_t\in\{\text{SchemaViewer, EntityLookup, GraphSQL, Search}\}$) or an Answer. The agent resolves temporal references into specific dates before calling tools and providing final answers.

### Key Designs

1. **Hybrid Entity-Event Ontology + Temporally Anchored Facts**:
    - **Function**: Provides a unified semantic structure for dialogue memory, capable of expressing both entity properties and the temporal evolution of facts.
    - **Mechanism**: Defines 35 entity classes (Person / Organization / Product / Place / Event / Software / ...). Facts are written as temporally anchored tuples $f=(s,p,v,\delta,[t_{\text{from}},t_{\text{to}}],c,\mathcal{E})$, where $s$ is the subject entity, $p$ the property name, $v$ the value, $\delta$ the data type, $[t_{\text{from}},t_{\text{to}}]$ the validity interval, $c\in[0,1]$ the confidence, and $\mathcal{E}$ the supporting evidence set. All facts must be attached to dialogue events $\varepsilon=(\text{type},T,L,P,F,\mathcal{E}_\varepsilon)$.
    - **Design Motivation**: Designs like Mem0 that treat information directly as "entity-relation" cannot represent property evolutions such as "Alice's favorite was Italian Garden on 2024-01-15 and changed to Sakura Sushi on 2024-03-20." Treating events as first-class citizens with temporal intervals makes fine-grained temporal reasoning truly writable and searchable.

2. **Append-Only Event Storage + Temporal Resolution at Retrieval**:
    - **Function**: Never overwrites old facts during construction; all conflicts and revisions are appended to the graph as new events. Conflict resolution is postponed until query time, adjudicated based on temporal validity.
    - **Mechanism**: Entity and property resolution use a RAG-style approach—given a mention $m$, dense embedding retrieves top-k candidates $C=\{(\text{id}_i,\text{text}_i,s_i)\}$, then a structured LLM outputs a decision $d\in\{\text{choose\_existing, propose\_new, none}\}$ and confidence, but only appends nodes without deleting old ones. During queries, GraphSQL sorts by `created_at` or `from_date`, and the agent selects the latest one (or the one valid at the time specified by the question).
    - **Design Motivation**: Early merging equals information loss, causing systems to fail on simple temporal questions (e.g., Mem0 drops 18% on multi-hop, MIRIX drops 20% on temporal; see ablation table). Append-only preserves the timeline, converting silent failure modes into "queryable" states.

3. **Multi-Tool ReAct Retrieval Agent (SchemaViewer / EntityLookup / GraphSQL / Search)**:
    - **Function**: Dynamically combines retrieval strategies at query time based on question characteristics, merging structured reasoning (SQL), entity normalization, and semantic search into a single reasoning loop.
    - **Mechanism**: SchemaViewer provides DB schema and tool usage suggestions as a meta-planner; EntityLookup normalizes surface mentions to graph IDs and returns timestamped fact snapshots; GraphSQL is a read-only SELECT interface (whitelist tables: entities/properties/facts/events/evidence/turns), supporting julianday temporal calculation, aggregation, and multi-hop JOINs; Search performs hybrid dense+lexical retrieval, returning subgraphs $(E_q,P_q,\mathcal{V}_q,\mathcal{T}_q)$ related to the question. The agent iteratively decides $(r_t,a_t)\sim\pi_\theta(\cdot\mid x,h_t)$ up to 40 steps.
    - **Design Motivation**: No single tool is sufficient—pure GraphSQL requires 27,282 calls to solve a problem set (3.3× waste), and pure EntityLookup multi-hop reasoning caps at 77%. Different questions (single-hop / multi-hop / temporal / open-domain / adversarial) require different optimal tool combinations; letting the agent choose outperformed any single-tool solution.

### Loss & Training
No retraining is involved. Fact extraction uses Claude Sonnet 4.5, and entity/property resolution uses Claude Haiku 4.5 (balancing cost and quality). Queries use strong models like GPT-5 or Claude Sonnet 4.5 with tools. For ultra-long dialogues (>$10^3$ documents), APEX-MEM Online is used: filtering a subset with $\mathrm{Relevance}(d_i|Q)>\Theta_{\text{rel}}=0.2$ before building local graphs. All experiments use temperature=0 with a 40-call tool limit.

## Key Experimental Results

### Main Results
Evaluated on LOCOMO (multi-session long-term memory), LongMemEval (long-input memory), and SealQA-Hard (noisy multi-document fact-based QA), compared against MIRIX / Mem0 / Zep / A-MEM / Nemori / MemGPT / OpenAI Memory.

| Dataset | Method | Overall | Single-hop | Multi-hop | Temporal | Open-domain | Adversarial |
|---|---|---|---|---|---|---|---|
| LOCOMO | **APEX-MEM (GPT-5)** | **88.88%** | **89.88%** | **86.29%** | **90.63%** | **91.68%** | 86.77% |
| LOCOMO | APEX-MEM (Claude 4.5 Sonnet) | 88.41% | 89.36% | 86.92% | 90.63% | 87.75% | 86.10% |
| LOCOMO | MIRIX | 85.38% | 85.11% | 83.70% | 65.62% | 88.39% | N/A |
| LOCOMO | Nemori | 79.40% | 84.90% | 75.10% | 77.60% | 51.00% | N/A |
| LOCOMO | Mem0 | 68.44% | 65.71% | 47.19% | 75.71% | 58.13% | N/A |
| LOCOMO | Zep | 75.14% | 61.70% | 41.35% | 76.60% | 49.31% | N/A |
| LOCOMO | Full Context GPT-4o | 87.52% | 88.53% | 77.70% | 71.88% | 92.70% | N/A |
| LongMemEval | **APEX-MEM (Sonnet)** | **86.2%** | - | - | - | - | - |
| LongMemEval | Nemori | 74.6% | - | - | - | - | - |
| SealQA-Hard | **APEX-MEM (GPT-5)** | **40.15%** | - | - | - | - | - |
| SealQA-Hard | O3 | 34.6% | - | - | - | - | - |

### Ablation Study (Tool combination, LOCOMO, Claude 4.5 Haiku backend)

| Configuration | Single-hop | Multi-hop | Temporal | Open-domain | Adversarial | Overall |
|---|---|---|---|---|---|---|
| SchemaViewer + EntityLookup | 80.85 | 76.64 | 72.92 | 76.34 | 77.80 | 77.19 |
| + GraphSQL | 80.78 | 79.75 | 82.29 | 78.00 | 81.16 | 79.45 |
| **+ Search (All 4 tools)** | **85.46** | **84.74** | 79.17 | **89.18** | **87.22** | **87.00** |

### Key Findings
- **Temporal reasoning is the biggest advantage**: APEX-MEM 90.63% vs MIRIX 65.62% (25-point gap) and Mem0 75.71% (15-point gap), validating the effectiveness of the append-only + GraphSQL temporal operation combination.
- **Search tool is critical for open-domain (+11 points)**: Improved from 78.00 to 89.18; pure structured queries cannot cover semantically vague questions.
- **GraphSQL cannot stand alone**: Using it alone requires 27,282 calls (3.3× waste) and peaks at 79.45% accuracy; combined with Search, calls drop to 8,260, improving efficiency by 3x.
- **Strong LLMs require fewer tool calls**: Claude 4.5 Sonnet reaches 84-86% accuracy with 10 calls, while GPT-4o requires more due to higher SQL generation error rates, resulting in lower final precision (86.35% vs 88.88%).
- **Token cost sweet spot**: APEX-MEM averages 30,000 tokens per query, nearly 4x lower than MIRIX's 112,000, while maintaining higher accuracy; graph construction costs only 16.6%.
- **SQL error recovery mechanism**: 87% of SQL failures are automatically recovered via SchemaViewer re-querying (45%), EntityLookup fallback (28%), or Search fallback (14%), demonstrating strong multi-tool architecture resilience.

## Highlights & Insights
- **"Subtraction during writing, addition during reading" is an elegant paradigm**: Similar to database Write-Ahead Logs (WAL), intelligence is concentrated at the reading end. Writing is always append-only, turning conflict resolution into a "query problem adjudicated by query time" rather than a "prediction problem guessing which is correct in advance."
- **Using SQLite for graph storage is a pragmatic engineering choice**: Allows GraphSQL to reuse SQL's julianday, aggregation, and JOINs, which is much more versatile than custom Cypher/SPARQL languages. LLMs are also much more proficient in SQL than Cypher (Sonnet SQL success rate 97.6%).
- **Balance of 35 entity ontology classes**: Fine enough to distinguish Software/Service/Device yet broad enough for cross-domain application; properties can be flexibly attached, finding a sweet spot between a strict schema and completely schemaless.
- **Converting silent failure into queryable state**: Allows questions like "When did Alice's favorite restaurant change?" or "What was Bob's previous job?"—questions unanswerable in overwriting systems like Mem0.
- **Indirect empirical evidence for append-only advantages (Appendix F)**: APEX-MEM Temporal 90.63% vs Mem0 75.71% (–14.92) vs MIRIX 65.62% (–25.01) vs Zep 76.60% (–14.03) strongly suggests that the update strategy, rather than LLM strength, is the true bottleneck for temporal ceilings.

## Limitations & Future Work
- Graph construction relies on large models for fact extraction (Claude Sonnet 4.5 extraction precision is 97.3%, schema coverage 91.1%, entity resolution 98.2%), leading to high real-time deployment costs; the authors plan to research small model alternatives and caching.
- Multi-tool ReAct averages 20-30 tool calls to converge, potentially affecting latency in interactive scenarios; marginal returns diminish after 20 calls, necessitating RL for smarter stop criteria.
- The 35-class domain-agnostic ontology may lack detail for specialized fields (Medical / Legal), lacking automatic ontology refinement capabilities.
- The 40.15% score on SealQA-Hard suggests room for improvement in high-noise multi-document scenarios, mainly due to insufficient fact extraction coverage of implicit relations and temporal details.
- The system only supports text dialogue; multimodal (image/audio) inputs are not covered.
- Evaluation is limited to QA; tasks requiring narrative synthesis like event summarization or dialogue generation remain unverified.

## Related Work & Insights
- **vs Mem0 / Mem0g**: Mem0 has limited entity types and merges upon writing; APEX-MEM uses 35 ontology classes + append-only, gaining +15 points in temporal reasoning.
- **vs MIRIX**: MIRIX uses 6 specialized memories + multi-agent routing to reach 85.4%, but eager state merge drops temporal awareness to 65.62%; APEX-MEM's simpler architecture achieves 90.63% temporal accuracy.
- **vs Zep / Graphiti**: Zep builds temporal KGs but relies heavily on text retrieval; GraphSQL provides missing precise structured query capabilities; APEX-MEM improves temporal reasoning by 14 points.
- **vs A-MEM / Nemori**: A-MEM uses Zettelkasten-style autonomous linking and Nemori draws from cognitive science for self-organizing memory, but neither solves the fundamental "merge upon writing" problem.
- **vs Full Context GPT-4o**: Directly providing a 128K context reaches 87.52%, which APEX-MEM has surpassed with only 1.2x the average token consumption (30K vs 25K)—proving structured memory doesn't have to sacrifice efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "event ontology + append-only + resolution at retrieval" is clever; while individual elements aren't entirely new, the paradigm they form is a breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across three benchmarks, tool ablation, model comparisons, token cost breakdown, and SQL failure recovery analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical formalization, intuitive case studies, and extremely detailed appendices.
- Value: ⭐⭐⭐⭐⭐ Provides a ready-to-implement engineering template for "long-term dialogue memory"; the append-only philosophy is transferable to broader agent memory systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ReIn: Conversational Error Recovery with Reasoning Inception](../../ICLR2026/dialogue/rein_conversational_error_recovery_with_reasoning_inception.md)
- [\[ICLR 2026\] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](../../ICLR2026/dialogue/think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)
- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)
- [\[ACL 2026\] ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking](reactod_bounded_neuro-symbolic_agentic_nlu_for_zero-shot_dialogue_state_tracking.md)

</div>

<!-- RELATED:END -->
