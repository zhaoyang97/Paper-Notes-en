---
title: >-
  [Paper Note] REMem: Reasoning with Episodic Memory in Language Agents
description: >-
  [ICLR 2026][LLM Agent][episodic memory] This paper proposes REMem, an episodic memory framework for language agents that employs a hybrid memory graph (temporally-aware gist nodes combined with factual triple nodes) and tool-augmented agentic reasoning, achieving improvements of 3.4% and 13.4% over the state of the art on episodic recall and episodic reasoning tasks, respectively.
tags:
  - ICLR 2026
  - LLM Agent
  - episodic memory
  - language agent
  - hybrid memory graph
  - temporal reasoning
  - agentic retrieval
  - gist extraction
date: 2026-05-08
content_hash: 0c948a44bb9a6975
---

# REMem: Reasoning with Episodic Memory in Language Agents

**Conference**: ICLR 2026
**arXiv**: [2602.13530](https://arxiv.org/abs/2602.13530)
**Code**: [intuit-ai-research/REMem](https://github.com/intuit-ai-research/REMem)
**Area**: LLM Agent
**Keywords**: episodic memory, language agent, hybrid memory graph, temporal reasoning, agentic retrieval, gist extraction

## TL;DR

This paper proposes REMem, an episodic memory framework for language agents that employs a hybrid memory graph (temporally-aware gist nodes combined with factual triple nodes) and tool-augmented agentic reasoning, achieving improvements of 3.4% and 13.4% over the state of the art on episodic recall and episodic reasoning tasks, respectively.

## Background & Motivation

Humans excel at recalling concrete experiences and reasoning within spatiotemporal contexts—a capability known as episodic memory—yet the memory systems of current language agents suffer from significant deficiencies:

**Dominance of semantic memory**: Existing systems (parametric memory, RAG, GraphRAG) primarily store decontextualized semantic knowledge, lacking temporal and spatial dimensions.

**Absence of event modeling**: Mem0 over-filters information, causing loss of details; Graphiti constructs entity-centric knowledge graphs that discard coherent event context; HippoRAG 2 lacks temporal dimension modeling.

**Retrieval insufficient for reasoning**: Existing approaches rely on simple similarity matching, which cannot support complex cross-event reasoning such as temporal range filtering, event ordering, and counting queries.

The core design philosophy of REMem draws on two insights:
- Cognitive science indicates that humans rely more on gist-based representations than verbatim memory for decision-making.
- Contextual dimensions such as time, location, and participants must be explicitly bound to event representations.

## Method

### Overall Architecture

REMem operates in two stages:
1. **Indexing stage**: Converts experiences into a hybrid memory graph.
2. **Agentic reasoning stage**: Iteratively retrieves and reasons over the memory graph using a carefully designed set of tools.

### Key Designs: Indexing Stage

**1) Gist extraction**:
- For each event or dialogue session, one or more natural-language gist statements are generated.
- Each gist is annotated with a timestamp (reference time), with relative temporal expressions converted to absolute dates.
- Gists capture core information including participants, actions, objects, locations, intentions, and quantities.

**2) Fact extraction**:
- Structured $(subject, predicate, object)$ triples are extracted from both raw text and gists.
- Each triple is augmented with Wikidata-style temporal qualifiers: `point_in_time`, `start_time`, and `end_time`.
- Potentially contradictory historical records are retained to support temporal retrospective queries.

**3) Graph construction**:
- **Gist nodes**: Context-level episodic representations, linked to phrase nodes from the same chunk.
- **Phrase nodes**: Concept-level representations; subject/object nodes are directly connected via predicate edges.
- **Synonymy edges**: Added between gist nodes whose embedding similarity exceeds a threshold of 0.8.
- The result is a **hybrid memory graph**: a unified representation integrating concept-level and context-level information.

### Key Designs: Agentic Reasoning Stage

A ReAct-style agent is employed, equipped with three categories of carefully designed tools:

| Tool Category | Tool Name | Key Parameters |
|---|---|---|
| Retrieval | `semantic_retrieve` | query, start_time, end_time, temporal operator |
| Retrieval | `lexical_retrieve` | query, start_time, end_time, temporal operator |
| Graph exploration | `find_gist_contexts` | gist_id, time range |
| Graph exploration | `find_entity_contexts` | subject, object, predicate, time range, limit, ordering, offset, aggregation |
| Flow control | `output_answer` | answer |

Reasoning follows a three-phase protocol:
1. **Retrieval**: Semantic/lexical retrieval to obtain seed nodes and preliminary evidence.
2. **Graph exploration**: Directed traversal from seed nodes to acquire event-level narratives and temporally anchored evidence.
3. **Flow control**: Output the final answer once sufficient evidence has been collected.

`find_entity_contexts` supports logical operations including temporal filtering, ordering, and aggregation, far surpassing simple similarity matching.

### Loss & Training

REMem does not involve model training. The indexing stage uses an LLM for gist and fact extraction; the reasoning stage employs an LLM agent for tool invocation and inference.

## Key Experimental Results

### Main Results — Episodic Recall

| Method | LoCoMo LLM-J | REALTALK LLM-J |
|---|---|---|
| NV-Embed-v2 (RAG) | 73.0 | 59.5 |
| Mem0 | 49.7 | 14.3 |
| Graphiti | 52.5 | 35.3 |
| HippoRAG 2 | 74.0 | 55.8 |
| **REMem-S** | **77.5** | **65.3** |
| **REMem-I** | 76.2 | 63.7 |

REMem-S achieves an LLM-J score of 77.5% on LoCoMo (+3.5 vs. HippoRAG 2) and 65.3% on REALTALK (+9.5 vs. HippoRAG 2).

### Main Results — Episodic Reasoning

| Method | Complex-TR LLM-J | Test of Time EM |
|---|---|---|
| NV-Embed-v2 (RAG) | 80.4 | 68.9 |
| NV-Embed-v2 + TISER | 88.3 | 68.9 |
| HippoRAG 2 | 81.5 | 66.9 |
| **REMem-I** | **89.6** | **93.1** |
| REMem-I + TISER | 92.0 | 90.6 |

REMem-I achieves an EM of **93.1%** on Test of Time, making it the only method to surpass 90%. Compared to Full-Context (79.7%), this represents a gain of **+13.4 pp**.

### Ablation Study

| Variant | LoCoMo LLM-J | Complex-TR LLM-J |
|---|---|---|
| REMem-I (full) | 76.2 | 89.6 |
| w/o Gists | 48.9 | 80.9 |
| w/o Facts | 74.1 | 87.2 |
| w/o synonymy edges | 76.4 | 89.2 |
| w/o semantic_retrieve | 72.8 | 88.1 |
| w/o lexical_retrieve | 76.8 | 87.5 |

- **Removing gists has the largest impact**: LLM-J on LoCoMo drops from 76.2 to 48.9 (−27.3), confirming that gists are the central carrier of episodic memory.
- **Facts are more critical for reasoning**: Removing facts causes a −2.4 drop on Complex-TR.
- **The two retrieval tools are complementary**: Semantic retrieval supports conceptual association, while lexical retrieval improves surface-form coverage.

### Key Findings

1. **Abstention behavior**: REMem achieves F1 = 64.0% (Precision 73.3%) on unanswerable questions, substantially outperforming Graphiti (F1 53.1%) and Mem0 (F1 13.5%), exhibiting more accurate and balanced abstention.
2. **Token efficiency**: REMem-I consumes an average of 9K input tokens per query (REMem-S: 0.9K), compared to 26K for Full-Context.
3. **Human evaluation**: LLM judges agree with human ratings 93% of the time, validating the reliability of the LLM-as-judge evaluation scheme.
4. **Error analysis**: The primary error categories are selection/localization errors (46%), temporal/numerical reasoning errors (19%), and abstentions despite available evidence (18%).

## Highlights & Insights

1. **Cognitive science–driven design**: The framework is grounded in gist-based memory theory and situation model theory, engineering psychological concepts into practical components.
2. **Flexibility of the hybrid graph structure**: The unified representation combining concept-level (fact triples) and context-level (gist) information supports both fine-grained and holistic understanding.
3. **Only method exceeding 90% EM**: REMem-I is the sole method surpassing 90% on the Test of Time benchmark, demonstrating strong temporal reasoning capability.
4. **Divergence between iterative reasoning and single-step retrieval**: REMem-I substantially outperforms REMem-S on reasoning tasks (EM 93.1 vs. 72.5), while the gap is smaller on recall tasks.
5. **Carefully designed tool interfaces**: `find_entity_contexts` supports temporal filtering, ordering, offset, and aggregation operations, serving as a critical enabler of reasoning capability.

## Limitations & Future Work

1. The indexing stage relies on LLM-based extraction; the quality of gists and facts is therefore constrained by LLM capability.
2. The offline batch indexing paradigm makes streaming memory construction an engineering challenge.
3. Multi-step tool invocation in agentic reasoning increases inference latency and cost.
4. Experiments primarily use GPT-4.1-mini; generalizability to other models has not been thoroughly validated.
5. Ablations show that synonymy edges have limited impact (LLM-J change of only −0.2 to −0.4), warranting re-examination of their cost-benefit ratio.

## Related Work & Insights

- **Relationship to HippoRAG**: HippoRAG constructs knowledge graphs inspired by the hippocampus for associative retrieval but lacks temporal and event dimensions; REMem explicitly models temporal timelines and contextual dimensions.
- **Relationship to Mem0**: Mem0's aggressive filtering results in sparse memory; REMem retains comprehensive gist and fact records.
- **Relationship to TISER**: TISER is a pure prompting method that guides temporal reasoning and can be used in complementary combination with REMem (+TISER raises Complex-TR LLM-J from 89.6 to 92.0).
- **Implications for agent systems**: Episodic memory is foundational for agent personalization and continual learning; REMem provides a practical engineering solution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The hybrid memory graph design is novel and the tool-augmented reasoning approach is conceptually coherent.
- **Practicality**: ⭐⭐⭐⭐⭐ — Directly applicable to long-term memory augmentation in conversational agents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four benchmarks, comprehensive comparisons, ablation analysis, human evaluation, and error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear and tables are information-dense, though some descriptions are overly verbose.
- **Overall**: ⭐⭐⭐⭐ — Establishes a strong baseline in the episodic memory domain; engineering contributions outweigh theoretical ones.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](../../ACL2026/llm_agent/lightweight_llm_agent_memory_with_small_language_models.md)
- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](../../CVPR2026/llm_agent/worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICLR 2026\] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution](the_tool_decathlon_benchmarking_language_agents_for_diverse_realistic_and_long-h.md)
- [\[ICLR 2026\] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)

<!-- RELATED:END -->
