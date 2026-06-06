---
title: >-
  [Paper Note] Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction
description: >-
  [AAAI 2026][Information Retrieval & RAG][Personalized Dialogue] This paper proposes H2Memory, a four-layer hierarchical heterogeneous memory structure (Log Graphs / Background Memory / Topic Outlines / Principles)…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "Personalized Dialogue"
  - "Long-term Memory"
  - "Hierarchical Memory"
  - "User Modeling"
  - "Dialogue Assistant"
date: 2026-05-08
content_hash: ae76ed5e6d16a7c4
---

# Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction

**Conference**: AAAI 2026
**arXiv**: [2511.13410](https://arxiv.org/abs/2511.13410)  
**Code**: [GitHub](https://github.com/hzp3517/Mem-PAL)  
**Area**: Information Retrieval
**Keywords**: Personalized Dialogue, Long-term Memory, Hierarchical Memory, User Modeling, Dialogue Assistant

## TL;DR
This paper proposes H2Memory, a four-layer hierarchical heterogeneous memory structure (Log Graphs / Background Memory / Topic Outlines / Principles), validated on the PAL-Set dataset (100 users × 8.4 months of interaction), improving BLEU-1 on demand paraphrasing and solution recommendation tasks from 13.59 to 26.67.

## Background & Motivation
**Background**: Personalized dialogue systems must retain users' long-term preferences and history. Existing methods (RecurSum, MemoryBank, etc.) primarily rely on simple summarization-based memory.

**Limitations of Prior Work**: (a) User preferences emerge from fragmented logs rather than explicit statements; (b) a single memory structure cannot simultaneously handle information at different granularities; (c) long-term personalized dialogue datasets are lacking.

**Key Challenge**: A semantic gap exists between users' daily behavioral logs (fragmented, implicit) and dialogue-level demands (structured, explicit).

**Goal**: Construct multi-granularity memory from users' long-term logs and conversation history to support personalized assistants in demand understanding and solution recommendation.

**Key Insight**: Four memory layers progressively aggregate information from raw logs to abstract principles, retrieving user information at different granularities from each layer.

**Core Idea**: A hierarchical heterogeneous memory structure (Log Graphs → Background → Topics → Principles) progressively abstracts fragmented user behavior into retrievable personalized knowledge.

## Method

### Overall Architecture
Input: user log stream (behavioral records) + multi-turn conversation history. Output: personalized demand understanding / solution recommendation. H2Memory consists of four layers: (1) Log Graphs → (2) Background Memory → (3) Topic Outlines → (4) Principles.

### Key Designs

1. **Log Graphs**:

    - **Function**: Construct situational descriptions from fragmented logs.
    - **Mechanism**: Related log fragments are connected via relational edges (two types: *Caused_by* and *Follows*) to form structured situational graphs; a situation description is then generated for each connected subgraph.
    - **Design Motivation**: Raw logs are too fragmented for direct retrieval and must be aggregated into meaningful situational units.

2. **Background Memory**:

    - **Function**: Maintain long-term summaries of fixed aspects (e.g., occupation, interests) across time.
    - **Mechanism**: Recursively updated — summaries of relevant aspects are revised upon each new log entry.
    - **Design Motivation**: Provides a persistent representation of the user profile.

3. **Topic Outlines**:

    - **Function**: Extract structured information on demands, solutions, and preferences from conversations.
    - **Mechanism**: Each dialogue turn yields triples (demand, solution, preference), organized by topic.
    - **Design Motivation**: Demands and preferences expressed in conversation are central to personalization.

4. **Principles**:

    - **Function**: Abstract high-level demand types from topic outlines.
    - **Mechanism**: KMeans clustering is applied to categorize demands and generalize them into universal principles.
    - **Design Motivation**: Provides the highest-level personalized prior.

### Loss & Training
During retrieval, the top $k=3$ most similar entries (via paraphrase-multilingual-mpnet embeddings) are fetched from each of the four layers; the aggregated context is then fed into an LLM for response generation.

## Key Experimental Results

### Main Results

| Method | Demand Paraphrase BLEU-1 | Solution Selection Score | GPT-4 Score |
|--------|--------------------------|--------------------------|-------------|
| Vanilla (no logs) | 13.59 | 18.95 | 17.50 |
| RecurSum | — | — | Lower |
| MemoryBank | — | — | Lower |
| **H2Memory** | **26.67** | **38.32** | **32.54** |

### Ablation Study

| Configuration | Impact | Note |
|---------------|--------|------|
| w/o Topic Memory ($M_T$) | ~20% drop | Most critical for demand understanding |
| w/o Background Memory ($M_B$) | ~6% drop | Important for preference alignment |
| w/o Principles | Minor drop | Provides high-level prior |

### Key Findings
- H2Memory improves demand paraphrase BLEU-1 from 13.59 to 26.67 (a 96% relative gain).
- **Topic Memory contributes most to demand understanding** (~20% drop when removed); **Background Memory contributes to preference alignment** (~6% drop).
- PAL-Set achieves high quality in human evaluation: logs 2.75/3.0, dialogues 2.67/3.0.
- Different memory layers serve distinct tasks, validating the necessity of heterogeneous design.

## Highlights & Insights
- **Systematic four-layer heterogeneous memory design**: Progressive aggregation from raw logs to abstract principles mirrors the "episodic–semantic–procedural" hierarchy of human memory.
- **PAL-Set dataset**: 100 users × 29 sessions × 996 logs × 8.4-month span, filling a gap in long-term personalized dialogue benchmarks.
- **Ablation validates design rationale**: The independent contribution of each memory layer is clearly quantifiable.

## Limitations & Future Work
- **Retrieval strategy is relatively simple**: A fixed $k=3$ per layer is used, with no dynamic or adaptive retrieval.
- **Dataset is LLM-synthesized**: Although human-validated, it may not fully reflect real user behavior.
- **Memory forgetting/decay is not considered**: In long-term interactions, outdated information may need to be discarded.
- Future work could introduce proactive memory update strategies (e.g., handling conflicts between new logs and existing memory).
- Cross-dataset validation on EMG-RAG demonstrates that H2Memory generalizes beyond PAL-Set, confirming the broader applicability of the design.

## Related Work & Insights
- **vs. RecurSum**: RecurSum employs recursive summarization, whereas this work uses a four-layer heterogeneous structure providing richer memory representations.
- **vs. MemoryBank**: MemoryBank relies on a single memory structure; the hierarchical design here offers greater flexibility.
- **vs. Persona-based Dialogue**: Traditional persona methods use fixed descriptive sentences, whereas H2Memory employs dynamically updated multi-granularity memory, better suited for evolving user preferences in long-term interaction.
- **Insight**: Personalization systems require user knowledge at varying granularities; a single representation is insufficient.
- **Insight**: Automated management of heterogeneous memory (when to merge, when to forget) remains a key open problem in personalized dialogue.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic and well-motivated four-layer heterogeneous memory design
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes ablation studies and human evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured
- Value: ⭐⭐⭐⭐ Practical value for long-term personalized dialogue

## Additional Notes
- The methodology and experimental design of this work offer meaningful reference for related research areas.
- Future work may validate generalizability and scalability across more scenarios and at larger scale.
- Potential research value exists in combining this work with recent related approaches (e.g., intersections with RL/MCTS or multimodal methods).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HyperMem: Hypergraph Memory for Long-Term Conversations](../../ACL2026/information_retrieval/hypermem_hypergraph_memory_for_long-term_conversations.md)
- [\[ICLR 2026\] AMemGym: Interactive Memory Benchmarking for Assistants in Long-Horizon Conversations](../../ICLR2026/information_retrieval/amemgym_interactive_memory_benchmarking_for_assistants_in_long-horizon_conversat.md)
- [\[AAAI 2026\] ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)
- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)
- [\[AAAI 2026\] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](mavis_a_benchmark_for_multimodal_source_attribution_in_long-form_visual_question.md)

</div>

<!-- RELATED:END -->
