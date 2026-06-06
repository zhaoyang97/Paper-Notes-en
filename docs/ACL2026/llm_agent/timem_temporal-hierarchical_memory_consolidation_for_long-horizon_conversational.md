---
title: >-
  [Paper Note] TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents
description: >-
  [ACL2026][LLM Agent][Temporal-Hierarchical Memory] TiMem organizes long-horizon conversational memory into a five-layer Temporal Memory Tree with explicit temporal containment. By using complexity-aware retrieval to dyna…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "Temporal-Hierarchical Memory"
  - "Long-Horizon Conversational Agents"
  - "Memory Consolidation"
  - "Adaptive Retrieval"
  - "Personalization"
date: 2026-05-08
content_hash: 10ee069695877713
---

# TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents

**Conference**: ACL2026  
**arXiv**: [2601.02845](https://arxiv.org/abs/2601.02845)  
**Code**: https://github.com/TiMEM-AI/timem  
**Area**: LLM Efficiency / Long-term Memory / Conversational Agents  
**Keywords**: Temporal-Hierarchical Memory, Long-Horizon Conversational Agents, Memory Consolidation, Adaptive Retrieval, Personalization

## TL;DR
TiMem organizes long-horizon conversational memory into a five-layer Temporal Memory Tree with explicit temporal containment. By using complexity-aware retrieval to dynamically balance fine-grained facts and high-level personas, it improves accuracy on LoCoMo and LongMemEval-S while significantly reducing recalled context length.

## Background & Motivation
**Background**: Long-horizon conversational agents must maintain consistent persona understanding and factual memory across multiple turns, days, or even weeks. Current approaches generally follow two paths: expanding model context/compressing KV/hidden states, or treats external memory as persistent storage managed via embedding retrieval, clustering, graph structures, or OS-style memory tiers.

**Limitations of Prior Work**: Existing methods often treat "semantic similarity" as the primary organizational principle, with time serving only as a secondary field. Consequently, similar segments from distant time points may be conflated, leading to a lack of clear evidentiary chains between short-term events and long-term preferences. Furthermore, if high-level summaries replace original segments too early, factual details necessary for answering specific questions are lost.

**Key Challenge**: Long-term memory requires compression to prevent indefinite growth of context length and retrieval costs, yet it also needs to preserve temporal continuity to maintain the causality of personality, preferences, and events. The problem is not just "how much to recall," but "how to organize temporally adjacent and hierarchically related memories into a retrievable structure."

**Goal**: The authors aim to build a plug-and-play memory framework that requires no additional fine-tuning and can be integrated with various LLM backends. It should distill representations like session, day, week, and profile from raw dialogues and select the appropriate level based on query complexity.

**Key Insight**: Borrowing the concept of memory consolidation from cognitive science, the paper proposes gradually transforming short-term episodic memory into stable semantic/persona memory. The key observation is that dialogue history is naturally chronological; thus, the memory hierarchy should first satisfy temporal containment before performing semantic integration at each level.

**Core Idea**: A Temporal Memory Tree is used to partition long dialogues into a structure that is "strictly nested in time and abstractly layered in semantics." Query complexity then determines which levels to recall, balancing accuracy, personalization, and context costs.

## Method
The design of TiMem consists of two phases: writing the dialogue stream into a temporal hierarchy tree (offline or online) and retrieving nodes from this tree that match the query complexity. It does not train new models but utilizes LLM prompts for memory consolidation, complexity judgment, and recall gating.

### Overall Architecture
The input is a continuously growing user-assistant dialogue stream. TiMem first converts each round or short segment into base-level factual segments, which are then incrementally merged into five levels: session, day, week, and profile. Each node contains a time interval and textual memory. A parent node's time interval must cover those of its children, ensuring that any low-level evidence can be traced upward to its corresponding long-term pattern or persona representation.

The retrieval phase follows three steps: first, a recall planner classifies the query as simple, hybrid, or complex and extracts keywords; second, candidate leaves are selected from L1 memory using a hybrid of semantic similarity and BM25, and ancestor nodes of the selected levels are recalled along the tree; finally, an LLM gating mechanism evaluates which candidates truly assist in answering the query, filtering redundant or conflicting memories before sorting them by hierarchy and temporal distance for the response model.

### Key Designs
1.  **Temporal Memory Tree as Memory Skeleton**:
    - **Function**: It organizes raw dialogues, session summaries, daily patterns, weekly patterns, and long-term personas within the same temporal tree to ensure high-level memories are abstractions of continuous time rather than arbitrary clusters.
    - **Mechanism**: Each memory node $m$ stores a time interval $tau(m)$ and semantic text $sigma(m)$. Parent-child edges require the parent's time to cover the child's. Higher levels contain fewer nodes and more abstract semantics (L1: segments, L2: sessions, L3: days, L4: weeks, L5: profiles).
    - **Design Motivation**: Long-term personalization often depends on how specific events gradually form preferences. Explicit temporal containment links factual evidence with long-term profiles, avoiding temporal misalignment caused by pure semantic clustering.

2.  **Instruction-Driven Hierarchical Memory Consolidation**:
    - **Function**: It gradually compresses low-level memories into non-redundant high-level representations while retaining recent history within the same level to minimize cross-window fragmentation.
    - **Mechanism**: The level-$i$ consolidator receives a set of child nodes, the $w_i=3$ most recent historical nodes from the same level, and level-specific instructions to output new level-$i$ memory. L1 is written instantly, while L2-L5 are triggered periodically at the end of session/day/week windows.
    - **Design Motivation**: Keeping only raw dialogue inflates retrieval costs, while keeping only summaries loses detail. Hierarchical consolidation allows short-term facts and long-term patterns to coexist without requiring dataset-specific fine-tuning.

3.  **Complexity-Aware Hierarchical Recall and Gating**:
    - **Function**: It automatically determines the recall range based on the query type, retrieving less for simple questions and more context for complex ones.
    - **Mechanism**: The planner maps queries to simple, hybrid, or complex. Simple queries search factual L1-L2 and profile L5; hybrid adds pattern layers; complex retrieves the full L1-L5. Base candidates use mixed semantic and BM25 scoring: $lambda s_{sem} + (1-lambda)s_{lex}$ (with $lambda=0.9$). Finally, LLM gating retains only query-relevant memories.
    - **Design Motivation**: Different questions require different memory granularities. Fixed recall ranges either under-recall or flood the context with irrelevant summaries. The planner + gating separation controls both accuracy and token costs.

### Loss & Training
TiMem does not involve supervised training or parameter updates. In all experiments, memory writing, planning, gating, and response generation use a unified LLM configuration. Qwen3-Embedding-0.6B is used for embeddings, while gpt-4o-mini-2024-07-18 handles response and memory operations. Key hyperparameters include a base recall budget $k=20$, a historical window $w_i=3$, and a hybrid coefficient $lambda=0.9$.

## Key Experimental Results

### Main Results
The paper evaluates TiMem on LoCoMo and LongMemEval-S benchmarks, comparing it against MemoryBank, A-MEM, Mem0, MemoryOS, and MemOS. Metrics primarily use LLM-as-a-Judge (LLJ) accuracy, along with F1/ROUGE-L and efficiency metrics for LoCoMo.

| Dataset | Metric | TiMem | Strongest Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| LoCoMo | Overall LLJ | 75.30 | MemOS 69.24 | +6.06 |
| LoCoMo | F1 / ROUGE-L | 54.40 / 54.68 | MemoryOS 45.36 / MemOS 47.41 | +9.04 F1 / +7.27 RL |
| LongMemEval-S, gpt-4o-mini | Overall LLJ | 76.88 | MemOS 68.68 | +8.20 |
| LongMemEval-S, gpt-4o | Overall LLJ | 78.96 | MemOS 73.07 | +5.89 |

On the four categories of LoCoMo, TiMem achieved 81.43 in Single-Hop, 77.63 in Temporal, 52.08 in Open-Domain, and 62.20 in Multi-Hop, outperforming all baselines. In LongMemEval-S, TiMem maintains leadership in Knowledge Update, Multi-Session, and Temporal Reasoning, demonstrating that the temporal tree supports both factual QA and cross-session reasoning.

### Ablation Study
The planner, gating, and hierarchical structure are the primary objects of ablation. "Mem Len" represents the average number of tokens recalled into the response context.

| Configuration | LoCoMo LLJ | LoCoMo Mem Len | LME-S LLJ | LME-S Mem Len | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Fixed Simple, No gating | 73.51 | 3710.30 | 73.20 | 3371.53 | Narrow range but high redundancy |
| Fixed Hybrid, With gating | 73.38 | 691.59 | 75.00 | 1673.93 | Most stable fixed strategy |
| Planner, No gating | 72.99 | 4411.09 | 73.80 | 3941.98 | Dynamic levels but high noise |
| Planner + Gating | 75.30 | 511.25 | 76.88 | 1270.62 | Full TiMem, best balance |

Further ablation on the hierarchy shows that neither base facts nor high-level summaries alone are sufficient.

| Hierarchy & Recall | LoCoMo LLJ | LoCoMo Mem Len | LME-S LLJ | LME-S Mem Len | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| L1 only, flat recall | 70.06 | 995.15 | 57.40 | 1823.98 | Details exist but lack structure |
| L1 only, hierarchical recall | 73.18 | 361.23 | 72.40 | 437.42 | Propagation restores some dependencies |
| L2-L5 only, hierarchical recall | 57.08 | 3786.44 | 64.20 | 2344.92 | Summaries alone lose factual evidence |
| L1-L5, flat recall | 70.71 | 1715.65 | 55.40 | 4519.26 | Unstable without tree-aligned recall |
| L1-L5, hierarchical recall | 75.30 | 511.25 | 76.88 | 1270.62 | Complementary detail and abstraction |

### Key Findings
*   The full TiMem recalls only 511.25 tokens on LoCoMo, which is 52.20% less than Mem0's 1070.10 tokens, despite higher accuracy. This indicates structured compression and filtering rather than context stuffing.
*   Consolidation calls for the five-layer TMT are 25%-30% higher than L1-only, but this is an amortized cost during the writing phase; the online inference context is significantly shorter.
*   Accuracy degrades as L1 segment granularity coarsens: from 75.30% at 1 turn to 65.26% at 8 turns, highlighting the importance of atomic evidence.
*   The hybrid retrieval is relatively insensitive to $lambda$, with LoCoMo accuracy maintained between 73.96%-75.30% for $lambda \in [0.7, 1.0]$, peaking at $lambda=0.9$.

## Highlights & Insights
*   The primary highlight is elevating "temporal continuity" from metadata to a structural constraint. While many systems record timestamps, retrieval and aggregation are typically dominated by similarity; TiMem ensures high-level personas have traceable temporal evidence.
*   The combination of planner and gating is practical: the planner decides which layers to inspect, and gating decides which nodes enter the context. This separation is more interpretable and allows for easier tuning of latency and costs.
*   Manifold analysis provides a diagnostic perspective: on LoCoMo, high-level memory separates 10 user clusters more clearly; on LongMemEval-S, hierarchical consolidation reduces embedding dispersion by ~50%, suggesting that "abstraction" helps distinguish personas or suppress noise depending on data distribution.
*   The framework is transferable to other long-term modeling tasks, such as educational tutors, medical follow-ups, and personal assistants, provided events accumulate chronologically.

## Limitations & Future Work
*   TiMem relies on LLMs for consolidation, planning, and gating. While no fine-tuning is needed, system stability depends on the internal LLM's quality; a planner misjudgment can lead to over- or under-recall.
*   The five-layer window is an empirical configuration. Different applications have varying time scales (e.g., customer service by tickets vs. education by course units), so the segment/session/day/week/profile setup may not be universal.
*   Experiments focused on QA-style memory evaluation and have not fully addressed continuous maintenance issues like changing user preferences, privacy deletion, or conflicting memory overwrites in real multi-turn interactions.
*   Future work could combine TMT with learnable routing or lightweight RL to automatically adjust windows and budgets based on feedback, or incorporate explicit conflict resolution.

## Related Work & Insights
*   **vs Mem0 / A-MEM**: These focus on semantic organization and agentic updates. TiMem differs by using a temporal tree to constrain all consolidation paths, providing clearer chains of evidence at the cost of predefined windows.
*   **vs RAPTOR / MemTree**: RAPTOR-like methods use tree-based summaries based on semantic clustering; TiMem emphasizes temporal containment, making it more suitable for scenarios where user states evolve over time.
*   **vs MemoryOS / MemOS**: While OS-style systems focus on memory tiers and system management, TiMem serves as a data structure and retrieval protocol specifically for long-term personalization. The two can be complementary.
*   **Insight**: Memory for long-horizon agents should be treated as a "structural formation at write-time" problem rather than just a retrieval problem. High-quality memory layout may be more decisive for performance than a stronger reranker.

## Rating
*   **Novelty**: ⭐⭐⭐⭐☆ Systematic use of temporal containment and hierarchical consolidation for long-term conversation, although components are prompt-driven.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Comprehensive main experiments and ablations, though real-time long-term conflict handling could be expanded.
*   **Writing Quality**: ⭐⭐⭐⭐☆ Logical flow from motivation to ablation; strong tabular support.
*   **Value**: ⭐⭐⭐⭐⭐ Highly valuable for long-term personalized agents, particularly for systems requiring context cost control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents](recmem_recurrence-based_memory_consolidation_for_efficient_and_effective_long-ru.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ACL 2026\] SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning](solar-rl_semi-online_long-horizon_assignment_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
