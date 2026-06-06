---
title: >-
  [Paper Note] RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents
description: >-
  [ACL2026][LLM Agent][Long-term Agent] RecMem draws on the human memory principle that "consolidation only happens upon repetition." It initially places raw interactions into a lightweight subconscious memory and only inv…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "Long-term Agent"
  - "Memory Consolidation"
  - "Recurrence Triggering"
  - "Semantic Memory"
  - "Cost Efficiency"
date: 2026-05-08
content_hash: 9e976538f13bca3e
---

# RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents

**Conference**: ACL2026  
**arXiv**: [2605.16045](https://arxiv.org/abs/2605.16045)  
**Code**: https://github.com/CaiusDai/RecMem  
**Area**: LLM Agent / Long-term Memory / Memory System  
**Keywords**: Long-term Agent, Memory Consolidation, Recurrence Triggering, Semantic Memory, Cost Efficiency

## TL;DR
RecMem draws on the human memory principle that "consolidation only happens upon repetition." It initially places raw interactions into a lightweight subconscious memory and only invokes the LLM to generate episodic and semantic memory when semantic recurrence is detected. This allows it to achieve or exceed the QA accuracy of mainstream memory systems on LoCoMo and LongMemEval-S with significantly lower token construction costs.

## Background & Motivation
**Background**: Long-term LLM agents need to retain user facts, preferences, events, and task states across multiple rounds and sessions. Existing external memory systems typically process interaction content into summaries, facts, knowledge graphs, or memory nodes, which are then used to enhance answers through retrieval.

**Limitations of Prior Work**: While systems like Mem0, A-Mem, and MemoryOS differ in structure, most adopt eager memory consolidation, meaning they invoke the LLM to extract, summarize, or merge memories for every new interaction. The primary issue with this strategy is the high construction cost, and many one-time, noisy, or low-information interactions do not actually need to enter long-term memory immediately.

**Key Challenge**: Long-term agents must avoid losing information without paying LLM-level consolidation costs for every interaction. Premature consolidation wastes tokens and may over-structure temporary information; a total lack of consolidation would result in a lack of cross-temporal organization for subsequent retrieval.

**Goal**: Design a training-free, text-based external memory system that reduces LLM calls during the memory construction phase in streaming interactions while maintaining long-term QA accuracy.

**Key Insight**: The authors start from the multi-store theory and Complementary Learning Systems in cognitive science, suggesting that isolated experiences should first be retained in a fast-encoding layer, and only repeatedly activated patterns deserve consolidation into long-term memory.

**Core Idea**: Use a cheap embedding store to first receive all raw interactions, then use semantic similarity and recurrence counts to trigger LLM consolidation. This treats "when to remember" as a first-class problem rather than defaulting to every interaction being summarized by an LLM.

## Method
RecMem is a three-layer memory system: subconscious memory preserves raw interaction units and supports low-cost retrieval; episodic memory stores event narratives across multiple rounds; semantic memory stores fine-grained facts. Its key is not proposing a new retrieval model, but changing the timing of memory construction: the system only triggers higher-layer consolidation via LLM when new interactions find enough semantically similar historical interactions in the subconscious layer.

### Overall Architecture
When streaming interactions arrive, the system first takes a user-assistant exchange as an atomic unit $u_i=(m_i^{usr},m_i^{ast},\tau_i)$, encodes it using an embedding model as $v_i=\Phi(u_i)$, and writes it to the subconscious store. For each new unit, the system retrieves top-$k$ neighbors in the subconscious store and maintains a relevant set $\mathcal{R}_i$ with similarity no lower than $\theta_{sim}$. When $|\mathcal{R}_i|\geq \theta_{count}$, indicating a persistent recurrence of the topic, the system sends $\mathcal{R}_i\cup\{s_i\}$ to the episodic and semantic layers for consolidation; otherwise, the raw interaction is only kept in subconscious memory.

During querying, RecMem retrieves evidence simultaneously from the subconscious, episodic, and semantic layers. Default retrieval budgets are $k_{sub}=10$, $k_{epi}=5$, and $k_{sem}=10$, where the semantic budget is twice that of the episodic budget to allow fine-grained facts to supplement event summaries.

### Key Designs
1.  **Subconscious memory and recurrence triggering**:
    - **Function**: Preserves raw interactions in their entirety at minimum LLM cost and determines which content is worth entering long-term memory.
    - **Mechanism**: Each interaction unit undergoes only lightweight structuring and vectorization; when a new unit arrives, it retrieves similar history. Only if the number of neighbors exceeding similarity $\theta_{sim}$ reaches $\theta_{count}$ is LLM consolidation triggered. The paper recommends $\theta_{sim}=0.7, \theta_{count}=5$ for open chit-chat settings and $\theta_{sim}=0.6, \theta_{count}=4$ for long task-oriented interactions.
    - **Design Motivation**: Much information appears only once and is not worth the LLM tokens to summarize; recurring themes tend to be more stable and possess higher future query value.

2.  **Episodic memory with merge-first strategy**:
    - **Function**: Organizes the temporal evolution of the same topic into time-anchored event narratives.
    - **Mechanism**: New interactions first attempt to merge with the most recent episodic entry. If similarity is high enough, the existing episode is updated using LLM merge; otherwise, after a recurrence trigger, relevant interactions are sorted by timestamp for the LLM to generate a new episode. This prevents a single topic from being fragmented into multiple parallel summaries.
    - **Design Motivation**: Topics in long-term conversations often recur and change incrementally; merge-first maintains narrative coherence for each topic.

3.  **Semantic refinement**:
    - **Function**: Recovers fine-grained facts missed by episodic summaries during compression.
    - **Mechanism**: For each episode, relevant existing semantic facts are first retrieved. The LLM is then tasked with two things regarding raw interactions, episode summaries, and historical facts: recovering key entities and details omitted by summaries, and maintaining existing facts while handling preference changes. Each fact is ultimately stored as an independent semantic entry.
    - **Design Motivation**: Event-level summaries tend to generalize; precise questions may require a single fact hit. Semantic memory acts as detailed compensation for episodic abstraction.

### Loss & Training
RecMem is a training-free external memory system that does not require LLM fine-tuning. The system primarily relies on embedding retrieval, static thresholds, and LLM prompts to complete consolidation, merge, refinement, and answering. Experiments use GPT-4o-mini and GPT-4.1-mini as backends with temperature=0.0. The embedding model is text-embedding-3-small.

## Key Experimental Results

### Main Results
| Dataset / Model | Metric | RecMem (Ours) | Strongest Baseline System | Construction Cost Comparison |
|---------------|------|--------|------------------|--------------|
| LoCoMo / GPT-4.1-mini | Overall accuracy | 81.10 | A-Mem 68.83 / MemoryOS 67.60 | 193.2K construction tokens vs Mem0 1520.8K, A-Mem 1459.93K |
| LoCoMo / GPT-4o-mini | Overall accuracy | 72.47 | MemoryOS 63.64 / A-Mem 60.84 | 202.4K construction tokens vs Mem0 1233.5K, A-Mem 1143.3K |
| LongMemEval-S / GPT-4.1-mini | Overall accuracy | 76.80 | MemoryOS 74.40 / A-Mem 71.60 | 365.49K construction tokens vs Mem0 1626.54K, A-Mem 1264.25K |
| LongMemEval-S / GPT-4o-mini | Overall accuracy | 69.20 | MemoryOS 67.80 / Mem0 64.00 | 329.55K construction tokens vs Mem0 1244.87K, A-Mem 1180.23K |

### Ablation Study
| Configuration | LoCoMo GPT-4.1-mini Overall | Note |
|------|-----------------------------|------|
| Full RecMem | 81.10 | Complete three-layer memory |
| w/o subconscious memory | 51.88 | Removing the raw interaction layer causes the largest drop |
| w/o episodic memory | 79.94 | Removing event narratives has minor impact |
| w/o semantic memory | 70.58 | Missing fine-grained facts causes significant drop |
| Direct semantic extraction | 74.22 | Lower than 79.94 when extracting without episode refinement |

### Key Findings
- RecMem uses approximately 87.3% fewer construction tokens than Mem0 and 86.8% fewer than A-Mem on LoCoMo GPT-4.1-mini while achieving higher overall accuracy.
- On the longer LongMemEval-S, Full Context is no longer dominant; RecMem achieves the highest overall accuracy with lower construction costs.
- Temporal reasoning is a strength of RecMem because subconscious clustering aggregates coreferential topics across time, and episodic consolidation restores the evolutionary process through chronological sorting.
- Ablations show that subconscious memory is the system foundation; semantic memory is more critical to final accuracy than episodic memory, as many questions require precise facts rather than coarse-grained event summaries.

## Highlights & Insights
- The paper shifts the focus from "what to remember" to "when it is worth consolidating." This perspective is practical because the cost bottleneck for long-term agents often occurs during continuous writing rather than single queries.
- The value of the subconscious layer is not just cost savings, but also a high-fidelity backup. Even if a piece of information does not recur enough to be consolidated, it can still be retrieved directly during a query.
- Semantic refinement explains why simple summary memory is insufficient: as summaries merge, they become more abstract, losing the fine-grained evidence—such as user preferences, time, and entity relationships—needed for QA.

## Limitations & Future Work
- Recurrence triggering depends on static thresholds $\theta_{sim}$ and $\theta_{count}$; different domains and interaction densities may require re-tuning.
- Using recurrence as a salience proxy may miss information that appears only once but is highly important, such as one-time deadlines, medical reminders, or contract terms. Although subconscious memory retains the original text, it will not actively form high-level memory.
- The 10/5/10 retrieval budget and three-layer structure are effective on current benchmarks, but their optimality in multi-user, multi-modal, or tool execution logs remains to be verified.
- Future work could implement adaptive triggers: dynamically adjusting thresholds based on user, task type, and risk level rather than using fixed empirical values.

## Related Work & Insights
- **vs Mem0**: Mem0 tends to extract interactions into atomic facts and update them continuously; RecMem delays this step, performing fact extraction only after theme recurrence.
- **vs A-Mem**: A-Mem organizes interactions using Zettelkasten-like memory notes and connection relationships; RecMem emphasizes cost control and recurrence triggering during streaming writes.
- **vs MemoryOS**: MemoryOS simulates operating system-style management with hierarchical memory; RecMem's three-layer structure is simpler but achieves high cost-performance through the division of labor between subconscious, episodic, and semantic layers.
- **Insights**: In long-term agents, it is not necessary to immediately summarize all interactions into "permanent memory"; a cheap, searchable, and delay-consolidatable buffer layer can be established first.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The idea of recurrence-triggered consolidation is clear and effective, representing a paradigm shift rather than complex model innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Coverage includes two long-memory benchmarks, two LLM backends, multiple memory systems, and ablations; real-world online deployment analysis could be further strengthened.
- Writing Quality: ⭐⭐⭐⭐☆ Method motivation is smooth, the three-layer structure is clearly explained, and the cost metrics are very persuasive.
- Value: ⭐⭐⭐⭐⭐ Highly practical for the design of memory systems in long-term agents, especially in scenarios where token cost is a core evaluation metric.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](timem_temporal-hierarchical_memory_consolidation_for_long-horizon_conversational.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[ACL 2026\] PersonaAgent: Bridging Memory and Action for Personalized LLM Agents](personaagent_bridging_memory_and_action_for_personalized_llm_agents.md)
- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)
- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)

</div>

<!-- RELATED:END -->
