---
title: >-
  [Paper Note] Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization
description: >-
  [ICLR 2026][Graph Learning][Personalized Embodied Intelligence] This paper systematically evaluates the memory utilization capabilities of LLM-driven embodied agents through the Memento framework. It finds that existing…
tags:
  - "ICLR 2026"
  - "Graph Learning"
  - "Personalized Embodied Intelligence"
  - "Memory Utilization"
  - "Episodic Memory"
  - "Knowledge Graph"
  - "LLM Agent"
date: 2026-05-08
content_hash: 54b7576ae6219fd7
---

# Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization

**Conference**: ICLR 2026
**arXiv**: [2505.16348](https://arxiv.org/abs/2505.16348)  
**Code**: [https://github.com/Connoriginal/MEMENTO](https://github.com/Connoriginal/MEMENTO)  
**Area**: Graph Learning
**Keywords**: Personalized Embodied Intelligence, Memory Utilization, Episodic Memory, Knowledge Graph, LLM Agent

## TL;DR
This paper systematically evaluates the memory utilization capabilities of LLM-driven embodied agents through the Memento framework. It finds that existing agents can recall simple object semantics but fail to process sequential information in user behavior patterns. A hierarchical knowledge graph-based user profile memory module is proposed to effectively improve performance on personalized assistance tasks.

## Background & Motivation

**Background**: LLM-driven embodied agents have achieved notable progress on traditional object rearrangement tasks; however, these tasks typically involve only single-turn interactions and static instructions, requiring no understanding of personalized user preferences or historical behaviors.

**Limitations of Prior Work**: Existing memory systems for embodied agents primarily focus on semantic memory (scene graphs, semantic maps) and procedural memory (skill libraries), while episodic memory is used merely as a passive task buffer or contextual history, lacking systematic evaluation of personalized knowledge extraction and utilization.

**Key Challenge**: Personalized user knowledge (e.g., "favorite mug," "morning routine") requires agents to extract information from past interactions and apply it flexibly to new tasks. However, agents face two critical bottlenecks: information overload (performance degrades as retrieved memories increase) and coordination failure (inability to simultaneously leverage multiple memory entries).

**Goal**: 1) Systematically evaluate the memory utilization capabilities of embodied agents in personalized assistance tasks; 2) Diagnose critical bottlenecks in memory utilization; 3) Design improved memory architectures to support personalized tasks.

**Key Insight**: The approach addresses two dimensions of memory utilization—object semantics (recognizing objects with personal significance) and user patterns (recalling sequential information within behavioral routines)—by constructing an end-to-end evaluation framework.

**Core Idea**: By decoupling personalized knowledge management, a hierarchical knowledge graph-based user profile memory module is constructed to independently manage object semantic and user pattern information, thereby overcoming information overload and coordination failure in LLM episodic memory.

## Method

### Overall Architecture
Memento is a two-phase evaluation framework. In Phase 1 (memory acquisition), agents accumulate episodic memories through multi-turn interactions with users and establish performance baselines. In Phase 2 (memory utilization), agents must apply accumulated personalized knowledge to complete new assistance tasks. Tasks are divided into single-memory tasks (requiring one piece of personalized knowledge) and joint-memory tasks (requiring simultaneous coordination of multiple memory entries).

### Key Designs

1. **Memento Evaluation Framework**:

    - Function: Constructs an end-to-end benchmark for evaluating personalized embodied agents.
    - Mechanism: Personalized knowledge is categorized into object semantics and user patterns. Object semantics refers to personal meanings users assign to physical objects (e.g., "the red cup in the coffee set"); user patterns refer to sequential information in behavioral routines (e.g., "breakfast routine"). Evaluation metrics include Percent Complete ($PC$) for task completion ratio and Success Rate ($SR$) for task success.
    - Design Motivation: Existing evaluations focus solely on single-turn static instructions and fail to reflect the genuine challenges of personalized assistance.

2. **Memory Bottleneck Diagnostic Experiments**:

    - Function: Identifies critical obstacles in memory utilization through controlled variable experiments.
    - Mechanism: Information overload effects are assessed by varying top-$k$ retrieval counts ($k=3,5,7,10$) in single-memory tasks; coordination capability is evaluated by requiring agents to simultaneously use two memory entries in joint-memory tasks. Memory format simplification experiments (summarized vs. instruction-only) are also conducted.
    - Design Motivation: Understanding the conditions under which agent memory utilization fails is necessary before targeted improvements can be designed.

3. **Hierarchical Knowledge Graph User Profile Memory**:

    - Function: Independently manages personalized knowledge, providing agents with clearer and more structured information.
    - Mechanism: A three-level hierarchy is constructed—user layer → knowledge type layer (object semantics, user patterns) → specific element layer (objects, patterns, locations)—using hierarchical edges for structural relationships and temporal edges for sequential ordering within user patterns. This module coexists with episodic memory rather than replacing it.
    - Design Motivation: Episodic memory simultaneously provides personalized knowledge and in-context learning benefits (simplification degrades smaller model performance); thus, an additional module dedicated to managing personalized knowledge is required.

## Key Experimental Results

### Main Results

| Model | Phase | Task Type | PC (%) | SR (%) | ΔSR |
|------|------|---------|--------|--------|-----|
| GPT-4o | Acquisition | - | 96.3 | 95.0 | - |
| GPT-4o | Utilization | Single-Memory | 88.0 | 85.1 | -9.9 |
| GPT-4o | Utilization | Joint-Memory | 86.7 | 63.9 | -30.5 |
| Qwen-2.5-72b | Acquisition | - | 93.5 | 91.0 | - |
| Qwen-2.5-72b | Utilization | Single-Memory | 72.6 | 67.2 | -23.8 |
| Qwen-2.5-72b | Utilization | Joint-Memory | 68.9 | 36.1 | -58.3 |
| Llama-3.1-8b | Acquisition | - | 78.1 | 68.5 | - |
| Llama-3.1-8b | Utilization | Single-Memory | 48.1 | 35.0 | -33.5 |

### Ablation Study

| Model | Memory Format | PC (%) | SR (%) |
|------|---------|--------|--------|
| GPT-4o | Full Episodic Memory | 90.0 | 83.3 |
| GPT-4o | Summarized | 88.0 | 83.3 |
| GPT-4o | Instruction-Only | 62.4 | 50.0 |
| Llama-3.1-8b | Full Episodic Memory | 72.8 | 63.3 |
| Llama-3.1-8b | Summarized | 49.4 | 43.3 |
| Llama-3.1-8b | Instruction-Only | 40.0 | 30.0 |

### Key Findings
- All models exhibit SR drops exceeding 20% on personalized tasks; GPT-4o shows a 30.5% SR decrease on joint-memory tasks.
- Agents can effectively recall object semantics but struggle severely with sequential understanding of user patterns.
- Increasing the number of retrieved memories (larger top-$k$) consistently degrades performance across all models, indicating that information overload is a critical bottleneck.
- Memory summarization has limited impact on large models but substantially degrades smaller model performance, demonstrating that episodic memory simultaneously provides in-context learning benefits.
- The user profile memory module yields significant performance improvements on both single-memory and joint-memory tasks.

## Highlights & Insights
- **Systematic Diagnosis of Memory Utilization Bottlenecks**: Controlled variable experiments clearly reveal two core bottlenecks—information overload and coordination failure—establishing foundational understanding of personalization capabilities in embodied agents.
- **Discovery of the Dual Role of Episodic Memory**: Episodic memory is shown to serve not only as a source of personalized knowledge but also as a demonstration for in-context learning, explaining why naive memory summarization strategies are detrimental for smaller models.

## Limitations & Future Work
- The evaluation employs gold perception and motor skills, bypassing challenges at the perception and execution layers.
- Personalized knowledge is synthetically generated by LLMs and may not fully reflect the complex knowledge structures of real users.
- Knowledge graph construction for user profile memory relies on LLM extraction, which may introduce noise in production environments.
- Long-term adaptation scenarios involving memory evolution and updates over time remain unexplored.

## Related Work & Insights
- **vs. ProgPrompt/VOYAGER**: These methods focus on procedural memory (skill libraries) to improve task completion efficiency, whereas this paper examines the role of episodic memory in personalization; the two represent complementary memory dimensions.
- **vs. Xu et al. (2024)**: Their work infers user preferences from limited demonstrations, whereas this paper requires agents to extract structured personalized knowledge from explicitly provided interaction histories, placing greater emphasis on systematic memory management.

## Rating
- Novelty: ⭐⭐⭐⭐ First framework to systematically evaluate memory utilization in embodied agents, with a clearly defined problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic ablations across multiple models and memory conditions yield insightful findings.
- Writing Quality: ⭐⭐⭐⭐ Three research questions are developed progressively with clear logical structure.
- Value: ⭐⭐⭐⭐ Provides important reference value for the direction of personalization in embodied agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](../../NeurIPS2025/graph_learning/esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](../../AAAI2026/graph_learning/echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)
- [\[ICML 2026\] Unsat Core Prediction through Polarity-Aware Representation Learning over Clause-Literal Hypergraphs](../../ICML2026/graph_learning/unsat_core_prediction_through_polarity-aware_representation_learning_over_clause.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](../../CVPR2026/graph_learning/graph2eval_automatic_multimodal_task_generation_for_agents_via_knowledge_graphs.md)
- [\[AAAI 2026\] Commonality in Few: Few-Shot Multimodal Anomaly Detection via Hypergraph-Enhanced Memory](../../AAAI2026/graph_learning/commonality_in_few_few-shot_multimodal_anomaly_detection_via_hypergraph-enhanced.md)

</div>

<!-- RELATED:END -->
