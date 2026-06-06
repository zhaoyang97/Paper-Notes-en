---
title: >-
  [Paper Note] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing
description: >-
  [ACL 2026][Multi-Agent][Multi-agent Orchestration] MASFactory models LLM multi-agent systems as Node/Edge computational graphs. It proposes a three-stage "Vibe Graphing" pipeline (Role Assignment → Structure Design → Sem…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Multi-agent Orchestration"
  - "Vibe Graphing"
  - "Computational Graph"
  - "Human-AI Collaboration"
  - "Context Adaptation"
date: 2026-05-08
content_hash: 3b2f3bcc73bf8cd2
---

# MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing

**Conference**: ACL 2026  
**arXiv**: [2603.06007](https://arxiv.org/abs/2603.06007)  
**Code**: https://github.com/BUPT-GAMMA/MASFactory (Available)  
**Area**: LLM Multi-agent Systems / Graph Orchestration / Engineering Framework  
**Keywords**: Multi-agent Orchestration, Vibe Graphing, Computational Graph, Human-AI Collaboration, Context Adaptation

## TL;DR
MASFactory models LLM multi-agent systems as Node/Edge computational graphs. It proposes a three-stage "Vibe Graphing" pipeline (Role Assignment → Structure Design → Semantic Completion) to compile natural language intents into executable MAS workflows. The framework provides Context/Message Adapters, ComposedGraph template reuse, and VS Code visualization. Across 7 benchmarks, it reproduces 5 representative MAS with comparable or superior performance. End-to-end Vibe Graphing reduces ChatDev's 1,511 lines of code to 45 lines, with API costs an order of magnitude lower than Vibe Coding.

## Background & Motivation

**Background**: LLM-based multi-agent systems (MAS) expand single-agent capabilities through role specialization, mutual verification, and iterative collaboration. Representative systems include AutoGen, MetaGPT, ChatDev, AgentVerse, and CAMEL. The mainstream orchestration abstraction is the directed computational graph—LangGraph models workflows as stateful graphs, while Dify provides a DAG canvas.

**Limitations of Prior Work**: ① Implementing a complete MAS entails extremely high engineering costs, requiring developers to manually write role prompts, link node routing, and define inter-agent communication protocols. ② Real-world applications must connect to heterogeneous context sources like memory (Mem0, MemGPT), RAG (LlamaIndex, GraphRAG), and MCP, yet existing frameworks rely on workflow-specific glue code, resulting in poor portability. ③ MAS contain numerous redundant subgraphs that are "globally similar but locally distinct," for which existing frameworks offer limited versioning or templated reuse. ④ Even with graph-based frameworks like LangGraph, writing complex MAS still requires thousands of lines of code (the original ChatDev is 1,511 lines of Python).

**Key Challenge**: User intentions are typically expressed in natural language (e.g., "I want a coding MAS where a PM breaks down requirements, a dev writes code, and a QA reviews"), but existing systems force developers to translate these intentions into a complex trio of graph wiring, prompts, and protocols, leading to high conversion costs and maintenance burdens.

**Goal**: To enable users to derive runnable, editable, and reusable MAS workflows directly from natural language intentions while ensuring performance comparable to manual implementations.

**Key Insight**: Taking inspiration from "Vibe Coding" but adopting a more structured approach—instead of directly generating code, the system generates a structured intermediate representation (graph skeleton + node configuration). This allows the LLM to focus on "graph design" while the framework handles "graph execution," with human-in-the-loop reviews inserted at each stage.

**Core Idea**: Reformulate MAS construction as a three-stage compilation: "intent → structured graph → executable workflow," supported by reusable ComposedGraph templates and pluggable Context/Message Adapters.

## Method

### Overall Architecture

The foundation is a Node/Edge computational graph skeleton: Nodes represent computational units (expandable to Graph, Loop, Agent, CustomNode, Interaction, or Switch), and Edges express dependencies and message pathways. Collaboration flows are explicitly categorized into three types: **Control flow** (advancing scheduling), **Message flow** (horizontal transmission of node outputs), and **State flow** (synchronizing shared context along parent-child graph hierarchies). The runtime utilizes readiness-based scheduling to allow multiple ready nodes to execute concurrently, with native support for serial, parallel, branching, and looping patterns. Agent nodes follow a Perception-Reasoning-Action loop and are decoupled via pluggable Message Adapters (JSON, Markdown, or free text) and Context Adapters (integrating Mem0, LlamaIndex, MCP, or RAG). The top layer provides three orchestration interfaces: (a) Vibe Graphing driven by natural language, (b) Imperative hand-written Python code, and (c) Declarative configuration files. A companion VS Code plugin, Visualizer, provides topology previews, runtime tracing, and human-in-the-loop interaction.

### Key Designs

1.  **Vibe Graphing Three-stage Compilation Pipeline**:

    - **Function**: Converts user natural language intentions into executable MAS workflows while allowing user review/modification at each stage.
    - **Mechanism**: Compilation occurs in three steps—(i) **Role Assignment**: The LLM maps task intent to a set of candidate agent roles with defined boundaries; (ii) **Structure Design**: Based on information dependencies and control constraints between roles, it generates a directed graph topology skeleton (determining connectivity and message/control propagation); (iii) **Semantic Completion**: Performs parameterized instantiation on the skeleton, configuring prompts and tools for each node to produce a ready-to-execute workflow. All three stages maintain a structured IR (readable and editable), allowing human-in-the-loop intervention via the Visualizer. The system uses gpt-5.2 for workflow construction and gpt-4o-mini for execution.
    - **Design Motivation**: Directly tasking LLMs with code generation (Vibe Coding) often results in logically flawed graphs and high API costs. Using a staged approach with structured intermediate representations separates "graph structural correctness" from free-form LLM generation, allowing the LLM to focus on semantics while the framework ensures executability.

2.  **Context Adapter / Message Adapter Dual-layer**:

    - **Function**: Decouples heterogeneous external dependencies (memory, RAG, MCP, communication protocols) from the collaboration graph.
    - **Mechanism**: The Context Adapter partitions various context sources (Mem0 long-term memory, LlamaIndex RAG, Anthropic MCP) into standardized units and exposes a unified interface to graph nodes. The Message Adapter formats agent IO according to specified protocols (JSON Schema, Markdown sections, or plain text) and allows custom user-defined protocols. Consequently, the same collaboration graph can seamlessly switch memory backends or communication protocols without altering its topology.
    - **Design Motivation**: Real-world MAS are highly dependent on external context sources, each with its own API/data format. Previously, glue code was required to stitch Mem0 or LlamaIndex into each agent, causing workflows to be tightly coupled to specific frameworks and reducing portability.

3.  **ComposedGraph + NodeTemplate Reuse Mechanism**:

    - **Function**: Enables recurring subgraph structures to be declared, reused, and version-managed as "templates."
    - **Mechanism**: NodeTemplate allows users to declare structural templates before instantiation, supporting the cloning of multiple graphs that are globally identical but locally parameterized. ComposedGraph is a specialized Graph with a predefined structure that can be instantiated by filling node configurations or activating specific branches. The framework includes common collaboration subgraphs (e.g., DyLan-style dynamic scheduling), and users can package their designs into reusable components.
    - **Design Motivation**: Collaboration patterns like "review-critique-revise" or "propose-vote-merge" recur frequently. Rewriting them every time is inefficient and inconsistent; templating reduces code volume and facilitates version management and team collaboration.

### Loss & Training
- The framework has no training objectives; all agents utilize LLM reasoning (defaulting to gpt-4o-mini). The Vibe Graphing construction stage uses gpt-5.2 for IR generation. Evaluation involves 5 reproduced MAS + 2 Vibe Graphing variants (a staged ChatDev version and a task-specific end-to-end version).

## Key Experimental Results

### Main Results (Percentage scale; "–" indicates the framework is not applicable to general reasoning benchmarks)

| Method | HumanEval | MBPP | BigCodeBench | SRDD | MMLU-Pro | GAIA | GPQA |
|---|---|---|---|---|---|---|---|
| ChatDev (orig) | 82.50 | 71.40 | 50.70 | 82.91 | – | – | – |
| ChatDev (Ours) | 81.30 | 74.20 | 53.30 | **84.23** | – | – | – |
| MetaGPT (orig) | 67.07 | 36.03 | 50.10 | 78.19 | – | – | – |
| MetaGPT (Ours) | **89.02** | **59.14** | 51.70 | 72.77 | – | – | – |
| AgentVerse (orig) | 85.00 | 74.54 | 65.92 | 87.55 | 64.64 | 12.12 | 38.39 |
| AgentVerse (Ours) | 85.00 | 75.15 | 64.12 | **91.06** | 64.16 | **12.73** | 37.50 |
| CAMEL (orig) | 62.20 | 60.60 | 63.51 | 89.42 | 50.08 | 9.70 | 32.59 |
| CAMEL (Ours) | **71.85** | 57.80 | **78.16** | 89.69 | **63.04** | **12.73** | 24.78 |
| HuggingGPT (orig) | 82.32 | 68.60 | 28.42 | 87.96 | 65.59 | 9.09 | 56.67 |
| HuggingGPT (Ours) | 80.49 | 64.40 | 29.91 | 83.26 | 63.66 | 10.91 | 47.32 |
| **Vibe Graphing-ChatDev** | 83.50 | 74.20 | 45.30 | 88.13 | – | – | – |
| **Vibe Graphing-Task Specific** | 84.76 | 72.37 | 51.67 | 90.71 | 51.73 | 12.12 | 39.51 |

Reproduction versions are broadly consistent or better (MetaGPT's +22 gain in HumanEval is due to ComposedGraph helping fix routing bugs in the original). Automatically generated Vibe Graphing workflows match or exceed manual originals in most tasks.

### Ablation Study (Code Volume + Cost)

| Implementation | Code Volume (lines) | Remarks |
|---|---|---|
| ChatDev (orig) | 1,511 | Hand-written |
| MASFactory Reproduction | 1,114 | ComposedGraph reuse |
| Vibe Graphing-ChatDev (Staged) | 203 | One VibeGraph component per stage |
| Vibe Graphing-Task Specific (End-to-End) | 45 | Compiled from a single VibeGraph intent |

| Workflow | Vibe Graphing Cost ($) | Vibe Coding low ($) | Vibe Coding medium ($) |
|---|---|---|---|
| ChatDev | **0.26** | 3.49 | 3.02 |
| AgentVerse | **0.59** | 4.43 | 6.08 |

Vibe Graphing is approximately 10× cheaper than Vibe Coding. Furthermore, Vibe Coding outputs frequently suffer from graph logic errors, rendering them non-executable (leading the authors to omit Vibe Coding performance comparisons and focus on cost).

### Key Findings
- Utilizing structured IR instead of direct code generation is central to making Vibe Graphing both affordable and stable—LLMs are only tasked with "graph structure + prompt design," which are more constrained tasks, while the framework ensures "executability."
- ComposedGraph helped the MetaGPT reproduction actually outperform the original (HumanEval +22, MBPP +23), suggesting that decoupling messy engineering implementations into templates allows the core methodology to shine without being obscured by noise.
- On general reasoning tasks like GAIA and GPQA, automatically generated Vibe Graphing workflows are comparable to or slightly better than manual AgentVerse (GPQA 39.51 vs 38.39), proving the paradigm generalizes beyond programming.
- The HuggingGPT reproduction was slightly lower (GPQA 47.32 vs 56.67), which the authors attribute to differences in tool-calling interface details—this highlights that Vibe Graphing requires more granular Context Adapters for MAS that are heavily dependent on external tool semantics.

## Highlights & Insights
- Decomposing MAS construction into "intent → structure → semantics" aligns with the natural design intuition of "deciding roles → mapping connections → writing prompts." This is significantly more controllable and cost-effective than the "one-shot magic" of Vibe Coding.
- The Context Adapter abstraction is highly engineering-oriented but offers immense reuse value. By normalizing Mem0, LlamaIndex, MCP, and RAG into standardized context units, graph topology is decoupled from the external ecosystem, allowing future context sources to be swapped or stacked without modifying the core logic.
- The separation of "Three flows" (control, message, and state) is a notable system design: explicitly splitting scheduling signals, data, and shared state allows complex control patterns like loops and branching to emerge naturally from readiness-based scheduling.
- Reducing 1,511 lines of ChatDev code into 45 lines of end-to-end Vibe Graphing intent is, in itself, a compelling advertisement for developer experience—the future of MAS frameworks likely lies in "coding less, and intending more."

## Limitations & Future Work
- Lack of support for checkpointing/recovery; failure in long workflows requires a full restart, which is a major issue for production deployment.
- The ComposedGraph component library is still being built and does not yet cover all collaboration patterns.
- Vibe Graphing construction relies heavily on gpt-5.2 (a top-tier model); the stability of using smaller models for IR generation has not been systematically evaluated.
- The experiment lacks system metrics such as latency and throughput, focusing primarily on quality and line counts; performance under high-concurrency readiness-based scheduling remains unknown.
- The performance gap in the HuggingGPT reproduction suggests that Context/Message Adapters are not yet fully comprehensive regarding tool protocols.
- There is no RL or learned routing policy; all control flow decisions are LLM-based or manually set, which may become a bottleneck in long-chain dynamic decision-making.

## Related Work & Insights
- **vs LangGraph / Dify**: While both use graphs for modeling, LangGraph and Dify still require manual coding for node logic. This paper adds Vibe Graphing (auto-generation), ComposedGraph (reuse), and Context Adapters, significantly simplifying the developer experience.
- **vs AutoGen / MetaGPT / ChatDev / CAMEL**: These are specific MAS methods, whereas this is a meta-framework. It can reproduce these methods with shorter code; the fact that the MetaGPT reproduction surpassed the original suggests the abstraction layer is well-designed.
- **vs CrewAI / Google ADK**: These are code-first orchestration frameworks. This paper utilizes a dual-driven "graph + natural language" approach, which is more friendly for visualization and human-in-the-loop interaction.
- **vs Vibe Coding (Direct code generation)**: This work demonstrates that structured IR and staged compilation are significantly superior to end-to-end code generation in terms of cost (10×) and success rate (Vibe Coding is often "broken"). This insight is valuable for all "LLM-generated software" research directions.

## Rating
- Novelty: ⭐⭐⭐ The primary contributions are the engineering framework and the "structured IR + three-stage compilation" Vibe Graphing pipeline. Innovation is moderate but well-integrated.
- Experimental Thoroughness: ⭐⭐⭐ Covers 7 benchmarks, 5 reproductions, 2 Vibe variants, plus code volume/cost comparisons. Broad coverage, though missing latency and failure mode analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear architecture diagrams and tables allow framework users to get started quickly.
- Value: ⭐⭐⭐⭐ For MAS developers, the 1,511→45 line improvement in DX is direct, tangible, and reusable. This is a framework that can be practically utilized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ACL 2026\] To Trust or Not to Trust: Attention-Based Trust Management for LLM Multi-Agent Systems](to_trust_or_not_to_trust_attention-based_trust_management_for_llm_multi-agent_sy.md)

</div>

<!-- RELATED:END -->
