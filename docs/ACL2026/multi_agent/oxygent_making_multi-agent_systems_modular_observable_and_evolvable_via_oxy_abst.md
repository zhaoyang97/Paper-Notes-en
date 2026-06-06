---
title: >-
  [Paper Note] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction
description: >-
  [ACL2026][Multi-Agent][Multi-Agent Systems] OxyGent unifies agents, tools, LLMs, and reasoning processes into pluggable Oxy atomic components. By utilizing permission-driven dynamic planning and the OxyBank data reflux m…
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Multi-Agent Systems"
  - "Agent Framework"
  - "Observability"
  - "Dynamic Planning"
  - "Continuous Evolution"
date: 2026-05-08
content_hash: a7e8f7eaa29525bc
---

# OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction

**Conference**: ACL2026  
**arXiv**: [2604.25602](https://arxiv.org/abs/2604.25602)  
**Code**: https://github.com/jd-opensource/OxyGent  
**Area**: Agent / Multi-Agent Systems  
**Keywords**: Multi-Agent Systems, Agent Framework, Observability, Dynamic Planning, Continuous Evolution

## TL;DR
OxyGent unifies agents, tools, LLMs, and reasoning processes into pluggable Oxy atomic components. By utilizing permission-driven dynamic planning and the OxyBank data reflux mechanism, it makes industrial-grade multi-agent systems easier to build, monitor, and continuously evolve.

## Background & Motivation
**Background**: LLM agents are transitioning from single chat assistants to multi-agent systems (MAS). Typical applications include intelligent customer service, enterprise knowledge bases, office automation, file management, and complex information retrieval. Mainstream frameworks usually provide agent definitions, tool calling, message passing, flow orchestration, and tracing capabilities, allowing developers to assemble multiple specialized roles into a task-solving system.

**Limitations of Prior Work**: The paper points out that while many MAS frameworks are effective for research prototypes, they expose three problems in industrial deployment. First, the abstraction layers are inconsistent; agents, tools, LLMs, and flows are often different types of objects, leading to high costs for reuse and hot-swapping. Second, fixed DAGs or hard-coded plan-and-execute workflows struggle to handle branches, uncertainty, and failure recovery in real-world environments. Third, there is a lack of a closed loop from execution trajectories to data labeling, evaluation, knowledge base updates, and model optimization once the system is online, causing agent capabilities to remain static at the time of deployment.

**Key Challenge**: Multi-agent systems need to be flexible enough to allow for dynamic path composition based on tasks at runtime, while remaining controllable enough for developers to see why each node was called, where time was spent, and how failures propagated. If only free collaboration is pursued, the system becomes a black box; if only fixed processes are pursued, the agent's adaptive capacity is sacrificed.

**Goal**: OxyGent aims to solve the engineering foundation issues of production-grade MAS rather than proposing a new reasoning model. It decomposes its goals into three parts: reducing component composition costs with a unified abstraction, improving observability through runtime visualization and lifecycle hooks, and precipitating online trajectories into auditable and refluxable AI assets using OxyBank.

**Key Insight**: The authors observe that the complexity of MAS often arises not from the capabilities of individual agents, but from inconsistent interfaces between different entities, unclear permission boundaries, chaotic state sharing, and untraceable execution processes. Therefore, rather than continuing to stack new planners, it is better to first transform all "capability units" in the agent ecosystem into the same type of manageable atomic nodes.

**Core Idea**: Use a unified Oxy abstraction to transform agents, tools, LLMs, and flows into isomorphic nodes, generate execution graphs at runtime based on permission relationships, and return execution trajectories to OxyBank to form a continuous evolution loop.

## Method
OxyGent's approach can be understood as an operating system layer for multi-agent systems: it does not replace specific LLMs or restrict a single fixed workflow but instead regulates how capabilities in the system are encapsulated, invoked, tracked, and how the system learns from historical trajectories. The technical main line of the paper consists of three parts: unified Oxy abstraction, permission-driven dynamic planning, and the OxyBank evolution engine.

### Overall Architecture
The input consists of user requests, business context, and a set of available agent/tool/LLM/flow components. Developers first register these components as Oxy nodes and configure the accessible data scope, invokable downstream nodes, and runtime permissions for each node. Upon receiving a request, the system does not follow a pre-defined DAG; instead, an orchestrator dynamically generates a calling path based on permission relationships, current state, and task requirements.

During execution, each Oxy node undergoes a standard lifecycle: pre-processing, input saving, core execution, post-processing, and output formatting. OxyGent injects logic for monitoring, security auditing, time statistics, and visualization into these lifecycle joinpoints, keeping business code clean while managing cross-cutting concerns centrally.

The output includes not only the final answer but also a complete call graph, inputs/outputs for each node, time distribution, failure points, and replayable trajectories. These trajectories further enter OxyBank, where they are deduplicated, prioritized, labeled, reviewed, and fed back into knowledge bases, becoming material for prompt optimization, knowledge updates, or model fine-tuning.

In terms of system form, OxyGent is not just an agent runner for single-turn inference but a closed-loop framework covering "construction-inference-observation-labeling-evolution." The paper demonstrates instances of this framework including a file management assistant, a GAIA task-solving system, and an e-commerce classification system with over 2,000 agents.

### Key Designs
1.  **Unified Oxy Abstraction and Four-Layer Data Scoping**:
    -   **Function**: Wraps agents, tools, LLMs, and flows into Oxy components, providing capability units with consistent interfaces, lifecycles, and state access methods.
    -   **Mechanism**: OxyGent treats agents, tools, and flows as atomic nodes that can be registered, called, replaced, and monitored equally. Each node accesses data through a unified request object, restricted to four scopes: Application (global context), Session Group (shared memory for related sessions), Request (temporary state during a single inference), and Node (local parameters for the current node).
    -   **Design Motivation**: State boundaries are the most common source of error in real-world MAS. Excessive global state leads to permission leaks and reproducibility issues, while excessive local state increases collaboration costs. The four-layer scope makes "which data can be shared and to what extent" explicit, ensuring the system can reuse context without mixing all data into an uncontrollable object.

2.  **Permission-Driven Dynamic Planning and Standard Execution Lifecycle**:
    -   **Function**: Defines potential collaboration spaces through authorization relationships between nodes, while generating real calling graphs at runtime based on the current task and retaining complete observable trajectories.
    -   **Mechanism**: Developers do not need to manually write a complete DAG; instead, they define callable relationships and permission boundaries for Oxy nodes. During execution, the orchestrator selects the next step only from allowed edges, resulting in a dynamic yet constrained path. Each node execution follows stages like `_pre_process`, `_pre_save_data`, `_execute`, `_post_process`, and `_format_output`, where the system can uniformly insert logs, monitoring, security checks, and format conversions.
    -   **Design Motivation**: Fixed workflows are fragile, while completely free agent collaboration is difficult to audit. Permission-driven planning limits "which paths can be taken" in advance, leaving the "actual path taken" to runtime decisions. Standard lifecycles and AOP injection allow execution graphs, latency, resource congestion, and failure recovery to be visible without requiring manual tracing code in every business agent.

3.  **OxyBank Evolution Engine and Quality Gating**:
    -   **Function**: Transforms online execution trajectories into manageable AI assets for knowledge base updates, sample labeling, system evaluation, and prompt optimization.
    -   **Mechanism**: OxyBank captures agent execution chains and precipitates trajectories as memory assets. Before entering the asset library, data is deduplicated via MD5 to avoid frequency bias and prioritized based on call chain metadata (e.g., end-to-end user interactions are prioritized). Data must pass through a state machine of `pending -> annotated -> approved`; only approved samples enter knowledge bases or training pipelines. The system also provides an AI-driven Optimize Prompt module to extract experience from verified trajectories.
    -   **Design Motivation**: The greatest risk for self-evolving agents is feeding low-quality, hallucinated, or repetitive trajectories back into the system, magnifying errors. OxyBank focuses on turning data reflux into an auditable asset process: precipitation, labeling, review, and then updating. This maintains human-in-the-loop reliability while using LLMs to automatically summarize experiences and reduce manual prompt engineering.

### Loss & Training
This paper does not train a new model; thus, there is no traditional end-to-end loss function. Its "training strategy" is reflected in the system evolution layer: first, multi-agent execution trajectories are collected in actual tasks; then, through OxyBank's deduplication, prioritization, templated labeling, and human review, high-quality samples are formed; finally, these samples are used for knowledge base updates, prompt optimization, or model fine-tuning.

In the GAIA experiments, the paper uses a combination of DeepSeek-R1, GPT-4o, and Claude-3.5-Sonnet to build a multi-layer agent hierarchy. A Master Agent handles overall scheduling, Task Agents handle high-level decomposition, Coordinator Agents further call sub-agents like Web Search, Document Processing, and Reasoning Coding, and an Answerer Agent synthesizes the final answer or re-delegates if evidence is insufficient. This demonstrates that OxyGent acts as a configurable collaboration runtime: model capabilities come from underlying LLMs, while system gains come from abstraction, planning, memory, and observability.

## Key Experimental Results

### Main Results
The experiments are divided into public benchmarks and industrial cases. The public benchmark uses GAIA to verify if OxyGent can effectively manage long-chain, multi-tool, and multi-modal complex tasks; the industrial case uses an e-commerce customer service classification system to verify the accuracy, scalability, and cost trade-offs of large-scale agent topologies.

| Scenario | Baseline / Control | OxyGent Result | Gain | Description |
| :--- | :--- | :--- | :--- | :--- |
| GAIA overall | Single agent 36.21% | 59.14% | +22.93 pct pts | Gradually improves complex task solving via multi-agent, dynamic planning, and memory. |
| GAIA open-source leaderboard | OWL++ 60.80% | 59.14% | 2nd among open-source | 2025-07-22 leaderboard result, slightly lower than the strongest open-source method OWL++. |
| E-commerce Classification | RAG + DeepSeek-R1-Distill-Qwen-32B 61.3% | 85.6% | +24.3 pct pts | Ultra-large scale few-shot scenario with 2000+ agents and 2400+ labels. |
| Category Self-Evolution | Manual discovery of new categories | Avg 5.4 new categories/week | Auto discovery & verification | Demonstrates the benefit of dynamic topology and data reflux for business taxonomy. |

Two points are noteworthy regarding the GAIA results. First, 59.14% was not achieved through raw model power alone, as the ablation study shows a step-by-step increase from single agent to the memory version. Second, the paper does not claim to surpass all systems but emphasizes being close to OWL++ among open-source methods, proving OxyGent's capacity to support complex agent collaboration.

The e-commerce case better reflects OxyGent's industrial positioning. The system classifies real customer requests into over 2,400 labels, with some categories having only 10 samples. A single agent + RAG struggled to cover fine-grained categories and long-tail samples. OxyGent's hierarchical multi-agent topology significantly improved accuracy, though at the cost of mean inference latency reaching 2.3x that of the single agent baseline.

### Ablation Study
The GAIA ablation study builds system capabilities starting from a single agent. Since the unified Oxy abstraction is the non-removable foundation, the table compares gains from orchestration, planning, and memory reflux.

| Configuration | Avg | Level 1 | Level 2 | Level 3 | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Single agent | 36.21 | 61.29 | 29.56 | 10.20 | Single agent baseline; insufficient for complex tasks and high-level levels. |
| + Multi-agent | 42.19 | 62.37 | 35.85 | 24.49 | Role specialization significantly improves Level 3, showing complex tasks need collaboration. |
| + Planning | 52.16 | 62.37 | 54.09 | 26.53 | Dynamic planning primarily boosts Level 2, showing path selection is key for mid-difficulty tasks. |
| + Memory | 59.14 | 77.42 | 56.60 | 32.65 | Memory reflux significantly improves Level 1 and overall average, while continuing to aid hard tasks. |

### Key Findings
- Dynamic planning is one of the most significant sources of structural gain in GAIA; adding planning improved the overall score from 42.19% to 52.16%, and Level 2 from 35.85% to 54.09%. This shows that in tasks requiring multi-step retrieval or tool calling, the ability to dynamically choose the next step is more important than simply stacking agents.
- The Memory mechanism further increased the overall score from 52.16% to 59.14%, particularly Level 1 from 62.37% to 77.42%. This indicates that historical trajectories and experience reflux not only serve complex tasks but also reduce repetitive errors and format deviations in simple tasks.
- The industrial case demonstrates a clear trade-off between accuracy and latency: hierarchical MAS increased classification accuracy from 61.3% to 85.6%, but mean latency increased to 2.3x. The paper argues this is acceptable for offline or accuracy-priority scenarios but may not suit real-time cases.
- Failure cases often concentrated on highly ambiguous semantic queries, where the system might enter repetitive ReAct loops or generate hallucinations. The authors rely on trace replay and human review to mitigate this, indicating that observability is the foundation for diagnosis but does not automatically eliminate agent reasoning failures.

## Highlights & Insights
- The greatest highlight is how specifically the paper addresses "system engineering problems" for MAS. It doesn't just use abstract terms like modularity or observability; it grounds them in implementable mechanisms like unified node interfaces, four-layer data scoping, lifecycle hooks, real-time call graphs, and OxyBank state machines.
- Permission-driven dynamic planning is a practical compromise: it is neither as rigid as a fixed DAG nor as uncontrollable as an open agent group chat. For enterprise systems, this "dynamic choice within an authorized space" design is much easier to deploy.
- The value of OxyBank lies in upgrading agent logs into AI assets rather than stopping at an observability dashboard. While many frameworks can trace, OxyGent attempts to connect traces to labeling, auditing, and optimization, forming a true feedback loop.
- The e-commerce classification case with 2000+ agents is enlightening: multi-agent systems need not be limited to open-ended reasoning but can be used for hierarchical decision-making under large-scale taxonomies. This can be transferred to medical coding, legal classification, or financial risk attribution.
- AOP-style lifecycle injection is suitable for promotion to other agent runtimes. Security audits, permission checks, time statistics, and visualization should be unified runtime capabilities rather than being scattered across business logic.

## Limitations & Future Work
- The experiments are more focused on system demonstration and case verification than on rigorous cross-framework benchmarking. While the GAIA results show OxyGent can support a strong system, it lacks a systematic comparison with LangGraph, AutoGen, or CrewAI under identical model and tool settings.
- OxyBank's self-evolution still relies on manual resource allocation and review. The authors acknowledge that large-scale training currently requires manual configuration, meaning a fully automated agent development lifecycle is still some distance away.
- The latency cost in the industrial case is significant. A 2.3x latency increase is acceptable for offline classification but may be a bottleneck for interactive office assistants or online transaction risk control. Future work needs finer routing strategies to allow simple requests to take shorter paths.
- The paper does not fully discuss the consequences of misconfigured permission boundaries. Permission-driven planning assumes the permission graph itself is reliable; if authorizations are too broad, dynamic planning might expose sensitive tools to inappropriate agents.
- Future improvements could follow three paths: adding automatic resource scheduling for OxyBank, establishing standardized MAS observability benchmarks, and connecting permission graphs and trajectories to formal auditing or policy verification.

## Related Work & Insights
- **vs LangGraph**: LangGraph emphasizes low-level graph orchestration and stateful runtimes, suitable for explicitly building long-running flows. OxyGent emphasizes unifying agents, tools, LLMs, and flows into the same Oxy node type and generating graphs at runtime based on permissions, making it more prominent in dynamic topology and visualization.
- **vs CrewAI / MetaGPT**: These are more role-specialization or SOP-driven. OxyGent differs by turning fixed scripts into dynamic paths under permission constraints while using lifecycle hooks for unified monitoring and replay.
- **vs AutoGen**: AutoGen centers on asynchronous message passing and agent dialogues. OxyGent acts more as a production runtime, emphasizing data scoping, permission boundaries, time tracking, and asset management for data reflux.
- **vs OpenAI Agents SDK / Strands Agents**: These frameworks already provide tracing or guardrails. OxyGent's differentiator is integrating observability with dynamic planning and the OxyBank evolution engine; its goal is not just debugging but enabling trajectories to serve continuous system evolution.
- **vs OWL / AWorld / EvoAgent**: These focus on agent learning or automatic expansion. OxyGent's insight is that the learning loop requires a foundational asset management layer, otherwise online experiences struggle to transition from logs into stable, auditable training signals.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Unified abstraction, permission-driven planning, and OxyBank are systematic integrations of existing directions, but the problem definition for industrial MAS is very clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ GAIA ablations and real business cases are persuasive, though systematic comparisons with mainstream frameworks are missing.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, and both methods and cases are easy to understand; some system features rely heavily on diagrams, and implementation details could be deeper.
- **Value**: ⭐⭐⭐⭐⭐ Highly valuable for those building production-grade agent systems, especially the designs for state isolation, permission graphs, lifecycle tracing, and the data reflux loop.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives](social_dynamics_as_critical_vulnerabilities_that_undermine_objective_decision-ma.md)
- [\[ACL 2026\] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
