---
title: >-
  [Paper Note] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts
description: >-
  [ACL 2026][LLM Agent][Autonomous Agents] Proposes AgencyBench—a comprehensive benchmark containing 138 real-world tasks that evaluate 6 core agent capabilities. Each scenario averages 90 tool calls and 1 million tokens…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Autonomous Agents"
  - "Long-horizon tasks"
  - "Real-world benchmark"
  - "User simulation"
  - "Docker sandbox evaluation"
date: 2026-05-08
content_hash: cd460f857a5afcf3
---

# AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts

**Conference**: ACL 2026  
**arXiv**: [2601.11044](https://arxiv.org/abs/2601.11044)  
**Code**: [GitHub](https://github.com/GAIR-NLP/AgencyBench)  
**Area**: LLM Agent / Benchmark  
**Keywords**: Autonomous Agents, Long-horizon tasks, Real-world benchmark, User simulation, Docker sandbox evaluation

## TL;DR

Proposes AgencyBench—a comprehensive benchmark containing 138 real-world tasks that evaluate 6 core agent capabilities. Each scenario averages 90 tool calls and 1 million tokens, achieving fully automated evaluation through user simulation agents and Docker sandboxes.

## Background & Motivation

**Background**: LLM-based autonomous agents are penetrating multiple domains such as software development, scientific research, and daily usage; however, evaluation benchmarks lag significantly behind agent capability development.

**Limitations of Prior Work**: (1) Existing benchmarks focus on single capabilities (e.g., tool use or software engineering), failing to capture the multidimensional and long-horizon nature of real-world tasks; (2) Evaluation of real-world tasks relies on human-in-the-loop feedback, creating a bottleneck for automated evaluation; (3) Task complexity is insufficient—most benchmarks require only dozens of tool calls.

**Key Challenge**: The capabilities of frontier agents have far exceeded the testing scope of existing benchmarks, necessitating more challenging evaluations.

**Goal**: To build a high-complexity, multidimensional, and fully automated real-world agent benchmark.

**Key Insight**: Collect tasks from real work scenarios via 20 human experts (AI researchers, developers) to construct a hierarchical capability-scenario-task system.

**Core Idea**: Replace human feedback with user simulation agents and perform visual evaluations via Docker sandboxes to achieve fully automated rollout collection and scoring for long-horizon complex tasks.

## Method

### Overall Architecture

Hierarchical design: 6 core capabilities (game development, frontend, backend, code generation, research, MCP tools) $\rightarrow$ 32 real-world scenarios $\rightarrow$ 138 specific tasks. Each scenario contains 1-5 sequential tasks of increasing difficulty, where results of preceding tasks affect subsequent ones. Performance isolation is ensured through a three-space separation of workspace, sandbox, and evalspace.

### Key Designs

1.  **User Simulation Agent**:

    - **Function**: Replaces humans to provide iterative feedback in multi-turn interactions.
    - **Mechanism**: Simulates real user behavior—when the agent submits intermediate results, the simulation agent provides revision suggestions and confirmations based on the task description and rubric.
    - **Design Motivation**: To eliminate the human-in-the-loop bottleneck, allowing rollouts spanning several hours to be completed fully automatically.

2.  **Docker Sandbox Evaluation**:

    - **Function**: Performs visual and functional evaluations of agent-produced code/files.
    - **Mechanism**: Synchronizes deliverables into a Docker container, simulates human-machine operations (UI rendering, mouse clicks, screen recording), generates visual artifacts, and uses evaluation scripts and LLM judges to score based on the rubric.
    - **Design Motivation**: Many real-world task outputs (e.g., games, websites) cannot be evaluated by text alone and require actual execution and visual inspection.

3.  **Hierarchical Task Design**:

    - **Function**: Simulates the progressive complexity of real workflows.
    - **Mechanism**: Difficulty increases across 1-5 tasks per scenario, with prior results impacting subsequent steps—e.g., a "Gomoku Game" scenario evolves from a basic board to adding an AI opponent, undo functionality, and theme switching.
    - **Design Motivation**: Real-world tasks are rarely completed in one step; this design tests the agent's context retention and long-horizon planning capabilities.

### Loss & Training

Evaluation employs a rubric-based 0-10 score, combining rule-based scripts and LLM-based judges. A full-consensus policy is used for data quality—all 4 experts must agree for a task to be included.

## Key Experimental Results

### Main Results

| Model Type | Average Score | Highest | Lowest |
| :--- | :--- | :--- | :--- |
| Closed-source | 48.4% | GPT-5.2 (56.5%) | Grok-4.1-Fast (44.3%) |
| Open-source | 32.1% | GLM-4.6 (38.6%) | Qwen-3-235B (27.0%) |

### Key Behavioral Differences

| Model | Characteristics | Description |
| :--- | :--- | :--- |
| GPT-5.2 | Strong feedback self-correction | Best at utilizing user feedback for improvement |
| Grok-4.1-Fast | High token efficiency | Completes tasks with fewer tokens |
| Claude-4.5-Opus | Preference for Shell tools | More frequent use of command-line operations |
| Gemini-3-Pro | Preference for file management | More frequent use of file and memory management tools |

### Key Findings

- Closed-source models significantly outperform open-source models (48.4% vs 32.1%), with the gap wider than on short-task benchmarks.
- A "Home Field Advantage" effect is evident—models perform best within their native frameworks (e.g., Claude + Claude-Agent-SDK).
- Even the strongest current models reach only 56.5%, indicating that long-horizon real-world tasks remain a massive challenge.
- Different models exhibit distinct tool-use preference variations, suggesting influences from architecture and training data.

## Highlights & Insights

- Task complexity far exceeds existing benchmarks—an average of 90 tool calls and 1 million tokens represents a qualitative leap.
- The combination of user simulation agents and Docker sandboxes solves the core challenge of automated evaluation for long-horizon tasks.
- The discovery of "Home Field Advantage" provides important insights for agent framework design—general frameworks may be inferior to specialized ones.

## Limitations & Future Work

- 138 tasks may still be insufficient to fully cover real-world scenarios.
- The quality of the user simulation agent sets the upper bound for evaluation reliability.
- Complexity in Docker sandbox environment configuration may limit community adoption.
- Future extensions could include more domains (e.g., data analysis, design, writing).

## Related Work & Insights

- **vs SWE-bench**: SWE-bench focuses specifically on software engineering, whereas AgencyBench covers 6 core capabilities.
- **vs GAIA**: GAIA averages only 10K tokens; AgencyBench is 100x more complex.
- **vs ToolLLM**: ToolLLM focuses on tool-call correctness, whereas AgencyBench focuses on end-to-end task completion.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Significantly exceeds existing benchmarks in scale and authenticity.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model comparisons, behavioral analysis, and framework comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich examples.
- **Value**: ⭐⭐⭐⭐⭐ Sets a new standard for next-generation agent evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](../../AAAI2026/llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)
- [\[ACL 2026\] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](shopping_companion_a_memory-augmented_llm_agent_for_real-world_e-commerce_tasks.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)

</div>

<!-- RELATED:END -->
