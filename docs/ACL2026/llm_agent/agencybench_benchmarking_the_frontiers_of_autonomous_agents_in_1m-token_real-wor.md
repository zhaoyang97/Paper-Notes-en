---
title: >-
  [Paper Note] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts
description: >-
  [ACL 2026][LLM Agent][autonomous agents] This paper proposes AgencyBench — a comprehensive benchmark comprising 138 real-world tasks that evaluates six core agent capabilities. Each scenario requires an average of 90 too…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "autonomous agents"
  - "long-horizon tasks"
  - "real-world benchmark"
  - "user simulation"
  - "Docker sandbox evaluation"
date: 2026-05-08
content_hash: 4bfa638925a681e5
---

# AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts

**Conference**: ACL 2026
**arXiv**: [2601.11044](https://arxiv.org/abs/2601.11044)  
**Code**: [GitHub](https://github.com/GAIR-NLP/AgencyBench)  
**Area**: LLM Agent / Benchmark
**Keywords**: autonomous agents, long-horizon tasks, real-world benchmark, user simulation, Docker sandbox evaluation

## TL;DR

This paper proposes AgencyBench — a comprehensive benchmark comprising 138 real-world tasks that evaluates six core agent capabilities. Each scenario requires an average of 90 tool calls and 1M tokens. Fully automated evaluation is achieved via a user simulation agent and Docker sandbox.

## Background & Motivation

**Background**: LLM-based autonomous agents are increasingly deployed across software development, scientific research, and everyday use, yet evaluation benchmarks have lagged significantly behind the growth in agent capabilities.

**Limitations of Prior Work**: (1) Existing benchmarks focus on isolated capabilities (e.g., tool use or software engineering) and fail to capture the multi-dimensional, long-horizon nature of real-world tasks; (2) real-task evaluation relies on human-in-the-loop feedback, becoming a bottleneck for automated assessment; (3) task complexity is insufficient — most benchmarks require only tens of tool calls.

**Key Challenge**: The capabilities of frontier agents have far surpassed the scope of existing benchmarks, necessitating substantially more challenging evaluations.

**Goal**: Construct a high-complexity, multi-dimensional, fully automated benchmark for evaluating real-world agents.

**Key Insight**: Twenty human experts (AI researchers and developers) collect tasks from authentic work scenarios, organized into a hierarchical capability–scenario–task taxonomy.

**Core Idea**: Replace human feedback with a user simulation agent and employ Docker sandbox execution with visual evaluation to enable fully automated rollout collection and scoring for long-horizon, complex tasks.

## Method

### Overall Architecture

A hierarchical design is adopted: six core capabilities (game development, frontend, backend, code generation, research, MCP tools) → 32 real-world scenarios → 138 specific tasks. Each scenario contains 1–5 sequentially ordered tasks of increasing difficulty, where the results of earlier tasks affect subsequent ones. Evaluation is conducted across three isolated spaces — workspace, sandbox, and evalspace — to ensure environmental separation.

### Key Designs

1. **User Simulation Agent**:

    - **Function**: Provides iterative feedback in place of a human during multi-turn interactions.
    - **Mechanism**: Simulates authentic user behavior — when the agent submits an intermediate result, the simulation agent provides revision suggestions or confirmations based on the task description and rubric.
    - **Design Motivation**: Eliminates the human-in-the-loop bottleneck, enabling rollouts that would otherwise take hours to complete in a fully automated manner.

2. **Docker Sandbox Evaluation**:

    - **Function**: Performs visual and functional evaluation of code and files produced by the agent.
    - **Mechanism**: Deliverables are synchronized into a Docker container, where human-computer interactions are simulated (UI rendering, mouse clicks, screen recording) to generate visual artifacts, which are then scored by evaluation scripts and an LLM judge against the rubric.
    - **Design Motivation**: Many real-world task outputs (e.g., games, web pages) cannot be assessed through text alone and require actual execution and visual inspection.

3. **Hierarchical Task Design**:

    - **Function**: Simulates the progressive complexity of real-world workflows.
    - **Mechanism**: Each scenario's 1–5 tasks increase in difficulty, with earlier completions affecting later ones — for example, a "Gomoku game" scenario progresses from a basic board to adding an AI opponent, undo functionality, and theme switching.
    - **Design Motivation**: Real-world tasks are never completed in a single step; this design tests the agent's ability to maintain context and engage in long-horizon planning.

### Loss & Training

Evaluation uses rubric-based scoring on a 0–10 scale, combining rule-based evaluation scripts with an LLM-based judge. A unanimous agreement strategy is applied for data quality — all four expert reviewers must approve a task before it is included.

## Key Experimental Results

### Main Results

| Model Type | Avg. Score | Highest | Lowest |
|-----------|------------|---------|--------|
| Closed-source | 48.4% | GPT-5.2 (56.5%) | Grok-4.1-Fast (44.3%) |
| Open-source | 32.1% | GLM-4.6 (38.6%) | Qwen-3-235B (27.0%) |

### Key Behavioral Differences

| Model | Characteristic | Description |
|-------|---------------|-------------|
| GPT-5.2 | Strong feedback self-correction | Most effective at incorporating user feedback |
| Grok-4.1-Fast | High token efficiency | Completes tasks with fewer tokens |
| Claude-4.5-Opus | Preference for shell tools | More frequent use of command-line operations |
| Gemini-3-Pro | Preference for file management | More frequent use of file and memory management tools |

### Key Findings
- Closed-source models substantially outperform open-source models (48.4% vs. 32.1%), with a larger gap than observed on short-task benchmarks.
- A pronounced "home advantage" effect is observed — models perform best within their native frameworks (e.g., Claude + Claude-Agent-SDK).
- Even the strongest current model reaches only 56.5%, indicating that long-horizon real-world tasks remain a formidable challenge.
- Distinct tool-use preferences across models suggest influences from architectural differences and training data.

## Highlights & Insights
- Task complexity far exceeds existing benchmarks — an average of 90 tool calls and 1M tokens represents a qualitative leap.
- The combination of user simulation agent and Docker sandbox addresses the core challenge of automated evaluation for long-horizon tasks.
- The "home advantage" finding has important implications for agent framework design — general-purpose frameworks may be outperformed by specialized ones.

## Limitations & Future Work
- 138 tasks may still be insufficient to comprehensively cover real-world scenarios.
- The quality of the user simulation agent constitutes an upper bound on evaluation reliability.
- The complexity of Docker sandbox configuration may hinder community adoption.
- Future work could extend coverage to additional domains such as data analysis, design, and writing.

## Related Work & Insights
- **vs. SWE-bench**: SWE-bench focuses on a single software engineering capability, whereas AgencyBench covers six capability dimensions.
- **vs. GAIA**: GAIA averages only 10K tokens; AgencyBench operates at 100× the complexity.
- **vs. ToolLLM**: ToolLLM targets tool-call correctness, while AgencyBench focuses on end-to-end task completion.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Substantially surpasses existing benchmarks in scale and real-world fidelity.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model comparison, behavioral analysis, and framework-level evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich illustrative examples.
- **Value**: ⭐⭐⭐⭐⭐ Sets a new standard for next-generation agent evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](../../AAAI2026/llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)

</div>

<!-- RELATED:END -->
