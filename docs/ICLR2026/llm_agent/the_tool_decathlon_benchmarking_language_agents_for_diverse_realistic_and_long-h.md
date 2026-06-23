---
title: >-
  [Paper Note] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution
description: >-
  [ICLR 2026][LLM Agent][language agent] This paper introduces Toolathlon, a language agent benchmark covering 32 software applications, 604 tools, and 108 tasks. It emphasizes realistic and diverse environment states and long-horizon multi-step interactions (averaging ~20 tool calls). The strongest model, Claude-4.5-Sonnet, achieves only a 38.6% success rate
tags:
  - ICLR 2026
  - LLM Agent
  - language agent
  - benchmark
  - MCP
  - tool calling
  - long-horizon
date: 2026-05-08
content_hash: 768f5e3deec8345b
---
# The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution

**Conference**: ICLR 2026  
**arXiv**: [2510.25726](https://arxiv.org/abs/2510.25726)  
**Code**: To be confirmed (Based on MCP servers)  
**Area**: LLM Agent / Benchmark / Tool Use  
**Keywords**: language agent, benchmark, MCP, tool calling, long-horizon, multi-app interaction, execution-based evaluation

## TL;DR
This paper introduces Toolathlon, a language agent benchmark covering 32 software applications, 604 tools, and 108 tasks. It emphasizes realistic and diverse environment states and long-horizon multi-step interactions (averaging ~20 tool calls). The strongest model, Claude-4.5-Sonnet, achieves only a 38.6% success rate.

## Background & Motivation
**Background**: Language agents are expected to complete complex workflows across applications and multiple steps in the real world, such as coordinating calendars with emails or monitoring databases to generate reports. This requires agents to possess integrated capabilities for tool discovery, multi-turn reasoning, state tracking, and cross-system coordination.

**Limitations of Prior Work**: Existing agent benchmarks suffer from three major shortcomings: (1) narrow application domains (mostly focusing on a single tool or API), (2) oversimplified tasks (completable in one or two steps), and (3) unrealistic environment states (using empty or minimalist initial states rather than the complex data found in real software), failing to adequately evaluate an agent's actual deployment capabilities.

**Key Challenge**: Researchers need to reliably assess the real-world performance of agents, but the massive gap between "toy-level" tasks in existing benchmarks and actual workflows leads to a disconnect between benchmark scores and real-world utility—agents perform well on simple benchmarks but fail frequently in real scenarios.

**Key Insight**: The emergence of the Model Context Protocol (MCP) provides the infrastructure for building standardized tool interfaces. Toolathlon builds its tool layer on high-quality MCP servers, some implemented or revised by the team to ensure interface quality and consistency.

**Goal**: Existing benchmarks often rely on LLM judgments or string matching to evaluate outputs, which are prone to error and non-reproducible. A rigorous verification mechanism based on program execution is needed to ensure deterministic determination of task completion.

**Core Idea**: Construct the first comprehensive agent benchmark that simultaneously satisfies four dimensions: diverse application coverage, realistic environment states, long-horizon complex tasks, and execution-based verification.

## Method

### Overall Architecture
Toolathlon is a language agent benchmark that exposes 32 real software applications as 604 tools via MCP servers, requiring agents to complete 108 long-horizon tasks in environments with realistic initial data. The benchmark consists of four components: a diverse application tool layer, realistic environment states, complex tasks averaging ~20 calls, and deterministic verification based on program execution.

### Key Designs

These four design points simultaneously increase the difficulty across four dimensions: the tool layer complicates tool selection, environment states provide "dirty" starting points, task design extends the logic chain, and execution-based evaluation enforces strict scoring.

**1. Application and Tool Layer: Turning "Tool Discovery" into a Challenge**

In single-tool benchmarks, agents do not need to select tools, but in real deployment, "finding the right tool among many" is itself a difficulty. Toolathlon spans 32 software applications, ranging from daily productivity (Google Calendar, Notion, Gmail) to professional tools (WooCommerce e-commerce, Kubernetes container orchestration, BigQuery data analysis), totaling 604 tools/APIs. The tool layer is standardized on MCP servers. The team revised or re-implemented several existing MCP servers to ensure interface quality and cross-application behavioral consistency. Consequently, agents must locate the correct tool combination among hundreds of options at each step rather than facing pre-selected APIs, turning "tool retrieval" back into a real obstacle.

**2. Realistic Environment States: Working with Existing "Dirty" Data**

While tools may be available in many benchmarks, the initial environments are often blank, artificially lowering difficulty. Toolathlon injects initial states from real software for every task—Canvas courses contain actual enrollment and grade data for dozens of students, financial sheets have existing accounts, and e-commerce systems contain historical orders. Agents must retrieve, filter, and correlate within existing complex data before modification, mimicking a new employee taking over an existing system rather than performing a demo in a clean sandbox. Many failures stem from "writing data without understanding the existing state."

**3. Task Design and Long-Horizon Complexity: Maintaining Planning Pressure over ~20 Calls**

Short tasks end in one or two steps, failing to expose weaknesses in planning and context maintenance. All 108 tasks in Toolathlon are manually authored or distilled from real workflows, each requiring cross-application collaboration and averaging ~20 tool calls—typical long-horizon tasks. Task types include information retrieval and aggregation, cross-system data synchronization, conditional execution, and report generation. Agents must track state across dozens of interactions, handle conditional branches, and recover from mid-way errors—any deviation accumulates in subsequent steps, revealing capabilities that short tasks cannot measure.

**4. Execution-Based Evaluation: Deterministic Judgment via Post-Task State Checks**

Using LLM scoring or string matching introduces noise and reduces reproducibility. Toolathlon provides a dedicated evaluation script for each task that directly checks system state changes—database records, file contents, and API states—to strictly determine completion. The evaluation is deterministic, covering both correctness (goal achievement) and completeness (no missing steps). Some tasks include intermediate checkpoints to locate exactly where an agent failed. Compared to "grading based on final responses," this "world-state-based" scoring is both objective and reproducible.

## Key Experimental Results

### Main Results: Model Success Rate Comparison

| Model | Success Rate (%) | Avg. Tool Calls | Type | Note |
|-------|------------------|-----------------|------|------|
| Claude-4.5-Sonnet | **38.6** | 20.2 | Closed | Strongest Model |
| GPT-4o / GPT-5 series | 25-35 (Est.) | ~20 | Closed | Moderate Performance |
| DeepSeek-V3.2-Exp | **20.1** | ~20 | Open | Strongest Open-source |
| Other Open-source | <20 | Highly Variable | Open | Generally Insufficient |

### Performance Across Application Categories

| Category | Typical Apps | Performance Trend | Difficulty Analysis |
|----------|--------------|-------------------|---------------------|
| Productivity | Calendar, Gmail | Relatively Good | Highly structured APIs |
| Project Mgmt | Notion, Canvas | Moderate | Requires understanding complex data structures |
| DevOps | Kubernetes, Git | Poor | Requires domain expertise |
| Data Analysis | BigQuery, Sheets | Poor | Requires multi-step data processing |
| E-commerce | WooCommerce | Poor | Complex business logic |

### Key Findings
- Even the strongest model, Claude-4.5-Sonnet, reached only 38.6% success, indicating significant deficiencies in current agents regarding realistic, long-horizon, multi-app tasks.
- A gap of approximately 18 percentage points exists between open-source and closed-source models (38.6% vs 20.1%); tool-use capability remains a major weakness for open-source weights.
- Primary failure reasons include: incorrect tool selection, loss of key information in long contexts, cross-app data format mismatches, and lack of error recovery.
- Success rates for specialized tools (e.g., Kubernetes, BigQuery) are significantly lower than for productivity tools, identifying domain knowledge as a critical bottleneck.
- While MCP standardization reduces integration complexity, agents still struggle to "understand tool capability boundaries."

## Highlights & Insights
- **The Philosophy of "Realistic Environment States"**: Toolathlon provides complex real-world data environments rather than just tool interfaces. This forces agents to "work in chaos"—much like a human employee facing existing data systems. This reflects deployment challenges more accurately than blank-box testing.
- **MCP as Infrastructure**: Using MCP servers as the tool layer is forward-looking; as the MCP ecosystem grows, Toolathlon can naturally expand to more applications.
- **Value of Long-Horizon Evaluation**: Task lengths averaging 20 calls effectively test planning, context retention, and error recovery—traits invisible in short-duration tasks.

## Limitations & Future Work
- **Limited Task Scale**: Although high-quality and manually constructed, 108 tasks may be insufficient to evaluate generalization across all scenarios.
- **Static Environments**: Environment states are fixed at the start of evaluation and do not involve dynamic changes (e.g., real-time notifications, concurrent users).
- **High Labeling Cost**: Manually writing evaluation scripts for each task makes scaling to more tasks/apps expensive.
- **Lack of Interactive Scenarios**: Tasks are currently one-way; there are no scenarios where a user provides feedback or modifies requirements mid-task.
- **Model Inference Cost**: An average of 20 calls per task implies significant API costs, limiting the feasibility of evaluating a wide range of models comprehensively.
- **Geographic and Linguistic Limitations**: Current applications and tasks are primarily English-based and centered on global/North American tools, lacking coverage for non-English environments.
- **Missing Safety Dimension**: The benchmark focuses on task completion rates and does not systematically evaluate agent safety (e.g., data leaks, permission escalation, or unintended side effects).

## Related Work & Insights
- **vs AgentBench (Liu et al., 2023)**: AgentBench covers OS, DB, and Web, but its task length and state complexity are inferior to Toolathlon. Toolathlon is a significant upgrade in diversity and realism.
- **vs ToolBench (Qin et al., 2024)**: ToolBench offers massive API sets and automated task generation, but tasks are often single-step or few-step with simple states. Toolathlon emphasizes long-horizon multi-step tasks and realistic states.
- **vs SWE-bench (Jimenez et al., 2024)**: SWE-bench focuses on the single domain of software engineering (code fixes). Toolathlon covers 32 different domains to test general tool-use capability.
- **vs GAIA (Mialon et al., 2023)**: GAIA tests general assistants on search and reasoning with shorter tasks. Toolathlon's MCP framework makes it closer to real-world application integration.
- **vs $\tau$-bench (Yao et al., 2024)**: $\tau$-bench also focuses on tool use but designs tasks more toward reasoning verification. Toolathlon emphasizes the breadth of application coverage and state realism.
- **vs AgentDojo (Debenedetti et al., 2024)**: AgentDojo emphasizes safety (defense against prompt injection), while Toolathlon emphasizes functionality (completion rate). Their perspectives are complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of MCP + Realistic states + Long-horizon tasks is a new paradigm for agent evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive testing of SOTA models across 32 apps × 604 tools × 108 tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear descriptions, though some details are quite dense.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in "realism" for agent evaluation; likely to become a community standard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LongHorizonUI: A Unified Framework for Robust Long-Horizon Task Automation of GUI Agent](longhorizonui_a_unified_framework_for_robust_long-horizon_task_automation_of_gui.md)
- [\[CVPR 2026\] WebGym: Scaling Training Environments for Long-Horizon Visual Web Agents with Realistic Tasks](../../CVPR2026/llm_agent/webgym_scaling_training_environments_for_long-horizon_visual_web_agents_with_rea.md)
- [\[ICLR 2026\] MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](mem1_learning_to_synergize_memory_and_reasoning_for_efficient_long-horizon_agent.md)
- [\[ICLR 2026\] Benchmarking LLM Tool-Use in the Wild](benchmarking_llm_tool-use_in_the_wild.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)

</div>

<!-- RELATED:END -->
