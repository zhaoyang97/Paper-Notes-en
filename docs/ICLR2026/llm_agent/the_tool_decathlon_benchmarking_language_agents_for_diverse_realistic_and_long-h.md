---
title: >-
  [Paper Note] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution
description: >-
  [ICLR 2026][LLM Agent][language agent] This paper introduces Toolathlon, a language agent benchmark covering 32 software applications, 604 tools, and 108 tasks, emphasizing realistic and diverse environment states alongside long-horizon multi-step interactions (averaging ~20 tool calls per task). The strongest evaluated model, Claude-4.5-Sonnet, achieves only 38.6% task success rate.
tags:
  - ICLR 2026
  - LLM Agent
  - language agent
  - benchmark
  - MCP
  - tool calling
  - long-horizon
  - multi-application interaction
  - execution-based evaluation
date: 2026-05-08
content_hash: 8f79275558b68d33
---

# The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution

**Conference**: ICLR 2026
**arXiv**: [2510.25726](https://arxiv.org/abs/2510.25726)
**Code**: To be confirmed (based on MCP servers)
**Area**: LLM Agent / Benchmark / Tool Use
**Keywords**: language agent, benchmark, MCP, tool calling, long-horizon, multi-application interaction, execution-based evaluation

## TL;DR
This paper introduces Toolathlon, a language agent benchmark covering 32 software applications, 604 tools, and 108 tasks, emphasizing realistic and diverse environment states alongside long-horizon multi-step interactions (averaging ~20 tool calls per task). The strongest evaluated model, Claude-4.5-Sonnet, achieves only 38.6% task success rate.

## Background & Motivation
**State of the Field**: Language agents are increasingly expected to complete complex, multi-step workflows across real-world applications—such as managing calendar-email coordination or monitoring databases to generate reports. This demands agents capable of tool discovery, multi-turn reasoning, state tracking, and cross-system coordination.

**Limitations of Prior Work**: Existing agent benchmarks suffer from three major shortcomings: (1) narrow application coverage, often focusing on a single tool or API; (2) overly simplified tasks solvable in one or two tool calls; and (3) unrealistic environment states that use blank or minimal initial conditions rather than the complex data found in real software. These limitations prevent meaningful evaluation of agents' real-world deployment capabilities.

**Root Cause**: Researchers need reliable evaluations of agents' real-world performance, yet a substantial gap exists between the "toy-level" tasks in current benchmarks and genuine workflows. This disconnect causes benchmark scores to diverge from actual capability—agents perform well on simple benchmarks but fail frequently in realistic scenarios.

**Opportunity from MCP**: The emergence of the Model Context Protocol (MCP) provides infrastructure for building standardized tool interfaces. Toolathlon constructs its tool layer atop high-quality MCP servers, some of which were implemented or revised by the authors themselves to ensure interface quality and consistency.

**Evaluation Reliability**: Many existing benchmarks rely on LLM judgment or string matching to assess agent outputs, introducing substantial noise and reproducibility issues. A rigorous, execution-based verification mechanism is needed to ensure deterministic assessment of task completion.

**Core Idea**: To construct the first comprehensive agent benchmark that simultaneously satisfies four dimensions: diverse application coverage, realistic environment states, long-horizon complex tasks, and execution-based evaluation.

## Method

### Overall Architecture
Toolathlon is organized around four core design principles: application diversity (Diverse Apps), environment realism (Realistic Setup), task complexity (Long-Horizon Tasks), and evaluation reliability (Execution-Based Evaluation).

### Key Designs

1. **Application and Tool Coverage**:

    - Covers 32 software applications spanning everyday productivity tools (Google Calendar, Notion, Gmail) to specialized platforms (WooCommerce for e-commerce, Kubernetes for container orchestration, BigQuery for data analytics), encompassing 604 distinct tools/APIs.
    - The tool layer is implemented via MCP servers; the authors revised or re-implemented select MCP servers to ensure interface quality and behavioral consistency.
    - Design intent: agents must confront the "tool discovery" challenge—identifying the correct combination of tools from a large pool to accomplish a given task.

2. **Realistic Environment States**:

    - Unlike benchmarks that merely provide functional tool interfaces against empty environments, Toolathlon supplies each task with an initial environment state drawn from real software.
    - Examples include Canvas courses populated with dozens of students' genuine course data, real financial spreadsheets, and e-commerce systems with existing orders.
    - Value: agents must search, filter, and relate information within complex pre-existing data, rather than simply creating records in a blank environment.

3. **Task Design and Complexity**:

    - The benchmark comprises 108 tasks, all manually authored or distilled from real workplace scenarios, each requiring agent interaction with multiple applications.
    - On average, each task requires approximately 20 tool calls, qualifying as long-horizon tasks that demand sustained context maintenance and multi-step planning.
    - Task types include: information retrieval and aggregation, cross-system data synchronization, conditional logic and workflow execution, and report generation.

4. **Execution-Based Evaluation**:

    - Each task is paired with a dedicated evaluation script that verifies task completion by inspecting system state changes (e.g., database records, file contents, API states).
    - Evaluation is deterministic—it does not rely on LLM judgment or fuzzy matching, eliminating noise introduced by the evaluation process itself.
    - Evaluation scripts assess both correctness (whether the task was completed) and completeness (whether any steps were missed), with some tasks incorporating intermediate checkpoint checks.

## Key Experimental Results

### Main Results: Model Success Rate Comparison

| Model | Success Rate (%) | Avg. Tool Calls | Type | Notes |
|-------|-----------------|-----------------|------|-------|
| Claude-4.5-Sonnet | **38.6** | 20.2 | Closed-source | Best overall model |
| GPT-4o / GPT-5 series | 25–35 (est.) | ~20 | Closed-source | Moderate performance |
| DeepSeek-V3.2-Exp | **20.1** | ~20 | Best open-source | Best open-weight result |
| Other open-source models | <20 | Variable | Open-source | Broadly insufficient |

### Performance Variation Across Application Categories

| Application Category | Typical Apps | Performance Trend | Key Challenges |
|---------------------|-------------|-------------------|----------------|
| Everyday productivity | Calendar, Gmail | Relatively strong | Highly structured APIs |
| Project management | Notion, Canvas | Moderate | Complex data structures |
| DevOps | Kubernetes, Git | Weak | Requires specialized domain knowledge |
| Data analytics | BigQuery, Sheets | Weak | Multi-step data processing |
| E-commerce | WooCommerce | Weak | Complex business logic |

### Key Findings
- Even the strongest model, Claude-4.5-Sonnet, achieves only 38.6% success, indicating that current agents remain substantially insufficient for realistic, long-horizon, multi-application tasks.
- A gap of approximately 18 percentage points exists between closed-source and open-source models (38.6% vs. 20.1%), revealing tool use as a notable weakness of open-source models.
- Primary failure modes include: incorrect tool selection, loss of critical information over long contexts, failure to adapt data formats across applications, and lack of error recovery capability.
- Success rates on specialized domain tools (e.g., Kubernetes, BigQuery) are markedly lower than on everyday productivity tools, identifying domain knowledge as a key bottleneck for current agents.
- While MCP server standardization reduces tool integration complexity, agents still fall short in "understanding the capability boundaries" of individual tools.

## Highlights & Insights
- **The philosophy of "realistic environment states"**: Toolathlon not only provides tool interfaces but also supplies complex, pre-populated data environments. This forces agents to "work in the midst of existing complexity"—analogous to the challenge a new employee faces when inheriting an established system with real data. This more accurately reflects the difficulties encountered in actual deployment than blank-environment testing.
- **MCP as infrastructure**: The choice to build the tool layer on MCP servers is forward-looking—as the MCP ecosystem grows, Toolathlon can naturally scale to additional applications and tools.
- **The evaluation value of long-horizon tasks**: An average task length of 20 tool calls effectively probes agents' planning capability, context maintenance, and error recovery—qualities that remain invisible in short-horizon evaluations.

## Limitations & Future Work
- **Limited task scale**: Although the 108 tasks are manually constructed with high quality, their relatively small number may be insufficient to evaluate model generalization across a broader range of scenarios.
- **Static environments**: Task environments are fixed at the start of evaluation and do not involve dynamically changing conditions (e.g., real-time notifications, concurrent user actions) that are common in real-world deployments.
- **High annotation cost**: Each task requires a manually authored evaluation script, making expansion to additional tasks and applications costly.
- **Absence of interactive scenarios**: All tasks are unidirectional—agents execute tasks without any mid-task user feedback or requirement changes.
- **Model API cost**: An average of 20 tool calls per task incurs substantial API overhead, limiting the feasibility of comprehensive evaluation across a wider range of models.
- **Geographic and linguistic limitations**: Current applications and tasks are primarily English-language and oriented toward North American or globally standardized software tools, with insufficient coverage of non-English environments and localized applications.
- **Missing safety dimension**: The benchmark primarily measures task completion rate and does not systematically evaluate the safety of agent tool use (e.g., data leakage, privilege escalation, unintended side effects).

## Related Work & Insights
- **vs. AgentBench (Liu et al., 2023)**: AgentBench covers operating systems, databases, and web environments, but its task length and environment state complexity are considerably lower than Toolathlon's. Toolathlon represents a significant upgrade in application diversity and environment realism.
- **vs. ToolBench (Qin et al., 2024)**: ToolBench provides a large-scale API collection with automated task generation, but tasks are predominantly single-step or few-step calls with simple environment states. Toolathlon emphasizes long-horizon multi-step interaction and realistic states.
- **vs. SWE-bench (Jimenez et al., 2024)**: SWE-bench focuses on a single domain—software engineering (code repair)—whereas Toolathlon spans 32 distinct application domains, evaluating agents' general-purpose tool use capability.
- **vs. GAIA (Mialon et al., 2023)**: GAIA tests general AI assistants on web search and reasoning with relatively short tasks. Toolathlon's MCP-based framework positions it closer to real application integration scenarios.
- **vs. τ-bench (Yao et al., 2024)**: τ-bench also addresses tool use but is oriented more toward reasoning capability verification. Toolathlon places greater emphasis on breadth of application coverage and environment state realism.
- **vs. AgentDojo (Debenedetti et al., 2024)**: AgentDojo focuses on security (prompt injection defense), while Toolathlon focuses on functionality (task completion rate); the two benchmarks offer complementary evaluation perspectives.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of MCP, realistic environment states, and long-horizon tasks establishes a new paradigm for agent evaluation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation of state-of-the-art models across 32 applications × 604 tools × 108 tasks
- Writing Quality: ⭐⭐⭐⭐ Benchmark description is clear, though some sections are detail-dense
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in agent evaluation regarding "realism"; has strong potential to become a community-standard benchmark

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](../../CVPR2026/llm_agent/carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](harnessing_uncertainty_entropy-modulated_policy_gradients_for_long-horizon_llm_a.md)
- [\[AAAI 2026\] Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution](../../AAAI2026/llm_agent/cook_and_clean_together_teaching_embodied_agents_for_paralle.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)

<!-- RELATED:END -->
