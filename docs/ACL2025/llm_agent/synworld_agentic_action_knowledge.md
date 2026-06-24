---
title: >-
  [Paper Note] SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement
description: >-
  [ACL 2025][LLM Agent][action knowledge] SynWorld proposes enabling agents to explore and refine action knowledge (tool descriptions and workflows) through Monte Carlo Tree Search (MCTS) in synthesized virtual scenarios. This allows agents to autonomously adapt to tool usage in new environments, achieving approximately a 9% improvement over the ReAct baseline on ToolBench.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "action knowledge"
  - "MCTS"
  - "scenario synthesis"
  - "tool learning"
  - "agent exploration"
date: 2026-05-08
content_hash: 58f87597ce21b79b
---

# SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement

**Conference**: ACL 2025  
**arXiv**: [2504.03561](https://arxiv.org/abs/2504.03561)  
**Code**: [https://github.com/zjunlp/SynWorld](https://github.com/zjunlp/SynWorld)  
**Area**: LLM Agent / Tool Learning  
**Keywords**: action knowledge, MCTS, scenario synthesis, tool learning, agent exploration

## TL;DR
SynWorld proposes enabling agents to explore and refine action knowledge (tool descriptions and workflows) through Monte Carlo Tree Search (MCTS) in synthesized virtual scenarios. This allows agents to autonomously adapt to tool usage in new environments, achieving approximately a 9% improvement over the ReAct baseline on ToolBench.

## Background & Motivation

**Background**: LLM-based agents complete tasks by interacting with environments through tool (API) calls, but tool description documents often mismatch actual usage.

**Limitations of Prior Work**:
   - Manually maintaining tool documentation is labor-intensive, and documentation is often missing or outdated in new environments.
   - Existing methods (e.g., EasyTool, DRAFT) learn in synthesized single-step scenarios, failing to handle multi-step tool compositions.
   - Linear iterative optimization lacks clear direction and is prone to local optima.

**Key Challenge**: Agents need to efficiently learn tool usage in unknown environments but lack structured exploration and optimization mechanisms.

**Goal**: Enable agents to autonomously explore environments and refine tool descriptions (action descriptions) and task workflows.

**Key Insight**: Synthesize virtual scenarios involving multi-tool combinations using LLMs, and employ MCTS within these virtual scenarios to explore and optimize action knowledge.

**Core Idea**: Sample subsets from a toolset $\rightarrow$ synthesize multi-step scenarios $\rightarrow$ agent performs trial-and-error using MCTS in virtual scenarios $\rightarrow$ iteratively refine tool descriptions and workflows $\rightarrow$ transfer refined knowledge to real-world tasks.

## Method

### Overall Architecture
**Phase 1: Scenario Synthesis**: Select a subset $t$ from the toolset $T$ $\rightarrow$ LLM generates virtual scenarios containing background $B$ and target $G$ $\rightarrow$ filter out highly similar scenarios. **Phase 2: MCTS Exploration**: Initialize the root node with the initial action knowledge $\rightarrow$ UCB selection $\rightarrow$ LLM-based expansion powered by historical optimization experience (generating a new version of action knowledge) $\rightarrow$ agent execution in virtual scenarios to receive feedback/score $\rightarrow$ backpropagation update $\rightarrow$ iteration.

### Key Designs

1. **Multi-Step Scenario Synthesis**:

    - Function: Generate virtual task scenarios requiring the coordination of multiple tools.
    - Mechanism: Select 2-4 tools as a group $\rightarrow$ LLM generates 2-3 scenarios per group using few-shot prompting $\rightarrow$ deduplication (cosine similarity $< \epsilon$).
    - Design Motivation: Single-tool scenarios cannot learn coordination workflows between tools; "gold tools" annotation renders evaluation more reliable.

2. **MCTS Action Knowledge Exploration**:

    - Function: Systematically explore the optimization direction of action knowledge within a tree search framework.
    - Mechanism: Node = a version of action knowledge $\rightarrow$ Expansion = LLM generates a new version based on historical optimization experience $\mathcal{E}$ (pre- and post-optimization scores + modification content) $\rightarrow$ Evaluation = agent executes with the new action knowledge (AK) in virtual scenarios to obtain scores $\rightarrow$ Backpropagation to the root node to update UCB values.
    - Design Motivation: MCTS outperforms linear iteration in exploration—UCB balances exploration and exploitation, preventing premature convergence to local optima.

3. **Bi-directional Action Knowledge Optimization**:

    - Function: Simultaneously optimize action descriptions (single-tool descriptions) and cognitive workflows (multi-tool workflows).
    - Mechanism: During optimization, the LLM analyzes failure trajectories in virtual scenarios to determine whether the issue is inaccurate descriptions or unreasonable workflows $\rightarrow$ targeted modifications.
    - Design Motivation: Descriptions and workflows represent two complementary levels of agent action understanding, requiring bi-directional alignment.

## Key Experimental Results

### Main Results

| Model | Method | ToolBench PASS | ToolBench WIN | HotpotQA |
|------|------|-------------|-------------|----------|
| GPT-4-turbo | ReAct | 50.67 | 67.00 | 54.61 |
| GPT-4-turbo | Self-Refine | 56.80 | 73.00 | 55.85 |
| GPT-4-turbo | DRAFT | 54.83 | 72.00 | 57.71 |
| GPT-4-turbo | **SynWorld** | **59.33** | **73.00** | **59.93** |

### Ablation Study

| Configuration | ToolBench PASS |
|------|---------------|
| SynWorld (Full) | **59.33** |
| w/o MCTS (Linear Optimization) | 55.20 |
| w/o Multi-step Scenarios (Single-step) | 53.80 |
| w/o workflow optimization | 56.10 |

### Key Findings
- **MCTS outperforms linear optimization by over 4 percentage points**: Structured exploration is more effective than blind iteration.
- **Multi-step scenarios are more valuable than single-step scenarios**: Multi-step scenarios force the agent to learn coordination among tools.
- **Knowledge learned in virtual scenarios transfers to real tasks**: Performance improves on both ToolBench and HotpotQA.

## Highlights & Insights
- **The integration of virtual scenario synthesis and MCTS exploration is an elegant paradigm for agent autonomous learning**: It bypasses the need for manual annotations or feedback from real environments, enabling agents to self-optimize within "imagined" scenes. This is transferable to tool learning in other agent systems.
- **Employing MCTS for meta-optimization (optimizing knowledge representation rather than direct action) is a key innovation**: Traditional MCTS is used for planning action sequences, whereas here it is leveraged to search for the optimal knowledge representation.

## Limitations & Future Work
- **Domain gap between virtual and real scenarios**: Synthesized scenarios may fail to cover the full complexity of real-world scenarios.
- **High cost of MCTS exploration**: Each node expansion and evaluation requires the agent to execute complete tasks.
- **Moderate improvement margin**: Compared to DRAFT, the improvement is approximately 5 percentage points.

## Related Work & Insights
- **vs EasyTool (Yuan et al., 2024)**: EasyTool optimizes tool descriptions using single-step scenarios, whereas SynWorld is more comprehensive by employing multi-step scenarios combined with MCTS.
- **vs DRAFT (Qu et al., 2024)**: DRAFT relies on linear iterative optimization, while SynWorld utilizes MCTS to avoid local optima.

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm of virtual scenarios + MCTS meta-optimization is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets + multiple models + ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear system architecture and complete formal definitions.
- Value: ⭐⭐⭐⭐ Methodological contributions to agent tool learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning](table_critic_multi_agent.md)
- [\[ICML 2025\] KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search](../../ICML2025/llm_agent/kbqa-o1_agentic_knowledge_base_question_answering_with_monte_carlo_tree_search.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)
- [\[ACL 2025\] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](os_genesis_gui_agent_trajectory.md)
- [\[ACL 2025\] Explorer: Scaling Exploration-Driven Web Trajectory Synthesis for Multimodal Web Agents](explorer_scaling_exploration-driven_web_trajectory_synthesis_for_multimodal_web_.md)

</div>

<!-- RELATED:END -->
