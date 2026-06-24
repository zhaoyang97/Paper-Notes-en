---
title: >-
  [Paper Note] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration
description: >-
  [CVPR 2025][Multi-Agent][Embodied AI] This paper proposes the Cooperative Tree Search (CoTS) framework, which integrates a modified Monte Carlo Tree Search with an LLM-driven reward function to guide multiple embodied agents in long-term strategic planning and highly efficient collaboration. By incorporating a plan evaluation module to prevent action confusion caused by frequent plan updates, CoTS significantly outperforms existing methods in both CWAH and TDW-MAT environment…
tags:
  - "CVPR 2025"
  - "Multi-Agent"
  - "Embodied AI"
  - "Collaborative Planning"
  - "Monte Carlo Tree Search"
  - "LLM"
  - "Task Allocation"
date: 2026-05-08
content_hash: e35dee1623b7d200
---

# Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration

**Conference**: CVPR 2025  
**Code**: To be confirmed  
**Area**: Robotics  
**Keywords**: Embodied AI, Multi-Agent, Collaborative Planning, Monte Carlo Tree Search, LLM, Task Allocation

## TL;DR
This paper proposes the Cooperative Tree Search (CoTS) framework, which integrates a modified Monte Carlo Tree Search with an LLM-driven reward function to guide multiple embodied agents in long-term strategic planning and highly efficient collaboration. By incorporating a plan evaluation module to prevent action confusion caused by frequent plan updates, CoTS significantly outperforms existing methods in both CWAH and TDW-MAT environments.

## Background & Motivation

**Background**: Embodied LLM agents have developed rapidly in recent years, with single agents achieving promising results in tasks such as navigation and manipulation. Multi-agent collaboration scenarios are closer to real-world demands (e.g., multi-robot home service, warehouse collaboration) but impose higher requirements on communication efficiency and task allocation.

**Limitations of Prior Work**: (1) **Simple communication patterns**: Most existing multi-agent methods adopt simple turn-taking dialogues or broadcast communication, which are prone to information redundancy and inconsistency. (2) **Propagation of behavioral errors**: An agent's erroneous behavior or unreasonable plan can propagate to other agents through communication, leading to chain errors. (3) **Lack of long-term planning**: Existing methods mostly rely on reactive decision-making, lacking global strategic planning capabilities for long-horizon tasks. (4) **Unstable plan updates**: Frequently updating plans causes action confusion among agents, whereas not updating leads to the execution of outdated plans.

**Key Challenge**: Multi-agent collaboration requires both flexible real-time communication to handle dynamic environments and stable long-term planning to avoid action confusion. Simple communication modalities fail to strike a balance between the two.

**Goal**: How to enable multiple LLM-driven embodied agents to perform efficient collaborative planning in complex long-term tasks while avoiding error propagation and action confusion.

**Key Insight**: Drawing on the success of Monte Carlo Tree Search (MCTS) in gaming and planning, multi-agent collaborative planning is modeled as a tree search problem. LLM-driven reward functions are leveraged to evaluate the quality of different collaborative schemes, searching for the optimal cooperation strategy within a tree structure.

**Core Idea**: Systematically organize multi-agent discussion and collaborative planning using an MCTS framework, search for the most promising cooperation plans via LLM-driven reward functions, and control the frequency of plan updates with a plan evaluation module.

## Method

### Overall Architecture
CoTS consists of three core components: (1) a modified MCTS module that organizes multi-agent collaborative discussion and strategy search; (2) an LLM-driven reward function that evaluates the feasibility and expected gains of different cooperation plans; and (3) a plan evaluation module that decides whether the currently executed plan needs to be updated. The overall workflow operates as follows: agents first generate and evaluate multiple candidate collaborative plans using tree search, choose the optimal plan for execution, and determine if re-planning is necessary during execution through the plan evaluation module.

### Key Designs

1. **Modified MCTS for Cooperation**

    - **Function**: Model multi-agent collaborative planning as a tree search process to systematically explore the space of cooperation strategies.
    - **Mechanism**: Each node in the tree represents a collaborative state (including the current task allocation of each agent and the environment state), and edges represent potential collaborative decisions (such as task reallocation or communication content). Unlike standard MCTS, the expansion and simulation steps here are driven by the LLM, leveraging its commonsense reasoning capabilities to generate reasonable collaborative candidates rather than expanding randomly. The selection phase utilizes the UCB formula to balance exploration and exploitation.
    - **Design Motivation**: Compared to simple dialogue-based collaboration, tree search systematically covers a larger strategy space to avoid local optima. The LLM-driven expansion ensures the rationality of the generated plans, avoiding the inefficient random search of traditional MCTS in large action spaces.

2. **LLM-Driven Reward Functions**

    - **Function**: Evaluate the quality of each candidate collaborative plan.
    - **Mechanism**: Prompt the LLM to evaluate collaborative plans from multiple dimensions: expected task completion efficiency, rationality of division of labor, potential conflict risks, resource utilization, etc. The reward signal is backpropagated to update the estimated values of nodes in the tree (similar to MCTS backpropagation).
    - **Design Motivation**: Traditional reward functions struggle to capture subtle nuances in complex collaborative scenarios (e.g., "two agents going to pick up the same object simultaneously" is problematic). The semantic comprehension of LLMs can better evaluate the rationality of collaborative schemes.

3. **Plan Evaluation Module**

    - **Function**: Control the frequency of plan updates to balance stability and adaptability.
    - **Mechanism**: Evaluate the executability and applicability of the current plan at each timestep. Re-planning is triggered only when the current plan becomes clearly inapplicable (e.g., significant changes in the environment, sub-goals completed, or unexecutable steps detected). It establishes thresholds and evaluation criteria to avoid two extremes: overly frequent updating (action confusion) and never updating (executing outdated plans).
    - **Design Motivation**: Directly addresses the issue in existing methods where frequent plan alterations cause agents to experience action confusion. This design mirrors the human cooperative strategy of "plan-then-execute" rather than "adapting as one goes."

## Key Experimental Results

### Main Results

| Method | CWAH Efficiency ↑ | CWAH Success Rate ↑ | TDW-MAT Efficiency ↑ | TDW-MAT Success Rate ↑ |
|------|-----------|-------------|---------------|----------------|
| ReAct | Baseline | Baseline | Baseline | Baseline |
| RoCo | Medium | Medium | Medium | Medium |
| CoELA | Better | Better | Better | Better |
| **CoTS (Ours)** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Performance |
|------|------|
| CoTS (Full) | Best |
| - w/o MCTS (LLM Dialogue Only) | Significant drop |
| - w/o Plan Evaluation Module (Re-planning at Every Step) | Moderate drop |
| - w/o LLM Reward (Random Reward) | Drastic drop |

### Key Findings
- The improvement of CoTS is most pronounced in long-term, complex tasks, showing that the long-range planning capability of tree search is a core advantage.
- The plan evaluation module effectively reduces unnecessary re-planning occurrences (by approximately 40-60%) while maintaining adaptability to environmental changes.
- In tasks requiring fine-grained division of labor (e.g., multiple items distributed across different rooms), the rationality of CoTS's task allocation is significantly superior to baseline methods.
- The search depth and breadth of MCTS have a substantial impact on final performance, presenting a calculation-performance trade-off.

## Highlights & Insights
- **Introducing MCTS into multi-agent collaborative planning** is a natural yet highly effective innovation. Collaborative planning is essentially a combinatorial optimization problem, and tree search is much more systematic than simple dialogue.
- The **plan evaluation module** addresses a practical but often overlooked problem: when to update a plan. This design reflects deep consideration of real-world deployment scenarios.
- **LLM as a reward function** cleverly leverages the LLM's commonsense reasoning schema, bypassing the difficulty of manually designing complex reward functions.
- The framework possesses strong generalizability and can, in principle, be extended to more agents and more complex collaborative settings.

## Limitations & Future Work
- The computational overhead of MCTS search can be large; real-time performance may be constrained by search budgets.
- LLM-driven reward functions may suffer from bias and instability, as different LLMs might yield different evaluations.
- Validation is restricted to simulation environments (CWAH, TDW-MAT); the sim-to-real gap in real physical environments has not been considered.
- Scalability with respect to the number of agents has not been thoroughly investigated—as the number of agents increases, the branching factor of tree search grows exponentially.
- The threshold setting for the plan evaluation module may need hand-tuning for different tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](../../ACL2026/multi_agent/efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)
- [\[CVPR 2025\] ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)
- [\[ICLR 2026\] AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning](../../ICLR2026/multi_agent/agentpo_enhancing_multi-agent_collaboration_via_reinforcement_learning.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](../../ACL2026/multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)
- [\[CVPR 2025\] NADER: Neural Architecture Design via Multi-Agent Collaboration](nader_neural_architecture_design_via_multi-agent_collaboration.md)

</div>

<!-- RELATED:END -->
