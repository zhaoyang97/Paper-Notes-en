---
title: >-
  [Paper Note] R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory
description: >-
  [ACL 2025][LLM Agent][web agent] R2D2 proposes a Web Agent framework that integrates the Remember (experience replay buffer + A* search navigation) and Reflect (error reflection + reflective memory storage) paradigms. It transforms web navigation from an Unknown MDP to a Known MDP, reducing navigation errors by 50% and tripling the task completion rate on WebArena, outperforming the SOTA by 17%.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "web agent"
  - "replay buffer"
  - "reflective memory"
  - "A* search"
  - "WebArena"
  - "known MDP"
date: 2026-05-08
content_hash: 2c1f7d89e29fa4dc
---

# R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory

**Conference**: ACL 2025  
**arXiv**: [2501.12485](https://arxiv.org/abs/2501.12485)  
**Code**: None  
**Area**: LLM Agent / Web Agent  
**Keywords**: web agent, replay buffer, reflective memory, A* search, WebArena, known MDP

## TL;DR
R2D2 proposes a Web Agent framework that integrates the Remember (experience replay buffer + A* search navigation) and Reflect (error reflection + reflective memory storage) paradigms. It transforms web navigation from an Unknown MDP to a Known MDP, reducing navigation errors by 50% and tripling the task completion rate on WebArena, outperforming the SOTA by 17%.

## Background & Motivation

**Background**: Web Agents (such as ReACT) utilize LLMs to execute web navigation and interaction tasks, but approximately 60% of failures stem from navigation errors—the Agent being unable to find the correct target page.

**Limitations of Prior Work**:
   - **Unknown MDP Assumption**: Agents have limited visibility into the consequences of their actions, exploring from scratch in each reasoning episode without leveraging historical experiences.
   - **Disposable Experience**: Traditional methods discard trajectories immediately after a single episode, wasting valuable exploration information.
   - **Insufficient Reflection**: Existing reflection methods (such as Reflexion) focus only on execution-level errors, remaining ineffective against navigation failures.

**Key Challenge**: Web environments are complex, but the agent starts its exploration from scratch each time, meaning historical experience cannot be effectively reused.

**Goal**: Construct a structured representation ("map") of the web environment to allow the Agent to make decisions based on known information.

**Key Insight**: Inspired by research on human cognition and robotic exploration—humans iteratively improve their strategies through memory and reflection rather than starting from scratch each time.

**Core Idea**: Use a replay buffer to construct a directed graph "map" (Known MDP) of the web environment, replacing blind navigation with A* search; use a reflection mechanism to learn from execution errors, storing corrected trajectories in memory for future retrieval.

## Method

### Overall Architecture
**Exploration Phase**: ReACT Agent executes tasks $\rightarrow$ collects observation sequences $\rightarrow$ constructs a replay buffer graph $\rightarrow$ classifies failed trajectories (navigation/execution errors) $\rightarrow$ corrects them using Remember/Reflect $\rightarrow$ stores them in reflective memory.  
**Inference Phase**: Encodes new queries $\rightarrow$ retrieves relevant trajectories as in-context demos $\rightarrow$ guides agent execution.

### Key Designs

1. **Remember Paradigm**:

    - Function: Constructs a structured "map" of the web environment from all historical observations.
    - **Replay Buffer Construction**: Represents the web environment as a directed graph $G = (O, E)$ where nodes represent webpage observations and edges represent actions (clicking, typing, etc.), storing differences between consecutive observations instead of full page states.
    - **A* Search Navigation**: For navigation-failed trajectories, A* search is executed on the replay buffer graph to find the optimal path to the target page. The heuristic function is evaluated by an LLM to estimate the relevance distance from each node to the target.
    - Design Motivation: Converts an Unknown MDP into a Known MDP—the Agent no longer navigates "in the dark" but plans paths on a known "map."

2. **Reflect Paradigm**:

    - Function: Performs error diagnosis and strategy correction on execution-failed trajectories.
    - Mechanism: LLM identifies the first erroneous action $a_i$ in the trajectory $\rightarrow$ truncates to $\{a_1, ..., a_{i-1}\}$ (the correct prefix) $\rightarrow$ generates reflections and correction suggestions for the erroneous action $\rightarrow$ stores them in reflective memory.
    - Design Motivation: Complements Remember—Remember addresses "where to go," while Reflect addresses "how to do it."

3. **Reflective Memory**:

    - Function: A key-value store, where encoded query vectors serve as keys and corrected trajectories + reflections serve as values.
    - Lookup: Encodes new queries and retrieves the most relevant historical trajectories via vector similarity.
    - Update: If a newly corrected trajectory is superior to existing ones, the LLM evaluates and updates it.
    - Design Motivation: Enables the Agent to continuously improve by accumulating experiences, similar to human learning.

4. **Error Classification**:

    - **Navigation Failure**: The agent fails to reach the critical page $\rightarrow$ corrected using Remember (A* search).
    - **Execution Failure**: The agent reaches the correct page but executes incorrect operations $\rightarrow$ corrected using Reflect.
    - Approximately 60% of failures are navigation failures, which is precisely what the Remember paradigm is designed to solve.

## Key Experimental Results

### Main Results (WebArena)

| Method | Task Success Rate | Navigation Error Rate |
|------|----------|----------|
| ReACT (GPT-4o) | ~14% | ~60% |
| Tree-search + reflection | ~20% | ~45% |
| **R2D2** | **~42%** | **~30%** |
| Gain vs SOTA | **+17%** | **-50%** |

Comparison across task domains:

| Domain | ReACT | R2D2 | Gain |
|----|-------|------|------|
| CMS | Low | High | Significant |
| Reddit | Low | High | Significant |
| Shopping | Medium | High | Significant |
| Map | Medium | High | Moderate |

### Ablation Study

| Configuration | Task SR |
|------|---------|
| R2D2 (Remember + Reflect) | **~42%** |
| w/o Remember (Reflect only) | ~25% |
| w/o Reflect (Remember only) | ~35% |
| w/o Reflective Memory | ~30% |
| Base ReACT | ~14% |

### Key Findings
- **Remember contributes more than Reflect**: Removing Remember drops performance by 17%, while removing Reflect drops it by 7%—because navigation failures account for 60% of all failures.
- **Synergy of both paradigms exceeds individual sums**: Remember resolves navigation obstacles, allowing Reflect to focus strictly on execution optimization.
- **A* search is more efficient than random exploration**: It accurately locates the target page path within the replay buffer.
- **Continuous learning in reflective memory**: As the number of exploration episodes increases, memory quality improves, benefiting new tasks.

## Highlights & Insights
- **Elegant conversion from Unknown MDP to Known MDP**: Changing web navigation from "walking in the dark" to "map navigation" is a fundamental paradigm shift. This can be transferred to agents in any interactive environment (such as GUI Agents, game Agents).
- **Combination of A* search and LLM heuristics**: Uses classical search algorithms to guarantee structural efficiency and LLMs to provide semantic-level heuristics—an outstanding fusion of classical algorithms and modern AI.
- **Divide-and-conquer strategy for error types**: Instead of applying the same correction method to all failures, R2D2 classifies them (navigation vs. execution) and treats them accordingly. This classification paradigm is widely applicable.

## Limitations & Future Work
- **Replay buffer requires prior exploration**: Multiple exploration episodes are needed to build the "map", causing high cold-start costs.
- **Dynamic nature of web environments**: If the webpage structure changes, the old information in the replay buffer may become obsolete.
- **Only tested on WebArena (simulated environment)**: Real-world web environments feature higher noise and complexity.
- **High volume of LLM calls**: A* search requires LLM evaluation of heuristic values at each node.

## Related Work & Insights
- **vs Reflexion (Shinn et al., 2023)**: Reflexion only performs execution-level reflection, whereas R2D2 decouples and separately handles navigation and execution failures.
- **vs Tree-search methods (Koh et al., 2024)**: Tree-search methods explore online in an Unknown MDP, whereas R2D2 builds a Known MDP offline for efficient search.
- **vs Agent-Q (Putta et al., 2024)**: Agent-Q uses RL fine-tuning, whereas R2D2's pure prompting approach is more flexible.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Unknown $\rightarrow$ Known MDP conversion, A* + LLM heuristics, and error divide-and-conquer strategy are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation and error analysis on WebArena, though evaluated on only one benchmark.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions and intuitive diagrams.
- Value: ⭐⭐⭐⭐⭐ Significant practical boost for Web Agents (3x success rate), with highly transferable concepts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LLM Agents Making Agent Tools](llm_agents_making_agent_tools.md)
- [\[ICLR 2026\] Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning](../../ICLR2026/llm_agent/language_agents_for_hypothesis-driven_clinical_decision_making_with_reinforcemen.md)
- [\[ICML 2026\] Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach](../../ICML2026/llm_agent/post-training_llms_as_better_decision-making_agents_a_regret-minimization_approa.md)
- [\[ICLR 2026\] LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities](../../ICLR2026/llm_agent/llms_are_greedy_agents_effects_of_rl_fine-tuning_on_decision-making_abilities.md)
- [\[ICLR 2026\] Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory](../../ICLR2026/llm_agent/seeing_listening_remembering_and_reasoning_a_multimodal_agent_with_long-term_mem.md)

</div>

<!-- RELATED:END -->
