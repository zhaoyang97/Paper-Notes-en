---
title: >-
  [Paper Note] MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines
description: >-
  [ICML 2025][Multi-Agent][Multi-Agent System] Proposes MetaAgent, a framework based on Finite State Machines (FSMs) that automatically designs multi-agent systems given only task descriptions, without requiring external training data. Supporting tool invocation and state backtracking, it outperforms existing automated design methods and approaches the performance of hand-crafted systems across text-based, ML, and software development tasks.
tags:
  - "ICML 2025"
  - "Multi-Agent"
  - "Multi-Agent System"
  - "Finite State Machine"
  - "LLM Agent"
  - "Automated Design"
  - "Tool Integration"
date: 2026-05-08
content_hash: 79157aff2e5f3ae7
---

# MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines

**Conference**: ICML 2025  
**arXiv**: [2507.22606](https://arxiv.org/abs/2507.22606)  
**Code**: [SaFoLab-WISC/MetaAgent](https://github.com/SaFoLab-WISC/MetaAgent/)  
**Area**: Optimization  
**Keywords**: Multi-Agent System, Finite State Machine, LLM Agent, Automated Design, Tool Integration

## TL;DR

Proposes MetaAgent, a framework based on Finite State Machines (FSMs) that automatically designs multi-agent systems given only task descriptions, without requiring external training data. Supporting tool invocation and state backtracking, it outperforms existing automated design methods and approaches the performance of hand-crafted systems across text-based, ML, and software development tasks.

## Background & Motivation

Existing multi-agent systems face two core challenges:

**High Hand-Crafting Costs**: Systems like MetaGPT and ChatDev require significant human effort to implement complex codebases and are limited to specific scenarios, resulting in poor generalization.

**Limitations of Prior Work**:
   - SPP, AutoAgents, and EvoAgent design systems for each specific case individually, lacking generalization.
   - SPP does not support tool use.
   - ADAS and Symbolic Learning rely heavily on large external datasets and iterative training steps.
   - All existing methods employ rigid communication structures (e.g., linear, debate, coordinator), lacking state backtracking capabilities and making it difficult to correct previous steps when errors occur.

| Feature | MetaGPT | AutoAgents | SPP | EvoAgent | ADAS | Symbolic | **MetaAgent** |
|------|---------|------------|-----|----------|------|----------|---------------|
| Automated Design | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| Generalization | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | **✓** |
| Tool Support | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | **✓** |
| Backtracking | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| No External Data | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | **✓** |

MetaAgent is the only framework that simultaneously satisfies all five key characteristics.

## Method

### Overall Architecture

MetaAgent models a multi-agent system as a Finite State Machine (FSM) $\mathcal{M} = (\Sigma, S, s_0, F, \delta)$:

- $\Sigma$: Input alphabet, representing the set of specific cases in the task domain.
- $S$: Finite set of states.
- $s_0 \in S$: Initial state.
- $F \subseteq S$: Set of final (accepting) states.
- $\delta$: State transition function.

Each state consists of four core components:
1. **Task-Solving Agent**: The agent responsible for executing the current subtask.
2. **State Instruction**: Natural language instructions describing the subtask to be completed in this state.
3. **Condition Verifier**: Checks whether the output satisfies the state transition conditions.
4. **Listeners**: Downstream agents that receive the output of the current state.

The framework operates in two phases: the **Construction Phase** (automated FSM generation) and the **Deployment Phase** (FSM-controlled execution).

### Key Designs

#### 1. Agent Design (Construction Phase - Step 1)

Upon receiving the task description, the Designer LLM:
- Generates comprehensive task analysis and system objectives.
- Designs the most minimal yet effective set of agents (to reduce costs).
- Outputs a structured JSON configuration for each agent, containing its name, system prompts, and assigned tools (e.g., code interpreter, search engine).

#### 2. State and Transition Condition Design (Construction Phase - Step 2)

The Designer LLM acts as a "foresightful planner" to construct the FSM:
- Anticipates various scenarios that may arise during task execution (e.g., different input types, intermediate result discrepancies, diverse outputs).
- Defines for each state: State Instruction, Assigned Agent, Condition Verifier, and Listeners.
- Transition conditions are **defined in natural language** and evaluated by the LLM acting as the Condition Verifier.

**Generality of FSMs**: Other multi-agent communication structures are constrained special cases of FSMs:
- Linear structures = FSMs with only one transition per state (no backtracking, no verifiers).
- Decentralized debate = FSMs that only retract from the end back to the beginning.
- Coordinator pattern = FSMs with shared verifiers.

#### 3. FSM Optimization Algorithm (Optimization Phase)

Initially designed FSMs often contain redundant states and excessively long propagation chains. The optimization algorithm:
- Traverses every pair of states in the FSM.
- Uses the LLM to judge whether any two states can be merged.
- Eliminates redundant states, shortening the execution pipeline.
- **Requires no external data** and **no iterative training**.

#### 4. Deployment and Execution (Deployment Phase)

Starting from the initial state $s_0$:
1. User query + current state instruction $\rightarrow$ executed by the Task-Solving Agent.
2. Agent output $\rightarrow$ validated by the Condition Verifier.
3. If transition conditions are met $\rightarrow$ transitions to the next state (**with the ability to backtrack to previously visited states**).
4. Saves the output as memory for downstream Listeners prior to transitioning.
5. Halts upon reaching a final state $F$ or exceeding the maximum number of transitions.

**State backtracking** is a key advantage: for example, in software development tasks, if the Product Manager Agent detects incorrect information, the FSM can backtrack to the information-gathering state and re-execute.

### Loss & Training

MetaAgent does not rely on traditional loss functions or gradient training. Its "optimization" is achieved via an LLM-driven state-merging algorithm:
- Input: Initially constructed FSM (which may contain redundant states).
- Process: For each state pair $(s_i, s_j)$, the LLM determines whether their functionalities overlap or can be merged.
- Output: Pruned FSM.
- Core Idea: Reducing state count $\rightarrow$ shortening the execution pipeline $\rightarrow$ mitigating error propagation $\rightarrow$ improving robustness.

## Key Experimental Results

### Main Results

Experiments cover three types of tasks: text-based reasoning tasks, machine learning tasks, and software development tasks.

| Task Type | Metric | MetaAgent | Prev. SOTA (Automated) | Prev. SOTA (Manual) | Gain |
|----------|------|-----------|-----------------|-----------------|------|
| Text-Based (Creative Writing + GPQA) | Accuracy | SOTA | Prev. prompt-based SOTA | — | +9% |
| ML Bench | Avg. Performance | 97% of best | Other Auto/Manual Methods | Best Manual System | Outperforms all other frameworks |
| Software Development | Passed Checkpoints | +50% | — | Hand-crafted System | +50% |

### Ablation Study

| Configuration | Change in Key Metrics | Description |
|------|-------------|------|
| W/o Tool Use | Performance decline | Agents cannot interact with the external environment, limiting their capabilities |
| W/o FSM Optimization | Performance decline | Redundant states lead to excessively long information propagation chains, causing error accumulation |
| W/o State Backtracking | Performance decline | Unable to correct previous steps when encountering errors, equivalently degrading to a linear structure |

Consistent performance degradation was observed across all task types when any of the three components were removed, validating the necessity of tool integration, FSM optimization, and state backtracking mechanisms.

### Key Findings

1. **Automated design can approach manual design**: MetaAgent achieves 97% of the performance of the best hand-crafted system on ML Bench and even outperforms manual systems by 50% in software development.
2. **FSM structure outperforms rigid communication structures**: Customized condition verifiers coupled with unconstrained state transitions yield maximum flexibility.
3. **State backtracking is a key differentiator**: MetaAgent is the only automated design framework equipped with backtracking capabilities.
4. **Optimization without external data is viable**: The LLM-driven state-merging algorithm operates both effectively and efficiently.

## Highlights & Insights

1. **FSM as a Unified Paradigm**: Unifying multi-agent communication structures under the FSM framework reveals that linear, debate, and coordinator structures are all constrained special cases of FSMs, making this an elegant theoretical contribution.
2. **Natural Language Transition Conditions**: Utilizing an LLM instead of hard-coded string matching for state transition decisions significantly enhances adaptability to complex scenarios.
3. **"Meta-Design" Philosophy**: Implementing a Designer LLM to architect the entire multi-agent system realizes automation at a meta-level.
4. **Lightweight Optimization**: Eliminating the need for heavy iterations and external datasets (as required by ADAS/Symbolic Learning) lowers the deployment barrier.
5. **Tool Integration**: Combining code interpreters and search engines empowers agents to resolve real-world tasks.

## Limitations & Future Work

1. **Dependency on the Designer LLM**: The quality of FSM design strictly relies on the LLM's planning capabilities; weaker models may output low-quality FSMs.
2. **Granularity Control in State Merging**: The current pairwise traversal strategy leads to $O(|S|^2)$ time complexity, reducing efficiency as the state count increases.
3. **Robustness of Natural Language Conditions**: Employing an LLM as a Condition Verifier might yield inaccurate decisions in edge cases.
4. **Lack of Online Adaptive Learning**: Once constructed, the FSM topology remains static and cannot be dynamically adjusted based on online execution feedback.
5. **Limited Evaluation Scope**: Primarily validated across text, ML, and software development tasks; generalized capability in other domains (e.g., multimodal, scientific research tasks) warrants further evaluation.

## Related Work & Insights

- **MetaGPT / ChatDev**: Hand-crafted software-development multi-agent systems that are functional but limited in generalization.
- **ADAS / Symbolic Learning**: Automated design methods based on self-iteration, but restricted by their reliance on external data.
- **SPP**: A prompt-based automated method that designs systems on a per-case basis and lacks tool support.
- **CodeAct**: The concept of treating code as agent actions inspired MetaAgent's tool integration.
- **FSM Applications in Agents**: Prior studies (e.g., AgentLite, StateFlow) used hard-coded FSMs to control agents, whereas MetaAgent automates the FSM design process itself.

## Rating

- **Novelty**: ★★★★☆ — The unified FSM paradigm and automated design represent clean and clear innovations.
- **Technical Depth**: ★★★☆☆ — The methodology is essentially prompt engineering combined with LLM-based decision making, lacking deep technical contributions.
- **Experimental Thoroughness**: ★★★★☆ — Covers multiple task categories and features ablation studies, though comparisons across different LLM backbones are lacking.
- **Value**: ★★★★☆ — Lowers the barrier to constructing multi-agent systems and provides open-source code.
- **Writing Quality**: ★★★★☆ — The problem definition is clear, the motivation is sound, and the presentation is clean.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GlobeDiff: State Diffusion Process for Partial Observability in Multi-Agent Systems](../../ICLR2026/multi_agent/globediff_state_diffusion_process_for_partial_observability_in_multi-agent_syste.md)
- [\[ICML 2025\] AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML](automl-agent_a_multi-agent_llm_framework_for_full-pipeline_automl.md)
- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](../../NeurIPS2025/multi_agent/maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[CVPR 2025\] ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](../../CVPR2025/multi_agent/comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)
- [\[NeurIPS 2025\] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](../../NeurIPS2025/multi_agent/masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)

</div>

<!-- RELATED:END -->
