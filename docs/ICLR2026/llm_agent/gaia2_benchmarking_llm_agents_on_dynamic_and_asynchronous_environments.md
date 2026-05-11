---
title: >-
  [Paper Note] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments
description: >-
  [ICLR 2026 (Oral)][LLM Agent][dynamic environments] This paper introduces the Gaia2 benchmark for evaluating LLM agents in dynamic and asynchronous environments. It incorporates realistic scenarios including time constra…
tags:
  - "ICLR 2026 (Oral)"
  - "LLM Agent"
  - "dynamic environments"
  - "asynchronous interaction"
  - "benchmark"
  - "reinforcement learning"
date: 2026-05-08
content_hash: 0fe274aab0c88911
---

# Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments

**Conference**: ICLR 2026 (Oral)  
**arXiv**: [2602.11964](https://arxiv.org/abs/2602.11964)  
**Code**: Built on the Agents Research Environments (ARE) platform; open-source  
**Area**: LLM Agent Evaluation  
**Keywords**: LLM Agent, dynamic environments, asynchronous interaction, benchmark, reinforcement learning

## TL;DR

This paper introduces the Gaia2 benchmark for evaluating LLM agents in dynamic and asynchronous environments. It incorporates realistic scenarios including time constraints, noisy events, ambiguity resolution, and multi-agent collaboration. A write-action verifier with verifiable rewards enables direct use for RLVR training. Evaluation results show that the strongest model, GPT-5 (high), achieves only 42% pass@1.

## Background & Motivation

Current LLM agent evaluation suffers from a fundamental flaw: most benchmarks rely on **static** or **synchronous** environments, in which the environment does not evolve independently of the agent's actions. The agent retains full temporal control, can pause and deliberate at will, and the environment always waits for the next action.

Real-world task environments, however, are fundamentally different:
- **Time sensitivity**: Flight prices fluctuate, inventory changes, and deadlines approach.
- **Asynchronous events**: New messages arrive and state updates occur independently.
- **Noise and ambiguity**: Information is incomplete or contradictory, and requirements need clarification.
- **Multi-party collaboration**: Coordination with other agents or humans is required.

Existing benchmarks (e.g., the original GAIA) test only static question answering and tool invocation, and cannot assess agent capabilities along these **real-world dimensions**. This creates a severe **sim-to-real gap** — strong benchmark performance does not predict real-world deployment behavior.

Gaia2 is designed to provide a more realistic evaluation platform while maintaining quantifiability and reproducibility.

## Method

### Overall Architecture

Gaia2 is built on top of a consumer environment and is implemented on the open-source Agents Research Environments (ARE) platform. Each evaluation scenario comprises:
- A **dynamic environment** that evolves independently of agent actions
- A **task description** specifying the goal the agent must achieve within the environment
- A **write-action verifier** that provides fine-grained assessment of agent correctness at each critical action point

### Key Designs

1. **Dynamic Asynchronous Environment**:

   Unlike the request–response paradigm of traditional benchmarks, the Gaia2 environment **runs continuously**. Environment state evolves as simulated time progresses, and new information arrives asynchronously. Agents must:
   - Make decisions within time windows (or miss opportunities)
   - Monitor environmental changes and adjust strategies accordingly
   - Handle unexpected events and state transitions

   This design forces agents to decide under uncertainty, testing adaptive capabilities beyond simple planning.

2. **Multi-Dimensional Capability Testing**:

   Gaia2 scenarios are designed to cover multiple core capability dimensions:
   - **Time-sensitive decision making**: Selecting optimal actions under time constraints
   - **Noise robustness**: Extracting key facts from incomplete or contradictory information
   - **Ambiguity resolution**: Proactively seeking clarification or selecting the most reasonable interpretation among multiple plausible ones
   - **Multi-agent collaboration**: Exchanging information and coordinating actions with other agents
   - **Environmental adaptation**: Responding to dynamic changes and revising plans accordingly

3. **Write-Action Verifier**:

   This is one of the most important technical innovations in Gaia2. While traditional benchmarks typically evaluate only the final answer, Gaia2 assesses **every critical action** taken by the agent throughout the task.

   - Each scenario defines a set of "write-action" checkpoints.
   - At each checkpoint, the verifier assesses whether the agent's action is correct.
   - Evaluation granularity is refined from "final answer correctness" to "decision quality at each step."

   More importantly, this verifiable reward signal enables Gaia2 to be **used directly for reinforcement learning training** via RLVR (Reinforcement Learning from Verifiable Rewards), providing infrastructure for a closed loop from benchmarking to training.

4. **Scalable Architecture Based on the ARE Platform**:

   Gaia2 is built on the open-source ARE (Agents Research Environments) framework and is designed for extensibility:
   - New scenarios can be added through a standard interface
   - Environment logic and verification logic are decoupled
   - Multiple agent frameworks are supported
   - Consumer environments (e.g., shopping, travel planning) are aligned with everyday applications

### Evaluation Protocol

- **Primary metric**: pass@1 (single-attempt pass rate)
- **Fine-grained analysis**: Performance profiles decomposed by capability dimension
- **Efficiency metrics**: Trade-off between task completion speed and API call cost

## Key Experimental Results

### Main Results: Overall Model Performance

| Model | pass@1 | Type | Notable Characteristics |
|-------|--------|------|------------------------|
| GPT-5 (high) | 42% | Closed-source | Overall strongest; weak on time-sensitive tasks |
| Claude-4 Sonnet | ~35–38% | Closed-source | Balanced accuracy and speed; better cost efficiency |
| Kimi-K2 | 21% | Open-source | Best among open-source models |
| Other open-source models | <20% | Open-source | Significantly behind closed-source models |

### Capability Dimension Analysis

| Capability Dimension | GPT-5 | Claude-4 | Kimi-K2 | Notes |
|----------------------|-------|----------|---------|-------|
| Time-sensitive decision making | Weak | Moderate | Weak | Most challenging dimension |
| Noise robustness | Strong | Strong | Moderate | Closed-source models show clear advantage |
| Ambiguity resolution | Strong | Moderate | Weak | Requires strong reasoning ability |
| Multi-agent collaboration | Moderate | Moderate | Weak | Weak point across all models |
| Environmental adaptation | Moderate | Moderate | Weak | Ability to dynamically revise plans |

### Ablation Study

| Comparison Dimension | Key Finding |
|----------------------|-------------|
| Static vs. dynamic environment | All models show significant performance drops in dynamic environments |
| Synchronous vs. asynchronous | Asynchronous events further widen performance gaps between models |
| Single-agent vs. multi-agent | Multi-agent scenarios are the current largest bottleneck |
| Without vs. with time constraints | Time constraints have a greater negative impact on open-source models |

### Key Findings

1. **No model dominates across all dimensions**: GPT-5 achieves the best overall performance but fails on time-sensitive tasks; Claude-4 offers better cost efficiency.
2. **42% pass@1 reveals a substantial gap**: Even the strongest model fails on nearly 60% of scenarios, demonstrating that real-world agent tasks remain extremely challenging.
3. **Open-source vs. closed-source divide**: The gap of 21% vs. 42% indicates that open-source models remain substantially less capable in agent scenarios.
4. **The sim-to-real gap is real**: Models that perform similarly on static benchmarks show amplified differences in Gaia2's dynamic environments.
5. **Potential of RLVR**: The fine-grained reward signal provided by the write-action verifier opens a path toward reinforcement learning-based agent training.

## Highlights & Insights

- **Paradigm shift from "question answering" to "acting"**: Gaia2 evaluates not agents' knowledge or reasoning, but their ability to take correct actions in dynamic environments.
- **The write-action verifier is the key innovation**: It enables the benchmark to serve both evaluation and training purposes simultaneously, greatly enhancing its practical value.
- **Asynchrony is an overlooked core challenge**: Virtually all existing agent systems assume synchronous interaction; Gaia2 is the first to systematically evaluate asynchronous scenarios.
- **ICLR 2026 Oral recognition reflects the field's urgency**: Selection as an oral presentation signals the community's pressing need for realistic agent evaluation.
- **Ecosystem value of the open-source ARE platform**: Gaia2 is not merely a benchmark but a sustainable and extensible research infrastructure.

## Limitations & Future Work

1. **Consumer environments may not generalize to all domains**: Scenarios such as shopping and travel planning differ considerably from agent requirements in scientific research, software engineering, and other professional domains.
2. **Reproducibility challenges in dynamic environments**: The stochasticity of dynamic environments may cause result fluctuations across different evaluation runs.
3. **Manual effort required for write-action verifier design**: Checkpoints and correctness criteria must be manually defined for each scenario, limiting automated scalability.
4. **Insufficient testing of tool-use capabilities**: Although the environments are dynamic, the complexity of the tool set and API interfaces may not be sufficient.
5. **Limited scale of multi-agent scenarios**: Current scenarios likely involve primarily two agents; larger-scale collaboration settings remain to be developed.

## Related Work & Insights

- **Inheritance from GAIA (2023)**: Gaia2 builds on its predecessor by introducing dynamic evolution and asynchrony as two qualitatively distinct new dimensions.
- **Distinction from WebArena and AgentBench**: These benchmarks focus on static web interaction or API invocation, whereas Gaia2 emphasizes temporal evolution of the environment.
- **Complementary to SWE-bench**: SWE-bench tests code generation ability, while Gaia2 evaluates environment interaction and decision-making capability.
- **Implications for agent training methods**: The RLVR-ready design positions Gaia2 as a potential key data source for training stronger agents.
- **Implications for agent architecture design**: Future architectures should incorporate time-awareness, asynchronous event-handling modules, and dynamic plan revision mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Dynamic asynchronous agent evaluation + RLVR-ready design; field-leading contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers mainstream models, though the number of scenarios is not specified)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured with clear analysis)
- Value: ⭐⭐⭐⭐⭐ (An important milestone in agent evaluation; Oral acceptance is well deserved)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[AAAI 2026\] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](../../AAAI2026/llm_agent/llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](../../AAAI2026/llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)
- [\[AAAI 2026\] AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments](../../AAAI2026/llm_agent/agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)
- [\[NeurIPS 2025\] DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments](../../NeurIPS2025/llm_agent/defenderbench_a_toolkit_for_evaluating_language_agents_in_cybersecurity_environm.md)

</div>

<!-- RELATED:END -->
