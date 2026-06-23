---
title: >-
  [Paper Note] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems
description: >-
  [ACL 2026][Multi-Agent][Paper Note] TraceElephant advocates that failure attribution in multi-agent systems should be evaluated under full execution traces visible to developers. It provides 220 failed traces with annotations for responsible agents and critical failure steps, demonstrating that full observability improves step-level attribution from 16%
tags:
  - ACL 2026
  - Multi-Agent
date: 2026-05-08
content_hash: fa71010a967a9473
---
# Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems

**Conference**: ACL 2026  
**arXiv**: [2604.22708](https://arxiv.org/abs/2604.22708)  
**Code**: https://github.com/TraceElephant/TraceElephant  
**Area**: LLM Evaluation / Multi-agent Debugging  
**Keywords**: Failure Attribution, Multi-agent Systems, Execution Trace, Observability, Debugging Benchmark

## TL;DR
TraceElephant advocates that failure attribution in multi-agent systems should be evaluated under full execution traces visible to developers. It provides 220 failed traces with annotations for responsible agents and critical failure steps, demonstrating that full observability improves step-level attribution from 16% (output-only) to over 28%-30%.

## Background & Motivation
**Background**: LLM-based multi-agent systems (MAS) are widely used for web search, software engineering, complex task decomposition, and tool invocation. However, these systems involve complex interactions between agents, prompts, tools, and environments, making it difficult to locate the responsible component and the earliest error step upon failure.

**Limitations of Prior Work**: Existing failure attribution benchmarks primarily use partially observable traces that only record agent outputs, omitting input contexts, system prompts, agent configurations, tool logs, and environment states. While suitable for black-box observation, this setup does not align with developer debugging scenarios where full logs are typically accessible.

**Key Challenge**: Failure attribution aims to provide actionable diagnostics, which depend on "what the agent saw, how it was configured, what tools it called, and what the environment returned." Relying solely on output sequences conflates upstream information deficiency with downstream component errors.

**Goal**: To build a developer-facing failure attribution benchmark that provides both complete execution traces and reproducible experimental environments. This allows researchers to evaluate the impact of static trace analysis, dynamic replay, and counterfactual probing on attribution accuracy.

**Key Insight**: The paper uses the metaphor "seeing the whole elephant" to represent complete observability. The authors analyzed 184 failure cases and found that at least 21% cannot be reliably attributed under output-only logs. Consequently, they redesigned the benchmark schema to treat inputs, outputs, tool logs, and system architecture as first-class information.

**Core Idea**: Failure attribution should not merely ask "which output in the log looks wrong," but rather "in the full execution narrative, which component made which decision under what input conditions that rendered failure inevitable."

## Method
TraceElephant is a benchmark rather than a new attribution model. Its contribution lies in defining the task, collecting complete traces, designing annotation protocols, and systematically comparing different observability levels and attribution techniques. The unit of evaluation can be either a distinct agent or a functional component with independent decision-making responsibilities within a single-agent scaffold.

### Overall Architecture
Each sample consists of a complete execution trace of a failed task, a runnable system environment, a responsible agent/component label, and a decisive failure step label. Step-level attribution seeks the earliest step where failure became inevitable; agent-level attribution identifies the component primarily responsible for that step.

Data is sourced from three representative systems: Captain-Agent, Magentic-One, and SWE-Agent. These cover dynamic team assembly, centralized multi-agent orchestration, and single-agent software engineering scaffolds. Tasks originate from GAIA, AssistantBench, and SWE-Bench. A total of 380 traces were collected, with 220 failed traces used for attribution benchmarking.

### Key Designs
**1. Full execution trace schema: Capturing inputs, configurations, and tool logs essential for developers**

Most prior benchmarks only retain agent output sequences, causing "upstream information errors" and "current component reasoning errors" to be indistinguishable. TraceElephant redesigns the trace schema to include metadata and inputs: trace-level records include `task_id`, `task_instruction`, `system_name`, `agent_configuration`, and `system_architecture`. Step-level records include input fields (`step_id`, `agent_id`, `agent_name`, `input_context`) and output fields (`output_content`, `tool_logs`). `tool_logs` specifically store tool names, input parameters, outputs, and execution status. This allows attribution methods to distinguish whether a failure arose because the agent received incomplete info, a prompt was malformed, or a tool returned an anomaly.

**2. Multi-round expert failure attribution: Labeling both responsible components and critical failure steps**

Locating the "point of no return" (step-level) is difficult for a single annotator. TraceElephant employs multi-round expert annotation: experts independently provide agent-level and step-level labels, then resolve disagreements through consensus. Initial inter-annotator agreement (Krippendorff's alpha) was 0.72 for agent-level and 0.64 for step-level, quantifying the inherent difficulty of fine-grained failure localization.

**3. Static and dynamic attribution evaluation: Testing both passive log reading and active hypothesis verification**

Real-world debugging involves re-running systems and modified inputs. TraceElephant provides two settings: the **Static** configuration provides full traces (methods include All-at-Once, Binary Search, Step-by-Step, and Static Agentic); the **Dynamic** configuration provides a replayable environment. The Dynamic Agentic method uses static proposals and then performs counterfactual checks to verify the failure point via replay.

### Loss & Training
This work does not train new models. Evaluation metrics are Agent-level Accuracy and Step-level Accuracy. The default Static Agentic method navigates trace information via a setup similar to mini-SWE-agent; the dynamic method adds replay and counterfactual probing. Experiments distinguish between scenarios with/without ground truth (GT) to simulate the presence of an oracle signal.

## Key Experimental Results

### Main Results
TraceElephant covers three systems and three task sources, with 220 failed traces annotated.

| System | Task Source | # Traces | # Failed |
|--------|-------------|----------|----------|
| Captain-Agent | GAIA | 126 | 73 |
| Captain-Agent | AssistantBench | 21 | 12 |
| Magentic-One | GAIA | 119 | 74 |
| Magentic-One | AssistantBench | 30 | 17 |
| SWE-Agent | SWE-Bench | 84 | 44 |
| Total | All | 380 | 220 |

Performance of different attribution techniques shows that Dynamic Agentic performs best.

| Configuration | Ground Truth | Agent Acc | Step Acc |
|---------------|--------------|-----------|----------|
| All-at-Once | w/ | 62.2 | 28.1 |
| Binary Search | w/ | 38.9 | 12.9 |
| Step-by-Step | w/ | 60.9 | 16.7 |
| Static Agentic | w/ | 65.9 | 30.3 |
| Dynamic Agentic | w/ | 66.7 | 33.3 |
| All-at-Once | w/o | 58.0 | 22.7 |
| Static Agentic | w/o | 59.1 | 26.1 |
| Dynamic Agentic | w/o | 60.6 | 27.6 |

### Ablation Study
Observability ablations verify that inputs and metadata are indispensable. Degrading to output-only traces leads to a significant drop in step-level accuracy.

| Observability | All-at-Once Agent | All-at-Once Step | Static Agentic Agent | Static Agentic Step |
|---------------|-------------------|------------------|----------------------|---------------------|
| Full trace | 0.62 | 0.28 | 0.66 | 0.30 |
| w/o metadata | 0.55 | 0.21 | 0.57 | 0.23 |
| w/o input | 0.54 | 0.18 | 0.56 | 0.19 |
| w/o metadata & input | 0.51 | 0.16 | 0.54 | 0.17 |

### Key Findings
- Full traces are critical for step-level attribution; All-at-Once accuracy drops from 0.28 to 0.16 when moving to output-only logs.
- Dynamic replay provides an additional ~10% relative improvement in step-level accuracy (e.g., 30.3% to 33.3%).
- Performance drops across all methods when ground truth is unavailable, yet Agentic methods are more robust as active probing compensates for the lack of a reference signal.
- Environment interactions and specific action agents are the primary sources of failure (>50%); however, planners/orchestrators contribute 18%-29% of failures via erroneous decomposition or dispatching.

## Highlights & Insights
- The core contribution is refocusing failure attribution on the "developer debugging" context. Output-only formats, while simple for academic benchmarks, are insufficient for real-world fixes that require knowledge of inputs and configurations.
- The TraceElephant schema is highly practical, particularly the `input_context` and `tool_logs`. Many agent bugs originate in prompt construction or tool formatting which are invisible without such records.
- Moving from a static dataset to an experimental environment allows for hypothesis testing and counterfactual re-runs.
- Results indicate that step-level attribution remains challenging. Even with full traces and dynamic probing, accuracy peaks at 33.3%, leaving significant room for specialized causal tracking and graph-based reasoning.

## Limitations & Future Work
- The benchmark covers only three MAS paradigms. While diverse, they may not represent all future architectures (e.g., highly asynchronous systems or those with long-term memory).
- By focusing on developer-facing full observability, the benchmark does not address black-box debugging where only external logs are available.
- Labeling remains subjective; the 0.64 alpha for step-level attribution indicates that the "earliest failure point" is not always uniquely identifiable in complex traces.
- Future work could explicitly model agent interaction graphs or train specialized small models for efficient attribution beyond general LLM-based agents.

## Related Work & Insights
- **vs Who&When**: Who&When provides output-only, partially observable traces for black-box attribution. TraceElephant provides full observability for developer-centric debugging.
- **vs ECHO / AgenTracer**: While those focus on attribution methods, TraceElephant provides the infrastructure and reproducible scenarios to evaluate such methods.
- **vs Traditional Debugging**: Traditional software debugging assumes discrete states and deterministic execution. LLM-based MAS require full context and counterfactual verification due to natural language states and non-deterministic outputs.
- **Insight**: Future agent platforms should log structured execution traces by default. Without observability, the effectiveness of automated attribution methods is capped by information loss.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Related benchmarks exist, but the shift to full observability and replayable environments is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Diverse systems and tasks, though extending to more architectures would be beneficial.
- Writing Quality: ⭐⭐⭐⭐☆ Clear motivation and well-defined data schema.
- Value: ⭐⭐⭐⭐⭐ High utility as infrastructure for researching agent observability and debugging tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ACL 2026\] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](../../ICML2026/multi_agent/maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
