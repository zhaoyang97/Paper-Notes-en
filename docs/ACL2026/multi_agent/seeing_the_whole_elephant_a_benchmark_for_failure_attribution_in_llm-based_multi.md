---
title: >-
  [Paper Note] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems
description: >-
  [ACL 2026][Multi-Agent][Paper Note] TraceElephant advocates that multi-agent failure attribution should be evaluated under complete execution traces visible to developers. It provides 220 failed traces with annotations for responsible agents and critical failure steps, demonstrating that full observability improves step-level attribution from 16% (output
tags:
  - ACL 2026
  - Multi-Agent
date: 2026-05-08
content_hash: 5a42dfaf87a12624
---
# Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems

**Conference**: ACL2026  
**arXiv**: [2604.22708](https://arxiv.org/abs/2604.22708)  
**Code**: https://github.com/TraceElephant/TraceElephant  
**Area**: LLM Evaluation / Multi-Agent Debugging  
**Keywords**: Failure Attribution, Multi-Agent Systems, Execution Trace, Observability, Debugging Benchmark

## TL;DR
TraceElephant advocates that multi-agent failure attribution should be evaluated under complete execution traces visible to developers. It provides 220 failed traces with annotations for responsible agents and critical failure steps, demonstrating that full observability improves step-level attribution from 16% (output-only) to over 28%-30%.

## Background & Motivation
**Background**: LLM-based multi-agent systems are widely used for web retrieval, software engineering, complex task decomposition, and tool calling. These systems comprise multiple agents, prompts, tools, and environmental interactions, making it difficult to locate responsible components and the earliest erroneous steps upon failure.

**Limitations of Prior Work**: Existing failure attribution benchmarks primarily use partially observable traces that only record agent outputs, omitting input contexts, system prompts, agent configurations, tool logs, and environmental states. This setup suits black-box observation but does not align with developer debugging scenarios, where full execution logs are typically accessible.

**Key Challenge**: The goal of failure attribution is to provide actionable diagnostics. Actionable diagnostics depend on "what the agent saw at the time, how it was configured, what tools it called, and what the environment returned." Relying solely on output sequences often conflates missing upstream information with errors inherent to the downstream agent.

**Goal**: To build a developer-facing failure attribution benchmark providing full execution traces and reproducible experimental environments. This allows researchers to evaluate the impact of static trace analysis, dynamic replay, and counterfactual probing on attribution accuracy.

**Key Insight**: The paper uses the metaphor "seeing the whole elephant" to represent complete observability. The authors analyzed 184 failure cases (Who & When) and found that at least 21% could not be reliably attributed under output-only logs. This led to redesigning the benchmark schema where inputs, outputs, tool logs, and system architecture are treated as first-class information.

**Core Idea**: Failure attribution should not merely ask "which output in the log looks wrong," but rather "in the complete execution narrative, which component made which decision under what input conditions that made failure inevitable."

## Method
TraceElephant is a benchmark rather than a new attribution model. Its contribution lies in defining the task, collecting complete traces, designing an annotation protocol, and systematically comparing different observability levels and attribution techniques. The evaluation units can be actual multiple agents or functional components with independent decision-making responsibilities within a single-agent scaffold.

### Overall Architecture
Each sample consists of a full execution trace of a failed task, a runnable system environment, a label for the responsible agent/component, and a label for the decisive failure step. Step-level attribution identifies the earliest step where failure becomes inevitable; agent-level attribution identifies the component primarily responsible during that step.

Data is sourced from three representative systems: Captain-Agent, Magentic-One, and SWE-Agent. These cover dynamic team assembly, centralized multi-agent orchestration, and software engineering single-agent scaffolds, respectively. Tasks originate from GAIA, AssistantBench, and SWE-Bench. A total of 380 traces were collected, with 220 failed traces used for attribution labels.

### Key Designs
**1. Full execution trace schema: Recording inputs, configurations, and tool logs that developers actually inspect, rather than just outputs.**

Most existing benchmarks only retain agent output sequences, causing "upstream information errors" and "current component reasoning errors" to be conflated. TraceElephant redesigns the trace schema to include inputs and metadata as first-class information: the trace level records `task_id`, `task_instruction`, `system_name`, `agent_configuration`, and `system_architecture`; each step records input fields (`step_id`, `agent_id`, `agent_name`, `input_context`) and output fields (`output_content`, `tool_logs`), where `tool_logs` further save tool names, input parameters, outputs, and execution status. With full context, attribution methods can determine if an error occurred because the agent received incomplete information, the prompt was malformed, or the tool returned an anomaly—thereby distinguishing "upstream errors" from "local reasoning errors."

**2. Multi-round expert failure attribution annotation: Labeling both the responsible component and the critical failure step for each failed trace.**

Failure attribution is not a simple classification, especially at the step level where one must determine "where the failure started to become irreversible." TraceElephant employs multi-round expert annotation: experts independently provide agent-level and step-level labels, then resolve disagreements through discussion to reach a consensus. The first-round inter-annotator agreement (Krippendorff's alpha) was 0.72 for agent-level and 0.64 for step-level—indicating that while agent attribution is relatively consistent, step attribution is significantly harder, a gap that quantifies the difficulty of pinpointing the earliest irreversible failure point.

**3. Static and dynamic attribution evaluation: Testing both passive log reading and active replay to verify hypotheses.**

Real-world debugging rarely stops at reading logs; it often involves rerunning systems, modifying inputs, and verifying guesses. TraceElephant provides two configurations: the static configuration provides only the full trace using methods like All-at-Once, Binary Search, Step-by-Step, and Static Agentic; the dynamic configuration provides a replayable execution environment. Dynamic Agentic first proposes candidate attributions using static methods, then performs counterfactual checks by rerunning from the candidate failure point. Incorporating active debugging transitions the benchmark from passive log classification to interactive diagnosis.

### Loss & Training
The paper does not train a new model. The evaluation metrics are agent-level accuracy and step-level accuracy. The default Static Agentic method navigates trace information based on mini-SWE-agent; the dynamic method executes replay and counterfactual probing based on static candidates. Experiments distinguish between "w/ ground truth" and "w/o ground truth" scenarios to simulate whether the correct answer or test-pass signal is provided.

## Key Experimental Results

### Main Results
TraceElephant covers three types of systems and three task sources, with 380 total traces and 220 failed traces annotated.

| System | Task Source | # Traces | # Failed |
|--------|-------------|----------|----------|
| Captain-Agent | GAIA | 126 | 73 |
| Captain-Agent | AssistantBench | 21 | 12 |
| Magentic-One | GAIA | 119 | 74 |
| Magentic-One | AssistantBench | 30 | 17 |
| SWE-Agent | SWE-Bench | 84 | 44 |
| Total | All | 380 | 220 |

The overall average performance of different attribution techniques shows that Dynamic Agentic performs best, while Static Agentic is the strongest static method.

| Configuration | Ground Truth | Agent Acc | Step Acc |
|------|--------------|-----------|----------|
| All-at-Once | w/ | 62.2 | 28.1 |
| Binary Search | w/ | 38.9 | 12.9 |
| Step-by-Step | w/ | 60.9 | 16.7 |
| Static Agentic | w/ | 65.9 | 30.3 |
| Dynamic Agentic | w/ | 66.7 | 33.3 |
| All-at-Once | w/o | 58.0 | 22.7 |
| Static Agentic | w/o | 59.1 | 26.1 |
| Dynamic Agentic | w/o | 60.6 | 27.6 |

### Ablation Study
The observability ablation confirms that "inputs and metadata are indispensable." When degraded to output-only (w/o metadata & input), results align closely with the Who & When setup, and step-level accuracy significantly decreases.

| Observability Config | All-at-Once Agent | All-at-Once Step | Static Agentic Agent | Static Agentic Step |
|--------------------|-------------------|------------------|----------------------|---------------------|
| Full trace | 0.62 | 0.28 | 0.66 | 0.30 |
| w/o metadata | 0.55 | 0.21 | 0.57 | 0.23 |
| w/o input | 0.54 | 0.18 | 0.56 | 0.19 |
| w/o metadata & input | 0.51 | 0.16 | 0.54 | 0.17 |

### Key Findings
- Complete traces are critical for step-level attribution. All-at-Once drops from 0.28 on full traces to 0.16 on output-only; Static Agentic drops from 0.30 to 0.17.
- Dynamic replay provide additional gains for step-level attribution: with ground truth, Dynamic Agentic step accuracy is 33.3%, compared to 30.3% for Static Agentic, a ~10% improvement. Agent-level gains are smaller as responsible components can often be inferred from static roles.
- All methods perform worse when ground truth is unavailable, indicating that correct answers or test signals significantly aid fine-grained attribution; however, agentic methods show smaller declines, suggesting active retrieval and verification can partially compensate for missing reference signals.
- External environment interactions and specific worker agents are the primary sources of failure, accounting for over 50%; planner/orchestrator components contribute 18%-29%, often due to incorrect decomposition, dispatching, or propagation of coordination logic.

## Highlights & Insights
- The most significant contribution of the paper is placing failure attribution back into the "developer debugging" context. While output-only logs are simple for academic benchmarks, they are insufficient for real-world fixes which require inputs, configurations, and tool contexts.
- The TraceElephant schema is highly practical, particularly `input_context` and `tool_logs`. Many agent bugs occur in prompt construction, information passing, and tool output formatting; without these recorded, attribution is inherently ambiguous.
- The dynamic configuration transforms the benchmark from a static dataset into an experimental environment. This is crucial as future attribution methods can perform hypothesis testing, counterfactual reruns, and automated fault injection.
- The results indicate that step-level attribution remains challenging. Even with full trace + Dynamic Agentic, accuracy is only 33.3%, leaving substantial room for specialized causal tracing, graph reasoning, and structured trace summarization.

## Limitations & Future Work
- The authors acknowledge the benchmark only covers three MAS paradigms. Although diverse, they may not represent all future agent architectures, particularly those with more asynchronous operations, long-term memory, or real-user interaction.
- The emphasis on developer-facing full observability means it does not cover black-box platform debugging. For third-party analysts with only external logs, TraceElephant's setting might be overly ideal.
- The labeling task involves subjectivity. The Krippendorff's alpha for step-level attribution is 0.64, indicating the "earliest failure point" is not always uniquely clear in complex traces.
- Current Static Agentic methods only use basic tools to view step I/O. Future work could explicitly model agent interaction graphs, tool call graphs, and temporal dependencies, or train specialized small models for efficient attribution.

## Related Work & Insights
- **vs. Who & When**: Who & When provides output-only, partially observable traces suitable for black-box attribution; TraceElephant provides full inputs, outputs, tools, configurations, and runnable environments, closer to developer debugging.
- **vs. ECHO / AgenTracer / GraphTracer / FAMAS**: These methods focus on the *how* of agent failure attribution; TraceElephant acts as benchmarking infrastructure by providing more complete and reproducible evaluation scenarios for these methods.
- **vs. Traditional Delta Debugging / Statistical Debugging**: Traditional software debugging assumes discrete states, traceable execution, and deterministic components; the natural language states and non-deterministic outputs of LLM-based MAS make attribution more reliant on full context and counterfactual validation.
- **Insight**: Future agent platforms should record structured execution traces by default, including input prompts, visible history, tool logs, and environmental states. Without observability, attribution methods will always be limited by missing information.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Related benchmarks exist, but the developer perspective of full observability + replayable environment is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Diverse systems and configurations; clear observability ablations, though the number of systems is still limited.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and data schemas are well-explained; experimental conclusions offer practical debugging insights.
- Value: ⭐⭐⭐⭐⭐ High infrastructure value for research into multi-agent observability, debugging tools, and failure attribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ACL 2026\] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
