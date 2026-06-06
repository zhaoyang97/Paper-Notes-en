---
title: >-
  [Paper Note] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems
description: >-
  [ACL2026][Multi-Agent][Failure Attribution] TraceElephant argues that failure attribution in multi-agent systems should be evaluated using full execution traces visible to developers. It provides 220 failure traces with…
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Failure Attribution"
  - "Multi-Agent Systems"
  - "Execution Trace"
  - "Observability"
  - "Debugging Benchmark"
date: 2026-05-08
content_hash: b22ff5a78305d555
---

# Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems

**Conference**: ACL2026  
**arXiv**: [2604.22708](https://arxiv.org/abs/2604.22708)  
**Code**: https://github.com/TraceElephant/TraceElephant  
**Area**: LLM Evaluation / Multi-Agent Debugging  
**Keywords**: Failure Attribution, Multi-Agent Systems, Execution Trace, Observability, Debugging Benchmark

## TL;DR
TraceElephant argues that failure attribution in multi-agent systems should be evaluated using full execution traces visible to developers. It provides 220 failure traces with annotations for responsible agents and decisive failure steps, demonstrating that full observability improves step-level attribution from 16% (output-only) to over 28%-30%.

## Background & Motivation
**Background**: LLM-based multi-agent systems are widely utilized for web retrieval, software engineering, complex task decomposition, and tool calling. These systems consist of multiple agents, prompts, tools, and environmental interactions, making it difficult to pinpoint responsible components and the earliest error steps when failures occur.

**Limitations of Prior Work**: Existing failure attribution benchmarks primarily utilize partially observable traces that only record agent outputs, omitting input contexts, system prompts, agent configurations, tool logs, and environmental states. Such setups are suitable for black-box observation but do not align with developer debugging scenarios, where full execution logs are typically accessible.

**Key Challenge**: The goal of failure attribution is to provide actionable diagnoses, which depend on knowing "what the agent saw at that time, how it was configured, what tools it called, and what the environment returned." Relying solely on output sequences tends to conflate upstream information deficiency with downstream agent errors.

**Goal**: Construct a developer-facing failure attribution benchmark that provides both complete execution traces and reproducible experimental environments. This enables researchers to evaluate the effects of static trace analysis, dynamic replay, and counterfactual probing on attribution accuracy.

**Key Insight**: The paper uses the metaphor of "seeing the whole elephant" to represent complete observability. The authors analyzed 184 failure cases from Who&When and found that at least 21% cannot be reliably attributed under output-only logs. Consequently, they redesigned the benchmark schema to treat inputs, outputs, tool logs, and system architecture as first-class information.

**Core Idea**: Failure attribution should not merely ask "which output in the log appears incorrect," but rather "in the complete execution narrative, which component made a decision under what input conditions that made the failure inevitable."

## Method
TraceElephant is a benchmark rather than a new attribution model. Its contributions lie in defining the task, collecting complete traces, designing an annotation protocol, and systematically comparing different observability levels and attribution techniques. The evaluation unit can be a true multi-agent system or independent functional components within a single-agent scaffold.

### Overall Architecture
Each sample consists of a complete execution trace of a failed task, a runnable system environment, a label for the responsible agent/component, and a label for the decisive failure step. Step-level attribution aims to find the earliest step where failure becomes inevitable; agent-level attribution identifies the component primarily responsible for that step.

Data is sourced from three representative systems: Captain-Agent, Magentic-One, and SWE-Agent. These cover dynamic team assembly, centralized multi-agent orchestration, and software engineering single-agent scaffolds, respectively. Tasks originate from GAIA, AssistantBench, and SWE-Bench, with 380 traces collected, of which 220 failed traces are used for attribution annotation.

### Key Designs
1.  **Complete Execution Trace Schema**:
    - **Function**: Records information that developers actually need during debugging, rather than just agent outputs.
    - **Mechanism**: Trace-level metadata includes `task_id`, `task_instruction`, `system_name`, `agent_configuration`, and `system_architecture`. Each step records input and output fields: input includes `step_id`, `agent_id`, `agent_name`, and `input_context`; output includes `output_content` and `tool_logs`. `tool_logs` store tool names, input parameters, outputs, and execution status.
    - **Design Motivation**: The root cause of many errors is not the output itself but incomplete input information, prompt construction errors, or anomalous tool returns. The complete schema allows attribution methods to distinguish between "upstream information transmission errors" and "current component reasoning errors."

2.  **Multi-round Expert Failure Attribution Annotation**:
    - **Function**: Provides responsible component and critical failure step labels for each failure trace.
    - **Mechanism**: Experts first independently annotate agent-level and step-level labels, then resolve uncertain cases through joint discussion. The first round of annotation achieved a Krippendorff's alpha of 0.72 (agent-level) and 0.64 (step-level), indicating higher consistency in agent attribution while step attribution remains more difficult.
    - **Design Motivation**: Failure attribution is not a simple classification; step-level attribution especially requires judging the "earliest irreversible failure point." Multi-round annotation improves reliability and reflects the fine-grained difficulty of the task.

3.  **Static and Dynamic Attribution Evaluation**:
    - **Function**: Compares different observability and attribution techniques in realistic debugging scenarios.
    - **Mechanism**: Static configurations use only the complete trace. Dynamic configurations provide a replayable environment, allowing for re-running from candidate failure points and performing counterfactual checks. Static methods include All-at-Once, Binary Search, Step-by-Step, and Static Agentic; Dynamic Agentic first proposes candidates using static methods and then validates them via re-runs.
    - **Design Motivation**: Developer debugging is not limited to log reading; it often involves re-running the system, modifying inputs, and validating hypotheses. The benchmark incorporates this active debugging capability to push from passive log classification toward interactive diagnosis.

### Loss & Training
This paper does not train a new model. Evaluation metrics are agent-level accuracy and step-level accuracy. The default Static Agentic method navigates trace information based on mini-SWE-agent; the dynamic method executes replay and counterfactual probing on top of static candidates. Experiments also distinguish between scenarios "w/ ground truth" and "w/o ground truth" to simulate the presence or absence of a correct answer or test pass signal.

## Key Experimental Results

### Main Results
TraceElephant covers three types of systems and three task sources, with a total of 380 execution traces, 220 of which are failed and annotated.

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
Observability ablation directly validates that "inputs and metadata are indispensable." When degraded to output-only (w/o metadata & input), settings resemble those in Who&When, and step-level accuracy drops significantly.

| Observability Config | All-at-Once Agent | All-at-Once Step | Static Agentic Agent | Static Agentic Step |
|----------------------|-------------------|------------------|----------------------|---------------------|
| Full trace | 0.62 | 0.28 | 0.66 | 0.30 |
| w/o metadata | 0.55 | 0.21 | 0.57 | 0.23 |
| w/o input | 0.54 | 0.18 | 0.56 | 0.19 |
| w/o metadata & input | 0.51 | 0.16 | 0.54 | 0.17 |

### Key Findings
- Complete traces are particularly critical for step-level attribution. All-at-Once drops from 0.28 with full traces to 0.16 with output-only; Static Agentic drops from 0.30 to 0.17.
- Dynamic replay provides an additional boost for step-level attribution: w/ ground truth, the Dynamic Agentic step accuracy is 33.3%, which is a ~10% gain over Static Agentic (30.3%). The gain in agent-level attribution is smaller, as responsible components can often be inferred from static roles and interaction structures.
- All methods decline when ground truth is unavailable, indicating that correct answers or test signals significantly aid fine-grained attribution; however, agentic methods show smaller declines, suggesting that active retrieval and verification can partially compensate for missing reference signals.
- External environmental interactions and specific execution agents are the primary sources of failure, accounting for over 50%; planners/orchestrators also contribute 18%-29% of failures, typically due to incorrect decomposition, dispatching, or coordination logic propagation.

## Highlights & Insights
- The most important perspective of the paper is placing failure attribution back into the context of "developer debugging." Relying solely on outputs may be sufficient for a concise academic benchmark, but it is inadequate for real-world fixes, which require knowledge of inputs, configurations, and tool contexts.
- The schema of TraceElephant is highly practical, particularly the `input_context` and `tool_logs`. Many agent bugs occur during prompt construction, information passing, and tool return formatting; without recording these, attribution is inherently ambiguous.
- The dynamic configuration transforms the benchmark from a static dataset into an experimental environment. This is crucial because future attribution methods can perform hypothesis testing, counterfactual re-runs, and automatic fault injection rather than just having an LLM read long logs.
- Results indicate that step-level attribution remains challenging. Even with full traces and Dynamic Agentic, accuracy is only 33.3%, leaving significant room for specialized causal tracing, graph reasoning, and structured trace summarization.

## Limitations & Future Work
- The authors acknowledge that the benchmark only covers three MAS. While these represent diverse design paradigms, they may not represent all future agent architectures, particularly those with asynchronous operations, long-term memory, or real user interactions.
- The paper emphasizes developer-facing full observability, thus it does not cover black-box platform debugging scenarios. For third-party analysts who only receive external output logs, the setup of TraceElephant may be overly idealistic.
- The annotation task itself involves subjectivity. The Krippendorff's alpha for step-level annotation was only 0.64, indicating that the "earliest failure point" is not always uniquely clear in complex traces.
- Current Static Agentic methods only use basic tools to view step I/O. Future work could explicitly model agent interaction graphs, tool call graphs, and temporal dependencies, or train specialized small models for efficient attribution.

## Related Work & Insights
- **vs Who&When**: Who&When provides output-only, partially observable traces suitable for black-box attribution; TraceElephant provides complete inputs, outputs, tools, configurations, and runnable environments, which is closer to developer debugging.
- **vs ECHO / AgenTracer / GraphTracer / FAMAS**: These methods focus on how to perform agent failure attribution, while TraceElephant serves as benchmark infrastructure, providing more complete and reproducible evaluation scenarios for these methods.
- **vs Traditional Delta Debugging / Statistical Debugging**: Traditional software debugging assumes discrete states, traceable execution, and deterministic components; the natural language states and non-deterministic outputs of LLM-based MAS make attribution more dependent on complete context and counterfactual validation.
- **Insight**: Future agent platforms should record structured execution traces by default, including input prompts, visible histories, tool logs, and environment states. Without observability, even the strongest automated attribution methods will be limited by missing information.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Related benchmarks exist, but the developer perspective with full observability and replayable environments is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Systems, tasks, and configurations are diverse, and observability ablation is clear; however, the number of systems is still limited.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and data schema are well-explained; experimental conclusions have practical debugging implications.
- Value: ⭐⭐⭐⭐⭐ Provides foundational value for research in multi-agent system observability, debugging tools, and failure attribution.

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
