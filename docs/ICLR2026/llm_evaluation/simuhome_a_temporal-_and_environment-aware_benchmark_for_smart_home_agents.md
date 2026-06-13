---
title: >-
  [Paper Note] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents
description: >-
  [ICLR 2026][LLM Evaluation][smart_home] SimuHome is a high-fidelity smart home simulator built on the Matter protocol and a 600-episode evaluation benchmark supporting dynamic environmental variable updates and time-acce…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "smart_home"
  - "LLM_agent"
  - "benchmark"
  - "temporal_reasoning"
  - "workflow_scheduling"
date: 2026-05-08
content_hash: 566c4e4d7a4abf50
---

# SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents

**Conference**: ICLR 2026  
**arXiv**: [2509.24282](https://arxiv.org/abs/2509.24282)  
**Code**: [holi-lab/SimuHome](https://github.com/holi-lab/SimuHome)  
**Area**: LLM Evaluation  
**Keywords**: smart_home, LLM_agent, benchmark, temporal_reasoning, workflow_scheduling  

## TL;DR

SimuHome is a high-fidelity smart home simulator built on the Matter protocol and a 600-episode evaluation benchmark supporting dynamic environmental variable updates and time-accelerated scheduling evaluation, revealing that workflow scheduling remains the most persistent challenge for current LLM agents.

## Background & Motivation

**Challenges in smart home agent development**:
- Amazon Alexa and Google Home have been commercially deployed, yet many everyday requests remain unhandled.
- User requests are far more complex than simple commands: "It feels stuffy in here" requires inferring implicit intent and triggering a dehumidifier.
- Device operations have dependency constraints: a robot vacuum must be powered on before switching to mopping mode.
- Temporally coordinated requests such as "turn on the kitchen light after the dishwasher finishes" require estimating completion times and scheduling accordingly.

**Limitations of Prior Work**:
- **HomeBench**: evaluates by comparing API call sequences but does not simulate environmental changes.
- **Sasha**: focuses on creative goal interpretation (e.g., "make the atmosphere cozy") and relies on human judgment.
- **SAGE**: supports sequential tool calls but does not simulate the effect of devices on environmental variables.
- **Common shortcomings**: none simulate the continuous impact of device operations on environmental variables such as temperature, humidity, or illuminance; none support operation dependency constraints; none support temporal scheduling evaluation.

## Method

### Overall Architecture

SimuHome makes two core contributions:
1. **Time-accelerated simulator**: built on the Matter protocol, modeling the continuous influence of device operations on environmental variables.
2. **600-episode evaluation benchmark**: covering 6 query types (each with feasible/infeasible variants), verified by human annotators.

### Key Designs

**Simulator Architecture**:

Three core components address three requirements:

1. **Smart Home Environment (dependency modeling)**:
    - Defines device communication and operational rules based on the Matter protocol.
    - Each room contains a set of devices and 4 environmental variables (temperature, illuminance, humidity, air quality).
    - 17 device types are defined, each with Matter clusters (feature groups).
    - Device operations follow dependency constraints (e.g., an air conditioner must be powered on before its temperature can be adjusted).

2. **Real-Time State Update Mechanism (real-time environmental feedback)**:
    - Time is discretized into ticks (0.1 seconds/tick), ensuring full determinism.
    - Each tick computes the cumulative effect of all active devices on environmental variables.
    - Example: two high-speed air conditioners cool a room faster than one.
    - Device sensors are updated synchronously (e.g., the temperature sensor on an air conditioner reflects the current ambient temperature).

3. **Agent-Simulator Interface**:
    - Agents query device states and environmental variables, execute Matter commands, and schedule workflows via API.
    - `schedule_workflow`: registers a command sequence for a future time point; returns only a confirmation without pre-validation.
    - The simulator supports time acceleration, enabling immediate forwarding to the scheduled execution time for result verification.

**6 Query Types**:

| Query Type | Description | Example |
|---------|------|------|
| QT1 Status Query | Retrieve environmental variables/device states | "What is the humidity in the kitchen?" |
| QT2 Implicit Intent Inference | Infer needs from indirect expressions | "It's too stuffy here" → activate dehumidifier |
| QT3 Explicit Device Control | Execute specific commands | "Set the living room air purifier to maximum fan speed" |
| QT4-1 Timed Scheduling | Control devices at a future time | "Turn off the lights and humidifier in ten minutes" |
| QT4-2 Event-Driven Scheduling | Trigger based on device completion events | "Turn off the kitchen light after the dishwasher finishes" |
| QT4-3 Coordinated Scheduling | Synchronize completion times of multiple devices | "Make the dishwasher and washing machine finish at the same time" |

Each type includes **infeasible variants**: non-existent devices, physical limits, and temporal contradictions.

**Episode Generation Pipeline** (three steps):

1. **STEP1 Initial State Construction**: randomize room layout and device configuration; sequentially randomize device states following dependency order; accelerate time until environmental variables stabilize.
2. **STEP2 Goal and Prerequisite Action Generation**: generate structured goals according to query type, along with prerequisite actions that must appear in the agent's tool-call history.
3. **STEP3 Query Synthesis**: GPT-4o mini generates natural language queries; two graduate students independently review them (Cohen's κ = 0.92).

### Evaluation Method

- **Simulator-based evaluation**: directly compare final device states/environmental variables against targets (for QT2/3/4 feasible variants).
- **LLM-as-Judge**: evaluate accuracy of natural language responses (for QT1 feasible + all infeasible variants).
- Judge reliability: Cohen's κ = 0.826; each episode is queried 3 times with majority voting.

### Loss & Training

This work presents an evaluation benchmark and involves no model training. Evaluation uses a binary success/failure criterion (target state match + prerequisite action presence).

## Key Experimental Results

### Main Results: Success Rate (%) of 18 LLM Agents

| Model | QT1-F | QT2-F | QT3-F | QT4-1-F | QT4-2-F | QT4-3-F |
|------|-------|-------|-------|---------|---------|---------|
| GPT-5.1 (reasoning) | **100** | **80** | **86** | **60** | **72** | **56** |
| Gemini-2.5-Pro (reasoning) | 96 | 60 | 76 | 44 | 60 | 46 |
| GPT-4.1 | 98 | 44 | 84 | 50 | 46 | 34 |
| GPT-4.1-mini | 96 | 62 | 64 | 26 | 40 | 10 |
| Llama4-Maverick | 96 | 52 | 88 | 22 | 18 | 32 |
| Qwen3-32B | 82 | 62 | 52 | 18 | 14 | 16 |
| Qwen3-235B-A22B | 86 | 32 | 84 | 26 | 38 | 28 |
| Gemma3-27B-it | 80 | 54 | 48 | 24 | 4 | 6 |
| Llama3.2-1B-it | 0 | 0 | 0 | 0 | 0 | 0 |

Core finding: **Workflow scheduling (QT4) is the weakest link for all models.** Even the strongest model, GPT-5.1, achieves only 56% on QT4-3.

### Infeasible Request Detection

| Model | QT1-IF | QT2-IF | QT3-IF | QT4-1-IF | QT4-2-IF | QT4-3-IF |
|------|--------|--------|--------|----------|----------|----------|
| GPT-5.1 | **94** | **50** | **92** | **100** | **92** | 44 |
| Gemini-2.5-Pro | 78 | 56 | 72 | **94** | 76 | **50** |
| GPT-4.1 | 82 | 44 | 88 | 12 | 34 | 32 |
| Qwen3-32B (SFT) | 88 | 32 | 74 | 32 | 10 | 14 |

Reasoning models substantially outperform non-reasoning models on infeasible request detection, particularly for QT4-IF. However, GPT-5.1 incurs latencies exceeding 100 seconds per episode, making real-time deployment impractical.

### Ablation Study: Error Analysis (GPT-4.1)

**Error type distribution for feasible episodes**:

| Error Type | QT2 | QT4 |
|---------|---------|---------|
| Device Control (DC) | **71%** | **40%** |
| Temporal Reasoning (TR) | 0% | **25%** |
| Action Planning (AP) | 7% | **19%** |
| Intent Inference (II) | 11% | 0% |
| Environment Perception (EP) | 11% | 16% |

**Error type distribution for infeasible episodes**:

| Error Type | QT2 | QT4 |
|---------|---------|---------|
| Contradiction Mishandling (CM) | **dominant** | minor |
| Contradiction Blindness (CB) | minor | **dominant** |

Key findings: QT2 errors concentrate on device control (operating the wrong device); QT4 errors are more diverse, with temporal reasoning and action planning each accounting for a significant share. For infeasibility detection, QT2 models detect contradictions but handle them incorrectly, whereas QT4 models fail to detect contradictions altogether.

### Critical Role of Tool Feedback

| Query Type | First-Attempt Success Rate | Recovery Success Rate After Error |
|---------|----------|---------------|
| QT3 | ~60% | >40% (recovery via error messages) |
| QT4 | Near 100% first-attempt only | ~0% (no feedback from `schedule_workflow`) |

This explains the performance gap between QT3 and QT4: QT3 provides immediate feedback enabling error correction, whereas QT4 requires success on the first attempt.

### SFT Experiments

Fine-tuning Gemma3-4B-it and Qwen3-32B on successful trajectories from GPT-5.1:
- Infeasibility detection improves substantially (up to +26 percentage points).
- QT4-3 shows almost no improvement — because scheduling requires dynamic environmental interaction, and imitating successful trajectories is insufficient.

### Key Findings

1. **Workflow scheduling is the most persistent challenge**: all models perform worst on QT4, particularly QT4-3 coordinated scheduling.
2. **Reasoning models offer significant advantages but prohibitive latency**: GPT-5.1 is the best-performing model but requires 100+ seconds per episode, making real-time deployment infeasible.
3. **Immediate feedback is critical to QT3 success**: over 40% of successes stem from error recovery; the absence of feedback is the bottleneck in QT4.
4. **Temporal contradiction detection is a blind spot**: non-reasoning models are nearly incapable of identifying unsatisfiable temporal constraints.
5. **SFT provides limited gains**: imitation learning improves infeasibility detection but cannot resolve dynamic scheduling problems.
6. **The bottleneck lies in models, not frameworks**: replacing ReAct with HiAgent, multi-turn interaction, and self-correction do not fundamentally resolve the issues.

## Highlights & Insights

- **Rigorous simulator design**: built on the industry-standard Matter protocol; tick-based deterministic simulation ensures reproducibility.
- **Fine-grained evaluation taxonomy**: 6 query types × feasible/infeasible = 12 categories, 50 episodes each, spanning a full spectrum from simple to complex tasks.
- **Thorough error analysis**: goes beyond reporting success rates to provide detailed error categorization and tool-feedback analysis.
- **Time acceleration as a key innovation**: enables immediate verification of scheduling task outcomes without waiting for real-time execution.
- **Deployment-oriented perspective**: explicitly addresses the latency issue of reasoning models rather than optimizing solely for accuracy.

## Limitations & Future Work

1. Only 17 device types are modeled; real smart homes may involve more complex device interactions.
2. Environmental influence models use linear superposition; real physical processes are more complex (e.g., window ventilation, sunlight).
3. Cross-room device effects are not supported (e.g., a living room air conditioner affecting bedroom temperature).
4. Natural language queries are generated by GPT-4o mini and reviewed by humans, which may not fully capture the diversity of real user expressions.
5. LLM-as-Judge has been human-validated (κ = 0.826) but may still be inaccurate in edge cases.
6. The ReAct framework is used primarily (with limited HiAgent testing); other agent frameworks (e.g., Plan-and-Execute) are not sufficiently explored.

## Related Work & Insights

- **AI2-THOR / ALFRED / VirtualHome**: 3D embodied agent benchmarks focusing on physical navigation and object manipulation, which represent a different problem from smart home API invocation.
- **HomeBench (Li et al., 2025)**: large-scale instruction-following evaluation, but relies on static API sequence comparison.
- **SAGE (Rivkin et al., 2024)**: supports sequential tool use but does not simulate environmental variables.
- **Matter Protocol**: global smart home standard, enabling simulation results to transfer to real devices.
- Insights: effective agent evaluation requires interactive environments rather than static datasets; deferred scheduling feedback is an important and unresolved problem.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Time acceleration combined with environment-aware simulation is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 18 models × 12 categories × 600 episodes with detailed analysis.
- **Value**: ⭐⭐⭐⭐⭐ — The open-source simulator provides direct value for smart home agent research.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with informative tables.
- **Overall**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation](../../ACL2026/llm_evaluation/aj-bench_benchmarking_agent-as-a-judge_for_environment-aware_evaluation.md)
- [\[ICLR 2026\] In-Context Learning of Temporal Point Processes with Foundation Inference Models](in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)
- [\[ACL 2026\] Rethinking Meeting Effectiveness: A Benchmark and Framework for Temporal Fine-grained Automatic Meeting Effectiveness Evaluation](../../ACL2026/llm_evaluation/rethinking_meeting_effectiveness_a_benchmark_and_framework_for_temporal_fine-gra.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ICLR 2026\] AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)

</div>

<!-- RELATED:END -->
