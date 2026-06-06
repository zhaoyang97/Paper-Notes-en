---
title: >-
  [Paper Note] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents
description: >-
  [ICLR 2026][LLM Agent][Smart Home] This paper proposes SimuHome, a time-accelerated smart home simulator based on the Matter protocol along with a 600-episode benchmark. It is the first benchmark to simulate the continuo…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Smart Home"
  - "Workflow Scheduling"
  - "Temporal Reasoning"
  - "Interactive Simulator"
date: 2026-05-08
content_hash: 5134d621045958de
---

# SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents

**Conference**: ICLR 2026
**arXiv**: [2509.24282](https://arxiv.org/abs/2509.24282)  
**Code**: [https://github.com/holi-lab/SimuHome/](https://github.com/holi-lab/SimuHome/)  
**Area**: LLM Agent
**Keywords**: Smart Home, LLM Agent, Workflow Scheduling, Temporal Reasoning, Interactive Simulator

## TL;DR
This paper proposes SimuHome, a time-accelerated smart home simulator based on the Matter protocol along with a 600-episode benchmark. It is the first benchmark to simulate the continuous effects of device operations on environmental variables and to evaluate workflow scheduling capabilities. Results reveal that workflow scheduling remains the most challenging frontier for current LLM agents, including GPT-5.1.

## Background & Motivation
**Background**: Smart home agents (e.g., Amazon Alexa, Google Home) are among the earliest large-scale commercial tool agents, yet many everyday home requests still exceed their capabilities. Recent research leverages LLMs to build more capable smart home agents that must handle tasks ranging from simple commands to complex temporal coordination.

**Limitations of Prior Work**:
   - **No simulation of environmental dynamics**: Benchmarks such as HomeBench, Sasha, and SAGE do not simulate how device operations continuously affect environmental variables (e.g., temperature, humidity). Setting an air conditioner to 25°C does not instantly change the temperature—it decreases gradually, and agents need to observe this process.
   - **No modeling of operation dependencies**: Real devices have operational dependencies (e.g., an air conditioner must be powered on before its temperature can be adjusted), which existing benchmarks do not model.
   - **No support for temporal scheduling evaluation**: Tasks such as "turn on the kitchen light after the dishwasher finishes" require querying remaining time, computing completion moments, and registering timed tasks—capabilities no existing benchmark evaluates.
   - **Static data is insufficient**: A single user request may admit multiple valid action sequences, which fixed annotations cannot cover. Agents need to operate in an interactive environment and verify outcomes.

**Key Challenge**: LLM agents need to perform complex temporal reasoning in dynamic, physically constrained environments, yet no suitable simulator or benchmark exists to train and evaluate such capabilities.

**Goal**: To construct a high-fidelity, interactive, time-accelerated smart home simulator and a systematic benchmark covering 6 query types (with feasible/infeasible variants).

**Key Insight**: Device behavior is modeled based on the Matter protocol (the global smart home communication standard), ensuring that operational constraints in the simulator are consistent with real physical devices and enabling sim-to-real transfer.

**Core Idea**: Matter protocol + tick-based deterministic environment simulation + time acceleration + 6 query types × feasible/infeasible variants = comprehensive evaluation of LLM agent capabilities in realistic smart home scenarios.

## Method

### Overall Architecture
SimuHome consists of two components: (1) a **Simulator**—an interactive smart home environment based on the Matter protocol that supports device operations, continuous environmental variable updates, and time acceleration; and (2) a **Benchmark**—600 manually validated episodes covering 12 evaluation categories (6 query types × feasible/infeasible). Agents interact with the simulator via APIs and complete tasks under the ReAct framework.

### Key Designs

1. **Matter Protocol-Based Device Modeling**:

    - Function: Models operational rules and dependencies for 17 device types.
    - Mechanism: Device behavior strictly follows the Matter protocol—e.g., an air conditioner must execute `PowerOn` before temperature adjustment, and a washing machine has multi-stage operating cycles. Each device defines a set of supported Matter clusters (capability groups).
    - Design Motivation: Ensures that simulator behavior is consistent with real devices, enabling knowledge acquired in the simulator to transfer to physical environments.

2. **Real-Time Environmental State Update Mechanism**:

    - Function: Computes the cumulative effects of all active devices on environmental variables (temperature, illuminance, humidity, air quality) at each tick (0.1 second).
    - Mechanism: Effects from multiple devices are superimposed (e.g., two air conditioners running at high speed cool the room faster), and device sensor attributes are synchronously updated to reflect the current environment. Deterministic time steps guarantee full reproducibility.
    - Design Motivation: Real physical environment responses are gradual; agents must be able to observe and determine whether a target state has been reached.

3. **Agent–Simulator Interface and Workflow Scheduling**:

    - Function: Provides three categories of tools: querying device states, executing Matter commands, and registering timed workflows.
    - Mechanism: `schedule_workflow` accepts an absolute start time and a list of commands. Crucially, scheduling only returns a registration confirmation without pre-validating whether commands will succeed at execution time. Command failures at execution time also return no error—mimicking the behavior of real smart home platforms.
    - Design Motivation: This "delayed feedback" design is intentional—it simulates real-world scenarios where device states may change between scheduling and execution, and is the structural reason why QT4 tasks are harder than QT3.

4. **Six Query Type Design**:

    - **QT1 Status Query**: Querying environmental variables or device settings (e.g., "What is the humidity in the kitchen?").
    - **QT2 Implicit Intent Inference**: Users express needs indirectly (e.g., "It feels stuffy" → infer need for dehumidification → activate dehumidifier).
    - **QT3 Explicit Device Control**: Precisely specifying devices and target values while adhering to operational dependencies.
    - **QT4-1 Time-Based Scheduling**: Controlling devices at a specified future time (e.g., "Turn off the lights in 10 minutes").
    - **QT4-2 Event-Driven Scheduling**: Triggering actions upon device completion events (e.g., "Turn off the lights after the dishwasher finishes" → query remaining time → compute completion moment → register workflow).
    - **QT4-3 Coordinated Scheduling**: Synchronizing the timing of multiple devices (e.g., "Schedule the dishwasher and washing machine to finish at the same time" → compute remaining time for both → adjust start times accordingly).
    - Each query type has an **infeasible variant** (non-existent device / physical limits exceeded / temporal contradiction), requiring agents to identify and explain the reason.

5. **Episode Generation and Evaluation**:

    - Function: Constructs diverse episodes by randomizing home layouts, device states, and environmental variables.
    - Mechanism: A three-step pipeline—(a) dependency-aware random initialization of device states; (b) structured goal and prerequisite action generation; (c) query generation by GPT-4o mini followed by independent validation by two graduate students (Cohen's κ = 0.92). Evaluation uses two methods: direct simulator verification (for feasible QT2–4) and LLM-as-a-Judge (for infeasible episodes and QT1), with majority voting over three runs.
    - Design Motivation: Prerequisite action requirements (e.g., agents must first call `get_room_devices()`) prevent agents from succeeding through guesswork.

### Loss & Training
This paper is primarily a benchmark contribution. SFT experiments use 204 successful trajectories from GPT-5.1 to fine-tune Gemma3-4B-it and Qwen3-32B.

## Key Experimental Results

### Main Results (Success Rate %)

| Model | QT1-F | QT2-F | QT3-F | QT4-1-F | QT4-2-F | QT4-3-F |
|---|---|---|---|---|---|---|
| Llama4-Maverick | 96 | 52 | 88 | 22 | 18 | 32 |
| Qwen3-235B | 86 | 32 | 84 | 26 | 38 | 28 |
| Gemini-2.5-Flash | 92 | 66 | 82 | 22 | 40 | 12 |
| GPT-4.1 | 98 | 44 | 84 | 50 | 46 | 34 |
| Gemini-2.5-Pro | 96 | 60 | 76 | 44 | 60 | 46 |
| **GPT-5.1** | **100** | **80** | **86** | **60** | **72** | **56** |

### Ablation Study (Reasoning Capability vs. Latency Trade-off)

| Model | QT3-F Time (s) | QT4-2-F Time (s) | QT4-3-F Time (s) | Reasoning Model |
|---|---|---|---|---|
| GPT-4.1 | 22.9 | 28.7 | 29.7 | No |
| Gemini-2.5-Pro | 66.1 | 57.7 | 53.7 | Yes |
| GPT-5.1 | 78.6 | 135.1 | 112.7 | Yes |

### Key Findings
- **Workflow scheduling is the most persistent challenge**: Even GPT-5.1 achieves only 56% on QT4-3 (coordinated scheduling). Success rates drop sharply from QT1/QT3 to QT4.
- **Immediate feedback is key to QT3 success**: Over 40% of successful QT3 episodes involve recovery from initial errors via tool feedback; the `schedule_workflow` interface in QT4 lacks this mechanism, preventing agents from detecting their own mistakes.
- **Reasoning models offer large gains at high cost**: GPT-5.1 outperforms non-reasoning GPT-4.1 by 26 percentage points on QT4-2 (46%→72%), but requires 3–5× more time (135s vs. 29s), which is impractical for real-time smart home applications.
- **SFT yields limited improvement**: Fine-tuning most substantially improves infeasible request detection (up to +26%), but provides almost no benefit for feasible workflow scheduling (QT4-3-F remains at 0%), as temporal computations vary across episodes and cannot be learned through imitation.
- **Small models are nearly incapable**: Models with fewer than 7B parameters achieve 0% success on most tasks; only Gemma3-4B-it achieves limited success on QT1.
- **Error analysis**: QT2 failures are dominated by device control errors (DC, 71%); QT4 errors are more evenly distributed—DC (40%), temporal reasoning (TR, 25%), and action planning (AP, 19%).

## Highlights & Insights
- **The "delayed feedback" design philosophy is the central insight**: The performance gap between QT3 and QT4 is not merely a matter of task complexity but reflects a fundamental difference in feedback structure—agents can recover through trial and error when immediate feedback is available, but are nearly incapable of self-correction without it. This insight has broad implications for the design of all agent systems.
- **Practical value of the time-accelerated simulator**: Beyond evaluation, the paper proposes using the simulator as a pre-validation environment for agents—testing scheduling plans in the time-accelerated simulator before submitting them to the real environment. This offers a concrete path toward addressing the delayed feedback problem.
- **Adoption of the Matter protocol is a well-motivated design choice**: It ensures consistency between simulated and real device behavior, giving benchmark results practical reference value beyond a purely synthetic setting.

## Limitations & Future Work
- The environmental model is relatively simple—it only models device–environment interactions within a single room and does not account for cross-room effects (e.g., a living room air conditioner affecting a bedroom) or cross-device interactions (e.g., combined effects of a humidifier and air conditioner).
- While 17 device types cover common household appliances, more complex IoT devices (e.g., security systems, voice assistant integrations) are not included.
- Infeasible episode design primarily covers three categories (non-existent device / physical limits / temporal contradiction); real-world scenarios involve additional constraints such as energy consumption limits and conflicting user preferences.
- The benchmark scale of 600 episodes is relatively small, with only 50 episodes per evaluation category, which may result in high statistical variance.
- Evaluation relies on LLM-as-a-Judge for certain categories, and the accuracy of the judge may affect results.

## Related Work & Insights
- **vs. HomeBench**: HomeBench evaluates agents by comparing API call sequences without simulating environmental dynamics. SimuHome provides an interactive environment that supports multiple valid action sequences.
- **vs. SAGE**: SAGE allows device states to change dynamically but does not simulate the continuous evolution of environmental variables and does not support temporal scheduling evaluation.
- **vs. AI2-THOR/ALFRED**: These are 3D environment benchmarks for physical navigation and object manipulation, which are fundamentally different from smart home API control tasks.
- **vs. Sasha**: Sasha focuses on creative intent interpretation (e.g., "make the environment more comfortable") and evaluates through user surveys. SimuHome's evaluation is more objective (simulator-verified) and covers a broader range of task types.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The first smart home benchmark to simulate continuous device–environment interactions, support time acceleration, and evaluate workflow scheduling; the problem formulation is systematic and thorough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 18 models, 6 query types × feasible/infeasible variants, detailed error analysis, and multiple improvement strategies (SFT / framework substitution / multi-turn interaction / self-correction), with exceptionally thorough analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical flow is clear and coherent, connecting simulator design, benchmark construction, and experimental findings in a well-structured narrative; the analysis of the delayed feedback mechanism is particularly insightful.
- Value: ⭐⭐⭐⭐ — Makes an important contribution to smart home agent research, though the domain is relatively specialized and its generality is narrower than more broadly applicable works.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ICLR 2026\] VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Reasoning](videomind_a_chain-of-lora_agent_for_temporal-grounded_video_reasoning.md)
- [\[ICLR 2026\] A Benchmark for Deep Information Synthesis (DeepSynth)](a_benchmark_for_deep_information_synthesis.md)
- [\[AAAI 2026\] SoMe: A Realistic Benchmark for LLM-based Social Media Agents](../../AAAI2026/llm_agent/some_a_realistic_benchmark_for_llm-based_social_media_agents.md)

</div>

<!-- RELATED:END -->
