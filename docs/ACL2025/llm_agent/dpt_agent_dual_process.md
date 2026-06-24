---
title: >-
  [Paper Note] Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration
description: >-
  [ACL 2025][LLM Agent][Dual Process Theory] DPT-Agent is proposed, which is the first method to systematically integrate Dual Process Theory into a language agent framework. It employs a Finite State Machine (FSM) + code-as-policy as the fast, intuitive System 1, and an LLM with Theory of Mind (ToM) + asynchronous reflection as the slow, deliberative System 2. This achieves autonomous, real-time simultaneous human-AI collaboration for the first time (in a challenging version o…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Dual Process Theory"
  - "Real-time Collaboration"
  - "Human-AI Collaboration"
  - "Theory of Mind"
  - "Finite State Machine"
date: 2026-05-08
content_hash: f137b45aacfce73c
---

# Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration

**Conference**: ACL 2025  
**arXiv**: [2502.11882](https://arxiv.org/abs/2502.11882)  
**Code**: [https://github.com/sjtu-marl/DPT-Agent](https://github.com/sjtu-marl/DPT-Agent)  
**Area**: LLM Agent  
**Keywords**: Dual Process Theory, Real-time Collaboration, Human-AI Collaboration, Theory of Mind, Finite State Machine  

## TL;DR
DPT-Agent is proposed, which is the first method to systematically integrate Dual Process Theory into a language agent framework. It employs a Finite State Machine (FSM) + code-as-policy as the fast, intuitive System 1, and an LLM with Theory of Mind (ToM) + asynchronous reflection as the slow, deliberative System 2. This achieves autonomous, real-time simultaneous human-AI collaboration for the first time (in a challenging version of Overcooked).

## Background & Motivation
**Background**: LLM agents excel in turn-by-turn human-AI collaboration (e.g., writing, coding), but face severe challenges in tasks requiring real-time synchronization (e.g., collaborative operations in shared spaces).

**Limitations of Prior Work**: (a) **Latency issues**—Large models possess strong reasoning capabilities but suffer from high latency (e.g., o3-mini takes too long to think), while small models respond quickly but have poor capabilities (scoring efficiency close to zero); (b) **Lack of autonomy**—Most collaborative frameworks still rely on human input to act, failing to proactively infer and adapt to human intentions; (c) **Poor policy adaptability**—LLMs struggle to cope with the dynamically changing policies of human collaborators.

**Key Challenge**: The irreconcilable conflict between performance and latency—models with strong reasoning capabilities are too slow to respond in real-time, whereas models that respond quickly lack the capability to act effectively.

**Goal** To achieve autonomous, real-time simultaneous human-AI collaboration while ensuring real-time responsiveness.

**Key Insight**: Operationalizing Dual Process Theory from cognitive psychology—System 1 uses FSM to achieve millisecond-level decision-making (independent of LLM inference), and System 2 uses an LLM for asynchronous deep thinking (without blocking action). The two systems run in parallel and guide each other.

**Core Idea**: FSM for fast actions + LLM for asynchronous thinking + ToM for inferring human intentions = real-time autonomous human-AI collaboration.

## Method

### Overall Architecture
DPT-Agent consists of two parallel systems: (1) **System 1 (Fast System)**—low-level action decision-making driven by FSM, which encodes System 2's policy guidance using code-as-policy to achieve millisecond-level responsiveness; (2) **System 2 (Slow System)**—high-level reasoning driven by LLM, comprising Theory of Mind (ToM) to infer human intentions and asynchronous reflection to learn and improve policies from experience, passing reasoning results to System 1 in the form of code.

### Key Designs

1. **FSM + Code-as-Policy's System 1**:

    - **Function**: Makes real-time action decisions without waiting for LLM inference
    - **Mechanism**: Predefines a finite state machine to describe basic behavioral patterns (e.g., "retrieve ingredients -> process -> serve food"). System 2 dynamically modifies FSM state transition rules by generating control code (code-as-policy), thereby indirectly controlling System 1's behavior.
    - **Design Motivation**: The state transitions of FSM have O(1) complexity, which is significantly faster than LLM inference. Code-as-policy allows the results of slow thinking to be immediately executed by the fast action system.

2. **Theory of Mind (ToM)'s System 2**:

    - **Function**: Proactively infers the intentions and strategies of human collaborators
    - **Mechanism**: The LLM infers the human's current goal and preferred policy based on observed human action trajectories, adjusting its own policy accordingly to cooperate—for instance, if it infers "the human is preparing soup ingredients", it will deliver the previous round of food.
    - **Design Motivation**: In real-world collaboration, humans do not explicitly state their strategies to the AI—the AI needs to infer human intentions just as humans infer those of their partners.

3. **Asynchronous Reflection Mechanism**:

    - **Function**: Learns from experience without blocking actions
    - **Mechanism**: The reflection process of System 2 is executed asynchronously in the background. While System 1 acts according to the current policy, System 2 analyzes recent action sequences and environmental feedback, identifies inefficient patterns, and updates the policy code. Once updated, the code is automatically pushed to System 1.
    - **Design Motivation**: Synchronous reflection blocks action and increases latency; the asynchronous design enables learning and action to run in parallel.

### Loss & Training
- **Training-free**—a pure inference-time framework
- Consists of a predefined FSM for System 1 and dynamic updates of LLM-generated code for System 2
- Supports various LLM backbones (GPT-4o, Llama-3.3-70B, DeepSeek-R1, etc.)

## Key Experimental Results

### Main Results (Collaborating with Rule Agents, Overcooked Challenging Edition)

| Method | Map 1 Score | Map 2 Score | Description |
|------|----------|----------|------|
| FSM (Upper Bound Ref) | High | High | Hardcoded optimal |
| ReAct (GPT-4o) | Mid-Low | Mid-Low | Latency causes missed opportunities |
| Reflexion (GPT-4o) | Mid | Mid | Reflection improves but still limited by latency |
| **DPT-Agent (GPT-4o)** | **Highest** | **Highest** | Synergy between fast and slow systems |

### Human-AI Collaboration Experiments

| Method | Map 1 Score | Map 2 Score | Human Perception Rank |
|------|----------|----------|-----------|
| ReAct | Low | Low | Low |
| Reflexion | Mid | Mid-Low | Mid |
| DPT-Agent w/o ToM | Mid-High | Mid | Mid-High |
| **DPT-Agent** | **Highest** | **Highest** | **Highest** |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| w/o ToM | Performance degradation (especially when collaborating with humans) | ToM is crucial for adapting to human policies |
| w/o Asynchronous Reflection | Performance degradation | Unable to improve from experience |
| Reasoning model (o3-mini) in conventional framework | Extremely low score | Latency too high; longer thinking leads to fewer actions |
| Reasoning model (DeepSeek-R1) + DPT-Agent | **Significant performance recovery** | The DPT framework effectively converts slow thinking into fast actions |

### Key Findings
- **DPT-Agent is the first agent framework to successfully achieve real-time simultaneous human-AI collaboration in the challenging version of Overcooked**
- Reasoning models (o3-mini, DeepSeek-R1) fail due to latency when used independently, but experience a substantial performance recovery under the DPT-Agent framework—proving the capability of the DPT framework to "translate proper thinking into effective actions."
- The ToM module contributes the most during human collaboration—inferring human intentions enables the agent to cooperate proactively rather than passively waiting for instructions.
- Human evaluators consistently ranked DPT-Agent highest in subjective rankings, validating the improvement in actual collaboration experience.
- FSM as System 1 achieves millisecond-level responsiveness, completely resolving the latency bottleneck.

## Highlights & Insights
- **The full operationalization of Dual Process Theory into an agent architecture** is the core contribution. Instead of a simple "large and small model combination", it is a heterogeneous synergy of FSM (non-LLM) + LLM, truly realizing a split between 'fast' and 'slow' processes. Prior works using small LLMs for System 1 were still limited by LLM inference latency.
- **Using code-as-policy as the interface between the fast and slow systems** is highly ingenious. System 2 generates code to modify System 1's FSM rules, facilitating the transmission mechanism of "slow thinking guiding fast actions."
- **Asynchronous reflection** eliminates the issue of "actions stopping while thinking", which is critical in real-time tasks.
- **The combination of reasoning models + DPT-Agent** reveals a new paradigm for using reasoning models—instead of letting them act directly, they are utilized to provide policy guidance asynchronously.
- The "functional" implementation of ToM (a closed-loop of inference -> decision-making -> action) goes beyond the purely "deliberative" ToM of previous LLMs, which made inferences but failed to apply them to decisions.

## Limitations & Future Work
- FSM requires manual state space design for each new task, limiting generality.
- Currently only validated in the Overcooked environment; more complex real-world tasks require validation.
- The accuracy of ToM depends on the reasoning capability of the LLM—the ToM of certain models (e.g., Llama-70B) actually degrades performance.
- The update frequency of asynchronous reflection requires manual adjustment.
- Simultaneous collaboration between multiple AI entities remains unexplored (only human-AI collaboration was studied).

## Related Work & Insights
- **vs ReAct/Reflexion**: Synchronous reasoning + action leads to unacceptable latency; DPT-Agent's asynchronous design resolves this issue.
- **vs Large-Small Model Combination (Liu et al. 2024)**: Previous works using small LLMs for System 1 still suffered from latency; DPT-Agent completely eliminates the inference latency of System 1 by using FSM.
- **vs VirSci (Multi-Agent Idea Generation)**: VirSci's multi-agent collaboration is non-real-time; DPT-Agent addresses collaboration under real-time constraints.
- It provides direct reference value for real-time decision-making scenarios such as game AI, robot collaboration, and autonomous driving.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first to fully operationalize Dual Process Theory, featuring a unique heterogeneous cooperative design of FSM + LLM.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Combines rule agents + real human experiments + subjective evaluations + ablations + multi-model comparisons.
- Writing Quality: ⭐⭐⭐⭐ Data-driven motivation analysis (the latency-performance analysis in Figure 2), with clear framework diagrams.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to real-time human-AI collaboration and LLM Agent architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration](../../ICLR2026/llm_agent/collaborative_gym_a_framework_for_enabling_and_evaluating_human-agent_collaborat.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] EMULATE: A Multi-Agent Framework for Determining the Veracity of Atomic Claims by Emulating Human Actions](emulate_a_multi-agent_framework_for_determining_the_veracity_of_atomic_claims_by.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)

</div>

<!-- RELATED:END -->
