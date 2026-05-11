---
title: >-
  [Paper Note] EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation
description: >-
  [AAAI 2026][LLM Agent][Device-Cloud Collaboration] This paper proposes EcoAgent, a closed-loop device-cloud collaborative multi-agent framework for mobile automation. By combining Dual-ReACT two-level reasoning and plann…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "Device-Cloud Collaboration"
  - "Multi-Agent System"
  - "Mobile Automation"
  - "Privacy Protection"
  - "Dual-ReACT"
date: 2026-05-08
content_hash: a28124262ba1dbc5
---

# EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation

**Conference**: AAAI 2026
**arXiv**: [2505.05440](https://arxiv.org/abs/2505.05440)
**Code**: [https://github.com/Yi-Biao/EcoAgent](https://github.com/Yi-Biao/EcoAgent)
**Area**: Agent / Mobile Automation
**Keywords**: Device-Cloud Collaboration, Multi-Agent System, Mobile Automation, Privacy Protection, Dual-ReACT

## TL;DR
This paper proposes EcoAgent, a closed-loop device-cloud collaborative multi-agent framework for mobile automation. By combining Dual-ReACT two-level reasoning and planning, lightweight on-device verification feedback, and a Pre-Understanding text compression module, EcoAgent achieves success rates comparable to fully cloud-based agents on AndroidWorld while substantially reducing latency (3.9s vs. 15.3s), cloud invocations (−89%), and upstream data volume (−48.6×).

## Background & Motivation
**Background**: MLLM-based mobile agents primarily adopt three architectures—fully cloud-based (e.g., M3A with dual GPT-4o), which offer strong reasoning but high latency and cost; fully on-device (e.g., ShowUI 2B), which offer low latency but struggle with complex task planning; and open-loop device-cloud collaborative (e.g., UGround), which combine cloud planning with on-device execution but lack a closed feedback loop.

**Limitations of Prior Work**: (1) Open-loop collaboration requires frequent upload of device screenshots to the cloud for verification, exposing user privacy and increasing latency. (2) When execution errors occur, the on-device component cannot feed back to the cloud for replanning, forcing the system to continue executing incorrectly without recovery.

**Key Challenge**: Cloud-based MLLMs offer strong reasoning but incur high communication costs and privacy risks, whereas on-device MSLMs offer low latency and privacy preservation but weak reasoning. An optimal task allocation strategy between the two is needed.

**Goal**: To achieve closed-loop device-cloud collaboration—enabling the on-device component to autonomously verify execution results and provide lightweight feedback to the cloud when needed, allowing the cloud to reflect and replan accordingly.

**Key Insight**: On-device general-purpose MSLMs (not fine-tuned on large-scale GUI data), while not proficient at precise UI element grounding, are capable of understanding screen semantics and judging whether a description matches the current screen—making them suitable for lightweight execution verification.

**Core Idea**: Dual-ReACT enables the cloud to generate an execution plan with expected outcomes in one pass; the on-device lightweight model verifies each step and provides textual feedback upon failure, realizing privacy-preserving closed-loop collaboration.

## Method

### Overall Architecture
EcoAgent consists of three agents: (1) a cloud-based **Planning Agent** (GPT-4o) responsible for Dual-ReACT planning and reflective replanning; (2) an on-device **Execution Agent** (ShowUI 2B or OS-Atlas 4B) responsible for precise UI action execution; and (3) an on-device **Observation Agent** (Qwen2-VL 2B) responsible for verifying each execution step and compressing screenshots into text via the Pre-Understanding module. Workflow: the Planning Agent generates a plan with per-step expectations → the Execution Agent executes steps sequentially → the Observation Agent verifies whether each outcome matches the expectation → on success, execution continues; on failure, textual feedback is sent to the cloud → the Planning Agent reflects and replans.

### Key Designs

1. **Dual-ReACT Two-Level Reasoning and Planning**:

   - **Function**: Extends conventional ReACT into two-level global and local reasoning, generating a complete executable plan in a single pass.
   - **Mechanism**: Given user instruction $Ins$ and initial screen $S_0$, Global ReACT first decomposes the task into a sequence of sub-goals; Local ReACT then generates concrete action steps $ST_t$ and corresponding expected outcomes $EX_t$ for each sub-goal. The plan is formalized as $P_0 = \text{GlReACT}(Ins, S_0) = \{\text{LoReACT}(ST_1, EX_1), \ldots, \text{LoReACT}(ST_t, EX_t)\}$.
   - **Design Motivation**: The generated $EX_t$ is the critical enabler—it allows the on-device Observation Agent to verify execution without understanding the global task semantics, reducing the complex verification problem to a simple text-image matching task that lightweight models can handle.

2. **Pre-Understanding Module**:

   - **Function**: Compresses a screenshot into a 50–150 token textual description, replacing raw image transmission to the cloud.
   - **Mechanism**: $T_{t+1} = \text{PreUnderstand}(S_{t+1})$, where the Observation Agent (Qwen2-VL 2B) summarizes the screenshot into a 3–5 sentence functional description using a simple prompt.
   - **Design Motivation**: (1) Raw screenshots consume approximately 1,400+ tokens, whereas compressed descriptions use only 50–150 tokens, substantially reducing communication overhead and MLLM token consumption. (2) Transmitting text rather than raw images fundamentally eliminates privacy risks from screen content leakage. (3) Replanning requires only the semantic change trajectory across screens, not full visual detail.

3. **Memory + Reflection Replanning**:

   - **Function**: Upon execution failure, the Planning Agent reflects using stored screen description trajectories and failure reasons to generate a revised plan.
   - **Mechanism**: The Memory Module stores textual screen trajectories and action history. Upon failure, the Reflection Module analyzes the error trajectory and produces a new plan $P_n = \text{Reflection}(Ins, P_{n-1}, \text{Memory})$.
   - **Design Motivation**: Inspired by Reflexion-style iterative improvement, this mechanism enables the system to learn from mistakes and recover adaptively—a core advantage of closed-loop over open-loop systems.

### Loss & Training
EcoAgent does not involve end-to-end training. The Execution Agent uses existing GUI fine-tuned models (ShowUI 2B / OS-Atlas 4B); the Observation Agent uses the general-purpose Qwen2-VL 2B without fine-tuning; the Planning Agent uses GPT-4o. The entire framework is plug-and-play.

## Key Experimental Results

### Main Results
Task success rate (SR) and operational cost comparison on the AndroidWorld benchmark:

| Architecture | Agent | Base Model | SR (%) | MC (Cloud Calls) | MT (Tokens) |
|---|---|---|---|---|---|
| On-Device | ShowUI | ShowUI 2B | 7.0 | 0 | 0 |
| On-Device | V-Droid | V-Droid 8B | 59.5 | – | – |
| Cloud | AppAgent | GPT-4o | 11.2 | 6.46 | 15309 |
| Cloud | M3A | GPT-4o×2 | 28.4 | 13.39 | 87469 |
| Open-Loop | UGround-2B | GPT-4o×2+2B | 32.8 | 12.21 | 45192 |
| **Closed-Loop** | **EcoAgent (OS-Atlas)** | GPT-4o+4B+2B | **27.6** | **1.53** | **3240** |

Latency comparison:

| Agent | Latency (s) |
|---|---|
| M3A | 15.3 |
| UGround-2B | 18.2 |
| **EcoAgent (ShowUI)** | **3.9** |

### Ablation Study

| Configuration | SR (%) | MC | MT | Notes |
|---|---|---|---|---|
| Execution Agent only (ShowUI) | 7.0 | 0 | 0 | No planning capability |
| + Planning Agent | 15.5 | 1 | 2149 | +8.5%, Dual-ReACT effective |
| + Observation Agent | **25.6** | 1.87 | 3545 | +10.1%, closed-loop feedback critical |

### Key Findings
- EcoAgent achieves success rates comparable to M3A (27.6% vs. 28.4%) while reducing cloud invocations by 89% and token consumption by 96%.
- Upstream data volume is only 120 kB/task, 48.6× lower than M3A's 5,831 kB, demonstrating significant privacy protection.
- Failure analysis reveals that 50% of failures stem from visual grounding errors (MSLM limitations) and 35% from planning errors, indicating that EcoAgent's performance ceiling is primarily constrained by on-device model capability—stronger edge models can directly improve results.
- The Observation Agent contributes the largest gain: from 15.5% to 25.6% (+10.1%), confirming that closed-loop feedback is the key design element.

## Highlights & Insights
- The "expectation" design in Dual-ReACT is particularly elegant: it downgrades the verification problem from "understanding whether the task is complete" to "judging whether the screen matches a description," enabling a 2B on-device model to perform verification—a critical enabling technique for closed-loop operation.
- The Pre-Understanding Module achieves three objectives simultaneously: cost reduction, privacy protection, and latency reduction. The approach of compressing visual information into text is generalizable to any agent scenario requiring device-cloud communication.
- The framework is model-agnostic and can be readily upgraded with stronger on-device models (e.g., V-Droid 8B as the Execution Agent), offering strong scalability.

## Limitations & Future Work
- The absolute success rate of 27.6% remains modest; V-Droid (fully on-device, 8B) achieves 59.5%, indicating that the performance ceiling is constrained by on-device model capability.
- The Pre-Understanding module's information compression may discard critical details, leaving the cloud with insufficient information for replanning.
- Evaluation is currently limited to AndroidWorld (116 tasks); validation at larger scale and across more application scenarios is lacking.
- On-device inference latency (~3.9s/step) remains high for real-world user interaction.

## Related Work & Insights
- **vs. M3A**: M3A employs two GPT-4o instances for planning and verification; EcoAgent achieves comparable performance with one GPT-4o and two 2–4B on-device models, reducing cost by an order of magnitude. The closed-loop design is the key differentiator.
- **vs. UGround**: UGround is an open-loop device-cloud collaboration system that still requires frequent screenshot uploads to the cloud for verification. EcoAgent enables on-device verification through the expectation mechanism in Dual-ReACT, avoiding privacy leakage.
- **vs. V-Droid**: V-Droid is a fully on-device 8B model with the highest success rate (59.5%) but limited capacity for complex long-horizon planning. Integrating V-Droid as EcoAgent's Execution Agent is a promising direction.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of closed-loop device-cloud collaboration, Dual-ReACT, and Pre-Understanding is novel, though individual modules offer limited standalone innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three-dimensional evaluation (success rate / cost / latency) combined with ablation studies, failure analysis, and system overhead analysis is relatively comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture and comparison figures are clear; problem motivation is well articulated.
- **Value**: ⭐⭐⭐⭐ Provides a practical low-cost solution for real-world deployment of mobile agents with high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis](coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)
- [\[CVPR 2026\] CarePilot: A Multi-Agent Framework for Long-Horizon Computer Task Automation in Healthcare](../../CVPR2026/llm_agent/carepilot_a_multi-agent_framework_for_long-horizon_computer_task_automation_in_h.md)
- [\[AAAI 2026\] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)
- [\[AAAI 2026\] COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)
- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)

</div>

<!-- RELATED:END -->
