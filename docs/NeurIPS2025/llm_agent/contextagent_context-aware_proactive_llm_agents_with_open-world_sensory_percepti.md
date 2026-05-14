---
title: >-
  [Paper Note] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions
description: >-
  [NeurIPS 2025][LLM Agent][Proactive Agent] This paper proposes ContextAgent, the first LLM agent framework that leverages multimodal sensory perception from wearable devices (video + audio + notifications) to understand…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Proactive Agent"
  - "context awareness"
  - "wearable sensors"
  - "tool invocation"
  - "benchmark"
date: 2026-05-08
content_hash: 5e651d80db38b063
---

# ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions

**Conference**: NeurIPS 2025
**arXiv**: [2505.14668](https://arxiv.org/abs/2505.14668)
**Code**: [https://github.com/openaiotlab/ContextAgent](https://github.com/openaiotlab/ContextAgent)
**Area**: Agent
**Keywords**: Proactive Agent, context awareness, wearable sensors, tool invocation, benchmark

## TL;DR
This paper proposes ContextAgent, the first LLM agent framework that leverages multimodal sensory perception from wearable devices (video + audio + notifications) to understand user intent and proactively deliver tool-augmented services. It also introduces ContextAgentBench, a benchmark of 1,000 samples, achieving improvements of 8.5% in proactive prediction accuracy and 6.0% in tool invocation accuracy.

## Background & Motivation

**Background**: Current LLM agents predominantly operate in a "passive response" mode, executing tasks only upon explicit user instruction. A small number of pioneering works have attempted to build proactive agents, but these are largely confined to desktop UI environments (e.g., monitoring screenshots and keyboard inputs).

**Limitations of Prior Work**: Existing proactive agents suffer from two major shortcomings: ① a narrow perceptual scope, limited to closed environments (desktop interfaces) and unable to perceive the open-world contexts of daily life; and ② limited functionality, relying on direct LLM inference to generate text replies without invoking external tools to provide richer services.

**Key Challenge**: Realizing a truly proactive personal assistant requires an agent to continuously perceive the surrounding environment through multiple senses (vision, hearing)—much like a human—and to understand user intent and deliver assistance automatically without requiring user action. Existing approaches fall critically short in both perceptual scope and service capability.

**Goal**: This work investigates how an LLM agent can leverage multimodal sensory data from wearable devices (smart glasses, earphones, etc.) to understand user context and intent in an open-world setting, determine whether proactive assistance is warranted, and automatically invoke appropriate tools to fulfill that assistance.

**Key Insight**: The authors observe that devices such as smart glasses and earphones offer a hands-free interaction modality whose egocentric perspective is co-located with the user, making them ideal perceptual front-ends for proactive agents. Incorporating user persona information further enables personalized service decisions.

**Core Idea**: Acquire multi-dimensional sensory context via wearable devices along with user persona, then employ a CoT fine-tuned LLM for "think-before-act" proactive reasoning and tool invocation.

## Method

### Overall Architecture
ContextAgent takes as input multimodal sensory data $\mathcal{S}$ collected from wearable devices—egocentric video $\mathcal{S_V}$, audio $\mathcal{S_A}$, and smartphone notifications $\mathcal{N}$—together with a user persona $\mathcal{P}$. The system operates in two stages: ① proactive-oriented context extraction, which transforms raw sensory data into structured perceptual context $\mathcal{C}$; and ② context-aware reasoning, in which an SFT-trained LLM performs inference to produce a chain-of-thought $\mathcal{T}$, a proactivity score $\mathcal{P_S}$ (on a scale of 1–5), a tool-call chain $\mathcal{T_C}$, and a final response $\mathcal{R}$.

### Key Designs

1. **Proactive-oriented Context Extraction**:

    - **Function**: Extracts key cues from raw sensory data that are useful for determining whether proactive service is needed.
    - **Mechanism**: A VLM (e.g., Qwen-2.5-VL) combined with in-context learning (ICL) is used to generate proactive-oriented visual context $\mathcal{C_V}$ from egocentric video; a speech recognition model generates audio context $\mathcal{C_A}$; these are then integrated as $\mathcal{C} = [\mathcal{C_V}, \mathcal{C_A}, \mathcal{N}]$.
    - **Design Motivation**: Zero-shot VLMs tend to produce overly generic or redundant descriptions that omit details critical for proactive service decisions (e.g., specific exercise equipment in a gym). The ICL approach guides the VLM to generate more targeted descriptions. Ablation experiments show that zero-shot context extraction yields a 3.0% lower Acc-P compared to ICL.

2. **Persona Context**:

    - **Function**: Incorporates information about the user's identity, preferences, and behavioral history.
    - **Mechanism**: User persona is extracted from historical sensory data (e.g., conversation logs) using an LLM, and is provided as part of the input at inference time.
    - **Design Motivation**: The necessity of proactive service is highly dependent on individual user preferences (e.g., a health-conscious user may need nutritional advice in a restaurant, whereas a general user may not). Ablation results show that removing the persona leads to a 9.0% drop in Acc-P and a 12.3% drop in F1.

3. **Context-aware Reasoner**:

    - **Function**: Integrates perceptual context and user persona to reason about whether proactive service is needed and which tools to invoke.
    - **Mechanism**: Trained via SFT with CoT supervision, where chain-of-thought data is distilled from Claude-3.7-Sonnet. At inference time, the model first generates an explicit reasoning process within `<think>...</think>` tags, then outputs the proactivity score $\mathcal{P_S}$ and the tool-call chain. Proactive service is triggered when $\mathcal{P_S} \geq \theta$. LoRA rank=8, lr=0.0001, trained for 5 epochs.
    - **Design Motivation**: Direct LLM inference performs poorly when mapping sensory context to tool invocations. CoT-based "think-before-act" training significantly improves both proactive prediction accuracy and tool-call correctness.

### Loss & Training
Standard SFT loss is applied on training data $\mathcal{D}_{SFT} = \{(\mathcal{X}, \mathcal{T}, \mathcal{Y})\}$, where $\mathcal{X}$ contains perceptual context and user persona, $\mathcal{T}$ is the chain-of-thought distilled from a stronger LLM, and $\mathcal{Y}$ comprises the proactivity score and tool-call chain. LoRA is used for parameter-efficient fine-tuning, with 8× A6000 GPUs and the AdamW optimizer.

## Key Experimental Results

### Main Results
Evaluated on ContextAgentBench using Llama-3.1-8B-Instruct as the base model:

| Method | Acc-P↑ | MD↓ | FD↓ | RMSE↓ | F1↑ | Acc-Args↑ |
|--------|--------|-----|-----|-------|-----|-----------|
| Proactive Agent | 0.676 | 0.017 | 0.306 | 1.915 | 0.318 | 0.081 |
| ICL-All | 0.757 | 0.229 | 0.012 | 1.872 | 0.582 | 0.270 |
| Vanilla SFT | 0.813 | 0.068 | 0.117 | 1.572 | 0.580 | 0.405 |
| **ContextAgent** | **0.874** | **0.030** | **0.095** | **1.408** | **0.626** | **0.448** |

With Qwen2.5-7B as the backbone, ContextAgent achieves 0.894 Acc-P, 0.645 F1, and 0.459 Acc-Args, reaching performance on par with GPT-4o (0.886 Acc-P, 0.639 F1) and 70B-scale models.

### Ablation Study

| Configuration | Acc-P | F1 | Acc-Args | Notes |
|---------------|-------|----|----------|-------|
| Full (DeepSeek-R1-7B) | 0.888 | 0.647 | 0.468 | Complete model |
| w/o vision | 0.709 | 0.414 | 0.163 | Acc-P −17.9%, tool calls −23.3% |
| w/o audio | 0.720 | 0.493 | 0.212 | Smaller drop than removing vision |
| w/o persona | ~0.78 | ~0.50 | ~0.32 | Acc-P −9%, F1 −12.3% |
| Zero-shot VLM | ~0.86 | ~0.61 | ~0.45 | Context extraction Acc-P −3% |

### Key Findings
- **Vision is the most critical modality**: Removing visual context causes the largest performance degradation (Acc-P −17.9%), confirming that egocentric visual perception is essential for understanding the user's environment.
- **User persona is indispensable**: Removing the persona leads to a 12.6% drop in Acc-Args, demonstrating that personalized information is critical for both service necessity judgments and tool selection.
- **Strong OOD generalization**: The model achieves 90.9% Acc-P and 68.9% F1 on unseen scenarios, surpassing the best baseline by 8.3% Acc-P.
- **Small models remain competitive**: 7B/8B models trained with ContextAgent match or exceed the baseline performance of 70B models and GPT-4o.

## Highlights & Insights
- **Novel problem formulation**: This is the first work to incorporate wearable device sensory data into the proactive agent paradigm, filling a research gap at the intersection of open-world perception and tool-augmented proactive service.
- **Task-oriented ICL context extraction**: Rather than simply prompting the VLM to describe a scene, the proposed approach uses ICL to steer the VLM toward cues relevant to proactive service decisions—a principle transferable to any scenario requiring task-oriented perception.
- **Complete benchmark construction methodology**: The data construction pipeline—200 human-authored seeds → LLM-based expansion → human validation → format verification—offers a replicable reference for future benchmark development.

## Limitations & Future Work
- **Text-mediated sensory context**: In practical deployment, video and audio must first be converted to textual descriptions, and the resulting information loss is unavoidable. End-to-end processing with a multimodal LLM could be a promising alternative.
- **Limited benchmark scale**: With only 1,000 samples, 9 scenarios, and 20 tools, the benchmark falls short of the diversity encountered in real-world deployments.
- **Privacy and ethical concerns underexplored**: Continuous video and audio recording by wearable devices poses serious privacy risks that present significant challenges for real-world deployment.
- **Fixed proactivity threshold $\theta$**: While the paper notes that users can adjust the threshold, no adaptive tuning strategy is provided.
- **Fixed tool set**: The 20 predefined tools constrain the scope of services, and the framework lacks a mechanism for dynamic tool discovery or composition.

## Related Work & Insights
- **vs. Proactive Agent [Li et al.]**: Restricted to desktop UI environments, relies on direct LLM inference, and lacks tool invocation capability. ContextAgent extends to open-world sensing and supports tool chains, achieving approximately 20% higher Acc-P.
- **vs. CodingGenie**: A desktop proactive agent focused on programming assistance requiring keyboard and screen inputs. ContextAgent targets general daily-life scenarios.
- **vs. Traditional rule-based proactive services** (e.g., Apple Watch fall detection): Rule-triggered versus LLM-reasoning-based; ContextAgent is more flexible but incurs higher computational cost.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to propose a systematic framework combining open-world proactive agents with tool augmentation; however, the core techniques (SFT + CoT + ICL) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 13 LLMs, 6 baselines, and comprehensive ablation/OOD experiments, though real physical deployment experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, experimental organization is systematic, and the appendix provides sufficient detail.
- **Value**: ⭐⭐⭐⭐ Opens a new direction of "perception-augmented proactive agents," and the benchmark will facilitate follow-up research, though practical deployment remains distant.

## Key Experimental Results

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](../../ACL2026/llm_agent/toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)
- [\[ICCV 2025\] Less is More: Empowering GUI Agent with Context-Aware Simplification](../../ICCV2025/llm_agent/less_is_more_empowering_gui_agent_with_context-aware_simplification.md)
- [\[NeurIPS 2025\] AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents](agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](../../ICLR2026/llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[NeurIPS 2025\] EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law](eu-agent-bench_measuring_illegal_behavior_of_llm_agents_under_eu_law.md)

</div>

<!-- RELATED:END -->
