---
title: >-
  [Paper Note] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer
description: >-
  [NEURIPS2025][LLM Agent][video anomaly detection] This paper proposes PANDA, an agentic AI engineer framework built upon MLLMs, which achieves training-free and human-intervention-free generalist video anomaly detection through four core capabilities: adaptive scene-aware strategy planning, goal-driven heuristic reasoning, tool-augmented self-reflection, and chain-of-memory.
tags:
  - NEURIPS2025
  - LLM Agent
  - video anomaly detection
  - agentic AI
  - RAG
  - self-reflection
  - chain-of-memory
date: 2026-05-08
content_hash: 086d976409e5156a
---

# PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer

**Conference**: NEURIPS2025
**arXiv**: [2509.26386](https://arxiv.org/abs/2509.26386)
**Code**: [GitHub](https://github.com/showlab/PANDA)
**Area**: LLM Agent
**Keywords**: video anomaly detection, agentic AI, RAG, self-reflection, chain-of-memory

## TL;DR

This paper proposes PANDA, an agentic AI engineer framework built upon MLLMs, which achieves training-free and human-intervention-free generalist video anomaly detection through four core capabilities: adaptive scene-aware strategy planning, goal-driven heuristic reasoning, tool-augmented self-reflection, and chain-of-memory.

## Background & Motivation

**Existing Problems**: Video anomaly detection (VAD) methods can be categorized into training-dependent and training-free paradigms; both require human involvement for new scenes (e.g., labeled data, handcrafted prompt templates, or post-processing rules), precluding true generalization.

**Limitations of Prior Work (Training-dependent)**: Semi-supervised, weakly supervised, and instruction-tuned methods suffer severe performance degradation under out-of-distribution environments and novel anomaly types, and incur high annotation and training costs.

**Limitations of Prior Work (Training-free)**: Although pre-trained LLMs/VLMs eliminate training, these methods still rely on manually designed preprocessing steps, prompt templates, rule management, and post-processing pipelines, resulting in static pipelines that lack adaptive capacity.

**Root Cause**: Under complex conditions such as low resolution, poor illumination, high noise, and long-duration anomalies, conventional methods consistently fail.

**Opportunity from the Agentic Paradigm**: Analogous to how human engineers systematically analyze problems, adapt to complex environments, and iteratively improve through tool use and accumulated experience, an agent framework can achieve autonomous perception, strategy formulation, reasoning, tool invocation, and self-improvement.

**Paper Goals**: Realize Generalist VAD — automatically handling arbitrary scenes and anomaly types with zero training data and zero human intervention.

## Method

### Overall Architecture

PANDA is built on LangGraph, employing a VLM (Qwen2.5VL-7B) for perception and reasoning and an MLLM (Gemini 2.0 Flash) for planning and reflection. The workflow proceeds as: (1) environment perception + RAG-based strategy planning → (2) goal-driven heuristic reasoning → (3) tool-augmented self-reflection triggered upon uncertainty → (4) chain-of-memory for continuous experience accumulation. Both offline and online inference modes are supported.

### Key Designs 1: Adaptive Scene-Aware Strategy Planning

**Environment Perception**: $M$ keyframes are uniformly sampled and, together with the user query, composed into a perception prompt fed to the VLM to obtain structured environmental information:
- Scene Overview (scene type and activities)
- Potential Anomalies (possible anomaly types for the scene)
- Weather Condition (time and weather)
- Video Quality (resolution and clarity)

**RAG-Based Strategy Planning**:
1. Based on the user query, the MLLM generates a structured anomaly knowledge base $\kappa_a$ (event types + anomaly rules + applicable scenes), with $H=20$ rule–scene pairs predefined per anomaly category.
2. Environmental information is used as a query to retrieve the top-$k=5$ most relevant anomaly rules via FAISS.
3. The user query, environmental information, and retrieved rules are integrated; the MLLM then generates a detection strategy plan comprising: preprocessing steps, a list of potential anomalies, and heuristic reasoning prompts.

### Key Designs 2: Goal-Driven Heuristic Reasoning

Guided by the strategy plan, the VLM performs reasoning on each video segment ($s=5$ frames per segment):

- **Input**: augmented video segment + visual memory + reasoning prompt (containing potential anomalies, rules, heuristic prompts, and reflection information)
- **Short-term Memory**: textual reasoning records and corresponding visual frames from the most recent $l=5$ steps
- **Output**: triplet \{Status, Score, Reason\}
    - Status: Normal / Abnormal / Insufficient
    - Score: anomaly probability in $[0,1]$
    - Reason: rationale provided by the VLM
- **Key Mechanism**: when Status is Insufficient, the reflection module is triggered to acquire additional information before re-reasoning

### Key Designs 3: Tool-Augmented Self-Reflection

Activated when the reasoning result is Insufficient, equipped with a tool set $\tau$ (image deblurring, denoising, brightness enhancement, image retrieval, object detection, web search, etc.).

**Experience-Driven Reflection**:
1. The long-term chain-of-memory (Long CoM) is queried to retrieve the most similar historical reflection cases.
2. A reflection prompt is constructed (incorporating user query, environmental information, strategy, rules, short-term memory, uncertainty reason, and historical experience).
3. The MLLM analyzes the source of uncertainty and generates a reflection plan: tools to invoke with their parameters + new anomaly rules + new heuristic prompts.

**Tool Invocation and Refined Reasoning**:
- Tools specified in the reflection plan are executed to obtain textual enhancements (detection results, search results) and visual enhancements (processed frames + retrieved historical keyframes).
- The reasoning prompt is updated with the augmented information, and the VLM re-reasons.
- A maximum of $r=3$ reflection rounds is allowed; if Insufficient persists, a default score is assigned and the segment is skipped.

### Key Designs 4: Self-Improving Chain-of-Memory (CoM)

- **Short-term CoM**: records textual and visual memory from the most recent $l$ steps during reasoning; records historical reflection outputs during reflection.
- **Long-term CoM**: maintains a complete decision trajectory across time steps, $\text{Long-CoM} = \{M_1, M_2, \ldots, M_T\}$, where each $M_t = \{\text{Result}_\text{reasoning}, \text{Result}_\text{reflection}, \text{Result}_\text{reasoning}^\text{refined}\}$.
- **Role**: Long CoM is initially empty and relies on short-term memory; as more segments are processed, long-term memory progressively accumulates, supporting memory-consistent reasoning and reflection planning.

### Loss & Training

PANDA is a training-free framework with no learnable loss functions. The core components are RAG retrieval (all-MiniLM-L6-v2 encoding + FAISS indexing) and prompt engineering for VLM/MLLM.

## Key Experimental Results

### Main Results: Multi-Scene / Open-Set / Complex-Scene Performance Comparison

| Method | Supervision | Explainable | Human-Free | UCF-Crime AUC% | XD-Violence AP% | UBnormal AUC% | CSAD AUC% |
|--------|-------------|-------------|------------|----------------|-----------------|---------------|-----------|
| HL-Net | Weakly supervised | ✗ | ✗ | 82.44 | 73.67 | - | - |
| RTFM | Weakly supervised | ✗ | ✗ | 84.30 | 77.81 | 64.94 | - |
| VadCLIP | Weakly supervised | ✗ | ✗ | 88.02 | 84.51 | - | - |
| Holmes-VAU | Instruction-tuned | ✓ | ✗ | 88.96 | 87.68 | 56.77 | 72.47 |
| LAVAD | Training-free | ✓ | ✗ | 80.28 | 62.01 | 64.23 | 57.26 |
| AnomalyRuler | Training-free | ✓ | ✗ | - | - | 71.90 | - |
| **PANDA (Offline)** | **Training-free** | **✓** | **✓** | **84.89** | **70.16** | **75.78** | **73.12** |
| PANDA (Online) | Training-free | ✓ | ✓ | 82.57 | 63.57 | 72.41 | 71.25 |

PANDA achieves comprehensive state-of-the-art performance among all training-free methods; surpasses all training-based methods on the UBnormal open-set benchmark; and outperforms Holmes-VAU (an instruction-tuned method) on the complex-scene CSAD benchmark.

### Ablation Study: Contribution of PANDA's Core Capabilities

| Planning | Reflection | Memory | UCF-Crime AUC% |
|----------|------------|--------|----------------|
| ✗ | ✗ | ✗ | 75.25 |
| ✓ | ✗ | ✗ | 80.37 (+5.12%) |
| ✓ | ✓ | ✗ | 82.63 (+2.26%) |
| ✓ | ✓ | ✓ | **84.89 (+2.26%)** |

### Hyperparameter Analysis (UCF-Crime)

| Reflection rounds $r$ | AUC% | Rule count $k$ | AUC% | Short-term memory length $l$ | AUC% |
|-----------------------|------|----------------|------|------------------------------|------|
| 1 | 83.83 | 1 | 82.79 | 1 | 82.92 |
| **3** | **84.89** | **5** | **84.89** | **5** | **84.89** |
| 5 | 84.91 | 9 | 83.92 | 9 | 84.03 |

### Key Findings

1. **Strategy planning yields the largest gain**: +5.12% AUC; RAG-retrieved anomaly rules combined with scene-aware strategy planning most significantly improve reasoning accuracy.
2. **Reflection and memory contribute comparably**: each +2.26%; tool-augmented reflection resolves ambiguous cases, while chain-of-memory provides cross-temporal consistency.
3. **Clear hyperparameter sweet spots**: $r=3$, $k=5$, $l=5$ achieve optimal balance; both insufficient and excessive values are detrimental (information deficit vs. noise interference).
4. **Advantage on CSAD complex scenes**: 73.12% vs. Holmes-VAU's 72.47%, demonstrating greater robustness under low-quality video and long-duration anomalies.
5. **Strongest open-set generalization**: 75.78% on UBnormal surpasses all methods, validating the Generalist positioning through detection of 12 unseen anomaly categories.

## Highlights & Insights

1. **A new paradigm for Generalist VAD**: The first approach to achieve training-free, human-intervention-free generalist video anomaly detection, pioneering the agentic paradigm in the VAD domain.
2. **Elegant combination of RAG and scene awareness**: Rather than naive RAG, the method first perceives the environment and then uses environmental information as the retrieval query, ensuring that the strategy aligns with the scene.
3. **Three-tier reasoning architecture (reasoning–reflection–refinement)**: The three-state design of Normal/Abnormal/Insufficient elegantly handles low-confidence cases, avoiding forced decisions.
4. **Dual short-term and long-term memory mechanism**: Short-term memory provides local temporal context, while long-term memory accumulates global experience, realizing a "the more it is used, the smarter it becomes" effect.
5. **Contribution of the CSAD benchmark**: A dedicated complex-scene anomaly detection benchmark (low resolution, poor illumination, high noise, long-duration anomalies) is constructed, filling an evaluation gap.

## Limitations & Future Work

1. **High inference cost**: VLM inference is required per frame; reflection additionally involves MLLM and tool calls, making real-time performance and API costs bottlenecks for practical deployment.
2. **Gap behind training-based methods on UCF-Crime and XD-Violence**: AUC 84.89% vs. VadCLIP's 88.02%, AP 70.16% vs. Holmes-VAU's 87.68%; a performance gap remains when training data is available.
3. **Fixed tool set**: The current tools (deblurring, retrieval, detection, etc.) are predefined and may lack the necessary tools for entirely novel scenarios.
4. **Dependence on a specific model combination**: Qwen2.5VL-7B + Gemini 2.0 Flash; the effectiveness and substitutability of alternative models have not been validated.

## Related Work & Insights

- **vs. LAVAD / AnomalyRuler**: PANDA upgrades from static prompts to a dynamic agent, with adaptive perception, reflection, and memory enabling comprehensive superiority over all training-free methods across all scenarios.
- **vs. Holmes-VAU**: Holmes-VAU requires instruction-tuning data, whereas PANDA is entirely training-free yet achieves comparable or superior performance on complex scenes.
- **vs. General-purpose Agents (AutoGPT/HuggingGPT)**: PANDA is tailored for VAD with a RAG knowledge base, heuristic reasoning, and CoM memory mechanisms, rather than a straightforward application of a general agent framework.
- **Insights**: The agentic paradigm advances from "tool invocation" to a complete closed loop of "strategy planning + reasoning + reflection + memory," which is transferable to other video understanding tasks such as action recognition and event prediction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First Generalist VAD agent; four-capability design is systematic and cohesive)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 datasets, 3 settings, detailed ablation and hyperparameter analysis; efficiency analysis is absent)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, comprehensive formulations, intuitive figures)
- Value: ⭐⭐⭐⭐⭐ (Opens a new direction for VAD agents; CSAD benchmark has community value; code is open-sourced)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[AAAI 2026\] AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture](../../AAAI2026/llm_agent/aquasentinel_next-generation_ai_system_integrating_sensor_ne.md)
- [\[NeurIPS 2025\] Enhancing Demand-Oriented Regionalization with Agentic AI and Local Heterogeneous Data for Adaptation Planning](enhancing_demand-oriented_regionalization_with_agentic_ai_and_local_heterogeneou.md)
- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[NeurIPS 2025\] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers](lc-opt_benchmarking_reinforcement_learning_and_agentic_ai_for_end-to-end_liquid_.md)

<!-- RELATED:END -->
