---
title: >-
  [Paper Note] VoxMind: An End-to-End Agentic Spoken Dialogue System
description: >-
  [ACL 2026][Dialogue Systems][End-to-end spoken dialogue] This paper proposes VoxMind, a unified framework that endows end-to-end spoken dialogue models with agentic capabilities: explicit reasoning through a "Think-before-Speak" mechanism, combined with a multi-agent dynamic tool management architecture that decouples reasoning latency from tool scale, improving task completion rate from baseline 34.88% to 74.57%, surpassing Gemini-2.5-Pro.
tags:
  - ACL 2026
  - Dialogue Systems
  - End-to-end spoken dialogue
  - tool invocation
  - think-speak mechanism
  - multi-agent dynamic tool management
  - speech agent
date: 2026-05-08
content_hash: 88e27e2e2a8c04ca
---

# VoxMind: An End-to-End Agentic Spoken Dialogue System

**Conference**: ACL 2026  
**arXiv**: [2604.15710](https://arxiv.org/abs/2604.15710)  
**Code**: [GitHub](https://github.com/MM-Speech/VoxMind)  
**Area**: Dialogue Systems / Agent  
**Keywords**: End-to-end spoken dialogue, tool invocation, think-speak mechanism, multi-agent dynamic tool management, speech agent

## TL;DR

This paper proposes VoxMind, a unified framework that endows end-to-end spoken dialogue models with agentic capabilities: explicit reasoning through a "Think-before-Speak" mechanism, combined with a multi-agent dynamic tool management architecture that decouples reasoning latency from tool scale, improving task completion rate from baseline 34.88% to 74.57%, surpassing Gemini-2.5-Pro.

## Background & Motivation

**Background**: End-to-end spoken dialogue models (e.g., Kimi-Audio, Qwen2.5-Omni, StepAudio2) have developed rapidly in recent years, capable of directly modeling paralinguistic information and generating expressive speech responses, avoiding information loss and latency in traditional cascaded ASR-LLM-TTS pipelines.

**Limitations of Prior Work**: (1) Existing end-to-end speech models primarily optimize reactive dialogue, lacking reasoning, planning, and external knowledge acquisition capabilities; (2) The speech domain lacks a unified definition and evaluation standard for "end-to-end speech agents"; (3) Speech input requires more tokens than text, and when combined with large-scale tool descriptions, produces significant computational overhead; (4) Absence of speech data with agent behavior annotations (reasoning traces, tool interactions).

**Key Challenge**: There is a trade-off between agentic capabilities (tool invocation + reasoning planning) and inference efficiency in speech models—integrating more tools enhances capabilities but increases latency, while speech interaction is sensitive to response time.

**Goal**: (1) Define end-to-end speech agents; (2) Endow speech models with reasoning and tool invocation capabilities; (3) Decouple reasoning latency from tool library scale.

**Key Insight**: Drawing on successful experiences from text agents (ReAct, tool invocation), but adapting to the special requirements of speech scenarios—low latency, preservation of paralinguistic information, scarcity of speech data.

**Core Idea**: Use a Think-before-Speak mechanism to have the speech model first generate text reasoning traces before generating speech responses, and use asynchronous auxiliary models to select candidate tools from the global tool library to maintain a dynamic local tool space.

## Method

### Overall Architecture

VoxMind receives speech input, with the main model first generating reasoning traces (CoT), then executing two processes in parallel: (1) The main model selects actions in the local tool space based on reasoning results; (2) The auxiliary LLM proposes candidate tools from the global tool library based on the same reasoning results. If the main model determines current tools are insufficient, it triggers tool space expansion; otherwise, it directly executes and generates speech responses.

### Key Designs

1. **Think-before-Speak Mechanism**:

    - Function: Introduce explicit reasoning capability to speech models
    - Mechanism: Before generating speech responses, first generate text reasoning traces $\mathbf{c}_t \sim \pi_\theta^{\text{think}}(\mathbf{c} | \mathbf{o}_t, \mathcal{H}_{t-1}, \mathcal{T}_t^{local})$, capturing intent understanding, context analysis, and task planning. Then select actions based on reasoning traces $\mathbf{a}_t \sim \pi_\theta^{\text{act}}(\mathbf{a} | \mathbf{c}_t, \mathbf{o}_t, \mathcal{H}_{t-1})$
    - Design Motivation: Direct $x \to y$ mapping in end-to-end speech models is insufficient for complex planning, requiring intermediate reasoning steps $x \to z \to y$. Reasoning trace training data is constructed through backward conditional generation

2. **Multi-Agent Dynamic Tool Management**:

    - Function: Decouple reasoning latency from tool library scale
    - Mechanism: Maintain local tool space $\mathcal{T}_t^{local} \subset \mathcal{T}^{all}$. Main model and auxiliary LLM execute in parallel: main model selects actions in local space, auxiliary LLM proposes candidates from global library. When main model outputs $a_{\text{retrieve}}$ (judging current tools insufficient), merge candidates into local space $\mathcal{T}_{t+1}^{local} = \mathcal{T}_t^{local} \cup \mathcal{T}_t^{cand}$
    - Design Motivation: Processing all tool descriptions each time leads to token count growing linearly with number of tools. Asynchronous parallel + on-demand expansion ensures latency does not increase significantly with growing tool library

3. **AgentChat Dataset Construction**:

    - Function: Provide data with reasoning annotations for speech agent training
    - Mechanism: Contains tool interaction corpus (14,805 entries from ToolACE, APIGen-MT, and self-built data) and general dialogue corpus (31,481 entries), totaling 470 hours. Reasoning traces generated through backward conditional generation $R \sim p_{\text{LM}}(R | Q, A)$, filtered iteratively (quality threshold 7/10, maximum 3 retries) and text polishing
    - Design Motivation: Severe lack of agent behavior annotation data in speech domain, requiring construction from text data and speech synthesis

### Loss & Training

Fine-tuned based on StepAudio2, using AdamW optimizer, learning rate 1e-5, DeepSpeed ZeRO-3, bfloat16 precision, gradient checkpointing. Trained on 2 H20-NVLink GPUs.

## Key Experimental Results

### Main Results

| Model | Single Task TS/PF | Task Decomp TS/PF | Parallel Proc TS/PF | Proactive TU | Overall |
|------|------------|-------------|-------------|-----------|---------|
| StepAudio2 | 78.70/48.87 | 60.32/26.98 | 53.33/33.33 | 3.12 | 34.88 |
| Kimi-Audio | 78.45/56.89 | 48.15/22.75 | 79.05/55.24 | 13.64 | 54.94 |
| Gemini-2.5-pro | 90.98/75.19 | 82.54/52.38 | 88.57/69.52 | 26.87 | 71.51 |
| **VoxMind** | **98.50/72.18** | **95.24/38.10** | **89.52/61.59** | **68.66** | **74.57** |

### Ablation Study

| Config | Overall | Note |
|------|---------|------|
| w/o think, 1:1 | 68.83 | No reasoning, tool/dialogue 1:1 |
| w/o think, 1:0.5 | 70.97 | No reasoning, less dialogue data |
| w/ think, 1:1 | 71.97 | With reasoning |
| w/ think, 1:0.5 | **74.57** | Reasoning + higher tool data ratio |

### Key Findings

- Think-before-Speak mechanism provides average improvement of ~3-6%, with greatest improvement on "proactive tool seeking" capability (from 31.34% to 68.66%)
- Tool/dialogue data ratio of 1:0.5 outperforms 1:1, indicating higher proportion of agent data benefits tool capabilities
- Dynamic tool management ensures latency does not increase significantly with number of tools; at 40 tools, latency increases only ~20%
- VoxMind maintains general dialogue quality on VoiceBench, with no degradation from agent training

## Highlights & Insights

- **Formalized definition of end-to-end speech agents** fills a domain gap—the four-dimensional framework of Profile, Memory, Planning, and Action provides a standard for future research
- **Asynchronous parallel dynamic tool management** design is ingenious—auxiliary model shares reasoning traces with main model but retrieves independently, achieving decoupling of capability and efficiency
- **Backward conditional generation of reasoning traces** data construction method is practical—generating reasoning process after having Q&A pairs is more efficient than manual annotation

## Limitations & Future Work

- AgentChat data primarily synthesized by TTS, may lack richness of natural speech
- Evaluation primarily on self-built test sets, lacking community-recognized speech agent benchmarks
- Impact of auxiliary LLM selection and scale on overall performance not sufficiently ablated
- Streaming reasoning scenarios not explored (starting reasoning during user speech)

## Related Work & Insights

- **vs Cascaded Systems (Qwen3+Whisper)**: Cascaded systems leverage text LLM agent capabilities but lose paralinguistic information and increase latency, VoxMind maintains end-to-end advantages
- **vs WavRAG/Stream RAG**: Only support single agent functions like retrieval augmentation, VoxMind supports complete tool invocation + reasoning planning
- **vs Gemini-2.5-pro**: Closed-source model has advantages in individual capabilities, but VoxMind surpasses in overall agent tasks and is open-source and deployable

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic end-to-end speech agent framework, complete with definition + data + methods
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison but lacks community standard benchmarks
- Writing Quality: ⭐⭐⭐⭐ Clear formalized definitions, intuitive architecture diagrams
- Value: ⭐⭐⭐⭐⭐ Open-source framework + dataset significantly advances speech agent field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[AAAI 2026\] Chatsparent: An Interactive System for Detecting and Mitigating Cognitive Fatigue in LLMs](../../AAAI2026/dialogue/chatsparent_an_interactive_system_for_detecting_and_mitigating_cognitive_fatigue.md)
- [\[ACL 2026\] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)

</div>

<!-- RELATED:END -->
