---
title: >-
  [Paper Note] VoxMind: An End-to-End Agentic Spoken Dialogue System
description: >-
  [ACL 2026][Audio & Speech][end-to-end spoken dialogue] VoxMind is proposed as a unified framework that endows end-to-end spoken dialogue models with agentic capabilities. Through a "Think-before-Speak" mechanism for expl…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "end-to-end spoken dialogue"
  - "tool calling"
  - "think-before-speak mechanism"
  - "multi-agent dynamic tool management"
  - "speech agent"
date: 2026-05-08
content_hash: 05ed186665478f55
---

# VoxMind: An End-to-End Agentic Spoken Dialogue System

**Conference**: ACL 2026  
**arXiv**: [2604.15710](https://arxiv.org/abs/2604.15710)  
**Code**: [GitHub](https://github.com/MM-Speech/VoxMind)  
**Area**: Dialogue System / Agent  
**Keywords**: end-to-end spoken dialogue, tool calling, think-before-speak mechanism, multi-agent dynamic tool management, speech agent

## TL;DR

VoxMind is proposed as a unified framework that endows end-to-end spoken dialogue models with agentic capabilities. Through a "Think-before-Speak" mechanism for explicit reasoning and a multi-agent dynamic tool management architecture to decouple inference latency from tool library scale, it improves task completion rates from a baseline of 34.88% to 74.57%, surpassing Gemini-2.5-Pro.

## Background & Motivation

**Background**: End-to-end spoken dialogue models (e.g., Kimi-Audio, Qwen2.5-Omni, StepAudio2) have developed rapidly, directly modeling paralinguistic information and generating expressive speech responses while avoiding information loss and latency inherent in traditional cascaded ASR-LLM-TTS pipelines.

**Limitations of Prior Work**: (1) Existing end-to-end speech models primarily optimize for reactive dialogue, lacking reasoning, planning, and external knowledge acquisition capabilities; (2) The field lacks a unified definition and evaluation standard for "end-to-end speech agents"; (3) Speech input requires more tokens than text, leading to significant computational overhead when combined with large-scale tool descriptions; (4) There is a lack of speech data annotated with agentic behaviors (reasoning trajectories, tool interactions).

**Key Challenge**: A trade-off exists between the agentic capabilities of speech models (tool calling + reasoning/planning) and inference efficiency—integrating more tools increases capability but adds latency, whereas speech interaction is sensitive to response time.

**Goal**: (1) Define end-to-end speech agents; (2) Empower speech models with reasoning and tool-calling capabilities; (3) Decouple inference latency from the scale of the tool library.

**Key Insight**: Successes from text-based agents (ReAct, tool calling) can be adapted, but special requirements for speech—low latency, paralinguistic preservation, and data scarcity—must be addressed.

**Core Idea**: A "Think-before-Speak" mechanism allows the speech model to generate text reasoning trajectories before generating speech responses. An asynchronous auxiliary model is used to maintain a dynamic local tool space by selecting candidate tools from a global library.

## Method

### Overall Architecture

VoxMind receives speech input, and the primary model first generates a reasoning trajectory (CoT). Subsequently, two processes are executed in parallel: (1) The primary model selects an action within a local tool space based on the reasoning result; (2) An auxiliary LLM proposes candidate tools from the global library based on the same reasoning result. If the primary model determines the current tools are insufficient, a tool space expansion is triggered; otherwise, execution proceeds directly to generate a speech response.

### Key Designs

1.  **Think-before-Speak Mechanism**:

    - **Function**: Introduces explicit reasoning capabilities to the speech model.
    - **Mechanism**: Before generating a speech response, the model generates a text reasoning trajectory $\mathbf{c}_t \sim \pi_\theta^{\text{think}}(\mathbf{c} | \mathbf{o}_t, \mathcal{H}_{t-1}, \mathcal{T}_t^{local})$ to capture intent understanding, context analysis, and task planning. It then selects an action $\mathbf{a}_t \sim \pi_\theta^{\text{act}}(\mathbf{a} | \mathbf{c}_t, \mathbf{o}_t, \mathcal{H}_{t-1})$ based on this trajectory.
    - **Design Motivation**: Direct $x \to y$ mapping in end-to-end speech models is insufficient for complex planning; intermediate reasoning steps $x \to z \to y$ are required. Training data for reasoning trajectories is constructed via reverse conditional generation.

2.  **Multi-agent Dynamic Tool Management**:

    - **Function**: Decouples inference latency from the tool library scale.
    - **Mechanism**: A local tool space $\mathcal{T}_t^{local} \subset \mathcal{T}^{all}$ is maintained. The primary model and auxiliary LLM execute in parallel: the primary model selects actions in the local space while the auxiliary LLM proposes candidates from the global library. When the primary model outputs $a_{\text{retrieve}}$ (determining current tools are insufficient), candidates are merged into the local space $\mathcal{T}_{t+1}^{local} = \mathcal{T}_t^{local} \cup \mathcal{T}_t^{cand}$.
    - **Design Motivation**: Processing all tool descriptions linearly increases tokens with the number of tools. Asynchronous parallelization and on-demand expansion ensure that latency does not increase significantly as the tool library grows.

3.  **AgentChat Dataset Construction**:

    - **Function**: Provides data with reasoning annotations for speech agent training.
    - **Mechanism**: Contains tool interaction corpora (14,805 entries from ToolACE, APIGen-MT, and self-constructed data) and general dialogue corpora (31,481 entries), totaling 470 hours. Reasoning trajectories are generated via $R \sim p_{\text{LM}}(R | Q, A)$, followed by iterative filtering (quality threshold 7/10, max 3 retries) and text refinement.
    - **Design Motivation**: The speech domain lacks agentic behavior annotations, necessitating construction and speech synthesis based on text data.

### Loss & Training

Ours is fine-tuned based on StepAudio2 using the AdamW optimizer with a learning rate of 1e-5, DeepSpeed ZeRO-3, bfloat16 precision, and gradient checkpointing. Training was conducted on 2 H20-NVLink GPUs.

## Key Experimental Results

### Main Results

| Model | Single-task TS/PF | Task Deco. TS/PF | Parallel TS/PF | Active TU | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| StepAudio2 | 78.70/48.87 | 60.32/26.98 | 53.33/33.33 | 3.12 | 34.88 |
| Kimi-Audio | 78.45/56.89 | 48.15/22.75 | 79.05/55.24 | 13.64 | 54.94 |
| Gemini-2.5-pro | 90.98/75.19 | 82.54/52.38 | 88.57/69.52 | 26.87 | 71.51 |
| **VoxMind** | **98.50/72.18** | **95.24/38.10** | **89.52/61.59** | **68.66** | **74.57** |

### Ablation Study

| Configuration | Overall | Description |
| :--- | :--- | :--- |
| w/o think, 1:1 | 68.83 | No reasoning, tools/dialogue 1:1 |
| w/o think, 1:0.5 | 70.97 | No reasoning, less dialogue data |
| w/ think, 1:1 | 71.97 | With reasoning |
| w/ think, 1:0.5 | **74.57** | Reasoning + higher tool data ratio |

### Key Findings

- The "Think-before-Speak" mechanism provides an average improvement of approximately 3-6%, with the largest gain in "Active Tool Seeking" (from 31.34% to 68.66%).
- A tool/dialogue data ratio of 1:0.5 outperforms 1:1, indicating that a higher proportion of agentic data benefits tool capabilities.
- Dynamic tool management ensures that latency does not grow significantly with the number of tools; with 40 tools, latency increased by only about 20%.
- VoxMind maintains general dialogue quality on VoiceBench, showing no degradation due to agentic training.

## Highlights & Insights

- **Formal definition of end-to-end speech agents**: Fills a gap in the field by providing a four-dimensional framework (Profile, Memory, Planning, Action) for future research.
- **Asynchronous parallel dynamic tool management**: The design is ingenious; the auxiliary model and primary model share reasoning trajectories but perform independent retrieval, decoupling capability from efficiency.
- **Reverse conditional generation for reasoning trajectories**: This data construction method is practical, generating reasoning processes from existing Q&A pairs, which is more efficient than manual annotation.

## Limitations & Future Work

- AgentChat data consists mainly of TTS synthesis, which may lack the richness of natural speech.
- Evaluation was primarily conducted on a self-constructed test set, lacking a community-recognized speech agent benchmark.
- The impact of the choice and scale of the auxiliary LLM on overall performance has not been fully ablated.
- Streaming reasoning scenarios (starting reasoning while the user is still speaking) have not been explored.

## Related Work & Insights

- **vs. Cascaded Systems (Qwen3 + Whisper)**: Cascaded systems utilize the agentic capabilities of text LLMs but lose paralinguistic information and increase latency; VoxMind maintains end-to-end advantages.
- **vs. WavRAG/Stream RAG**: These support only single agentic functions like retrieval augmentation; VoxMind supports full tool calling and reasoning/planning.
- **vs. Gemini-2.5-pro**: While the closed-source model has advantages in individual capabilities, VoxMind surpasses it in overall agentic tasks and is open-source and deployable.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic end-to-end speech agent framework, complete with definition, data, and methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons, though lacking standardized community benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear formal definitions and intuitive architectural diagrams.
- Value: ⭐⭐⭐⭐⭐ The open-source framework and dataset provide a significant push for the speech agent field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ACL 2026\] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)
- [\[ACL 2026\] Speculative End-Turn Detector for Efficient Speech Chatbot Assistant](speculative_end-turn_detector_for_efficient_speech_chatbot_assistant.md)
- [\[ACL 2026\] SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness](sdiareward_modeling_and_benchmarking_spoken_dialogue_rewards_with_modality_and_c.md)

</div>

<!-- RELATED:END -->
