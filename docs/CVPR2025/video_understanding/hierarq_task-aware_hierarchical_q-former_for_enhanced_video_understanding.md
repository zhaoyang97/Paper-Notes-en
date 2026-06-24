---
title: >-
  [Paper Note] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Q-Former] Proposes HierarQ, a task-aware hierarchical Q-Former framework that achieves autoregressive frame-by-frame video processing through a two-stream language-guided feature modulator (entity stream + scene stream) and short/long-term memory banks. It bypasses the LLM context length limit without frame sampling, achieving SOTA or near-SOTA performance on 10 video understanding benchmarks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Q-Former"
  - "Hierarchical"
  - "Task-Aware"
  - "Memory Bank"
date: 2026-05-08
content_hash: f509b586c0d06ed9
---

# HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2503.08585](https://arxiv.org/abs/2503.08585)  
**Code**: None (Project page available)  
**Area**: Video Understanding  
**Keywords**: Video Understanding, Q-Former, Hierarchical, Task-Aware, Memory Bank

## TL;DR

Proposes HierarQ, a task-aware hierarchical Q-Former framework that achieves autoregressive frame-by-frame video processing through a two-stream language-guided feature modulator (entity stream + scene stream) and short/long-term memory banks. It bypasses the LLM context length limit without frame sampling, achieving SOTA or near-SOTA performance on 10 video understanding benchmarks.

## Background & Motivation

Current multimodal large language models (MLLMs) face three major bottlenecks in medium-to-long video understanding:

1. **Context Length Constraint**: The context window of LLMs limits the number of frames that can be processed. While extending context length is a potential way out, it is computationally expensive and struggles to achieve theoretical promises.
2. **Information Loss from Frame Sampling**: Commonly used frame sampling methods (uniform/key-frame) may miss critical information in long videos and lack task relevance—the model blindly processes all frames without prioritizing task-related content.
3. **Oversimplification of Spatiotemporal Compression**: Methods like token compression and spatiotemporal pooling reduce the token count but risk losing key details.

The core idea of HierarQ is to emulate human cognition by **simultaneously focusing on frame-level entity details (who is doing what) and cross-frame scene context (how events evolve)**, dynamically adjusting the focus based on the task (prompt). By processing frames autoregressively, frame sampling is completely avoided.

## Method

### Overall Architecture

Given a video $V$ and a text prompt $T_P$, the processing flow of HierarQ is as follows:
1. Extract visual features frame-by-frame using a frozen ViT: $f_i = \mathcal{V}(v_i)$
2. Modulate features in a task-relevant manner using a two-stream feature modulator.
3. Store the modulated features into short-term/long-term memory banks.
4. A hierarchical Q-Former (HierarQ) queries and fuses information from the memory banks.
5. The output at the final time step is projected via an FC layer and fed into the LLM to generate the response.

### Key Designs

1. **Two-stream Language-guided Feature Modulator**:
    - **Function**: Dynamically modulates the visual features of each frame based on the prompt's semantics, enabling the model to focus on task-relevant frames.
    - **Mechanism**:
        - **Entity-guided modulator $L_f^e$**: Nouns (people/objects) are extracted from the prompt and encoded into $T_P^e$ using BERT, which then interacts with frame features via cross-attention: $f_i^e = C.Attn(T_P^e, f_i, f_i)$. This focuses the frame on entities mentioned in the prompt.
        - **Scene-guided modulator $L_f^s$**: Uses the BERT encoding of the full prompt, $T_P^s$, to perform cross-attention: $f_i^s = C.Attn(T_P^s, f_i, f_i)$, capturing macro-level scene relationships.
    - **Design Motivation**: The entity stream and scene stream focus on different granularities: the entity stream localizes "who/what" within a frame, while the scene stream comprehends "events/relationships". The two are complementary—entity details support scene understanding. A lightweight Transformer design ensures efficiency.

2. **Short/Long-term Memory Banks**:
    - **Function**: Provides rich temporal context for the Q-Former, balancing immediate details and long-term evolution.
    - **Mechanism**:
        - **Short-term memory $M_e$**: Stores entity-modulated visual features and query history, updated using FIFO (discarding the oldest entries when capacity $M$ is reached), which is low-cost.
        - **Long-term memory $M_s$**: Stores scene-modulated features, updated using Memory Bank Compression (MBC)—finding the adjacent token pair with the highest similarity $k = \arg\max_t \cos(f_t, f_{t+1})$ and merging them via averaging, which preserves temporal order while compressing redundancy.
    - **Design Motivation**: Entity information consists of frame-level, short-term details, for which FIFO is sufficient (entity details from old frames are less critical). Scene information requires long-term context spanning the entire video; simple FIFO would lose key scene continuity, necessitating a smarter compression strategy.

3. **Hierarchical Q-Former (HierarQ)**:
    - **Function**: Hierarchically integrates entity-level and scene-level information, outputting a fixed number of tokens (32) to the LLM.
    - **Mechanism**: Comprises two Q-Formers:
        - **Entity-level $QF_e$**: A standard Q-Former containing self-attention (interaction among queries + short-term query memory) and cross-attention (interaction between queries and short-term visual memory) to summarize frame-level entity details.
        - **Scene-level $QF_s$**: An extended Q-Former with 4 sub-modules: ① cross-attention (interaction with long-term visual memory) $\rightarrow$ ② self-attention (interaction with long-term query memory) $\rightarrow$ ③ self-attention (interaction among queries) $\rightarrow$ ④ cross-attention (interaction with outputs of $QF_e$), achieving information integration from entity to scene.
    - **Design Motivation**: The hierarchical design mimics human cognition—focusing on specific entities first, and then understanding relationships between entities at the scene level. The final-step cross-Q-Former attention ($Q=\hat{z}_t^s, K=z_t^e, V=z_t^e$) is key to injecting short-term entity details into long-term scene comprehension. Ultimately, outputting only $N$ tokens (instead of $N \times T$) fundamentally resolves LLM context constraints.

### Loss & Training

Trained on video-text pairs using the standard cross-entropy loss. ViT G/14 (EVA-CLIP) and Vicuna 7B are frozen, while the feature modulators, HierarQ, and the FC layer are micro-tuned. The LLM is fine-tuned with LoRA (rank=32). HierarQ weights are initialized from InstructBLIP. Training is conducted on 4 A100 GPUs.

## Key Experimental Results

### Main Results

**Medium-to-Long Video Understanding (LVU/Breakfast/COIN)**:

| Model | LVU Avg | Breakfast | COIN |
|------|---------|-----------|------|
| S5 | 60.9 | 90.7 | 90.8 |
| MA-LMM | 61.1 | 93.0 | 93.2 |
| VideoMamba | 57.8 | 94.3 | 86.2 |
| **HierarQ** | **67.9** (+6.8) | **97.4** (+3.1) | **96.0** (+2.8) |

**Short Video Question Answering (MSRVTT-QA / MSVD-QA / ActivityNet-QA)**:

| Model | MSR-QA | MSVD-QA | ANet-QA |
|------|--------|---------|---------|
| Mirasol3B | 50.4 | - | 51.1 |
| MA-LMM | 48.5 | 60.6 | 49.8 |
| **HierarQ** | **54.1** (+3.7) | **66.2** (+5.6) | **57.1** (+6.0) |

### Ablation Study

**Contribution of Each Component (LVU / Breakfast)**:

| Configuration | LVU | Breakfast | Description |
|------|-----|-----------|------|
| Baseline (MA-LMM) | 60.7 | 93.0 | Standard Q-Former + memory only |
| + Entity Mod. | 58.7 | 88.5 | Entity modulation alone degrades performance (lacks scene context) |
| + Prompt Mod. | 62.0 | 94.1 | Scene modulation alone is effective |
| + HierarQ + Two-stream | 66.8 | 96.1 | Hierarchical integration yields substantial gains |
| + LLM LoRA | **67.9** | **97.4** | Full model |

**Memory Update Strategies**:

| Short-term Update | Long-term Update | LVU | Breakfast |
|---------|---------|-----|-----------|
| FIFO | FIFO | 65.2 | 93.6 |
| MBC | MBC | 67.4 | 97.3 |
| **FIFO** | **MBC** | **67.9** | **97.4** |

### Key Findings

- Using the entity modulator alone degrades performance due to the lack of scene context; however, it achieves the best results when combined with the scene modulator and HierarQ, validating the hierarchical design where "entity details supplement scene understanding".
- Long-term memory is more critical than short-term memory (62.5 vs 61.8), but combining both yields the optimal performance (66.8).
- The optimal length for short-term memory is around 10; beyond this, excessive entity details interfere with scene understanding.
- Isolating $QF_e$ and $QF_s$ (canceling hierarchical interaction and switching to concatenation) leads to a 3.6% drop on LVU, proving the necessity of hierarchical modeling.
- Increasing the parameters of a single Q-Former to match HierarQ's scale still lags behind by 4.7%, showing that the gains stem from the architecture, not the parameter count.
- As video length increases, the performance of MA-LMM continuously drops, whereas HierarQ remains stable.

## Highlights & Insights

- **Cognitive Science-Inspired Design**: The two-stream design of entity and scene streams directly corresponds to the human cognitive paradigm of "local attention + global comprehension".
- **Engineering Elegance**: Autoregressive processing combined with a fixed $N$-token output fundamentally solves the LLM context length constraint ($N$ tokens instead of $N \times T$).
- **Task-Awareness**: Feature modulation is guided by LLM prompts, allowing different tasks to automatically "focus" on different frames, which aligns better with how humans watch videos than blindly processing all frames.
- **Refined Memory Strategy**: The differentiated update strategy—using FIFO for short-term and MBC for long-term memory—is simple yet highly effective.

## Limitations & Future Work

- Dependence on BERT for extracting nouns means NLP parsing errors will affect the accuracy of the entity stream.
- Although the framework is general, ablation experiments are mainly conducted on medium-to-long videos, with limited evaluation on ultra-long videos (>10 mins).
- Adaptive adjustment of memory bank capacity has not been explored (currently fixed at $M=10$).
- HierarQ is initialized from InstructBLIP, imposing a certain dependency on pre-training data.

## Related Work & Insights

- **Relationship with MA-LMM**: Building upon MA-LMM's Q-Former + memory configuration, HierarQ introduces two-stream modulation and a hierarchical design, acting as a direct improvement and extension.
- **Relationship with MovieChat**: Both utilize memory mechanisms to handle long videos, but MovieChat's memory merging is relatively coarse, whereas HierarQ's MBC is more refined.
- **Insights**: (1) The two-stream/hierarchical paradigm can be generalized to other multi-granularity understanding tasks (e.g., document understanding: word-level + paragraph-level); (2) Task-aware feature modulation represents a generic strategy for efficiency improvement.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of a hierarchical Q-Former and two-stream task-aware modulation is novel, though individual components follow established paradigms.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive ablations across 10 benchmarks covering three major tasks: video understanding, QA, and captioning.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, features professional diagrams, and offers in-depth ablation analysis.
- Value: ⭐⭐⭐⭐⭐ Provides a practical and efficient solution for medium-to-long video MLLMs, achieving SOTA on most benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [\[CVPR 2025\] Object-Shot Enhanced Grounding Network for Egocentric Video](object-shot_enhanced_grounding_network_for_egocentric_video.md)
- [\[CVPR 2025\] Progress-Aware Video Frame Captioning](progress-aware_video_frame_captioning.md)
- [\[CVPR 2025\] Context-Enhanced Memory-Refined Transformer for Online Action Detection](context-enhanced_memory-refined_transformer_for_online_action_detection.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)

</div>

<!-- RELATED:END -->
