---
title: >-
  [Paper Note] ProAct-VL: A Proactive VideoLLM for Real-Time AI Companions
description: >-
  [ICML 2026][Video Understanding][Video Large Language Model] ProAct-VL enables VideoLLMs to autonomously decide **when to respond** and generate short segment comments under streaming input through a chunked input-output…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Video Large Language Model"
  - "Streaming Inference"
  - "Proactive Response"
  - "Real-time Interaction"
  - "Game Commentary"
date: 2026-05-08
content_hash: 57183d28ef502ef3
---

# ProAct-VL: A Proactive VideoLLM for Real-Time AI Companions

**Conference**: ICML 2026  
**arXiv**: [2603.03447](https://arxiv.org/abs/2603.03447)  
**Code**: To be confirmed  
**Area**: Video Understanding / Real-time Multimodal Interaction  
**Keywords**: Video Large Language Model, Streaming Inference, Proactive Response, Real-time Interaction, Game Commentary

## TL;DR
ProAct-VL enables VideoLLMs to autonomously decide **when to respond** and generate short segment comments under streaming input through a chunked input-output paradigm, a lightweight FLAG decision head, and a transition-aware loss function. It achieves ~1s low latency and strong proactivity—obtaining a TimeDiff of only 1.20s and a trigger F1 of 63.25% in game commentary tasks, significantly surpassing offline models like GPT-4o.

## Background & Motivation

**Background**: Recent developments in Video Large Language Models (VideoLLMs) support video perception and real-time user interaction. However, most existing works adopt either a passive "chunk-sequential processing" response mode or a low-latency passive streaming mode that lacks response control.

**Limitations of Prior Work**:
- Proactive models decide when to speak but generate complete long answers once triggered, resulting in high latency and coarse temporal granularity.
- Real-time models emphasize fast generation but lack explicit control over "speaking behavior," often leading to over-talking.
- Existing methods struggle to balance proactive timing with content quality.

**Key Challenge**: A real AI companion (e.g., a game commentator) requires coordination across three layers: (1) low-latency inference, (2) autonomous decision on when to respond, and (3) control over the quality and quantity of generated content. These three form a triangle that is difficult to optimize simultaneously.

**Goal**: Construct a unified framework to solve "when to speak," "what to say," and "how fast to speak" simultaneously.

**Key Insight**: Scenarios such as game commentary and game guidance possess rich, automatically evaluable interaction patterns and are thus selected as specific evaluation scenarios. A large-scale annotated dataset is constructed to drive model training.

**Core Idea**: A chunk-level I/O paradigm + FLAG token decision head + transition-aware loss function are used to unify streaming video understanding and proactive response modeling.

## Method

### Overall Architecture
At each time step $t$ (1-second chunk):
1. **Input**: Triplet $(V_t, Q_t, B_t)$—visual content of the current time window, optional user query, and environmental context (including summaries of previous comments).
2. **Processing**: Persistent KV cache $\mathcal{K}_{t-1}$ maintains the full context, processed by a causal Transformer.
3. **Decision**: Speaking probability $p_t$ is extracted from the hidden state $h_t$ of a special `<|FLAG|>` token and compared with a threshold $\tau$ to obtain a binary decision $a_t$.
4. **Output**: If $a_t = 1$, a short segment comment $U_t$ (approx. 1s) is generated; otherwise, a silence token is output. The generated $U_t$ is automatically appended to the context as input for $t+1$.

### Key Designs

1. **Chunk-level I/O Paradigm**:
    - **Function**: Discretizes continuous video streams into fixed-duration chunks (1s in this paper) to support online causal processing.
    - **Mechanism**: At each step $t$, the model generates $(U_t, \mathcal{K}_t)$ from $(V_t, Q_t, B_t)$ and the persistent KV cache $\mathcal{K}_{t-1}$. The generated $U_t$ is immediately appended as part of the input for $t+1$, forming a continuous dialogue history.
    - **Design Motivation**: Chunks combined with persistent caching avoid the computational waste of re-processing the entire text at each step while maintaining full temporal context; long answers spanning multiple chunks can naturally connect across subsequent time steps.

2. **Lightweight Proactive Response Mechanism**:
    - **Function**: Inserts a special FLAG token at the end of each user message. The speaking probability is calculated from its hidden state via a lightweight MLP + sigmoid, and a binary decision is made using threshold $\tau$.
    - **Mechanism**: $p_t = \sigma(\text{MLP}(h_t))$, $a_t = \mathbb{I}[p_t \geq \tau]$. The decision head is extremely lightweight and does not add an inference bottleneck.
    - **Design Motivation**: Compared to designs that couple proactivity and generation, the decoupled decision mechanism allows the model to independently learn "when to speak" strategies, improving training and inference efficiency.

3. **Multi-level Stability Loss (Transition-aware + Stability Regularization)**:
    - **Function**: Composes $\mathcal{L}_{\text{resp}}$ through "transition-aware classification loss" and "stability regularization," combined with the main language modeling loss $\mathcal{L}_{\text{main}}$ via weighting.
    - **Mechanism**: Transition-aware classification loss $\mathcal{L}_{\text{cls}}$ uses weights $w_t = \gamma$ (when $y_t \neq y_{t-1}$) and $w_t = 1$ (when state persists), emphasizing rare but critical "speaking-silence" state transitions. Stability regularization $\mathcal{L}_{\text{reg}}$ contains two terms: local temporal consistency $\mathbb{E}[(p_t - p_{t-1})^2 \mid y_t = y_{t-1}]$ (smoothing probability changes during state persistence) and a global speaking rate constraint $(\mathbb{E}[p_t] - \mathbb{E}[y_t])^2$ (aligning the model's average speaking duration with human commentators). Finally, $\mathcal{L} = \mathcal{L}_{\text{main}} + \alpha \mathcal{L}_{\text{resp}}$.
    - **Design Motivation**: Treating the response state as a sequence learning problem rather than independent binary classification significantly improves learning of "when to transition states"; the global speaking rate constraint prevents over- or under-talking.

## Key Experimental Results

### Main Results (Live Gaming Benchmark)

| Model Category | Model | CC ↑ | LiveU ↑ | FinalQ ↑ | TimeDiff ↓ | F1 ↑ |
|:---|:---|:---|:---|:---|:---|:---|
| Offline | GPT-4o | 39.42 | 4.62 | 4.80 | 3.07 | 54.88 |
| Offline | Gemini 2.5 Pro | — | 4.70 | 4.82 | 2.59 | 49.23 |
| Proactive | VideoLLM-online | 13.78 | 3.56 | 1.74 | 12.59 | 6.54 |
| Proactive | MMDuet | 20.08 | 2.67 | 2.68 | 26.72 | 0.18 |
| Proactive | Livestar | 8.59 | 3.14 | 2.41 | 27.33 | 0.20 |
| Real-time | LiveCC-7B-Base | 38.88 | 3.85 | 3.83 | 11.35 | 36.10 |
| Real-time | StreamingVLM | 14.89 | 3.49 | 2.65 | 2.21 | 50.67 |
| **Ours** | **ProAct-VL** | **49.23** | **6.52** | **5.03** | **1.20** | **63.25** |

CC = Win rate against Gemini 2.5 Pro; LiveU = Quality of streaming segment comments; FinalQ = Overall script quality; TimeDiff = Response time deviation (seconds); F1 = Trigger precision. ProAct-VL is optimal across all indicators, especially in response timing (1.20s) and trigger precision (63.25%), far exceeding baselines.

### Ablation Study

| Configuration | CC | TimeDiff | P | R | F1 | Description |
|:---|:---|:---|:---|:---|:---|:---|
| Only $\mathcal{L}_{\text{cls}}$ | 45.54 | 18.50 | 12.13 | 14.00 | 11.03 | Classification loss alone |
| Only $\mathcal{L}_{\text{reg}}$ | 47.53 | 8.28 | 45.20 | 67.02 | 47.39 | Stability regularization alone |
| **Full** | **50.91** | **3.41** | **65.72** | **62.41** | **60.08** | Combination of both loss terms |

### Key Findings
- Removing $\mathcal{L}_{\text{reg}}$ has the greatest impact—F1 drops by 49.05 and TimeDiff increases by 15.09, showing that stability regularization is crucial.
- Removing $\mathcal{L}_{\text{cls}}$ also leads to performance degradation, though the impact is smaller than regularization; the two loss terms are complementary.
- Long-sequence stability: Streaming Commentary increased from 73.75% to 82.03% (10-50 minute videos); although response quality slightly decays, it tends to stabilize (F1 drops from 74.42% to 69.23%); long-term stability is significantly better than StreamingVLM.

## Highlights & Insights
- **Unification of Proactivity and Streaming Real-time Performance**: Traditional trade-offs were "passive but fast" or "proactive but slow"; this paper achieves strong proactivity under ~1s latency by decoupling decision and generation. This design can be transferred to interactive tasks requiring real-time decision-making (customer service systems, real-time subtitling).
- **Transition-aware Weighting Mechanism**: Treating state transitions as rare events and assigning high weights ($\gamma = 5$) provides the core insight that "transitions are often more important than persistence in sequential decision-making," which is inspiring for any time-series classification task.
- **High-quality Annotation Pipeline for Live Gaming Dataset**: A three-stage process of WhisperX ASR + Qwen3 sentiment annotation + DeepSeek domain error correction ensures high-precision transcription; the pipeline (especially LLM correction + cleaning) can be reused for other multimodal datasets.

## Limitations & Future Work
- The dataset is limited to the gaming domain (though covering 12 popular games, the focus remains entertainment); generalization to fields like sports commentary or news broadcasting is limited.
- Metrics such as CC / LiveU / FinalQ are calculated by closed-source LLMs (GPT-5.1), limiting reproducibility; manual verification across languages/modalities still needs supplementation.
- The response decision mechanism is relatively simple—using only FLAG token hidden states + MLP might ignore fine-grained visual signals (motion magnitude, scene changes).
- Future directions: Expanding to more real-time interaction fields; introducing multimodal features (audio emotion, gestures) to enhance decisions; exploring threshold-free decision strategies (directly regressing latency instead of binary classification).

## Related Work & Insights
- **vs Proactive Models (VideoLLM-online / MMDuet)**: These models generate full answers when "speaking," resulting in high latency (> 10s) and low trigger accuracy (F1 < 10%); this paper enforces short segment generation (1s) + decoupled decisions to ensure proactivity while avoiding long-tail latency.
- **vs Low-latency Models (LiveCC / StreamingVLM)**: They optimize inference speed but lack control over "when to speak," often leading to over-generation. ProAct-VL adds "silence" capability via an explicit response head, allowing it to interact with restraint like a human.
- **vs Offline Models (GPT-4o / Gemini)**: They have strong understanding but cannot operate in real-time; ProAct-VL achieves close performance (CC 49.23 vs GPT-4o 39.42) while supporting true real-time deployment.

## Rating
- Novelty: ⭐⭐⭐⭐ Unified framework for proactivity and real-time performance; although individual components like transition-aware loss + FLAG mechanism are not overly complex, the engineering combination is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 interaction scenarios + 2 test sets (in-domain + out-of-domain) + long-sequence stability + ablation + inference efficiency + manual verification.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and intuitive charts; some details (ChatML format, RoPE correction) are relegated to the appendix.
- Value: ⭐⭐⭐⭐⭐ Addresses real problems for AI companion applications; provides a deployable system + 561-hour annotated dataset; directly drives advancements in live streaming, gaming, and virtual assistants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos](../../AAAI2026/video_understanding/uncovering_zero-shot_generalization_gaps_in_time-series_foundation_models_using_.md)
- [\[AAAI 2026\] Learning Time in Static Classifiers](../../AAAI2026/video_understanding/learning_time_in_static_classifiers.md)
- [\[CVPR 2026\] StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](../../CVPR2026/video_understanding/streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [\[CVPR 2026\] Envisioning the Future, One Step at a Time](../../CVPR2026/video_understanding/envisioning_the_future_one_step_at_a_time.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](../../ACL2026/video_understanding/response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)

</div>

<!-- RELATED:END -->
