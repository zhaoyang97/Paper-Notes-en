---
title: >-
  [Paper Note] Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video
description: >-
  [NeurIPS 2025][Model Compression][streaming video] This paper defines a new task, "First-Person Streaming Video Proactive Understanding." Given ego-streaming video, an AI assistant proactively answers diverse, evolving questions at the appropriate moments while maintaining synchronized perception and reasoning. The authors propose the ESTP-Bench evaluation framework, the ESTP-F1 metric, and a complete technical pipeline (VideoLLM-EyeWO) containing a data engine…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "streaming video"
  - "proactive response"
  - "ego-centric"
  - "just-in-time"
  - "dynamic compression"
  - "ESTP-Bench"
date: 2026-05-08
content_hash: 84adff395c468ee4
---

# Eyes Wide Open: Ego Proactive Video-LLM for Streaming Video

**Conference**: NeurIPS 2025  
**arXiv**: [2510.14560](https://arxiv.org/abs/2510.14560)  
**Code**: [https://zhangyl4.github.io/publications/eyes-wide-open/](https://zhangyl4.github.io/publications/eyes-wide-open/)  
**Area**: Model Compression  
**Keywords**: streaming video, proactive response, ego-centric, just-in-time, dynamic compression, ESTP-Bench  
**Authors**: Yulin Zhang (ShanghaiTech), Cheng Shi (HKU), Yang Wang (ShanghaiTech), Sibei Yang† (SYSU)

## TL;DR
This paper defines a new task, "First-Person Streaming Video Proactive Understanding." Given ego-streaming video, an AI assistant proactively answers diverse, evolving questions at the appropriate moments while maintaining synchronized perception and reasoning. The authors propose the ESTP-Bench evaluation framework, the ESTP-F1 metric, and a complete technical pipeline (VideoLLM-EyeWO) containing a data engine, multi-stage training, and proactive dynamic compression. VideoLLM-EyeWO outperforms the strongest baseline, MiniCPM-V, by 11.8% on ESTP-Bench.

## Background & Motivation
Existing Video-LLMs operate in a **passive response mode**—answering only when prompted by users, and typically processing pre-recorded full videos (offline). However, AI assistants in real-world scenarios (such as smart wearables, autonomous driving copilots, etc.) need to process **real-time streaming video** and **proactively** offer information at critical moments. This new task requires three key attributes:

1. **Proactive Coherence**: Processing diverse questions, proactively responding even if answers depend on future video frames, and maintaining contextual coherence across related questions. For example, a green segment in a dialogue may semantically depend on the content of a purple segment, requiring the cross-temporal integration of past and current information.
2. **Just-in-Time Responsiveness**: Determining when to answer based on visual readiness—responding too early may cause errors due to insufficient evidence, while responding too late misses the opportunity to assist; the model should remain silent when uncertain to avoid unnecessary redundancy.
3. **Synchronized Efficiency**: Answering and visual perception must occur concurrently; the model cannot afford to miss new visual inputs while generating responses, and it must maintain temporal and memory efficiency as the frame count increases.

Existing evaluation frameworks (such as StreamingBench, OVO-Bench, etc.) fail to comprehensively evaluate these three dimensions. Most online benchmarks suffer from monolithic question types and lack contextual continuity, rarely assessing just-in-time responsiveness or synchronized efficiency.

## Core Problem
How can Video-LLMs be upgraded from a passive "answering-only-when-asked + offline processing" paradigm to a proactive "autonomously deciding when to respond + real-time streaming processing" paradigm while simultaneously satisfying the three constraints of proactive coherence, just-in-time responsiveness, and synchronized efficiency?

## Method

### 1. ESTP-Bench and ESTP-F1 Metric
Built upon the Ego4D validation set, the benchmark includes 890 videos, 100+ scenarios, and 2,264 human-verified QA instances, covering 14 task types (such as object recognition, attribute perception, action recognition, intent prediction, etc.). The questions are classified into three proactive categories:
- **Explicit**: 8 task types, answering directly using visual information (OR/AP/TRU/OL/OSC/EOL/EOSC/AR)
- **Implicit**: 4 task types, requiring reasoning beyond immediate observation (OFR/IFR/NAR/TU)
- **Contextual**: 2 task types, requiring cross-temporal dialogue coherence stability (ORC/TRC)

Each question is annotated with an average of 3.96 valid response time intervals, and 46% of the questions are contextually linked. ESTP-F1 comprehensively measures three dimensions:
- **Answer Quality**: Evaluating the accuracy of the predicted content against the ground truth using an LLM, denoted as $\mathcal{S}_{\text{answer}}$.
- **Response Timing**: Measuring timeliness via $\mathcal{S}_{\text{time}}$, with false negatives (FN) penalizing missed responses.
- **Temporal Precision**: Penalizing false positives (FP) for misalignment.

$$\text{ESTP-F1} = \frac{2 \times \sum_{k=1}^{M} S(g_k)}{2\sum_{k=1}^{M} S(g_k) + \text{FP} + \text{FN}}$$

### 2. Data Engine (ESTP-Gen)
Leveraging the Ego4D training set, the authors automatically generate 60K single-turn and 20K multi-turn training data using a three-stage pipeline:
- **One-to-One**: Utilizing an LVLM to generate captions and extract initial QA pairs (single time interval).
- **One-to-Many**: Expanding each answer to multiple valid time intervals using RAG.
- **Many-to-Many**: Merging related QA pairs into coherent multi-turn dialogues.

### 3. Multi-Stage Training Strategy
Based on the LLaMA3 + SigLIP architecture and trained with LoRA, the model is progressively endowed with three tiers of capability:

**Stage-1: Passive Interval Response**  
Weighted supervision (rather than simple binary classification) is imposed within the valid answer intervals. A linear decay function $f$ is used as weights to modulate the supervision intensity based on the distance between the time step and the end of the interval, resolving training conflicts caused by highly similar adjacent frames.

**Stage-2: Proactive Just-in-Time Response and Precise Answering**  
A third action $a_{\text{ask\_high}}$ is introduced—proactively requesting high-resolution frames during uncertain moments. The model first learns when to request high resolution ($\mathcal{L}_{\text{ask\_high}}$), and then determines whether it is the correct response time based on this high-resolution information to deliver a precise answer ($\mathcal{L}_{\text{determine}}$). The total loss is the sum of both.

**Stage-3: Multi-turn QA Coherence**  
Training solely on multi-turn QA data, which enhances contextual understanding and cross-turn coherence while preserving the ability for just-in-time responsiveness.

### 4. Proactive Dynamic Compression Mechanism
To guarantee memory efficiency, a two-tiered compression strategy is proposed:
- **Two-Level Compression**: The model proactively decides the timing and level of compression. When a potential response is anticipated, it requests high-resolution input (low compression); otherwise, high compression rates are applied to historical content. Upon completing a response, the preceding content is further compressed.
- **Unified Compression Method**: Inserting a special compression token `⟨ct⟩` (initialized with the EOS embedding) after the input segments. The causal attention mechanism is then leveraged to compress preceding information into a compact KV cache representation. The average token consumption is reduced to approximately 1/10 of the original sequence.

## Key Experimental Results

### ESTP-Bench Main Table

| Model | Overall ESTP-F1 |
|------|----------------|
| LIVE (th=0.9) | 15.5 |
| MMDuet | 17.8 |
| MiniCPM-V (Polling) | 22.9 |
| Qwen2-VL (Polling) | 21.3 |
| **VideoLLM-EyeWO** | **34.7** |

- Outperforms the baseline LIVE by **+19.2%** and the strongest Polling model MiniCPM-V by **+11.8%**.
- Achieves 23.6 vs. 9.5 (baseline) on explicit tasks, 52.5 vs. 25.6 on implicit tasks, and 43.6 vs. 20.3 on contextual tasks.

### Ablation Study
- The ESTP-IT data yields a boost of +7.1 / +6.8 (single-turn / contextual) for the LIVE baseline.
- Stage-1 progressive training successfully resolves binary classification training conflicts without requiring manual threshold tuning.
- Proactive dynamic compression reduces KV cache consumption to approximately **0.11%** of the baseline (from 9,636 to 942 tokens).
- Stage-2 (multi-turn coherence) further yields +4.9 on contextual tasks.

### Other Benchmarks
- OVO-Bench zero-shot: 32.76 vs. 20.79 of VideoLLM-online (+57.6%)
- COIN benchmark Top-1 accuracy: 66.0 vs. 63.4 (+2.6%), validating the architectural generalization ability.

## Highlights & Insights
- **"Proactive Streaming Video Understanding" stands as a paradigm-shifting new task definition**, shifting from passive to proactive, and offline to streaming.
- **Formalization of three key attributes** (Proactive Coherence / Just-in-Time / Sync Efficiency) and their corresponding "impossible trinity" analysis.
- **ESTP-Bench is the first benchmark to comprehensively evaluate streaming proactive understanding**, featuring 14 task types, 3 proactive categories, and precise response interval annotations.
- **Proactively requesting high-resolution frames** serves as a core innovation—the model autonomously determines when finer-grained visual information is required.
- **Dynamic compression compresses the KV cache to 0.11% of the baseline**, making long streaming video processing a reality.
- A **complete technical stack** spanning task definition, data construction, training strategies, to inference optimization.

## Limitations & Future Work
- There exists a notable negative correlation between Recall and Precision; determining the active response timing still generates false positives.
- Scores for NAR/TU tasks are deceptively high due to the large proportion of valid intervals, suggesting that metrics may require normalization.
- First-person data collection and precise temporal annotation incur substantial costs.
- Evaluation is currently restricted to Ego4D; generalization to other domains (e.g., autonomous driving, surveillance) remains to be explored.
- In terms of synchronized efficiency, a trade-off persists between APS (Actions Per Second) and model performance.

## Related Work & Insights
- **vs. LIVE / VideoLLM-Online**: Sharing the LLaMA3+SigLIP architecture and the Ego4D data source, but LIVE utilizes a simple binary classification supervision and lacks a proactive high-resolution request mechanism; EyeWO significantly outperforms them through multi-stage training and dynamic compression.
- **vs. MMDuet**: MMDuet pursues high recall but at the cost of extremely low precision (over-responsiveness), whereas EyeWO achieves a better balance between the two.
- **vs. Offline MLLMs (such as Qwen2-VL / MiniCPM-V, etc.)**: Offline models can achieve acceptable performance using a Polling strategy but fail to achieve true real-time synchronization; performance drops significantly under the Response-in-Last strategy, exposing weak temporal localization capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proactive streaming video understanding represents a paradigm-shifting new task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Brand-new benchmark + novel metric + multi-baseline comparisons + extensive ablation + cross-benchmark validation.
- Writing Quality: ⭐⭐⭐⭐ Clear formalization of the three attributes; the impossible trinity analysis is insightful.
- Value: ⭐⭐⭐⭐⭐ Defines an important new direction for Video-LLMs, presenting a deployable complete technical stack.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[NeurIPS 2025\] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)
- [\[NeurIPS 2025\] Smooth Regularization for Efficient Video Recognition](smooth_regularization_for_efficient_video_recognition.md)
- [\[NeurIPS 2025\] AI-Generated Video Detection via Perceptual Straightening](ai-generated_video_detection_via_perceptual_straightening.md)
- [\[NeurIPS 2025\] Mitigating Semantic Collapse in Partially Relevant Video Retrieval](mitigating_semantic_collapse_in_partially_relevant_video_retrieval.md)

</div>

<!-- RELATED:END -->
