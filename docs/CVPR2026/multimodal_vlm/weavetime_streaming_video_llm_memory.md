---
title: >-
  [Paper Note] WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs
description: >-
  [CVPR 2026][Multimodal VLM][Video-LLM] This work diagnoses the "Time-Agnosticism" issue in current Video-LLMs and proposes the WeaveTime framework. It endows the model with temporal awareness through a Streaming Temporal Perception Enhancement (SOPE) auxiliary task during training. At inference, it implements efficient adaptive memory retrieval via an uncertainty-gated Past-Current Dynamic Focus Cache (PCDF-Cache), achieving significant improvements in streaming video QA.
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Video-LLM"
  - "streaming VQA"
  - "temporal order"
  - "memory cache"
  - "uncertainty-gated retrieval"
date: 2026-05-08
content_hash: c9b3192f2a590dea
---

# WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs

**Conference**: CVPR 2026  
**arXiv**: [2602.22142](https://arxiv.org/abs/2602.22142)  
**Code**: None (Coming soon)  
**Area**: Multimodal / Streaming Video Understanding  
**Keywords**: Video-LLM, streaming VQA, temporal order, memory cache, uncertainty-gated retrieval

## TL;DR

This work diagnoses the "Time-Agnosticism" issue in current Video-LLMs and proposes the WeaveTime framework. It endows the model with temporal awareness through a Streaming Temporal Perception Enhancement (SOPE) auxiliary task during training. At inference, it implements efficient adaptive memory retrieval via an uncertainty-gated Past-Current Dynamic Focus Cache (PCDF-Cache), achieving significant improvements in streaming video QA.

## Background & Motivation

**Background**: Modern visual understanding systems are increasingly deployed in streaming scenarios where frame sequences arrive in real-time (e.g., autonomous driving, human-computer interaction, real-time monitoring). Video-LLM-based methods (e.g., LLaVA-Video, Qwen2-VL) perform excellently in offline settings but face fundamental challenges in streaming contexts.

**Limitations of Prior Work**:
1. Current Video-LLMs suffer from **Time-Agnosticism**: they treat videos as unordered bags of evidence rather than causally ordered sequences. Experiments show that shuffling frame order has almost no impact on model accuracy and even improves performance on certain temporal tasks (whereas human performance drops sharply).
2. Existing streaming methods (e.g., StreamBridge, VideoLLM-Online) either require large-scale specialized streaming datasets and high-cost training or rely on customized memory mechanisms with sub-optimal results.
3. Compressed memory (selecting/merging/dropping visual features) leads to information loss; retrieval-based memory retains information but suffers from unnecessary long-range reloading and loss of temporal focus.

**Key Challenge**: Video-LLMs lack genuine temporal reasoning capabilities, and existing streaming augmentation methods fail to balance "temporal awareness" with "memory efficiency."

**Goal**: To address two coupled issues caused by Time-Agnosticism: **Temporal Order Ambiguity** and **Past-Current Focus Blindness**.

**Key Insight**: Teach temporal order first, then utilize it during inference—"first teach order, then use order."

**Core Idea**: Empower the model to perceive frame order through a lightweight temporal reconstruction auxiliary task, followed by demand-driven backtracking using uncertainty-driven coarse-to-fine retrieval.

## Method

### Overall Architecture

WeaveTime targets the "Time-Agnosticism" of Video-LLMs—the tendency of models to treat videos as unordered bags of evidence where shuffling frames barely affects accuracy. It is a plug-and-play, model-agnostic streaming QA framework following the principle of "first teach order, then use order." During training, Streaming Temporal Perception Enhancement (SOPE) teaches the model that "frames have a sequence" via a temporal reconstruction task. During inference, the Past-Current Dynamic Focus Cache (PCDF-Cache) utilizes uncertainty gating and coarse-to-fine retrieval, allowing the model to trace back through history on demand rather than reloading excessively.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Video Frame Stream + Question"] --> B["SOPE: Streaming Temporal Perception Enhancement (Training)<br/>Timestamp tokens + Shuffled frame reconstruction"]
    B --> C["PCDF-Cache Inference<br/>Short-term window response + Entropy H_t calculation"]
    C --> D{"H_t < δ ?"}
    D -->| "Yes, skip retrieval" | F["Output Answer"]
    D -->| "No" | E["Coarse-to-fine Retrieval<br/>Frame-level screening + Late-interaction matching"]
    E --> F
```

### Key Designs

**1. Time-Agnosticism Diagnosis: Proving models neglect temporal order**

This is motivated by a set of control experiments: when video frames are shuffled, model accuracy remains nearly unchanged or even improves in temporal awareness and action recognition tasks. In contrast, human performance on shuffled videos without timestamps collapses (dropping from 1.0 to 0.0–0.2) and only recovers when timestamps are provided. Heatmaps further reveal temporal position biases—focusing on the start and end of short videos and biased towards the beginning of long videos. These diagnostics indicate that Video-LLMs rely on spatio-temporal shortcuts and position biases rather than causal reasoning, justifying the need for temporal enhancement.

**2. Streaming Temporal Perception Enhancement (SOPE): Forcing the model to order frames**

To teach temporal reasoning, a Temporal Reconstruction (TR) auxiliary task is designed. Given a sequence of patched video tokens $\mathbf{X} = [\tilde{\mathbf{v}}_{1,1}, \ldots, \tilde{\mathbf{v}}_{1,N_f}, \tilde{\mathbf{v}}_{2,1}, \ldots]$, timestamp tokens $\mathbf{ts}_i$ are inserted before each frame before shuffling the content. The prompt is prepended with: "These segments are shuffled, please list the true time range for each segment." This transforms temporal prediction into a next-token prediction task, leveraging the LLM's text-reordering capabilities without adding extra modules or losses. It upgrades memory from an "unordered cache" to an "ordered state chain," enabling retrieval to locate event times rather than just content. It is efficient, using only 30k samples from LLaVA-Video-178K for 1 epoch of LoRA training on 8 GPUs.

**3. Past-Current Dynamic Focus Cache (PCDF-Cache): Look Now, Recall if Needed**

With temporal awareness established, PCDF-Cache addresses "when and how much to backtrack" via a "Look Now, Recall if Needed" strategy. When query $q$ arrives at time $t$, the model first predicts an answer $a_t^{(0)}$ using only the short-term window $\mathcal{M}_{t-1}[-C:]$ and calculates its prediction entropy $H_t = \text{Entropy}(a_t^{(0)})$. If $H_t < \delta$, the answer is adopted directly to save computation. If $H_t \geq \delta$, coarse-to-fine retrieval is triggered. This involves two layers: first, frame-level cosine similarity $\text{Sim}(f_i^v, f^q)$ identifies a coarse set $\mathcal{M}_{\text{coarse}}$, followed by fine matching via late-interaction max-sim: $\text{maxSim}(\{f_{i,k}^v\}, \{f_j^q\}) = \sum_{j=1}^{N_q}\max_{1\leq k\leq N_i}\langle f_j^q, f_{i,k}^v \rangle$. The top-$K$ frames (up to 64) are retrieved, achieving token-level accuracy at frame-level computational costs while avoiding OOM issues.

### Loss & Training

- Standard next-token prediction language modeling loss is used, with the TR auxiliary task and original QA merged into a single-turn conversation.
- 30k samples of offline video IT data are randomly sampled for LoRA fine-tuning (lr=$1\times10^{-5}$) for 1 epoch.
- The entropy threshold during inference is set to $\delta=0.6$ (optimal value from ablation); implemented based on ReKV with a maximum of 64 recalled frames.

## Key Experimental Results

### Main Results

Streaming Multi-Turn Evaluation based on LLaVA-OV-7B:

| Method | OVO-Bench Overall | Streaming-Bench Real-Time |
|------|-------------------|--------------------------|
| LLaVA-OV-7B + StreamBridge | 61.72 | 68.39 |
| LLaVA-OV-7B + ReKV | 61.72 | 66.15 |
| LLaVA-OV-7B + **WeaveTime** | **68.82** (+7.10) | **72.13** (+3.74) |

Evaluation based on Qwen2-VL-7B:

| Method | OVO-Bench Overall | Streaming-Bench Real-Time |
|------|-------------------|--------------------------|
| Qwen2-VL-7B + StreamBridge | 63.35 | 72.01 |
| Qwen2-VL-7B + ReKV | 59.72 | 70.07 |
| Qwen2-VL-7B + **WeaveTime** | **66.28** | **75.39** |

Improvements in temporal-sensitive sub-tasks are particularly significant: ACP +7.56%, EU +9.04%, ACR +11.09%.

### Ablation Study

| SOPE w/ TP | SOPE w/ TR | PCDF-Cache | OVO-Bench | Gain | Streaming-Bench | Gain |
|:---:|:---:|:---:|---:|---:|---:|---:|
| | | | 53.56 | — | 66.15 | — |
| ✔ | | | 49.88 | -3.68 | 65.91 | -0.54 |
| ✔ | ✔ | | 55.70 | +5.82 | 68.49 | +2.58 |
| ✔ | ✔ | ✔ | **57.57** | +1.87 | **72.13** | +3.64 |

Retrieval strategy comparison (LLaVA-OV-7B):

| Method | QAEGO4D Recall↑ | QAEGO4D Acc↑ | MLVU Acc↑ | EventHALL Acc↑ |
|------|---:|---:|---:|---:|
| LLaVA-OV | 14.0 | 52.8 | 64.7 | 60.1 |
| + ReKV | 23.9 | 54.3 | 68.5 | 60.6 |
| + C2F (Ours) | **25.2** | **55.2** | **68.9** | **61.4** |
| + Fine-only | OOM | — | — | — |

### Key Findings

1. Fine-tuning on small-scale offline data with only timestamp prompts (no TR) leads to a decline in streaming performance (-3.68%), indicating distribution mismatch.
2. Adding Temporal Reconstruction (TR) significantly improves performance (+5.82%) under the same data budget, proving the effectiveness of SOPE.
3. The optimal entropy threshold $\delta$ is 0.6: too low causes interference from frequent recalls, while too high results in insufficient temporal evidence.
4. Using only 30k offline samples and 8 GPUs matches the performance of StreamForest, which uses 121k streaming samples and 32 GPUs, demonstrating high efficiency.
5. Fine-only token-level retrieval causes OOM, validating the necessity of the coarse-to-fine (C2F) strategy.

## Highlights & Insights

1. **Compelling Diagnostic Experiments**: Showing that frame shuffling affects humans but not models clearly reveals the fundamental flaw of Video-LLMs.
2. **"Teach First, Use Later" Philosophy**: The two-stage design is elegant—injecting temporal awareness during training and utilizing it to guide retrieval during inference.
3. **Practical Uncertainty Gating**: Using current frames for low-uncertainty responses and backtracking only when uncertainty is high avoids redundant computation.
4. **Data Efficiency**: Significant gains are achieved without specialized streaming data, using only 30k random samples from general offline datasets.

## Limitations & Future Work

1. Validated only on 7B-scale models; performance on larger scales (e.g., 72B) remains untested.
2. The entropy threshold $\delta$ is a global hyperparameter and does not adapt to specific task types.
3. Temporal reconstruction assumes clear temporal cues; effectiveness may be limited in static scenes or slow-changing videos.
4. The coarse-to-fine retrieval in PCDF-Cache still requires two-stage computation, which might be a bottleneck for ultra-low latency scenarios.
5. Does not discuss the impact of historical QA context in multi-turn dialogues on current retrieval decisions.

## Related Work & Insights

- **StreamBridge**: Enhances Video-LLMs through a streaming training pipeline but requires substantial streaming data and resources.
- **ReKV**: A retrieval-based KV cache method that retains all visual memory but lacks temporal awareness.
- **StreamForest**: Manages streaming memory using clustering and forest structures, requiring 121k specialized samples and 32 GPUs.
- **Insight**: Temporal awareness may be a foundational capability for all video tasks, not just streaming; uncertainty-driven adaptive computation is a versatile design pattern.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Interactive Episodic Memory with User Feedback](interactive_episodic_memory_with_user_feedback.md)
- [\[CVPR 2026\] VisMem: Latent Vision Memory Unlocks Potential of Vision-Language Models](vismem_latent_vision_memory_unlocks_potential_of_vision-language_models.md)
- [\[ICLR 2026\] Visual Symbolic Mechanisms: Emergent Symbol Processing in Vision Language Models](../../ICLR2026/multimodal_vlm/visual_symbolic_mechanisms_vlm.md)
- [\[CVPR 2026\] MVLM: Template-Free Tracking via Vision-Language Margin Confidence and Memory-Gated Tracking](mvlm_template-free_tracking_via_vision-language_margin_confidence_and_memory-gat.md)
- [\[CVPR 2026\] Bridging the Modality Gap in Compositional Zero-Shot Learning via Sparse Alignment and Unimodal Memory Bank](bridging_the_modality_gap_in_compositional_zero-shot_learning_via_sparse_alignme.md)

</div>

<!-- RELATED:END -->
