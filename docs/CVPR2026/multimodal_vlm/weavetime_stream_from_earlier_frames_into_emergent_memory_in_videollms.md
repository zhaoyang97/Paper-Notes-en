---
title: >-
  [Paper Note] WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs
description: >-
  [CVPR 2026][Multimodal VLM][Video-LLM] This paper diagnoses the *Time-Agnosticism* problem in current Video-LLMs and proposes the WeaveTime framework. During training, a temporal reconstruction auxiliary task (SOPE) endows the model with temporal awareness; during inference, an uncertainty-gated coarse-to-fine memory cache (PCDF-Cache) enables efficient adaptive memory retrieval, achieving significant gains on streaming video QA.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Video-LLM
  - streaming VQA
  - temporal order
  - memory cache
  - uncertainty-gated retrieval
date: 2026-05-08
content_hash: f0f02e46686f92f0
---

# WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs

**Conference**: CVPR 2026
**arXiv**: [2602.22142](https://arxiv.org/abs/2602.22142)
**Code**: None (coming soon)
**Area**: Multimodal / Streaming Video Understanding
**Keywords**: Video-LLM, streaming VQA, temporal order, memory cache, uncertainty-gated retrieval

## TL;DR

This paper diagnoses the *Time-Agnosticism* problem in current Video-LLMs and proposes the WeaveTime framework. During training, a temporal reconstruction auxiliary task (SOPE) endows the model with temporal awareness; during inference, an uncertainty-gated coarse-to-fine memory cache (PCDF-Cache) enables efficient adaptive memory retrieval, achieving significant gains on streaming video QA.

## Background & Motivation

**Background**: Modern visual understanding systems are increasingly deployed in streaming scenarios where frames arrive in real time (autonomous driving, human–computer interaction, live surveillance, etc.). Video-LLM-based approaches (e.g., LLaVA-Video, Qwen2-VL) perform well in offline settings but face fundamental challenges in streaming contexts.

**Limitations of Prior Work**:
1. Current Video-LLMs suffer from **Time-Agnosticism**: they treat video as an unordered bag of evidence rather than a causally ordered sequence. Experiments show that shuffling video frame order has virtually no effect on model accuracy, and performance even improves on certain temporal tasks (whereas human performance drops sharply).
2. Existing streaming methods (e.g., StreamBridge, VideoLLM-Online) either require large dedicated streaming datasets and costly training, or rely on customized memory mechanisms with limited effectiveness.
3. Compressive memory (selecting/merging/discarding visual features) loses information; retrieval-based memory preserves information but suffers from unnecessary long-range reloading and loss of temporal focus.

**Key Challenge**: Video-LLMs lack genuine temporal understanding, and existing streaming enhancement methods cannot simultaneously achieve temporal awareness and efficient memory management.

**Goal**: Address two coupled problems caused by Time-Agnosticism — **Temporal Order Ambiguity** and **Past-Current Focus Blindness**.

**Key Insight**: Teach temporal order first (during training), then exploit it (during inference) — "first teach order, then use order."

**Core Idea**: A lightweight temporal reconstruction auxiliary task enables the model to perceive frame order; uncertainty-driven coarse-to-fine retrieval then supports on-demand retrospection at inference time.

## Method

### Overall Architecture

WeaveTime is a plug-and-play, Video-LLM-agnostic streaming video QA framework comprising two core components:
1. **Training phase**: Streaming-Order Perception Enhancement (SOPE) — imparts temporal awareness via a temporal reconstruction auxiliary task.
2. **Inference phase**: Past-Current Dynamic Focus Cache (PCDF-Cache) — uncertainty gating combined with coarse-to-fine retrieval.

### Key Designs

1. **Time-Agnosticism Diagnosis**:
   - Core experiment: model vs. human performance after shuffling video frame order.
   - Model accuracy is nearly unchanged after shuffling and even improves on temporal tasks such as Temporal Perception and Action Recognition (highlighted in red).
   - Human accuracy collapses on temporal/action tasks under shuffled, timestamp-free videos (dropping from 1.0 to 0.0–0.2) and recovers when timestamps are provided.
   - Attention heatmap analysis reveals temporal positional bias: models tend to focus on the beginning and end of short videos, and predominantly on the beginning of long videos.
   - Conclusion: models rely on spatiotemporal shortcuts and positional biases rather than genuine causal reasoning.

2. **Streaming-Order Perception Enhancement (SOPE)**:
   - Designs a **Temporal Reconstruction (TR)** auxiliary task: video frames are shuffled while timestamp tokens are retained, and the model is required to recover the correct temporal order before answering.
   - Concretely, for the patch-tokenized video token sequence $\mathbf{X} = [\tilde{\mathbf{v}}_{1,1}, \ldots, \tilde{\mathbf{v}}_{1,N_f}, \tilde{\mathbf{v}}_{2,1}, \ldots]$, a timestamp token $\mathbf{ts}_i$ is inserted before each frame, after which frame contents are shuffled.
   - An instruction is prepended to the original QA prompt: "These video segments are shuffled. List each segment's true time range."
   - The LLM's inherent text reordering ability is leveraged, treating temporal prediction as a next-token prediction task without additional modules or loss functions.
   - Only 30k offline video IT samples (from LLaVA-Video-178K) are used; LoRA training for 1 epoch on 8 GPUs suffices.
   - Effect: upgrades memory from an "unordered cache" to an "ordered state chain," enabling retrieval at inference time to localize *when* events occur rather than merely *what* they contain.

3. **Past-Current Dynamic Focus Cache (PCDF-Cache)**:
   - Core strategy: "Look Now, Recall if Needed."
   - When query $q$ arrives at time $t$, the model first generates an answer $a_t^{(0)}$ using only the recent window $\mathcal{M}_{t-1}[-C:]$.
   - Prediction entropy $H_t = \text{Entropy}(a_t^{(0)})$ is computed and compared against threshold $\delta$:
     - If $H_t < \delta$: use the current answer directly (no retrospection needed).
     - If $H_t \geq \delta$: trigger Coarse-to-Fine Recall (C2F Recall).
   - **Coarse-to-Fine Retrieval**: frame-level cosine similarity ($\text{Sim}(f_i^v, f^q)$) first filters $\mathcal{M}_{\text{coarse}}$; late-interaction max-sim then performs fine-grained matching: $\text{maxSim}(\{f_{i,k}^v\}, \{f_j^q\}) = \sum_{j=1}^{N_q} \max_{1 \leq k \leq N_i} \langle f_j^q, f_{i,k}^v \rangle$
   - The top-$K$ frames are selected (capped at 64 frames), achieving token-level precision at only frame-level computational cost.

### Loss & Training

- Standard next-token prediction language modeling loss is used during training; the TR auxiliary task and original QA are merged into a single-turn dialogue.
- 30k offline video IT samples are randomly drawn; LoRA fine-tuning ($\text{lr}=1 \times 10^{-5}$), 1 epoch.
- Inference-time entropy threshold $\delta = 0.6$ (determined by ablation).
- Implemented on the ReKV codebase; maximum recalled frames capped at 64.

## Key Experimental Results

### Main Results

Streaming multi-turn evaluation based on LLaVA-OV-7B:

| Method | OVO-Bench Overall | Streaming-Bench Real-Time |
|--------|-------------------|--------------------------|
| LLaVA-OV-7B + StreamBridge | 61.72 | 68.39 |
| LLaVA-OV-7B + ReKV | 61.72 | 66.15 |
| LLaVA-OV-7B + **WeaveTime** | **68.82** (+7.10) | **72.13** (+3.74) |

Evaluation based on Qwen2-VL-7B:

| Method | OVO-Bench Overall | Streaming-Bench Real-Time |
|--------|-------------------|--------------------------|
| Qwen2-VL-7B + StreamBridge | 63.35 | 72.01 |
| Qwen2-VL-7B + ReKV | 59.72 | 70.07 |
| Qwen2-VL-7B + **WeaveTime** | **66.28** | **75.39** |

Gains are particularly pronounced on temporally sensitive sub-tasks: ACP +7.56%, EU +9.04%, ACR +11.09%.

### Ablation Study

| SOPE w/ TP | SOPE w/ TR | PCDF-Cache | OVO-Bench | Δ | Streaming-Bench | Δ |
|:---:|:---:|:---:|---:|---:|---:|---:|
| | | | 53.56 | — | 66.15 | — |
| ✔ | | | 49.88 | -3.68 | 65.91 | -0.54 |
| ✔ | ✔ | | 55.70 | +5.82 | 68.49 | +2.58 |
| ✔ | ✔ | ✔ | **57.57** | +1.87 | **72.13** | +3.64 |

Retrieval strategy comparison (LLaVA-OV-7B):

| Method | QAEGO4D Recall↑ | QAEGO4D Acc↑ | MLVU Acc↑ | EventHALL Acc↑ |
|--------|---:|---:|---:|---:|
| LLaVA-OV | 14.0 | 52.8 | 64.7 | 60.1 |
| + ReKV | 23.9 | 54.3 | 68.5 | 60.6 |
| + C2F (Ours) | **25.2** | **55.2** | **68.9** | **61.4** |
| + Fine-only | OOM | — | — | — |

### Key Findings

1. Direct fine-tuning on small-scale offline data (timestamp prompts only, without TR) actually degrades streaming performance (−3.68%), indicating distribution mismatch.
2. Adding Temporal Reconstruction yields substantial gains (+5.82%) under the same data budget, validating the effectiveness of SOPE.
3. The optimal entropy threshold is $\delta = 0.6$: too low causes frequent recall that introduces noise; too high leaves temporal evidence insufficient.
4. Using only 30k offline samples and 8 GPUs matches the performance of StreamForest trained on 121k streaming samples with 32 GPUs, demonstrating exceptional data and compute efficiency.
5. Fine-only full token-level retrieval directly causes OOM, confirming the necessity of the C2F strategy.

## Highlights & Insights

1. **The Time-Agnosticism diagnostic experiment is highly convincing** — shuffling frames has no effect on models but causes human performance to collapse, cleanly exposing a fundamental flaw in Video-LLMs.
2. **The two-stage philosophy of "teach order first, then use order"** is elegant: temporal awareness is injected during training and then exploited to guide retrieval at inference time.
3. **Uncertainty gating** is practically motivated: low-uncertainty queries are answered from the current frame window, while high-uncertainty queries trigger historical retrospection, avoiding unnecessary computation.
4. **Exceptional data efficiency**: no dedicated streaming data is required; randomly sampling 30k instances from general offline data suffices.

## Limitations & Future Work

1. Validation is limited to 7B-scale models; the effect on larger models (e.g., 72B) remains untested.
2. The entropy threshold $\delta$ is a global hyperparameter and does not adapt to task type.
3. The temporal reconstruction task assumes salient temporal cues between frames; its effectiveness may be limited for static or slowly changing videos.
4. The two-stage computation of PCDF-Cache's coarse-to-fine retrieval may become a bottleneck in extremely low-latency scenarios.
5. The influence of historical QA context in multi-turn dialogues on current retrieval decisions is not discussed.

## Related Work & Insights

- **StreamBridge**: Enhances Video-LLMs via a streaming training pipeline but requires large amounts of streaming data and significant compute.
- **ReKV**: A retrieval-based KV cache method that retains all visual memory but lacks temporal awareness.
- **StreamForest**: Manages streaming memory via clustering and forest structures, requiring 121k dedicated samples and 32 GPUs.
- **Insights**: Temporal awareness may be a foundational capability for all video understanding tasks, not just streaming scenarios; uncertainty-driven adaptive computation allocation is a broadly applicable design pattern.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)

</div>

<!-- RELATED:END -->
