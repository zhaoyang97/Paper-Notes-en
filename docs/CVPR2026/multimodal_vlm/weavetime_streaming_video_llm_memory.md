---
title: >-
  [Paper Note] WeaveTime: Stream from Earlier Frames into Emergent Memory in VideoLLMs
description: >-
  [CVPR 2026][Multimodal VLM][Video-LLM] This paper diagnoses the *Time-Agnosticism* problem in current Video-LLMs and proposes WeaveTime…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Video-LLM"
  - "streaming VQA"
  - "temporal order"
  - "memory cache"
  - "uncertainty-gated retrieval"
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

This paper diagnoses the *Time-Agnosticism* problem in current Video-LLMs and proposes WeaveTime, a framework that endows models with temporal awareness via a Shuffled-Order Prediction Enhancement (SOPE) auxiliary task during training, and achieves efficient adaptive memory retrieval at inference via an uncertainty-gated coarse-to-fine memory cache (PCDF-Cache), yielding significant gains on streaming video QA benchmarks.

## Background & Motivation

**Background**: Modern visual understanding systems are increasingly deployed in streaming scenarios where frames arrive in real time (e.g., autonomous driving, human-computer interaction, real-time surveillance). Video-LLM-based methods such as LLaVA-Video and Qwen2-VL perform well in offline settings but face fundamental challenges in streaming scenarios.

**Limitations of Prior Work**:
1. Current Video-LLMs suffer from **Time-Agnosticism**: they treat videos as unordered bags of evidence rather than causally ordered sequences. Experiments show that shuffling video frame order has almost no effect on model accuracy, and performance even improves on certain temporal tasks (whereas human performance drops drastically).
2. Existing streaming methods (e.g., StreamBridge, VideoLLM-Online) either require large dedicated streaming datasets and costly training, or rely on customized memory mechanisms with unsatisfactory results.
3. Compressive memory (selecting/merging/discarding visual features) loses information; retrieval-based memory preserves information but suffers from unnecessary long-range reloading and loss of temporal focus.

**Key Challenge**: Video-LLMs lack genuine temporal understanding, and existing streaming enhancement methods cannot simultaneously achieve temporal awareness and efficient memory management.

**Goal**: Address two coupled problems caused by Time-Agnosticism — **Temporal Order Ambiguity** and **Past-Current Focus Blindness**.

**Key Insight**: Teach temporal order first (during training), then exploit it (during inference) — "first teach order, then use order."

**Core Idea**: Enable the model to perceive frame order via a lightweight temporal reconstruction auxiliary task, then use uncertainty-driven coarse-to-fine retrieval for on-demand retrospection.

## Method

### Overall Architecture

WeaveTime is a plug-and-play, Video-LLM-agnostic streaming video QA framework comprising two core components:
1. **Training stage**: Streaming-Order Perception Enhancement (SOPE) — endows the model with temporal awareness via a temporal reconstruction auxiliary task.
2. **Inference stage**: Past-Current Dynamic Focus Cache (PCDF-Cache) — uncertainty gating + coarse-to-fine retrieval.

### Key Designs

1. **Time-Agnosticism Diagnosis**:
    - Core experiment: test model vs. human performance after shuffling video frame order.
    - Model accuracy is nearly unchanged after shuffling, and even improves on temporal tasks such as Temporal Perception and Action Recognition (highlighted in red).
    - Human accuracy on shuffled videos without timestamps collapses on temporal/action tasks (from 1.0 to 0.0–0.2) but recovers when timestamps are provided.
    - Attention heatmap analysis reveals temporal positional bias: models tend to focus on the beginning and end of short videos, and predominantly on the beginning of long videos.
    - Conclusion: models rely on spatiotemporal shortcuts and positional biases rather than genuine causal reasoning.

2. **Streaming-Order Perception Enhancement (SOPE)**:
    - Introduces the **Temporal Reconstruction (TR)** auxiliary task: video frames are shuffled with timestamp tokens retained, and the model is required to first restore the correct temporal order before answering the question.
    - Concretely, for a patch-tokenized video token sequence $\mathbf{X} = [\tilde{\mathbf{v}}_{1,1}, \ldots, \tilde{\mathbf{v}}_{1,N_f}, \tilde{\mathbf{v}}_{2,1}, \ldots]$, a timestamp token $\mathbf{ts}_i$ is inserted before each frame, and then the frame content is shuffled.
    - An instruction is prepended to the original QA prompt: "These video segments are shuffled. List each segment's true time range."
    - The temporal prediction is formulated as a next-token prediction task, leveraging the LLM's intrinsic text reordering ability without requiring additional modules or loss functions.
    - Only 30k offline video instruction-tuning samples (from LLaVA-Video-178K) are used; LoRA fine-tuning for 1 epoch on 8 GPUs suffices.
    - Effect: upgrades memory from an "unordered cache" to an "ordered state chain," enabling retrieval at inference to localize *when* events occur rather than merely *what* they contain.

3. **Past-Current Dynamic Focus Cache (PCDF-Cache)**:
    - Core strategy: "Look Now, Recall if Needed."
    - When query $q$ arrives at time $t$, the model first generates an answer $a_t^{(0)}$ using only the short temporal window $\mathcal{M}_{t-1}[-C:]$.
    - The predictive entropy $H_t = \text{Entropy}(a_t^{(0)})$ is computed and compared against threshold $\delta$:
      - If $H_t < \delta$: the current answer is used directly (no retrospection needed).
      - If $H_t \geq \delta$: coarse-to-fine recall (C2F Recall) is triggered.
    - **Coarse-to-Fine Retrieval**: frame-level cosine similarity ($\text{Sim}(f_i^v, f^q)$) is first used to select $\mathcal{M}_{\text{coarse}}$, followed by late-interaction max-sim for fine-grained matching: $\text{maxSim}(\{f_{i,k}^v\}, \{f_j^q\}) = \sum_{j=1}^{N_q} \max_{1 \leq k \leq N_i} \langle f_j^q, f_{i,k}^v \rangle$
    - Top-$K$ frames are selected (capped at 64 frames), achieving token-level precision at only frame-level computational cost.

### Loss & Training

- Standard next-token prediction language modeling loss is used during training; the TR auxiliary task and the original QA are merged into a single-turn dialogue.
- 30k offline video IT samples are randomly drawn; LoRA fine-tuning ($\text{lr}=1 \times 10^{-5}$), 1 epoch.
- Inference entropy threshold $\delta = 0.6$ (determined as optimal via ablation).
- Implementation is based on the ReKV codebase; maximum recalled frames capped at 64.

## Key Experimental Results

### Main Results

Streaming multi-turn evaluation based on LLaVA-OV-7B:

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

Gains are particularly pronounced on temporally sensitive subtasks: ACP +7.56%, EU +9.04%, ACR +11.09%.

### Ablation Study

| SOPE w/ TP | SOPE w/ TR | PCDF-Cache | OVO-Bench | Δ | Streaming-Bench | Δ |
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

1. Direct fine-tuning on small-scale offline data using only timestamp prompts (without TR) degrades streaming performance (−3.68%), indicating a distribution mismatch.
2. Adding Temporal Reconstruction yields substantial improvements (+5.82%) under the same data budget, validating the effectiveness of SOPE.
3. The optimal entropy threshold is $\delta = 0.6$: a lower value causes frequent recalls that introduce noise, while a higher value provides insufficient temporal grounding.
4. Using only 30k offline samples and 8 GPUs, WeaveTime matches StreamForest trained on 121k streaming samples with 32 GPUs, demonstrating exceptional data and computational efficiency.
5. Fine-only full token-level retrieval results in OOM, confirming the necessity of the C2F strategy.

## Highlights & Insights

1. **The Time-Agnosticism diagnostic experiment is highly convincing** — shuffling frames has no effect on models but collapses human performance, clearly exposing a fundamental deficiency in Video-LLMs.
2. **The two-stage philosophy of "teach order first, then use order"** is elegant: temporal awareness is injected during training and then leveraged to guide retrieval during inference.
3. **The uncertainty-gated design is practically appealing**: low-uncertainty queries are answered from current frames, while high-uncertainty queries trigger retrospection, avoiding unnecessary computation.
4. **Exceptional data efficiency**: no dedicated streaming data is required; randomly sampling 30k examples from general offline data suffices.

## Limitations & Future Work

1. Validation is limited to 7B-scale models; the effect on larger models (e.g., 72B) has not been assessed.
2. The entropy threshold $\delta$ is a global hyperparameter with no task-adaptive adjustment.
3. The temporal reconstruction task assumes salient temporal cues between frames, which may be less effective for static scenes or slowly changing videos.
4. The two-stage computation of PCDF-Cache's coarse-to-fine retrieval may become a bottleneck in extremely low-latency settings.
5. The influence of historical QA context in multi-turn dialogues on current retrieval decisions is not discussed.

## Related Work & Insights

- **StreamBridge**: enhances Video-LLMs via a streaming training pipeline, but requires large amounts of streaming data and computational resources.
- **ReKV**: a retrieval-based KV cache method that retains all visual memories but lacks temporal awareness.
- **StreamForest**: manages streaming memory using clustering and forest structures, requiring 121k dedicated samples and 32 GPUs.
- **Insights**: Temporal awareness may be a foundational capability for all video understanding tasks, not merely for streaming scenarios; uncertainty-driven adaptive computation allocation is a broadly applicable design pattern.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)
- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_highfidelity_incontext_learning_for_multimo.md)

</div>

<!-- RELATED:END -->
