---
title: >-
  [Paper Note] M-LLM Based Video Frame Selection for Efficient Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Video Frame Selection] A lightweight M-LLM frame selector is proposed. Trained via spatial and temporal pseudo-labels, it adaptively selects the most question-relevant frames for downstream video LLMs, improving performance across multiple video QA benchmarks without requiring fine-tuning of the downstream models.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Frame Selection"
  - "Multimodal Large Large Language Models"
  - "Video QA"
  - "Pseudo-Labeling"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: b08a23088fc2ce07
---

# M-LLM Based Video Frame Selection for Efficient Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2502.19680](https://arxiv.org/abs/2502.19680)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Video Frame Selection, Multimodal Large Large Language Models, Video QA, Pseudo-Labeling, Plug-and-Play

## TL;DR

A lightweight M-LLM frame selector is proposed. Trained via spatial and temporal pseudo-labels, it adaptively selects the most question-relevant frames for downstream video LLMs, improving performance across multiple video QA benchmarks without requiring fine-tuning of the downstream models.

## Background & Motivation

Current video M-LLMs commonly adopt uniform sampling strategies to extract a fixed number of frames from a video. This "one-size-fits-all" approach has distinct limitations:

1. **Information Loss**: Uniform sampling may miss frames representing key events. Especially in long videos, sampling once every few seconds can easily miss short-duration actions.
2. **Redundant Frame Interference**: The sampled frames may be redundant with each other or irrelevant to the question, wasting the precious context window.
3. **Efficiency Bottleneck**: Although dense uniform sampling can cover more timestamps, increasing the number of input frames $n$ significantly increases inference overhead.

Key Insight: Most video QA questions can be answered with only a few key frames. If frame selection can adaptively prioritize key frames based on the question, equivalent or superior performance compared to dense sampling can be attained using fewer frames.

## Method

### Overall Architecture

The system adopts a two-stage architecture: a lightweight frame selector is first utilized to choose $k$ key frames from $n=128$ densely sampled frames, and the selected frames are then fed into a frozen downstream video M-LLM for question answering. The frame selector works in a plug-and-play manner, requiring only a single training phase to enhance multiple different downstream models.

### Key Designs

1. **M-LLM Frame Selector Architecture**: Finetuned based on a small Qwen2.5-1.5B LLM. Taking $n$ video frames and the question text as inputs, a learnable score query $q \in \mathbb{R}^{1 \times d}$ is appended to the end of the input sequence. Utilizing the causal attention mechanism, $q$ aggregates information from all visual and textual tokens. The hidden representation $e^q$ of $q$ is extracted from the penultimate Transformer block and mapped to an $n$-dimensional importance vector $s = \text{MLP}(e^q) \in \mathbb{R}^n$ via an MLP. Key efficiency design: Aggressive spatial pooling is applied to the visual tokens of each frame, compressing them from $12 \times 12 = 144$ to $3 \times 3 = 9$ tokens, as identifying frame importance does not require fine-grained visual details.

2. **Spatial-Temporal Pseudo-Label Generation**: Due to the lack of frame-level importance annotations, two automatic pseudo-label generation strategies are designed:
    - **Spatial Pseudo-Labels**: Qwen2-VL-7B is used to score each frame independently. CoT prompting is adopted to let the model generate explanations before outputting True/False, with the importance score computed as $s = p_{\text{True}} / (p_{\text{True}} + p_{\text{False}})$.
    - **Temporal Pseudo-Labels**: An M-LLM is first used to generate captions for all frames, and then all captions and the question are fed into GPT-4o mini, prompting the LLM to perform cross-frame temporal reasoning and output a list of indices of the most relevant frames.
    - The final pseudo-labels are obtained by averaging the two, balancing single-frame spatial information and multi-frame temporal relationships.

3. **Greedy NMS Frame Sampling**: Once importance scores are obtained, instead of directly selecting the top-$k$ frames (which causes redundancy due to similar scores of adjacent frames), a greedy strategy with non-maximum suppression (NMS) is employed. After selecting the frame with the highest score at each step, its neighboring frames (distance $\leq n/4k$) are suppressed to ensure a reasonable distribution of selected frames along the timeline.

### Loss & Training

A two-stage training strategy is adopted:

- **Stage 1**: The visual encoder and the LLM backbone are frozen, and the alignment projector $g_a$, score query $q$, and score projector $g_s$ are trained. Two tasks are optimized alternately: (1) visual instruction following (cross-entropy loss) to train the projector for feature space alignment; (2) importance score prediction (binary cross-entropy loss) to initialize the scoring module.
- **Stage 2**: LoRA weights of the LLM are introduced, and only the importance score prediction task is trained to adapt the LLM to the frame selection task. Learning rate is set to $10^{-5}$ with a cosine scheduler for 5 epochs.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (PLLaVA 34B + Selector) | PLLaVA 34B baseline | Gain |
|--------|------|------|----------|------|
| ActivityNet-QA | Acc/Score | 62.3/3.6 | 60.9/3.7 | +1.4 |
| NExT-QA | Acc | 69.3 (LLaVA-NeXT 34B) | 68.1 | +1.2 |
| EgoSchema | Acc | 65.9 (Qwen2-VL 7B) | 64.6 | +1.3 |
| VideoMME | Avg Acc | 58.7 (Qwen2-VL 7B) | 58.1 | +0.6 |
| LongVideoBench | Acc | 57.0 (Qwen2-VL 7B, 32 frames) | 53.3 | +3.7 |

Consistent improvements are achieved across all tested downstream models (PLLaVA, LLaVA-NeXT-Video, Idefics2, Qwen2-VL).

### Ablation Study

| Configuration | ActivityNet-QA | NExT-QA | Description |
|------|---------|------|------|
| Uniform Sampling | 53.5 | 62.4 | baseline |
| CLIP Similarity Frame Selection | 53.7 | 62.2 | Simple text-image matching is insufficient |
| SeViLA Pseudo-Labels | 54.0 | 63.2 | Single-frame evaluation lacks temporal context |
| Spatial Pseudo-Labels Only | 54.2 | 63.6 | CoT improves evaluation quality |
| Spatial + Temporal Pseudo-Labels | 55.5 | 63.9 | Temporal reasoning is highly effective |
| Post-training Selector | 55.1 | 63.4 | Lightweight selector approaches pseudo-label upper bound |

### Key Findings

- **The frame selector achieves equivalent performance with half the frames**: The performance of 128 $\to$ 8 frame selection $\approx$ 16 frames uniform sampling, with approximately 1.5x faster inference.
- **Only 9 tokens per frame are required**: The variations are minimal from 1 to 25 tokens, validating the hypothesis that "identifying frame importance does not require fine-grained visual information."
- **A 1.5B backbone is sufficient**: The improvement from 0.5B $\to$ 1.5B is significant, whereas the gain from 1.5B $\to$ 7B is limited, demonstrating the effectiveness of the lightweight design.

## Highlights & Insights

- **Plug-and-play design**: The frame selector does not modify the parameters of downstream models. It can be trained once to serve multiple different video LLMs, showing high practicality.
- **Complementarity of spatial and temporal pseudo-labels**: Spatial labels capture single-frame content relevance, while temporal labels perform cross-frame reasoning via captions. Their integration yields the best results.
- **Aggressive token compression**: Compressing each frame to 9 tokens shows excellent design intuition—frame selection only requires coarse outlines, not detail.

## Limitations & Future Work

- Improvements on baseline models that are already highly capable (such as Qwen2-VL) are limited (+0.6% ~ 1.3%), potentially because strong models possess inherent robustness to input frame selection.
- Higher cost of pseudo-label generation (requiring prompting an M-LLM for each frame). Although only used during training, the data annotation overhead remains high.
- Underperforms SeViLA in Video Grounding (QVHighlights), indicating a gap still exists between frame selection and temporal localization.
- The selector and the downstream model are trained separately, preventing end-to-end optimization and risking suboptimal performance.

## Related Work & Insights

- SeViLA selects frames by grading each frame independently via an M-LLM, but lacks temporal reasoning and incurs high inference overhead (requiring individual inference per frame).
- The score query design in this work is similar to the object query in DETR, utilizing a learnable token to aggregate global information for prediction.
- The token compression concept can be transferred to other video understanding scenarios, such as video summarization and long video retrieval.

## Rating

- Novelty: ⭐⭐⭐⭐ The overall architecture of the frame selector is novel but not revolutionary; the pseudo-labeling strategy is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks, 4 downstream models, and extensive ablation studies render it highly complete.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear and the description of the method is detailed, though some LaTeX formulas are not standard in typesetting.
- Value: ⭐⭐⭐⭐ The plug-and-play practical design holds direct application value for industrial video QA systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Frame Selection for Long Video Understanding via Reinforcement Learning](../../CVPR2026/video_understanding/efficient_frame_selection_for_long_video_understanding_via_reinforcement_learnin.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](../../ICCV2025/video_understanding/q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[CVPR 2025\] VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](videorefer_suite_advancing_spatial-temporal_object_understanding_with_video_llm.md)
- [\[CVPR 2025\] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)
- [\[CVPR 2025\] Progress-Aware Video Frame Captioning](progress-aware_video_frame_captioning.md)

</div>

<!-- RELATED:END -->
