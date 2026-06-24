---
title: >-
  [Paper Note] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow
description: >-
  [ICCV 2025][Video Understanding][long-form video understanding] Flow4Agent is the first work to introduce optical flow motion priors into LLM-based video understanding. It employs Temporal Granularity Optimization (TGO) to cluster video events via coarse-grained optical flow and filter redundant scenes using semantic priors, and Motion Token Pruning (MTP) to remove intra-frame static redundant tokens via fine-grained optical flow. The method achieves state-of-the-art performa…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "long-form video understanding"
  - "optical flow"
  - "motion prior"
  - "token pruning"
  - "MLLM"
  - "temporal granularity"
  - "key content extraction"
date: 2026-05-08
content_hash: 5d4921e84b752bde
---

# Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow

**Conference**: ICCV 2025
**arXiv**: [2510.05836](https://arxiv.org/abs/2510.05836)  
**Code**: To be confirmed  
**Area**: Long-form Video Understanding / Multimodal Large Language Models / Optical Flow Prior
**Keywords**: long-form video understanding, optical flow, motion prior, token pruning, MLLM, temporal granularity, key content extraction

## TL;DR
Flow4Agent is the first work to introduce optical flow motion priors into LLM-based video understanding. It employs Temporal Granularity Optimization (TGO) to cluster video events via coarse-grained optical flow and filter redundant scenes using semantic priors, and Motion Token Pruning (MTP) to remove intra-frame static redundant tokens via fine-grained optical flow. The method achieves state-of-the-art performance on long-video benchmarks including VideoMME, MLVU, and LongVideoBench.

## Background & Motivation

Multimodal large language models (MLLMs) have achieved notable progress on short video understanding, yet face fundamental challenges when processing hour-long videos:

**Uniform sampling causes severe information loss**: For a one-hour video, uniform sampling allocates fewer than one frame per minute, causing large amounts of critical content to be missed.

**Dense sampling introduces heavy redundancy**: Long videos contain substantial redundancy along both the temporal dimension (irrelevant frames) and the spatial dimension (repeated content within the same scene), which can overwhelm the limited context window of LLMs.

**Limitations of existing semantics-driven methods**: Current approaches for extracting key content rely heavily on semantic priors (e.g., CLIP retrieval, dense captioning), suffering from two issues: (a) strong dependence on the informativeness of the user query, causing failure when the query lacks detail; (b) susceptibility to errors in the prior models themselves, as mistakes in CLIP or captioning models propagate downstream.

Motion information derived from optical flow, as an underexplored prior for video understanding, naturally captures dynamic changes in a scene without requiring detailed user instructions or dense caption generation, enabling more robust key-content extraction at lower computational cost.

## Method

### Overall Architecture
Flow4Agent builds upon a standard video MLLM backbone (e.g., LLaVA-Video-Qwen) and introduces two modules at the input stage to handle inter-frame and intra-frame redundancy respectively: TGO addresses inter-frame redundancy by selecting key events, while MTP addresses intra-frame redundancy by pruning tokens. The two modules operate in a coarse-to-fine manner driven by optical flow priors.

### Module 1: Temporal Granularity Optimization (TGO)

TGO operates in two stages:

**Dynamic Event Split (DES)**:
- **Stage 1 (Coarse filtering)**: Video frames are converted to the HSV color space (robust to illumination changes), and the mean squared error between adjacent frames is computed; frames exceeding threshold $\theta$ are marked as candidate event boundaries.
- **Stage 2 (Precise segmentation)**: For each candidate boundary, optical flow is computed over $M=3$ frames within a temporal window using SeaRAFT (requiring only a few iterations for coarse flow); if the maximum optical flow magnitude within the window exceeds threshold $\eta$, the boundary is confirmed as a final event boundary.
- Through this motion-aware segmentation, the video is partitioned into event units, each containing a semantically consistent scene.

**Event-Center Cross-modal Query (ECQ)**:
- The middle frame of each event is selected as an anchor frame, and semantic similarity with the user query is computed using SigLiP.
- **Key innovation**: Rather than directly selecting the top-$k$ most similar frames, a statistical hypothesis testing constraint is introduced. The significance of each event $\alpha(S_i)$ is defined based on softmax-normalized similarity scores, and the selected event set $S_\text{out}$ is required to satisfy $p\text{-value} < 0.05$ while minimizing the number of selected events.
- Effect: When the query is informative, highly relevant events are individually selected; when the query lacks detail, a conservative strategy ensures that no important scene is missed, filtering only scenes with negligible significance.

### Module 2: Motion Token Pruning (MTP)

To address intra-frame redundancy, where most background regions remain static and only a small foreground region changes:

1. Fine-grained optical flow between the current frame and the next is computed using SeaRAFT (12 iterations).
2. Camera motion interference is eliminated via homography matrix compensation.
3. The primary motion region mask is obtained using U2-Net saliency detection.
4. Within the motion region, tokens corresponding to the top-50% pixels by optical flow magnitude are selected to generate the final valid token mask.
5. Anchor frames retain all tokens to preserve complete context; MTP is applied only to adjacent frames.
6. The token budget freed by pruning is used to increase the number of sampled frames, keeping the total visual context length constant.

### Loss & Training
- No additional training is required. Flow4Agent operates as an inference-time sampling and pruning strategy that can be plug-and-play applied to different MLLM backbones.
- Input image resolution is 336; the LLM context length is 8k; initial sampling uses 64 frames.
- Experiments are conducted on $2\times$ A100 GPUs.

## Key Experimental Results

### Main Results

| Model | Params | NextQA | EgoSchema | PercepTest | MLVU | L-VideoBench | VideoMME-Long | VideoMME-Overall |
|------|------|--------|-----------|------------|------|-------------|---------------|------------------|
| LLaVA-Video | 7B | 83.2 | 57.3 | 67.9 | 70.8 | 58.2 | 50.6 | 62.6 |
| Apollo | 7B | - | - | 67.3 | 70.9 | 58.5 | - | 61.3 |
| GPT-4V | - | - | 55.6 | - | - | 59.1 | 56.9 | 60.7 |
| **Flow4Agent** | **7B** | **84.0** | **61.4** | **69.6** | **71.4** | **60.4** | **54.2** | **64.7** |

- Outperforms LLaVA-Video by 3.6% and LLaVA-OneVision by 7.5% on the long-video benchmark (VideoMME-Long).
- As a 7B model, surpasses GPT-4V on most metrics.

### Cross-Model Generalization

| Base Model | Original Overall | +Flow4Agent Overall | Long Gain |
|----------|-------------|--------------------:|----------:|
| LLaVA-NeXT (7B) | 44.9 | 47.0 | +3.7 |
| LLaVA-OneVision (7B) | 58.2 | 59.9 | +3.2 |
| Qwen2-VL (7B) | 61.7 | 63.9 | +2.7 |
| LLaVA-Video (7B) | 62.6 | 64.7 | +3.6 |
| LLaVA-Video (72B) | 67.1 | 69.0 | +2.0 |

Consistent improvements are observed across all models, with the most significant gains on long videos.

### Ablation Study

| DES | ECQ | MTP | Short | Medium | Long | Overall |
|:---:|:---:|:---:|------:|-------:|-----:|--------:|
| | | | 75.9 | 61.2 | 50.6 | 62.6 |
| ✓ | | | 77.0 | 61.7 | 50.8 | 63.2 |
| | ✓ | | 75.8 | 62.3 | 52.0 | 63.4 |
| ✓ | ✓ | | 77.1 | 62.2 | 52.9 | 64.0 |
| | | ✓ | 75.9 | 61.5 | 52.4 | 63.3 |
| ✓ | ✓ | ✓ | 77.2 | 62.6 | 54.2 | 64.7 |

- DES primarily benefits short videos; ECQ primarily benefits long videos; MTP further enhances long-video understanding.
- The three modules are complementary, and their combination yields the best overall performance.

### Optical Flow Model Selection
Sea-RAFT (4 iter for TGO / 12 iter for MTP) achieves the best performance, outperforming NeuFlow and StreamFlow by 1.2% and 0.3% respectively on the Long category.

## Highlights & Insights

- **Pioneering contribution**: This is the first work to introduce optical flow motion priors into LLM-based video understanding, opening a new direction for motion-assisted long-form video comprehension.
- **Coarse-to-fine optical flow utilization**: TGO uses coarse-grained optical flow (4 iter) for rapid event clustering, while MTP uses fine-grained optical flow (12 iter) for precise token pruning, with a well-balanced computational budget allocation.
- **Statistical hypothesis testing for event selection**: Rather than naively selecting top-$k$ events, the method adaptively determines the number of selected events via a $p\text{-value} < 0.05$ constraint, elegantly balancing information retention and redundancy removal.
- **Model-agnostic plug-and-play design**: No training is required; the method directly enhances arbitrary video MLLMs, with consistent effectiveness from 7B to 72B models.
- **High frame efficiency**: The advantage is most pronounced under frame budget constraints, demonstrating that motion priors genuinely help the model select the right frames rather than simply observe more frames.

## Limitations & Future Work

- The optical flow computation introduces additional inference overhead (SeaRAFT + U2-Net), and end-to-end latency comparisons are not reported.
- The HSV threshold $\theta$ and optical flow threshold $\eta$ require manual tuning, which may need adjustment for different video types.
- The significance level $p = 0.05$ is fixed, with no sensitivity analysis across different threshold values.
- The top-50% token retention ratio in MTP is fixed, which may be suboptimal for scenes with sparse or dense motion.
- The method has not been validated in streaming or online inference settings; the current design assumes access to the full video.
- Optical flow estimation itself may be unreliable for fast motion or occluded scenes, potentially introducing noise in extreme cases.

## Related Work & Insights

- **vs. VideoAgent**: VideoAgent relies on GPT-4 for dense captioning and CLIP-based keyframe retrieval, incurring high cost and strong dependence on semantic priors. Flow4Agent replaces part of this semantic dependence with motion priors, yielding greater robustness and efficiency.
- **vs. LongVU**: LongVU uses DINOv2 to filter spatially redundant tokens but ignores motion information. Flow4Agent's MTP leverages optical flow to more precisely identify dynamic foreground regions.
- **vs. MovieChat/Flash-VStream**: These methods handle long videos via streaming memory or sliding windows. Flow4Agent approaches the problem from an information selection perspective, packing more critical information into a limited context.
- Motion priors have been applied in traditional video understanding tasks (e.g., action recognition, data curation); this work is the first to transfer them to the MLLM paradigm, inspiring future exploration of additional modality priors (e.g., audio, depth).

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce optical flow priors into LLM-based video understanding; the hypothesis-testing-based event selection in TGO is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six benchmarks, five base models, and detailed ablations covering components, frame counts, optical flow models, and semantic models.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic method description, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Plug-and-play with consistent effectiveness; represents a substantive contribution to the field of long-form video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)
- [\[ICCV 2025\] HERMES: temporal-coHERent long-forM understanding with Episodes and Semantics](hermes_temporal-coherent_long-form_understanding_with_episodes_and_semantics.md)
- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](../../CVPR2025/video_understanding/re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)

</div>

<!-- RELATED:END -->
