---
title: >-
  [Paper Note] Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM
description: >-
  [NeurIPS 2025][Multimodal VLM][audio-visual understanding] This paper proposes TriSense — a tri-modal (visual + audio + speech) large language model that adaptively modulates per-modality weights via a Query-Based Connector for robust video temporal understanding, supported by the TriSense-2M dataset containing 2 million annotated samples.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - audio-visual understanding
  - multimodal fusion
  - temporal understanding
  - modality adaptation
  - video moment retrieval
date: 2026-05-08
content_hash: 9fdbd2a61d6b671b
---

# Watch and Listen: Understanding Audio-Visual-Speech Moments with Multimodal LLM

**Conference**: NeurIPS 2025
**arXiv**: [2505.18110](https://arxiv.org/abs/2505.18110)
**Code**: [GitHub](https://github.com/zinuoli/TriSense)
**Area**: Multimodal VLM
**Keywords**: audio-visual understanding, multimodal fusion, temporal understanding, modality adaptation, video moment retrieval

## TL;DR

This paper proposes TriSense — a tri-modal (visual + audio + speech) large language model that adaptively modulates per-modality weights via a Query-Based Connector for robust video temporal understanding, supported by the TriSense-2M dataset containing 2 million annotated samples.

## Background & Motivation

Humans naturally integrate visual and auditory cues when understanding video. For instance, localizing a moment such as "a scientist enthusiastically discussing wildlife conservation, with background music and an applauding audience" requires simultaneously processing visual, audio, and speech signals. However, existing MLLMs face two core challenges in audio-visual fusion:

**1. Insufficient and incomplete training data**: Existing datasets mostly consist of short clips and lack large-scale, complete annotations across three modalities. More critically, in real-world videos, not all modalities are simultaneously present — there may be silent footage, pure background music, or scenes that naturally lack certain signals. When models are trained only on data with all three modalities present, they degrade severely when modalities are missing.

**2. Lack of modality-adaptive mechanisms**: Existing MLLMs cannot assess the relative importance of each modality based on task or query context. LongVALE compresses all modality tokens into a single representation, causing information loss and preventing handling of missing modalities; Qwen2.5-Omni introduces temporal positional encoding but still underperforms on fine-grained temporal tasks over long videos.

These limitations motivate the authors to build a model capable of flexibly handling arbitrary modality combinations while adaptively regulating modality contributions.

## Method

### Overall Architecture

The TriSense architecture comprises four core modules: (1) three dedicated encoders for visual, audio, and speech processing respectively; (2) modality-specific projectors for dimensionality transformation; (3) a Query-Based Connector for query-adaptive multimodal feature fusion; and (4) an LLM backbone with a Time Encoder for generating temporally aligned outputs. The model supports 8 task configurations (AVS/VS/AV/V × segment captioning/moment retrieval).

### Key Designs

1. **Multimodal feature extraction**: $n=64$ frames are uniformly sampled from the video; timestamps are recorded for each frame and audio clips spanning ±1 second around each frame are extracted. Pretrained expert encoders extract visual tokens $f^v_i$, audio tokens $f^a_i$, and speech tokens $f^s_i$ respectively. Slot-Based Compression reduces each modality's tokens to a fixed 16 tokens to control computational cost. Timestamps are encoded by the Time Encoder as 6 character tokens (e.g., `⟨0⟩⟨1⟩⟨2⟩⟨3⟩⟨.⟩⟨4⟩`).

2. **Query-Based Connector**: This is the core innovation of TriSense. Compressed modality features first interact with the query representation via Cross-Attention to obtain query-aware features $f^{v,q}_i, f^{a,q}_i, f^{s,q}_i$. An adaptive weighting mechanism then determines the importance of each modality:

   Global average pooling is applied to each modality to obtain compact representations $c_v, c_a, c_s$, which are concatenated and fed into a single-layer MLP to produce unnormalized weights, followed by softmax normalization such that $w_v + w_a + w_s = 1$:

   $w_m = \frac{\exp(\tilde{w}_m)}{\sum_{m' \in \{v,a,s\}} \exp(\tilde{w}_{m'})}, \quad m \in \{v,a,s\}$

   The fused multimodal representation is:

   $\mathcal{X}_m = \hat{\mathcal{F}}(C_{comp}(\text{concat}(w_v f^{v,q}_i, w_a f^{a,q}_i, w_s f^{s,q}_i)))$

   where $C_{comp}$ is slot compression that reduces the tripled token count back to the original size, and $\hat{\mathcal{F}}$ is a two-layer MLP for feature refinement. This design enables the model to emphasize the most relevant modality and suppress irrelevant or missing ones based on query content.

3. **Causal event prediction**: The video is segmented into an event sequence $\{e_1, e_2, \cdots, e_K\}$, where each event contains a timestamp and a description. The model predicts the next event conditioned on preceding events. A special `⟨sync⟩` token enables adaptive switching between the temporal head and the language head, allowing the model to naturally alternate between timestamp prediction and text description generation.

### Loss & Training

Training proceeds in two stages: Stage 1 performs modality alignment (without temporal information); Stage 2 conducts temporal understanding training on TriSense-2M.

**TriSense-2M dataset construction**: Starting from 5 million initial samples drawn from InternVid and VAST, two models fine-tuned on Qwen2.5-72B are employed — a Generator that merges modality-specific descriptions into cross-modal descriptions, and a Judger that evaluates quality on a 0–5 scale (samples scoring ≥3 are retained). The final dataset contains 2 million high-quality samples covering approximately 38,000 long videos with an average duration of 905 seconds. The dataset explicitly supports missing-modality scenarios (AVS/AV/VS and other combinations).

## Key Experimental Results

### Main Results

**TriSense-2M Segment Captioning**

| Model | AVS-SC (B/M/R/C) | VS-SC (B/M/R/C) | AV-SC (B/M/R/C) | V-SC (B/M/R/C) |
|-------|-------------------|------------------|------------------|-----------------|
| VTimeLLM | 0.8/8.2/16.1/2.4 | 1.2/8.8/16.9/3.1 | 1.3/10.3/17.9/2.6 | 1.4/10.4/18.2/4.0 |
| TRACE-uni | 1.1/8.2/14.7/1.4 | 1.5/8.3/15.1/2.2 | 1.6/9.5/16.3/2.3 | 1.3/9.9/17.6/8.8 |
| LongVALE | 1.2/8.6/16.7/4.9 | 2.3/10.0/20.1/5.5 | 2.5/11.4/21.3/5.9 | 1.5/11.5/18.8/0.9 |
| Qwen2.5-Omni | 0.8/8.8/13.1/1.7 | 0.8/8.6/13.1/0.8 | 1.2/9.8/15.1/1.3 | 1.1/10.1/14.6/1.1 |
| **TriSense** | **3.4/10.1/20.1/8.3** | **3.0/10.0/22.2/11.8** | **5.3/12.2/26.3/15.4** | **7.3/12.6/30.7/36.3** |

**Zero-shot Moment Retrieval (Public Benchmarks)**

| Model | Charades-STA (R@0.5/R@0.7/mIoU) | ActivityNet (R@0.5/R@0.7/mIoU) |
|-------|----------------------------------|-------------------------------|
| TRACE-uni | 43.7/21.0/41.5 | 38.2/24.7/39.4 |
| NumPro-FT | 42.0/20.6/41.4 | 37.5/20.6/38.8 |
| **TriSense** | 42.3/**27.6**/39.8 | **39.6**/**27.2**/**40.1** |

### Ablation Study

| Configuration | AVS-MR (R@0.5/R@0.7) | Note |
|---------------|----------------------|------|
| Stage 1 Only | 0.07/0.01 | Modality alignment only, no temporal modeling |
| Stage 1+2 | 0.52/0.19 | Large gain after adding temporal training |
| Addition (simple sum) | 0.71/0.22 | Cannot differentiate modality importance |
| Fixed Weights (equal) | 0.89/0.38 | Equal weight assignment |
| **TriSense (adaptive weights)** | **1.12/0.42** | Query-driven dynamic weights are optimal |
| 32 frames | 0.74/0.27 | Performance drops with halved frame count |
| 64 frames | 1.12/0.42 | Default configuration |
| 128 frames | 1.12/0.43 | Marginal returns diminish with more frames |

### Key Findings

- TriSense shows the most significant advantage in the tri-modal (AVS) setting, with CIDEr improving from 4.9 (LongVALE) to 8.3
- In the V-SC (vision-only captioning) setting, CIDEr improves from 8.8 (best baseline) to 36.3, demonstrating that multimodal fusion also benefits single-modality tasks
- Query-driven adaptive weights substantially outperform fixed weights (+26%) and simple addition (+58%)
- TriSense achieves the best R@0.7 on both public benchmarks, indicating higher precision in moment localization

## Highlights & Insights

- **Modality-adaptive fusion** is the core contribution — different queries may require different modality support, and learned softmax weights enable flexible regulation
- Missing modalities are explicitly addressed via a Modality Dropout training strategy, enabling the model to operate under arbitrary modality combinations
- The Generator+Judger dual-model pipeline for dataset construction is a noteworthy practice: fine-tuned LLMs replace API calls to enable large-scale automatic annotation
- The `⟨sync⟩` token for adaptive switching between the temporal head and the language head is an elegant engineering design

## Limitations & Future Work

- Only 64 frames are used at test time; for very long videos (average 905 seconds), the sampling density may be insufficient for fine-grained localization
- TriSense slightly underperforms vision-specialized models such as TRACE on vision-only moment retrieval (0.43 vs. 0.48 R@0.5)
- Performance on the LongVALE captioning benchmark is less prominent than on the model's own dataset, possibly due to differences in description style
- Computational overhead is non-trivial — three encoders, Cross-Attention, and the Connector impose costs that warrant future optimization for inference efficiency

## Related Work & Insights

- LongVALE attempts to integrate three modalities but compresses all tokens into a single representation, incurring severe information loss
- TRACE introduces causal event modeling to improve temporal understanding; TriSense inherits this paradigm
- Insight: The modality-adaptive weighting design is generalizable to other multimodal settings (e.g., text + image + table); the key principle is "letting the model learn when to attend to what"

## Rating

- Novelty: ⭐⭐⭐⭐ The Query-Based Connector enabling tri-modal adaptive fusion is a significant contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 task types × multiple benchmarks × detailed ablations × zero-shot evaluation — highly comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear structure with thorough description of the dataset construction process
- Value: ⭐⭐⭐⭐ The 2M-scale high-quality tri-modal dataset and modality-adaptive framework advance the state of audio-visual understanding

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/av_speakerbench_audiovisual_human_speech_understanding_mllms.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] UniTok: A Unified Tokenizer for Visual Generation and Understanding](unitok_a_unified_tokenizer_for_visual_generation_and_understanding.md)
- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)

<!-- RELATED:END -->
