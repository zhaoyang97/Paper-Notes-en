---
title: >-
  [Paper Note] FastVID: Dynamic Density Pruning for Fast Video Large Language Models
description: >-
  [Video Understanding] This paper proposes FastVID, which systematically eliminates video token redundancy along both temporal and visual dimensions via Dynamic Temporal Segmentation (DySeg) and Density Spatiotemporal Pruning (STPrune). On LLaVA-OneVision-7B, FastVID retains 98% accuracy after pruning 90.3% of video tokens, achieving a 7.1× speedup in the LLM prefill stage.
tags:
  - "Video Understanding"
date: 2026-05-08
content_hash: 80a49cc96df44536
---

# FastVID: Dynamic Density Pruning for Fast Video Large Language Models

## Basic Information
- **arXiv**: 2503.11187
- **Conference**: NeurIPS 2025
- **Authors**: Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, Pengzhang Liu, Sicheng Zhao, Guiguang Ding
- **Institutions**: Tsinghua University, JD.com
- **Code**: https://github.com/LunarShen/FastVID

## TL;DR
This paper proposes FastVID, which systematically eliminates video token redundancy along both temporal and visual dimensions via Dynamic Temporal Segmentation (DySeg) and Density Spatiotemporal Pruning (STPrune). On LLaVA-OneVision-7B, FastVID retains 98% accuracy after pruning 90.3% of video tokens, achieving a 7.1× speedup in the LLM prefill stage.

## Background & Motivation
Video LLMs incur high inference costs due to the enormous number of video tokens. Existing approaches include:
- **Image token compression** (FastV, VisionZip, LLaVA-PruMerge): addresses only spatial redundancy, ignoring inter-frame temporal dependencies.
- **Video compression** (DyCoke, PruneVID, FrameFusion): either disrupts temporal structure, achieves insufficient compression, or introduces significant latency.

Core insight: video token redundancy must be analyzed from two dimensions—
1. **Temporal context**: frame order and continuity affect semantic understanding (shuffling or dropping frames causes errors).
2. **Visual context**: both "representativeness" and "distinctiveness" must be preserved simultaneously.

## Core Problem
How to extremely compress video tokens (>90%) at inference time while preserving temporal structure and visual semantic integrity?

## Method

### 1. Dynamic Temporal Segmentation (DySeg)
The video is adaptively partitioned into temporally ordered segments with high intra-segment redundancy.

Cosine similarity between adjacent frames is computed as $t_i = \cos(\mathbf{f}_i, \mathbf{f}_{i+1})$, and split points are selected as:
$$\mathbf{S} = \mathbf{S}_1 \cup \mathbf{S}_2$$
- $\mathbf{S}_1$: the $c-1$ least similar transition points (guaranteeing a minimum number of segments).
- $\mathbf{S}_2$: transition points with similarity below threshold $\tau$ (adaptively capturing scene changes).

Simple videos yield fewer segments; complex videos yield more. This contrasts with fixed-interval splitting (which does not guarantee intra-segment similarity) and clustering-based splitting (which destroys temporal order).

### 2. Density Spatiotemporal Pruning (STPrune)
A two-stage pruning scheme is applied within each high-redundancy segment:

**Density-based Token Merging (DTM)**: preserves segment-level visual context.
- Anchor frames are selected every $p$ frames; density peak clustering identifies anchor tokens:
$$\rho_i = \exp\left(-\frac{1}{k}\sum_{v_j \in \text{kNN}(v_i)} d(v_i, v_j)^2\right)$$
$$\delta_i = \min_{j: \rho_j > \rho_i} d(v_i, v_j)$$
- Tokens with high $\rho_i \times \delta_i$ serve as anchors, capturing both representativeness and distinctiveness.
- Anchor-centered aggregation: $a^* = \beta a + \frac{1-\beta}{n}\sum_{i=1}^n b_i$ (with $\beta=0.6$).
- Crucially, original positional information of anchor tokens is preserved to maintain the spatiotemporal structure of RoPE encodings.

**Attention-based Token Selection (ATS)**: captures salient visual details.
- [CLS] attention scores are obtained from the pretrained SigLIP head.
- Tokens with the highest attention scores are selected, complementing DTM.

An allocation ratio of $d=0.4$ (DTM 40%, ATS 60%) yields the best performance.

## Key Experimental Results

### LLaVA-OneVision-7B (32 frames)

| Method | Retention Rate | TFLOPs | MVBench | VideoMME | Avg% |
|---|---|---|---|---|---|
| Vanilla | 100% | 48.82 | 56.9 | 58.6 | 100% |
| VisionZip* | 9.7% | 4.04 | 51.7 | 53.1 | 89.6% |
| PruneVID* | 10.1% | 4.23 | 54.2 | 55.9 | 95.4% |
| **FastVID** | **9.7%** | **4.04** | **55.9** | **57.3** | **98.0%** |

### Efficiency Comparison (LLaVA-OneVision)

| Method | Prefill Time | Speedup | Avg Acc |
|---|---|---|---|
| Vanilla | 476.3ms | 1.0× | 100% |
| PruneVID | 101.5ms | 4.7× | 95.4% |
| **FastVID** | **67.2ms** | **7.1×** | **98.0%** |

### Cross-Model Generalization
- LLaVA-Video-7B (64 frames): 25% retention → 98.1% accuracy.
- Qwen2-VL (768 frames): 25% retention → 96.2% accuracy.
- Qwen2.5-VL (768 frames): 24.1% retention → 93.3% accuracy.

### Length Extrapolation

| Setting | Frames | Tokens | VideoMME |
|---|---|---|---|
| Vanilla | 32 | 6272 | 58.6 |
| FastVID (r=25%) | 128 | 6272 | 60.4 (+1.8) |
| FastVID (r=10%) | 320 | 6080 | 61.4 (+2.8) |

→ Under the same token budget, sampling more frames with fewer tokens per frame is strictly superior.

## Highlights & Insights
1. **Systematic analysis**: video redundancy is dissected along both temporal and visual dimensions, with targeted solutions designed for each.
2. **Robustness under extreme compression**: 98% accuracy is maintained at a 90.3% pruning rate, far surpassing comparable methods.
3. **Training-free, plug-and-play**: a purely inference-time method compatible with FlashAttention, KV cache, and multi-turn dialogue.
4. **Broad generalizability**: achieves state-of-the-art results across four distinct Video LLM architectures.
5. **Density peak selection**: anchor tokens simultaneously satisfy representativeness and distinctiveness, outperforming uniform sampling and clustering.

## Limitations & Future Work
1. Query-agnostic pruning — question-relevant frames in long videos may be erroneously pruned.
2. Performance gains are less pronounced on long-frame models (LLaVA-Video 64 frames, Qwen2-VL 768 frames) compared to short-frame models.
3. Despite parallel optimization, density score computation still introduces additional latency (5.6ms).
4. DySeg hyperparameters ($c=8, \tau=0.9$) are fixed uniformly across all videos.

## Related Work & Insights
- **vs. FastV**: FastV prunes single-frame tokens based on LLM attention and does not address inter-frame redundancy.
- **vs. VisionZip**: VisionZip applies content-agnostic uniform sampling and merging; FastVID's density-guided selection better preserves semantics.
- **vs. DyCoke**: DyCoke performs cross-frame merging and KV cache compression but achieves insufficient prefill-stage compression.
- **vs. PruneVID**: PruneVID uses clustering-based segmentation and merging, losing positional information and exhibiting slower speed (6.1× slower).
- **vs. LLaVA-PruMerge**: PruMerge performs spatial-only pruning, whereas FastVID performs joint spatiotemporal pruning.

Regarding broader connections: FastVID complements streaming temporal modeling methods (e.g., Eyes Wide Open) and could be combined for efficient streaming video understanding. The length extrapolation results establish that "more frames with fewer tokens per frame" is preferable to "fewer frames with full tokens," offering a new design principle for Video LLMs. Furthermore, spatial pruning and spatiotemporal pruning can be applied in a cascaded fashion.

## Rating
- Novelty: ★★★★☆ — Novel application of density peak clustering to token pruning.
- Technical Depth: ★★★★☆ — DySeg + STPrune design is elegant, with thorough ablations.
- Experimental Thoroughness: ★★★★★ — Covers 4 models × 4 benchmarks × multiple compression rates; exceptionally comprehensive.
- Writing Quality: ★★★★☆ — Well-structured but slightly verbose.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[CVPR 2025\] DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models](../../CVPR2025/video_understanding/divprune_diversity-based_visual_token_pruning_for_large_multimodal_models.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](../../CVPR2025/video_understanding/video_summarization_with_large_language_models.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)

</div>

<!-- RELATED:END -->
