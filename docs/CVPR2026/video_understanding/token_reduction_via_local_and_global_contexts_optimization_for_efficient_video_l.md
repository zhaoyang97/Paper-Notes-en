---
title: >-
  [Paper Note] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models
description: >-
  [CVPR2026][Video Understanding][Video LLM] This paper proposes the AOT framework, which establishes local-global token anchors and employs Optimal Transport (OT) to aggregate the semantic information of pruned/merged tok…
tags:
  - "CVPR2026"
  - "Video Understanding"
  - "Video LLM"
  - "Token Reduction"
  - "optimal transport"
  - "training-free"
  - "Spatiotemporal Compression"
date: 2026-05-08
content_hash: f30b008457d5850d
---

# Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models

**Conference**: CVPR2026
**arXiv**: [2603.01400](https://arxiv.org/abs/2603.01400)  
**Code**: [AOT Project](https://github.com/) (to be confirmed)  
**Area**: Video Understanding
**Keywords**: Video LLM, Token Reduction, optimal transport, training-free, Spatiotemporal Compression

## TL;DR

This paper proposes the AOT framework, which establishes local-global token anchors and employs Optimal Transport (OT) to aggregate the semantic information of pruned/merged tokens at both intra-frame and inter-frame levels. The method achieves training-free video token compression, retaining 97.6% of original performance while discarding 90% of tokens.

## Background & Motivation

**Computational bottleneck in Video LLMs**: When processing video, visual encoders in Video LLMs convert sampled frames into a massive number of tokens (potentially reaching millions for long videos), with the prefilling stage accounting for the vast majority of FLOPs (approximately 98%), resulting in prohibitively high inference costs.

**Limitations of training-based compression**: Some methods compress tokens via trainable modules, but require substantial training resources and GPU overhead, making them difficult to deploy broadly.

**Spatial compression neglects temporal dependencies**: Methods such as VisionZip and LLaVA-PruMerge primarily address intra-frame spatial redundancy, and suffer severe performance degradation at low retention rates (e.g., an 8.4% drop at 10% retention) due to their failure to exploit inter-frame temporal redundancy.

**Limited efficiency of intra-LLM pruning**: Approaches such as FastV and PDrop perform token pruning within LLM layers, but shallow-layer overhead remains, and they struggle to effectively exploit the compressibility of long contexts.

**Information loss from naive merging or dropping**: Existing methods either directly discard low-importance tokens or naively merge similar ones, ignoring the subtle yet useful semantic and contextual information embedded in those tokens.

**Lack of a globally optimal information aggregation perspective**: Prior methods lack a systematic framework for measuring the relationship between pruned and retained tokens and for optimally routing useful information into the retained tokens.

## Method

### Overall Architecture

AOT (Anchors + Optimal Transport) operates in three steps:

1. **Token Anchor Construction**: Semantically important and spatially diverse token anchors are selected within each frame via local-global attention guidance.
2. **Intra-frame OT Aggregation (Phase I)**: Optimal transport is applied to aggregate the information of intra-frame pruned tokens into the anchors.
3. **Inter-frame OT Aggregation (Phase II)**: Frames are partitioned into clips, with the first frame serving as the temporal anchor; OT is used to fuse information from similar frames while retaining temporally dynamic tokens.

### Key Designs

**Local-Global Token Anchors**:

- **Global anchors**: Multi-head attention scores of the [CLS] token from the last layer of the visual encoder are used to select the Top-K high-attention tokens as global anchors $\mathbf{x}_V^g$.
- **Local anchors**: Image features are divided into $W$ non-overlapping grid windows; within each window at shallow layers, $K_w = K/W$ tokens are selected by [CLS] attention as local anchors $\mathbf{x}_V^l$.
- The final anchor set is $\mathbf{X}_V^{\text{anchors}} = \mathbf{x}_V^g \cup \mathbf{x}_V^l$, with equal global and local quotas to balance coverage.

**Intra-frame OT Aggregation**:

- Anchors $\mathbf{X}_V^a$ and non-anchors $\mathbf{X}_V^u$ are treated as two discrete distributions.
- The cost matrix is defined as $\bm{C} = \bm{1} - (\mathbf{X}_V^a)^\top \mathbf{X}_V^u$ (inverse cosine similarity).
- The optimal transport plan $\bm{T}^*_{intra}$ is solved via Sinkhorn-Knopp iterations.
- Aggregation is weighted by transport mass: $\tilde{\mathbf{x}}_j^a = \frac{\mathbf{x}_j^a + \lambda_{intra} \sum_i T^*_{ij} \mathbf{x}_i^u}{1 + \lambda_{intra} m_j}$

**Inter-frame OT Aggregation**:

- Frame sequences are divided into clips; the first frame's tokens serve as temporal anchors.
- OT transport plans are computed frame by frame for subsequent frames.
- After row-normalizing the transport plan, the maximum assignment probability for each token is computed as $q_i^{(\ell)} = \max_j p_{ij}^{(\ell)}$.
- If $q_i < \tau$ (threshold), the token is considered to carry unique information with significant temporal variation and is retained without merging.
- Otherwise, the token is aggregated into the temporal anchors via OT weights, progressively updating the anchor representations.

### Loss & Training

- No training loss is involved; the method is entirely training-free.
- The core optimization objective is to minimize the OT distance $d_{\text{OT}}(\bm{u}, \bm{v} | \bm{C})$.
- Fast solving is achieved via Sinkhorn-Knopp iterations (100 by default), with entropic regularization coefficient $\lambda$ controlling smoothness.
- The weighting coefficients $\lambda_{intra}$ and $\lambda_{inter}$ are both set to 1.0 by default, controlling the contribution strength of aggregation.

## Key Experimental Results

### Main Results

Comparison on LLaVA-OneVision-7B (32-frame input, 4 video understanding benchmarks):

| Method | FLOPs (T) | Retention | MVBench | EgoSchema | LongVideoBench | VideoMME | Avg. | Score% |
|--------|-----------|-----------|---------|-----------|----------------|----------|------|--------|
| LLaVA-OV-7B (vanilla) | 40.8 | 100% | 58.3 | 60.4 | 56.4 | 58.6 | 58.4 | 100 |
| VisionZip | 3.4 | 10% | 53.5 | 58.0 | 49.3 | 53.4 | 53.5 | 91.6 |
| PruneVid | 3.4 | 10% | 56.2 | 59.8 | 54.5 | 56.0 | 56.6 | 96.9 |
| FastVID | 3.4 | 10% | 55.9 | - | 56.3 | 57.3 | - | - |
| **AOT** | **3.4** | **10%** | **57.2** | **60.3** | **53.8** | **56.6** | **57.0** | **97.6** |

Comparison on LLaVA-Video-7B (64-frame input, 25% retention):

| Method | FLOPs (T) | Retention | MVBench | EgoSchema | LongVideoBench | VideoMME | Avg. | Score% |
|--------|-----------|-----------|---------|-----------|----------------|----------|------|--------|
| LLaVA-Video-7B (vanilla) | 80.2 | 100% | 60.4 | 57.2 | 58.9 | 64.3 | 60.2 | 100 |
| VisionZip | 9.3 | 25% | 56.7 | 54.7 | 54.7 | 60.7 | 56.7 | 94.2 |
| **AOT** | **9.3** | **25%** | **59.2** | **55.6** | **55.9** | **62.4** | **58.3** | **96.8** |

### Ablation Study

Contribution of each Token Anchor component (10% retention rate):

| Configuration | MVBench | EgoSchema | LongVideoBench | VideoMME | Avg. | Score% |
|---------------|---------|-----------|----------------|----------|------|--------|
| w/o Local Anchors | 56.5 | 60.1 | 54.0 | 55.7 | 56.6 | 96.9 |
| w/o Global Anchors | 55.5 | 59.4 | 53.4 | 53.1 | 55.4 | 94.9 |
| w/o OT | 56.1 | 60.2 | 53.5 | 55.8 | 56.4 | 96.6 |
| OT w/o Intra-frame | 57.1 | 60.2 | 53.6 | 54.6 | 56.3 | 96.6 |
| OT w/o Inter-frame | 56.1 | 60.0 | 53.6 | 55.9 | 56.4 | 96.6 |
| **AOT (Full)** | **57.2** | **60.3** | **53.8** | **56.6** | **57.0** | **97.6** |

Aggregation strategy comparison: No Merging (56.4) vs. Cosine Merging (52.4) vs. **AOT (57.0)**, confirming that OT-based aggregation substantially outperforms naive cosine merging.

### Key Findings

- **Negligible Sinkhorn computational overhead**: Intra-frame and inter-frame OT with 100 iterations requires only 2.11ms, accounting for less than 1% of total inference time.
- **Surpassing vanilla model performance**: On certain benchmarks, AOT after compression outperforms the original model, suggesting that a large number of redundant or irrelevant tokens act as noise that interferes with the LLM's focus on key information.
- **Scaling advantage with frame count**: From 16 to 128 frames, AOT consistently outperforms other compression methods; at 128 frames, the vanilla model is constrained by context length limits, whereas AOT remains fully functional.
- **Global anchors are more critical than local anchors**: Removing global anchors incurs a larger performance drop (−3.0 Avg.) than removing local anchors (−0.4 Avg.).

## Highlights & Insights

- **Novel perspective**: This is the first work to systematically study how to *optimally* aggregate information from pruned/merged tokens back into retained tokens, rather than simply discarding them.
- **Elegant application of OT**: Token compression is formulated as a discrete optimal transport problem, where suppliers (pruned tokens) transfer context to consumers (anchors), with efficient solving via Sinkhorn iterations.
- **Two-level optimization architecture**: Intra-frame spatial redundancy elimination and inter-frame temporal redundancy elimination are handled by complementary OT stages, covering the full spatiotemporal compression spectrum.
- **Fully training-free**: The method can be applied as a plug-and-play module to various Video LLMs without fine-tuning.
- **Outstanding performance under extreme compression**: At 10% retention, 97.6% of original performance is maintained, substantially surpassing VisionZip's 91.6%.

## Limitations & Future Work

- Validation is limited to LLaVA-series 7B models; generalization to larger models (e.g., 72B) or alternative architectures (e.g., Qwen-VL, InternVL) has not been demonstrated.
- Inter-frame clip partitioning relies on uniform sampling or simple clustering, without exploring adaptive frame-level importance-aware grouping.
- The threshold $\tau$ and weights $\lambda$ require manual tuning and may need adjustment for different video scenarios.
- Although OT solving is fast, it introduces additional memory overhead (cost matrix $M \times N$); scalability to extremely high frame rates or resolutions remains to be verified.
- Current evaluation is limited to multiple-choice benchmarks; assessment on open-ended generation tasks (e.g., video captioning, video grounding) is absent.

## Related Work & Insights

- **Intra-frame spatial compression**: VisionZip (CLS attention selection + token merging), LLaVA-PruMerge (adaptive spatial redundancy detection and token pruning), ToMe (bipartite matching-based token merging).
- **Intra-LLM pruning**: FastV (prefilling attention-guided token selection), PDrop (layer-wise progressive pruning), SparseVLM (text-guided visual token ranking).
- **Video-specific compression**: DyCoke (cross-frame merging + dynamic KV cache pruning), PruneVid (joint spatial-temporal clustering-based pruning), FastVID (temporal segmentation + spatiotemporal token merging), TempMe (progressive spatial pruning + adjacent clip merging), FrameFusion (shallow-layer cross-frame merging + pruning).
- **Optimal transport foundations**: Sinkhorn distance and Wasserstein distance are widely applied in distribution comparison tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introducing OT into video token compression is a novel perspective; the "aggregate rather than discard" paradigm is inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 4 benchmarks, 2 models, multiple retention rates, and thorough ablations; evaluation on larger models and open-ended tasks is lacking.
- Writing Quality: ⭐⭐⭐⭐ — The framework is clearly presented, formulations are rigorous, and figures are intuitive; some paragraphs are slightly verbose.
- Value: ⭐⭐⭐⭐ — Training-free with strong performance; high practical value with immediate relevance to Video LLM inference acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_for_efficient_video_understanding.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](../../ICLR2026/video_understanding/flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)

</div>

<!-- RELATED:END -->
