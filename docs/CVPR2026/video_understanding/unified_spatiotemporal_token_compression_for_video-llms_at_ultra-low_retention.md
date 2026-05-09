---
title: >-
  [Paper Note] Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention
description: >-
  [CVPR 2026][Video Understanding][Visual token compression] This paper proposes a unified spatiotemporal token compression method that jointly evaluates token contribution and semantic redundancy via a global retention pool, and introduces a text-aware merging mechanism inside the LLM. At an extreme compression ratio retaining only ~2% of visual tokens, the method preserves 90.1% of baseline performance while reducing FLOPs to ~2.6%.
tags:
  - CVPR 2026
  - Video Understanding
  - Visual token compression
  - video large language models
  - unified spatiotemporal compression
  - inference acceleration
  - training-free
date: 2026-05-08
content_hash: e01da3acd791dd75
---

# Unified Spatiotemporal Token Compression for Video-LLMs at Ultra-Low Retention

**Conference**: CVPR 2026
**arXiv**: [2603.21957](https://arxiv.org/abs/2603.21957)
**Code**: None
**Area**: Video Understanding / Multimodal VLM / LLM Efficiency
**Keywords**: Visual token compression, video large language models, unified spatiotemporal compression, inference acceleration, training-free

## TL;DR
This paper proposes a unified spatiotemporal token compression method that jointly evaluates token contribution and semantic redundancy via a global retention pool, and introduces a text-aware merging mechanism inside the LLM. At an extreme compression ratio retaining only ~2% of visual tokens, the method preserves 90.1% of baseline performance while reducing FLOPs to ~2.6%.

## Background & Motivation

1. **State of the Field**: Video-LLMs (e.g., LLaVA-OneVision-7B) achieve strong performance on complex video understanding tasks, but generate 196 visual tokens per frame — accumulating to 6,272 tokens for a 32-frame video — with substantial redundancy, leading to high inference latency and memory consumption.

2. **Limitations of Prior Work**: Existing training-free video token compression methods fall into three categories: spatial pruning (VisionZip, PruMerge), temporal pruning (DyCoke, TempMe), and staged spatiotemporal methods (FastVid, HoliTom). These methods typically adopt two-stage strategies (temporal-then-spatial or spatial-then-temporal) with independent scoring, implicitly assuming that spatial and temporal redundancy are separable.

3. **Root Cause**: At ultra-low retention rates (≤5%), the spatiotemporal separability assumption breaks down. Staged decision-making leads to imbalanced spatiotemporal resource allocation — retaining non-critical tokens while discarding critical ones. For example, FastVid retains only 83.3% of original performance at 2% retention. Moreover, intra-LLM pruning methods (e.g., FastV, PDrop) rely solely on the last token's attention weights for selection, introducing positional bias and weakening the semantic influence of critical query tokens.

4. **Paper Goals**: (a) How to globally allocate spatiotemporal tokens under a unified budget to maximize information contribution while minimizing redundancy? (b) How to further compress tokens inside the LLM based on query relevance?

5. **Starting Point**: Reframe token compression as a global spatiotemporal token allocation problem rather than a staged independent process. Jointly evaluate all tokens using attention weights and semantic similarity.

6. **Core Idea**: Replace two-stage compression with a unified global retention pool; use dual criteria of contribution and redundancy for token selection; apply DPC-KNN clustering-based merging on the recycle pool; and introduce text-aware merging inside the LLM — achieving efficient compression at ultra-low retention rates.

## Method

### Overall Architecture
The method consists of two core components: (1) an external unified spatiotemporal token compression module — maintaining a retention pool and a recycle pool, globally selecting high-contribution, low-redundancy tokens via attention scores and cosine similarity, with unselected tokens merged via DPC-KNN clustering and backfilled into the retention pool; (2) an intra-LLM text-aware merging mechanism — further retaining visual tokens most relevant to the query based on cross-attention between text and visual tokens and semantic similarity.

### Key Designs

1. **Spatiotemporal Pruning**:

    - **Function**: Select high-contribution, low-redundancy tokens from all visual tokens.
    - **Mechanism**: Attention scores from the CLS token $A_h = \text{Softmax}(Q_h K_h^\top / \sqrt{d})$ are used to quantify each token's contribution. For encoders without a CLS token (e.g., SigLIP), the average attention of each token over all others serves as a substitute. After selecting top-k high-attention tokens, the maximum cosine similarity of each candidate token to existing tokens in the retention pool is computed as $S = \max_{p \in \mathcal{P}} \frac{c \cdot p}{\|c\|\|p\|}$. Only tokens with similarity below threshold $\tau$ enter the retention pool; others are sent to the recycle pool. This process iterates until the retention pool reaches its target capacity.
    - **Design Motivation**: Attention scores measure contribution while cosine similarity detects redundancy. Their combination prevents retaining high-attention but highly redundant tokens, resolving the imbalanced spatiotemporal allocation problem of two-stage methods.

2. **Spatiotemporal Clustering**:

    - **Function**: Preserve semantic information from recycle pool tokens to avoid information loss from direct discarding.
    - **Mechanism**: DPC-KNN clustering is applied. For each token in the recycle pool, local density $\rho_i$ and the distance to the nearest higher-density token $\delta_i$ are computed. Cluster centers are selected by decision score $\gamma_i = \rho_i \times \delta_i$; remaining tokens are assigned to their nearest center and averaged to form merged tokens, which are then backfilled into the retention pool in original spatiotemporal order.
    - **Design Motivation**: Direct discarding compromises semantic completeness. Clustering-based merging preserves the overall semantic structure, ensuring the retention pool contains both high-contribution selected tokens and supplementary information from clustered compression.

3. **Text-Aware Merging**:

    - **Function**: Further compress visual tokens inside the LLM based on semantic relevance to the text query.
    - **Mechanism**: The text-to-visual submatrix $A_{qv}$ is extracted from the attention matrix. The maximum cross-attention score $A_m$ per visual token is computed and normalized. Simultaneously, the maximum cosine similarity $S_m(v_i)$ between each visual token and all text tokens is computed. The final decision score $I(v_i) = (1-\lambda) \cdot A_m^{\text{norm}} + \lambda \cdot S_m^{\text{norm}}$ combines both signals. The top-R% tokens are retained, and pruned tokens are merged into their nearest retained token based on cosine similarity.
    - **Design Motivation**: Using only the last token's attention (as in FastV) introduces positional bias due to RoPE's relative position encoding, favoring adjacent tokens. This method leverages attention from all text tokens to globally identify the most query-relevant visual information, with cosine similarity complementing to reduce position sensitivity.

### Loss & Training
The entire method is completely training-free, serving as a plug-and-play module compatible with existing Video-LLMs without modifying original model parameters. Hyperparameter settings: similarity threshold $\tau=0.7$, clustering ratio 0.3, intra-LLM merging activated from layer 18, retaining top 50% of visual tokens, $\lambda=0.5$.

## Key Experimental Results

### Main Results
Comparison on LLaVA-OneVision-7B (average across 5 benchmarks):

| Retention | Method | FLOPs(T) | MVBench | EgoSchema | MLVU | LVBench | VideoMME | Avg | Score% |
|-----------|--------|----------|---------|-----------|------|---------|----------|-----|--------|
| 100% | Original | 41.4 | 58.3 | 60.4 | 47.7 | 56.4 | 58.6 | 56.3 | 100% |
| 2% | FastVID | 1.2 | 48.0 | 52.3 | 37.6 | 47.3 | 49.2 | 46.9 | 83.3% |
| 2% | HoliTom | 1.1 | 52.6 | 57.2 | 37.4 | 48.5 | 51.1 | 49.4 | 87.7% |
| 2% | **Ours** | **1.1** | **52.8** | **57.6** | **40.3** | **50.8** | **51.8** | **50.7** | **90.1%** |

Cross-backbone results (LLaVA-Video-7B, 2% retention):

| Method | FLOPs ratio | MVBench | MLVU | VideoMME | Avg | Score% |
|--------|-------------|---------|------|----------|-----|--------|
| HoliTom | 1.7% | 50.2 | 39.9 | 55.3 | 48.5 | 82.5% |
| **Ours** | **1.7%** | **50.1** | **40.8** | **56.2** | **48.8** | **83.0%** |

### Ablation Study

| Configuration | Avg @ 5% retention | Avg @ 2% retention | Notes |
|---------------|--------------------|--------------------|-------|
| Full model | 53.7 | 50.7 | Complete method |
| w/o internal merging | 53.4 | 50.4 | Removing text-aware merging, −0.3 |
| HoliTom (two-stage) | 52.9 | 49.4 | Two-stage baseline; larger gap at lower retention |

### Key Findings
- At ultra-low retention (2%, ~4 tokens per frame), the proposed method outperforms the two-stage method HoliTom by 2.4% (Score%: 87.7→90.1), validating the advantage of unified spatiotemporal allocation.
- Cross-backbone experiments (LLaVA-Video-7B, LLaVA-OV-0.5B, Qwen2.5-VL-7B) all demonstrate effectiveness, confirming the method's generalizability.
- Text-aware merging contributes more noticeably at lower retention rates, indicating that query-guided secondary compression is more critical for preserving key information when tokens are extremely scarce.
- FLOPs can be reduced to ~2.6% of the original, with substantial reductions in end-to-end inference latency and memory consumption.

## Highlights & Insights
- **Global Retention Pool Design**: Transforms token compression from staged independent optimization to global joint optimization — analogous to upgrading from "local greedy" to a "global perspective." This idea is transferable to any scenario involving multi-dimensional resource allocation.
- **Clustering Backfill from Recycle Pool**: Rather than simply discarding low-score tokens, clustering-based merging and backfilling preserve information completeness — a practical "no information wasted" principle.
- **Fully Training-Free Plug-and-Play Design**: Requires no fine-tuning of model weights and is directly compatible with multiple Video-LLMs, lowering the barrier to deployment.

## Limitations & Future Work
- Relies on the quality of attention scores from the visual encoder; if the encoder's attention distribution is suboptimal, pruning effectiveness may be limited.
- Hyperparameters such as the similarity threshold $\tau$ and clustering ratio require manual tuning; adaptive adjustment has not been explored.
- Evaluation is limited to multiple-choice benchmarks; open-ended generation tasks (e.g., video captioning) are not assessed.
- Text-aware merging requires access to intermediate layers of the LLM, making it difficult to apply to API-based models that do not expose internal activations.

## Related Work & Insights
- **vs. HoliTom**: HoliTom uses dynamic programming for frame segmentation with two-stage pruning and merging; this work uses a global retention pool for unified processing. HoliTom performs reasonably at moderate retention rates but degrades more rapidly at ultra-low retention.
- **vs. FastV**: FastV performs intra-LLM pruning using only the last token's attention, lacking external compression and suffering from positional bias. This work applies dual compression both externally and internally, using multi-token attention to mitigate bias.
- **vs. VisionZip**: VisionZip performs spatial compression only, without addressing temporal redundancy, limiting its effectiveness in video scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The global unified spatiotemporal allocation approach is more elegant than two-stage methods, though the core technical components (attention-based selection + clustering merging) are relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Systematic evaluation across multiple backbones, benchmarks, and retention rates, with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear paper structure with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ High practical value for real-world Video-LLM deployment; retaining 90% performance at 2% token retention is compelling for production scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] StreamingTOM: Streaming Token Compression for Efficient Video Understanding](streamingtom_streaming_token_compression_video.md)
- [\[ICLR 2026\] FlashVID: Efficient Video Large Language Models via Training-free Tree-Based Spatiotemporal Token Merging](../../ICLR2026/video_understanding/flashvid_efficient_video_large_language_models_via_training-free_tree-based_spat.md)
- [\[CVPR 2026\] UTPTrack: Towards Simple and Unified Token Pruning for Visual Tracking](utptrack_towards_simple_and_unified_token_pruning_for_visual_tracking.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](../../ICLR2026/video_understanding/floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)

<!-- RELATED:END -->
