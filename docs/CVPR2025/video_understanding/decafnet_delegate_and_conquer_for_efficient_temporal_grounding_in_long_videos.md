---
title: >-
  [Paper Note] DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos
description: >-
  [CVPR 2025][Video Understanding][Long Video Temporal Grounding] DeCafNet is proposed, which outperforms all prior methods on long video temporal grounding tasks with a **47% reduction in TFLOPs**, utilizing a **delegate-and-conquer dual-encoder strategy** (where a lightweight sidekick encoder extracts dense features and generates saliency maps, while an expert encoder only processes the top-c% key clips) combined with **DeCaf-Grounder** to unify features across different temp…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Long Video Temporal Grounding"
  - "Efficient Inference"
  - "Dual Encoder"
  - "Saliency Selection"
  - "Multi-Scale Temporal Refinement"
date: 2026-05-08
content_hash: 30122edfa6d17198
---

# DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos

**Conference**: CVPR 2025  
**arXiv**: [2505.16376](https://arxiv.org/abs/2505.16376)  
**Code**: [https://github.com/ZijiaLewisLu/CVPR2025-DeCafNet](https://github.com/ZijiaLewisLu/CVPR2025-DeCafNet)  
**Area**: Video Understanding  
**Keywords**: Long Video Temporal Grounding, Efficient Inference, Dual Encoder, Saliency Selection, Multi-Scale Temporal Refinement

## TL;DR

DeCafNet is proposed, which outperforms all prior methods on long video temporal grounding tasks with a **47% reduction in TFLOPs**, utilizing a **delegate-and-conquer dual-encoder strategy** (where a lightweight sidekick encoder extracts dense features and generates saliency maps, while an expert encoder only processes the top-c% key clips) combined with **DeCaf-Grounder** to unify features across different temporal resolutions.

## Background & Motivation

Long Video Temporal Grounding (LVTG) aims to localize temporal segments corresponding to user text queries from long videos ranging from minutes to hours. It is applied in scenarios such as video summarization, content recommendation, and surveillance.

**Limitations of Prior Work**: Current SOTA methods (e.g., RGNet, SnAG) follow a two-stage paradigm of "clipping $\to$ passing each clip through an expert encoder $\to$ grounding". While this approach is feasible for short videos, the number of clips in long videos is massive (up to hundreds). Processing each clip sequentially with a large-scale pretrained expert encoder leads to an **explosion in computation**—requiring up to 668 TFLOPs, 224GB GPU memory, and 17 seconds of inference time.

**Key Challenge**: In long videos, query-relevant segments typically account for only a tiny fraction of the entire video (an average of only 1.7% in Ego4D-NLQ). A large number of clips are irrelevant to the query but consume the same amount of computational resources.

**Key Insight**: Inspired by "computational heterogeneity"—that not all temporal positions are equally complex or important—this work proposes a delegate-and-conquer strategy: delegating most of the computation to an efficient sidekick encoder, and using the expert encoder only for the most salient clips to perform fine-grained processing, thereby overcoming the computational bottleneck.

## Method

### Overall Architecture

DeCafNet consists of three components: (1) **Sidekick Encoder** $\Psi_D$: efficiently extracts dense features for all clips and generates a saliency map; (2) **Expert Encoder** $\Psi_E$: processes only the top-c% key clips selected by the saliency map; (3) **DeCaf-Grounder**: unifies the differing resolution features from both encoders through query-aware temporal aggregation and multi-scale temporal refinement to predict the temporal intervals.

### Key Designs

1. **Sidekick Encoder (Lightweight Sidekick Encoder)**
    - **Function**: Extracts dense features for all clips and generates a query-related saliency map at an extremely low computational cost.
    - **Mechanism**: Based on the ViT architecture, spatiotemporal convolutional pooling layers are inserted before the transformer blocks, reducing the temporal dimension $L$ and spatial dimensions $(H, W)$ by 4 times respectively, which significantly decreases the computation of subsequent blocks. Meanwhile, **temporal interpolation** is utilized—sampling once every $\tau=2$ clips, and the unsampled clips are interpolated from adjacent clip features via an FFN.
    - **Design Motivation**: Adjacent clips in long videos are highly similar and can be inferred from neighboring clips without computing from scratch. Combining this with convolutional pooling achieves a 31x reduction in TFLOPs.
    - **Loss & Training**: Employs **saliency loss** (contrastive learning to align video-text features) and **distillation loss** ($L_2$ loss to distill expert encoder features) to ensure the quality of the lightweight encoder's features.

2. **Saliency Selection Mechanism**
    - **Function**: Dynamically filters the most relevant clips based on the query text, reducing the workload of the expert encoder.
    - **Mechanism**: Computes the inner product between the dense features $\mathbf{F}_D$ from the sidekick encoder and the text query CLS token $\mathbf{q}_{cls}$ to obtain the saliency score $\mathbf{S}=\mathbf{F}_D \cdot \mathbf{q}_{cls}$, then selects the top-c% clips to feed into the expert encoder.
    - **Design Motivation**: To balance information retention and computational efficiency—since the sidekick encoder inevitably loses information due to pooling, the expert encoder is still required to provide high-quality, fine-grained features for key clips.

3. **DeCaf-Grounder (Unified Refinement Module)**
    - **Function**: Unifies features from both encoders with different temporal resolutions to perform precise temporal grounding.
    - **Mechanism**:
     - **Query-aware temporal aggregation**: Performs zero-padding on non-salient clips in $\mathbf{F}_S$ to align the temporal dimensions, concatenates $\mathbf{F}_D$, $\hat{\mathbf{F}}_S$, and the index-aligned saliency score $\mathbf{S}$, and then enhances query-relevant information via video-text cross-attention.
     - **Multi-scale temporal refinement**: Constructs an $L=8$ level feature pyramid (with temporal length halved at each level) using a temporal transformer, generates confidence scores for each scale using an FFN classifier, and synchronizes localization information across scales using dilated temporal convolutions.
    - **Design Motivation**: The features of the two encoders exhibit different temporal resolutions and semantic granularities. Standard grounding modules perform poorly when directly applied, necessitating a specialized aggregation-refinement architecture.
    - **Loss & Training**: Utilizes Focal loss + Distance-IoU loss to train the classification and regression heads.

## Key Experimental Results

### Main Results

| Dataset | Method | R1@0.3 | R1@0.5 | R5@0.3 | R5@0.5 | AVG |
|---------|------|--------|--------|--------|--------|-----|
| Ego4D-NLQ | SnAG | 15.87 | 11.26 | 38.26 | 27.16 | 23.14 |
| | RGNet | 18.28 | 12.04 | 34.02 | 22.89 | 21.81 |
| | **DeCafNet-50%** | **18.10** | **12.55** | **38.85** | **28.27** | **24.44** |
| Ego4D-Goalstep | SnAG | 18.34 | 15.12 | 45.95 | 38.55 | 29.49 |
| | **DeCafNet-50%** | **21.29** | **17.46** | **47.27** | **40.40** | **31.61** |
| MAD | SnAG | 10.28 | 8.46 | 24.42 | 20.60 | 13.84 |
| | **DeCafNet** | **13.25** | **10.96** | **27.73** | **23.68** | **16.47** |

### Comparison of Computational Efficiency

| Encoder Configuration | TFLOPs | GPU Memory (G) | Inference Time (s) |
|-----------|--------|-----------|------------|
| $\Psi_E$ only (Prior methods) | 668.2 | 224.2 | 17.1 |
| DeCafNet-50% | 355.7 (**↓47%**) | 126.2 (**↓44%**) | 8.4 (**↓51%**) |
| DeCafNet-30% | 222.1 (**↓66%**) | 79.9 (**↓65%**) | 5.7 (**↓67%**) |
| $\Psi_D$ only | 21.6 | 10.9 | 0.6 |

### Ablation Study

- Using only $\mathbf{F}_D$: AVG=21.51; using only $\mathbf{F}_S$: AVG=22.57; combining both: AVG=23.91; adding the saliency score $\mathbf{S}$: **AVG=24.41**, indicating that the three features are complementary.
- The sidekick encoder outperforms random and uniform selection by an average of 1.8%+ in saliency selection.
- DeCaf-Grounder also outperforms SnAG on short video datasets: Charades-STA by +1.37%, TACoS by +0.81%.

### Key Findings

- Query-associated segments in long videos are extremely sparse (only 1.7% in Ego4D-NLQ), justifying aggressive clip filtering.
- The sidekick encoder achieves a 31x reduction in TFLOPs and a 22x reduction in GPU memory.
- Processing only 30% of the clips matches the previous SOTA, while processing 50% entirely outperforms it.

## Highlights & Insights

- The **delegate-and-conquer philosophy** is highly versatile: delegating expensive computations to a lightweight proxy and investing heavy resources only at critical locations can be generalized to other long-sequence processing scenarios.
- Saliency selection is **query-dependent**, selecting different clips for different queries, which is much more flexible than fixed sampling strategies.
- DeCaf-Grounder still significantly outperforms prior methods when using only expert encoder features (on the MAD dataset), demonstrating that its multi-scale refinement architecture holds independent value.

## Limitations & Future Work

- The sidekick encoder requires separate training (two losses + distillation), which increases training complexity.
- The saliency selection ratio c% is a fixed hyperparameter and cannot adjust adaptively based on video content.
- Currently verified only on egocentric videos (Ego4D) and movies (MAD); more diverse long video scenarios remain to be explored.

## Related Work & Insights

- Methods for **Short Video Temporal Grounding** (SVTG) (e.g., Moment-DETR which directly predicts timestamps) cannot scale to long video scenarios.
- The coarse-to-fine strategy of **CONE** and the late fusion of **SnAG** establish the foundation for LVTG, but both ignore the computational overhead of feature extraction.
- Inspiration from the delegate-and-conquer concept of this paper: In the field of efficient inference, it is not always necessary to compress individual models; instead, one can **design dynamic computational resource allocation strategies**.

## Rating

⭐⭐⭐⭐ — The proposed method is simple yet effective, achieving a new Pareto optimum in the efficiency-performance trade-off. The dual-encoder + saliency selection framework possesses great generalizability and practical engineering value, comprehensively outperforming SOTA on three LVTG benchmarks while significantly reducing computation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Number it: Temporal Grounding Videos like Flipping Manga](number_it_temporal_grounding_videos_like_flipping_manga.md)
- [\[CVPR 2025\] VideoGEM: Training-Free Action Grounding in Videos](videogem_training-free_action_grounding_in_videos.md)
- [\[ECCV 2024\] RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](../../ECCV2024/video_understanding/rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [\[ICCV 2025\] Sparse-Dense Side-Tuner for Efficient Video Temporal Grounding](../../ICCV2025/video_understanding/sparse-dense_side-tuner_for_efficient_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
