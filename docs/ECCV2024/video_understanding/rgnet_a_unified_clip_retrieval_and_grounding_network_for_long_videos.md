---
title: >-
  [Paper Note] RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos
description: >-
  [ECCV 2024][Video Understanding][long video temporal grounding] RGNet is proposed to deeply unify the two stages of long video temporal grounding—clip retrieval and temporal localization—into a single network. Through the sparse attention of the RG-Encoder and contrastive clip sampling, end-to-end optimization is achieved, yielding SOTA performance on MAD and Ego4D.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "long video temporal grounding"
  - "clip retrieval"
  - "unified network"
  - "sparse attention"
  - "moment localization"
date: 2026-05-08
content_hash: 02d9b72f5c36dd15
---

# RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos

**Conference**: ECCV 2024  
**arXiv**: [2312.06729](https://arxiv.org/abs/2312.06729)  
**Code**: [https://github.com/Tanveer81/RGNet](https://github.com/Tanveer81/RGNet)  
**Area**: Video Understanding / Temporal Grounding  
**Keywords**: long video temporal grounding, clip retrieval, unified network, sparse attention, moment localization

## TL;DR
RGNet is proposed to deeply unify the two stages of long video temporal grounding—clip retrieval and temporal localization—into a single network. Through the sparse attention of the RG-Encoder and contrastive clip sampling, end-to-end optimization is achieved, yielding SOTA performance on MAD and Ego4D.

## Background & Motivation
**Background**: Long Video Temporal Grounding (LVTG) requires localizing a specific moment (usually only a few seconds) from a 20-120 minute video based on a textual query, like looking for a needle in a haystack. Existing methods follow a two-stage paradigm: first retrieving relevant clips, then performing precise localization within those clips.

**Limitations of Prior Work**: The two stages are disconnected—the retrieval module typically adopts text-video retrieval techniques (e.g., CLIP) which only require understanding the high-level theme of the video, lacking fine-grained event understanding. If the retrieval fails, the localization network cannot recover from the error.

**Key Challenge**: Retrieval in LVTG requires fine-grained event understanding (e.g., "find the moment where the mother is hanging laundry outside the farmhouse"), but general video retrieval models are designed for coarse-grained theme matching (e.g., "a movie x about a farm family").

**Goal**: To unify clip retrieval and temporal localization so that the retrieval module acquires fine-grained event understanding capabilities.

**Key Insight**: Designing a unified Transformer encoder to model both the clip level and frame level simultaneously, allowing retrieval to directly benefit from localization targets through sparse attention and end-to-end optimization.

**Core Idea**: Utilizing the RG-Encoder to unify retrieval and localization with shared features for mutual optimization. The retrieval module directly learns fine-grained event understanding through localization annotations.

## Method

### Overall Architecture
The input long video is segmented into multiple equal-length clips using a sliding window. The RG-Encoder processes all clips and the text query, outputting the features of the most relevant retrieved clips. The localization decoder then predicts the precise temporal boundaries $(\tau_c, \tau_w)$ within the retrieved clips. The entire network is trained end-to-end.

### Key Designs

1. **RG-Encoder (Unified Retrieval-Localization Encoder)**

    - **Function**: Unifies retrieval and localization into a single encoder, simultaneously modeling at both clip-level (context) and frame-level (content).
    - **Core components consist of four parts**:
        - **Cross Attention**: Frame features act as queries, and text features act as keys/values, generating text-conditioned frame features: $F^i = \text{softmax}(Q^i K^T) V + Q^i$
        - **Sparsifier**: Uses Gumbel-softmax to compute the relevance of each frame $G^j \in [0,1]$, classifying frames as relevant/irrelevant, and generating an attention mask $M(j,k) = \begin{cases} 0 & \text{if } G^j > 0.5 \text{ or } j=k \\ -\infty & \text{otherwise} \end{cases}$
        - **Retrieval Attention**: Introduces a learnable retrieval token $R^i$ concatenated with frame features for self-attention, aggregated based on the sparse mask: $\tilde{Q}^i = \text{softmax}(Q^i K^T + M) V + Q^i$. Outputs clip-level context $R^i$ and frame-level content $F_c^{i,j}$
        - **Feature Fusion**: Retrieved clip feature = content + context $\times$ relevance: $P^{i,j} = F_c^{i,j} + R^i \times G^j$
    - **Design Motivation**: Sparse attention forces the retrieval to focus on event-relevant frames rather than the entire clip, leading to better synergy with the localization task.

2. **Contrastive Clip Sampling**

    - **Function**: Mimics the clip retrieval scenario in long videos during training.
    - **Mechanism**: Multiple clips from the same video form a batch, and an InfoNCE loss is utilized to contrast the positive clip against negative clips:
    $$\mathcal{L}_{\text{cont}} = -\sum_i \log \frac{\exp(l_{\text{cont}}(R^{i,i}))}{\sum_j \exp(l_{\text{cont}}(R^{i,j}))}$$
    - A large batch of negative samples (different clips from the same scene) simulates the real-world inference challenge of retrieving from a long video.
    - **Design Motivation**: The gap between the training and testing phases is a key issue in LVTG. Previous methods only observe a few clips during training, while needing to retrieve from hundreds of clips during testing.

3. **Intra-Clip Attention Loss**

    - **Function**: Guides the sparsifier to differentiate between frames inside and outside the temporal boundary.
    - **Mechanism**: A margin-based ranking loss requiring frames belonging to the ground truth moment to have higher relevance scores:
    $$\mathcal{L}_{\text{attn}} = \max(0, \Delta + S_c(i, j_{\text{out}}) - S_c(i, j_{\text{in}}))$$
    - Where $S_c(i,j) = R^i_{\text{proj}} \cdot P^{i,j}_{\text{proj}}$, $\Delta=0.2$
    - **Design Motivation**: Directs retrieval attention to focus on frames aligned with the query event.

### Loss & Training
- Total loss: $\mathcal{L}_{\text{total}} = \lambda_{\text{attn}} \mathcal{L}_{\text{attn}} + \lambda_{\text{cont}} \mathcal{L}_{\text{cont}} + \mathcal{L}_g$
- Localization loss $\mathcal{L}_g$: L1 + gIoU + CE (Hungarian algorithm matching for predicted and GT boundaries).
- Hyperparameters: $\lambda_{L1}=10, \lambda_{\text{gIoU}}=1, \lambda_{\text{CE}}=4, \lambda_{\text{cont}}=10$
- Trained for 35 epochs on MAD, and 200 epochs on Ego4D.
- Features are extracted using frozen CLIP and EgoVLP.
- Clip length: 180s for MAD, 48s for Ego4D.

## Key Experimental Results

### Main Results

| Dataset | Metric | RGNet | CONE | SOONet | Gain |
|--------|------|-------|------|--------|------|
| Ego4D-NLQ | R1@0.3 | **20.63** | 14.15 | 8.00 | +6.48 |
| Ego4D-NLQ | R5@0.3 | **41.67** | 30.33 | 22.40 | +11.34 |
| Ego4D-NLQ | Avg | **24.96** | 17.67 | 11.31 | +7.29 |
| MAD | R1@0.1 | **12.43** | 8.90 | 11.26 | +1.17 |
| MAD | R5@0.1 | **25.12** | 20.51 | 23.21 | +1.91 |
| MAD | Avg | **13.70** | 11.01 | 13.59 | +0.11 |

### Disconnected vs. Unified Architecture Comparison

| Dataset | Stage | Disconnected Baseline | RGNet | Gain |
|--------|------|----------|-------|------|
| Ego4D | Retrieval R@1 | 31.71 | **42.08** | +10.37 |
| Ego4D | Retrieval R@5 | 64.63 | **76.28** | +11.65 |
| Ego4D | Localization R1@0.3 | 29.84 | **36.53** | +6.69 |
| MAD | Retrieval R@1 | 12.41 | **25.01** | +12.60 |
| MAD | Retrieval R@5 | 24.50 | **50.02** | +25.52 |
| MAD | Localization R1@0.3 | 29.49 | **33.42** | +3.93 |

### Ablation Study

| Configuration | R1@0.3 | R5@0.3 | Explanation |
|------|--------|--------|------|
| RGNet (Full) | 18.28 | 34.02 | Default (w/o NaQ) |
| w/o Retrieval Token | 17.80 | 33.99 | -0.48, clip context modeling is important |
| w/o Sparsifier | 16.12 | 31.57 | -2.16, frame-level filtering is crucial |
| w/o RG-Encoder (Disconnected Baseline) | 14.15 | 30.33 | -4.13, the advantage of the unified architecture is clear |
| w/o Contrastive Loss | 17.41 | 32.12 | -0.87, negative sample simulation is effective |
| w/o Attention Loss | 16.21 | 31.59 | -2.07, frame-level event differentiation is key |

### Key Findings
- The retrieval stage is the performance bottleneck in LVTG: oracle localization R1@0.3 = 36.53 is far higher than LVTG's 20.63, indicating that the performance gap stems from retrieval errors.
- The unified architecture increases retrieval R@1 by 10.4% (Ego4D) and 12.6% (MAD), far exceeding expectations.
- Even when retrieving only 1 clip, RGNet outperforms the best results of the baseline.
- Sparse attention and attention loss contribute the most, with performance dropping 2.16 and 2.07 validation points respectively when removed.

## Highlights & Insights
- **Thorough Problem Analysis**: By systematically dissecting the two stages, experiments demonstrate that retrieval (rather than localization) is the bottleneck, providing a clear motivation for the unified architecture. The design of the oracle experiment is clean and powerful.
- **Elegant Unified Architecture**: The separation and fusion of clip-level context features and frame-level content features ($P = F_c + R \times G$) naturally allow the same encoder to serve both retrieval (requires clip representation) and localization (requires frame representation).
- **Doubled Retrieval R@1** (MAD: 12.41 $\rightarrow$ 25.01) demonstrates that end-to-end training enables the retrieval module to acquire fine-grained event understanding that detached methods could not learn.

## Limitations & Future Work
- Still relies on pre-trained image encoders to extract visual features (CLIP/EgoVLP), without end-to-end training of the visual encoder.
- The performance gain on MAD (Avg +0.11) is significantly smaller than on Ego4D (Avg +7.29), possibly because visual features of movie videos are harder to distinguish.
- Fixed clip lengths (180s/48s) might not adapt to target moments of varying lengths.
- The decoder still requires multiple queries and Hungarian matching, which may lack precision for ultra-short moments.

## Related Work & Insights
- **vs CONE**: CONE is also a proposal-based method, but retrieval and localization are disconnected. RGNet improves Avg by 7.29% on Ego4D, proving that unification excels over disconnection.
- **vs Moment-DETR**: Moment-DETR is designed for short videos. Directly applying it to long videos results in poor performance because target moments are relatively microscopic; RGNet addresses this through multi-granularity modeling.
- **vs X-Pool/TS2-Net**: Video retrieval methods target high-level themes, whereas RGNet demonstrates that clip retrieval requires much finer-grained event understanding.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of unifying the two stages is clear, and the RG-Encoder is reasonably designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive two-stage analysis, oracle experiments, ablation studies, and qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorously derived motivation, clear figures, and deep experimental analysis.
- Value: ⭐⭐⭐⭐ A significant step forward in long video temporal grounding, with a 46% improvement in Ego4D R1@0.3.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AMEGO: Active Memory from Long EGOcentric Videos](amego_active_memory_from_long_egocentric_videos.md)
- [\[CVPR 2025\] DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](../../CVPR2025/video_understanding/decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)
- [\[ECCV 2024\] Goldfish: Vision-Language Understanding of Arbitrarily Long Videos](goldfish_vision-language_understanding_of_arbitrarily_long_videos.md)
- [\[ECCV 2024\] SAFNet: Selective Alignment Fusion Network for Efficient HDR Imaging](safnet_selective_alignment_fusion_network_for_efficient_hdr_imaging.md)
- [\[CVPR 2025\] Object-Shot Enhanced Grounding Network for Egocentric Video](../../CVPR2025/video_understanding/object-shot_enhanced_grounding_network_for_egocentric_video.md)

</div>

<!-- RELATED:END -->
