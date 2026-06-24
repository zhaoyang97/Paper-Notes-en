---
title: >-
  [Paper Note] Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Key-Frame Extraction
description: >-
  [ACL 2025][Video Understanding][Key-Frame Extraction] This paper proposes Attention-Seeker, an unsupervised method that dynamically analyzes the attention score distribution in self-attention layers of Transformer models to extract the most representative key-frames from videos without any supervision signals. It outperforms existing unsupervised methods on multiple video summarization benchmarks.
tags:
  - "ACL 2025"
  - "Video Understanding"
  - "Key-Frame Extraction"
  - "Self-Attention Mechanism"
  - "Dynamic Scoring"
  - "Unsupervised Method"
  - "Video Summarization"
date: 2026-05-08
content_hash: 329881cf4fad58f8
---

# Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Key-Frame Extraction

**Conference**: ACL 2025  
**Code**: None  
**Area**: Video Understanding / Unsupervised Learning  
**Keywords**: Key-Frame Extraction, Self-Attention Mechanism, Dynamic Scoring, Unsupervised Method, Video Summarization

## TL;DR
This paper proposes Attention-Seeker, an unsupervised method that dynamically analyzes the attention score distribution in self-attention layers of Transformer models to extract the most representative key-frames from videos without any supervision signals. It outperforms existing unsupervised methods on multiple video summarization benchmarks.

## Background & Motivation

**Background**: Key-frame extraction is a fundamental task in video understanding, aiming to select a small subset of frames that best represent the video content. Existing methods can be categorized into three groups: clustering-based methods (e.g., K-means clustering on frame features), change detection-based methods (detecting salient mutation points in visual content), and deep learning-based supervised methods (training models to directly predict key-frame likelihood). Recently, the success of Vision Transformers (ViTs) and video-language models has provided powerful feature representations for key-frame extraction.

**Limitations of Prior Work**: Supervised methods require large amounts of human-annotated key-frame data, which is expensive to collect and suffers from poor annotation consistency (since different annotators may define "key-frames" differently). Although unsupervised methods bypass the need for annotations, existing ones (e.g., clustering-based) typically only consider visual similarity and diversity while neglecting the semantic importance of frames. Moreover, conventional approaches first extract frame features and then perform selection, leaving feature extraction and frame selection as two decoupled steps.

**Key Challenge**: An ideal key-frame should be both representative (summarizing the video content) and informative (containing crucial semantic details). However, existing unsupervised methods struggle to determine the semantic importance of frames without supervision. Although the attention mechanism implicitly encodes the model's judgment on the importance of input parts, this information has not been fully exploited for key-frame extraction.

**Goal**: To design a key-frame scoring method without any supervision by leveraging the intrinsic structure of the self-attention mechanism in pretrained Transformer models.

**Key Insight**: The authors observe that when a pretrained video Transformer processes a video, certain frames consistently act as "attention centers" in the self-attention matrix (i.e., other frames frequently attend to them). These frames often correspond to the semantically most important key-frames.

**Core Idea**: To use the column sum of the self-attention matrix as a proxy indicator for frame importance, and design a dynamic layer selection mechanism to adaptively extract importance signals from the most relevant attention layers.

## Method

### Overall Architecture
The pipeline of Attention-Seeker is as follows: (1) The video frame sequence is fed into a pretrained video Transformer (e.g., TimeSFormer, VideoMAE) to obtain the self-attention matrices of all layers. (2) A dynamic layer selection module identifies which attention layers/heads best reflect the semantic importance of the frames. (3) Frame importance scores are extracted from the selected attention layers. (4) Key-frame selection is performed based on these scores (with temporal diversity constraints to avoid selecting overly clustered frames). The entire process requires no training or fine-tuning.

### Key Designs

1. **Column-wise Attention Scoring (CAS)**:

    - **Function**: Extracts the importance score of each frame from the attention matrix.
    - **Mechanism**: For a self-attention matrix $A \in \mathbb{R}^{T \times T}$ (where $T$ is the number of frames), $A_{ij}$ denotes the attention weight when frame $i$ attends to frame $j$. Summing each column yields the total degree to which frame $j$ is attended by all other frames: $s_j = \sum_{i=1}^{T} A_{ij}$. Intuitively, a frame heavily attended by many other frames holds high semantic "centrality". Furthermore, intra-layer normalization and temporal smoothing (Gaussian smoothing with an adaptive window size based on video length) are applied to CAS scores to eliminate noise in the attention matrix.
    - **Design Motivation**: The self-attention matrix intrinsically encodes the correlation and importance between tokens. Column-wise summation can be seen as a graph centrality measure (similar to PageRank), providing useful signals for frame importance without any extra learning.

2. **Dynamic Attention Layer Selection (DALS)**:

    - **Function**: Adaptively selects attention layers and heads that best reflect the semantic structure of the video.
    - **Mechanism**: Different attention layers/heads may encode information at different levels (e.g., shallow layers focus on low-level visual features, while deep layers focus on high-level semantics). DALS evaluates the quality of each attention layer by calculating the "structuredness" of its attention matrix using attention entropy: $H_l = -\sum_{i,j} A_{ij}^{(l)} \log A_{ij}^{(l)}$. A low attention entropy indicates highly concentrated attention, where the model has clear judgment on importance; high entropy suggests scattered attention, which is unsuitable for importance scoring. The CAS scores of the top-$k$ layers/heads with the lowest entropy are fused via weighted combination as the final frame importance score. The weights are inversely proportional to the attention entropy: $w_l \propto 1/H_l$.
    - **Design Motivation**: Directly average-pooling attention maps across all layers yields suboptimal results because many layers (especially shallow ones and specialized heads) exhibit attention patterns that do not contribute to semantic importance. Dynamic selection automatically filters out these noisy layers.

3. **Temporal Diversity Constrained Selection (TDCS)**:

    - **Function**: Selects a temporally dispersed, non-redundant subset from high-importance frames.
    - **Mechanism**: Simply selecting the highest-scoring frames often leads to temporal clustering (e.g., consecutive frames during high-action climaxes all getting high scores). TDCS employs a greedy selection strategy: in each step, the frame with the highest current score is added to the key-frame set, and then a score decay is applied to neighbors within its temporal vicinity (the decay factor decays exponentially with temporal distance). Mathematically, after selecting frame $f_t$, the score of a neighboring frame $f_s$ is updated as: $s'_{f_s} = s_{f_s} \cdot (1 - \exp(-|t-s|^2 / 2\sigma^2))$, where $\sigma$ controls the suppression range. This ensures a balanced distribution of key-frames across the timeline.
    - **Design Motivation**: A core requirement of key-frames is to cover different semantic segments of the video. Purely ranking by importance leads to temporal bias in frame selection.

### Loss & Training
The proposed method is a training-free solution, involving no loss functions or training processes. All computations are based on forward inference of a pretrained Transformer model, and key-frame extraction is completed back-to-back with a single forward pass to retrieve the attention matrices.

## Key Experimental Results

### Main Results

| Dataset | Metric | Attention-Seeker | K-Means | VSUMM | DR-DSN | CA-SUM |
|--------|------|-----------------|---------|-------|--------|--------|
| SumMe | F1 | 52.8 | 41.2 | 44.6 | 42.1 | 50.8 |
| TVSum | F1 | 61.3 | 50.5 | 53.8 | 57.6 | 59.2 |
| YouTube | Precision | 74.5 | 62.3 | 66.1 | 65.8 | 71.2 |
| OVP | F1 | 68.2 | 55.4 | 59.7 | 61.3 | 65.8 |

### Ablation Study

| Configuration | SumMe F1 | TVSum F1 | Note |
|------|---------|---------|------|
| Full (CAS+DALS+TDCS) | 52.8 | 61.3 | Full method |
| w/o DALS (Average of all layers) | 48.3 | 56.7 | Remove dynamic layer selection, -4.5/-4.6 |
| w/o TDCS (Pure top-k) | 49.1 | 57.8 | Remove diversity constraint, -3.7/-3.5 |
| Row sum instead of column sum | 45.6 | 52.1 | Use row sum as importance, -7.2/-9.2 |
| Last layer only | 47.8 | 55.3 | Use only the last layer's attention, -5.0/-6.0 |
| Random layer selection | 44.2 | 51.8 | Random layer selection as baseline |

### Key Findings
- Dynamic Layer Selection (DALS) and Temporal Diversity Constrained Selection (TDCS) each contribute to an improvement of about 4 F1 points, demonstrating that both are indispensable components.
- Column-wise summation significantly outperforms row-wise summation (a gap of 7-9 F1 points), validating that "how much a frame is attended to" is a much better proxy for importance than "how much a frame attends to others".
- On TimeSFormer, the attention from intermediate layers (Layers 6-9) is more suitable for key-frame extraction than that of the shallowest and deepest layers, aligning with the finding in ViTs that "intermediate layers are the most discriminative".
- The advantage is more pronounced on long videos (>5 minutes) compared to short videos, as the global receptive field of self-attention in longer videos helps identify crucial content across different time periods.

## Highlights & Insights
- The core insight is extremely simple yet elegant: "frames that are heavily attended to by other frames are the important ones." While intuitive, this idea has not been systematically validated and exploited in prior work. This approach of mining unsupervised signals from attention matrices can be transferred to other modalities (e.g., extracting key sentences from the attention maps of text Transformers).
- Dynamic Layer Selection automatically identifies "informative" attention layers using attention entropy. This serves as a training-free layer selection method that could be applied to attention visualization, model interpretability, and other scenarios.
- The entire method is training-free and requires only a single forward pass, making it extremely cost-effective to deploy in real-world applications.

## Limitations & Future Work
- The method depends heavily on the quality of the pretrained video Transformer. If the attention distribution of the pretrained model is flawed, the extracted key-frames will also be inaccurate.
- The current approach assumes a positive correlation between key-frames and attention concentration. However, in certain scenarios (such as anomaly/emergency detection), key-frames might instead be "unusual" frames with highly scattered attention.
- The method has not been compared with attention patterns in the latest video-language models (e.g., Video-LLaVA).
- Future work could extend Attention-Seeker to video summarization (not just frame selection, but summary generation) and video question answering (using key-frames as evidence frames).

## Related Work & Insights
- **vs DINO self-attention visualization**: The DINO series demonstrates that ViT self-attention contains semantic segmentation information. Attention-Seeker extends a similar idea from the spatial dimension to the temporal dimension.
- **vs CA-SUM (Apostolidis et al.)**: CA-SUM uses supervised attention training to learn frame importance. Attention-Seeker demonstrates that pretrained attention is already sufficient without additional supervision.
- **vs DR-DSN**: DR-DSN utilizes adversarial learning for unsupervised video summarization, which involves complex modeling. Attention-Seeker provides a simpler and more efficient alternative.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing self-attention column-wise sum as a proxy for frame importance is an elegant insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation, detailed ablation studies, and convincing comparisons between row and column sums.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and intuitive explanation of insights.
- Value: ⭐⭐⭐⭐ Highly practical as a training-free unsupervised method, with highly transferable concepts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](../../CVPR2025/video_understanding/seal_semantic_attention_learning_for_long_video_representation.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](../../CVPR2025/video_understanding/learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)

</div>

<!-- RELATED:END -->
