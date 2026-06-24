---
title: >-
  [Paper Note] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders
description: >-
  [ECCV 2024][Self-Supervised Learning][Masked Autoencoder] ViC-MAE unifies contrastive learning and masked autoencoders into a single framework. By treating short video clips as augmented views (rather than duplicating images into video frames), it achieves excellent performance on both image and video downstream tasks—reaching 87.1% top-1 on ImageNet-1K (+2.4% over OmniMAE) and 75.9% on SSv2.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Masked Autoencoder"
  - "Contrastive Learning"
  - "Video Representation"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 23f729c4dd8bda25
---

# ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders

**Conference**: ECCV 2024  
**arXiv**: [2303.12001](https://arxiv.org/abs/2303.12001)  
**Code**: [https://github.com/jeffhernandez1995/ViC-MAE](https://github.com/jeffhernandez1995/ViC-MAE)  
**Area**: Self-Supervised Learning / Video-Image Representation Learning  
**Keywords**: Self-supervised Learning, Masked Autoencoder, Contrastive Learning, Video Representation, Vision Transformer

## TL;DR

ViC-MAE unifies contrastive learning and masked autoencoders into a single framework. By treating short video clips as augmented views (rather than duplicating images into video frames), it achieves excellent performance on both image and video downstream tasks—reaching 87.1% top-1 on ImageNet-1K (+2.4% over OmniMAE) and 75.9% on SSv2.

## Background & Motivation

There are two major paradigms in self-supervised visual representation learning:

**Joint Embedding Methods** (Contrastive Learning): Learn invariance to specific transformations, excelling in global semantic representation but lacking in local feature learning.

**Masked Image Modeling** (MAE): Learn local features by reconstructing masked regions, but yield lower-quality global representations (only 68% in linear probing).

Limitations of Prior Work:
- **Difficulty in Video-to-Image Transfer**: Models pre-trained on video perform poorly on image tasks, and vice versa.
- **Imbalance in Joint Image-Video Training**: Methods such as OmniMAE treat images as videos (by duplicating frames), which is computationally inefficient and underutilizes semantic information.
- **Non-Trivial Combination of Contrastive Learning and MAE**: Plain combinations lead to representation collapse (e.g., failed attempts using [CLS] tokens + VicReg/SimSiam).

Key Insight: **Natural variations between video frames (pose, viewpoint, deformation) provide strong augmentation signals that standard image augmentations cannot simulate. Treating short videos as temporal augmentations of the same view, rather than pretending images are videos, offers a more efficient multimodal learning mechanism.**

## Method

### Overall Architecture

ViC-MAE employs a Siamese ViT encoder architecture (with shared weights). For two inputs (either two frames from the same video or two augmented views of the same image), it performs:
1. **Masked Image Modeling (Independent per frame)**: Learns local features.
2. **Contrastive Learning (Cross-frame/Cross-view)**: Aggregates local features into a global representation via global pooling, then aligns them using the InfoNCE loss.

### Key Designs

1. **Temporal Augmentation Instead of Image Duplication**:

    - Given a video $\{I_1, ..., I_T\}$, "Distant Sampling" is adopted: the video is partitioned into $n$ equal segments, and one frame is randomly sampled from each segment.
    - The temporal distance between the two frames is approximately 1.06 seconds, providing a sufficiently strong "natural augmentation".
    - For image data, standard dual-view augmentations (cropping, flipping, color jitter, and Gaussian blur) are applied.
    - Video and image data are mixed and trained together within the same batch.

2. **Masked Reconstruction Branch (Local Features)**:

    - Random masks $M_i, M_j$ (75% masking ratio) are generated for the two frames respectively, yielding visible tokens $X_i^{(v)}, X_j^{(v)}$.
    - The ViT encoder computes representations for the visible tokens, and the decoder reconstructs the original patches after appending mask tokens.
    - Reconstruction loss: $\mathcal{L}_i^{MASK} = \|{\hat{I}_i - I_i}\|_2^2$ (computed only on masked positions).

3. **Contrastive Learning Branch (Global Features)**:

    - **Global Pooling Layer $\Omega$**: Average pooling is applied to the local token features output by the encoder to obtain the global representation.
    - Predictor $\mathcal{P}$ and target encoder $\mathcal{T}$ (symmetric design, Linear→BN→ReLU×2).
    - InfoNCE loss: $\mathcal{L}^{NEG}_{p_i, z_j} = -\log \frac{\exp(p_i \cdot z_j / \tau)}{\sum_{k=1}^{2N} \mathbb{1}[p_i \neq z_k] \exp(p_i \cdot z_k / \tau)}$
    - Sources of negative samples: frames from other videos, temporally distant frames from the same video, and other images in the batch.
    - **Ingenious Pooling Design to Avoid Representation Collapse**: The pooled features are simultaneously constrained by the MAE reconstruction loss (preventing them from becoming zero vectors), thereby eliminating the need for an extra stop-gradient operation or entropy regularization.

4. **Loss Combination and Scheduling**:

    - Total loss: $\mathcal{L} = \mathcal{L}^{MASK} + \lambda \mathcal{L}^{NEG}$
    - A progressive introduction schedule is utilized: the MAE loss dominates early in training to allow the model to learn local features first, and the weight of the contrastive loss is gradually increased later.

### Loss & Training

- Architecture: ViT-B/16 and ViT-L/16, with a small MAE decoder.
- Pre-training data: Kinetics-400 (~300K videos) + ImageNet-1K (~1.2M images); the largest scale version uses K710 + MiT + IN1K.
- Optimizer: AdamW, batch size 512.
- During video fine-tuning: The spatial tokenizer is duplicated and scaled along the temporal dimension to initialize the temporal tokenizer (standard 16 frames, skip 4).
- Data augmentation: Videos use spatial augmentation only (cropping + flipping, scale [0.5,1]), while images use strong augmentation (+ color jittering + Gaussian blur).

## Key Experimental Results

### Main Results

Transfer learning results of ViT-L/16 on multiple benchmarks:

| Method | Pre-training Data | IN1K↑ | K400↑ | Places-365↑ | SSv2↑ |
|------|-----------|:---:|:---:|:---:|:---:|
| MAE | IN1K | 85.5 | 82.3 | 59.4 | 57.7 |
| ST-MAE | K400 | 81.7 | 84.8 | 58.1 | 73.2 |
| OmniMAE | K400+IN1K | 84.7 | 84.0 | 59.4 | 73.4 |
| **ViC-MAE** | K400+IN1K | **86.0** | **86.8** | **60.0** | **75.0** |
| **ViC-MAE** | K710+MiT+IN1K | **87.1** | **87.8** | **60.7** | **75.9** |

### Ablation Study

Linear probing of ViT-B/16 on ImageNet-1K:

| Ablation Dimension | Configuration | Top-1 | Description |
|---------|------|:---:|------|
| Frame Distance | 0 (same frame) | 63.25 | No temporal augmentation |
| Frame Distance | 4 | 65.25 | Consecutive sampling |
| Frame Distance | D (distant sampling) | **67.66** | Strongest augmentation |
| Pooling Type | GeM | 66.92 | Generalized mean |
| Pooling Type | max | 67.01 | Max pooling |
| Pooling Type | **mean** | **67.66** | Mean pooling is optimal |
| Augmentation Strategy | Color only | 65.40 | Insufficient |
| Augmentation Strategy | Spatial only | 66.03 | Spatial is more important |
| Augmentation Strategy | **Color + Spatial** | **67.66** | Strong augmentation needed for images |

### Key Findings

1. **Distant Sampling is Key**: As the frame distance increases from 0 to D (approx. 1.06s), linear probing performance improves from 63.25% to 67.66% (+4.4%), demonstrating that natural temporal changes in video provide a very strong augmentation signal.
2. **Outperforming OmniMAE Across the Board**: Under the same data (K400+IN1K) and the same architecture (ViT-L), ViC-MAE outperforms OmniMAE on all four benchmarks (IN1K +1.3%, K400 +2.8%, Places +0.6%, SSv2 +1.6%).
3. **SOTA Video-to-Image Transfer**: ViC-MAE trained solely on video (MiT) achieves 85.3% on IN1K, representing the best image transfer result for self-supervised video pre-training.
4. **Contrastive Learning Outperforms Mask-only**: In the image-to-video ratio ablation, ViC-MAE consistently outperforms OmniMAE, indicating that the combination of contrastive and mask-based learning is superior to pure mask pre-training.
5. **COCO Object Detection**: ViC-MAE (IN1K+K710+MiT) achieves $AP^{Box}=53.2$ and $AP^{Mask}=46.9$, both of which are optimal among comparable methods.

## Highlights & Insights

1. **Dual Role of the Pooling Layer**: Global average pooling serves as both a necessary aggregation step for contrastive learning and a natural mechanism to avoid representation collapse (as the pooled features are still constrained by MAE reconstruction). This represents an elegant architectural design.
2. **Video as Augmentation**: Treating the variation between video frames as natural data augmentation (rather than disguising images as static videos) is conceptually simple and highly efficient, reducing the number of training tokens (98 vs. 157 for OmniMAE).
3. **Impressive Data Efficiency**: Achieving accuracy comparable to CAN (which utilizes 300M data points from JFT-300M) with only about 4.25M data points cleanly demonstrates that the information density of video data far exceeds that of images.
4. **Importance of Negative Samples**: Experiments show that using negative samples (InfoNCE) outperforms negative-free methods (VicReg, SimSiam), consistent with other successful video-to-image transfer methods.

## Limitations & Future Work

1. **Gap with Pure Video Methods**: On K400, ViC-MAE falls behind VideoMAEV2 by 0.8% (ViT-L) and TubeViT by 2.4% (ViT-L), likely because ViC-MAE does not explicitly model temporal tokens.
2. **Gap with DINOv2**: On IN1K, it falls behind DINOv2 by 1.2% (ViT-L), which leverages substantially more data and advanced training recipe scaling.
3. **Temporal Tokenizer Initialization during Fine-tuning**: Initializing the temporal dimension by duplicating spatial tokens is somewhat simplistic; more sophisticated initialization strategies could be explored.
4. **Unutilized Modalities**: Future iterations could incorporate text supervision (e.g., UMT) or audio signals for further performance improvement.
5. **Relatively Fixed Frame Sampling Strategy**: Although distant sampling yields strong results, it might overlook intermediate-scale temporal motion information.

## Related Work & Insights

- **Core Difference from OmniMAE**: ViC-MAE treats videos as augmented views and incorporates a contrastive loss, whereas OmniMAE treats images as videos and relies purely on masked modeling. The former is more efficient and exhibits superior performance.
- **Difference from C-MAE**: C-MAE employs the [CLS] token for contrastive learning, whereas ViC-MAE leverages pooled features to circumvent the representation collapse problem.
- **Insight**: The local features of MAE and the global semantics of contrastive learning are complementary. **Bridging the two through a pooling layer** serves as a general design paradigm that can be generalized to self-supervised learning in other modalities (such as point clouds and audio).

## Rating

- **Novelty**: ⭐⭐⭐⭐ Although the combination of contrastive learning and MAE is not entirely new, the contribution is distinct through the collapse-avoidance pooling design, the perspective of video as augmentation, and the progressive loss scheduling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers four major benchmarks, eight transfer datasets, COCO detection and segmentation, and rich ablations (frame distance, pooling style, augmentation type, data volume, and image-to-video ratio).
- **Writing Quality**: ⭐⭐⭐⭐ The methodology section is clear and the experimental section provides a solid scaling analysis (Figure 3), though some mathematical notations are slightly redundant.
- **Value**: ⭐⭐⭐⭐⭐ Offers a unified image-video self-supervised pre-trained model that acts as a plug-and-play solution for both image and video downstream tasks, with the code and weights already open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Self-supervised Video Copy Localization with Regional Token Representation](self-supervised_video_copy_localization_with_regional_token_representation.md)
- [\[ECCV 2024\] Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders](efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)
- [\[CVPR 2026\] Recurrent Video Masked Autoencoders](../../CVPR2026/self_supervised/recurrent_video_masked_autoencoders.md)
- [\[ECCV 2024\] Adaptive Multi-head Contrastive Learning](adaptive_multihead_contrastive_learning.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)

</div>

<!-- RELATED:END -->
