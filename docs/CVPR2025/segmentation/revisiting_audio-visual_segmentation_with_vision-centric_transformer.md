---
title: >-
  [Paper Note] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer
description: >-
  [CVPR 2025][Segmentation][Audio-Visual Segmentation] This paper proposes a Vision-Centric Transformer (VCT) framework to address the audio-visual segmentation task. By replacing traditional audio-derived queries with queries derived from visual features and pairing them with a Prototype Prompting Query Generation (PPQG) module, VCT achieves new state-of-the-art results on three AVSBench subsets, with particularly significant improvements on the challenging AVSS subset.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Audio-Visual Segmentation"
  - "Vision-Centric Transformer"
  - "Prototype Prompting"
  - "Query Design"
  - "Sounding Object Segmentation"
date: 2026-05-08
content_hash: cc90a8e9f22639ba
---

# Revisiting Audio-Visual Segmentation with Vision-Centric Transformer

**Conference**: CVPR 2025  
**arXiv**: [2506.23623](https://arxiv.org/abs/2506.23623)  
**Code**: [https://github.com/spyflying/VCT_AVS](https://github.com/spyflying/VCT_AVS)  
**Area**: Image Segmentation / Multimodal  
**Keywords**: Audio-Visual Segmentation, Vision-Centric Transformer, Prototype Prompting, Query Design, Sounding Object Segmentation

## TL;DR

This paper proposes a Vision-Centric Transformer (VCT) framework to address the audio-visual segmentation task. By replacing traditional audio-derived queries with queries derived from visual features and pairing them with a Prototype Prompting Query Generation (PPQG) module, VCT achieves new state-of-the-art results on three AVSBench subsets, with particularly significant improvements on the challenging AVSS subset.

## Background & Motivation

**Background**: Audio-Visual Segmentation (AVS) aims to perform pixel-level segmentation of sounding objects in a video using its audio signals. Existing mainstream methods adopt an audio-centric Transformer architecture, which uses audio features as or incorporates them into object queries, locating sounding objects through progressive interactions in a Transformer decoder. Representative methods include COMBO, AQFormer, and CATR.

**Limitations of Prior Work**: Audio-centric Transformers suffer from two fundamental limitations. (1) **Perceptual Ambiguity**: Audio in real-world scenes is usually a mixture of multiple sound sources, including sounds from both inside and outside the video frame. For example, an audio track may simultaneously contain human voices, guitar sounds, and out-of-screen car noises. Queries derived from such mixed audio interfere with each other, making it difficult to distinguish different sounding objects, and out-of-screen noise may lead to false positive predictions. (2) **Weakened Dense Prediction Capability**: AVS is inherently a vision-centric dense prediction task. Queries must simultaneously contain abstract audio semantics (to determine if an object is making sound) and concrete visual details (to delineate precise outlines). However, audio-derived queries initially possess only audio semantics, and the delayed integration of visual information leads to a loss of key details.

**Key Challenge**: There exists a conflict between the mixed nature of audio signals and the precision required for visual dense prediction. Searching for sounding objects in the visual scene starting from mixed audio is less direct and accurate than starting from visual regions and matching them with their corresponding audio signals.

**Goal**: Redesign the query mechanism for AVS to place visual information at the center, forcing queries to naturally encompass rich visual details while progressively acquiring audio semantics.

**Key Insight**: Shift queries from the audio domain to the visual domain—each query initially focuses on a different region of the image and progressively becomes "audio-aware" through multi-layer interactions with audio and visual features. Consequently, each query can independently extract its corresponding sound information from the mixed audio, avoiding mutual interference.

**Core Idea**: Replace audio-derived queries with vision-derived queries, along with audio prototype prompting and pixel context grouping, to achieve more accurate sounding-object differentiation and boundary delineation.

## Method

### Overall Architecture

Given a video of $T$ frames and corresponding audio segments, a vision encoder (Swin Transformer) is first used to extract multi-scale visual features $\{V_i\}_{i=2}^5$, and VGGish is used to extract audio features $A \in \mathbb{R}^{T \times S \times C^a}$. The highest-resolution visual features $V_2$ are fed into the PPQG module to generate $N$ vision-derived queries. These queries alternately interact with audio features and multi-scale visual features in an iterative audio-visual Transformer decoder, and finally output segmentation results through a classification head and a mask head.

### Key Designs

1. **Prototype Prompting Query Generation (PPQG)**:

    - **Function**: Generate vision-derived queries that contain rich visual details while possessing audio semantic perception.
    - **Mechanism**: Three-step generation. **Step 1 (Visual Embedding Aggregation)**: Project the high-resolution features $V_2$ through convolutional layers and an MLP and aggregate spatial information to obtain $N$ visual embeddings $V^e \in \mathbb{R}^{N \times C^h}$. **Step 2 (Audio Prototype Prompting)**: Define $K$ learnable audio prototypes $P \in \mathbb{R}^{K \times C^h}$ (where $K$ is the number of audio event categories), and inject class-specific audio priors into the visual embeddings via cross-attention: $\bar{V}^e = V^e + \text{Softmax}(\frac{(V^e W_1^q)(P W_1^k)^T}{\sqrt{C^h}})(P W_1^v)$. Concurrently, a prototype-audio contrastive loss $\mathcal{L}_{pac}$ is designed to ensure prototypes learn correct audio semantics. **Step 3 (Pixel Context Grouping)**: Use Gumbel-Softmax to achieve hard yet differentiable allocation, grouping image pixel contexts into individual queries to force them to focus on different image regions.
    - **Design Motivation**: Audio prototype prompting informs the queries of potential sound events in the scene before entering the decoder, allowing more targeted audio feature extraction in subsequent interactions. Gumbel-Softmax hard allocation ensures that different queries focus on distinct regions, thereby enhancing differentiability.

2. **Iterative Audio-Visual Transformer Decoder**:

    - **Function**: Enable vision-derived queries to progressively acquire corresponding sound information and detailed visual features.
    - **Mechanism**: The decoder is composed of interaction units $\mathcal{U} = \{A_t, V_5, V_4, V_3\}$ repeated $D$ times. Each unit consists of an **audio information extraction block** (where queries perform cross-attention with the current frame's audio features, with the audio acting as keys/values) and three **visual information enhancement blocks** (where queries sequentially perform cross-attention with $V_5, V_4, V_3$). The audio block allows each query to acquire sound information for its represented region, while the visual blocks capture finer visual features to precisely predict the mask. Following Mask2Former, the predicted mask from the previous layer is used as the attention mask for the current layer.
    - **Design Motivation**: Vision-derived queries, focused on different visual regions, can independently extract their respective audio information from the mixed audio, bypassing the mutual interference that plagues audio-derived queries. Progressive visual enhancement from low to high resolution ensures precise boundary prediction.

3. **Prototype-Audio Contrastive Loss (PAC Loss)**:

    - **Function**: Ensure that randomly initialized audio prototypes learn semantic information of different audio event categories.
    - **Mechanism**: Project and globally pool audio features, then perform an inner product with each prototype to obtain matching predictions $M \in \mathbb{R}^K$. Use dataset annotations to obtain the true audio event categories as the ground truth $M^*$. Train using BCE loss: $\mathcal{L}_{pac} = \frac{1}{K} \sum_k \mathcal{L}_{bce}(M_k, M_k^*)$. This loss decreases the distance between audio features and corresponding prototypes while increasing the distance to unrelated prototypes.
    - **Design Motivation**: Randomly initialized prototypes without loss constraints fail to learn meaningful audio priors (as verified by ablation studies). By contrastive learning with audio features rather than visual features, prototypes can acquire more explicit and clear audio event category priors.

### Loss & Training

The total loss is formulated as $\mathcal{L} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{mask}\mathcal{L}_{mask} + \lambda_{pac}\mathcal{L}_{pac}$, where $\lambda_{cls}=2, \lambda_{mask}=5, \lambda_{pac}=1$. The classification loss is CE loss, and the mask loss includes BCE + Dice loss. The model is trained using the AdamW optimizer with a learning rate of $1e^{-4}$. The number of training iterations is 45K for the S4 subset, 40K for the MS3 subset, and 45K for the AVSS subset. The number of vision-derived queries is $N=100$, and the decoder repetition count is $D=2$.

## Key Experimental Results

### Main Results

| Method | Backbone | AVSS $\mathcal{M_J}$ | AVSS $\mathcal{M_F}$ | S4 $\mathcal{M_J}$ | MS3 $\mathcal{M_J}$ |
|------|----------|-----|-----|-----|-----|
| COMBO | PVT-v2 | 42.1 | 46.1 | 84.7 | 59.2 |
| AVSBias | Swin-B(384) | 44.4 | 49.9 | 83.3 | 67.2 |
| TeSO | Swin-B(384) | 39.0 | 45.1 | 83.3 | 66.0 |
| **VCT (Ours)** | PVT-v2(224) | **44.7** | **49.5** | **84.8** | 62.0 |
| **VCT (Ours)** | Swin-B(224) | **47.9** | **52.9** | 84.7 | **67.5** |
| **VCT (Ours)** | Swin-B(384) | **51.2** | **55.5** | **86.2** | **67.6** |

### Ablation Study (AVSS Subset, ResNet-50)

| Configuration | $\mathcal{M_J}$ | $\mathcal{M_F}$ | Description |
|------|-----|-----|------|
| ACT (audio-derived queries) | 33.2 | 37.0 | Audio-centric baseline |
| VCT + Naive Vision Queries | 35.2 | 39.3 | Visual embedding aggregation only |
| + Cross-Attention | 35.8 | 39.8 | Normal softmax |
| + Group-Attention (Gumbel) | 36.3 | 40.5 | Gumbel-Softmax hard allocation |
| + Audio Prototypes (PAC) | **37.5** | **42.2** | Complete PPQG |

### Key Findings

- **Vision-Centric vs. Audio-Centric Fundamental Advantage**: Using only the simplest visual embeddings as queries (35.2 vs 33.2) already outperforms the audio-derived query baseline, proving the directional correctness of the vision-centric paradigm.
- **Most Significant Gains on the AVSS Subset**: On the most challenging semantic segmentation subset, VCT (Swin-B, 384) achieves 51.2 $\mathcal{M_J}$, which is 6.8 points higher than AVSBias (44.4). This indicates that vision-derived queries yield the greatest advantage when distinguishing multiple sounding object categories.
- **PVT-v2 (224) Competes with Swin-B (384)**: VCT with PVT-v2 and a 224 resolution (44.7 $\mathcal{M_J}$) outperforms AVSBias configured with Swin-B and a 384 resolution (44.4), demonstrating the high efficiency of the proposed architecture.
- **PAC Loss Must Be Contrasted with Audio Features**: Contrastive learning with visual queries (36.5) yields inferior results compared to contrasting with audio features (37.5), indicating that prototypes must learn semantics from audio signals rather than visual correlations.
- **Direct Fusion of Audio Features into Queries is Suboptimal**: Multiply/concat/add fusions of audio and visual features (33.9–36.3) are all inferior to the complete VCT scheme (37.5), confirming the hypothesis that late fusion is superior to early fusion.

## Highlights & Insights

- **Shift in Query Design Paradigm**: Transitioning from "audio looking for vision" to "vision looking for audio" is a simple yet highly effective perspective shift. The key insight is that queries from multiple visual regions can independently extract their respective information from the mixed audio, whereas queries derived from mixed audio are inherently entangled from the start.
- **Clever Utilization of Gumbel-Softmax**: Gumbel-Softmax is leveraged to achieve hard allocation, forcing different queries to focus on distinct image regions while maintaining differentiability. This design draws inspiration from GroupViT but produces new and effective results in multi-modal segmentation scenarios.
- **Dual Role of Audio Prototypes**: They function both as class priors in PPQG to guide query generation and as self-supervised elements that learn audio semantics via PAC loss. This represents an elegant self-supervised design.

## Limitations & Future Work

- The audio encoder uses the older VGGish; switching to stronger audio models (e.g., AudioMAE, BEATs) could yield further improvements.
- The current number of prototypes is fixed to $K$ (the number of audio event classes); open-world scenarios would require a dynamic prototype mechanism.
- Temporal modeling across multiple frames is not considered, and the current frame-by-frame processing might overlook temporal consistency.
- Under certain configurations on the MS3 subset, the improvements are less significant compared to those on AVSS, suggesting limited advantages in simpler scenarios.

## Related Work & Insights

- **vs. COMBO**: COMBO adds audio features to learnable queries and then performs bidirectional fusion with visual features. VCT is completely vision-centric, allowing audio information to be progressively obtained through decoder interaction, which avoids the information entanglement of early fusion.
- **vs. AQFormer**: AQFormer directly uses audio as queries to aggregate visual features. VCT does the opposite by using vision as queries to extract audio information. Logit map visualizations clearly demonstrate the advantage of vision-derived queries in focusing on more diverse regions.
- **vs. GAVS**: GAVS achieves 80.1 on S4 and 63.7 on MS3 using ViT-B, both of which are surpassed by VCT using Swin-B.

## Rating

- Novelty: ⭐⭐⭐⭐ The shift in query design paradigm is the key contribution. Though simple in concept, it yields remarkable effectiveness.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three subsets, ablation studies covering each component, and persuasive visual analyses.
- Writing Quality: ⭐⭐⭐⭐ Clearly explained motivations, professional diagrams, and a complete explanation of the methodology.
- Value: ⭐⭐⭐⭐ Provides a new design paradigm for the AVS field, with the PPQG module showing strong transfer potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[CVPR 2025\] Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)
- [\[CVPR 2025\] Audio-Visual Instance Segmentation](audio-visual_instance_segmentation.md)
- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)

</div>

<!-- RELATED:END -->
