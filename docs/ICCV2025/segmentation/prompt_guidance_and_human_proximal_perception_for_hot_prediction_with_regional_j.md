---
title: >-
  [Paper Note] Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss
description: >-
  [ICCV 2025][Segmentation][Human-Object Contact Detection] This paper proposes P3HOT, a framework that achieves state-of-the-art performance on Human-Object Contact (HOT) detection by incorporating text prompt guidance to…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Human-Object Contact Detection"
  - "Text Guidance"
  - "Depth Perception"
  - "Regional Joint Loss"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: 201eeb18e13377dc
---

# Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss

**Conference**: ICCV 2025
**arXiv**: [2507.01630](https://arxiv.org/abs/2507.01630)  
**Code**: [github.com/YuxiaoWang-AI/P3HOT](https://github.com/YuxiaoWang-AI/P3HOT)  
**Area**: Image Segmentation
**Keywords**: Human-Object Contact Detection, Text Guidance, Depth Perception, Regional Joint Loss, Semantic Segmentation

## TL;DR

This paper proposes P3HOT, a framework that achieves state-of-the-art performance on Human-Object Contact (HOT) detection by incorporating text prompt guidance to focus on human contact regions, a depth-aware module to filter irrelevant backgrounds, and a Regional Joint Loss to enforce intra-region category consistency.

## Background & Motivation

HOT (Human-Object conTact) detection originates from HOI (Human-Object Interaction) and aims to identify **which specific body parts** are in contact with objects, partitioning the human body into 18 categories (17 parts + background). This task holds significant value for human-robot interaction, VR, and gesture recognition.

Limitations of existing methods (DHOT, PIHOT):

**Single modality**: Reliance solely on image features, neglecting semantic information from text guidance.

**Intra-region category inconsistency**: Cross-entropy loss cannot constrain category consistency within regions, leading to spurious predictions within a region (e.g., small "head" predictions appearing inside a palm region).

**Interference from low-interaction regions**: Over-segmentation in areas with minimal interaction.

**Evaluation metric flaw**: The C-Acc. metric is flawed — predicting the entire image as a single contact category yields 100% C-Acc.

Core motivation: Introduce multimodal information (text prompts + pseudo-3D depth) to enhance HOT detection, and design a dedicated loss function to ensure intra-region segmentation consistency.

## Method

### Overall Architecture

P3HOT consists of four components:
- **Image Encoder** (ResNet-50 + CLIP initialization + Attention Pooling)
- **Text Encoder** (CLIP Text Encoder, frozen parameters)
- **Human Proximal Perception (HPP) Module** (SAM + ZoeDepth + learnable parameter $\tau$)
- **Image Decoder** (progressive upsampling + multi-scale feature fusion)

### Key Designs

1. **Text Prompt Guidance Mechanism**: Seventeen templates of the form "A [body part] of the human body is in contact with an object" are constructed by substituting each of the 17 body parts. Text features $\mathbf{F}_{TE} \in \mathbb{R}^{17 \times 1024}$ are extracted via the CLIP text encoder, and their cosine similarity $\mathbf{S}$ with image features $\mathbf{F}_{IE}$ is computed:
$$\mathbf{S} = \frac{\mathbf{F}_{IE} \cdot \mathbf{F}_{TE}^T}{\|\mathbf{F}_{IE}\| \cdot \|\mathbf{F}_{TE}\|}$$
The similarity $\mathbf{S}$ is applied channel-wise to the decoder output, enhancing responses for body parts attended to by the text. **Core Idea**: Image-text matching scores are used to dynamically reweight each body part channel.

2. **Human Proximal Perception (HPP) Module**: Addresses the ambiguity of human-object overlap in 2D views:

    - SAM (with text prompt "person") generates a human mask $\mathbf{M}$
    - ZoeDepth extracts a depth map $\mathbf{D}$, normalized to $[0,1]$
    - The average depth $\mathbf{m}_i^{da}$ is computed for each detected person
    - A learnable parameter $\tau$ defines a depth range $[\mathbf{m}_i^{da} - \tau, \mathbf{m}_i^{da} + \tau]$
    - A filter mask retains the human body and surrounding objects while excluding irrelevant background

   To make $\tau$ differentiable, hard threshold comparisons are replaced with ReLU:
    $\Theta_i = (D_{Norm} - (\mathbf{m}_i^{da} - \tau)) \otimes ((\mathbf{m}_i^{da} + \tau) - D_{Norm})$
    $FM = FM + \sum_{i=1}^N \text{ReLU}(\Theta_i)$

3. **Multi-Scale Feature Fusion Decoder**: Outputs from the four encoder blocks (downsampling rates $S_i = \{4, 8, 16, 32\}$) are progressively upsampled and concatenated:
$$\mathbf{x}_{i-1} = \text{DoubleConv}(\text{Up}(\mathbf{x}_i) \text{ⓒ} \mathbf{F}_{i-1})$$
This compensates for texture detail loss caused by downsampling.

### Loss & Training

**Regional Joint Loss (RJLoss)** consists of two components:

**(a) Local Joint Loss** — penalizes the appearance of other categories within each GT-defined category region:
$$\mathcal{L}_c^L = \frac{\sum(|\mathbf{O}_c - \mathbf{GT}_c| \otimes \mathbf{GT}_c)}{\sum \mathbf{GT}_c}$$

**(b) Global Joint Loss** — over the full prediction map, identifies connected regions (closed regions) surrounded by a given category and penalizes misclassified pixels within them. Connected component analysis detects "island" regions enclosed by boundary pixels:
$$\mathcal{L}_c^G = \sum(\neg \mathbf{O}_c \otimes \mathbf{O}_c^M)$$

**Total Loss**:
$$\mathcal{L} = \text{CE}(\mathbf{O}, \mathbf{GT}) + \alpha \mathcal{L}^L + \beta \mathcal{L}^G + \gamma \text{BE}(\mathbf{S}, \mathbb{C})$$
where $\alpha = 0.3, \beta = 0.1, \gamma = 1.0$. BE denotes the binary cross-entropy loss for image-text matching.

Training setup: 8×NVIDIA A6000 GPUs, AdamW optimizer, batch size 4 per GPU.

## Key Experimental Results

### Main Results

| Method | Dataset | SC-Acc. | mIoU | wIoU | AD-Acc. (New Metric) |
|------|--------|---------|------|------|-----------------|
| DHOT-Full | HOT-Annotated | 40.7 | 21.5 | 26.0 | - |
| PIHOT | HOT-Annotated | 45.3 | 23.6 | 28.6 | 31.3 |
| **P3HOT (Ours)** | **HOT-Annotated** | **46.0** | **25.6** | **30.2** | **42.3** |
| DHOT-Full | HOT-Generated | 30.4 | 13.9 | 16.7 | - |
| PIHOT | HOT-Generated | 34.9 | 16.9 | 21.2 | 25.4 |
| **P3HOT (Ours)** | **HOT-Generated** | **35.2** | **18.0** | **23.1** | **30.6** |

Gains on HOT-Annotated: SC-Acc. +0.7, mIoU +2.0, wIoU +1.6, **AD-Acc. +11.0**.

### Ablation Study

**Contribution of each component** (HOT-Annotated):

| Configuration | SC-Acc. | mIoU | AD-Acc. | Note |
|------|---------|------|---------|------|
| Baseline | 37.2 | 19.0 | 30.3 | Encoder + Decoder only |
| +Fine (multi-scale fusion) | 38.9 | 20.1 | 34.1 | +3.8 AD-Acc. |
| +TE (text encoder) | 40.3 | 21.0 | 37.1 | +3.0 AD-Acc. |
| +TE+DM+SAM (HPP) | 43.2 | 23.1 | 38.9 | Significant HPP contribution |
| +RJLoss | **46.0** | **25.6** | **42.3** | Additional +3.4 from RJLoss |

**Effect of loss functions**:

| Loss Configuration | SC-Acc. | mIoU | AD-Acc. |
|---------|---------|------|---------|
| CE only | 43.2 | 23.1 | 38.9 |
| +BE | 44.5 | 23.8 | 40.1 |
| **+RJLoss** | **46.0** | **25.6** | **42.3** |

**Depth map normalization**:

| Depth Range | SC-Acc. | AD-Acc. | Note |
|---------|---------|---------|------|
| $D \in R$ (unnormalized) | 44.1 | 40.2 | $\tau$ difficult to optimize |
| **$D \in [0,1]$** | **46.0** | **42.3** | Optimal after normalization |

### Key Findings

- **RJLoss contributes the most**: AD-Acc. improves by 3.4 points from CE-only to +RJLoss, demonstrating the importance of intra-region consistency constraints.
- The proposed **AD-Acc. metric** corrects a critical flaw in C-Acc. (which yields 100% by predicting a single class over the entire image).
- **Depth normalization** is critical for learning $\tau$: without normalization, large depth range variations across images prevent $\tau$ from converging.
- The HPP module filters irrelevant regions from a pseudo-3D perspective, outperforming the use of SAM masks or depth alone.

## Highlights & Insights

- **AD-Acc. as a new evaluation metric**: Accounts for both positive and negative samples, resolving the C-Acc. flaw and providing a fairer benchmark for the HOT community.
- **Connected component analysis in RJLoss is conceptually distinctive**: By detecting "island" regions enclosed by a given category to localize intra-region misclassifications, this approach has broad applicability in segmentation tasks.
- **Learnable depth threshold $\tau$**: Differentiability is achieved by replacing hard thresholding with ReLU, elegantly enabling adaptive depth range estimation.
- **First introduction of text guidance to HOT**: Establishes a foundation for subsequent multimodal HOT research.

## Limitations & Future Work

- The ResNet-50 backbone is relatively dated; stronger visual encoders (ViT, Swin) could be explored.
- SAM and ZoeDepth as external modules introduce inference overhead; lightweight alternatives are worth investigating.
- Validation is limited to only two datasets (HOT-Annotated and HOT-Generated) of modest scale.
- Text prompts rely on fixed templates; dynamic generation or context-aware prompts could be explored.

## Related Work & Insights

- The text guidance mechanism is inspired by CLIP-based applications in HOI detection (GEN-VLKT, FreeA).
- The depth-aware mechanism complements 3D HOT detection methods (PROX+SMPL-X): 2D approaches are more efficient and broadly applicable.
- The intra-region consistency principle of RJLoss is transferable to medical image segmentation and semantic segmentation tasks.

## Rating

- Novelty: ⭐⭐⭐ — Individual components are not entirely novel, but their combination is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations with multi-dimensional analysis.
- Practicality: ⭐⭐⭐⭐ — Open-source code; AD-Acc. metric is practically useful.
- Overall: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](../../NeurIPS2025/segmentation/haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[ICCV 2025\] Joint Self-Supervised Video Alignment and Action Segmentation](joint_self-supervised_video_alignment_and_action_segmentation.md)
- [\[ICCV 2025\] Advancing Visual Large Language Model for Multi-granular Versatile Perception](advancing_visual_large_language_model_for_multi-granular_versatile_perception.md)
- [\[ICCV 2025\] Temporal Rate Reduction Clustering for Human Motion Segmentation](temporal_rate_reduction_clustering_for_human_motion_segmentation.md)
- [\[ICCV 2025\] ConformalSAM: Unlocking the Potential of Foundational Segmentation Models in Semi-Supervised Semantic Segmentation with Conformal Prediction](conformalsam_unlocking_the_potential_of_foundational_segmentation_models_in_semi.md)

</div>

<!-- RELATED:END -->
