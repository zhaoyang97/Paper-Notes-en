---
title: >-
  [Paper Note] I0T: Embedding Standardization Method Towards Zero Modality Gap
description: >-
  [ACL 2025][Modality gap] The I0T framework is proposed to reduce the modality gap to near zero by identifying and eliminating modality-specific features (manifested as peak activations in normalized embeddings) independently learned by the image and text encoders in CLIP. It maintains or even improves downstream task performance and introduces I0T-Score, an automatic evaluation metric that is more interpretable than CLIPScore.
tags:
  - "ACL 2025"
  - "Modality gap"
  - "CLIP"
  - "Embedding standardization"
  - "Batch normalization"
  - "Cross-modal alignment"
date: 2026-05-08
content_hash: 1148b30f4115a84c
---

# I0T: Embedding Standardization Method Towards Zero Modality Gap

**Conference**: ACL 2025  
**arXiv**: [2412.14384](https://arxiv.org/abs/2412.14384)  
**Code**: [GitHub](https://github.com/xfactlab/I0T)  
**Area**: Others  
**Keywords**: Modality gap, CLIP, Embedding standardization, Batch normalization, Cross-modal alignment

## TL;DR

The I0T framework is proposed to reduce the modality gap to near zero by identifying and eliminating modality-specific features (manifested as peak activations in normalized embeddings) independently learned by the image and text encoders in CLIP. It maintains or even improves downstream task performance and introduces I0T-Score, an automatic evaluation metric that is more interpretable than CLIPScore.

## Background & Motivation

**The Modality Gap Problem**: After contrastive learning training, vision-language models like CLIP project image and text embeddings onto different manifolds in the latent space, creating a significant modality gap. Consequently, intra-modality similarity is consistently higher than cross-modality similarity, preventing CLIP from accurately measuring the true semantic relationships between images and texts.

**Limitations of Prior Work**: As a widely used automatic evaluation metric for image captioning, CLIPScore is based on the cosine similarity of image and text embeddings. However, due to the modality gap, CLIPScore counterintuitively assigns higher scores to mismatched image-image pairs than to correct image-text pairs.

**Limitations of Prior Work**: Previous methods, such as Mind-the-Gap, pull positive sample pairs closer by shifting embeddings but fail to identify the root cause of the modality gap. This paper uncovers the true attribution factor: modality-specific features learned independently by each encoder.

## Method

### Overall Architecture

The I0T framework is divided into two stages:
1. **First Stage** (plug-and-play, optional): Enhances the semantic representation capability of CLIP, allowing the frozen encoders to retain rich semantics.
2. **Second Stage** (core): Eliminates the modality gap through embedding standardization, proposing two methods: post-processing $\text{I0T}_{\text{post}}$ and trainable $\text{I0T}_{\text{async}}$.

### Key Designs

**Root Cause Analysis of the Modality Gap**: The authors analyze the activation patterns of CLIP's normalized embeddings and find:
- All image samples exhibit a consistent negative peak activation at the 93rd dimension.
- All text samples exhibit consistent positive peak activations at the 134th and 313th dimensions.
- These peak activations have extremely small standard deviations and are irrelevant to the semantic content of the samples.

**Theoretical Analysis**: Assuming the image embedding has a negative peak of magnitude $p$ and the text embedding has two positive peaks of magnitude $q$, with other dimensions uniformly distributed, the upper bound of the cosine similarity converges to $\sqrt{(1-p^2)(1-2q^2)}$. Substituting the actual values of Long-CLIP ($p=-1/2$, $q=1/3$), the upper bound converges to 0.76, which is far less than 1, directly explaining the existence of the modality gap.

**Modality Gap Severity Classification**:
- Severe: Centroid Distance $\triangle_{\text{CD}} \geq 0.63$
- Moderate: $0.19 \leq \triangle_{\text{CD}} < 0.63$
- Low: $\triangle_{\text{CD}} < 0.19$

**$\text{I0T}_{\text{post}}$ (Post-processing Method)**: Standardizes the normalized embeddings of the frozen encoders—subtracting the mean vector of each modality and then re-performing Frobenius normalization:

$$\mathbf{x}_i' = \text{Normalize}(\mathbf{x}_i - \bar{\mathbf{x}}), \quad \mathbf{y}_i' = \text{Normalize}(\mathbf{y}_i - \bar{\mathbf{y}})$$

This is more effective than simple clipping because it removes modality-specific features across all dimensions rather than just handling peak dimensions.

**$\text{I0T}_{\text{async}}$ (Trainable Method)**: Adds independent Batch Normalization (BN) layers, $\text{BN}_{\text{img}}$ and $\text{BN}_{\text{txt}}$, to each encoder and trains them asynchronously (training the encoders first, then freezing them to train the BN layers). The BN layers learn the mean and variance of each modality's normalized embeddings, automatically standardizing them during inference.

**MCSIE (Multimodal Contrastive Learning of Sentence and Image Embeddings)**: An unsupervised positive enhancement strategy is utilized when training the BN layers. Dropout (rate=0.1) is applied as data augmentation to both ViT and Transformer encoders, enhancing the robustness of BN in learning modality-specific features.

### Loss & Training

The first stage uses the CyCLIP loss: $\mathcal{L}_{\text{CyCLIP}} = \mathcal{L}_{\text{CLIP}} + 0.25\mathcal{L}_{\text{I-Cyclic}} + 0.25\mathcal{L}_{\text{C-Cyclic}}$

Key Training Decisions: Adopt the Long-CLIP-only strategy (using only the ShareGPT4V long captions from COCO), which reduces training time to 1/10 of Long-CLIP while achieving better performance. Utilizes the AdamW optimizer with lr=1e-6, weight decay=1e-2, batch size=128, training for 3 epochs.

## Key Experimental Results

### Main Results

**Reduction in Modality Gap** (Table 2):

| Model | Centroid Distance (CD↓) | Linear Separability (LS↓) | Severity |
|------|-------------|---------------|--------|
| CLIP (Original) | 0.7642 | 0.9985 | Severe |
| $\text{I0T}_{\text{async}}$ | 0.4795 | 0.9960 | Moderate |
| $\text{I0T}_{\text{post}}$ | **0.0102** | **0.5374** | **Low** |
| Mind-the-Gap (λ=0.375) | 0.0291 | 0.5632 | Low |

**Maintenance of Downstream Task Performance** (Table 2):

| Model | I2T Retrieval | T2I Retrieval | CIFAR Classification | Flickr-Expert |
|------|---------|---------|-----------|--------------|
| CLIP | 69.60 | 67.10 | 65.05 | 51.00 |
| $\text{I0T}_{\text{async}}$ | 72.50 | 73.80 | 62.97 | 53.33 |
| $\text{I0T}_{\text{post}}$ | **73.30** | **76.30** | 63.07 | 53.97 |

- $\text{I0T}_{\text{post}}$ improves text-to-image (T2I) retrieval by 9.2% (67.10 $\rightarrow$ 76.30).
- CIFAR classification outperforms PAC-S by 4.46%.

### Ablation Study

**First-Stage Strategy Comparison** (Table 1):
- Long-CLIP-only (LCO): Achieves comparable performance to Long-CLIP while reducing training time by approximately 90%.
- Long-CyCLIP-only (LCCO): Incorporates cyclic losses to further improve performance.
- Add Layer Normalization (+LN): Fails to reduce the modality gap (remains at the severe level).
- Add Batch Normalization (+BN): Effectively reduces the gap to a moderate level.
- Asynchronous training of BN (+BN*): Outperforms synchronous BN training and serves as the final $\text{I0T}_{\text{async}}$.

**Comparison with BLIP** (Figure 4): With only 2/5 of BLIP's parameter count, I0T reduces the centroid distance by 94.68% and 47.74%, while achieving comparable Flickr-Expert correlation.

### Key Findings

1. Peak activations independently learned by each encoder are the direct cause of the modality gap, rather than the narrow cone effect in high-dimensional space as previously assumed.
2. Simple clipping cannot eliminate the modality gap; the modality-specific mean across all dimensions must be removed.
3. Batch Normalization (BN) rather than Layer Normalization (LN) is required to effectively reduce the modality gap since BN's statistics computation aligns with the distribution of modality-specific features.
4. There is no direct causal relationship between the modality gap and downstream performance—reducing the modality gap does not imply sacrificing performance.

## Highlights & Insights

- **Thorough Root Cause Analysis**: Not only identifies the phenomenon of peak activations but also provides a mathematical derivation proving their impact on the upper bound of cosine similarity.
- **Simple and Effective**: $\text{I0T}_{\text{post}}$ requires only mean subtraction and renormalization with zero parameter overhead, yet reduces the modality gap to near zero.
- **Practical Value of I0T-Score**: The cosine similarity distribution changes from the skewed distribution in CLIP-S to a wider distribution centered around zero, naturally assigning higher scores to correct pairs and negative scores to incorrect pairs without requiring a scaling factor $\omega$.
- **Two-Stage Decoupled Design**: Separates semantic enhancement from modality gap elimination, allowing each to be optimized independently, resulting in an elegant design.

## Limitations & Future Work

- $\text{I0T}_{\text{post}}$ requires statistics from the entire test set, which does not support single-sample zero-shot inference.
- Although $\text{I0T}_{\text{async}}$ addresses the aforementioned issue, it fails to reduce the gap to near zero (remaining at a moderate level).
- Validated only on the ViT-B/32 architecture; the effectiveness on larger models (such as ViT-L/14) and other architectures remains unverified.
- Only image and text modalities are handled; expansion to other modalities like audio and video has not been explored.
- Batch statistics of the BN layer may be sensitive to small batch sizes or out-of-distribution (OOD) data.
- The mechanism linking modality gap reduction to downstream performance improvement requires deeper theoretical analysis.

## Related Work & Insights

- Compared with Mind-the-Gap (Liang et al., 2022): I0T identifies the root cause of the modality gap (modality-specific features), whereas MG fails to eliminate peak activations through mere shifts.
- Compared with training methods like CyCLIP, CLOOB, and Unif-Align: The post-processing method of I0T achieves the best results with zero training cost.
- Inspiration for Future Research: The BN layer can be reinterpreted as an effective tool for eliminating the modality gap rather than merely a training stabilization technique.

## Rating

- **Novelty**: 8/10 — The root cause analysis of the modality gap and the concept of embedding standardization are highly novel.
- **Technical Depth**: 8/10 — Solid theoretical analysis and a well-structured methodological design.
- **Experimental Thoroughness**: 8/10 — Comprehensive evaluations with comparisons across multiple methods, diverse tasks, and detailed ablation studies.
- **Value**: 9/10 — I0T-Score can be directly deployed as a superior evaluation metric.
- **Overall Rating**: 8/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)
- [\[ACL 2025\] Entropy-UID: A Method for Optimizing Information Density](entropy-uid_a_method_for_optimizing_information_density.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method](towards_comprehensive_argument_analysis_in_education_dataset_tasks_and_method.md)
- [\[ACL 2025\] TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)

</div>

<!-- RELATED:END -->
