---
title: >-
  [Paper Note] TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition
description: >-
  [CVPR 2025][Video Understanding][Cross-Domain Few-Shot Action Recognition (CDFSAR)] This paper proposes TAMT, a decoupled "pre-train, fine-tune" paradigm for cross-domain few-shot action recognition (CDFSAR). By efficiently recalibrating intermediate features of frozen models with a Temporal-Aware Adapter (TAA) and generating strong representations with Global Temporal Moment Tuning (GTMT) to capture long- and short-term temporal covariance, TAMT outperforms existing methods…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Cross-Domain Few-Shot Action Recognition (CDFSAR)"
  - "Temporal-Aware Adapter"
  - "Model Fine-Tuning"
  - "Covariance Features"
  - "Decoupled Training"
date: 2026-05-08
content_hash: c9fc6d498bfd07e7
---

# TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition

**Conference**: CVPR 2025  
**arXiv**: [2411.19041](https://arxiv.org/abs/2411.19041)  
**Code**: [https://github.com/TJU-YDragonW/TAMT](https://github.com/TJU-YDragonW/TAMT)  
**Area**: Video Understanding / Few-Shot Learning  
**Keywords**: Cross-Domain Few-Shot Action Recognition (CDFSAR), Temporal-Aware Adapter, Model Fine-Tuning, Covariance Features, Decoupled Training

## TL;DR

This paper proposes TAMT, a decoupled "pre-train, fine-tune" paradigm for cross-domain few-shot action recognition (CDFSAR). By efficiently recalibrating intermediate features of frozen models with a Temporal-Aware Adapter (TAA) and generating strong representations with Global Temporal Moment Tuning (GTMT) to capture long- and short-term temporal covariance, TAMT outperforms existing methods by 13% to 31% across multiple cross-domain scenarios while requiring 5 times lower training costs.

## Background & Motivation

**Background**: Few-shot action recognition (FSAR) aims to classify video actions using a small number of labeled samples. Cross-domain FSAR (CDFSAR) further introduces domain discrepancy between source and target domains, requiring knowledge to be learned on a label-rich source domain and transferred to a label-scarce target domain.

**Limitations of Prior Work**: Existing CDFSAR methods (e.g., SEEN, CDFSL-V) adopt a joint-training paradigm, training source and target data together to mitigate domain shift. However, they suffer from two key limitations: (1) with one source domain and multiple target domains, joint training requires retraining the model for each target domain, causing computation costs to scale linearly with the number of target domains; (2) they use simple nearest neighbor classifiers or fine-tuned classifiers during inference, failing to fully leverage the potential of pretrained models.

**Key Challenge**: The conflict between the heavy computational cost of joint training versus the multi-target domain adaptation demands, compounded by the underutilization of the pretrained model's representation capabilities.

**Goal**: (1) Avoid repetitive training across multiple target domains; (2) efficiently adapt pretrained models to target domains; (3) generate superior video representations for few-shot matching.

**Key Insight**: Adopting a decoupled paradigm—performing source domain pre-training once and lightweight, rapid target domain fine-tuning. Additionally, designing a temporal-aware adapter, since temporal modeling is central to video understanding while existing adapters focus primarily on spatial information.

**Core Idea**: Replacing joint training with a decoupled pre-train and fine-tune paradigm, achieving highly efficient cross-domain few-shot action recognition with extremely few trainable parameters via a lightweight temporal-aware adapter and temporal covariance feature representations based on first- and second-order moments.

## Method

### Overall Architecture

TAMT consists of two stages: (1) Source Domain Pre-training: First, a VideoMAE encoder is trained via Self-Supervised Learning (SSL) reconstruction to acquire general spatiotemporal structures, followed by Supervised Learning (SL) classification to enhance semantic discriminability. (2) Target Domain Fine-tuning: The pretrained encoder is frozen, and few-shot adaptation is performed via Hierarchical Temporal Tuning Network (HTTN). HTTN includes local TAA adapters embedded in the last $L$ layers of the Transformer, and a global GTMT module at the end. Metric learning (Euclidean distance) is ultimately applied to compare query and support representations for classification.

### Key Designs

1. **Temporal-Aware Adapter (TAA)**:

    - **Function**: Calibrates intermediate video features of the frozen model using very few learnable parameters.
    - **Mechanism**: For the output features $\mathbf{F} \in \mathbb{R}^{T \times M \times C}$ from each Transformer layer, TAA generates temporal-aware scaling factors $\gamma$ and shift factors $\beta$ to perform $\mathbf{F'} = \gamma \odot \mathbf{F} \oplus \beta$. $\gamma$ and $\beta$ are generated by applying global average pooling to the features, followed by two temporal convolutional layers (kernel size $k_t=3$) and a bottleneck dimensionality reduction ($C \to C/\rho \to C$, $\rho=4$). $\gamma$ and $\beta$ share the weights of the reduction layers to further reduce parameters.
    - **Design Motivation**: Traditional full-parameter fine-tuning is prone to overfitting in few-shot scenarios and demands heavy computation. Unlike spatial adapters in NLP/image classification, TAA explicitly captures inter-frame dynamic details using temporal convolutions, making it more suitable for video tasks. It requires only 2.8M parameters (vs. 29.9M for FFT) and 1.9GB VRAM (vs. 17.5GB for FFT).

2. **Global Temporal Moment Tuning (GTMT) + Efficient Long-Short Temporal Covariance (ELSTC)**:

    - **Function**: Generates stronger global video representations using the first- and second-order moments of feature distributions.
    - **Mechanism**: The final representation is formulated as $\mathbf{Z} = \mathcal{H}(\mathbf{M}_2) \oplus \mathbf{M}_1$, where the first-order moment $\mathbf{M}_1$ is obtained via global average pooling, and the second-order moment $\mathbf{M}_2$ is calculated via ELSTC. ELSTC divides the temporal dimension into $G$ groups and computes the inter-frame covariance matrix $\mathbf{R}_{t,t'}$ within each group, capturing multi-scale temporal correlations from short-term (intra-frame appearance) to long-term (cross-frame motion). Finally, the covariance matrices from all groups are aggregated through two convolutional layers and aligned in dimensionality via linear projection before being added to the first-order moment.
    - **Design Motivation**: Traditional methods use only global average pooling (first-order moment) as representations, discarding rich second-order statistical information. The covariance matrix describes the shape of the feature distribution, containing descriptions of inter-frame motion patterns. The grouping strategy reduces the computational cost by $G$ times, making the second-order moment calculation feasible.

3. **Two-Stage Pre-training Strategy**:

    - **Function**: Learns feature representations with both generalization and discriminability capabilities on the source domain.
    - **Mechanism**: The encoder is first trained for 400 epochs using VideoMAE's masked reconstruction objective (SSL) to learn general spatiotemporal structures, and subsequently trained for 140 epochs using a cross-entropy classification loss (SL) to enhance semantic discriminability.
    - **Design Motivation**: Pure SSL captures low-level features but lacks high-level semantic meaning; pure SL provides insufficient generalization in few-shot cross-domain scenarios. The two-stage strategy strikes a balance between generalizability and representation capability.

### Loss & Training

- Pre-training stage: SSL uses Mean Squared Error (MSE) loss, SL uses Cross-Entropy (CE) loss.
- Fine-tuning stage: CE loss; Euclidean distance serves as the metric function.
- Optimizer: SGD + Cosine annealing learning rate, with only 40 epochs for fine-tuning.
- Inference: 5-way 1/5-shot, averaged over 10,000 episodes.

## Key Experimental Results

### Main Results

**K-400 → Five Target Domains (5-way 5-shot Accuracy %)**:

| Method | HMDB | SSV2 | Diving | UCF | RareAct | Average |
|------|------|------|--------|-----|---------|------|
| CDFSL-V | 53.23 | 49.92 | 17.84 | 65.42 | 49.80 | 47.24 |
| SEEN | - | - | - | - | - | - |
| **TAMT (Ours)** | **74.14** | **59.18** | **45.18** | **95.92** | **67.44** | **68.37** |
| Gain | +20.91 | +9.26 | +27.34 | +30.50 | +17.64 | **+21.13** |

### Ablation Study

**Pre-training Strategy + Fine-Tuning Paradigm (K-400 Source Domain, 5-way 5-shot)**:

| Pre-training | Fine-tuning Method | SSV2 | Diving | UCF | Average |
|--------|---------|------|--------|-----|------|
| SSL only | Frozen | 29.27 | 22.10 | 55.30 | 35.56 |
| SL only | TAMT | 45.15 | 37.96 | 89.73 | 56.48 |
| SSL+SL | FFT | 55.99 | 42.85 | 94.95 | 64.30 |
| **SSL+SL** | **TAMT** | **59.18** | **45.18** | **95.92** | **66.76** |

**Efficiency Comparison**:

| Metric | FFT | TAMT |
|------|-----|------|
| VRAM | 17.5GB | 1.9GB |
| Parameters | 29.9M | 2.8M |
| Training Time | 10.6h | 7.3h |

### Key Findings

- With only 9.4% of the trainable parameters (2.8M vs. 29.9M), TAMT outperforms full-parameter fine-tuning (FFT) by 2.46% in average accuracy, demonstrating that parameter-efficient fine-tuning (PEFT) is superior to FFT in few-shot scenarios.
- The two-stage SSL+SL pre-training significantly outperforms either stage alone, where SSL grants generalization (+10.28%) and SL ensures execution accuracy and semantics discriminability.
- TAMT's computational training cost is approximately 1/5 that of CDFSL-V (19 vs. 88 GPU days), yet delivers a 21% performance improvement.
- The largest improvement is observed on the Diving48 dataset (+27.3%), verifying that temporal modeling is crucial for fine-grained action recognition.

## Highlights & Insights

- **Efficiency Advantages of the Decoupled Paradigm**: The source domain only requires a single pre-training phase, and transferring to multiple target domains only requires lightweight fine-tuning. This significantly reduces deployment costs and is generalizable to other cross-domain few-shot tasks.
- **Temporal-Aware Parameter-Efficient Fine-Tuning**: Unlike spatial-focused adapters in the image domain, TAA explicitly models inter-frame relationships via temporal convolutions, offering a meaningful exploration of video PEFT.
- **The Power of Second-Order Statistics**: ELSTC utilizes the covariance matrix to capture inter-frame correlations. Compared to using only the mean (first-order statistics), it encodes richer motion patterns while the grouping strategy maintains computational efficiency.

## Limitations & Future Work

- Evaluated only on ViT-S/B backbones, leaving scalability with larger models (e.g., ViT-L) untouched.
- Hyperparameters such as the number of groups $G$ and the dimension reduction ratio $\tau$ in ELSTC require manual tuning.
- The integration with vision-language priors (e.g., CLIP) has not yet been explored.
- Performance gains on rare action datasets (e.g., RareAct) are relatively modest, potentially due to severe domain shifts from the source domain distribution.
- Incorporating attention mechanisms into GTMT to dynamically select crucial temporal covariance components could be investigated.

## Related Work & Insights

- **vs. SEEN**: SEEN employs joint training and contrastive learning to alleviate domain shifts, yet requires retraining for each target domain. TAMT's decoupled paradigm is more efficient and leverages the pretrained model more comprehensively during fine-tuning.
- **vs. CDFSL-V**: CDFSL-V relies on a two-stage joint training and curriculum learning process with high training costs and lower performance than TAMT. TAMT yields an average improvement of 31.15% on the K-100 source domain.
- **vs. Image Adapters (e.g., AdaptFormer)**: While such methods focus primarily on spatial information, TAMT's TAA captures inter-frame dynamics through temporal convolutions, making it more tailored for video tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The integration of the decoupled paradigm and temporal-aware adapters represents a highly reasonable and effective innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across 5 source and 5 target domains with exhaustive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the motivations are thoroughly articulated.
- Value: ⭐⭐⭐⭐ Provides a simple yet effective baseline for CDFSAR, displaying distinct advantages in performance and efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](../../ICCV2025/video_understanding/trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](../../ECCV2024/video_understanding/efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](../../AAAI2026/video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
