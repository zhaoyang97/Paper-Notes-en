---
title: >-
  [Paper Note] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation
description: >-
  [CVPR2025][Medical Imaging][federated learning] This work proposes FedMEPD, a federated learning framework that simultaneously addresses inter-modality heterogeneity and client personalization in multimodal MRI brain tumor segmentation through modality-specific encoders (globally federated) and a partially personalized fusion decoder. It achieves an average client mDSC of 75.70%/75.90% on BraTS 2018/2020.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "federated learning"
  - "brain tumor segmentation"
  - "multimodal MRI"
  - "personalized FL"
  - "missing modality"
date: 2026-05-08
content_hash: fa0623bdd5ece1da
---

# Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation

**Conference**: CVPR2025  
**arXiv**: [2603.04887](https://arxiv.org/abs/2603.04887)  
**Code**: [GitHub](https://github.com/ccarliu/FedMEPD)  
**Area**: Medical Image  
**Keywords**: federated learning, brain tumor segmentation, multimodal MRI, personalized FL, missing modality

## TL;DR

This work proposes FedMEPD, a federated learning framework that simultaneously addresses inter-modality heterogeneity and client personalization in multimodal MRI brain tumor segmentation through modality-specific encoders (globally federated) and a partially personalized fusion decoder. It achieves an average client mDSC of 75.70%/75.90% on BraTS 2018/2020.

## Background & Motivation

Federated learning (FL) enables collaborative training across multiple institutions without compromising privacy. Existing medical image FL methods primarily address intra-modality heterogeneity (e.g., differences in data distribution) but overlook inter-modality heterogeneity: in multimodal MRI brain tumor segmentation, different institutions may only possess a subset of the four modalities (T1, T1c, T2, FLAIR). This introduces two concurrent challenges: (1) how to effectively train a global model in the presence of incomplete modalities, and (2) how to provide each participant with a personalized model tailored to their local data characteristics. Existing approaches either require all clients to have identical modalities (Xiong et al.), train a single global model that fails to meet personalization needs (FedIoT), or require data sharing, which violates privacy constraints (CreamFL).

## Method

### Overall Architecture

FedMEPD consists of four core components:

1. **Modality-specific Encoders**: Each modality $m \in \{T1, T1c, T2, FLAIR\}$ has an independent encoder $E_m$, which is globally shared and federated.
2. **Multimodal Fusion Decoder**: Fuses all modality features on the server side, while being partially federated and partially personalized on the client side.
3. **Multi-Anchor Multimodal Representation**: Extracts class anchors from the server's fused features and distributes them to clients.
4. **LACCA Module**: Clients calibrate missing-modality features toward global anchors using cross-attention.

### Modality-specific Encoders (Fully Federated)

Each modality uses an independent encoder for feature extraction, allowing high parameter specialization. The server aggregates the encoder parameters of the same modality: $W_m^s = \frac{1}{N_m} \sum_i W_m^i$. The server-side fusion decoder bridges the distribution gap among different modalities via backpropagation.

### Partially Personalized Fusion Decoder

Core idea: Dynamically determine which filters in the decoder are federated (shared) and which are personalized (retained) based on the consistency between global and local parameter updates.

- **Client aggregation**: $W_d^{i,agg} = (1 - B^{i,r-1}) W_d^{i,r-1} + B^{i,r-1} W_d^{s,r-1}$
- **Consistency judgment**: Calculate the cosine similarity of global/local updates for each filter: $\delta_j^{i,r} = \cos(\Delta \mathbf{w}_j^{s,r}, \Delta \mathbf{w}_j^{i,r})$
- **Personalization rule**: If a filter has $\delta_j^{i,r} < 0$ for $P$ consecutive rounds, it is permanently personalized.
- **Server aggregation**: Uses an EMA strategy to balance the contributions of the server and clients, with $\lambda$ dynamically set to 1 (fully personalized) or 0.3 (others).

### Multi-Anchor Multimodal Representation + LACCA

- The server extracts $N_k = 4$ anchors per class from the fused features, clustering them via K-means and smoothing updates with EMA ($\omega = 0.999$).
- Clients calibrate local missing-modality features toward global anchors using scaled dot-product cross-attention:
$$F_l^{cal} = \text{softmax}\left[\frac{F_l W_0 (A_l W_1)^T}{\sqrt{C_l}}\right] A_l W_2$$

### Loss & Training

Dice loss + cross-entropy loss (standard medical image segmentation loss), Adam optimizer, lr=0.0002.

## Key Experimental Results

**BraTS 2018 (285 cases, divided into 9 sites, 8 clients + 1 server)**:

| Method | Average Client mDSC (%) | Server mDSC (%) |
|------|-------------------|----------------|
| Local models | 66.95 | 82.56 |
| FedAvg | 59.04 | 80.10 |
| FedMSplit | 71.23 | 79.93 |
| FedIoT | 69.18 | 84.89 |
| CreamFL* | 67.21 | 82.83 |
| **FedMEPD** | **75.70** | **84.98** |

**BraTS 2020 (369 cases)**:

| Method | Average Client mDSC (%) | Server mDSC (%) |
|------|-------------------|----------------|
| FedMSplit | 73.80 | 86.88 |
| FedIoT | 71.20 | 88.77 |
| **FedMEPD** | **75.90** | **89.39** |

- The average client mDSC outperforms the second-best method by 4.47% (BraTS 2018) and 2.10% (BraTS 2020).
- Single-modality clients (e.g., T1c only) show the most significant improvement: 58.87% vs. 48.99% of FedMSplit.
- *CreamFL requires data sharing, which violates privacy constraints.

## Highlights & Insights

- Simultaneously optimizes both the global full-modality model and personalized missing-modality client models, balancing dual objectives.
- The partially personalized strategy is dynamically determined based on the consistency of parameter updates, offering clear theoretical intuition.
- Multi-anchor representation + cross-attention calibration: Only abstract population-level prototypes are transmitted, preserving privacy while compensating for missing modality information.
- Compared to a fully personalized decoder (prior work), the partial federated strategy significantly improves client performance.
- The framework is model-agnostic and can be adapted to various multimodal segmentation backbones.

## Limitations & Future Work

- Assumes the existence of a server with full-modality data, which may be difficult to satisfy in practice.
- Once a filter is marked as personalized, it is irreversible, which may prematurely lock certain parameters.
- Communication cost analysis is insufficient (although tiny mask transmission overhead is mentioned).
- Only validated on the brain tumor segmentation task; other multimodal medical tasks (e.g., cardiac, liver) remain unexplored.
- The number of clients is relatively small (8), and scalability under large-scale scenarios has not been validated.

## Related Work & Insights

- **FedAvg** (McMahan et al., 2017): Classic FL baseline, does not handle modality heterogeneity.
- **FedMSplit** (Chen & Zhang, 2022): Multimodal FL but lacks a personalization mechanism.
- **FedNorm** (Bernecker et al., 2022): Adjusts only normalization parameters to handle modality differences, which is insufficient for high heterogeneity.
- **RFNet** (Ding et al., 2021): Centralized multimodal segmentation method, which serves as the backbone network of the proposed framework.
- **CreamFL** (Yu et al., 2023): Requires sharing multimodal data, which violates privacy constraints.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (Novel combined design of modality-specific encoders, partially personalized decoder, and multi-anchor calibration)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Two benchmarks, comparisons with multiple FL baselines, thorough ablation studies, and statistical significance tests)
- **Writing Quality**: ⭐⭐⭐⭐ (Complete structure and clear description of algorithms)
- **Value**: ⭐⭐⭐⭐⭐ (Addresses real pain points in multimodal FL with significant clinical relevance)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Uni-Encoder Meets Multi-Encoders: Representation Before Fusion for Brain Tumor Segmentation with Missing Modalities](../../CVPR2026/medical_imaging/uni-encoder_meets_multi-encoders_representation_before_fusion_for_brain_tumor_se.md)
- [\[ICLR 2026\] Sequential Information Bottleneck Fusion: Towards Robust and Generalizable Multi-Modal Brain Tumor Segmentation](../../ICLR2026/medical_imaging/sequential_information_bottleneck_fusion_towards_robust_and_generalizable_multi-.md)
- [\[CVPR 2025\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2025\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[CVPR 2025\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)

</div>

<!-- RELATED:END -->
