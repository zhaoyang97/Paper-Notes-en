---
title: >-
  [Paper Note] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Federated Learning] This paper proposes FedMEPD, a framework that employs modality-specific encoders to address intermodal heterogeneity…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Federated Learning"
  - "Multimodal Brain Tumor Segmentation"
  - "Intermodal Heterogeneity"
  - "Personalized FL"
  - "Cross-Attention Calibration"
date: 2026-05-08
content_hash: 3aa27f74a88551fe
---

# Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.04887](https://arxiv.org/abs/2603.04887)
**Code**: [GitHub](https://github.com/ccarliu/FedMEPD)
**Area**: Medical Imaging
**Keywords**: Federated Learning, Multimodal Brain Tumor Segmentation, Intermodal Heterogeneity, Personalized FL, Cross-Attention Calibration

## TL;DR

This paper proposes FedMEPD, a framework that employs modality-specific encoders to address intermodal heterogeneity, a filter-level dynamic partial personalization decoder to balance knowledge sharing and personalization, and a multi-anchor cross-attention calibration module to compensate for missing modality information. FedMEPD comprehensively outperforms existing multimodal federated learning methods on BraTS 2018/2020.

## Background & Motivation

**Background**: Federated learning (FL) enables multiple medical institutions to collaboratively train models without sharing private data. Brain tumor segmentation relies on four MRI modalities—T1, T1c, T2, and FLAIR—which provide complementary information (the former two highlight the tumor core, while the latter two highlight peritumoral edema).

**Limitations of Prior Work**: In practice, different medical institutions may possess only a subset of modalities (e.g., small clinics may only have T1), resulting in severe **intermodal heterogeneity** among FL participants. The vast majority of existing medical imaging FL methods address only intra-modality data heterogeneity (non-IID distributions) and cannot effectively handle modality-missing scenarios.

**Key Challenge**: FL must simultaneously achieve two objectives: (1) training an optimal global model for full-modality inputs (server-side), and (2) providing personalized models for clients that possess only a subset of modalities. These two objectives are fundamentally in tension under modality heterogeneity: fully federated aggregation is disrupted by heterogeneous modalities, while full personalization impedes knowledge sharing.

**Goal**: To leverage heterogeneous multimodal data effectively for global model training while providing missing-modality compensation and personalized adaptation for clients—all under privacy constraints.

**Key Insight**: The network is decomposed into modality-specific encoders (fully federated) and a multimodal fusion decoder (partially federated, partially personalized), augmented with multi-anchor multimodal representations that calibrate missing-modality features via cross-attention.

**Core Idea**: A filter-level dynamic binary mask, derived from the directional consistency of parameter updates, enables federated sharing of decoder parameters on which global and local clients agree, while preserving personalized parameters where local divergence exists. Concurrently, multi-anchor full-modality representations maintained on the server compensate for missing modalities on clients via cross-attention.

## Method

### Overall Architecture

FedMEPD consists of three major components:

- **Server** (assumed to have full-modality data): four modality-specific encoders $E_m$ (one per modality) and a multimodal fusion decoder $D_M$. Fused features are clustered via K-means to generate multi-anchor representations, which are distributed to clients together with model parameters.
- **Clients** (possessing 1–4 modality subsets): federated encoders for the available modalities, a partially personalized fusion decoder $D_m$, and a LACCA calibration module. Encoders are fully federated; the decoder employs a dynamic binary mask $B^i$ to determine which filters are federated and which are personalized.
- **Backbone**: RFNet (Ding et al., 2021) is adopted as the backbone, as it natively supports the separation of modality-specific encoders and a fusion decoder. Encoders additionally share an auxiliary segmentation decoder for regularization.

### Key Designs

1. **Federated Modality-specific Encoders**

    - **Function**: Learn dedicated feature representations for each MRI modality, fully accommodating the pronounced distributional differences across modalities.
    - **Mechanism**: Each modality $m$ has an independent encoder $E_m$ whose parameters $W_m^s$ are fully federated between the server and clients. In each FL round, clients receive global parameters to replace their local copies, train locally, and upload updates; the server averages parameters across clients for the same modality: $W_m^s = \frac{1}{N_m}\sum_i W_m^i$.
    - **Design Motivation**: FedAvg uses a shared encoder for all modalities, but T1/T1c/T2/FLAIR exhibit vastly different distributions, causing shared parameters to interfere with one another. Modality-specific encoders allow full parameter specialization. Ablation studies confirm this is the largest single contributor to performance gain (client mean mDSC improves from 55.37% to 68.70%, +13.33%).

2. **Partially Personalized Fusion Decoder**

    - **Function**: Dynamically balance global knowledge sharing and client-side personalization.
    - **Mechanism**: Based on the directional consistency between global and local parameter updates, the federated/personalized status of each filter is determined dynamically at the filter level. Specifically, the cosine similarity between server and client parameter updates for filter $j$ is computed as $\delta_j^{i,r} = \cos(\Delta \mathbf{w}_j^{s,r}, \Delta \mathbf{w}_j^{i,r})$. If the cosine similarity remains negative for $P$ consecutive rounds (patience), the filter is irreversibly converted to personalized. The aggregation rule is: $W_d^{i,agg} = (1-B^{i,r-1})W_d^{i,r-1} + B^{i,r-1}W_d^{s,r-1}$, where $B$ is the dynamic binary mask. The server aggregates using an EMA strategy, with $\lambda$ switching between 0.3 and 1.0 according to the personalization status of each filter, and a normalization term $H^{i,r}$ is introduced to mitigate client bias.
    - **Design Motivation**: A fully personalized decoder impedes knowledge sharing (client mean of only 68.70% at $P=0$), while a fully federated decoder is disrupted by heterogeneous modalities (mDSC 68.49%). Filter-level operation preserves the integrity of feature detection with minimal communication overhead (only 1 byte per filter for status encoding).

3. **Multi-Anchor Multimodal Representation**

    - **Function**: Extract class-level representations from server-side full-modality fused features and distribute them to clients to compensate for missing modality information.
    - **Mechanism**: Ground-truth masks are used to extract per-class features from the feature maps of fusion decoder $D_M$, and K-means clustering is applied to each class to obtain $N_k=4$ anchors (rather than a single prototype). Cluster membership is determined based on the most abstract feature level $l=4$ (bottleneck layer), and anchors are computed separately for all four feature scales. Anchors are smoothed via EMA ($\omega=0.999$) to prevent cluster instability.
    - **Design Motivation**: 3D medical images exhibit large inter-subject variability, making a single prototype overly compressed ($N_k=1$: client mean 71.19%; $N_k=4$: 72.84%). Multi-anchor representations serve as population-level abstractions that do not leak individual privacy and impose negligible transmission overhead.

4. **LACCA Module (Localized Adaptive Calibration via Cross-Attention)**

    - **Function**: Clients adaptively calibrate missing-modality feature representations using multimodal anchors via cross-attention.
    - **Mechanism**: Local feature maps $F_l$ serve as queries, while multimodal anchors $A_l$ serve as keys and values, processed through scaled dot-product cross-attention: $F_l^{cal} = \text{softmax}\left[\frac{F_l W_0 (A_l W_1)^T}{\sqrt{C_l}}\right] A_l W_2$. Eight-head attention is used, inserted at all four feature scale levels of the decoder.
    - **Design Motivation**: Different clients are missing different modalities, necessitating adaptive extraction of the most relevant information from full-modality anchors given each client's modality combination. LACCA operates entirely on the client side; at inference, the pre-trained anchors are used directly.

### Loss & Training

- **Loss Function**: Dice Loss + Cross-Entropy Loss (standard for medical segmentation).
- **Optimizer**: Adam, learning rate 0.0002, weight decay $10^{-5}$.
- **Federated Training**: 1000 communication rounds; server and each client train for 1 epoch per round.
- **Input**: $80 \times 80 \times 80$ voxel crops, batch size = 1.
- **Regularization**: All encoders share an auxiliary segmentation decoder to enforce consistent discriminative feature learning.
- **Hardware**: 5 RTX 2080Ti GPUs (1 for the server, 4 for clients).

## Key Experimental Results

### Main Results

Experiments are conducted on BraTS 2018 (285 cases) and BraTS 2020 (369 cases), comparing against a local baseline, RFNet, and 8 FL state-of-the-art methods. Eight clients are configured, with modality combinations ranging from single-modality to full-modality (two clients per combination type).

**BraTS 2018 mDSC (%):**

| Method | Single T1c | Single T2 | Dual F/T1c | Dual T1/T2 | Triple F/T1c/T1 | Triple F/T1/T2 | Full (Client 1) | Full (Client 2) | Client Avg. | Server |
|--------|-----------|----------|-----------|------------|----------------|---------------|----------------|----------------|-------------|--------|
| Local | 42.37 | 48.13 | 87.74 | 64.93 | 71.59 | 63.99 | 89.15 | 67.67 | 66.95 | 82.56 |
| FedAvg | 18.46 | 42.12 | 82.11 | 59.59 | 61.13 | 61.91 | 84.88 | 62.09 | 59.04 | 80.10 |
| FedMSplit | 48.99 | 54.09 | 92.16 | 68.21 | 82.48 | 69.92 | 87.87 | 66.09 | 71.23 | 79.93 |
| FedIoT | 41.97 | 48.33 | 92.35 | 61.69 | 81.81 | 70.66 | 88.31 | 68.36 | 69.18 | 84.89 |
| **FedMEPD** | **58.87** | **59.35** | **93.73** | **75.83** | **82.99** | **74.58** | **90.69** | **69.62** | **75.70** | **84.98** |

**BraTS 2020 mDSC (%):**

| Method | Client Avg. | Server |
|--------|-------------|--------|
| Local | 71.38 | 88.07 |
| FedAvg | 61.91 | 87.61 |
| FedMSplit | 73.80 | 86.88 |
| FedIoT | 71.20 | 88.77 |
| **FedMEPD** | **75.90** | **89.39** |

**BraTS 2018 HD95 (voxels):**

| Method | Client Avg. | Server |
|--------|-------------|--------|
| FedAvg | 23.43 | 14.52 |
| FedMSplit | 18.01 | 12.40 |
| **FedMEPD** | **12.98** | **6.52** |

### Ablation Study

**Incremental Component Addition (BraTS 2018 validation set mDSC %):**

| Configuration | Encoder | Decoder | LACCA | Client Avg. | Server |
|---------------|---------|---------|-------|-------------|--------|
| (a) FedAvg shared E | Shared E | — | — | 55.37 | 82.60 |
| (b) FedAvg federated D | — | Federated D | — | 64.79 | 82.46 |
| (c) Modality-specific E | 4E federated | — | — | 68.70 | 82.72 |
| (d) + Fully federated D | 4E federated | Federated D | — | 68.49 | 83.00 |
| (e) + Partially personalized D | 4E federated | Partial. pers. D | — | 70.73 | 83.83 |
| (f) + Single-anchor LACCA | 4E federated | Partial. pers. D | Single anchor | 71.19 | 83.71 |
| **(h) Full model** | **4E federated** | **Partial. pers. D** | **Multi-anchor** | **72.84** | **83.83** |

**Sensitivity to Patience $P$ (validation set mDSC %):**

| $P$ | Client Avg. | Server |
|-----|-------------|--------|
| 0 (fully personalized) | 68.70 | 82.72 |
| 6 | 72.31 | 83.54 |
| 8 | 72.20 | 83.76 |
| **10** | **72.84** | **83.83** |
| 12 | 71.55 | 83.78 |
| 14 | 72.29 | 83.74 |

**Number of Anchors $N_k$ (validation set mDSC %):**

| $N_k$ | Client Avg. | Server |
|-------|-------------|--------|
| 1 | 71.19 | 83.71 |
| 2 | 71.91 | 83.56 |
| 4 | **72.84** | **83.83** |
| 6 | 71.33 | 83.05 |

**Robustness to Server Data Volume and Quality (BraTS 2018 test set mDSC %):**

| Data Configuration | Client Avg. | Server |
|--------------------|-------------|--------|
| Full server data | 75.70 | 84.98 |
| 50% server data | 74.34 | 82.98 |
| 30% server data | 73.81 | 80.68 |
| 10% server data | 72.81 | 78.30 |
| Label noise (±1px erosion/dilation) | 75.02 | 81.43 |
| FedMSplit (full data, reference) | 71.23 | 79.93 |

### Key Findings

- **Modality-specific encoders are the largest single contributor**: client mean mDSC jumps from 55.37% (FedAvg) to 68.70% (+13.33%), confirming that intermodal heterogeneity is the core bottleneck.
- **Partial personalization strictly outperforms both extremes**: a fully federated decoder (68.49%) and a fully personalized decoder (68.70%) both underperform the partially personalized strategy (70.73%), validating the necessity of balancing knowledge sharing and personalization.
- **FedAvg-family methods underperform even the local baseline** under modality heterogeneity (59.04% vs. 66.95%), demonstrating that naive federated aggregation is harmful in this setting.
- **Extreme robustness to server data volume**: even with only 10% server data (~9 cases), client mean mDSC (72.81%) surpasses all comparison methods using full data (FedMSplit: 71.23%).
- **Robustness to annotation noise**: random ±1-pixel erosion/dilation of server annotations causes only a marginal drop to 75.02%, with no statistically significant difference.

## Highlights & Insights

- The **filter-level dynamic personalization mechanism** is elegantly designed—the binary mask based on parameter update directional consistency automatically discovers what should be shared and what should be personalized, with minimal communication overhead and stable training guaranteed by the irreversibility design.
- **Multi-anchor representation** is a powerful upgrade over single prototypes: just 4 anchors yield a notable improvement (+1.65%) while preserving FL privacy properties as population-level abstractions.
- **The framework imposes modest requirements on server resources**—as little as 10% of full-modality data suffices to effectively drive the entire federated system, which is highly practical in real-world scenarios where large hospitals may not be able to provide large amounts of labeled data.
- The experimental design is comprehensive: varying numbers of clients (4/6), varying modality completeness (1–4 modalities per client), varying server data volume and quality, two datasets, and 10+ comparison methods.

## Limitations & Future Work

- **Assumes server access to full-modality data**: although experiments demonstrate robustness with limited data, fully decentralized scenarios with no full-modality data are not addressed. Peer-to-peer modality complementation mechanisms could be explored.
- **Irreversible personalization mask**: once a filter is personalized, it is permanently locked. In very long training scenarios, this could result in premature solidification of decisions; an "unfreeze" mechanism or periodic re-evaluation could be considered.
- **Validated only on brain tumor segmentation**: generalizability to other multimodal medical imaging tasks (cardiac, abdominal, pathology, etc.) beyond the BraTS dataset has not been verified.
- **Limited client scale**: experiments use at most 8 clients; communication efficiency and convergence at larger federation scales (e.g., hundreds of hospital nodes) remain to be validated.
- **Lack of formal privacy analysis**: while multi-anchor representations serve as population-level abstractions, no formal privacy guarantees such as differential privacy are provided.

## Related Work & Insights

- **Missing-modality segmentation**: centralized methods such as RFNet (Ding et al., 2021) perform well under missing-modality settings but are unsuitable for federated privacy-preserving scenarios; this paper adopts RFNet as the backbone and extends it to the federated setting.
- **Multimodal FL**: FedMSplit (2022) and FedIoT (2022) address modality heterogeneity but lack sufficient personalization; CreamFL (2023) requires shared server data, violating privacy; FedNorm (2022) specializes only normalization parameters, which is insufficient. The modality-specific encoders in this paper provide stronger parameter specialization.
- **Personalized FL**: methods such as perFL and IOP-FL achieve personalization through partial parameter sharing but do not account for intermodal heterogeneity. The filter-level dynamic mask mechanism proposed here is generalizable to other FL scenarios requiring fine-grained personalization.

## Rating

- Novelty: ⭐⭐⭐⭐ — Filter-level cosine-consistency-based dynamic personalization and multi-anchor cross-attention calibration are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two datasets, 10+ comparison methods, 7 ablation groups, and data volume/quality robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous mathematical derivations, and rich figures and tables.
- Value: ⭐⭐⭐⭐ — Addresses a critical bottleneck in multimodal federated medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modal.md)

</div>

<!-- RELATED:END -->
