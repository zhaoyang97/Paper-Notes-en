---
title: >-
  [Paper Note] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Federated Learning] This paper proposes FedMEPD, a framework that addresses two major challenges in federated multimodal brain tumor segmentation — inter-modality heterogeneity and client personalization — through modality-specific encoders (fully federated), a partially personalized multimodal fusion decoder, and a multi-anchor cross-attention calibration module. FedMEPD surpasses existing federated methods on BraTS 2018/2020.
tags:
  - CVPR 2026
  - Medical Imaging
  - Federated Learning
  - Multimodal Brain Tumor Segmentation
  - Modality Heterogeneity
  - Personalized Federated Learning
  - Cross-modal Calibration
date: 2026-05-08
content_hash: 2da86eacfac4904d
---

# Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.04887](https://arxiv.org/abs/2603.04887)
**Code**: [https://github.com/ccarliu/FedMEPD](https://github.com/ccarliu/FedMEPD)
**Area**: Medical Imaging
**Keywords**: Federated Learning, Multimodal Brain Tumor Segmentation, Modality Heterogeneity, Personalized Federated Learning, Cross-modal Calibration

## TL;DR

This paper proposes FedMEPD, a framework that addresses two major challenges in federated multimodal brain tumor segmentation — inter-modality heterogeneity and client personalization — through modality-specific encoders (fully federated), a partially personalized multimodal fusion decoder, and a multi-anchor cross-attention calibration module. FedMEPD surpasses existing federated methods on BraTS 2018/2020.

## Background & Motivation

1. **Background**: Federated learning (FL) is increasingly important in medical image analysis, enabling collaborative model training across institutions without privacy leakage. However, most existing FL methods address only intra-modality heterogeneity (e.g., distribution shifts), while neglecting inter-modality heterogeneity in multimodal data.
2. **Limitations of Prior Work**: Brain tumor segmentation requires four complementary MRI modalities — T1, T1c, T2, and FLAIR. In practice, different institutions may possess only a subset of these modalities, introducing inter-modality heterogeneity across participants and rendering conventional approaches such as FedAvg ineffective for training a globally capable model.
3. **Key Challenge**: There exists an inherent tension between optimizing a global server model that performs best under full-modality input and enabling each client to maintain a personalized model adapted to its specific modality combination. These two objectives are rarely addressed simultaneously in FL.
4. **Goal**: (a) How to handle inter-modality heterogeneity arising from different clients possessing different modality subsets? (b) How to achieve effective client personalization while sharing generalizable knowledge through federation?
5. **Key Insight**: The authors observe that different MRI modalities exhibit substantially different distributions (e.g., T1/T1c highlight tumor core, T2/FLAIR highlight edema), such that naive parameter averaging degrades performance. They further note that the decoder constitutes the majority of model parameters — full personalization hinders knowledge sharing, while full federation degrades performance due to distributional mismatch.
6. **Core Idea**: Modality-specific encoders handle inter-modality heterogeneity; a dynamic, partially personalized decoder based on parameter update consistency balances knowledge sharing and personalization; a multi-anchor multimodal representation with cross-attention calibration compensates for information gaps caused by missing modalities.

## Method

### Overall Architecture

FedMEPD takes multimodal MRI volumes as input. The server holds complete data across all four modalities, while each client may possess only a subset (1–4 modalities). The framework consists of three core components: (1) modality-specific encoders (one per modality, fully federated); (2) a multimodal fusion decoder (partially federated, partially personalized); and (3) a Locally Adaptive Cross-attention Calibration (LACCA) module. Both the server and clients maintain encoders and decoders. In each communication round, encoder parameters are fully synchronized, while decoder parameters are dynamically partitioned into federated and personalized subsets based on update consistency.

### Key Designs

1. **Federated Modality-specific Encoders**

   - **Function**: Assigns an independent feature extractor to each MRI modality, enabling large-scale parameter specialization.
   - **Mechanism**: A late-fusion strategy is adopted. Each modality $m$ has a dedicated encoder $E_m$, and the same-modality encoders are fully federated across all clients and the server. The server aggregates gradients from the fusion decoder to indirectly optimize each modality encoder. Clients retain only the encoders corresponding to their available modalities. At each communication round, the encoder for modality $m$ is aggregated by averaging over all clients possessing that modality: $W_m^s = \frac{1}{N_m}\sum_i W_m^i$.
   - **Design Motivation**: The distribution discrepancy among MRI modalities is severe; normalization-layer specialization alone (e.g., FedNorm) is insufficient to handle such heterogeneity. Modality-specific encoders allow greater parameter specialization while facilitating knowledge transfer across clients sharing the same modality.

2. **Partially Personalized Fusion Decoder**

   - **Function**: Strikes a balance between universal knowledge sharing and client-specific personalization, avoiding overfitting from full personalization and performance degradation from full federation.
   - **Mechanism**: Personalization decisions are made at the filter level. The cosine similarity between server and client decoder parameter updates is computed as $\delta_j^{i,r} = \cos(\Delta \mathbf{w}_j^{s,r}, \Delta \mathbf{w}_j^{i,r})$. If a filter's update direction consistently opposes the global update for $P$ consecutive rounds ($\delta < 0$), it is marked as personalized. The aggregation rule is $W_d^{i,agg} = (1 - B^{i,r-1})W_d^{i,r-1} + B^{i,r-1}W_d^{s,r-1}$, where $B$ is a binary mask. The server aggregates via EMA: $W_d^{s,agg} = \lambda W_d^{s,r-1} + (1-\lambda)W_d^{loc,r}$, with a normalization term $H$ introduced to reduce client bias.
   - **Design Motivation**: Once a filter is marked as personalized, it is not reverted to the federated state, preventing training instability. The filter is chosen as the minimal unit of personalization because convolutional filters typically encode specific feature patterns; personalizing at this granularity preserves learned representations while incurring negligible communication overhead (only 1 byte per filter to encode its state).

3. **Multi-Anchor Multimodal Representation + LACCA**

   - **Function**: Compensates for information loss due to missing modalities by aligning client local features toward the server's full-modality representations.
   - **Mechanism**: The server extracts class-specific features from multi-scale fusion decoder outputs via masked average pooling, then applies K-means clustering to obtain $N_k=4$ anchors per class, updated via EMA: $\bar{a}_c = \omega \bar{a}_c + (1-\omega)a_c$. These anchors are distributed to clients, where the LACCA module performs scaled dot-product cross-attention using local client features as queries and multimodal anchors as keys and values: $F_l^{cal} = \text{softmax}[F_l W_0 (A_l W_1)^T / \sqrt{C_l}] A_l W_2$. Calibration is applied at all four feature scales.
   - **Design Motivation**: A single prototype is overly compressed and insufficient to capture intra-class variation in 3D multimodal medical images. Multiple anchors preserve richer full-modality information while being abstracted at the population level, avoiding individual privacy leakage. The cross-attention mechanism enables each client to adaptively select the most relevant anchor components based on its own modality configuration and data distribution, offering greater flexibility than uniform calibration.

### Loss & Training

A standard combination of Dice loss and cross-entropy loss is used. Training spans 1,000 federated communication rounds, with 1 local epoch per round on both server and clients. The Adam optimizer is used with a learning rate of 0.0002 and weight decay of $10^{-5}$. Input patches are cropped to $80 \times 80 \times 80$ with a batch size of 1.

## Key Experimental Results

### Main Results

FedMEPD is compared against multiple FL methods on BraTS 2018 and BraTS 2020:

| Method | BraTS 2018 Avg mDSC (%) | BraTS 2018 Server mDSC (%) | BraTS 2020 Avg mDSC (%) | BraTS 2020 Server mDSC (%) |
|---|---|---|---|---|
| Local models | 66.95 | 82.56 | 71.38 | 88.07 |
| FedAvg | 59.04 | 80.10 | 61.91 | 87.61 |
| FedMSplit | 71.23 | 79.93 | 73.80 | 86.88 |
| CreamFL | 67.21 | 82.83 | 67.09 | 87.69 |
| FedIoT | 69.18 | 84.89 | 71.20 | 88.77 |
| **Ours** | **75.70** | **84.98** | **75.90** | **89.39** |

Consistent advantages are also observed in HD95: on BraTS 2018, Avg HD95 is reduced from 18.01 (second-best) to 12.98; on BraTS 2020, FedMEPD achieves the best Avg HD95 of 13.41.

### Ablation Study

| Configuration | BraTS 2020 Avg mDSC (%) | Note |
|---|---|---|
| Full model (FedMEPD) | 75.90 | Complete model |
| w/o Partial personalization (full personalization) | ~72 | Insufficient knowledge sharing |
| w/o LACCA | ~73 | Missing modality information not compensated |
| w/o Multi-anchor (single prototype) | ~74 | Single prototype lacks sufficient richness |
| FedNorm (normalization specialization only) | 63.09 | BN-layer specialization far insufficient |

### Key Findings

- Modality-specific encoders are fundamental to handling inter-modality heterogeneity, with the greatest gains observed for single-modality clients (e.g., the T1c-only client improves from 18.46% under FedAvg to 58.87%).
- The partially personalized decoder yields clear improvements over full personalization, confirming the importance of federated knowledge sharing.
- Multiple anchors ($N_k=4$) outperform single prototypes; further increasing anchor count yields diminishing returns.
- LACCA provides greater benefit to clients with more missing modalities and has minimal impact on full-modality clients.
- FedMEPD achieves statistical significance ($p < 0.05$) over competing methods in most comparisons, for both Avg and Server metrics.

## Highlights & Insights

- The **filter-level dynamic personalization** strategy is particularly elegant: by measuring consistency of parameter update directions, the framework automatically identifies which parameters are sensitive to data heterogeneity and which can be safely shared — a principle readily generalizable beyond modality heterogeneity to any FL setting with data heterogeneity.
- **Multi-anchor representation** is both more informative and more privacy-preserving than conventional prototype transfer: multiple cluster centroids per class retain richer distributional information than a single mean, while remaining population-level abstractions that do not expose individual data.
- Applying **cross-attention** for missing-modality feature calibration is a natural and principled design — each client adaptively attends to the most relevant full-modality anchors to compensate for its specific information deficit.

## Limitations & Future Work

- The framework assumes the server holds complete full-modality data, which may not be realistic in all deployment scenarios. Extension to settings where the server also has only partial modalities warrants further investigation.
- Experiments are conducted solely on brain tumor segmentation; generalization to other multimodal medical imaging tasks (e.g., cardiac or abdominal organ segmentation) remains to be validated.
- The number of clients is fixed at 8; scalability to large-scale federated settings has not been evaluated.
- The patience parameter $P$ and EMA coefficient $\lambda$ for partial personalization require manual tuning; automated hyperparameter adaptation strategies merit future study.

## Related Work & Insights

- **vs. FedAvg**: FedAvg averages all parameters indiscriminately and cannot accommodate inter-modality heterogeneity. FedMEPD replaces naive full-parameter averaging with modality-specific encoders and a partially personalized decoder.
- **vs. FedNorm**: FedNorm specializes only normalization-layer parameters, which experiments show is far insufficient for handling the substantial distributional differences across MRI modalities.
- **vs. FedMSplit**: FedMSplit is the closest competitor in the multimodal FL setting, but does not address missing-modality calibration or partial personalization. FedMEPD outperforms it by approximately 2–4 percentage points in Avg mDSC.
- **vs. CreamFL**: CreamFL requires sharing server data with all clients, which conflicts with medical privacy constraints.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Organically integrates modality-specific encoders, filter-level dynamic personalization, and multi-anchor calibration to address two long-overlooked challenges in federated multimodal medical imaging.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two BraTS benchmarks, 10 comparison methods, and comprehensive ablation studies; limited to a single brain tumor segmentation task.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete mathematical formulations, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Provides important reference for real-world deployment of federated learning in multimodal medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[CVPR 2026\] Unlocking Multi-Site Clinical Data: A Federated Approach to Privacy-First Child Autism Behavior Analysis](unlocking_multi-site_clinical_data_a_federated_approach_to_privacy-first_child_a.md)
- [\[CVPR 2026\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)

</div>

<!-- RELATED:END -->
