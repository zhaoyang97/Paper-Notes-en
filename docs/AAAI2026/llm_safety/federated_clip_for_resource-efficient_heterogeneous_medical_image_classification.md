---
title: >-
  [Paper Note] Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification
description: >-
  [AAAI 2026][LLM Safety][Federated Learning] This paper proposes FedMedCLIP, a federated CLIP framework for medical image classification. By freezing the CLIP encoder and combining a masked Feature Adaptation Module (FAM)…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Federated Learning"
  - "CLIP"
  - "Medical Image Classification"
  - "Data Heterogeneity"
  - "Parameter Efficiency"
date: 2026-05-08
content_hash: 3d4ba999d5a418a4
---

# Federated CLIP for Resource-Efficient Heterogeneous Medical Image Classification

**Conference**: AAAI 2026
**arXiv**: [2511.07929](https://arxiv.org/abs/2511.07929)  
**Code**: [github](https://github.com/AIPMLab/FedMedCLIP)  
**Area**: Medical Imaging
**Keywords**: Federated Learning, CLIP, Medical Image Classification, Data Heterogeneity, Parameter Efficiency

## TL;DR

This paper proposes FedMedCLIP, a federated CLIP framework for medical image classification. By freezing the CLIP encoder and combining a masked Feature Adaptation Module (FAM), a local masked MLP, and class-level KL distillation regularization, the framework achieves robust classification under data heterogeneity with minimal communication and computational overhead (surpassing the second-best method by 8% on ISIC2019 and running 120× faster than FedAVG).

## Background & Motivation

### Challenges of Federated Learning in Medical Imaging

Deep learning has demonstrated strong performance in medical imaging, but data privacy constraints hinder cross-institutional collaboration. Federated learning (FL) offers a distributed training paradigm, yet faces two major challenges:

**Data Heterogeneity**: Differences in devices, modalities, and disease distributions across hospitals introduce feature shifts, causing global model performance to degrade on local clients.

**Resource Overhead**: VLM-class models such as CLIP (~$10^8$ parameters) impose substantial communication and computation costs, making deployment on resource-constrained devices difficult.

### Difficulty of Adapting CLIP to the Medical Domain

- Vanilla CLIP performs poorly on medical data (e.g., only 24.1% accuracy on ISIC2019) due to the large domain gap between pretraining data and medical imagery.
- The recall of direct CLIP inference for skin cancer detection is approximately 50%.
- Existing PEFT methods (e.g., FedCLIP, PromptFL): (a) have not been validated on medical data, (b) do not address modality-level heterogeneity, and (c) may lose client-specific features through global aggregation.

### Core Problem

How to efficiently adapt CLIP to federated medical settings, achieving reliable performance under heterogeneous data with reasonable overhead?

## Method

### Overall Architecture

The framework operates in a three-stage loop:

1. **Local Training and Inference**: Frozen CLIP encoder extracts features → FAM generates masked features → FAM is trained with contrastive loss; local MLP is trained with CE loss; KL regularization aligns the two.
2. **Model Compression and Transmission**: FAM parameters are converted to float16 and compressed with zlib before being uploaded to the server.
3. **Global Aggregation**: FAM parameters are aggregated via simple averaging; MLP parameters remain local and private.

### Key Designs

#### 1. **Masked Feature Adaptation Module (Masked FAM)**

FAM applies an attention mask to CLIP image features, with the core idea of learning **sparse yet dominant feature representations**:

- Input image features: $\mathbf{I} = e_I(\mathbf{x}) \in \mathbb{R}^D$
- FAM generates mask $att_i(\mathbf{I}) \in [0,1]^D$; masked features: $\tilde{\mathbf{I}} = att_i(\mathbf{I}) \otimes \mathbf{I}$
- Mask generation: the mean magnitude $u_i$ of each layer's weights is computed and compared against a learnable threshold $\kappa_i$ to produce a binary mask:

$$m_i = \mathcal{S}(u_i - \kappa_i) = \begin{cases} 1 & \text{if } u_i \geq \kappa_i \\ 0 & \text{if } u_i < \kappa_i \end{cases}$$

- Masked linear layer: $\hat{W} = W \odot (\mathcal{M} \cdot \mathbf{1}^T), \; y = \hat{W}x + (b \odot \mathcal{M})$

**Design Motivation**: The masking mechanism directs FAM to focus on dominant feature dimensions shared across clients, reducing noise propagation. Furthermore, FAM contains only ~$5 \times 10^5$ parameters (far fewer than CLIP's $10^8$), substantially reducing communication overhead.

#### 2. **Masked Local MLP Classifier**

- Each client maintains a private masked MLP using the same masked linear layer structure.
- The MLP is not uploaded for aggregation, thereby preserving client-specific patterns and class distribution information.
- Trained with standard cross-entropy loss: $\mathcal{L}_{MLP} = -\frac{1}{B}\sum_{j=1}^{B}\mathcal{L}_{CE}(p_j, y)$

**Design Motivation**: Global aggregation may weaken client-specific features learned by the FAM; the private MLP is therefore introduced to capture local task characteristics. The two components are complementary: FAM learns globally shared features while the MLP learns locally specific ones.

#### 3. **Class-Level KL Distillation Regularization**

FAM and the MLP mutually learn from each other, with class-level information incorporated to prevent misalignment under heterogeneous data:

$$\mathcal{L}_{sim} = \frac{1}{C}\sum_{c=1}^{C}\sum_{i=1}^{B}\left(\varpi \hat{q}_i^{(c)}\log\frac{\hat{q}_i^{(c)}}{\hat{p}_i^{(c)}} + (1-\varpi)\hat{p}_i^{(c)}\log\frac{\hat{p}_i^{(c)}}{\hat{q}_i^{(c)}}\right)$$

The dynamic weight $\varpi = \frac{\mathcal{H}(p^v)}{\mathcal{H}(p^m) + \mathcal{H}(p^v)}$ adaptively balances the knowledge transfer direction (FAM→MLP vs. MLP→FAM) based on prediction entropy.

**Design Motivation**: When FAM is uncertain (high entropy), the MLP provides stronger guidance to FAM, and vice versa.

#### 4. **Ensemble Prediction and Model Compression**

- Inference: $p^{ens} = \varpi \cdot p^{MLP} + (1-\varpi) \cdot p^{FAM}$
- Compression: float32 → float16 + zlib reduces model size from 2.01 MB to 1.36 MB with negligible performance loss (80.48% → 80.4%).

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_{contr} + \mathcal{L}_{MLP} + \lambda \cdot \mathcal{L}_{sim}$

- $\mathcal{L}_{contr}$: image-text contrastive loss for training FAM
- $\mathcal{L}_{MLP}$: cross-entropy loss for training the local MLP
- $\lambda = 0.04$, temperature $T=2$
- AdamW optimizer with exponential learning rate scheduling (gamma = 0.97)
- Communication rounds: 100 for ISIC2019, 50 for other datasets; 1 local epoch per round

## Key Experimental Results

### Main Results

ISIC2019 skin cancer classification (7 clients partitioned by anatomical site; feature heterogeneity):

| Method | C₁ | C₂ | C₃ | C₄ | C₅ | C₆ | Global | AVG |
|------|-----|-----|-----|-----|-----|-----|--------|-----|
| CLIP zero-shot | 31.95 | 23.98 | 24.31 | 17.71 | 20.12 | 33.47 | 17.17 | 24.1 |
| FedAVG | 78.35 | 60.47 | 76.39 | 58.33 | 75.00 | 71.77 | 84.54 | 72.12 |
| FedAPT | 77.15 | 51.09 | 73.80 | 67.65 | 81.70 | 68.60 | 85.43 | 72.21 |
| **Ours** | **84.45** | **71.92** | **82.69** | **84.31** | **79.00** | **79.40** | 81.01 | **80.4** |

Resource efficiency comparison:

| Method | ISIC2019 Compute (min) | ISIC2019 Comm. (GB) | BraTS Compute (min) | BraTS Comm. (GB) |
|------|-------------------|------------------|----------------|----------------|
| FedAVG | 95.58 | 7.569 | 28.19 | 3.03 |
| FedAPT | 72.50 | 0.037 | 4.23 | 0.003 |
| **Ours** | **68.56** | **0.063** | **2.85** | **0.012** |

### Ablation Study

| $\mathcal{L}_{contr}$ | $\mathcal{L}_{MLP}$ | $\mathcal{L}_{sim}$ | Aggregation | AVG |
|:---:|:---:|:---:|:---:|:---:|
| ✓ | ✗ | ✗ | ✓ | 71.94 |
| ✓ | ✓ | ✗ | ✓ | 78.44 |
| ✗ | ✓ | ✗ | ✗ | 67.77 |
| ✓ | ✓ | ✓ | ✗ | 70.41 |
| ✓ | ✓ | ✓ | ✓ | **80.4** |

### Key Findings

1. **Vanilla CLIP performs poorly on medical data**: Zero-shot accuracy of only 24.1% confirms a large domain gap.
2. **MLP contributes most significantly**: Introducing $\mathcal{L}_{MLP}$ improves AVG from 71.94% to 78.44% (+6.5%).
3. **KL regularization yields further gains**: +1.96% AVG, though minor degradation may occur on individual clients with severe class imbalance.
4. **Substantial resource efficiency**: 120× faster and 120× less communication than FedAVG, with computation time comparable to FedAPT.
5. **Stability at scale**: On the ICH dataset with 5→10→15 clients, the proposed method consistently achieves AVG >66%, whereas FACMIC degrades by ~5%.
6. **Stronger adversarial robustness**: Under FGSM attack, AVG reaches 33.85%, outperforming FedCLIP (24%) and LoRA (23%).

## Highlights & Insights

1. **Dual-path global-local design**: FAM aggregates global knowledge while the MLP retains local characteristics; mutual learning via KL distillation forms an elegant complementary structure.
2. **Dual role of the masking mechanism**: It simultaneously induces parameter sparsity (reducing communication and computation costs) and encourages the model to focus on dominant features (improving generalization).
3. **Entropy-based dynamic weight**: The design of $\varpi$ is concise and effective, adaptively balancing the knowledge flow between the two models.
4. **Compression with negligible performance loss**: float16 + zlib compression reduces model size from 2.01 MB to 1.36 MB with only 0.08% AVG degradation.
5. **Genuine medical-domain validation**: Unlike most FL+CLIP works that only evaluate on natural image benchmarks such as OfficeHome, this work validates on real medical datasets.

## Limitations & Future Work

1. **Limited global generalization**: Under modality-level heterogeneity (e.g., different MRI modalities in BraTS), global ACC is lower than FedAPT, suggesting the CLIP encoder's prior knowledge is insufficient for handling medical modality gaps.
2. **Reliance on generic prompt templates**: The standard prompt "a picture of a {class}" is used without leveraging domain-specific medical prompt knowledge.
3. **Mask threshold $\kappa$ initialization and training**: The paper does not discuss gradient handling for the sign function during backpropagation (likely using STE).
4. **Classification tasks only**: The framework has not been extended to more complex medical tasks such as segmentation and detection.
5. **No formal privacy guarantees**: Although FAM parameters are small, whether they may leak partial feature information has not been analyzed.

## Rating

- Novelty: ⭐⭐⭐⭐ — The dual-path FAM+MLP design with mutual KL distillation is novel and well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four medical datasets, eight baselines, and multi-faceted analysis covering robustness, scalability, and statistical significance.
- Writing Quality: ⭐⭐⭐⭐ — Well-organized with complete derivations.
- Value: ⭐⭐⭐⭐ — Provides a practical solution for deploying federated CLIP in real-world medical settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ICCV 2025\] Cooperative Pseudo Labeling for Unsupervised Federated Classification](../../ICCV2025/llm_safety/cooperative_pseudo_labeling_for_unsupervised_federated_classification.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](../../ICLR2026/llm_safety/she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[ICLR 2026\] Heterogeneous Federated Fine-Tuning with Parallel One-Rank Adaptation](../../ICLR2026/llm_safety/heterogeneous_federated_fine-tuning_with_parallel_one-rank_adaptation.md)
- [\[AAAI 2026\] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
