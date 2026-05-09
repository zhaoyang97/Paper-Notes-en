---
title: >-
  [Paper Note] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers
description: >-
  [ICCV 2025][AI Safety][Data-free quantization] This paper proposes SARDFQ to address **semantic distortion** and **semantic insufficiency** in data-free quantization (DFQ) of ViTs. Attention Prior Alignment (APA) guides synthetic images to match the attention patterns of real images, while Multi-Semantic Reinforcement (MSR) enriches local patch semantics. SARDFQ achieves a 15.52% Top-1 accuracy improvement on ImageNet W4A4 ViT-B.
tags:
  - ICCV 2025
  - AI Safety
  - Data-free quantization
  - Vision Transformer
  - attention prior alignment
  - multi-semantic reinforcement
  - synthetic images
date: 2026-05-08
content_hash: 91f649271b9904b5
---

# Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers

**Conference**: ICCV 2025
**arXiv**: [2412.16553](https://arxiv.org/abs/2412.16553)
**Code**: [https://github.com/zysxmu/SARDFQ](https://github.com/zysxmu/SARDFQ)
**Area**: AI Safety / Model Quantization
**Keywords**: Data-free quantization, Vision Transformer, attention prior alignment, multi-semantic reinforcement, synthetic images

## TL;DR

This paper proposes SARDFQ to address **semantic distortion** and **semantic insufficiency** in data-free quantization (DFQ) of ViTs. Attention Prior Alignment (APA) guides synthetic images to match the attention patterns of real images, while Multi-Semantic Reinforcement (MSR) enriches local patch semantics. SARDFQ achieves a 15.52% Top-1 accuracy improvement on ImageNet W4A4 ViT-B.

## Background & Motivation

**Data-Free Quantization (DFQ)** enables model quantization without access to real data, addressing data privacy and security concerns. However, DFQ for ViTs faces unique challenges:

- **CNN-based DFQ methods** rely on Batch Normalization Statistics (BNS) to synthesize in-distribution data, but ViTs use Layer Normalization with statistics computed dynamically at inference, making BNS unavailable.
- Existing ViT DFQ methods (e.g., PSAQ-ViT) suffer from two major issues:
  1. **Semantic distortion**: The semantics of synthetic images deviate significantly from real images (t-SNE visualizations show severe feature cluster shifts; cosine similarity is only 0.44 vs. 0.68 for real images).
  2. **Semantic insufficiency**: Synthetic images contain large dull regions with monotonous content and overly simplified textures, providing little or even harmful learning signal.

In high-bit quantization, the model retains sufficient capacity and these issues have limited impact. However, under **low-bit quantization** (e.g., W4A4), model capacity is severely degraded, necessitating high-quality synthetic data for performance recovery; low-quality data leads to substantial generalization degradation.

## Method

### Overall Architecture

SARDFQ consists of two stages:
1. **Data synthesis**: Synthetic images are optimized from Gaussian noise using APA + MSR + SL + TV losses.
2. **Quantized network learning**: The quantized model is fine-tuned block-by-block using the synthetic data.

### Key Designs

1. **Attention Prior Alignment (APA)**:

    - **Design Motivation**: Existing methods overlook the intrinsic property of self-attention in ViTs to encode semantic correlations, resulting in disordered and unnatural attention patterns in synthetic images.
    - A **Gaussian Mixture Model (GMM)** is used to randomly generate structured attention priors $\tilde{\mathbf{A}}_{l,h}$.
    - For DeiT: the attention from the classification token to other tokens $\mathbf{A}^c_{l,h}$ is extracted; for Swin: the mean attention across all tokens is used.
    - Alignment is enforced via MSE loss: $\mathcal{L}_{l,h} = \text{MSE}(\mathbf{A}^c_{l,h} - \tilde{\mathbf{A}}_{l,h})$
    - **Depth weighting**: APA is applied only to the latter half of blocks ($S = L/2$), since shallow layers capture low-level information while deeper layers capture semantics.
    - The total loss includes a depth scaling factor $l/L$: $\mathcal{L}^{\text{APA}} = \sum_{l=S}^{L}\sum_{h=1}^{H}\frac{l}{L}\mathcal{L}_{l,h}$
    - Each head uses a distinct GMM, simulating the characteristic that different heads capture different patterns.

2. **Multi-Semantic Reinforcement (MSR)**:

    - **Design Motivation**: Global optimization causes synthetic images to exhibit low-rank structural regularity, where adjacent pixels are highly similar, producing dull regions; the patch-based mechanism of ViTs further exacerbates this issue.
    - $m$ ($m \in \{1,...,K_{MSR}\}$, $K_{MSR}=4$) non-overlapping patches are randomly selected from the synthetic image.
    - Each patch is cropped and resized to the model input resolution and optimized as an independent image with a distinct semantic label.
    - Gradients are back-propagated only to the corresponding patch regions in the original image.
    - Effect: Each patch learns a different semantic, eliminating dull regions and producing synthetic images with richer content and texture diversity.

3. **Soft Label Learning (SL)**:

    - **Design Motivation**: Since MSR causes a single synthetic image to contain patches with multiple distinct semantics, conventional one-hot loss is inappropriate.
    - Soft labels are generated for each patch and the overall image: $T_s = \text{softmax}(Z)$, where values at relevant class positions are sampled from $U(\epsilon_1, \epsilon_2)$ ($\epsilon_1=5, \epsilon_2=10$).
    - Soft cross-entropy replaces one-hot cross-entropy.
    - This ensures consistent, non-conflicting supervision for the multi-semantic images produced by MSR.

### Loss & Training

Total data synthesis loss:
$$\mathcal{L}_G = \alpha_1 \mathcal{L}^{\text{APA}} + \mathcal{L}^{\text{SL}} + 0.05 \mathcal{L}^{\text{TV}}$$

Quantized network learning uses block-wise reconstruction loss $\mathcal{L}_l = \|\mathbf{X}_l - \bar{\mathbf{X}}_l\|_2$.

- Only 32 synthetic images are generated; Adam optimizer; 1,000 iterations.
- Quantization learning: Adam optimizer, learning rate 4e-5, cosine decay, 100 iterations.
- Weight channel quantization, activation quantization, and log2 quantizer for attention scores.

## Key Experimental Results

### Main Results (Tables)

**ImageNet Top-1 Accuracy (%), W4A4 Setting**

| Model | Full Precision | Gaussian Noise | PSAQ-ViT | PSAQ-ViT V2 | SMI | **SARDFQ** |
|-------|---------------|----------------|----------|-------------|-----|------------|
| ViT-S | 81.39 | 6.02 | 47.24 | 41.53 | 24.33 | **50.32** |
| ViT-B | 84.54 | 0.15 | 36.32 | 26.32 | 35.27 | **51.84** |
| DeiT-T | 72.21 | 17.43 | 47.75 | 30.20 | 30.14 | **52.06** |
| DeiT-S | 79.85 | 20.89 | 58.28 | 45.53 | 42.77 | **62.29** |
| DeiT-B | 81.85 | 47.20 | 71.75 | 66.43 | 65.33 | **72.17** |
| Swin-S | 83.20 | 31.92 | 73.19 | 65.55 | 65.85 | **74.74** |
| Swin-B | 85.27 | 30.14 | 71.84 | 67.42 | 65.23 | **76.42** |

**W6A6 Setting Performance**

| Model | PSAQ-ViT | **SARDFQ** | Gain |
|-------|----------|------------|------|
| ViT-S | 77.20 | **78.40** | +1.20 |
| ViT-B | 76.65 | **79.16** | +2.51 |
| DeiT-S | 75.85 | **77.31** | +1.46 |
| Swin-B | 82.00 | **83.03** | +1.03 |

### Ablation Study (Tables)

**Contribution of Each Module on W4A4 DeiT-S**

| APA | MSR | SL | Accuracy (%) |
|-----|-----|-----|--------------|
| | | | 51.73 (baseline) |
| ✓ | | | 60.26 (+8.53) |
| | ✓ | | 50.75 (−0.98) |
| | | ✓ | 52.02 (+0.29) |
| ✓ | ✓ | | 61.58 |
| ✓ | | ✓ | 60.51 |
| | ✓ | ✓ | 56.08 |
| ✓ | ✓ | ✓ | **62.29** |

**Comparison of Attention Prior Distribution Types**

| Distribution | Top-1 (%) |
|-------------|-----------|
| GMM | 62.29 |
| Laplace | 62.16 |
| Real attention | 63.19 |

### Key Findings

- **APA is the most critical component**, yielding a standalone improvement from 51.73% to 60.26% (+8.53%), validating the importance of semantic alignment.
- MSR alone degrades performance (50.75%) due to conflicts between one-hot loss and multi-semantic images; combined with SL, it recovers to 56.08%.
- **ViT-B W4A4 gain of 15.52%** is the largest across all settings, indicating that lower-capacity models are more dependent on high-quality synthetic data.
- GMM-generated priors achieve performance within 0.9% of real attention patterns, demonstrating that exact replication of real attention is unnecessary.
- Significant quantization performance improvements are achievable with only 32 synthetic images.
- Depth weighting ($l/L$) contributes a 0.97% performance gain; applying APA only to deeper blocks outperforms full-layer application by 0.33%.

## Highlights & Insights

- **Precise problem formulation**: The paper clearly identifies and quantifies semantic distortion and semantic insufficiency, supported by compelling evidence via t-SNE and cosine similarity metrics.
- **Elegant APA design**: Random GMM-generated attention priors guide semantic alignment while maintaining diversity, without requiring direct replication of real attention.
- **Synergistic design of MSR + SL**: MSR enriches semantics but requires a compatible loss function; SL provides exactly the multi-semantics-compatible supervision signal.
- **Extreme data efficiency**: Substantial improvements in quantized model performance are achieved with only 32 synthetic images, making the approach highly valuable in data-privacy-constrained scenarios.

## Limitations & Future Work

- A significant performance gap relative to real-data quantization remains (e.g., W4A4 ViT-B: 51.84% vs. 68.16%), approximately 16%.
- A formal theoretical framework explaining how APA and MSR influence the properties of synthetic images is absent.
- The hyperparameter $\alpha_1$ requires per-model search (ranging from 1 to 1e5), increasing tuning cost.
- More advanced generative models (e.g., diffusion models) as alternatives to Gaussian-noise-based optimization are not explored.
- Validation on downstream tasks (detection, segmentation) is relegated to the appendix and is not sufficiently presented in the main paper.

## Related Work & Insights

- **PSAQ-ViT** (first ViT DFQ method): Proposes PSE loss to guide Gaussian noise into images with heterogeneous patches, but suffers from semantic distortion.
- **PSAQ-ViT V2**: Introduces adversarial learning but does not address semantic insufficiency.
- **SMI**: Proposes sparse generation to remove noisy and hallucinatory backgrounds, but exhibits unstable performance.
- **CRD / MoCo**: The principles of contrastive learning and momentum contrast share conceptual commonality with the attention pattern alignment in APA.
- **ZeroQ / GDFQ**: Classic CNN DFQ methods leveraging BNS, not applicable to ViTs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Addressing semantic alignment from the attention prior perspective is a genuinely novel angle; the local patch optimization in MSR is concise and clever.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 7 models × 3 bit-widths, comprehensive ablations (modules, distribution types, hyperparameters, depth weighting), and thorough visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; t-SNE and cosine similarity provide convincing quantitative evidence; method description is detailed.
- **Value**: ⭐⭐⭐⭐ — Significant practical value for ViT deployment in data-privacy-constrained settings, particularly under low-bit quantization.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Model Inversion with Layer-Specific Modeling and Alignment for Data-Free Continual Learning](../../NeurIPS2025/ai_safety/model_inversion_with_layer-specific_modeling_and_alignment_for_data-free_continu.md)
- [\[ICCV 2025\] Backdooring Self-Supervised Contrastive Learning by Noisy Alignment](backdooring_self-supervised_contrastive_learning_by_noisy_alignment.md)
- [\[ICCV 2025\] Staining and Locking Computer Vision Models without Retraining](staining_and_locking_computer_vision_models_without_retraining.md)
- [\[ICCV 2025\] Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation](controllable_feature_whitening_for_hyperparameter-free_bias_mitigation.md)
- [\[ICCV 2025\] Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)

<!-- RELATED:END -->
