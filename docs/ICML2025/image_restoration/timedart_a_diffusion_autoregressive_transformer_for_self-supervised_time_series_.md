---
title: >-
  [Paper Note] TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation
description: >-
  [ICML2025][Image Restoration][Self-Supervised Learning] TimeDART is proposed to unify autoregressive modeling and denoising diffusion processes within a self-supervised pre-training framework. It captures long-term dynamic evolution via a causal Transformer encoder and fine-grained local patterns through patch-level diffusion denoising, outperforming existing methods on both forecasting and classification tasks.
tags:
  - "ICML2025"
  - "Image Restoration"
  - "Self-Supervised Learning"
  - "Time Series Representation Learning"
  - "Diffusion Models"
  - "Autoregressive Transformer"
  - "Pre-training"
date: 2026-05-08
content_hash: 837638f58d4421d2
---

# TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation

**Conference**: ICML2025  
**arXiv**: [2410.05711](https://arxiv.org/abs/2410.05711)  
**Code**: [GitHub](https://github.com/Melmaphother/TimeDART)  
**Area**: Image Restoration  
**Keywords**: Self-Supervised Learning, Time Series Representation Learning, Diffusion Models, Autoregressive Transformer, Pre-training

## TL;DR
TimeDART is proposed to unify autoregressive modeling and denoising diffusion processes within a self-supervised pre-training framework. It captures long-term dynamic evolution via a causal Transformer encoder and fine-grained local patterns through patch-level diffusion denoising, outperforming existing methods on both forecasting and classification tasks.

## Background & Motivation

Self-supervised learning for time series primarily falls into three paradigms, each with its own limitations:

**Masked Autoencoders** (e.g., PatchTST, TimeMAE): Discrepancies exist between pre-training and fine-tuning due to the introduction of new mask embeddings during fine-tuning.

**Contrastive Learning** (e.g., CoST, TS2Vec): Focuses on sequence-level modeling, making it difficult to capture fine-grained temporal variations.

**Autoregressive Methods**: Naturally align with left-to-right temporal dynamics but are prone to overfitting noise. Moreover, the MSE loss implicitly assumes that the data follows a Gaussian distribution, causing predictions to collapse into a fixed-variance Gaussian distribution centered on historical values.

**Core Motivation**: How to model both long-term dynamic evolution and fine-grained local patterns within a unified framework? The authors propose embedding a diffusion denoising process into an autoregressive optimization framework, replacing the simple MSE with diffusion loss to enable the model to express richer, multimodal distributions.

## Method

TimeDART consists of three core components:

### 1. Normalization and Patch Embedding

- Apply Instance Normalization (zero mean, unit variance) to each sample.
- Segment the sequence of length $L$ into $N = L/P$ non-overlapping patches (stride $S = P$, preventing information leakage).
- A linear embedding layer maps the patches to high-dimensional representations: $\mathbf{z}_{1:N} = \text{Embedding}(\mathbf{x}_{1:N})$.

### 2. Causal Transformer Encoder

- Prepend a learnable SOS (Start-of-Sequence) token to the patch embedding sequence and discard the last patch.
- Add sinusoidal positional encodings and apply a causal mask $M$ in the encoder, restricting each patch to attend only to its preceding patches.
- Encoder output: $f(\mathbf{z}_{1:N}^{in}) = \text{Encoder}(\mathbf{z}_{1:N}^{in}, M)$.

### 3. Patch-level Diffusion Denoising

**Forward Process**: Noise is added to each patch independently using a cosine scheduler:

$$q(x_j^s | x_j^0) = \mathcal{N}(x_j^s; \sqrt{\gamma(s)} x_j^0, (1 - \gamma(s))I)$$

where $\gamma(s) = \prod_{s' \leq s} \alpha(s')$. The number of noise steps $s_j$ for each patch is independently and randomly sampled to prevent the task from becoming too trivial.

**Reverse Process**: The denoising decoder performs cross-attention, using the encoder outputs as keys/values and the noisy patch embeddings as queries. The decoder uses a self-only mask to ensure that the $j$-th input only attends to the $j$-th encoder output:

$$z_j^{out} = g(\hat{z}_j^{in}, f(\mathbf{z}_{1:j-1}^{in}))$$

### 4. Optimization Objective: Replacing MSE with Diffusion Loss

The traditional autoregressive MSE loss implicitly assumes a Gaussian distribution. The authors show that it is equivalent to:

$$-\log \mathcal{N}(x_j^0; \text{Projector}(f(\mathbf{z}_{1:j-1}^{in})), \sigma^2) + C$$

This assumes that the time series follows a unimodal Gaussian distribution, which contradicts real-world multimodal distributions. TimeDART utilizes diffusion loss (equivalent to ELBO):

$$\mathcal{L}_{diff} = \sum_{j=1}^{N} \mathbb{E}_{\epsilon, q(x_j^0)} \left[ \| x_j^0 - g(\hat{z}_j^{in}, f(\mathbf{z}_{1:j-1}^{in})) \|^2 \right]$$

### 5. Downstream Transfer

Upon completion of pre-training, the denoising decoder is discarded, leaving only the embedding layer and encoder to be connected to task-specific heads:
- **Forecasting**: flatten head + MSE loss
- **Classification**: max-pooling head + cross-entropy loss

## Key Experimental Results

### Forecasting Task (12 datasets, MSE/MAE, lower is better)

| Method | ETTh1 | ETTh2 | ETTm1 | ETTm2 | Electricity | Traffic | Weather | Exchange |
|------|-------|-------|-------|-------|-------------|---------|---------|----------|
| **TimeDART** | **0.411** | **0.346** | 0.344 | **0.257** | **0.163** | **0.388** | **0.226** | **0.359** |
| SimMTM | 0.409 | 0.353 | 0.348 | 0.263 | 0.162 | 0.392 | 0.230 | 0.451 |
| PatchTST(SSL) | 0.433 | 0.354 | **0.342** | 0.272 | 0.163 | 0.404 | 0.227 | 0.376 |
| TimeMAE | 0.434 | 0.402 | 0.350 | 0.270 | 0.196 | 0.410 | 0.227 | 0.427 |
| Random Init. | 0.439 | 0.358 | 0.351 | 0.269 | 0.177 | 0.410 | 0.231 | 0.440 |

- Achieves the best performance on **83.3%** of the 24 evaluation metrics.
- Reduces MSE by **6.8%** compared to random initialization, and by **3%** compared to the SOTA.

### Classification Task (Accuracy)

| Method | HAR | Epilepsy | EEG |
|------|-----|----------|-----|
| **TimeDART** | **0.9247** | **0.9712** | **0.8269** |
| SimMTM | 0.9200 | 0.9565 | 0.8165 |
| TimeMAE | 0.9204 | 0.9459 | 0.8148 |
| Random Init. | 0.8738 | 0.9265 | 0.7752 |

- The average accuracy after pre-training improves by **5.7%** compared to random initialization.

### Ablation Study

| Setup | ETTh2 MSE | ETTm2 MSE | Electricity MSE | HAR Acc. |
|------|-----------|-----------|-----------------|----------|
| TimeDART | **0.346** | **0.257** | **0.163** | **0.9247** |
| w/o AR | 0.365 | 0.281 | 0.193 | 0.8966 |
| w/o Diff | 0.352 | 0.265 | 0.164 | 0.9002 |
| w/o AR-Diff | 0.364 | 0.285 | 0.190 | 0.8785 |

Removing the autoregressive mechanism results in performance even worse than random initialization, demonstrating its key role in capturing long-term dynamics.

## Highlights & Insights

1. **Profound Theoretical Insight**: Reveals the limitations of the implicit Gaussian assumption of the MSE loss from a maximum likelihood perspective, naturally replacing it with diffusion loss, backed by complete theoretical derivations.
2. **Unification of Two Generative Paradigms**: Autoregressive modeling for global temporal dependencies combined with diffusion denoising for local details—a highly elegant complementary design.
3. **Independent Noise Addition Strategy**: Every patch independently and randomly samples its noise steps, avoiding oversimplifying the task.
4. **Strong Cross-Domain Transferability**: Pre-training on mixed multi-domain datasets followed by fine-tuning outperforms most in-domain pre-training methods.
5. **Generality of the Backbone**: Replacing the Transformer encoder with TCN remains effective, verifying the universality of the framework.

## Limitations & Future Work

1. **No Diffusion Process During Inference**: The pre-trained denoising decoder is discarded during downstream tasks, meaning the learned capability for local pattern modeling is not directly leveraged during inference.
2. **Channel-Independent Strategy**: Does not model the correlation between multiple variables, which may limit performance in strongly coupled multivariate scenarios.
3. **Pre-training Overhead**: The diffusion process introduces additional computational costs, and the paper does not analyze the pre-training time or resource consumption in detail.
4. **Cross-Domain Pre-training Performs Slightly Worse than In-Domain on the Weather Dataset**: Unified hyperparameters fail to adapt to the characteristics of all domains.
5. **Only 3 Datasets for Classification**: The evaluation scope is limited, lacking verification on a broader range of classification benchmarks.

## Related Work & Insights

- **SimMTM** (Dong et al., 2024): Masked time series modeling, manifold structure weighting
- **PatchTST** (Nie et al., 2023): Patch-level Transformer for self-supervised/supervised forecasting
- **TimeMAE** (Cheng et al., 2023): Decoupled masked autoencoders
- **CoST** (Woo et al., 2022): Contrastive learning in time and frequency domains
- **DDPM/Diffusion Models** (Ho et al., 2020): Denoising diffusion probabilistic models; TimeDART embeds them into an autoregressive framework rather than using them for direct generation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The design of the unified autoregressive + diffusion framework is novel, with in-depth theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across 9 datasets, ablations, parameter sensitivity analysis, cross-domain, and few-shot settings.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, intuitive diagrams, and complete theoretical derivations.
- Value: ⭐⭐⭐⭐ — Provides a solid new paradigm for self-supervised pre-training of time series.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Time Without Time: Pseudo-Temporal Representation for Space-Time Super-Resolution](../../CVPR2026/image_restoration/time_without_time_pseudo-temporal_representation_for_space-time_super-resolution.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](../../ICLR2026/image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)
- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[CVPR 2026\] SelfHVD: Self-Supervised Handheld Video Deblurring](../../CVPR2026/image_restoration/selfhvd_self-supervised_handheld_video_deblurring.md)
- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)

</div>

<!-- RELATED:END -->
