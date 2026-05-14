---
title: >-
  [Paper Note] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach
description: >-
  [AI Safety] This paper proposes the TGAF framework, which leverages diffusion models to encode target labels as 2D semantic tensors for guiding adversarial noise generation…
tags:
  - "AI Safety"
date: 2026-05-08
content_hash: c07a6cd946894c82
---

# Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach

- **Conference**: AAAI 2026
- **arXiv**: [2504.14137](https://arxiv.org/abs/2504.14137)
- **Code**: [GitHub - TemenosMistral/TGAF](https://github.com/TemenosMistral/TGAF)
- **Area**: AI Safety
- **Keywords**: Adversarial Examples, Targeted Attacks, Transferability, Diffusion Models, 2D Semantic Tensor, Multi-target Attacks

## TL;DR

This paper proposes the TGAF framework, which leverages diffusion models to encode target labels as 2D semantic tensors for guiding adversarial noise generation, and introduces a random masking strategy to preserve complete semantic information, significantly improving the transferability of targeted adversarial attacks.

## Background & Motivation

- **Attack taxonomy**: Untargeted attacks only need to cause misclassification, whereas targeted attacks must compel the model to output an attacker-specified class — the latter being more threatening (e.g., deceiving autonomous driving systems into recognizing stop signs as speed limit signs).
- **Transferability bottleneck**: Targeted attacks achieve substantially lower success rates than untargeted attacks in black-box settings, primarily due to overfitting to the surrogate model's decision boundary.
- **Limitations of prior work**: Multi-target generative methods such as C-GSP and CGNC encode labels as **1D tensors** (one-hot or CLIP embeddings), discarding spatial structural information and causing the generated adversarial noise to lack fine-grained visual features of the target class.
- **Core finding**: The authors systematically analyze two key factors from the perspective of "semantic feature implantation":
    - **Feature Quality**: The structural completeness of the implanted target features; missing critical discriminative information prevents some models from recognizing the target class.
    - **Feature Quantity**: The spatial sufficiency of the implanted target features; insufficient coverage limits victim models' attention to those features.

## Method

### Overall Architecture: TGAF Framework

TGAF comprises four core components: an image encoder $\mathcal{E}$, a text-to-image encoder $\mathcal{G}$, a feature integration module $\mathcal{F}$, and an image decoder $\mathcal{D}$.

Given an input image $\mathbf{I} \in \mathbb{R}^{C \times H \times W}$ and a target class $c_t$, the framework generates a perturbation $\delta$ such that the adversarial example $x_{adv} = x + \delta$ misleads the model under an $\ell_\infty$ norm constraint:

$$\arg\max f_\Phi(x_{adv}) = c_t, \quad \|\delta\|_\infty \leq \epsilon$$

### Key Design 1: 2D Target Semantic Tensor Generation (Text-to-Image Encoder)

**Core Idea**: The encoder of Stable-Diffusion-2 and its denoising UNet are leveraged to convert target class labels into 2D spatial representations, rather than conventional 1D vectors.

- The target label text is processed through the diffusion model to obtain a low-dimensional latent vector of shape $B \times 4 \times 64 \times 64$.
- Convolutional layers and average pooling align it to the image feature space: $\mathbf{z}_c \in \mathbb{R}^{4 \times \frac{H}{4} \times \frac{W}{4}}$.
- **Key advantage**: The diffusion model is invoked only once per target class prior to training; the resulting 2D tensors are saved to disk and loaded directly during training and inference, introducing no additional inference overhead.
- Low-level semantic information of the target class (spatial structure, texture details) is preserved, avoiding the information loss inherent in 1D encoding.

### Key Design 2: Dual-Path Feature Integration Strategy (Feature Integration Module)

The image representation $\mathbf{x}$ and the target-conditioned representation $\mathbf{z}_c$ are fused via two complementary strategies:

**Convolution-based Fusion (CbF)**: Local feature interactions are learned via $1 \times 1$ convolution:

$$\mathbf{f}_c = \text{Conv}_{1 \times 1}(\mathbf{x} \| \mathbf{z}_c)$$

**Transformer-based Fusion (TbF)**: Global spatial-channel dependencies are captured in three stages:
1. Channel alignment: $\mathbf{z}_t = \text{Conv}_{1 \times 1}(\mathbf{z}_c)$
2. Channel attention recalibration: $\mathbf{x}_{ca} = \mathbf{x}_c \odot \text{CHA}(\mathbf{x}_c)$
3. Self-attention + cross-attention fusion:

$$\mathbf{f}_t = \text{CA}(\text{SA}(\mathbf{x}_{ca}), \mathbf{z}_t)$$

where self-attention and cross-attention are defined as:

$$\text{SA}(\mathbf{x}_{ca}) = \text{Softmax}\left(\frac{\mathbf{Q}_{ca}\mathbf{K}_{ca}^T}{\sqrt{d_k}}\right)\mathbf{V}_{ca}$$

$$\text{CA}(\mathbf{x}_{sa}, \mathbf{z}_t) = \text{Softmax}\left(\frac{\mathbf{Q}_{sa}\mathbf{K}_{z_t}^T}{\sqrt{d_k}}\right)\mathbf{V}_{z_t}$$

### Key Design 3: Dynamic Block-wise Random Masking Strategy (Mask Mechanism)

- The image is divided into $N \times N$ blocks of varying sizes; during training, 2 blocks are randomly selected for masking.
- **Purpose**: To ensure that certain noise regions still retain complete semantic information of the target class, preventing noise from concentrating in regions that are easy to map.
- The decoder output is projected via $\tanh$: $\delta = \epsilon \cdot \tanh(\mathbf{o})$

### Loss & Training

End-to-end optimization of the cross-entropy loss:

$$\theta^* \leftarrow \arg\min_\theta \mathcal{L}_{\text{CE}}\left(f_\Phi(\mathbf{x}_s + \mathcal{D}_\theta(\mathcal{F}_\theta(\mathcal{E}_\theta, \mathcal{G}_\theta)), c_t\right)$$

## Key Experimental Results

### Experimental Setup

- **Dataset**: Trained on ImageNet training set; evaluated on ImageNet-NeurIPS 1k.
- **Surrogate models**: Inc-v3 and Res-152.
- **Perturbation budget**: $\epsilon = 16/255$; trained for 10 epochs with learning rate 2e-4.
- **Baselines**: Logit, SU, Everywhere (instance-specific); C-GSP, CGNC (instance-agnostic).

### Table 1: Attack Success Rate (%) on Normally Trained Models

| Surrogate | Method | Inc-v4 | Inc-Res-v2 | Res-152 | DN-121 | VGG-16 | ViT-B | Swin-T |
|---------|------|--------|------------|---------|--------|--------|-------|--------|
| Inc-v3 | CGNC | 59.41 | 47.98 | 42.50 | 62.91 | 52.63 | 24.81 | 28.16 |
| Inc-v3 | **TGAF** | **72.49** | **63.20** | **61.94** | **78.30** | **70.64** | **33.03** | **42.61** |
| Res-152 | CGNC | 51.59 | 34.18 | — | 85.60 | 63.36 | 34.81 | 40.84 |
| Res-152 | **TGAF** | **62.44** | **44.02** | — | **87.90** | **65.20** | **39.64** | **42.84** |

### Table 2: Attack Success Rate (%) on Robustly Trained Models

| Surrogate | Method | Inc-v3ADV | IR-v2ENS | Res50SIN | Res50IN | Res50FINE | Res50AUG |
|---------|------|-----------|----------|----------|---------|-----------|----------|
| Inc-v3 | CGNC | 24.30 | 22.51 | 8.88 | 40.81 | 52.13 | 22.83 |
| Inc-v3 | **TGAF** | **39.69** | **34.86** | **17.76** | **64.79** | **72.36** | **43.53** |
| Res-152 | CGNC | 22.15 | 26.70 | 29.81 | 79.82 | 84.05 | 63.66 |
| Res-152 | **TGAF** | **27.73** | **32.71** | **38.07** | **84.53** | **88.48** | **68.63** |

## Key Findings

1. **2D vs. 1D encoding**: 2D semantic tensors preserve spatial structural information, enabling the adversarial noise to contain more complete target-class features (e.g., the pointer of a barometer, the count of figs).
2. **Complementarity of dual fusion strategies**: Ablation studies show that removing either CbF (TGAF-Conv) or TbF (TGAF-CA) leads to performance degradation; CbF captures local features while TbF models global dependencies.
3. **Effectiveness of the masking strategy**: Removing the masking strategy (TGAF-N) causes an average ASR drop of approximately 6–8%; replacing it with the CGNC masking strategy (TGAF-C) also yields a noticeable decline.
4. **Sensitivity to block count**: $N=3$ achieves the best performance; $N=2$ results in overly large masked regions, while $N=4$ produces excessively fine-grained masking.
5. **Negligible image quality degradation**: TGAF and CGNC exhibit minimal differences on SSIM/LPIPS/FID metrics (LPIPS difference of only 0.013), with TGAF achieving slightly better PSNR.

## Highlights & Insights

- **Novel perspective**: This is the first work to systematically analyze targeted attack transferability along the two dimensions of feature quality and feature quantity.
- **Clever use of diffusion models**: The diffusion model is used only once in a preprocessing stage to generate and cache 2D tensors, without affecting training or inference efficiency.
- **Comprehensive experiments**: Evaluations cover normally trained models, robustly trained models, and various defense methods (preprocessing / denoising / diffusion purification), achieving state-of-the-art results across all settings.
- **Cross-architecture transferability**: Significant improvements are demonstrated on both CNN and Transformer architectures.

## Limitations & Future Work

- The approach depends on Stable Diffusion for 2D tensor generation, introducing a reliance on the generation quality of the diffusion model.
- ASR remains very low (<1%) against strong defenses such as DiffPure, indicating that this class of defenses remains a major challenge.
- Experiments are conducted solely on ImageNet classification; generalization to downstream tasks such as detection and segmentation has not been explored.
- The perturbation budget is fixed at $\epsilon=16/255$; behavior under larger budget settings is not investigated.

## Related Work & Insights

- **Instance-specific attacks**: Logit (Zhao et al. 2021), SU/DTMI-Logit-SU (Wei et al. 2023), Everywhere/CFM (Zeng et al. 2025) — per-sample iterative optimization with low efficiency.
- **Instance-agnostic attacks**: C-GSP (Yang et al. 2022) employs CLIP embeddings for 1D conditional generation; CGNC (Fang et al. 2024) introduces fine-tuned masking but remains constrained by 1D encoding.
- **Defense methods**: Adversarial training (Goodfellow et al. 2014), preprocessing defenses (JPEG / Bit-Squeezing / Smoothing), diffusion purification (DiffPure, NRP).

## Rating

⭐⭐⭐⭐ — The method motivation is clear (feature quality + quantity analysis), the 2D tensor guidance is a novel idea, and the experiments are comprehensive with significant gains; however, the approach relies fundamentally on a pretrained diffusion model, and considerable room for improvement remains in defensive settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[AAAI 2026\] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks](improving_the_convergence_rate_of_ray_search_optimization_for_query-efficient_ha.md)

</div>

<!-- RELATED:END -->
