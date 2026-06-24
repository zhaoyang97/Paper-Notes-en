---
title: >-
  [Paper Note] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach
description: >-
  [AI Safety] This study proposes the TGAF framework, which leverages diffusion models to encode target labels into 2D semantic tensors to guide adversarial noise generation, and designs a random masking strategy to preserve complete semantic information, significantly improving the transferability of targeted adversarial attacks.
tags:
  - "AI Safety"
date: 2026-05-08
content_hash: 5381dc5868d5e473
---

# Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach

- **Conference**: AAAI 2026
- **arXiv**: [2504.14137](https://arxiv.org/abs/2504.14137)
- **Code**: [GitHub - TemenosMistral/TGAF](https://github.com/TemenosMistral/TGAF)
- **Area**: AI Security
- **Keywords**: Adversarial Examples, Targeted Attacks, Transferability, Diffusion Models, 2D Semantic Tensor, Multi-target Attacks

## TL;DR

This study proposes the TGAF framework, which leverages diffusion models to encode target labels into 2D semantic tensors to guide adversarial noise generation, and designs a random masking strategy to preserve complete semantic information, significantly improving the transferability of targeted adversarial attacks.

## Background & Motivation

- **Classification of Adversarial Attacks**: Untargeted attacks only need to cause misclassification, whereas targeted attacks must force the model to output a specific class designated by the attacker. The latter is far more threatening (e.g., deceiving autonomous driving systems to recognize stop signs as speed limit signs).
- **Transferability Bottleneck**: The success rate of targeted attacks in black-box scenarios is much lower than that of untargeted attacks, primarily because they easily overfit the decision boundaries of surrogate models.
- **Limitations of Prior Work**: Multi-target generation methods such as C-GSP and CGNC encode labels as **1D tensors** (one-hot or CLIP embeddings), discarding spatial structural information. This results in generated adversarial noise that lacks fine-grained visual features of the target category.
- **Key Findings**: The authors systematically analyze two critical factors from the perspective of "semantic feature implantation":
    - **Feature Quality**: The structural integrity of the implanted target features. Missing critical discriminative information will prevent some models from recognizing them.
    - **Feature Quantity**: The spatial sufficiency of the implanted target features. Insufficiency limits the victim model's focus on these features.

## Method

### Overall Architecture: TGAF Framework

TGAF consists of four core components: an image encoder $\mathcal{E}$, a text-to-image encoder $\mathcal{G}$, a feature integration module $\mathcal{F}$, and an image decoder $\mathcal{D}$.

Given an input image $\mathbf{I} \in \mathbb{R}^{C \times H \times W}$ and a target class $c_t$, the perturbation $\delta$ is generated to mislead the model such that the adversarial example $x_{adv} = x + \delta$ satisfies the $\ell_\infty$ norm constraint:

$$\arg\max f_\Phi(x_{adv}) = c_t, \quad \|\delta\|_\infty \leq \epsilon$$

### Design 1: 2D Target Semantic Tensor Generation (Text-to-Image Encoder)

**Core Idea**: Utilizing the encoder and denoising UNet of Stable-Diffusion-2 to convert target class labels into 2D spatial representations rather than traditional 1D vectors.

- The target label text is input and processed by the diffusion model to obtain a low-dimensional latent vector of size $B \times 4 \times 64 \times 64$.
- It is aligned with the image feature space via convolutional layers and average pooling: $\mathbf{z}_c \in \mathbb{R}^{4 \times \frac{H}{4} \times \frac{W}{4}}$.
- **Design Motivation**: The diffusion model is run only once for each target category before training, and the generated 2D tensors are saved to disk. During training and inference, they are directly loaded without introducing additional inference overhead.
- This preserves the low-level semantic information (spatial structure, texture details) of the target category, avoiding the information loss inherent in 1D encoding.

### Design 2: Dual-Path Feature Integration Module

To integrate the image representation $\mathbf{x}$ and the target conditional representation $\mathbf{z}_c$, two complementary strategies are adopted:

**Convolution-based Fusion (CbF)**: Learning local feature interactions via a $1 \times 1$ convolution:

$$\mathbf{f}_c = \text{Conv}_{1 \times 1}(\mathbf{x} \| \mathbf{z}_c)$$

**Transformer-based Fusion (TbF)**: Capturing global spatial-channel dependencies through three stages:
1. Channel Alignment: $\mathbf{z}_t = \text{Conv}_{1 \times 1}(\mathbf{z}_c)$
2. Channel Attention Recalibration: $\mathbf{x}_{ca} = \mathbf{x}_c \odot \text{CHA}(\mathbf{x}_c)$
3. Self-Attention + Cross-Attention Fusion:

$$\mathbf{f}_t = \text{CA}(\text{SA}(\mathbf{x}_{ca}), \mathbf{z}_t)$$

where self-attention and cross-attention are defined as:

$$\text{SA}(\mathbf{x}_{ca}) = \text{Softmax}\left(\frac{\mathbf{Q}_{ca}\mathbf{K}_{ca}^T}{\sqrt{d_k}}\right)\mathbf{V}_{ca}$$

$$\text{CA}(\mathbf{x}_{sa}, \mathbf{z}_t) = \text{Softmax}\left(\frac{\mathbf{Q}_{sa}\mathbf{K}_{z_t}^T}{\sqrt{d_k}}\right)\mathbf{V}_{z_t}$$

### Design 3: Dynamic Block-Level Random Masking Strategy

- The image is divided into $N \times N$ blocks of varying sizes, and two blocks are randomly selected for masking during training.
- **Function**: To ensure that partial noise regions still retain complete target class semantic information, preventing noise from concentrating solely on easily mapped regions.
- The decoder output is projected via $\tanh$: $\delta = \epsilon \cdot \tanh(\mathbf{o})$

### Training Objectives

End-to-end optimization of the cross-entropy loss:

$$\theta^* \leftarrow \arg\min_\theta \mathcal{L}_{\text{CE}}\left(f_\Phi(\mathbf{x}_s + \mathcal{D}_\theta(\mathcal{F}_\theta(\mathcal{E}_\theta, \mathcal{G}_\theta)), c_t\right)$$

## Experimental Results

### Experimental Setup

- **Dataset**: Trained on the ImageNet training set, evaluated on the ImageNet-NeurIPS 1k dataset.
- **Surrogate Models**: Inc-v3 and Res-152.
- **Perturbation Budget**: $\epsilon = 16/255$, trained for 10 epochs with a learning rate of 2e-4.
- **Baselines**: Logit, SU, Everywhere (instance-specific); C-GSP, CGNC (instance-agnostic).

### Table 1: Attack Success Rates (%) on Normally Trained Models

| Surrogate Model | Method | Inc-v4 | Inc-Res-v2 | Res-152 | DN-121 | VGG-16 | ViT-B | Swin-T |
|---------|------|--------|------------|---------|--------|--------|-------|--------|
| Inc-v3 | CGNC | 59.41 | 47.98 | 42.50 | 62.91 | 52.63 | 24.81 | 28.16 |
| Inc-v3 | **TGAF** | **72.49** | **63.20** | **61.94** | **78.30** | **70.64** | **33.03** | **42.61** |
| Res-152 | CGNC | 51.59 | 34.18 | — | 85.60 | 63.36 | 34.81 | 40.84 |
| Res-152 | **TGAF** | **62.44** | **44.02** | — | **87.90** | **65.20** | **39.64** | **42.84** |

### Table 2: Attack Success Rates (%) on Robustly Trained Models

| Surrogate Model | Method | Inc-v3ADV | IR-v2ENS | Res50SIN | Res50IN | Res50FINE | Res50AUG |
|---------|------|-----------|----------|----------|---------|-----------|----------|
| Inc-v3 | CGNC | 24.30 | 22.51 | 8.88 | 40.81 | 52.13 | 22.83 |
| Inc-v3 | **TGAF** | **39.69** | **34.86** | **17.76** | **64.79** | **72.36** | **43.53** |
| Res-152 | CGNC | 22.15 | 26.70 | 29.81 | 79.82 | 84.05 | 63.66 |
| Res-152 | **TGAF** | **27.73** | **32.71** | **38.07** | **84.53** | **88.48** | **68.63** |

## Key Findings

1. **2D Encoding vs. 1D Encoding**: 2D semantic tensors preserve spatial structure, allowing adversarial noise to contain more complete target class features (such as the pointer of a barometer or the number of figs).
2. **Complementarity of Dual Fusion Strategies**: Ablation studies show that removing CbF (TGAF-Conv) or TbF (TGAF-CA) leads to performance degradation. CbF captures local features, while TbF models global dependencies.
3. **Effectiveness of Masking Strategy**: Removing the masking strategy (TGAF-N) results in an average ASR decline of approximately 6-8%, and replacing it with the CGNC masking strategy (TGAF-C) also leads to a notable decrease.
4. **Sensitivity to the Number of Blocks**: $N=3$ achieves the best performance. $N=2$ results in excessively large masked regions, while $N=4$ causes too fragmented masks.
5. **Almost Negligible Loss of Image Quality**: The difference between TGAF and CGNC in terms of SSIM/LPIPS/FID metrics is extremely small (LPIPS differ by only 0.013), and PSNR is slightly superior.

## Highlights & Insights

- **Novel Perspective**: This study is the first to systematically analyze targeted attack transferability from two dimensions: feature quality and feature quantity.
- **Clever Use of Diffusion Models**: The diffusion model is used only once during the preprocessing phase to generate and cache 2D tensors, which does not affect training or inference efficiency.
- **Comprehensive Experiments**: The method is evaluated across normal models, robust models, and various defense methods (preprocessing, denoising, and diffusion purification), consistently achieving SOTA.
- **Cross-Architecture Transfer**: Significant performance improvements are attained on both CNN and Transformer architectures.

## Limitations & Future Work

- The approach depends on Stable Diffusion to generate 2D tensors, introducing a degree of dependency on the generation quality of the diffusion model.
- Under strong defenses like DiffPure, the ASR remains very low (<1%), indicating that such defenses still pose a significant challenge.
- Experiments only validate the method on the ImageNet classification task without extending to downstream tasks such as detection or segmentation.
- The perturbation budget is fixed at $\epsilon=16/255$, and behaviors under larger budget settings have not been explored.

## Related Work & Insights

- **Instance-Specific Attacks**: Logit (Zhao et al. 2021), SU/DTMI-Logit-SU (Wei et al. 2023), Everywhere/CFM (Zeng et al. 2025) — optimized iteratively sample by sample, leading to low efficiency.
- **Instance-Agnostic Attacks**: C-GSP (Yang et al. 2022) utilizes CLIP embeddings for 1D conditional generation; CGNC (Fang et al. 2024) introduces fine-grained masking but is still limited by 1D encoding.
- **Defense Methods**: Adversarial training (Goodfellow et al. 2014), preprocessing defenses (JPEG/BitSqueezing/Smoothing), diffusion purification (DiffPure, NRP).

## Rating

⭐⭐⭐⭐ — Clear motivation (feature quality and quantity analysis), novel 2D tensor-guided mechanism, comprehensive experiments with significant improvements. However, the core relies on a pretrained diffusion model, and there is still substantial room for improvement under strong defenese scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[AAAI 2026\] Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)
- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)

</div>

<!-- RELATED:END -->
