---
title: >-
  [Paper Note] Enhancing Diffusion Model Guidance through Calibration and Regularization
description: >-
  [NeurIPS 2025 (SPIGM Workshop)][Image Generation][classifier guidance] To address the vanishing gradient problem caused by overconfident classifiers in classifier-guided diffusion models, this paper proposes two complementary approaches: (1) a Smooth ECE calibration loss for fine-tuning classifiers, yielding ~3% FID improvement; and (2) regularized sampling guidance based on f-divergences (RKL/FKL/JS) that requires no retraining, achieving FID 2.13 on ImageNet 128×128.
tags:
  - NeurIPS 2025 (SPIGM Workshop)
  - Image Generation
  - classifier guidance
  - diffusion model
  - f-divergence
  - calibration
  - conditional generation
date: 2026-05-08
content_hash: 1952ba25091fb40d
---

# Enhancing Diffusion Model Guidance through Calibration and Regularization

**Conference**: NeurIPS 2025 (SPIGM Workshop)
**arXiv**: [2511.05844](https://arxiv.org/abs/2511.05844)
**Code**: [ajavid34/guided-info-diffusion](https://github.com/ajavid34/guided-info-diffusion)
**Area**: Image Generation
**Keywords**: classifier guidance, diffusion model, f-divergence, calibration, conditional generation

## TL;DR

To address the vanishing gradient problem caused by overconfident classifiers in classifier-guided diffusion models, this paper proposes two complementary approaches: (1) a Smooth ECE calibration loss for fine-tuning classifiers, yielding ~3% FID improvement; and (2) regularized sampling guidance based on f-divergences (RKL/FKL/JS) that requires no retraining, achieving FID 2.13 on ImageNet 128×128.

## Background & Motivation

Classifier-guided DDPM steers the reverse diffusion process toward a target class using gradients $\nabla_x \log p(y|x)$ from an external classifier, making it a core technique for conditional image generation. However, a critical failure mode exists:

**Vanishing Gradient Problem**: Classifiers assign excessively high confidence (near one-hot distributions) to partially denoised images in the early stages of denoising, causing $\nabla_x \log p(y|x) \to 0$. Subsequent denoising steps then effectively degrade to unconditional generation, severely degrading conditional generation quality.

Existing remedies (e.g., entropy-constrained training) require training classifiers from scratch and cannot be applied to off-the-shelf classifiers.

## Core Problem

How can the vanishing gradient problem in classifier guidance be mitigated—without retraining the diffusion model or classifier—to improve the quality and diversity of conditional generation?

## Method

### 1. Smooth ECE Calibration Loss (Requires Fine-tuning)

A differentiable Huber-style calibration loss is defined as:

$$\mathcal{L}_{\text{ECE}} = \frac{1}{n} \sum_{b=1}^{B} \sum_{i: \hat{p}^{(i)} \in \mathcal{B}_b} \sqrt{(\hat{p}^{(i)} - a^{(i)})^2 + \beta}$$

where $\hat{p}^{(i)} = \max_y p_\phi(y|x^{(i)})$ is the predicted confidence, $a^{(i)} = \mathbb{I}[\hat{y}^{(i)} = y^{(i)}]$ is the correctness indicator, and $\beta > 0$ ensures differentiability.

This loss can be incorporated as a regularization term during fine-tuning to improve classifier calibration and, consequently, guidance quality.

### 2. f-Divergence Regularized Sampling Guidance (No Retraining Required)

**Core Idea**: An f-divergence between the classifier's output distribution $p(\cdot|x)$ and the target distribution $q_y(\cdot)$ is used as a regularization term to prevent premature distributional collapse. The guidance score is defined as:

$$\mathcal{S}_\mathcal{D}(x, y) = \log p_{\tau_1, \tau_2}(y|x) - \alpha D_f(q_y(\cdot) \| p(\cdot|x))$$

where $q_y(i) = (1-\epsilon)\frac{1}{N} + \epsilon \mathbb{I}_{i=y}$ is a label-smoothed target distribution, and $\tau_1, \tau_2$ are joint and marginal temperatures.

The general gradient form is:

$$\nabla_x \mathcal{S}_\mathcal{D} = \tau_1 \nabla_x f_y(x) - \tau_2 \sum_i p_{\tau_2}(i|x) \nabla_x f_i(x) - \alpha \sum_i w_f(q_y(i), p(i|x)) g_i(x)$$

where $w_f(q, p) = p f'(p/q)$ and $g_i(x) = \nabla_x f_i(x) - \sum_j p(j|x) \nabla_x f_j(x)$.

### 3. Three f-Divergence Instantiations

#### Reverse KL (Mode Covering)

$f(t) = -\log(t)$, weight $w_f(q,p) = -q$. The gradient decomposes as:

$$-\nabla_x D_{\text{KL}}(q_y \| p) = \underbrace{\sum_i q_y(i) \nabla_x f_i(x)}_{\text{target direction}} - \underbrace{\sum_j p(j|x) \nabla_x f_j(x)}_{\text{current direction}}$$

The **mode-covering** property of RKL ensures non-zero probability wherever $q_y$ has support, preventing mode dropping. Under a Gaussian mixture analysis, the guidance force comprises an enhanced target direction $(\tau_1 + \alpha\epsilon)(\mu_y - x)$ and a diversity term $\alpha\frac{1-\epsilon}{K}\sum_{k \neq y}(\mu_k - x)$.

#### Forward KL (Mode Seeking)

$f(t) = t\log(t)$, with weights containing a $\log(p(i|x)/q_y(i))$ term. This strongly penalizes probability mass placed by $p$ outside the support of $q_y$, producing sharper but less diverse samples. **Highest precision, lowest recall.**

#### Jensen-Shannon (Balanced Guidance)

Balances mode covering and mode seeking through an implicit mixture distribution $m = \frac{1}{2}(q_y + p)$. The weight $(q_y(i) - p(i|x))/m(i)$ is bounded and approaches zero when $q_y \approx p$, providing smooth gradient dynamics. Achieves the best empirical performance.

### 4. Tilted Sampling

Adjusts guidance weights using within-batch information:

$$\mathcal{S}_{\text{tilted}}(t; x, y) = \frac{1}{t} \log\left(\frac{1}{N}\sum_{i \in [N]} e^{t \log p_{\tau_1, \tau_2}(y|x)}\right)$$

$t > 0$ emphasizes high-confidence samples (improving quality); $t < 0$ emphasizes low-confidence samples (improving diversity).

## Key Experimental Results

### Smooth ECE Fine-tuning (10K ImageNet 128×128)

| Method | FID↓ | Precision↑ | Recall↑ |
|--------|------|-----------|---------|
| Standard fine-tuned classifier | 6.15 | 0.77 | **0.68** |
| **+Smooth ECE** | **5.94** | **0.79** | 0.66 |

~3% FID improvement with minimal fine-tuning.

### Sampling Guidance Comparison (10K samples, ResNet-50)

| Method | FID↓ | Precision↑ | Recall↑ |
|--------|------|-----------|---------|
| Baseline (ma2023) | 5.34 | **0.78** | 0.67 |
| Tilted sampling (t=-0.2) | 5.28 | 0.77 | 0.68 |
| Entropy regularization | 5.30 | 0.77 | **0.69** |
| **RKL guidance** | **5.12** | **0.78** | 0.68 |

### Comparison with SOTA (50K ImageNet 128×128)

| Method | Classifier | FID↓ | Precision↑ | Recall↑ |
|--------|-----------|------|-----------|---------|
| Dhariwal et al. | Fine-tuned | 2.97 | 0.78 | 0.59 |
| Entropy-aware classifier | Specialized | 2.68 | 0.80 | 0.56 |
| Classifier-free | - | 2.43 | - | - |
| ma2023 | ResNet-50 | 2.37 | 0.77 | 0.60 |
| FKL (ours) | ResNet-101 | 2.17 | **0.80** | 0.59 |
| RKL (ours) | ResNet-101 | 2.14 | 0.79 | 0.59 |
| **JS (ours)** | ResNet-101 | **2.13** | 0.79 | 0.60 |

JS divergence achieves the best FID of 2.13 without retraining the diffusion model or classifier.

### Divergence Characteristics Summary

| Divergence | FID | Precision | Recall | Characteristic |
|-----------|-----|-----------|--------|---------------|
| FKL | Mid | Highest | Lowest | Mode seeking, high sharpness |
| RKL | Mid-high | Mid | Mid | Mode covering, diversity preservation |
| **JS** | **Best** | Mid | **Highest** | Balanced coverage and seeking |

## Highlights & Insights

1. **Plug-and-play**: The f-divergence guidance method can be directly applied to off-the-shelf classifiers and diffusion models with zero retraining cost.
2. **Rigorous theoretical analysis**: Complete gradient derivations for all three f-divergences are provided (Proposition 2), along with closed-form analysis under a Gaussian mixture scenario (Proposition 3), establishing a solid mathematical foundation.
3. **Novel insight**: The superiority of JS divergence over both RKL and FKL challenges the conventional belief that mode-covering (RKL) guidance is optimal for generation.
4. **Smooth ECE loss** is simple yet effective, improving calibration and FID with only minimal fine-tuning.

## Limitations & Future Work

1. As a **workshop paper**, the experimental scope is limited—evaluation is restricted to ImageNet 128×128 without testing higher resolutions.
2. The f-divergence guidance hyperparameters ($\alpha, \epsilon, \tau_1, \tau_2$) require tuning, and sensitivity analysis is not sufficiently discussed.
3. The improvement from tilted sampling is marginal (FID 5.34 → 5.28), limiting its practical utility.
4. No comparison is made against more recent conditional generation methods (e.g., DiT + classifier-free guidance).
5. Smooth ECE fine-tuning and f-divergence sampling are not jointly evaluated.

## Related Work & Insights

| Dimension | Dhariwal et al. | Entropy-constrained | ma2023 | Ours |
|-----------|----------------|--------------------|----|------|
| Requires classifier retraining | ✓ | ✓ | ✗ | **✗** |
| Requires diffusion model retraining | ✗ | ✗ | ✗ | **✗** |
| Theoretical analysis | None | Limited | Energy perspective | **f-divergence framework** |
| FID (ResNet-101) | 2.97 | 2.68 | 2.19 | **2.13** |
| Diversity preservation | Weak | Moderate | Moderate | **Strong (JS/RKL)** |

**Additional insights:**

1. **Classifier confidence ≠ guidance strength**: Overconfident classifiers paradoxically provide weaker guidance, highlighting the importance of calibration for conditional generation.
2. **The f-divergence family defines a regularization design space**: Different divergences correspond to different precision-recall trade-offs, enabling application-specific selection.
3. **JS divergence's "symmetric penalty" mechanism**: Adaptive correction via the mixture distribution $m$ strengthens when predictions deviate from the target and diminishes at alignment—a mechanism potentially transferable to other guidance settings.
4. The tilted sampling approach of adaptive adjustment using batch-level information has potential connections to ensemble methods.

## Rating

- Novelty: ⭐⭐⭐⭐ — The f-divergence framework for diffusion guidance regularization is novel with solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Workshop-scale experiments limited to 128×128 resolution; large-scale validation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear, the Proposition chain is complete, and intuitive explanations are well-presented.
- Value: ⭐⭐⭐⭐ — Zero-cost plug-and-play guidance improvement has strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Increasing the Utility of Synthetic Images through Chamfer Guidance](increasing_the_utility_of_synthetic_images_through_chamfer_guidance.md)
- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)
- [\[NeurIPS 2025\] Tortoise and Hare Guidance: Accelerating Diffusion Model Inference with Multirate Integration](tortoise_and_hare_guidance_accelerating_diffusion_model_inference_with_multirate.md)
- [\[NeurIPS 2025\] Why Diffusion Models Don't Memorize: The Role of Implicit Regularization](why_diffusion_models_dont_memorize_the_role_of_implicit_regularization.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](token_perturbation_guidance_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
