---
title: >-
  [Paper Note] Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator
description: >-
  [ICML 2025 Spotlight][Image Generation][Generative Model Fine-Tuning] DDO proposes parameterizing the likelihood model itself as a GAN discriminator (via the likelihood ratio). This enables fine-tuning pre-trained diffusion/autoregressive models using GAN targets without an additional discriminator network, significantly improving the FID records on CIFAR-10 and ImageNet (EDM: 1.97 $\to$ 1.38, EDM2-S: 1.58 $\to$ 0.97).
tags:
  - "ICML 2025 Spotlight"
  - "Image Generation"
  - "Generative Model Fine-Tuning"
  - "GAN Discriminator"
  - "Likelihood Ratio Parameterization"
  - "Diffusion Models"
  - "Autoregressive Models"
date: 2026-05-08
content_hash: 6f6de72ce2183336
---

# Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is also a GAN Discriminator

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2503.01103](https://arxiv.org/abs/2503.01103)  
**Code**: [https://research.nvidia.com/labs/dir/ddo/](https://research.nvidia.com/labs/dir/ddo/)  
**Area**: Image Generation  
**Keywords**: Generative Model Fine-Tuning, GAN Discriminator, Likelihood Ratio Parameterization, Diffusion Models, Autoregressive Models

## TL;DR
DDO proposes parameterizing the likelihood model itself as a GAN discriminator (via the likelihood ratio). This enables fine-tuning pre-trained diffusion/autoregressive models using GAN targets without an additional discriminator network, significantly improving the FID records on CIFAR-10 and ImageNet (EDM: 1.97 $\to$ 1.38, EDM2-S: 1.58 $\to$ 0.97).

## Background & Motivation

**Background**: Diffusion models and autoregressive models are currently the mainstream paradigms for visual generation, renowned for their stability and scalability, and have achieved remarkable success in image and video synthesis.

**Limitations of Prior Work**: These likelihood-based models optimize the forward KL divergence $\min_\theta D_{\text{KL}}(p_{\text{data}} \| p_\theta)$, which naturally exhibits a "mode-covering" behavior. Under limited model capacity, the learned density tends to over-diffuse, leading to potentially blurry generated samples. Consequently, they heavily rely on guidance methods such as CFG to enhance generation quality.

**Key Challenge**: GANs optimize the JS divergence and tend to generate sharper, more realistic samples, but their training is unstable and prone to mode collapse. How can the sharpening advantages of GANs be introduced to pre-trained likelihood models while avoiding the engineering complexity of GANs?

**Goal**: To fine-tune pre-trained likelihood generative models using GAN objectives without altering network architectures or increasing inference costs, thereby exceeding the quality upper bound of MLE.

**Key Insight**: The likelihood model itself can serve as a GAN discriminator. By implicitly parameterizing the discriminator through the likelihood ratio of two likelihood models, it mimics how DPO parameterizes the reward model using policy log ratios.

**Core Idea**: Implicitly parameterize the discriminator as $d_\theta(\mathbf{x}) = \sigma\left(\log \frac{p_\theta(\mathbf{x})}{p_{\theta_{\text{ref}}}(\mathbf{x})}\right)$, simplifying GAN training into direct fine-tuning of the generative model itself.

## Method

### Overall Architecture

The pipeline of DDO is highly concise:
- **Input**: A pre-trained likelihood generative model $p_{\theta_{\text{ref}}}$ (acting as a frozen reference model) and a training dataset (real samples)
- **Process**: Initialize $\theta = \theta_{\text{ref}}$, generate fake samples using the reference model, and train $\theta$ with the GAN discriminator loss
- **Output**: The fine-tuned model $p_\theta$, which directly replaces the original model for inference with zero additional cost

### Key Designs

1. **Implicit Discriminator Parameterization**:

    - In standard GANs, the optimal discriminator is $d^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_{\theta_{\text{ref}}}(\mathbf{x})} = \sigma\left(\log \frac{p_{\text{data}}(\mathbf{x})}{p_{\theta_{\text{ref}}}(\mathbf{x})}\right)$
    - The core idea of DDO: replace the unknown $p_{\text{data}}$ with the learnable generative model $p_\theta$, defining the discriminator as $d_\theta(\mathbf{x}) = \sigma\left(\log \frac{p_\theta(\mathbf{x})}{p_{\theta_{\text{ref}}}(\mathbf{x})}\right)$
    - **Theoretical Guarantee**: The loss is minimized when $p_\theta^* = p_{\text{data}}$, meaning the optimal solution still matches the data distribution.
    - **Design Motivation**: This parameterization eliminates the need for an independent discriminator network as well as backpropagation through the generation process (which is highly expensive for diffusion models).

2. **Generalized Objective and Hyperparameter Control**:

    - Since $\log p_\theta(\mathbf{x})$ of likelihood models can reach the scale of $10^3$, directly applying Sigmoid results in vanishing gradients.
    - Introduce hyperparameters $\alpha, \beta$: $\mathcal{L}_{\alpha,\beta}(\theta) = -\mathbb{E}_{p_{\text{data}}}[\log \sigma(\beta \log \frac{p_\theta}{p_{\theta_{\text{ref}}}})] - \alpha \mathbb{E}_{p_{\theta_{\text{ref}}}}[\log(1 - \sigma(\beta \log \frac{p_\theta}{p_{\theta_{\text{ref}}}})]$
    - $\beta$ controls the scaling of the probability ratio, and $\alpha$ controls the relative weight of the two loss terms.
    - When $\beta < 1$, the optimal solution will "overshoot" the data distribution ($p_\theta^* \propto p_{\theta_{\text{ref}}}^{1-1/\beta} p_{\text{data}}^{1/\beta}$), which is theoretically aligned with guidance methods.

3. **Single-step Approximation for Diffusion Models**:

    - The likelihood ratio of diffusion models requires multi-step ELBO approximation: $\log \frac{p_\theta}{p_{\theta_{\text{ref}}}} \approx \mathbb{E}_{t,\epsilon}[\Delta_{\mathbf{x}_t, t, \epsilon}]$
    - Where $\Delta = -w(t)(||\epsilon_\theta(\mathbf{x}_t,t) - \epsilon||^2 - ||\epsilon_{\theta_{\text{ref}}}(\mathbf{x}_t,t) - \epsilon||^2)$
    - An upper bound is derived using Jensen's inequality, requiring only one forward pass per sample.
    - **Design Motivation**: To avoid the high cost of multi-step backpropagation, making the computational cost of diffusion DDO comparable to standard training.

4. **Multi-round Self-Play Refinement**:

    - After each round of fine-tuning, the optimal model is treated as the reference model for the next round: $p_{\theta_{n-1}^*} \to p_{\theta_n}$
    - This is similar to Iterative DPO and SPIN, but does not directly update the reference model.
    - Each round requires less than 1% of the pre-training iterations.
    - **Design Motivation**: A single round of DDO provides useful gradients but does not converge to the data distribution; multi-round iterations gradually approach the target.

### Loss & Training

- **Diffusion Models (EDM-DDO)**: Leveraging F-parameterization, the loss is:
  $$\mathcal{L}_{\alpha,\beta}^{\text{EDM-DDO}} = -\mathbb{E}_{t,\epsilon}[\mathbb{E}_{p_{\text{data}}} \log \sigma(-\beta(\|F_\theta - \hat{F}\|^2 - \|F_{\theta_{\text{ref}}} - \hat{F}\|^2)) + \alpha \mathbb{E}_{p_{\theta_{\text{ref}}}} \log \sigma(\beta(\|F_\theta - \hat{F}\|^2 - \|F_{\theta_{\text{ref}}} - \hat{F}\|^2))]$$
- **Autoregressive Models (VAR)**: Directly use the next-token log-likelihood ratio, construct reference samples online, and retain label dropout for compatibility with CFG.
- Disable mixed precision (for diffusion models) to maintain numerical stability; disable all dropout layers.
- Perform grid search for each round in the range of $\alpha \in [0.5, 6.0], \beta \in [0.01, 0.1]$ (for diffusion) or $\alpha \in [10, 100], \beta = 0.02$ (for VAR).

## Key Experimental Results

### Main Results

**CIFAR-10 (FID↓)**

| Method | NFE | Unconditional FID | Class-Conditional FID |
|------|-----|-----------|-----------|
| EDM (Baseline) | 35 | 1.97 | 1.85 |
| EDM + DG | 53 | 1.77 | 1.64 |
| **EDM + DDO** | **35** | **1.38** | **1.30** |
| StyleGAN-XL | 1 | - | 1.85 |

**ImageNet-64 (FID↓)**

| Method | NFE | FID |
|------|-----|-----|
| EDM2-S (Baseline) | 63 | 1.58 |
| EDM2-S + AG | 126 | 1.01 |
| **EDM2-S + DDO** | **63** | **0.97** |
| EDM2-XL | 63 | 1.33 |

**ImageNet 256×256 (VAR, FID↓)**

| Method | w/o CFG | w/ CFG |
|------|---------|--------|
| VAR-d30 (w/ tricks) | 2.17 | 1.90 |
| VAR-d30 (w/o tricks) | 4.74 | 1.92 |
| **VAR-d30 + DDO** | **1.79** | **1.73** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| $\alpha \in [0.5, 6.0]$, $\beta = 0.05$ | FID improved across all | Consistent effectiveness over a wide range of $\alpha$ |
| $\beta \in [0.01, 0.1]$, $\alpha = 4.0$ | FID improved across all | Consistent effectiveness over a wide range of $\beta$ |
| Single-round DDO (CIFAR-10) | FID 1.72/1.58 | Single round outperforms DG |
| EDM2-S 3 rounds (ImageNet-64) | FID 1.31 | 280M model outperforms the 1119M EDM2-XL |
| VAR-d16 + DDO w/o CFG | FID 3.12 $\to$ outperforms CFG baseline 3.30 | Halves inference cost |

### Key Findings

- **Record-breaking FID**: CIFAR-10 1.30, ImageNet-64 0.97, establishing new SOTAs.
- **Astonishing Efficiency**: Each round of fine-tuning requires $<1\%$ of pre-training iterations, taking approximately 3 hours per round for EDM.
- **Eliminating Sampling Tricks**: VAR fine-tuned with DDO achieves superior FID without requiring top-k/top-p.
- **Eliminating CFG Dependency**: VAR-d30 + DDO achieves an unguided FID of 1.79, outperforming the original CFG-enhanced baseline of 1.90.
- **Parameter Efficiency**: EDM2-S (280M) + DDO outperforms the 4x larger EDM2-XL (1119M).

## Highlights & Insights

1. **Elegant Theoretical Framework**: Parameterizing the discriminator with likelihood ratios establishes a profound connection between DPO and GANs.
2. **Unified Perspective with Guidance Methods**: DDO ($\beta < 1$) is equivalent to $p_\theta^* \propto p_{\text{ref}}^{1-1/\beta} p_{\text{data}}^{1/\beta}$, which is inherently identical to the "distribution sharpening" of CFG/AG.
3. **Zero Inference Overhead**: Unlike DG/AG/CFG which require auxiliary models or multiple forward passes, DDO directly replaces the original model.
4. **Generality**: The same framework is simultaneously applicable to continuous (diffusion) and discrete (autoregressive) generative models.

## Limitations & Future Work

1. Hyperparameters $\alpha, \beta$ require grid searching; automated tuning strategies are currently lacking.
2. Multi-round refinement introduces additional training costs (although each round is short), requiring parallel search across ~20 nodes.
3. Currently only validated on class-conditional image generation, not yet extended to more complex tasks such as text-to-image.
4. Theoretical analysis relies on the bounded likelihood ratio assumption, which might not hold under strong distribution shifts.

## Related Work & Insights

- **DPO to DDO Transfer**: DPO uses log-policy ratios to parameterize the reward model $\to$ DDO uses log-likelihood ratios to parameterize the discriminator. However, DDO focuses on distribution alignment rather than preference learning.
- **Complementary to Distillation**: DDO enhances model quality, while distillation improves inference speed; the two can be cascaded.
- **Insight**: The paradigm of "model as discriminator" could be extended to other modalities, such as audio, 3D, and video.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The insight of implicitly parameterizing the likelihood model as a GAN discriminator is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive dual validation on diffusion and autoregressive models, with multi-dataset and multi-round ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theory, clear motivations, and fluent exposition.
- Value: ⭐⭐⭐⭐⭐ Provides a concise and unified recipe for post-training generation quality enhancement, featuring high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Boost Your Human Image Generation Model via Direct Preference Optimization](../../CVPR2025/image_generation/boost_your_human_image_generation_model_via_direct_preference_optimization.md)
- [\[ICML 2025\] Discriminative Policy Optimization for Token-Level Reward Models](discriminative_policy_optimization_for_token-level_reward_models.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/image_generation/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[CVPR 2025\] Training Data Provenance Verification: Did Your Model Use Synthetic Data from My Generative Model for Training?](../../CVPR2025/image_generation/training_data_provenance_verification_did_your_model_use_synthetic_data_from_my_.md)

</div>

<!-- RELATED:END -->
