---
title: >-
  [Paper Note] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification
description: >-
  [CVPR 2025][Image Generation][Adversarial Purification] A heterogeneous noise diffusion purification strategy based on attention masks is proposed. It applies high-intensity noise to crucial pixels focused on by the classifier to eradicate adversarial perturbations while applying low-intensity noise to the remaining regions to preserve semantic information, significantly reducing computational overhead via single-step resampling.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Adversarial Purification"
  - "Diffusion Models"
  - "Heterogeneous Noise"
  - "Attention Mask"
  - "Adversarial Defense"
date: 2026-05-08
content_hash: eff3331ff7c16bd4
---

# Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification

**Conference**: CVPR 2025  
**arXiv**: [2503.01407](https://arxiv.org/abs/2503.01407)  
**Code**: [GitHub](https://github.com/GaozhengPei/Purification)  
**Area**: Image Generation/Adversarial Robustness  
**Keywords**: Adversarial Purification, Diffusion Models, Heterogeneous Noise, Attention Mask, Adversarial Defense

## TL;DR

A heterogeneous noise diffusion purification strategy based on attention masks is proposed. It applies high-intensity noise to crucial pixels focused on by the classifier to eradicate adversarial perturbations while applying low-intensity noise to the remaining regions to preserve semantic information, significantly reducing computational overhead via single-step resampling.

## Background & Motivation

Existing diffusion-based adversarial purification methods (such as DiffPure) uniformly inject noise of the same intensity to all pixels during the forward process, and then reconstruct clean images via the reverse process. However, this uniform operation exhibits a fundamental contradiction:

- **Overly strong noise**: Successfully removes adversarial perturbations but simultaneously destroys semantic information, causing the classifier to yield incorrect predictions for both clean and adversarial samples.
- **Overly weak noise**: Fails to effectively eliminate adversarial perturbations, leaving the attack successful.
- Striking a balance between perturbation elimination and semantic preservation remains the core challenge.

From the perspective of neural network interpretability, a classifier assigns varying attention weights to different regions of an image when making decisions. Adversarial perturbations have the most significant impact on highly-attended regions. Thus, noise of different intensities can be targetedly applied to different areas.

Furthermore, existing multi-step resampling methods are computationally expensive, making it extremely difficult to evaluate the full gradient of strong adaptive attacks on consumer-grade GPUs, which is also an urgent problem to be solved.

## Method

### Overall Architecture

Given an adversarial image $\boldsymbol{x}_{adv}$, the model first obtains the attention maps of each layer through forward propagation of the classifier to construct a binary attention mask $\mathcal{M}$. Next, it executes a heterogeneous forward process to inject high/low noise into the attended/unattended regions, respectively. Finally, a clean image is restored via a two-stage denoising process.

### Key Design 1: Heterogeneous Forward Process

- **Function**: Applies varying noise intensities to different regions of the image, balancing perturbation elimination and semantic preservation.
- **Mechanism**: Extracts the activation outputs of various blocks in the classifier, aggregates them via the $L^p$ norm, and upsamples them to spatial size. After normalizing with spatial softmax, they are binarized using a threshold $\tau$. The union of masks across all blocks yields $\mathcal{M} = \bigcup_{m=1}^{M} \mathbb{I}[\text{AM}_m > \tau]$. High-attention regions receive strong noise with a large timestep $t_l$, while low-attention regions receive weak noise with a small timestep $t_s$, resulting in the fused noisy image $\boldsymbol{x}(t_l, t_s) = \boldsymbol{x}(t_l) \odot \mathcal{M} + \boldsymbol{x}(t_s) \odot (1-\mathcal{M})$.
- **Design Motivation**: Adversarial perturbations damage classifier decisions most in highly-attended regions, where high-intensity noise can effectively eliminate them. In contrast, low-attention areas have less impact on classification results, meaning weak noise is sufficient and helps retain semantic information.

### Key Design 2: Two-stage Denoising Process

- **Function**: Restores clean images from heterogeneously noisy images.
- **Mechanism**: Stage 1 ($t_s < t < t_l$) is formulated as an inpainting problem. Regions outside the mask are directly obtained by adding noise to the original image to get the known pixels $\boldsymbol{x}(t)^{known}$, while regions inside the mask are reconstructed using the reverse process of the diffusion model to predict $\boldsymbol{x}(t)^{unknown}$, which are then combined using the mask. Stage 2 ($t < t_s$) has a uniform noise intensity, where standard denoising sampling is performed.
- **Design Motivation**: Since different regions in heterogeneously noisy images have distinct noise levels, they cannot be directly restored using standard denoising. Performing inpainting first resolves local high-intensity noise, followed by global standard denoising to eliminate the remaining weak noise.

### Key Design 3: Single-step Resampling

- **Function**: Resolves semantic inconsistency at the mask boundaries while dramatically reducing computational overhead.
- **Mechanism**: Replaces the multi-step resampling of repeating $U=20$ times in RePaint by: diffusing $\boldsymbol{x}(t)$ one step back to $\boldsymbol{x}(t+U)$ and then denoising back to $\boldsymbol{x}(t)$ in a single DDIM step, requiring only one additional denoising network forward pass per timestep.
- **Design Motivation**: Multi-step resampling incurs extreme GPU memory and time costs due to the need to store massive computation graphs. Single-step resampling achieves approximately 90% savings in both time and memory, enabling the evaluation of full-gradient adaptive attacks on a single 24GB GPU.

### Loss & Training

No extra training is required; the pretrained diffusion model is directly customized for inference-time purification.

## Key Experimental Results

### Main Results: CIFAR-10 $\ell_\infty$ ($\epsilon=8/255$) WideResNet-28-10

| Method | Type | Standard Acc. | Robust (AutoAttack) |
|------|------|--------------|-------------------|
| Gowal et al. | AT | 88.54 | 63.38 |
| Bai et al. | AP | 91.41 | 77.08 |
| Lee et al. | AP | 90.16 | 70.47 |
| Lin et al. | AP | 90.62 | 72.85 |
| **Ours** | AP | **93.16** | **80.45** |

### CIFAR-10 $\ell_\infty$ WideResNet-70-16

| Method | Type | Standard Acc. | Robust (AutoAttack) |
|------|------|--------------|-------------------|
| Rebuffi et al. (AT†) | AT | 92.22 | 66.56 |
| Bai et al. | AP | 92.97 | 79.10 |
| Lin et al. | AP | 91.99 | 76.37 |
| **Ours** | AP | **93.36** | **84.83** |

### Ablation Study (Efficiency Comparison)

| Resampling Method | Semantic Consistency | Extra Forward Passes per Step | GPU Memory Requirement |
|-----------|----------|------------------|------------|
| Multi-step Resampling $U=20$ | ✓ | 20 | Extremely High |
| Single-step Resampling $U=10$ | ✓ | 1 | Feasible on 24GB |
| No Resampling | ✗ (Boundary artifacts) | 0 | Lowest |

### Key Findings

- On WRN-70-16, the AutoAttack robust accuracy reaches 84.83%, surpassing the best AT baseline (66.56%) and the best AP baseline (79.10%).
- The standard accuracy of 93.36% also outperforms all compared methods, verifying that the purification process does not degrade clean samples.
- Single-step resampling successfully achieves semantic consistency at $U=10$, yielding an approximate 90% reduction in computational overhead.

## Highlights & Insights

1. **Intelligent Heterogeneous Noise Design**: Leverages the classifier's own attention maps as priors to guide localized treatment, effectively converting the global "noise-semantic trade-off" paradox into spatial separation.
2. **Denoising from an Inpainting Perspective**: Elegantly reformulates the heterogenous noise restoration process as an image inpainting task, leading to solid theoretical backing and a simple implementation.
3. **High Engineering Value of Single-step Resampling**: For the first time, a diffusion-based purification approach is capable of computing complete adaptive attack gradients on consumer-grade GPUs.

## Limitations & Future Work

- The quality of the attention mask is heavily dependent on the classifier's architecture; different networks may require tailored thresholds $\tau$ and noise levels $t_l, t_s$.
- The method was only evaluated on image classification tasks and has not yet been extended to other downstream tasks such as object detection or segmentation.
- If an adversary operates under a white-box assumption regarding the mask generation mechanism, they might design targeted attacks to bypass the defense.
- Scalability to higher-resolution images remains to be verified.

## Related Work & Insights

- **DiffPure**: The pioneering diffusion-based purification method, which, however, utilizes uniform noise levels.
- **GDMP**: Introduces contrastive loss guidance but is still limited by uniform noise addition.
- **RePaint**: Utilizes multi-step resampling to address boundary issues at extreme computational costs.
- The core concept of localized, heterogeneous processing can be readily extended to other image restoration tasks requiring spatial differentiations.

## Rating

⭐⭐⭐⭐ — The heterogeneous noise scheme is highly novel and intuitive, with experimental results significantly outperforming existing baselines. The engineering contribution of single-step resampling is highly practical. However, requiring the classifier's attention map as an extra input slightly increases deployment complexity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [\[CVPR 2025\] Enhancing Facial Privacy Protection via Weakening Diffusion Purification](enhancing_facial_privacy_protection_via_weakening_diffusion_purification.md)
- [\[CVPR 2025\] IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation](idprotector_an_adversarial_noise_encoder_to_protect_against_id-preserving_image_.md)
- [\[CVPR 2025\] Improving Diffusion Inverse Problem Solving with Decoupled Noise Annealing](improving_diffusion_inverse_problem_solving_with_decoupled_noise_annealing.md)
- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)

</div>

<!-- RELATED:END -->
