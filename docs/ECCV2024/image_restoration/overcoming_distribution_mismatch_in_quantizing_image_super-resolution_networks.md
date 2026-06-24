---
title: >-
  [Paper Note] Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks
description: >-
  [ECCV 2024][Image Restoration][image super-resolution] This paper proposes the ODM framework. By employing two simple strategies—cooperative mismatch regularization and layer-wise weight clipping correction—it resolves the distribution mismatch problem in SR network quantization without introducing dynamic modules during inference, achieving state-of-the-art (SOTA) performance with minimal extra overhead.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "image super-resolution"
  - "network quantization"
  - "distribution mismatch"
  - "quantization-aware training"
  - "gradient conflict"
date: 2026-05-08
content_hash: c8b2ad342ce8ce2a
---

# Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks

**Conference**: ECCV 2024  
**arXiv**: [2307.13337](https://arxiv.org/abs/2307.13337)  
**Code**: [https://github.com/Cheeun/ODM](https://github.com/Cheeun/ODM)  
**Area**: Image Restoration  
**Keywords**: image super-resolution, network quantization, distribution mismatch, quantization-aware training, gradient conflict

## TL;DR

This paper proposes the ODM framework. By employing two simple strategies—cooperative mismatch regularization and layer-wise weight clipping correction—it resolves the distribution mismatch problem in SR network quantization without introducing dynamic modules during inference, achieving state-of-the-art (SOTA) performance with minimal extra overhead.

## Background & Motivation

**Background**: Super-resolution (SR) networks (such as EDSR, RDN, SwinIR) exhibit excellent performance, but their heavy computational load limits deployment on resource-constrained platforms, such as mobile devices. Network quantization significantly reduces computational costs by mapping 32-bit floating-point operations to low-bit integer operations, a method that has already been validated in high-level vision tasks (e.g., classification).

**Limitations of Prior Work**: The performance of SR networks drops drastically when quantized to low-bit representations. The root cause is that SR networks typically do not use BatchNorm, which leads to massive variations in feature (activation) distributions across different channels and inputs (distribution mismatch). Consequently, a single quantization range $[l, u]$ cannot accommodate all distributions.

**Key Challenge**: Existing methods (e.g., DAQ, DDTB) use dynamic modules to adaptively adjust the quantization range during inference to fit different distributions. However, these dynamic modules introduce extra computational overhead (additional bitOPs and memory storage), undermining the computational efficiency gains brought by quantization. This is fundamentally a trade-off between "quantization accuracy vs. inference efficiency."

**Goal**: Without using any dynamic modules at inference time, pre-adjust the feature distribution to be quantization-friendly via regularization strategies during the training phase, while simultaneously determining more accurate layer-wise weight quantization ranges.

**Key Insight**: Looking through the lens of gradient conflict in multi-task learning—directly adding mismatch regularization creates a gradient conflict with the reconstruction loss, pushing about half of the parameters in opposite directions. Therefore, a cooperative strategy based on gradient cosine similarity is introduced to apply regularization only when the two loss gradient directions are aligned.

**Core Idea**: Use cooperative regularization during training to pre-shape the SR feature distributions to be quantization-friendly, instead of using dynamic modules during inference to adapt to unfriendly distributions.

## Method

### Overall Architecture

ODM is a quantization-aware training (QAT) framework applicable to any SR network (e.g., EDSR, RDN, SwinIR):
1. Start with a pre-trained 32-bit SR network.
2. Quantize the activations and weights of each convolutional/linear layer using a uniform quantization function $q(\mathbf{X}_i; l, u) = \text{Int}\left(\frac{\text{clip}(\mathbf{X}_i, l, u) - l}{s}\right) \cdot s + l$, where $s = \frac{u - l}{2^b - 1}$.
3. Alternately optimize both the reconstruction loss $\mathcal{L}_R$ and the mismatch regularization loss $\mathcal{L}_M$ during training, avoiding gradient conflict through a cooperative strategy.
4. Dynamically adjust the weight quantization range via layer-wise correction parameters $\gamma_w$.
5. Ensure zero extra modules during inference, as quantization ranges and correction parameters can be pre-computed.

### Key Designs

1. **Cooperative Mismatch Regularization**:

    - **Function**: Regularizes the distance (mismatch) between features of each layer and their quantization grids during training so that the feature distributions are pre-conditioned to be quantization-friendly.
    - **Mechanism**: Mismatch is defined as $M(\mathbf{X}_i) = \|\mathbf{X}_i - q(\mathbf{X}_i; l_a, u_a)\|_2$, and the total mismatch loss is $\mathcal{L}_M = \sum_i^{\#\text{layers}} M(\mathbf{X}_i)$. However, directly optimizing $\mathcal{L}_R + \mathcal{L}_M$ jointly leads to severe gradient conflict (about 50% of the parameters have opposing directions). Hence, a cooperative strategy is adopted:
    $$\theta^{t+1} = \theta^t - \beta^t \cdot \left(\lambda_R \nabla_\theta \mathcal{L}_R + \lambda_M \cdot \text{sim}(\nabla_\theta \mathcal{L}_R, \nabla_\theta \mathcal{L}_M) \cdot \nabla_\theta \mathcal{L}_M\right)$$
   where $\text{sim}(\mathbf{v}_a, \mathbf{v}_b) = \frac{\cos(\mathbf{v}_a, \mathbf{v}_b) + 1}{2} \in [0, 1]$. When the two gradients align, the similarity score is close to 1 (fully leveraging the regularization), and when they oppose, it approaches 0 (ignoring the regularization to follow the reconstruction loss).
    - **Design Motivation**: Experimental observations (Fig. 2) show that the gradient cosine similarity frequently becomes negative during naive joint optimization, with around 50% of parameters suffering from gradient conflict. Directly adding $\mathcal{L}_M$ makes the distribution quantization-friendly but deviates from the high-density regions of the original 32-bit distribution (near 0), causing a drop in reconstruction accuracy (decreasing by 0.13dB in ablation studies). The cooperative strategy balances both quantization friendliness and reconstruction accuracy.

2. **Weight Clipping Correction**:

    - **Function**: Determines more accurate symmetric quantization ranges $[-u_w, u_w]$ for each layer's weights, replacing a globally uniform strategy.
    - **Mechanism**: Existing methods use a fixed global strategy $u_w^t = f(\mathbf{W}^t)$ (such as max) to determine the quantization range for all layers. This paper introduces layer-wise learnable correction parameters $\gamma_w$:
    $$u_w^{t+1} = f(\mathbf{W}^{t+1}) \cdot \left(\gamma_w^t - \beta^t \cdot \nabla_{\gamma_w} \mathcal{L}_R(\gamma_w^t)\right)$$
   $\gamma_w$ is initialized to 1, and $f(\cdot)$ uses the 99th percentile function instead of max to prevent outliers from dominating the range.
    - **Design Motivation**: (a) Weight distributions vary significantly across different layers; a uniform max strategy causes the quantization grids of some layers to be stretched by outliers into low-density regions (Fig. 3), which formulaically limits information capacity under low-bit configurations. (b) A purely learnable $u_w$ has the drawback of optimizing based on the previous weight step $\mathbf{W}^{t-1}$ while quantizing the current weight $\mathbf{W}^t$, introducing a lag. The correction factor $\gamma_w$ disentangles the "current weight-based global strategy" and "layer-wise adjustment". The extra overhead is only a single scalar parameter per layer (0.06K for EDSR, 0.15K for RDN), which can be pre-computed into the quantization ranges for inference, yielding zero additional bitOPs.

3. **Quantization Range Initialization and Update**:

    - **Function**: Provides a reasonable initialization and learnable updates for the activation quantization range $[l_a, u_a]$.
    - **Mechanism**: Initialize $l_a, u_a$ using the average of the 1st and 99th percentiles of the training data (to avoid outliers), followed by end-to-end learning via straight-through estimators (STE). The clipping parameters are updated with a larger learning rate of $10 \times \beta^0$.
    - **Design Motivation**: Percentile initialization is more robust than min/max, and a larger learning rate allows clipping parameters to rapidly adapt to changes in the distribution.

### Loss & Training

- **Loss Function**: Reconstruction loss $\mathcal{L}_R = \mathcal{L}_1(\mathcal{Q}(\mathbf{I}_{LR}), \mathbf{I}_{HR})$ + Cooperative mismatch regularization $\mathcal{L}_M$.
- **Hyperparameters**: $\lambda_R = 1$, $\lambda_M = 10^{-5}$ (RDN uses $10^{-6}$ because its overall mismatch is larger), percentile $j = 99$.
- **Training Configuration**: DIV2K dataset, 60K iterations, batch size 8, initial learning rate $10^{-4}$ halved every 15K iterations, RTX 2080Ti GPU.

## Key Experimental Results

### Main Results (EDSR x4 Quantization)

| Method | Bit | Set5 PSNR↑ | Set5 SSIM↑ | Set14 PSNR↑ | Urban100 PSNR↑ | Urban100 SSIM↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| EDSR (FP32) | 32 | 32.10 | 0.894 | 28.58 | 26.04 | 0.785 |
| EDSR-PAMS | 4 | 31.59 | 0.885 | 28.20 | 25.32 | 0.762 |
| EDSR-DAQ | 4 | 31.85 | 0.887 | 28.38 | 25.73 | 0.772 |
| EDSR-DDTB | 4 | 31.85 | 0.889 | 28.39 | 25.69 | 0.774 |
| **EDSR-ODM** | **4** | **32.00** | **0.891** | **28.47** | **25.80** | **0.778** |
| EDSR-PAMS | 2 | 29.51 | 0.835 | 26.79 | 23.72 | 0.688 |
| EDSR-DAQ | 2 | 31.01 | 0.871 | 27.89 | 24.88 | 0.740 |
| EDSR-DDTB | 2 | 30.97 | 0.876 | 27.87 | 24.82 | 0.742 |
| **EDSR-ODM** | **2** | **31.50** | **0.882** | **28.14** | **25.17** | **0.755** |

### Ablation Study (EDSR 2-bit x4)

| Model | WCC | Cooperative | MR | Set5 PSNR / SSIM | Urban100 PSNR / SSIM |
|:---|:---:|:---:|:---:|:---:|:---:|
| (a) Baseline | - | - | - | 29.94 / 0.848 | 23.99 / 0.703 |
| (b) +Cooperative MR | - | ✓ | ✓ | 30.34 / 0.859 | 24.27 / 0.715 |
| (c) +WCC | ✓ | - | - | 31.12 / 0.876 | 24.91 / 0.746 |
| (d) +WCC+Naive MR | ✓ | - | ✓ | 30.99 / 0.871 | 24.79 / 0.735 |
| **(e) ODM** | **✓** | **✓** | **✓** | **31.50 / 0.882** | **25.17 / 0.755** |

### Key Findings

- **Most significant gains in 2-bit scenarios**: On EDSR 2-bit, ODM outperforms the second-best method (DAQ) by 0.49dB (Set5), RDN 2-bit by 0.66dB, and SwinIR 2-bit by 0.43dB. This shows that the distribution mismatch problem is more severe at lower bit-widths, and the benefits of ODM are larger.
- **4-bit performance close to FP32**: EDSR-ODM 4-bit is only 0.1dB behind FP32 on Set5, indicating that ODM effectively bridges the gap between quantized and full-precision networks.
- **Naive MR is counterproductive**: Directly optimizing $\mathcal{L}_R + \mathcal{L}_M$ jointly (Model d) achieves a PSNR 0.13dB lower than only using WCC (Model c), verifying that gradient conflict indeed hampers reconstruction.
- **Cooperative strategy turns the tide**: Model (e) outperforms Model (d) by 0.51dB, proving the effectiveness of filtering via gradient alignment.
- **WCC makes the largest contribution**: Model (c) vs. (a) shows an improvement of 1.18dB, indicating that the precise choice of layer-wise weight quantization ranges is crucial.
- **Minimal computational overhead**: The extra storage on EDSR is only 0.06K (vs. 1.9K for DDTB), with zero additional bitOPs (vs. 0.5T for DAQ), making it the only method that does not require channel-wise quantization or dynamic modules.
- **Generalizable across architectures**: Effective for both CNNs (EDSR, RDN) and Transformers (SwinIR), as well as both x2 and x4 scale factors.

## Highlights & Insights

1. **Solving at training time vs. Remedying at inference time**: Paradigm shift—instead of using more complex quantization functions to adapt to unfriendly distributions during inference, the distribution is pre-adjusted to be quantization-friendly during the training phase. This guarantees that inference efficiency is not compromised.
2. **Discovering and resolving gradient conflicts**: Drawing on gradient conflict theory from multi-task learning, the two objectives are cooperatively optimized using cosine similarity weighting. This concept can be extended to other scenarios requiring regularization.
3. **Intuitive explanation via distribution visualization**: Fig. 5 clearly shows the differences in activation distributions under the three training strategies—naive MR causes multi-modal distributions that deviate from the original high-density regions, while cooperative MR reduces outliers while maintaining the original distribution structure.
4. **Simplistic nature of the solution**: The entire framework does not require modifying the network architecture; it only introduces one loss term during training and one scalar parameter per layer, making it easy to integrate into existing QAT pipelines.

## Limitations & Future Work

1. **Only verified on a subset of SR networks**: EDSR, RDN, and SwinIR are classical networks; the performance on newer, larger SR models (such as HAT, SRFormer) remains unknown.
2. **Extra training overhead from gradient similarity computation**: Computing gradients for two losses separately before calculating cosine similarity increases training time (the paper does not report training time comparison).
3. **Hyperparameter $\lambda_M$ requires tuning for different networks**: RDN and EDSR use different values of $\lambda_M$; automated hyperparameter tuning strategies could be further explored.
4. **Initialization of layer-wise correction parameters**: Fixed initialization of $\gamma_w$ to 1 may not be optimal; adaptive initialization based on layer-wise distribution statistics could be considered.
5. **No exploration of mixed-precision**: All layers use the same bit-width; combining ODM with mixed-precision allocation strategies might further improve the efficiency-accuracy trade-off.
6. **Only L1 reconstruction loss is used**: The effects of combining this with perceptual losses or GAN losses have not been explored.

## Related Work & Insights

- **PAMS [Li et al., ECCV 2020]**: The first work to learn clipping parameters for SR quantization, serving as one of the baselines in this paper, but it cannot handle distribution mismatch along the channel or image dimensions.
- **DAQ [Hong et al., WACV 2022]**: Introduces channel-wise dynamic quantization functions. Though effective, it increases bitOPs. This paper proves that surpassing DAQ is possible without channel-wise quantization.
- **DDTB [Zhong et al., ECCV 2022]**: Uses input-adaptive dynamic modules, which incur large storage overhead (31.7x to 304.7x larger than ODM).
- **Gradient Conflict [Du et al., 2018]**: A gradient similarity weighting strategy for auxiliary losses in multi-task learning, which this paper creatively applies to quantization training.
- **QuantSR [Qin et al., NeurIPS 2024]**: A parallel work that introduces extra transformation functions in both forward and backward passes. ODM shows clear advantages at 2-bit (e.g., +0.51dB on SRResNet).
- **Insight**: The idea of "distribution friendliness" can be extended to other low-level vision tasks (denoising, deblurring, etc.), and gradient-conflict-aware regularization strategies have general value.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Approaching quantization regularization from the perspective of gradient conflict is a fresh angle, and the paradigm shift of solving inference issues during training is valuable, though the core technique (gradient cosine weighting) is borrowed from multi-task learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Highly comprehensive, covering three architectures (CNN x2 + Transformer), three bit-widths (4/3/2), four datasets, complete ablation studies, complexity analysis, and distribution visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definitions, logical progression (Observation $\rightarrow$ Motivation $\rightarrow$ Solution), supported well by distribution plots and gradient conflict illustrations.
- **Value**: ⭐⭐⭐⭐ High practical value—an SR quantization scheme without inference overhead has direct deployment significance, and the dramatic improvement in 2-bit scenarios is crucial for edge devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Accelerating Image Super-Resolution Networks with Pixel-Level Classification](accelerating_image_super-resolution_networks_with_pixel-level_classification.md)
- [\[ECCV 2024\] TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](ttt-mim_test-time_training_with_masked_image_modeling_for_denoising_distribution.md)
- [\[ECCV 2024\] Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)
- [\[ECCV 2024\] Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)
- [\[ECCV 2024\] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)

</div>

<!-- RELATED:END -->
