---
title: >-
  [Paper Note] Spatially-Variant Degradation Model for Dataset-free Super-resolution
description: >-
  [ECCV 2024][Image Restoration][Blind Image Super-Resolution] Proposing SVDSR, the first dataset-free spatially-variant degradation model. The degradation kernel of each pixel is represented as a linear combination of a learnable atomic kernel dictionary. The coefficient matrix is derived from image texture information via membership functions in fuzzy set theory, and inferred under the MAP framework using the Monte Carlo EM algorithm. It achieves an average improvement of aro…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Blind Image Super-Resolution"
  - "Spatially-Variant Degradation"
  - "Dataset-Free"
  - "Blur Kernel Estimation"
  - "Monte Carlo EM"
date: 2026-05-08
content_hash: ab2103db8e1d9903
---

# Spatially-Variant Degradation Model for Dataset-free Super-resolution

**Conference**: ECCV 2024  
**arXiv**: 2407.08252 (https://arxiv.org/abs/2407.08252)  
**Code**: [https://github.com/shaojieguoECNU/SVDSR](https://github.com/shaojieguoECNU/SVDSR)  
**Area**: Image Restoration  
**Keywords**: Blind Image Super-Resolution, Spatially-Variant Degradation, Dataset-Free, Blur Kernel Estimation, Monte Carlo EM

## TL;DR

Proposing SVDSR, the first dataset-free spatially-variant degradation model. The degradation kernel of each pixel is represented as a linear combination of a learnable atomic kernel dictionary. The coefficient matrix is derived from image texture information via membership functions in fuzzy set theory, and inferred under the MAP framework using the Monte Carlo EM algorithm. It achieves an average improvement of around 1 dB in $2\times$ super-resolution.

## Background & Motivation

The core challenge of blind image super-resolution (BISR) lies in accurately estimating the unknown degradation operator $\mathcal{D}$. The degradation model can be formulated as $\boldsymbol{y} = (\mathcal{D}\boldsymbol{x})\downarrow_s + \boldsymbol{n}$. Existing methods are limited in two aspects:

**Spatially-Invariant Assumption**: Most methods assume a uniform degradation kernel for the entire image, whereas in real-world scenarios, degradation varies significantly between flat and dense texture regions.

**Dataset Dependency**: Spatially-variant degradation methods (e.g., KOALA, DARM, LARPAR) rely on training with large-scale paired datasets, which reduces their practicality.

This work bridges these two aspects: presenting the **first BISR method that simultaneously achieves spatially-variant degradation modeling and dataset-free deep learning**. Compared to prior spatially-variant methods that utilize 72 pre-defined atomic kernels, this approach requires only 5 learnable atomic kernels (with only 3 parameters each), significantly reducing the number of parameters.

## Method

### Overall Architecture

The overall pipeline is based on the MAP framework + MCEM inference:
1. Construct a spatially-variant degradation model: a learnable atomic kernel dictionary + a texture-based coefficient matrix.
2. Design a probabilistic BISR model: incorporating dual-domain likelihood in both spatial and frequency domains, kernel priors, and image priors.
3. Alternatingly execute the E-step (sampling the latent variable $\boldsymbol{z}$) and the M-step (updating kernel parameters $\boldsymbol{\Gamma}$ and network weights $\boldsymbol{\phi}$).

### Key Designs

1. **Spatially-Variant Degradation Model**: Based on O'Leary's decomposition formula, the degradation of each pixel $[h,w]$ is represented as a weighted combination of $N_{\mathcal{D}}$ atomic kernels: $(\mathcal{D}\boldsymbol{x})[h,w] = \sum_{r,c}\sum_{i=1}^{N_{\mathcal{D}}} \boldsymbol{W}_i[h,w] \mathcal{D}_i \boldsymbol{x}[h-r,w-c]$. Each atomic kernel $\mathcal{D}_i$ is an anisotropic Gaussian kernel, which requires only 3 learnable parameters $\{\theta_i, \sigma_{i,1}, \sigma_{i,2}\}$ (rotation angle + two standard deviations) through decomposition. This is much more flexible and has significantly fewer parameters than the 72 pre-defined kernels used in previous approaches.

2. **Fuzzy Set Coefficient Matrix**: Instead of being learned by a neural network, the coefficient matrix $\boldsymbol{W}_i$ is derived from image textures using membership functions from fuzzy set theory: $\boldsymbol{W}_i = \frac{\boldsymbol{\mu}_i(\tilde{\boldsymbol{x}})}{\sum \boldsymbol{\mu}_i(\tilde{\boldsymbol{x}})}$, where $\boldsymbol{\mu}_i(\tilde{\boldsymbol{x}}) = \exp\left(-\frac{(N_{\mathcal{D}}-1)}{2\sigma_g^2}(\boldsymbol{h}(\tilde{\boldsymbol{x}}) - \frac{i-1}{N_{\mathcal{D}}-1})^2\right)$. The texture feature $\boldsymbol{h}(\tilde{\boldsymbol{x}}) = \boldsymbol{H} * (\nabla \tilde{\boldsymbol{x}})$ is obtained by extracting gradients with first-order derivatives and smoothing them with a median filter. This design exploits a key observation: **the shape of the degradation kernel is highly correlated with the texture density of its local region**.

3. **Dual-Domain Likelihood Function**: Unlike conventional methods that only define likelihood in the spatial domain, this work constrains both the spatial and frequency domains: $y \sim \mathcal{N}(\boldsymbol{y} | (\mathcal{D}\boldsymbol{x})\downarrow_s, \sigma_y) \cdot \mathcal{N}(\mathcal{F}(\boldsymbol{y}) | \mathcal{F}((\mathcal{D}\boldsymbol{x})\downarrow_s), \sigma_f)$. Frequency-domain constraints improve reconstruction performance (as shown by a drop in PSNR when removed in ablation studies).

4. **Image Prior (Deep Image Prior Variant)**: A 3-layer U-Net network $G(\boldsymbol{z}; \boldsymbol{\phi})$ is employed as an implicit image prior. A Gaussian prior $\mathcal{N}(0, \sigma_z)$ is placed on $\boldsymbol{z}$, and a Laplace gradient prior $\mathcal{L}(\nabla G | 0, \sigma_x)$ is applied to the network output to suppress overfitting. Instance Normalization and frequency-domain skip connections are further introduced to mitigate overfitting.

### Loss & Training

The MCEM inference algorithm alternates between:
- **E-Step**: Sampling the latent variable $\boldsymbol{z}$ via Stochastic Gradient Langevin Dynamics (SGLD) with a sampling step of $n_z = 5$.
- **M-Step**: Updating the kernel parameters $\boldsymbol{\Gamma}$ and network weights $\boldsymbol{\phi}$ with the ADAM optimizer to maximize the ELBO.

Key parameters: $N_{\mathcal{D}} = 5$ (number of atomic kernels), $\sigma_g = 0.5$, $\sigma_y = 1$, $\sigma_f = 2$, $\sigma_x = 2.5$, $\sigma_z = 1$, with a maximum of 5000 iterations.

## Key Experimental Results

### Main Results

| Dataset | Scale | SVDSR (Ours) | BSRDM (SOTA) | Gain |
|--------|------|-------------|-------------|------|
| Set5 | ×2 | **33.51/0.92** | 32.76/0.91 | +0.75 |
| Set14 | ×2 | **29.61/0.83** | 28.65/0.81 | +0.96 |
| Urban100 | ×2 | **26.40/0.79** | 25.46/0.76 | +0.94 |
| Manga109 | ×2 | **29.89/0.89** | 28.49/0.87 | +1.40 |
| DIV2K100 | ×2 | **29.46/0.82** | 28.32/0.78 | +1.14 |
| Set5 | ×3 | **31.37/0.89** | 30.96/0.88 | +0.41 |
| Set14 | ×3 | **28.14/0.79** | 27.67/0.77 | +0.47 |
| Set5 | ×4 | **29.29/0.85** | 29.02/0.85 | +0.27 |
| Urban100 | ×4 | **23.90/0.70** | 23.47/0.68 | +0.43 |

$2\times$ super-resolution achieves an average PSNR improvement of approximately 1 dB, and performs close to the non-blind method (ZSSR-NB with ground-truth kernels) across multiple datasets.

### Ablation Study

| Configuration | Set14 PSNR (×2) | Description |
|------|-----------------|------|
| Full Model | **29.61** | — |
| Case1: w/o frequency-domain likelihood | 29.61 | Frequency-domain constraint still contributes |
| Case2: w/o frequency-domain skip | 29.52 | -0.09 |
| Case3: w/o Instance Norm | 29.28 | -0.33, overfitting effect is obvious |

| Number of atomic kernels $N_{\mathcal{D}}$ | Set5 | Set14 | Urban100 | Manga109 | DIV2K100 |
|---------------------------|------|-------|----------|----------|----------|
| 1 (Degenerates to spatially-invariant) | 33.19 | 29.13 | 25.90 | 29.23 | 29.05 |
| 3 | 33.29 | 29.40 | 26.11 | 29.53 | 29.30 |
| **5** | **33.51** | **29.61** | **26.40** | **29.89** | **29.46** |
| 7 | 33.43 | 29.58 | 26.21 | 29.67 | 29.52 |
| 9 | 33.27 | 29.59 | 26.20 | 29.74 | 29.54 |

### Key Findings

- **Spatially-Variant vs. Spatially-Invariant**: $N_{\mathcal{D}}=5$ outperforms $N_{\mathcal{D}}=1$ by 0.3~0.7 dB, validating the merit of spatially-variant modeling.
- **5 Atomic Kernels as Optimal**: Retaining more kernels degrades performance, suggesting that the 72 kernels used in prior methods are redundant.
- Instance Normalization is crucial for suppressing overfitting (-0.33 dB).
- Model size is 850K parameters, with a run time of 33s ($256\times256, \times2$), comparable to BSRDM.

## Highlights & Insights

1. **Pioneering Dataset-Free Spatially-Variant Degradation**: Breaking the paradigm that 'spatially-variant degradation models must be trained on large datasets'.
2. **Ingenious Integration of Fuzzy Set Theory**: Deriving the coefficient matrix from image texture information avoids learning a massive pixel-wise coefficient matrix, while providing physical interpretability.
3. **Minimal Parameterization**: Each atomic kernel requires only 3 parameters (rotation angle + two variances), supported by the theory of anisotropic Gaussian kernel decomposition.
4. **Intuitive Visualization Results**: The visualization of the coefficient matrix clearly reflects the image texture structure, demonstrating significant domain shape differences in different regions.

## Limitations & Future Work

1. **Performance Drop in Large-Scale SR**: The advantage diminishes at $4\times$ scale, as excessive loss of texture details weakens the benefits of spatially-variant modeling.
2. **Color Distortion**: Color shift may occur in heavily noisy images, which is a common issue for most SR methods.
3. **Computational Efficiency**: Running 5000 EM iterations (each containing SGLD sampling) takes 33s in total, which is acceptable but far from real-time.
4. Future work can explore extending the concept of fuzzy set coefficient matrices to other image inverse problems (such as deblurring and deraining).

## Related Work & Insights

- The Deep Image Prior (DIP) framework provides a foundation for dataset-free methods. On this basis, this work introduces spatially-variant degradation modeling.
- Comparison with BSRDM demonstrates that spatially-variant degradation modeling can bring significant improvements with only a tiny amount of extra parameters (88K).
- The inference framework of Monte Carlo EM + SGLD provides an elegant mathematical structure for probabilistic image restoration.
- The application of fuzzy set theory in computer vision still holds vast potential for exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first dataset-free spatially-variant degradation model, featuring a unique fuzzy set coefficient matrix design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Assessed across 5 datasets and 3 scale factors with core components covered in ablation, though lacking a direct comparison with other spatially-variant methods like LARPAR.
- **Writing Quality**: ⭐⭐⭐⭐ — The mathematical derivations are clear but symbol-dense, presenting a somewhat high reader entry barrier for the probabilistic modeling part.
- **Value**: ⭐⭐⭐⭐ — Training without a dataset is a significant advantage, delivering solid performance gains. Code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)
- [\[ICLR 2026\] DISK: Differentiable Sparse Kernel Complex for Efficient Spatially-Variant Convolution](../../ICLR2026/image_restoration/disk_differentiable_sparse_kernel_complex_for_efficient_spatially-variant_convol.md)
- [\[ECCV 2024\] Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal](raindrop_clarity_a_dual-focused_dataset_for_day_and_night_raindrop_removal.md)
- [\[ECCV 2024\] Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)
- [\[ECCV 2024\] BAMM: Bidirectional Autoregressive Motion Model](bamm_bidirectional_autoregressive_motion_model.md)

</div>

<!-- RELATED:END -->
