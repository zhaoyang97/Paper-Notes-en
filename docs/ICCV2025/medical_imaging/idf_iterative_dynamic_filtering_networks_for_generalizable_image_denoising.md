---
title: >-
  [Paper Note] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising
description: >-
  [ICCV 2025][Medical Imaging][Image Denoising] This paper proposes Iterative Dynamic Filtering Networks (IDF), which achieves strong out-of-distribution (OOD) denoising performance using only ~0.04M parameters. By combini…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Image Denoising"
  - "Dynamic Filtering"
  - "Generalizability"
  - "Iterative Refinement"
  - "Lightweight Model"
date: 2026-05-08
content_hash: c7dc4ba4fdf81cbe
---

# IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising

**Conference**: ICCV 2025
**arXiv**: [2508.19649](https://arxiv.org/abs/2508.19649)
**Code**: [dongjinkim9.github.io/projects/idf](https://dongjinkim9.github.io/projects/idf)
**Area**: Medical Imaging
**Keywords**: Image Denoising, Dynamic Filtering, Generalizability, Iterative Refinement, Lightweight Model

## TL;DR

This paper proposes Iterative Dynamic Filtering Networks (IDF), which achieves strong out-of-distribution (OOD) denoising performance using only ~0.04M parameters. By combining per-pixel dynamic kernel prediction with an adaptive iterative refinement strategy, IDF generalizes to diverse unseen noise types (Gaussian, Poisson, salt-and-pepper, Monte Carlo rendering, and real noise) while trained exclusively on single-level Gaussian noise.

## Background & Motivation

- Deep learning-based denoising methods (DnCNN, SwinIR, Restormer) achieve excellent performance on their training noise distributions but suffer from severe **generalization deficiency** when facing unseen noise types and levels.
- Self-supervised methods require time-consuming test-time adaptation; data augmentation approaches are constrained by training distributions; prior-based methods depend on large-scale pretraining.
- MaskedDenoising regularizes learning via masked training but performs poorly in fine-grained texture regions.
- **Core Insight**: Dynamic kernel prediction combined with iterative refinement can prevent overfitting to specific noise patterns:
    - Dynamic kernels are generated per-pixel to adapt to local image context and noise patterns.
    - A sum-to-one normalization constraint forces each kernel to act as a weighted averaging operator, preventing memorization of training noise.
    - The iterative strategy enables progressive denoising, balancing efficiency and quality.

## Method

### Overall Architecture

IDF adopts an iterative denoising scheme: a noisy input image $\mathbf{I}_{Noisy}$ is progressively denoised through $T$ Dynamic Image Denoising (DID) blocks, all sharing weights to reduce parameters and mitigate overfitting. At inference, both fixed-iteration and adaptive-iteration strategies are supported.

### Key Designs

1. **Dynamic Image Denoising (DID) Block**: At each iteration $t$, the previous output $\hat{\mathbf{I}}_{Clean}^{(t-1)}$ is unfolded into overlapping patches of size $K \times K$. The DID block predicts a per-pixel spatially varying denoising kernel $\mathbf{w}^{(t)} \in \mathbb{R}^{1 \times K^2 \times (H \cdot W)}$, applies the convolution $\mathbf{x}_i^{(t)} = \mathbf{w}_i^{(t)} \circledast \mathbf{y}_i^{(t)}$, and folds the result back into an image. Alternating dilation rates (e.g., 2 and 1) encourage learning of diverse features.

2. **Feature Extraction Module (FEM)**: Applies **RMS normalization** for sample-level normalization: $\text{Norm}(\mathbf{a}) = \mathbf{a} / \sqrt{\frac{1}{N}\sum_i \mathbf{a}_i^2 + \epsilon}$, rendering features invariant to global noise level variations. This is followed by two 3×3 convolutional layers with ReLU activations for a shallow and efficient design.

3. **Global Statistics Module (GSM) + Local Correlation Module (LCM)**:

    - GSM leverages the iterative structure by computing the inter-iteration residual $\mathbf{I}_{Res}^{(t)} = \hat{\mathbf{I}}_{Clean}^{(t-1)} - \hat{\mathbf{I}}_{Clean}^{(t-2)}$, extracting its mean and standard deviation as global noise statistics, analogous to ISO information.
    - LCM computes per-patch Pearson correlation coefficients among pixels; high-correlation regions (flat areas) favor uniform kernels, while low-correlation regions (edges) employ selective weights.
    - The two modules are complementary: GSM provides global noise-level priors, and LCM provides local structural priors.

4. **Kernel Prediction Module (KPM)**: FEM features and GSM features are multiplied via channel-wise attention ($\mathbf{F}_{FE}^{(t)} \odot \mathbf{F}_{GS}^{(t)}$), concatenated with LCM features, normalized, and passed through a 3×3 convolution to produce kernel weights. **Power Normalization** ($p=3$) is used instead of softmax to enforce the sum-to-one constraint, providing greater robustness to outliers.

5. **Dynamic Iteration Control (DIC)**: Adaptively determines the number of iterations. The key observation is that upon convergence, the kernel center value approaches 1. A confidence map $\mathbf{C}^{(t)} = \mathbf{w}^{(t)}(c_x,c_y) - \mathbf{w}^{(t-1)}(c_x,c_y)$ monitors changes in kernel center values across consecutive iterations; early termination is triggered when the spatial mean satisfies $\frac{1}{HW}|\sum_i \mathbf{C}_i^{(t)}| < \kappa$ (with $\kappa=0.015$).

### Loss & Training

- Loss function: $\mathcal{L}_1$ distance, $|\hat{\mathbf{I}}_{Clean}^{(T)} - \mathbf{I}_{Clean}|$
- Training data: CBSD432 dataset with **Gaussian noise at σ=15 only**
- Training configuration: AdamW optimizer, learning rate 1e-4, 50k iterations, 128×128 patches, batch size 8
- Maximum iterations $T=10$; DIC threshold $\kappa=0.015$

## Key Experimental Results

### Main Results

**Synthetic OOD Noise Denoising (all methods trained on Gaussian noise at σ=15 only; PSNR/SSIM)**:

| Noise Type | Dataset | DnCNN | SwinIR | Restormer | MaskedDen. | **Ours** |
|------------|---------|-------|--------|-----------|------------|----------|
| Gaussian σ=50 | CBSD68 | 15.87/0.227 | 16.07/0.227 | 20.15/0.386 | 20.95/0.441 | **25.73/0.719** |
| Gaussian σ=50 | Urban100 | 16.08/0.297 | 16.26/0.298 | 19.69/0.428 | 21.10/0.502 | **25.06/0.755** |
| Poisson α=3.5 | CBSD68 | 19.36/0.430 | 19.45/0.420 | 22.23/0.560 | 24.17/0.635 | **27.47/0.801** |
| Salt&Pepper d=0.02 | CBSD68 | 24.02/0.708 | 23.23/0.674 | 23.60/0.679 | 29.74/0.843 | **33.38/0.913** |
| Speckle σ²=0.04 | CBSD68 | 24.55/0.696 | 24.08/0.681 | 25.16/0.720 | 27.94/0.814 | **29.18/0.855** |
| Mixture Level4 | Urban100 | 21.26/0.522 | 21.16/0.506 | 23.20/0.599 | 25.17/0.746 | **27.52/0.841** |

**Real Noise Denoising**:

| Dataset | Restormer | CODE | MaskedDen. | **Ours** |
|---------|-----------|------|------------|----------|
| SIDD | 22.54/0.370 | 26.71/0.515 | 28.65/0.604 | **32.08/0.758** |
| SIDD+ | 24.45/0.470 | 31.08/0.677 | 31.52/0.725 | **33.72/0.812** |
| PolyU | 27.27/0.831 | **38.11/0.950** | 34.65/0.933 | 37.93/0.960 |
| Nam | 27.71/0.815 | **39.78/0.954** | 34.80/0.942 | 38.96/0.968 |

### Ablation Study

**Effect of GSM and LCM (Urban100, Mixture / Spatial Gaussian)**:

| GSM | LCM | Mixture PSNR | Spatial Gaussian PSNR |
|-----|-----|-------------|----------------------|
| ✗ | ✗ | 27.07 | 27.70 |
| ✓ | ✗ | 27.33 | 27.44 |
| ✗ | ✓ | 27.25 | 27.76 |
| ✓ | ✓ | **27.52** | **27.78** |

**Impact of Individual DID Components**:

| Configuration | Mixture PSNR | Spatial Gaussian PSNR |
|---------------|-------------|----------------------|
| Full | **27.52** | **27.78** |
| w/o RMS Norm | 27.13 | 27.55 |
| w/o Dilation | 27.48 | 27.37 |
| w/o Power Norm | 23.02 | 24.04 |

**Model Efficiency Comparison**:

| Model | Params (M) | FLOPs (G) | Inference Time (s) |
|-------|-----------|----------|--------------------|
| CLIPDenoising | 19.55 | 8.77 | 0.004 |
| DnCNN | 0.67 | 17.2 | 0.002 |
| Restormer | 26.1 | 60.5 | 0.03 |
| **Ours** | **0.04** | 11.5 | 0.005 |
| **Ours†(DIC)** | **0.04** | **8.02** | **0.004** |

### Key Findings

- **Power Normalization is the most critical component**: its removal causes a PSNR drop exceeding 4 dB, indicating that the sum-to-one constraint is central to preventing overfitting.
- The DIC adaptive strategy reduces iteration count by approximately 30% while maintaining performance.
- With only 0.04M parameters (17× fewer than DnCNN), IDF comprehensively outperforms large-capacity models (Restormer at 26.1M) across diverse OOD noise types.
- IDF also performs strongly on Monte Carlo rendering noise (64/128 spp), achieving PSNRs of 28.09 and 31.13, respectively.

## Highlights & Insights

- **Minimalist yet effective**: 0.04M parameters + single-level training data → generalization to all noise types, challenging the "bigger is better" paradigm.
- The **sum-to-one constraint** is conceptually profound: it enforces each kernel to function as a weighted averaging operator, fundamentally preventing the network from memorizing training noise patterns.
- The **iterative strategy** naturally synergizes with dynamic kernel prediction: residual information across iterations (GSM) and kernel convergence signals (DIC) are obtained "for free."
- RMS normalization achieves noise-level invariance in a simple and effective manner.
- The choice of Power Normalization ($p=3$) over softmax is insightful — it provides greater robustness to outliers.

## Limitations & Future Work

- Inference speed is constrained by PyTorch's unfold operation; optimizing this step could yield significant speedups.
- Performance on PolyU and Nam real-noise datasets falls slightly short of CODE, suggesting an inherent ceiling for extreme lightweight models.
- Validation is limited to grayscale/RGB denoising and has not been extended to other low-level vision tasks (super-resolution, deblurring).
- Adaptive kernel size selection or multi-scale kernel prediction are promising directions for future exploration.

## Related Work & Insights

- The dynamic kernel prediction concept builds on Kernel-Predicting CNNs (Bako et al.) and KPN (Mildenhall et al.); the key innovation lies in combining this with iterative refinement.
- The approach is orthogonal to the masked regularization strategy of MaskedDenoising, and combining the two could be fruitful.
- RMS normalization is adapted from the NLP domain (RMSNorm in LLMs), demonstrating successful cross-domain transfer.
- The iterative refinement paradigm is consistent with prior work such as DBPN (super-resolution) and IDR (unsupervised denoising).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of dynamic kernels, iterative refinement, and adaptive control is novel; the Power Normalization constraint is conceptually deep.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluations span synthetic, real, and Monte Carlo noise; six synthetic noise types; four real-noise datasets; and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and module motivations are well-articulated.
- **Value**: ⭐⭐⭐⭐⭐ A general-purpose denoiser with 0.04M parameters offers exceptional practical value and is highly suitable for edge device deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SIC: Similarity-Based Interpretable Image Classification with Neural Networks](sic_similarity-based_interpretable_image_classification_with_neural_networks.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [\[CVPR 2026\] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](../../CVPR2026/medical_imaging/relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)
- [\[ICCV 2025\] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup](dictas_a_framework_for_class-generalizable_few-shot_anomaly_segmentation_via_dic.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](../../NeurIPS2025/medical_imaging/iterative_foundation_model_fine-tuning_on_multiple_rewards.md)

</div>

<!-- RELATED:END -->
