---
title: >-
  [Paper Note] Lightweight Optimal-Transport Harmonization on Edge Devices
description: >-
  [AAAI 2026 Oral][Model Compression][Color Harmonization] This paper proposes MKL-Harmonizer, which leverages the Monge-Kantorovich Linear (MKL) mapping from classical optimal transport theory to train a compact encoder that predicts 12-dimensional color transformation parameters, enabling real-time image color harmonization on edge devices. The method achieves state-of-the-art performance on the combined perceptual quality–speed metric in AR scenarios.
tags:
  - "AAAI 2026 Oral"
  - "Model Compression"
  - "Color Harmonization"
  - "Optimal Transport"
  - "Edge Devices"
  - "Augmented Reality"
  - "Lightweight Inference"
date: 2026-05-08
content_hash: 28a25cab7781ea9f
---

# Lightweight Optimal-Transport Harmonization on Edge Devices

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.12785](https://arxiv.org/abs/2511.12785)  
**Code**: [github](https://github.com/maria-larchenko/mkl-harmonizer)  
**Area**: Model Compression
**Keywords**: Color Harmonization, Optimal Transport, Edge Devices, Augmented Reality, Lightweight Inference

## TL;DR
This paper proposes MKL-Harmonizer, which leverages the Monge-Kantorovich Linear (MKL) mapping from classical optimal transport theory to train a compact encoder that predicts 12-dimensional color transformation parameters, enabling real-time image color harmonization on edge devices. The method achieves state-of-the-art performance on the combined perceptual quality–speed metric in AR scenarios.

## Background & Motivation

### State of the Field
Image harmonization aims to adjust the color of foreground objects in composite images so that they are visually consistent with the background. Existing methods are predominantly based on deep learning dense prediction models (e.g., DoveNet, RainNet), which deliver strong results but impose significant computational and memory demands, and are largely constrained to low-resolution 256×256 inputs.

### Limitations of Prior Work

**High computational resource requirements**: Dense prediction models (encoder-decoder architectures) cannot run in real time on edge devices such as mobile GPUs and XR headsets.

**Lack of AR support**: Mainstream AR platforms such as ARKit and ARCore rely solely on illumination estimation (directional light, environment maps, spherical harmonics) and lack advanced color harmonization.

**Exposure bias**: Masks in standard training datasets (e.g., iHarmony4) contain background pixel leakage at boundaries, causing models to over-rely on this information. In AR scenarios, rendering engines provide pixel-accurate masks, leading to a train–inference mismatch.

**Scarcity of AR evaluation data**: No dataset of real AR composite images with pixel-accurate masks exists.

### Root Cause
Color harmonization is critical for AR realism, yet existing methods cannot satisfy the latency and computational constraints of real-time AR.

### Starting Point & Core Idea
Color harmonization is fundamentally a problem of mapping color distributions. When both source and target distributions are approximated as multivariate Gaussians, the optimal transport mapping admits a closed-form linear solution—the Monge-Kantorovich Linear (MKL) filter—requiring only 12 parameters (a 3×3 matrix $A$ and a 3-dimensional offset $S$). The authors propose training a compact encoder to predict these 12 parameters, enabling extremely lightweight and fast inference.

## Method

### Overall Architecture
A 4-channel (RGB + mask) composite image is fed into an EfficientNet-B0 encoder, which outputs a 12-dimensional vector $[A, S]$. An MKL filter is constructed and applied as an affine color transformation to foreground pixels, producing the harmonized image.

### Key Designs

1. **Optimal Transport Theoretical Foundation**:

    - Color harmonization is formulated as a transport mapping problem from source distribution $\pi_0$ to target distribution $\pi_1$.
    - When both distributions are approximately Gaussian, the optimal transport mapping has a closed-form linear solution: $T^*_{im}(x) = \mu_1 + A(x - \mu_0)$
    - where $A = \Sigma_0^{-1/2}(\Sigma_0^{1/2}\Sigma_1\Sigma_0^{1/2})^{1/2}\Sigma_0^{-1/2}$
    - **Design Motivation**: The ideal MKL filter achieves an MSE of approximately 7.0 on iHarmony4, demonstrating that a linear filter is sufficient for most harmonization tasks.

2. **Ground-Truth Filter Generation**:

    - Exact MKL transformation parameters are computed for each image in iHarmony4 as supervision signals.
    - The problem is simplified from predicting the harmonized image to predicting 12 filter parameters.
    - Predicting $[A, S]$ directly rather than $[\mu_1, \Sigma_1]$ is shown experimentally to be more robust to prediction error.

3. **Theoretical Error Analysis**:

    - An upper bound is derived for the approximation error of the linear MKL mapping relative to the true nonlinear mapping:
    - $\mathcal{E} \leq 2\mathcal{E}_{clip} + 2\mathcal{E}_{lin}$
    - $\mathcal{E}_{lin} \leq 2B^2 + 2(\|A\|_{op} + L)^2 \cdot \text{tr}(\Sigma_0)$
    - The linear approximation is valid when the true mapping is smooth (small Lipschitz constant) and the color distribution is not concentrated at the gamut boundary. Dark objects, whose distributions cluster in gamut corners, may yield suboptimal results.

4. **ARCore Evaluation Dataset**:

    - An ARCore sample application is modified to construct a data collection tool.
    - 327 composite image–mask pairs are collected, covering indoor and outdoor scenes under varying times of day and weather conditions.
    - All masks are obtained directly from the rendering engine and are pixel-accurate.
    - This is the first open-source AR composite image dataset of its kind.

### Loss & Training

A composite loss function is employed:

$$L_{total} = L_{labels} + \alpha \cdot L_{content}$$

- **Label loss**: $L_{labels} = \|\text{Model}(im) - [A, S]\|_1$, using L1 rather than L2.
    - L2 loss forces predictions toward the arithmetic mean of all possible MKL solutions, which may correspond to an invalid filter.
    - L1 loss allows convergence to sharper solutions.
- **Content loss**: $L_{content} = \|M * X_0 - M * (X_0 \cdot A' + S')\|_1$, pixel-wise L1 loss.
    - Using content loss alone causes mode collapse, where the model learns a near-identity filter.
    - Content loss weight $\alpha = 10$.

The encoder is EfficientNet-B0 with 256×256, 4-channel (RGB+mask) input, trained for 210 epochs using the Adam optimizer with piecewise learning rate decay and a batch size of 64.

## Key Experimental Results

### Main Results

| Method | MSE↓ | PSNR↑ | fMSE↓ |
|--------|------|-------|-------|
| Ideal Linear OT | 7.6 | 43.6 | 45.9 |
| PCT-Net | 29.1 | 38.0 | 201 |
| Harmonizer | 40.1 | 36.6 | 258 |
| **Ours (L1)** | 65.0 | 34.1 | 438 |
| INR | 67.2 | 35.3 | 392 |
| Unharmonized | 182 | 31.0 | 984 |

| Method | 256×256 | 512×512 | 1024×2048 | 4096×4096 |
|--------|---------|---------|-----------|-----------|
| **Ours** | **175.01** | **166.76** | **137.21** | **40.85** |
| DoveNet | 123.39 | – | – | – |
| PCT-Net | 104.57 | 98.65 | 63.74 | 11.84 |
| Harmonizer | 95.01 | 89.82 | 47.63 | 7.45 |
| INR | 6.35 | 3.22 | 0.81 | 0.12 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Ours (L1 loss) | MSE=65.0 | L1 loss overall superior |
| Ours (L2 loss) | MSE=66.3 | Marginally worse, difference is small |
| Content loss only | Mode collapse | Model learns identity transform |
| Predict $[\mu_1, \Sigma_1]$ | Worse | Insufficient robustness to prediction error |
| Predict $[A, S]$ | Better | Directly predicting filter parameters is more stable |

### Key Findings
1. **Perceptual quality**: A user study (20 participants, 642 ratings) shows that MKL-Harmonizer achieves perceptual quality on par with leading baselines on real AR data.
2. **Speed–quality trade-off**: The proposed method simultaneously achieves the highest perceptual score and the fastest inference speed.
3. **Edge device deployment**: 12–15 fps is achieved on Google Pixel 4a/7; zero-copy optimization may push this to 24–30 fps.
4. **Exposure bias**: INR outperforms PCT-Net on MSE, yet receives lower perceptual quality ratings in human evaluation, indicating that MSE is an unreliable metric in AR scenarios.
5. **High-resolution advantage**: Dense prediction models produce degradation artifacts (banding, JPEG artifacts) at high resolutions, whereas filter-based methods do not.

## Highlights & Insights
1. Classical optimal transport theory and deep learning are elegantly combined, yielding a theoretically grounded and practically effective approach.
2. The paper is the first to identify and analyze "exposure bias" in image harmonization, and points out the inadequacy of standard evaluation metrics.
3. Predicting only 12 parameters results in an extremely lightweight model, establishing a new paradigm of filter-parameter prediction.
4. A complete theoretical error analysis is provided, characterizing the conditions under which the linear mapping is valid.
5. The first open-source AR composite image dataset with pixel-accurate masks is contributed.

## Limitations & Future Work
1. Performance on standard iHarmony4 metrics is below SOTA (higher MSE), though the authors attribute this to metric distortion caused by exposure bias.
2. Handling of dark objects is poor due to color distributions concentrated at gamut boundaries.
3. The method is not well-suited for video harmonization, as per-frame predictions may exhibit temporal flickering; only exponential moving averaging is currently applied as a mitigation.
4. iHarmony4 contains high-frequency artifacts; the authors cleaned the dataset but this has not become standard practice.
5. Validation is limited to EfficientNet-B0; more efficient architectures remain to be explored.

## Related Work & Insights
- **PCT-Net**: Predicts pixel-level affine transformation parameters → this work simplifies to 12 global parameters.
- **INR-Harmonization**: Implicit neural representations → computationally expensive, unsuitable for real-time use.
- **Harmonizer**: Regresses brightness/contrast filter coefficients → this work provides a more principled formulation grounded in optimal transport theory.
- The classical optimal transport approach to color transfer (Pitié 2007) remains a relevant and enduring reference.

## Rating
- Novelty: ⭐⭐⭐⭐ (The combination of optimal transport and encoder-based prediction is novel, though the overall framework is not highly complex)
- Experimental Thoroughness: ⭐⭐⭐⭐ (iHarmony4 + ARCore + user study + edge deployment, though standard metrics are weak)
- Writing Quality: ⭐⭐⭐⭐⭐ (Theoretical analysis is complete, motivation is clear, and the dataset contribution is valuable)
- Value: ⭐⭐⭐⭐ (High practical value with clear demand in AR scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](../../ICLR2026/model_compression/cross_domain_lossy_compression_optimal_transport.md)
- [\[NeurIPS 2025\] Optimizing Distributional Geometry Alignment with Optimal Transport for Generative Dataset Distillation](../../NeurIPS2025/model_compression/optimizing_distributional_geometry_alignment_with_optimal_transport_for_generati.md)
- [\[CVPR 2026\] MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction](../../CVPR2026/model_compression/memo_human-like_crisp_edge_detection_using_masked_edge_prediction.md)
- [\[ICLR 2026\] Boosted Trees on a Diet: Compact Models for Resource-Constrained Devices](../../ICLR2026/model_compression/boosted_trees_on_a_diet_compact_models_for_resource-constrained_devices.md)
- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](../../CVPR2026/model_compression/critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)

</div>

<!-- RELATED:END -->
