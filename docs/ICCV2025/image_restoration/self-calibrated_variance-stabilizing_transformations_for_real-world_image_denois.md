---
title: >-
  [Paper Note] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising
description: >-
  [ICCV 2025][Image Restoration][Image Denoising] This paper proposes Noise2VST, a framework that learns a model-free variance-stabilizing transformation (VST) via self-supervised learning, enabling off-the-shelf Gaussian denoisers to handle real-world noisy images without any additional training.
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Image Denoising"
  - "Variance-Stabilizing Transformation"
  - "Zero-Shot Learning"
  - "Blind-Spot Denoising"
  - "Spline Modeling"
date: 2026-05-08
content_hash: d7f2834dd7cc7e85
---

# Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising

**Conference**: ICCV 2025  
**arXiv**: [2407.17399](https://arxiv.org/abs/2407.17399)  
**Code**: [GitHub](https://github.com/sherbret/Noise2VST)  
**Area**: Image Restoration  
**Keywords**: Image Denoising, Variance-Stabilizing Transformation, Zero-Shot Learning, Blind-Spot Denoising, Spline Modeling

## TL;DR

This paper proposes Noise2VST, a framework that learns a model-free variance-stabilizing transformation (VST) via self-supervised learning, enabling off-the-shelf Gaussian denoisers to handle real-world noisy images without any additional training.

## Background & Motivation

Deep learning has achieved remarkable success in image denoising, yet dominant approaches are heavily reliant on scene-specific training data. Networks trained for Gaussian noise perform poorly in real-world scenarios due to complex noise sources (photon shot noise, readout noise, etc.) that deviate from simple Gaussian distributions. Existing solutions face the following challenges:

**Data dependency**: Collecting large amounts of scene-specific clean/noisy image pairs to train dedicated models is extremely time-consuming or infeasible in many applications (e.g., medical imaging, astronomical imaging).

**Limitations of classical VSTs**: Classical variance-stabilizing transformations (e.g., the Anscombe transform, the GAT) can map non-Gaussian noise to approximately Gaussian noise, but require prior knowledge of the parametric form of the noise distribution; inaccurate parameter estimation leads to severe performance degradation.

**Insufficiency of unsupervised methods**: Existing methods that require no ground truth (Noise2Noise, blind-spot denoising, etc.) do not need clean images but still lag behind supervised methods in performance.

The core insight of this paper is that **pretrained Gaussian denoisers encode rich signal prior knowledge**. If an appropriate transformation can be found to map real-world noise to Gaussian noise, these powerful denoisers can be directly reused. The key challenge is learning such a transformation without assuming a noise model.

## Method

### Overall Architecture

The Noise2VST pipeline is as follows. Given a noisy image $\boldsymbol{z}$, a pixel-wise VST $f_{\boldsymbol{\theta}}$ is first learned to map it into the Gaussian noise domain; an off-the-shelf Gaussian denoiser $D$ is then applied; finally, the learned inverse transform $f^{\text{inv}}_{\boldsymbol{\theta},\alpha,\beta}$ maps the result back to the original domain. The full pipeline is:

$$\hat{\boldsymbol{s}} = (f^{\text{inv}}_{\boldsymbol{\theta},\alpha,\beta} \circ D \circ f_{\boldsymbol{\theta}})(\boldsymbol{z})$$

During training, a blind-spot denoiser $\bar{D}$ with frozen weights is used; at inference, it is replaced by a standard denoiser for improved performance.

### Key Designs

1. **Continuous Piecewise-Linear (CPWL) VST Modeling**

    - **Function**: The variance-stabilizing transformation is modeled using piecewise-linear splines from the family of increasing functions.
    - **Mechanism**: The VST $f_\theta$ is parameterized as a CPWL function with $n=128$ knots. The abscissas $x_i$ are uniformly distributed over $[z_{\min}, z_{\max}]$, and the ordinates are parameterized as $y_i = \theta_1 + \sum_{j=2}^{i} \exp(\theta_j)$, ensuring strict monotonic increase. The inverse transform is designed as $f^{\text{inv}}_{\boldsymbol{\theta},\alpha,\beta}(z) = f_{\boldsymbol{\theta}}^{-1}(z) + \alpha z + \beta$, where the affine term corrects the bias of the algebraic inverse.
    - **Design Motivation**: Spline functions are universal approximators, and monotonicity preserves the pixel ordering; the total of only $n+2=130$ learnable parameters greatly reduces the risk of overfitting.

2. **Blind-Spot Self-Supervised Training Strategy**

    - **Function**: The blind-spot denoiser property is exploited to enable fully self-supervised VST learning.
    - **Mechanism**: The blind-spot denoiser $\bar{D}$ produces an output for each pixel that does not depend on that pixel's own value. Since the composition $f^{\text{inv}} \circ \bar{D} \circ f_\theta$ preserves the blind-spot property, under the assumption of spatially independent noise, the self-supervised loss differs from the ground-truth loss by only a constant: $\mathcal{L}^{\bar{D}}_{\boldsymbol{\theta},\alpha,\beta}(\boldsymbol{z}, \boldsymbol{z}) = \mathcal{L}^{\bar{D}}_{\boldsymbol{\theta},\alpha,\beta}(\boldsymbol{z}, \boldsymbol{s}) + \text{const}$. VST parameters can therefore be optimized without any clean images.
    - **Design Motivation**: The theoretical guarantees of blind-spot methods ensure that VST learning remains effective in a zero-shot setting with only a single noisy image; the 130-parameter model is sufficiently compact to prevent overfitting.

3. **Blind-Spot Replacement at Inference**

    - **Function**: A blind-spot denoiser is used during training and replaced by a standard denoiser at inference.
    - **Mechanism**: The optimal VST depends only on the noise distribution, not on the denoiser type. The trained VST can therefore be paired with any Gaussian denoiser. Using a standard denoiser such as DRUNet at inference avoids the checkerboard artifacts of blind-spot denoisers. Alternatively, a fast blind-spot denoiser (e.g., FFDNet-based Noise2VST†) can be used during training and replaced by a stronger standard denoiser at inference.
    - **Design Motivation**: Blind-spot denoisers deliberately exclude information from the target pixel, which limits their performance; the training/inference decoupling allows both theoretical correctness and high output quality.

### Loss & Training

- **Loss function**: $\ell_2$ self-supervised loss $\mathcal{L}^{\bar{D}}_{\boldsymbol{\theta},\alpha,\beta}(\boldsymbol{z}, \boldsymbol{z}) = \|(f^{\text{inv}}_{\boldsymbol{\theta},\alpha,\beta} \circ \bar{D} \circ f_{\boldsymbol{\theta}})(\boldsymbol{z}) - \boldsymbol{z}\|_2^2$
- **Optimizer**: Adam with initial learning rate 0.01, decayed by a factor of 10 at 1/3 and 2/3 of total iterations.
- **Training details**: Trained on randomly cropped $64 \times 64$ patches with batch size 4; data augmentation includes random flipping and 90° rotation; total iterations 2000 (5000 for raw-RGB data).
- **Denoiser choice**: DRUNet (non-blind, requires a specified noise level $\sigma=25/255$); FFDNet is used for the fast variant.

## Key Experimental Results

### Main Results

**Synthetic Poisson Noise (sRGB space)**

| Method | KODAK PSNR/SSIM | BSD300 PSNR/SSIM | SET14 PSNR/SSIM |
|--------|----------------|-----------------|----------------|
| Baseline (N2C+GAT) | 31.63/0.865 | 29.92/0.850 | 30.66/0.854 |
| B2UNB | 31.07/0.857 | 29.92/0.852 | 30.10/0.844 |
| SST-GP | 31.39/0.872 | 29.96/0.853 | 30.22/0.848 |
| **Noise2VST** | **31.60/0.865** | 29.89/0.849 | **30.60/0.850** |

**Real-World Noise Denoising (SIDD dataset, raw-RGB space)**

| Method | SIDD Benchmark PSNR/SSIM | SIDD Validation PSNR/SSIM |
|--------|--------------------------|---------------------------|
| Baseline (N2C) | 50.60/0.991 | 51.19/0.991 |
| B2UNB | 50.79/0.991 | 51.36/0.992 |
| SST-GP | 50.87/0.992 | 51.57/0.992 |
| **Noise2VST** | **51.07/0.991** | **51.66/0.992** |

### Ablation Study

**Computational Efficiency Comparison** (256×256 image)

| Method | GPU Time | CPU Time | Trainable Parameters |
|--------|---------|---------|---------------------|
| S2S | 35 min | 4.5 hr | 1M |
| ZS-N2N | 20 sec | 1 min | 22k |
| Noise2VST | 50 sec | 5 min | 130 |
| Noise2VST† | **20 sec** | **40 sec** | 130 |

**Fluorescence Microscopy Data (FMD + W2S)**

| Method | FMD Confocal Fish | FMD Two-Photon Mice | W2S ch0 avg1 |
|--------|-------------------|---------------------|--------------|
| B2UNB | 32.74/0.897 | 34.03/0.916 | - |
| Noise2VST | **32.88/0.904** | **34.06/0.926** | **35.65** |

### Key Findings

1. The performance gap between Noise2VST and the oracle baseline (DRUNet+GAT with true noise parameters) is within 0.06 dB, demonstrating that the learned model-free VST approaches the theoretical optimum.
2. Noise2VST outperforms all methods that require no ground-truth training on the SIDD benchmark, approaching the supervised baseline.
3. The fast variant Noise2VST† (using FFDNet) requires only 20 seconds of GPU time with negligible performance loss.
4. Classical GAT methods perform poorly in real-world scenarios, primarily due to inaccurate parameter estimation and overly simplified noise models.

## Highlights & Insights

- **Minimal parameter design**: Only 130 parameters suffice to learn an effective VST, constituting a strong counterexample to the "more is better" paradigm.
- **Reuse of pretrained knowledge**: Rather than training a new denoising network, the method bridges Gaussian denoisers and real-world noise by learning a transformation, fully leveraging the signal priors of pretrained models.
- **Zero-shot with high performance**: Without using any external data, Noise2VST achieves performance close to supervised methods and even surpasses them on fluorescence microscopy data.
- **Training/inference decoupling**: Using a blind-spot denoiser during training ensures theoretical correctness, while switching to a standard denoiser at inference improves output quality — an elegant and clean design.

## Limitations & Future Work

1. The method assumes spatially independent noise; performance may be limited on sRGB images where demosaicking has introduced spatial noise correlations.
2. A separate VST must be optimized for each image; although the runtime is acceptable (~50 seconds), instant denoising is not achievable.
3. The VST is a global pixel-wise mapping and does not account for spatially varying noise characteristics.
4. Piecewise-linear functions are universal approximators but are less smooth than higher-order splines.

## Related Work & Insights

- **Noise2Noise / Noise2Void family**: The blind-spot concept provides the theoretical foundation of this method.
- **GAT / Anscombe transform**: Classical VSTs provide methodological inspiration; this paper generalizes them from parametric to model-free formulations.
- **DRUNet**: The generalization capability of non-blind Gaussian denoisers is a critical prerequisite for the success of the approach.
- Insight: The value of pretrained denoisers is underestimated; lightweight adaptation can unlock their potential in non-Gaussian settings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Generalizes VST from parametric models to self-supervised model-free learning; the minimalist 130-parameter design is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three categories of scenarios — synthetic noise, fluorescence microscopy, and smartphone cameras — with comprehensive baselines.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and rigorous, though the dense notation requires careful reading.
- Value: ⭐⭐⭐⭐⭐ A zero-shot method approaching supervised performance has significant practical value in resource-constrained settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Next-Scale Prediction: A Self-Supervised Approach for Real-World Image Denoising](../../CVPR2026/image_restoration/next-scale_prediction_a_self-supervised_approach_for_real-world_image_denoising.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](../../CVPR2026/image_restoration/tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [\[ECCV 2024\] Asymmetric Mask Scheme for Self-supervised Real Image Denoising](../../ECCV2024/image_restoration/asymmetric_mask_scheme_for_self-supervised_real_image_denoising.md)

</div>

<!-- RELATED:END -->
