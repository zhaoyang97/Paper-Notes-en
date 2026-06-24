---
title: >-
  [Paper Note] Architecture-Agnostic Untrained Network Priors for Image Reconstruction with Frequency Regularization
description: >-
  [ECCV2024][Medical Imaging][untrained network prior] This paper proposes three architecture-agnostic frequency regularization techniques (bandwidth-constrained input, bandwidth-controllable upsampling, and Lipschitz-regularized convolutional layers) to address the issues of architectural sensitivity, overfitting, and operational inefficiency in untrained network priors, significantly narrowing the performance gap among different architectures in MRI reconstruction tasks.
tags:
  - "ECCV2024"
  - "Medical Imaging"
  - "untrained network prior"
  - "deep image prior"
  - "spectral bias"
  - "frequency regularization"
  - "MRI reconstruction"
date: 2026-05-08
content_hash: da9d350ea314c013
---

# Architecture-Agnostic Untrained Network Priors for Image Reconstruction with Frequency Regularization

**Conference**: ECCV2024  
**arXiv**: [2312.09988](https://arxiv.org/abs/2312.09988)  
**Code**: [GitHub](https://github.com/YilinLiu97/Untrained-Recon)  
**Area**: Medical Imaging  
**Keywords**: untrained network prior, deep image prior, spectral bias, frequency regularization, MRI reconstruction

## TL;DR

This paper proposes three architecture-agnostic frequency regularization techniques (bandwidth-constrained input, bandwidth-controllable upsampling, and Lipschitz-regularized convolutional layers) to address the issues of architectural sensitivity, overfitting, and operational inefficiency in untrained network priors, significantly narrowing the performance gap among different architectures in MRI reconstruction tasks.

## Background & Motivation

Untrained networks inspired by Deep Image Prior (DIP) can recover high-quality images from corrupted or partial measurements without requiring a training set. Their success is widely attributed to the spectral bias of appropriate network architectures—an implicit regularization effect where CNNs tend to fit low-frequency signals before high-frequency ones (such as noise).

However, existing methods face three interconnected core challenges:

1. **Difficult Architecture Selection**: Different architectures (depth and width) heavily affect reconstruction performance. A unified architecture selection criterion is lacking, and searching for the optimal architecture is highly expensive.
2. **Risk of Overfitting**: Networks easily overfit to noise or incomplete measurements, requiring strategies like early stopping, which are unstable and do not improve the capacity of the architecture itself.
3. **Low Operational Efficiency**: The image-specific optimization paradigm is inherently slow, a problem further exacerbated by over-parameterized architectures.

Most existing works address these issues separately (e.g., neural architecture search, early stopping, pre-training/fine-tuning). The core motivation of this paper is: since spectral bias is the fundamental reason behind DIP's success, is it possible to directly regulate the regularization effect of arbitrary architectures from a frequency perspective, thereby solving all three challenges simultaneously?

## Core Problem

How to design architecture-agnostic frequency frequency regularization methods so that untrained networks with different configurations (depth and width) can all achieve near-optimal reconstruction performance, while avoiding overfitting and reducing running time?

## Method

The authors identify three core elements that contribute to spectral bias within the DIP framework and propose corresponding regularization methods for each:

### 1. Bandwidth-Constrained Input

Traditional untrained networks use uniformly distributed white noise as input. From a frequency perspective, white noise contains all frequency components with uniform amplitude, which drives the network to quickly converge to high-frequency components, leading to high-frequency artifacts.

This paper proposes two methods to limit the input bandwidth:

- **Gaussian Blur**: Apply a Gaussian blur filter $\mathcal{G}_{s,\sigma}$ to the noise input to remove a portion of the high-frequency components. The filter size $s$ and $\sigma$ are hyperparameters.
- **Fourier Features**: Replace the noise input with Fourier features at a lower maximum frequency $f_c \propto L$ (such as $L=4$ or $L=8$) to introduce frequency diversity in a controlled manner.

Experiments show that when $L$ increases to 16, the frequency range of the Fourier features approaches that of raw noise, and the performance deteriorates accordingly, validating the effectiveness of limiting the input bandwidth.

### 2. Bandwidth-Controllable Upsampling

While limiting input bandwidth works significantly for shallow networks, its effect diminishes as network depth increases—since deeper layers can generate arbitrary new high-frequency components.

The authors design a bandwidth-controllable upsampler based on the Kaiser-Bessel window:

1. First, interleave the input feature maps with zeros (zero-interleaving).
2. Convolve the result with a customizable low-pass filter.

The Kaiser window provides explicit control over the trade-off between passband ripple and stopband attenuation:

$$w(n) = I_0(\beta\sqrt{1-(2n/M)^2}) / I_0(\beta)$$

where $M$ controls the spatial support of the window and $\beta$ controls the degree of stopband attenuation (larger values yield smoother images). This plug-and-play upsampler can use different $M$ and $\beta$ values at different layers, providing flexible and precise control.

### 3. Lipschitz-Regularized Layers

Convolutional layers (coupled with non-linearities) are the only operations capable of generating new frequencies. The sensitivity of the output to input perturbations is controlled by regularizing their Lipschitz constants:

- Set a learnable Lipschitz constant $k_\ell$ for each layer $\ell$.
- Constrain weights via matrix norm normalization: normalization is only performed when the matrix norm exceeds the learned constraint.
- Use SoftPlus to ensure non-negativity.

The final optimization objective is:

$$\min_{\Theta, K} \mathcal{L}(\mathbf{y}; \mathbf{AG_\Theta(z)}) + \lambda \sum_{l=1}^{L} \text{SoftPlus}(\mathbf{k}_\ell)^2$$

where $\lambda$ controls the scale of smoothness, and $K$ is the set of learnable Lipschitz constants for all layers.

### Combination of Methods

The three methods are complementary: bandwidth-constrained inputs mainly benefit shallow architectures, Kaiser upsampling primarily benefits deep architectures, and Lipschitz regularization provides additional gains across all configurations. The best performance is achieved when they are used in combination.

## Key Experimental Results

### Datasets and Settings
- fastMRI multi-coil knee and brain MRI, standard 4× acceleration
- Stanford 3D FSE knee dataset (out-of-domain evaluation)
- Base architecture: $N$-layer encoder-decoder with full skip connections, 3000 iterations

### Narrowing the Architectural Performance Gap (fastMRI Brain)

| Method | A2_256 PSNR | A8_64 PSNR | A2_256 SSIM | A8_64 SSIM |
|------|------------|------------|-------------|------------|
| No Regularization | 29.08 | 31.68 | 0.729 | 0.807 |
| Gaussian Blur + Lips. + Kaiser | 32.50 | 33.85 | 0.836 | 0.885 |

The worst-performing A2_256 architecture gets improved by +3.42 dB PSNR, leading to consistent performance across all configurations.

### Comparison with Baseline Methods (fastMRI Knee)

| Method | PSNR | SSIM | Running Time |
|------|------|------|---------|
| U-Net (Supervised) | 31.15 | 0.776 | ~1.5 days training + 0.1s inference |
| ZS-SSL | 32.00 | 0.773 | 26.1 min/slice |
| DIP | 29.16 | 0.628 | 9.2 min/slice |
| A2_64 (Ours) | 32.07 | 0.781 | 6.4 min/slice |
| A8_64 (Ours) | 31.73 | 0.768 | 12.3 min/slice |

The compact A2_64 combined with the proposed method matches or even exceeds ZS-SSL while running approximately 4 times faster.

### Out-of-Domain Generalization (Stanford FSE Knee)
- A2_64 (Ours): PSNR 31.43 / SSIM 0.790
- U-Net (Supervised): PSNR 29.16 / SSIM 0.724
- Unsupervised methods significantly outperform supervised ones on out-of-domain data.

### Operational Efficiency
- Up to 90× faster compared to ZS-SSL (reducing from ~1 hour/slice to ~5 minutes/slice).
- Small models with regularization can match the performance of large models, further reducing computational overhead.

## Highlights & Insights

1. **A Unified Framework Addressing Three Problems**: The proposed framework is the first to simultaneously address architectural sensitivity, overfitting, and operational inefficiency in untrained network priors, without requiring any modifications to the base architecture itself.
2. **Extremely Simple Implementation**: The core methods can be implemented with only a few lines of code (blurring the input noise, replacing the upsampler, and adding the Lipschitz regularization term).
3. **Insights from a Frequency Perspective**: Drawing an analogy between white noise input and Fourier features reveals the mechanism of how input bandwidth influences overfitting.
4. **Advantages in Out-of-Domain Generalization**: Unsupervised methods naturally avoid the distribution shift challenge of supervised methods, outperforming U-Net by approximately 2 dB in out-of-domain PSNR.
5. **Complementary to Early Stopping**: The proposed method can be combined with self-validation-based early stopping to further reduce reconstruction time.

## Limitations & Future Work

1. **Hyperparameter Tuning**: $\{s, \sigma\}$ for Gaussian blur and $\{M, \beta\}$ for the Kaiser window must be tuned for different datasets and acceleration factors (e.g., using different hyperparameters for knee and brain), which is not yet fully automated.
2. **Limited Evaluation Scope**: The method is primarily validated on MRI reconstruction; experiments on natural image denoising and inpainting are relatively brief.
3. **Evaluation Only under 4× Acceleration**: Main experiments were conducted under 4× undersampling; evaluations with higher acceleration factors (e.g., 8×) are not comprehensive enough.
4. **Insufficient Theoretical Analysis**: The theoretical foundation explaining why the three methods are complementary and what the optimal combination strategy should be is not yet fully developed.
5. **Missing Comparisons with the Latest Methods**: No comparison was made with diffusion-model-based MRI reconstruction methods.

## Related Work & Insights

| Method Type | Representative Work | Ours Advantage |
|---------|---------|---------|
| Architecture Search | NAS for DIP | No search required; improves arbitrary architectures |
| Early stopping | ZS-SSL, Wang et al. | Improves the architectural capability itself rather than just preventing overfitting, and is complementary |
| Transfer Learning | Pre-training + Fine-tuning | No extra training set required, avoiding expensive pre-training |
| Traditional Regularization | Total Variation (TV) | TV only partially alleviates overfitting; ours fundamentally addresses it from a frequency perspective |
| Supervised Methods | U-Net | No training set required; better out-of-domain generalization |

## Inspirations & Connections

- **Generality of the Frequency Perspective**: The paradigm of understanding and improving neural network prior through frequency regulation can be generalized to other inverse problems (e.g., CT reconstruction, super-resolution).
- **Importance of Input Design**: The frequency characteristics of white noise input have long been overlooked; this paper demonstrates that carefully designing the input bandwidth is a zero-cost yet powerful regularization mechanism.
- **Small Models + Strong Regularization vs. Large Models**: With frequency regularization, compact models can outperform large models. This finding holds practical value for resource-constrained medical imaging deployments.
- **Connection to INR/NeRF**: Analogizing the noise input of untrained networks to the Fourier features of implicit neural representations (INRs) bridges the gap between these two seemingly different paradigms.

## Rating

- Novelty: ⭐⭐⭐⭐ (The frequency-regularization perspective is novel, though individual techniques are not entirely new)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive MRI experiments, but relatively limited on natural images)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation, complete logical flow, and rich figures and tables)
- Value: ⭐⭐⭐⭐ (Addresses practical pain points with a simple, plug-and-play implementation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Cognitive Process-Inspired Architecture for Subject-Agnostic Brain Visual Decoding](../../ICLR2026/medical_imaging/a_cognitive_process-inspired_architecture_for_subject-agnostic_brain_visual_deco.md)
- [\[ICLR 2026\] Frequency-Balanced Retinal Representation Learning with Mutual Information Regularization](../../ICLR2026/medical_imaging/frequency-balanced_retinal_representation_learning_with_mutual_information_regul.md)
- [\[CVPR 2026\] D$^2$-FOSA: Dual-Diffusion Guided EEG-to-Image Reconstruction with Frequency-Oriented Semantic Alignment](../../CVPR2026/medical_imaging/d2-fosa_dual-diffusion_guided_eeg-to-image_reconstruction_with_frequency-oriente.md)
- [\[ECCV 2024\] Domesticating SAM for Breast Ultrasound Image Segmentation via Spatial-Frequency Fusion and Uncertainty Correction](domesticating_sam_for_breast_ultrasound_image_segmentation_via_spatial-frequency.md)
- [\[ECCV 2024\] Brain-ID: Learning Contrast-agnostic Anatomical Representations for Brain Imaging](brain-id_learning_contrast-agnostic_anatomical_representations_for_brain_imaging.md)

</div>

<!-- RELATED:END -->
