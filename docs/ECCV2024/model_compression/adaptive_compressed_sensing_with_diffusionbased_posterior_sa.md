---
title: >-
  [Paper Note] Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling
description: >-
  [ECCV 2024][Model Compression][Adaptive Compressed Sensing] This paper proposes AdaSense, which leverages the zero-shot posterior sampling capability of pre-trained diffusion models to quantify reconstruction uncertainty, thereby adaptively selecting the optimal measurement matrix. It achieves training-free adaptive compressed sensation across multiple domains including face images, MRI, and CT, outperforming non-adaptive methods and even the optimal PCA-based non-adaptive sc…
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Adaptive Compressed Sensing"
  - "Diffusion Models"
  - "Posterior Sampling"
  - "Medical Imaging"
  - "Active Measurement Acquisition"
date: 2026-05-08
content_hash: b6a94365a33949f0
---

# Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling

**Conference**: ECCV 2024  
**arXiv**: [2407.08256](https://arxiv.org/abs/2407.08256)  
**Code**: [https://github.com/noamelata/AdaSense](https://github.com/noamelata/AdaSense)  
**Area**: Model Compression / Compressed Sensing  
**Keywords**: Adaptive Compressed Sensing, Diffusion Models, Posterior Sampling, Medical Imaging, Active Measurement Acquisition

## TL;DR
This paper proposes AdaSense, which leverages the zero-shot posterior sampling capability of pre-trained diffusion models to quantify reconstruction uncertainty, thereby adaptively selecting the optimal measurement matrix. It achieves training-free adaptive compressed sensation across multiple domains including face images, MRI, and CT, outperforming non-adaptive methods and even the optimal PCA-based non-adaptive scheme.

## Background & Motivation
Compressed sensing (CS) enables rapid image acquisition by selecting a small number of key measurements, representing an essential technology for measurement-expensive scenarios such as medical imaging. Adaptive CS goes a step further by dynamically choosing subsequent measurement directions based on already acquired measurement information, which can theoretically reduce the total number of required measurements significantly. However, existing adaptive CS methods suffer from several key limitations: (1) most methods require complex task-specific training pipelines for specific degradation patterns, resulting in poor generalization; (2) many methods are restricted to specific degradation types (e.g., only supporting pixel subsampling); (3) in data-sensitive domains like medical imaging, acquiring sufficient training data is often challenging. Key Challenge: How to achieve adaptive measurement selection without additional training? Key Insight: Leverage the zero-shot posterior sampling capabilities of pre-trained diffusion models to estimate reconstruction uncertainty, thereby greedily selecting measurement directions that minimize uncertainty. Core Idea: Approximate the principal components of the posterior covariance matrix using posterior sampling and employ them as the next measurement directions.

## Method

### Overall Architecture
AdaSense is an iterative framework for adaptive measurement acquisition. In each step: (1) a zero-shot posterior sampler of a diffusion model is used to generate $s$ posterior samples; (2) PCA is performed on these samples to extract the top $r$ principal components of the posterior covariance matrix as the new measurement directions; (3) real measurements are acquired using the new measurement matrix and appended to the existing measurements. After repeating this for $N$ steps, the final image reconstruction is performed using all acquired measurements.

### Key Designs
1. **Posterior Covariance-Based Measurement Selection**:
    - Function: Adaptively select the optimal measurement direction at each step.
    - Mechanism: Formulate the measurement selection problem as a PCA problem. Given the existing measurements $\mathbf{y}_{0:nr}$, the optimal new measurement direction corresponds to the top $r$ eigenvectors of the posterior covariance matrix $\mathrm{Cov}[\mathbf{x}|\mathbf{y}_{0:nr}]$, as they represent the directions of maximum uncertainty.
    - Design Motivation: The error of a linear minimum MSE reconstructor equals the projection of the posterior covariance along the unmeasured directions; thus, choosing the directions of maximum variance maximizes information gain.

2. **Simplified Optimization under Constrained Measurement Scenarios**:
    - Function: Handle measurement selection under physical constraints, such as in MRI (k-space lines) and CT (projection angles).
    - Mechanism: When the measurement matrix is restricted to a feasible set $\mathcal{H}$, the reconstruction matrix is fixed to the pseudo-inverse $\tilde{\bm{H}}^{\dagger}$, simplifying the objective function to $\argmax_{\tilde{\bm{H}} \in \mathcal{H}} \mathbb{E}[(\mathbf{x} - \mathbb{E}[\mathbf{x}|\mathbf{y}])^{\top} \tilde{\bm{H}}^{\dagger} \tilde{\bm{H}} (\mathbf{x} - \mathbb{E}[\mathbf{x}|\mathbf{y}]) | \mathbf{y}]$, which can be approximated via posterior samples and solved through exhaustive search.
    - Design Motivation: Directly optimizing the full objective function requires the covariance matrix rank to be at least $r$. Since $r$ is typically very large in MRI/CT, generating a sufficient number of posterior samples is computationally impractical.

3. **Posterior Sampling Acceleration**:
    - Function: Avoid repetitive SVD computations at each step to improve efficiency.
    - Mechanism: Exploit the property of a consistent posterior sampler where the variance along already selected measurement directions is zero, meaning that new measurement directions are inevitably orthogonal to past ones. This trivializes the SVD of the measurement matrix to $(U, \Sigma, V^{\top}) = (I, I, \bm{H})$ without requiring auxiliary computation.
    - Design Motivation: Dynamically constructing the measurement matrix at runtime necessitates frequent SVD computations, which represents the primary computational bottleneck.

### Loss & Training
AdaSense itself requires no training; it is an inference-time measurement selection algorithm. It fundamentally relies on pre-trained diffusion models and zero-shot posterior samplers (e.g., DDRM). The final reconstruction can be performed using: (1) a single posterior sample, (2) an average of multiple posterior samples (approximating the MMSE estimate), or (3) a modality-specific specialized reconstruction network.

## Key Experimental Results

### Main Results
**Face Image Reconstruction** (CelebA-HQ 256, 192 dimensions of measurements):

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DeepFace↓ |
|------|-------|-------|--------|-----------|
| Block Downsampling | 20.50 | 0.6128 | 0.3035 | 0.6134 |
| Bicubic Downsampling | 20.86 | 0.6193 | 0.2964 | 0.5865 |
| PCA (Optimal Non-Adaptive) | 24.60 | 0.7190 | 0.2307 | 0.3010 |
| **AdaSense** | **26.20** | **0.7515** | **0.1950** | **0.2674** |

**MRI Active Subsampling** (FastMRI single-coil knee, vertical subsampling):

| Method | PSNR(R10)↑ | SSIM(R10)↑ |
|------|-----------|-----------|
| Random R10 | 24.56 | 0.4786 |
| Equi-spaced R10 | 24.73 | 0.4767 |
| **AdaSense R10** | **27.01** | **0.5229** |

### Ablation Study

| Configuration (Adaptive Steps N) | PSNR↑ | LPIPS↓ | Description |
|------|---------|------|------|
| N=1 (Non-adaptive) | ~24.5 | ~0.24 | No adaptation, equivalent to one-step PCA |
| N=4 | ~25.5 | ~0.21 | Moderate adaptation |
| N=8 | ~26.2 | ~0.20 | Highly adaptive, best performance |

Comparison with specialized active MRI acquisition methods (R8-30L):

| Method | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Low-to-High (Non-adaptive) | 28.32 | 0.5948 |
| SS-DDQN (RL-trained) | 28.99 | 0.6129 |
| Greedy Oracle (Upper bound) | 29.22 | 0.6264 |
| **AdaSense** | 28.89 | 0.6108 |

### Key Findings
- Without any additional training, AdaSense outperforms the non-adaptive PCA method using real training data by 1.6 dB PSNR in face image reconstruction.
- Greater adaptability (larger $N$) leads to monotonically improved reconstruction quality, validating the effectiveness of adaptive measurement selection.
- On MRI tasks, AdaSense achieves performance comparable to specialized methods requiring complex reinforcement learning training (SS-DDQN), approaching the theoretical upper bound (Greedy Oracle).
- It also shows potential in CT sparse-view reconstruction, demonstrating the cross-modality versatility of the framework.

## Highlights & Insights
- **Zero-Training Versatility**: Transferring to a new modality merely requires substituting the domain diffusion model without retraining for degradation patterns, which is particularly crucial in data-scarce fields like medical imaging.
- **Theoretical Elegance**: The adaptive CS problem is elegantly formulated as a PCA problem of the conditional posterior, backed by clear mathematical derivation.
- **Practical Simplicity**: The algorithmic workflow is simple (sampling $\rightarrow$ PCA $\rightarrow$ measurement $\rightarrow$ repeat), making it easy to understand and implement.
- Outperforming the optimal non-adaptive PCA scheme demonstrates that adaptability (adjusting subsequent strategies based on existing measurements) yields substantial benefits.

## Limitations & Future Work
- High computational overhead: Generating multiple posterior samples at each step is required, whereas diffusion sampling itself is time-consuming.
- The performance upper bound is constrained by the quality of the posterior sampler; sampler inaccuracies can propagate to measurement selection.
- Currently limited to linear measurements; non-linear measurement scenarios (e.g., phase retrieval) remain unexplored.
- The greedy strategy may lead to sub-optimal solutions as it does not account for the impact of current choices on future steps.
- More safety validation is required prior to clinical deployment.

## Related Work & Insights
- Compared to RL-based active MRI acquisition methods (SS-DDQN, DS-DDQN), AdaSense trades off partial performance for zero-shot flexibility.
- It shares similarities with blind inverse problem methods as both leverage diffusion models to adapt to degradation matrices. However, their objectives differ; this work aims to select optimal measurements rather than recovering unknown degradations.
- Insight: With the rapid advancement of diffusion posterior samplers (faster and more accurate), the practicality of AdaSense is expected to improve significantly.
- It can be extended to other active sensing tasks, such as burst photography and best-next-view selection in 3D reconstruction.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegantly combines diffusion posterior sampling with adaptive CS, though selecting measurement directions via PCA is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three domains (Faces, MRI, CT) with a clever design for the adaptive ablation study, although the scale of MRI/CT experiments is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous mathematical derivation, intuitive tables and figures, and a well-structured paper.
- Value: ⭐⭐⭐⭐ Provides a unified zero-training adaptive CS framework with direct application value for medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing](adaptive_selection_of_samplingreconstruction_in_fourier_comp.md)
- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](../../CVPR2025/model_compression/sampling_innovation-based_adaptive_compressive_sensing.md)
- [\[ECCV 2024\] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)
- [\[CVPR 2026\] Sampling-Aware Quantization for Diffusion Models](../../CVPR2026/model_compression/sampling-aware_quantization_for_diffusion_models.md)
- [\[ICML 2025\] Diffusion Sampling Correction via Approximately 10 Parameters](../../ICML2025/model_compression/diffusion_sampling_correction_via_approximately_10_parameters.md)

</div>

<!-- RELATED:END -->
