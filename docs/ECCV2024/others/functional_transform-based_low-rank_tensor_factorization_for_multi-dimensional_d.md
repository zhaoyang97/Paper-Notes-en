---
title: >-
  [Paper Note] Functional Transform-Based Low-Rank Tensor Factorization for Multi-Dimensional Data Recovery
description: >-
  [ECCV 2024][Low-rank tensor factorization] Proposes Functional Transform-based Low-Rank Tensor Factorization (FLRTF), which utilizes implicit neural representations instead of traditional discrete transforms to capture the continuous smoothness of data along the third dimension, effectively addressing temporal/spectral degradation.
tags:
  - "ECCV 2024"
  - "Low-rank tensor factorization"
  - "Implicit Neural Representation"
  - "Video frame interpolation"
  - "Multispectral image super-resolution"
  - "Continuous transform"
date: 2026-05-08
content_hash: 549ea96342b3d2f3
---

# Functional Transform-Based Low-Rank Tensor Factorization for Multi-Dimensional Data Recovery

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Low-level vision / Data recovery  
**Keywords**: Low-rank tensor factorization, Implicit Neural Representation, Video frame interpolation, Multispectral image super-resolution, Continuous transform

## TL;DR

Proposes Functional Transform-based Low-Rank Tensor Factorization (FLRTF), which utilizes implicit neural representations instead of traditional discrete transforms to capture the continuous smoothness of data along the third dimension, effectively addressing temporal/spectral degradation.

## Background & Motivation

1. **Background**: Transform-based Low-Rank Tensor Factorization (t-LRTF) has emerged as an important tool for multi-dimensional data recovery in recent years. This approach applies an invertible transform along a certain dimension of the tensor, transforms the data into the transform domain, and performs low-rank factorization, thereby effectively utilizing structural priors of the data. Representative methods include those based on FFT (Fast Fourier Transform), DCT (Discrete Cosine Transform), and data-driven learnable discrete transforms.

2. **Limitations of Prior Work**: 
    - **Limitations of discrete transforms**: Existing t-LRTF methods mainly use discrete transforms (e.g., FFT, DCT) along the third dimension (temporal or spectral dimension). These discrete transforms can only handle discrete sampling points and cannot model the smoothness of the data in the continuous domain.
    - **Temporal/spectral degradation**: In tasks such as video frame interpolation, video frame extrapolation, and multispectral image (MSI) spectral super-resolution, data needs to be recovered at missing time points or spectral bands, which represents "temporal/spectral degradation".
    - **Inability of discrete transforms to handle missing positions**: Discrete transforms require uniformly sampled complete data and cannot directly perform inference at missing temporal/spectral positions.

3. **Key Challenge**: Existing t-LRTF methods rely on discrete transforms, whereas temporal/spectral degradation tasks inherently require modeling and interpolation in the continuous domain—discrete transforms cannot bridge this gap.

4. **Goal**: To design a low-rank tensor factorization method capable of performing transforms in the continuous domain, enabling it to handle not only traditional data corruption (such as random missing values) but also the complete absence of time frames or spectral bands (temporal/spectral degradation).

5. **Key Insight**: Utilizing Implicit Neural Representation (INR) to parameterize the transform function. INRs are naturally continuous and can be evaluated at any continuous coordinate, making them highly suitable for modeling continuous transforms along temporal/spectral dimensions.

6. **Core Idea**: Replace traditional discrete transforms with implicit neural representations to parameterize the transform matrix in t-LRTF, equipping tensor factorization with continuous sampling and interpolation capabilities along the third dimension.

## Method

### Overall Architecture

The core framework of Functional Transform-based Low-Rank Tensor Factorization (FLRTF) is as follows:

1. Given an incomplete three-dimensional tensor $\mathcal{X} \in \mathbb{R}^{n_1 \times n_2 \times n_3}$ (e.g., height $\times$ width $\times$ temporal frames of a video) where some elements or entire frames are missing.
2. While traditional t-LRTF decomposes the tensor into a low-rank form after applying a discrete transform along the third dimension, FLRTF replaces the discrete transform with a continuous functional transform.
3. The missing data is recovered through joint optimization by learning the INR parameters of the transform function and the low-rank factorization coefficients.
4. Due to the continuity of the INR, it can be evaluated at any coordinate along the third dimension, enabling frame/spectral interpolation.

### Key Designs

1. **Functional Transform**: 
    - **Function**: Maps the tensor along the third dimension from the original domain to the transform domain.
    - **Mechanism**: Parameterizes the transform function using an implicit neural representation (such as an MLP) with positional encoding. Specifically, each element of the transform matrix is defined by a continuous function $f_\theta$, taking the continuous coordinate along the third dimension as input.
    - **Design Motivation**: Compared with discrete transforms, functional transforms possess continuity and can be evaluated at arbitrary positions; meanwhile, positional encoding empowers them to capture high-frequency details.

2. **INR-based Continuous Low-Rank Factorization**: 
    - **Function**: Represents the recovered tensor in a low-rank factorization form over a continuous domain.
    - **Mechanism**: In the transform domain, the tensor is decomposed into a superposition of several low-rank components. Since the transform itself is continuous, each decomposed component also maintains continuity along the third dimension, naturally capturing the temporal/spectral smoothness of the data.
    - **Design Motivation**: Continuity is the key to solving temporal/spectral degradation tasks—allowing direct evaluation at missing frame/channel positions.

3. **General Multi-Dimensional Data Recovery Model**: 
    - **Function**: Constructs a unified data recovery optimization framework based on FLRTF.
    - **Mechanism**: Embeds FLRTF into a general optimization model to recover data by alternately optimizing the INR parameters and the low-rank factorization coefficients. It supports multiple recovery tasks, including frame interpolation, frame extrapolation, and spectral super-resolution.
    - **Design Motivation**: A unified framework can flexibly adapt to different types of multi-dimensional data recovery tasks, avoiding the need to design task-specific methods.

### Loss & Training

- **Data Fidelity Term**: Minimizes the mean squared error between the recovered and observed values on observed positions.
- **Low-Rank Regularization**: Imposes low-rank constraints (such as nuclear norm minimization or truncated SVD) on the tensor in the transform domain.
- **INR Regularization**: Applies appropriate regularization on the INR parameters to prevent overfitting and encourage smoothness.
- **Alternative Optimization Strategy**: Alternately updates the INR parameters (using gradient descent) and low-rank factorization coefficients (using closed-form solutions or proximal operators) to ensure convergence.

## Key Experimental Results

### Main Results

The paper conducts extensive experimental validation across three categories of multi-dimensional data recovery tasks:

| Task | Metrics | Ours (FLRTF) | Prev. SOTA | Gain |
|------|------|------------|---------|------|
| Video Frame Interpolation | PSNR/SSIM | Outperforms all baseline methods | Traditional t-LRTF / Deep learning methods | Significant improvement |
| Video Frame Extrapolation | PSNR/SSIM | Outperforms all baseline methods | Traditional t-LRTF methods | Significant improvement |
| MSI Spectral Channel Interpolation | PSNR/SSIM | Outperforms all baseline methods | Traditional t-LRTF methods | Significant improvement |
| MSI Spectral Super-Resolution | PSNR/SSIM/SAM | Outperforms all baseline methods | Representative data recovery methods | Superior performance |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Discrete FFT transform | Lower PSNR | Traditional discrete transforms cannot handle missing frames/channels |
| Discrete learnable transform | Moderate PSNR | Better than fixed transforms, but still limited by discreteness |
| INR without positional encoding | Lower PSNR | Positional encoding is crucial for capturing high-frequency details |
| Complete FLRTF | Highest PSNR | Continuous functional transform + positional encoding achieves the best overall performance |

### Key Findings

- FLRTF demonstrates significant advantages over traditional discrete transform-based t-LRTF in temporal/spectral degradation scenarios (frame interpolation, spectral interpolation).
- The smoothness prior introduced by the continuous transform is key to handling degradation issues.
- Positional encoding helps the INR capture high-frequency components in the data, preventing over-smoothing.
- The FLRTF framework is general and can be directly applied to various data recovery tasks.

## Highlights & Insights

- **Outstanding theoretical contribution**: First to introduce implicit neural representations into the transform design of low-rank tensor factorization, bridging the theoretical gap between discrete and continuous transforms.
- **Solves a fundamental problem**: The root cause of traditional methods struggling with temporal/spectral degradation is the inability of discrete transforms to evaluate at missing locations. FLRTF elegantly solves this problem.
- **Unified framework**: The same framework can handle multiple tasks such as frame interpolation, frame extrapolation, and spectral super-resolution, illustrating the generality of the method.
- **Combination of INR + traditional optimization**: Combines INR techniques from deep learning with traditional optimization methods, reaping the benefits of both worlds.

## Limitations & Future Work

- The training of INRs relies on gradient descent, which may present computational efficiency issues for large-scale data.
- The current method mainly handles 3D tensors; extending to higher-dimensional tensors (e.g., 4D and above) requires further research.
- The choice of INR network architecture (depth, width) may require parameter tuning for different tasks.
- Compared to end-to-end deep learning methods, optimization-based frameworks can be slower in inference speed.
- More powerful INR architectures (e.g., SIREN, Hash Encoding) can be explored to further improve performance.

## Related Work & Insights

- **TNN / t-SVD**: Classic tensor nuclear norm and tensor SVD factorization methods, serving as the theoretical foundation for t-LRTF.
- **Instant-NGP / NeRF**: The successful application of implicit neural representations in 3D scene reconstruction inspired this work to use INR for transform function parameterization.
- **LRTC / TMac**: Traditional low-rank tensor completion methods; compared to these, the proposed method has the extra capability to handle degradation tasks.
- **Inspiration**: The continuity of INRs can be leveraged to solve more signal processing problems that require continuous modeling.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing INR into the transform design of tensor factorization is a brand-new approach, with clear theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple tasks and datasets, with comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and well-explained problem motivations.
- Value: ⭐⭐⭐⭐ Opens up a new direction of continuous transforms for low-rank tensor factorization, showing a wide range of application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Superpixel-Informed Implicit Neural Representation for Multi-Dimensional Data](superpixel-informed_implicit_neural_representation_for_multi-dimensional_data.md)
- [\[ECCV 2024\] Dropout Mixture Low-Rank Adaptation for Visual Parameters-Efficient Fine-Tuning](dropout_mixture_low-rank_adaptation_for_visual_parameters-efficient_fine-tuning.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](../../NeurIPS2025/others/the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)

</div>

<!-- RELATED:END -->
