---
title: >-
  [Paper Note] Blind Image Deblurring with Noise-Robust Kernel Estimation
description: >-
  [ECCV 2024][Image Restoration][Blind Deblurring] This paper proposes a blind deblurring method based on a noise-robust kernel estimation function and deep image prior (DIP). By designing a kernel estimation function capable of accurately estimating blur kernels even under strong noise, combined with a multiple-kernel estimation scheme to handle unknown noise levels, it achieves superior deblurring performance on both simulated and real images.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "Blind Deblurring"
  - "Noise-Robust"
  - "Kernel Estimation"
  - "Deep Image Prior"
  - "Inverse Problems"
date: 2026-05-08
content_hash: 60dda3cc30cc36e3
---

# Blind Image Deblurring with Noise-Robust Kernel Estimation

**Conference**: ECCV 2024  
**Code**: [https://github.com/csleemooo/BD_noise_robust_kernel_estimation](https://github.com/csleemooo/BD_noise_robust_kernel_estimation)  
**Area**: Image Restoration / Image Deblurring  
**Keywords**: Blind Deblurring, Noise-Robust, Kernel Estimation, Deep Image Prior, Inverse Problems

## TL;DR

This paper proposes a blind deblurring method based on a noise-robust kernel estimation function and deep image prior (DIP). By designing a kernel estimation function capable of accurately estimating blur kernels even under strong noise, combined with a multiple-kernel estimation scheme to handle unknown noise levels, it achieves superior deblurring performance on both simulated and real images.

## Background & Motivation

**Background**: Blind image deblurring is a classic inverse problem aiming to simultaneously recover a sharp image and a blur kernel from a single blurry image. This problem finds wide applications in fields such as photography, medical imaging, and remote sensing. Existing methods, including traditional optimization-based methods (e.g., alternating minimization) and deep learning-based end-to-end methods, have achieved decent results on clean (noise-free) blurry images.

**Limitations of Prior Work**: When strong noise exists in blurry images, the performance of existing blind deblurring methods drops dramatically. There are two core reasons: (1) Noise interferes with blur kernel estimation—priors such as image gradients relied upon by traditional methods become unreliable under noise; (2) Deep learning methods are prone to overfitting noise—the network may learn noise patterns instead of the true sharp image. Crucially, the noise level in real-world scenarios is typically unknown, further increasing the difficulty of the problem.

**Key Challenge**: Blind deblurring itself is a highly ill-posed inverse problem, requiring the recovery of two unknowns (sharp image and blur kernel) from a single observation. Once noise is introduced, the solution space expands even further, causing traditional image and kernel priors to fail due to noise perturbation. A noise-insensitive kernel estimation method is needed.

**Goal**: (1) How to accurately estimate the blur kernel under strong noise? (2) How to handle real-world scenarios with unknown noise levels? (3) How to perform effective denoising while deblurring?

**Key Insight**: The authors observe that given a "sufficiently good" estimation of the sharp image, a noise-robust kernel estimation function can be designed to extract the blur kernel. Meanwhile, deep image prior (DIP) can generate a good estimation of natural images without requiring training data. Therefore, DIP is utilized for sharp image estimation, coupled with noise-robust kernel estimation, forming an alternating optimization framework.

**Core Idea**: Designing a noise-robust kernel estimation function to accurately recover blur kernels, combining it with deep image prior to generate sharp images, and using a multiple-kernel estimation scheme to handle unknown noise levels.

## Method

### Overall Architecture

The overall pipeline is an alternating optimization process consisting of two main iterative steps: (1) Given the current sharp image estimation, recover the blur kernel using the noise-robust kernel estimation function; (2) Given the estimated blur kernel, optimize the sharp image using the DIP network. The input is a blurry and noisy image, and the outputs are the recovered sharp image and the estimated blur kernel.

### Key Designs

1. **Noise-Robust Kernel Estimation Function**:

    - **Function**: Accurately estimate the blur kernel from a noisy and blurry image given the sharp image estimation.
    - **Mechanism**: Traditional kernel estimation solves for kernel $k$ by minimizing $\|y - k * x\|_2^2$ (where $y$ is the blurry image, $x$ is the sharp image estimation, and $*$ denotes convolution). However, in the presence of noise, this least-squares problem is severely degraded by noise. The proposed kernel estimation function operates in the frequency domain, suppressing noise impact by exploiting the distinct distribution characteristics of signals and noise in the frequency domain. Specifically, it adaptively weights the high-frequency components of the signal, which are more susceptible to noise pollution, thereby downweighting the noise components. Meanwhile, a sparsity constraint is introduced for the kernel (since natural blur kernels are generally sparse), further enhancing estimation robustness through regularization.
    - **Design Motivation**: Traditional kernel estimation performs well in noise-free scenarios, but kernel estimation errors are amplified under noise—even minor noise can cause significant bias in kernel estimation. By performing adaptive denoising in the frequency domain, the noise robustness is substantially improved without losing kernel details.

2. **Deep Image Prior (DIP)**:

    - **Function**: Generate the sharp image estimation and provide a natural image prior.
    - **Mechanism**: DIP utilizes a key observation: the architecture of convolutional neural networks intrinsically holds a prior bias toward natural images. When training a network to fit a target image with random noise as input, the network is prone to learning low-frequency components (natural structures) first, followed by high-frequency components (noise). Thus, by controlling the iteration count (early stopping), a denoised sharp image estimation can be obtained. Specifically, the DIP network $f_\theta$ takes a fixed random vector $z$ as input and optimizes network parameters $\theta$ by minimizing $\|y - k * f_\theta(z)\|_2^2$, where $k$ is the currently estimated blur kernel.
    - **Design Motivation**: The advantage of DIP lies in being completely unsupervised, requiring no training dataset and operating solely on a single test image. This enables the method to generalize across images from any domain. Concurrently, the regularizing effect of DIP naturally suppresses noise, complementing the noise-robust kernel estimation.

3. **Multiple Kernel Estimation Scheme**:

    - **Function**: Handle practical scenarios where the noise level is unknown.
    - **Mechanism**: Because the denoising parameters in the noise-robust kernel estimation function depend on the noise level, which is unknown in practice, a multiple-kernel estimation strategy is adopted. Specifically, multiple kernel estimation functions configured with different denoising parameters are executed in parallel, each corresponding to a different assumed noise level. Then, the optimal kernel estimation is automatically selected by evaluating quality metrics of each estimated kernel (e.g., kernel sparsity, sharp image reconstruction quality). This strategy bypasses the explicit step of estimating noise levels.
    - **Design Motivation**: Noise level estimation is inherently difficult, and its estimation error propagates into kernel estimation. The multiple-kernel estimation scheme avoids the need for explicit noise estimation via a "multi-hypothesis and selection" strategy, making the method more practical.

### Loss & Training

The overall optimization alternates between two steps: (1) Kernel estimation step: Fix the DIP network output, and solve for the optimal kernel $k = \arg\min_k \|Y - K \cdot X\|_F^2 + \lambda_k \|k\|_1$ (frequency domain representation) via the noise-robust kernel estimation function, where $\lambda_k$ controls sparsity; (2) Image estimation step: Fix the kernel $k$, optimize the DIP network parameters $\theta = \arg\min_\theta \|y - k * f_\theta(z)\|_2^2$ using the Adam optimizer, and utilize early stopping to prevent overfitting to noise. The outer loop repeats for approximately 5-10 alternating rounds, while the inner-loop DIP optimization runs for about 2000-5000 steps.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Simulated Data (Noise-free) | PSNR↑ | Competitive | SelfDeblur, MPRNet | Comparable to SOTA |
| Simulated Data (Noise $\sigma=2.55$) | PSNR↑ | Superior | Traditional + DL Methods | Significantly Outperforms |
| Simulated Data (Noise $\sigma=7.65$) | PSNR↑ | Substantially Superior | Notable degradation in comparisons | Gap widens as noise increases |
| AFHQ-dog (motion blur+noise) | Visual Quality | Superior | SelfDeblur | More accurate kernel estimation |
| AFHQ-cat (motion blur+noise) | Visual Quality | Superior | Traditional + DL Methods | Fewer noise artifacts |
| Real-world Blurry Images | Visual Quality | Superior | Cascaded denoising + deblurring | Avoids cascaded error accumulation |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Standard Kernel Estimation (No noise robustness) | PSNR drops by 2-5dB | Validates the necessity of noise-robust kernel estimation |
| DIP replaced with TV prior | PSNR drops | DIP provides a stronger image prior |
| Single Kernel Estimation (Fixed noise assumption) | Sensitive to mismatched noise | Multiple kernel estimation strategy improves robustness |
| Multiple Kernel Estimation Scheme | Robust to noise levels | Automatically adapts to varying noise levels |
| Different iteration numbers | Stable convergence | 5-10 alternating rounds are sufficient |

### Key Findings

- Under strong noise scenarios ($\sigma > 5$), the performance of traditional blind deblurring methods almost completely collapses, whereas the proposed method still maintains reasonable recovery quality.
- Interference of noise on kernel estimation is the primary cause of performance degradation—utilizing noise-robust kernel estimation substantially reduces the estimation error of the blur kernel.
- The timing of early stopping in DIP has a profound impact on results: stopping too early leads to under-restoration, whereas stopping too late leads to noise overfitting.
- The multiple-kernel estimation scheme reliably selects the optimal kernel, and the automatically selected results are close to those configured with the ground-truth noise level.
- Qualitative results on real-world images demonstrate the practical applicability of the proposed method.

## Highlights & Insights

1. **Accurate Problem Diagnosis**: Correctly identifies that the core challenge of noise in blind deblurring is kernel estimation failure, rather than image reconstruction failure.
2. **Simple and Effective Design**: The combination of noise-robust kernel estimation and DIP is straightforward yet highly effective, avoiding over-engineering.
3. **Fully Unsupervised**: Requires no training data; the self-supervised paradigm based on DIP offers great practical flexibility.
4. **Practical Multi-Kernel Strategy**: Ingeniously bypasses the challenging prerequisite of explicit noise-level estimation.

## Limitations & Future Work

1. The optimization process of DIP is slow (requiring thousands of steps per image), which limits real-time applications.
2. The method assumes the blur kernel is spatially uniform; handling spatially-varying blur remains to be extended.
3. The multiple-kernel estimation scheme requires running multiple optimization processes, further increasing the computational overhead.
4. The early stopping of DIP relies on heuristic determination, lacking an automated optimal stop criterion.
5. Experiments are only conducted on motion blur; applicability to other types like defocus blur and atmospheric turbulence blur is not yet verified.
6. Comparisons with state-of-the-art diffusion-model-based restoration methods (e.g., DiffPIR) are lacking.

## Related Work & Insights

- **SelfDeblur (Ren et al., 2020)**: Pioneered the use of DIP for blind deblurring, but did not consider noisy scenarios.
- **Deep Image Prior (Ulyanov et al., 2018)**: Foundational work on DIP, revealing the implicit regularization effect of network structures.
- **Pan et al.**: Classic blind deblurring method based on dark channel priors, with limited noise robustness.
- **MPRNet / Restormer**: Deep learning-based end-to-end image restoration methods, requiring extensive training data.
- **Levin et al.**: Classic Bayesian blind deblurring framework, which provides the theoretical foundation.
- The proposed method could be combined with learned denoisers (e.g., replacing DIP with a pre-trained denoiser) to potentially improve both speed and quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core contribution lies in the design of the noise-robust kernel estimation function, while the DIP framework is relatively classic.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on both simulated and real scenarios, though the datasets and comparison methods could be further enriched.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, but methodological details (frequency-domain operations) may require a signal processing background.
- **Value**: ⭐⭐⭐⭐ Blind deblurring under noise is a highly practical demand; the method is practical but inference speed limits widespread deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Blind Noisy Image Deblurring Using Residual Guidance Strategy](../../ICCV2025/image_restoration/blind_noisy_image_deblurring_using_residual_guidance_strateg.md)
- [\[ICLR 2026\] Are Deep Speech Denoising Models Robust to Adversarial Noise?](../../ICLR2026/image_restoration/are_deep_speech_denoising_models_robust_to_adversarial_noise.md)
- [\[ECCV 2024\] EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](edformer_transformer-based_event_denoising_across_varied_noise_levels.md)
- [\[ECCV 2024\] Domain-Adaptive Video Deblurring via Test-Time Blurring](domain-adaptive_video_deblurring_via_test-time_blurring.md)
- [\[ECCV 2024\] Towards Real-world Event-guided Low-light Video Enhancement and Deblurring](towards_real-world_event-guided_low-light_video_enhancement_and_deblurring.md)

</div>

<!-- RELATED:END -->
