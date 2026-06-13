---
title: >-
  [Paper Note] Blind Noisy Image Deblurring Using Residual Guidance Strategy
description: >-
  [ICCV 2025][Image Restoration][Blind Deblurring] This paper proposes a Residual Guidance Strategy (RGS) for coarse-to-fine blind image deblurring within an image pyramid framework. At each scale transition…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Blind Deblurring"
  - "Noise Robustness"
  - "Residual Guidance"
  - "Image Pyramid"
  - "Blur Kernel Estimation"
date: 2026-05-08
content_hash: 5c23b85deaf182cd
---

# Blind Noisy Image Deblurring Using Residual Guidance Strategy

**Conference**: ICCV 2025
**arXiv**: N/A (CVF Open Access only)  
**CVF**: [Paper Link](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Blind_Noisy_Image_Deblurring_Using_Residual_Guidance_Strategy_ICCV_2025_paper.html) | [PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Liu_Blind_Noisy_Image_Deblurring_Using_Residual_Guidance_Strategy_ICCV_2025_paper.pdf)
**Code**: N/A  
**Authors**: Heyan Liu, Jianing Sun, Jun Liu (Corresponding), Xi-Le Zhao, Tingting Wu, Tieyong Zeng
**Affiliations**: Northeast Normal University, University of Electronic Science and Technology of China, Nanjing University of Posts and Telecommunications, BNU-HKBU United International College
**Area**: Image Restoration
**Keywords**: Blind Deblurring, Noise Robustness, Residual Guidance, Image Pyramid, Blur Kernel Estimation

## TL;DR

This paper proposes a Residual Guidance Strategy (RGS) for coarse-to-fine blind image deblurring within an image pyramid framework. At each scale transition, the convolution residual from the adjacent coarser scale is denoised via a guided filter and used to correct the blurred input at the current scale. This approach significantly improves kernel estimation accuracy and restoration quality under high noise levels (σ=0.1), surpassing multiple deep learning methods without requiring any training.

## Background & Motivation

Blind image deblurring is a classical ill-posed inverse problem: simultaneously recovering a sharp image and a blur kernel from a single blurry observation. The degradation model is given by $B = K \otimes L + N$, where $K$ is the blur kernel, $L$ is the latent sharp image, and $N$ is noise.

In practice, long-exposure photography introduces not only motion blur but also substantial noise. Existing methods—both traditional and learning-based—perform reasonably well under noise-free or mild-noise conditions, but suffer severe degradation in kernel estimation accuracy when noise increases (e.g., Gaussian noise with σ≥0.05). The **root cause** lies in a fundamental tension: deblurring requires preserving high-frequency information (edges, textures), whereas denoising requires suppressing it. Striking a balance between the two is the central challenge.

Through empirical observation, the authors identify an interesting phenomenon in coarse-to-fine pyramid frameworks: **coarser-scale kernel estimates are more accurate** (due to weakened noise and preserved dominant structures after downsampling), yet lack fine detail; **finer-scale estimates are more precise but increasingly corrupted by noise**. This observation directly motivates the proposed RGS.

## Core Problem

How can coarser-scale, more reliable estimates be leveraged within a coarse-to-fine blind deblurring framework to guide finer-scale kernel estimation and suppress noise-induced degradation?

## Method

### Overall Architecture

The proposed method is a **training-free traditional optimization approach** based on an alternating iterative scheme grounded in a physical degradation model:

1. **Input**: A blurry and noisy image $B$
2. **Image Pyramid Construction**: Progressive downsampling produces multi-scale images $B_1, B_2, \ldots, B_n$ ($B_1$ finest, $B_n$ coarsest)
3. **Coarse-to-Fine Iteration**: Starting from the coarsest scale $B_n$, alternately optimize the intermediate sharp image $L$ and blur kernel $K$
4. **Residual Guidance Correction**: At each cross-scale transition, RGS corrects the input to the next finer scale to suppress noise
5. **Non-blind Restoration**: Given the final estimated kernel $K_1$, a non-blind method (Zhong et al.) is applied for the final deblurring

### Key Designs

1. **Alternating Optimization Subproblems**:

    - **L-subproblem** (Eq. 5): Given $K^t$, estimate the sharp image $L$ using the prior from Liu et al.—$\ell_0$ norm for salient edge selection plus image surface area regularization—solved via half-quadratic splitting.
    - **K-subproblem** (Eq. 6): Given $L^{t+1}$, estimate the kernel $K$ using **image gradients** rather than raw pixel values (gradient domain is more stable), with $\ell_2$ regularization and efficient FFT-based solution.

2. **Residual Guidance Strategy (RGS)**: The core contribution. At the transition from scale $i+1$ (coarse) to scale $i$ (fine):

    - Compute the residual: $R_i = B_i - \text{Up}(L_{i+1} \otimes K_{i+1})$, i.e., the difference between the current-scale blurry image and the upsampled estimate from the coarser scale.
    - The residual primarily contains noise along with some structural information.
    - Apply a **guided filter** to the residual, using its Gaussian-smoothed version as the guidance image, yielding a denoised residual $\tilde{R}_i = g(R_i)$.
    - Correct the current-scale input: $\tilde{B}_i = \text{Up}(L_{i+1} \otimes K_{i+1}) + \tilde{R}_i$.
    - Substitute $\tilde{B}_i$ for the original $B_i$ in subsequent alternating optimization at this scale.

3. **Comparison with Naive Guidance Strategy (NGS)**: An intuitive baseline directly applies a guided filter to the upsampled estimate. Experiments show that NGS is inferior to RGS, as direct filtering of the blurry image may destroy important structural details. In contrast, **RGS filters only the residual**, preserving the dominant structural content.

### Loss & Training

No training is required. The optimization objective is:
$$\min_{K,L} \|K \otimes L - B\|_2^2 + \lambda P_l(L) + \mu P_k(K)$$

where $P_l(L) = \|\nabla L\|_0 + \gamma \sum_{i,j}\sqrt{1 + |\nabla_{i,j}L|^2}$ is the sharp image prior, and $P_k(K) = \|K\|_2^2$ is the kernel regularization term.

Parameter settings: $\lambda = \gamma = 0.004$, maximum outer iterations $M=5$. The guided filter window size $w$ and smoothing parameter $s$ are also specified. The non-blind restoration stage incorporates NLM-based denoising enhancement for improved performance under high-noise conditions.

## Key Experimental Results

All experiments are conducted under Gaussian noise with σ=0.1, which represents a challenging noise level.

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Lai et al. | PSNR/SSIM/LPIPS | **21.41**/0.75/0.18 | 18.48/0.53/0.34 (Li) | +2.93dB |
| Zhang et al. (Face) | PSNR/SSIM/LPIPS | **24.73**/0.66/0.46 | 23.37/0.60/0.47 (Li) | +1.36dB |
| Levin et al. | PSNR/SSIM | **Best** | — | Significant margin |
| RealBlur (Real) | PSNR/SSIM/LPIPS | **25.07**/0.72/0.20 | 24.06/0.67/0.23 (Anger) | +1.01dB |

**RGS as a plug-in module for other methods** (Table 2, σ=0.1):

| Method | Original PSNR (Lai) | +RGS PSNR (Lai) | Gain |
|------|----------------|-----------------|------|
| Dong et al. | 17.88 | 19.45 | +1.57dB |

### Ablation Study

- **NGS vs. RGS** (Table 3): On the Lai dataset, RGS achieves PSNR 19.45 vs. NGS 18.96 (+0.49dB), and kernel accuracy MNC 0.66 vs. 0.60, with a clear advantage for RGS.
- **Stability**: Across 100 independent runs, PSNR variance is only $5.1\times10^{-3}$ and SSIM variance is $6.93\times10^{-6}$, demonstrating high stability.
- **Mixed Noise** (Gaussian-Poisson mixture, σ=0.05, λ=20): Residual-based denoising in RGS naturally adapts to different noise types.
- **Generality of RGS**: Can be integrated into other blind deblurring methods (e.g., Dong et al.), yielding consistent PSNR and SSIM improvements across four datasets.

## Highlights & Insights

- **Minimal yet effective**: A training-free, purely traditional optimization approach that outperforms all deep learning methods under high noise, demonstrating that physical modeling combined with principled strategy design still holds considerable potential.
- **Precise core insight**: The observation that coarser-scale kernel estimates are more accurate but coarser, while finer-scale estimates are more detailed but noise-corrupted, directly drives the method design.
- **Residual rather than image**: Filtering the residual instead of the image directly avoids destroying structural information—a simple but critical design choice.
- **Universal plug-in**: RGS can serve as a drop-in module to enhance any blind deblurring method built on a coarse-to-fine framework.
- **Strong robustness**: Effective across unknown noise types (Gaussian, Poisson, mixed), as residual filtering is inherently insensitive to noise distribution.

## Limitations & Future Work

- **No handling of dynamic scene blur**: The method assumes a linear convolution degradation model (spatially invariant kernel) and is not applicable to dynamic scenes.
- **Fixed guided filter**: The authors acknowledge that the guided filter could be replaced by a more powerful filter, representing a potential direction for improvement.
- **Replaceable non-blind restoration**: The current non-blind restoration module could be substituted with more advanced alternatives.
- **Computational efficiency not discussed**: Traditional iterative methods are generally slower than end-to-end deep learning approaches.
- **Only spatially invariant kernels evaluated**: Real-world blur is often spatially varying.

## Related Work & Insights

- **vs. Traditional methods** (Dong et al., Zhong et al., Anger et al.): These methods are effective under low noise but suffer catastrophic kernel estimation failure at high noise levels. RGS substantially improves robustness.
- **vs. DIP/VDIP/WDIP** (deep image prior family): These unsupervised methods exploit network structural priors for blind deblurring but tend to overfit to noise under high-noise conditions. The proposed method does not suffer from this issue.
- **vs. End-to-end methods** (DeblurGAN-v2, Zhang et al.): These methods rely on large-scale training data, generalize poorly to high-noise scenarios, and cannot produce explicit kernel estimates. The proposed method offers stronger interpretability.
- **vs. Lee et al. (ECCV 2024)**: While also targeting noise-robust blind deblurring, the proposed method achieves superior performance.

**Further Inspirations**:

- **Learnable residual filtering**: Replacing the guided filter with a lightweight learnable network for adaptive filtering strength could be a valuable improvement direction.
- **Cross-task transfer**: The residual guidance concept—using coarse-scale information to correct fine-scale inputs—can potentially generalize to other coarse-to-fine image restoration tasks such as super-resolution and dehazing.
- **Integration with diffusion models**: Analogous cross-step residual guidance could be incorporated into the denoising process of diffusion-based restoration models.

## Rating

- Novelty: ⭐⭐⭐⭐ — The residual guidance concept is concise and elegant; while technically straightforward, the underlying insight is sharp.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four datasets, multiple noise types, complete ablations, stability verification, and plug-in experiments; computational efficiency comparison is absent.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, logic is coherent, and figures and tables are persuasive.
- Value: ⭐⭐⭐⭐ — A training-free traditional method outperforming deep learning approaches in high-noise scenarios, with a generalizable plug-in module offering high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Efficient Concertormer for Image Deblurring and Beyond](efficient_concertormer_for_image_deblurring_and_beyond.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[NeurIPS 2025\] The Effect of Optimal Self-Distillation in Noisy Gaussian Mixture Model](../../NeurIPS2025/image_restoration/the_effect_of_optimal_self-distillation_in_noisy_gaussian_mixture_model.md)
- [\[CVPR 2026\] BluRef: Unsupervised Image Deblurring with Dense-Matching References](../../CVPR2026/image_restoration/bluref_unsupervised_image_deblurring_with_dense-matching_references.md)
- [\[NeurIPS 2025\] DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance](../../NeurIPS2025/image_restoration/dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)

</div>

<!-- RELATED:END -->
