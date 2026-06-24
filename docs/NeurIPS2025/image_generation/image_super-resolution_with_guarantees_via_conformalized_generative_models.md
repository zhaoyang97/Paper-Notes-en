---
title: >-
  [Paper Note] Image Super-Resolution with Guarantees via Conformalized Generative Models
description: >-
  [NEURIPS2025][Image Generation][Super-Resolution] This work applies Conformal Prediction to construct binary "confidence masks" for generative image super-resolution models, reliably identifying trustworthy regions in generated images with rigorous statistical guarantees.
tags:
  - "NEURIPS2025"
  - "Image Generation"
  - "Super-Resolution"
  - "Uncertainty Quantification"
  - "Conformal Prediction"
  - "Diffusion Models"
  - "Confidence Masks"
date: 2026-05-08
content_hash: 97426f3af9c0fb0d
---

# Image Super-Resolution with Guarantees via Conformalized Generative Models

**Conference**: NEURIPS2025
**arXiv**: [2502.09664](https://arxiv.org/abs/2502.09664)  
**Code**: [adamesalles/experiments-conformal-superres](https://github.com/adamesalles/experiments-conformal-superres)  
**Area**: Image Generation
**Keywords**: Super-Resolution, Uncertainty Quantification, Conformal Prediction, Diffusion Models, Confidence Masks

## TL;DR

This work applies Conformal Prediction to construct binary "confidence masks" for generative image super-resolution models, reliably identifying trustworthy regions in generated images with rigorous statistical guarantees.

## Background & Motivation

- Generative foundation models (especially diffusion models) have achieved remarkable progress in image super-resolution, yet the reliability of their predictions lacks quantification — models may hallucinate in certain regions.
- Two major limitations of existing uncertainty quantification methods:
    1. **Poor interpretability**: Angelopoulos et al. (2022b) produce interval-valued images (each pixel represented by a confidence interval), which are difficult for users to intuitively understand.
    2. **Lack of probabilistic guarantees**: Kutiel et al. (2023) generate continuous confidence scores without rigorous statistical validity proofs.
- Practical deployment scenarios (consumer devices, medical imaging, etc.) urgently require an uncertainty quantification framework that is **both intuitively interpretable and theoretically grounded**.

## Core Problem

How can one determine, for each pixel in a super-resolved image, whether it is "trustworthy" or "untrustworthy" — without relying on the internal structure of the generative model (black-box / API mode included) — while providing controllable fidelity error guarantees?

## Method

### 2.1 Conformal Mask Calibration

**Input**: An arbitrary black-box super-resolution model $\mu$, an uncertainty estimation function $\sigma$, and $n$ labeled high-resolution calibration image pairs $(X_i, Y_i)_{i=1}^n$.

**Core Idea**: A binary mask $M_\alpha(X)$ is constructed by thresholding the output of $\sigma$; pixels marked as "trustworthy" satisfy an uncertainty value below threshold $t_\alpha$.

**Threshold Calibration Formula**:

$$t_\alpha = \sup\left\{t \in \mathbb{R} \cup \{+\infty\}: \frac{1}{n+1}\sum_{i=1}^n \sup_{p;[\sigma(X_i)]_p \le t} D_p(Y_i, \mu(X_i)) + \frac{3}{n+1} \le \alpha \right\}$$

where $D_p$ is a local image fidelity metric and $\alpha$ is the desired fidelity level.

**Mask Generation**: $M_\alpha(X) = \{p : [\sigma(X)]_p \le t_\alpha\}$

**Core Theorem (Theorem 2.1)**: If calibration and test samples are i.i.d., then:

$$\mathbb{E}\left[\sup_{p \in M_\alpha(X_{n+1})} D_p(\mu(X_{n+1}), Y_{n+1})\right] \le \alpha$$

That is, the maximum fidelity error within the trusted mask region does not exceed $\alpha$ in expectation.

**Computational Efficiency**: Using dynamic programming, $t_\alpha$ can be computed in $O(nd\log d)$ time ($n$ = number of calibration images, $d$ = number of pixels), far superior to the brute-force $\Omega(n^2 d^2)$.

### 2.2 Construction of Uncertainty Score Masks

A good $\sigma$ should assign higher values in regions of greater uncertainty. Three strategies are proposed:

1. **Per-pixel variance** $\sigma^{\text{var}}$: Generate $M$ super-resolution outputs for the same low-resolution input and compute the empirical variance per pixel. This approach is overly local — slight edge misalignments cause artificially high variance.
2. **Low-pass filtered patch variance**: Apply a convolution kernel $K$ to the second-moment decomposition of variance. Setting $K$ to a unit box kernel reduces this to per-pixel variance. Key formula: $[\sigma^K(X)]_p = \hat{\mathbb{E}}_M[[\mu(X)^2 * K]_p] - (\hat{\mathbb{E}}_M[[\mu(X) * K]_p])^2$
3. **Additional Gaussian smoothing**: Apply Gaussian blurring on top of patch variance to suppress over-emphasis of edge artifacts.

### 2.3 Selection of Local Fidelity Metric $D_p$

Three metrics are explored, all constrained to $0 \le D_p \le 3$:

| Metric | Definition | Characteristics |
|--------|------------|-----------------|
| Pointwise | $D_p = \|[Y]_p - [\hat{Y}]_p\|_1$ | Simplest, but sensitive to single-pixel errors |
| Neighborhood average | $D_p = \|[Y*K]_p - [\hat{Y}*K]_p\|_1$ | Incorporates spatial context; produces larger, more stable masks |
| Semantic | $D_p = [S(Y, \hat{Y})]_p$ (human-annotated difference) | Best captures semantic discrepancies, but requires human annotation |

All comparisons are performed in the Lab color space to ensure perceptual uniformity.

## Additional Theoretical Guarantees

### PSNR Control (Proposition 3.1)

Beyond controlling the custom fidelity error, the method also yields a lower bound on PSNR:

$$\mathbb{E}[\text{PSNR}(\mu(X_{n+1}), Y_{n+1} | M_\alpha(X_{n+1}))] \ge -20\log_{10}\alpha$$

For example, at $\alpha = 0.1$, the PSNR lower bound is 20 dB.

### Robustness to Data Leakage (Proposition 3.2)

When $n_{\text{leaked}}$ of the $n$ calibration samples actually originate from the training data, the fidelity error upper bound degrades to:

$$\text{Fidelity Error} \le \alpha \cdot \frac{n_{\text{new}} + n_{\text{leaked}} + 1}{n_{\text{new}} + 1}$$

When the leakage fraction is small, the guarantee degrades only mildly — a property particularly relevant for scenarios involving large-scale pretrained foundation models.

## Key Experimental Results

**Dataset**: Liu4K (1,600 training + 400 validation), real 4K high-resolution images.
**Base Model**: SinSR (a single-step diffusion-based super-resolution method).
**Hardware**: Intel Xeon E5-2696 v2 + NVIDIA RTX 6000 Ada 48 GB.

| Fidelity $\alpha$ | Semantic $D_p$ PSNR | Semantic Mask Size | Non-semantic $D_p$ PSNR | Non-semantic Mask Size |
|:---:|:---:|:---:|:---:|:---:|
| 0.075 | 32.75 ± 1.55 | 0.77 ± 0.07 | 30.23 ± 1.12 | 0.43 ± 0.09 |
| 0.100 | 32.65 ± 1.48 | 0.73 ± 0.07 | 28.64 ± 0.93 | 0.23 ± 0.06 |
| 0.200 | 31.63 ± 1.32 | 0.60 ± 0.08 | 26.82 ± 1.03 | 0.00 ± 0.00 |
| No method | 26.83 ± 1.06 | N/A | 26.82 ± 1.08 | N/A |

**Key Findings**:
- Fidelity error is strictly controlled below $\alpha$, with empirical values nearly matching the theoretical upper bound.
- Masks under the semantic metric $D_p$ are substantially larger than those under non-semantic metrics, demonstrating that incorporating semantic information improves trustworthy region coverage.
- Even when the base model fails (e.g., hallucinating high-frequency content in blurry regions), the confidence mask accurately identifies unreliable areas.

## Highlights & Insights

1. **Fully black-box compatible**: Only the input–output interface of the super-resolution model is required; no access to model weights or intermediate features is needed, supporting API-only models.
2. **Comprehensive statistical guarantee system**: A complete theoretical chain spanning fidelity error control (Theorem 2.1), PSNR lower bounds (Proposition 3.1), and data leakage robustness (Proposition 3.2).
3. **Efficient dynamic programming calibration**: $O(nd\log d)$ complexity makes calibration on large-scale data tractable.
4. **Flexible metric design**: $D_p$ can be freely chosen from pixel-level to semantic-level, accommodating diverse application scenarios.
5. **Beyond super-resolution**: Appendix experiments demonstrate successful transfer to image colorization, suggesting broad applicability of the framework.
6. **Counterexample against Kutiel et al.**: Appendix B rigorously proves that the method of Kutiel et al. does not satisfy its claimed statistical guarantees.

## Limitations & Future Work

1. **Exchangeability assumption**: The core theorem relies on the i.i.d. (or exchangeable) assumption between calibration and test data; significant distribution shift may invalidate the guarantees.
2. **Computational overhead**: Estimating $\sigma$ (pixel-level variance) requires multiple invocations of the generative model, increasing inference time.
3. **Decoupling of $\sigma$ and $\mu$**: Uncertainty estimation and super-resolution image generation are currently separate; joint estimation may yield further improvements.
4. **Annotation cost of semantic metric**: Using semantic $D_p$ requires human annotation of difference regions, limiting scalability.
5. **Loose PSNR lower bound**: The theoretical lower bound of Proposition 3.1 deviates noticeably from empirical values at larger $\alpha$.

## Related Work & Insights

| Method | Output Form | Statistical Guarantees | Interpretability | Model Requirements |
|--------|-------------|----------------------|------------------|--------------------|
| **Ours** | Binary confidence mask | ✅ Fidelity error + PSNR + data leakage robustness | ⭐⭐⭐ Intuitive | Black-box |
| Angelopoulos et al. (2022b) | Interval-valued image | ✅ Pixel interval coverage | ⭐ Hard to interpret | Black-box |
| Kutiel et al. (2023) | Continuous confidence scores | ❌ (disproved by counterexample in appendix) | ⭐⭐ Moderate | Requires internal access |
| BNN / MC Dropout | Variance map | ❌ No formal guarantees | ⭐⭐ Continuous values | Requires model architecture |

### Further Insights

- **New paradigm of conformal prediction in vision**: This work extends conformal prediction from classification/semantic segmentation to pixel-level generative quality assessment, providing a general post-hoc uncertainty quantification template.
- **Trustworthy AI deployment**: In high-stakes scenarios such as medical image super-resolution, statistically guaranteed confidence masks offer greater practical value than conventional heatmaps.
- **Applicability beyond super-resolution**: The framework is naturally suited to any "low-quality → high-quality" image restoration task (denoising, colorization, inpainting); appendix experiments confirm feasibility on colorization.
- **Dual role of low-pass filtering**: It both improves the estimation of $\sigma$ (reducing spurious edge variance) and relaxes the locality of $D_p$ (yielding larger masks) — a design principle worth borrowing in other pixel-level evaluation settings.

## Rating

- Novelty: ⭐⭐⭐⭐ (Introducing conformal risk control into image super-resolution confidence masks is methodologically novel and theoretically rigorous)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Liu4K dataset + distribution shift + data leakage + colorization transfer; comprehensive coverage)
- Writing Quality: ⭐⭐⭐⭐⭐ (Theorem statements are clear, experimental visualizations are intuitive, appendix is thorough)
- Value: ⭐⭐⭐⭐ (Provides a practical tool for trustworthy deployment of generative models, though computational overhead and the i.i.d. assumption limit direct applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](../../CVPR2026/image_generation/vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[ICCV 2025\] PatchScaler: An Efficient Patch-Independent Diffusion Model for Image Super-Resolution](../../ICCV2025/image_generation/patchscaler_an_efficient_patch-independent_diffusion_model_for_image_super-resol.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](../../ICCV2025/image_generation/bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[CVPR 2025\] Arbitrary-Steps Image Super-Resolution via Diffusion Inversion](../../CVPR2025/image_generation/arbitrary-steps_image_super-resolution_via_diffusion_inversion.md)
- [\[CVPR 2025\] FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-Resolution](../../CVPR2025/image_generation/faithdiff_unleashing_diffusion_priors_for_faithful_image_super-resolution.md)

</div>

<!-- RELATED:END -->
