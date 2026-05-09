---
title: >-
  [Paper Note] AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction
description: >-
  [AAAI 2026][Image Generation][Image Attribution] This paper proposes a training-free image attribution method based on the ratio of autoencoder double-reconstruction losses. By incorporating image uniformity calibration to eliminate texture complexity bias, the method achieves an average accuracy of 95.1% across 8 mainstream diffusion models, surpassing the strongest baseline by 24.7%, while being approximately 100× faster.
tags:
  - AAAI 2026
  - Image Generation
  - Image Attribution
  - Autoencoder Reconstruction
  - Training-Free
  - Latent Diffusion Models
  - Kernel Density Estimation
date: 2026-05-08
content_hash: 196516b499689924
---

# AEDR: Training-Free AI-Generated Image Attribution via Autoencoder Double-Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2507.18988v2](https://arxiv.org/abs/2507.18988v2)
**Code**: [Available](https://github.com/wangchao0708/AEDR)
**Area**: Image Generation
**Keywords**: Image Attribution, Autoencoder Reconstruction, Training-Free, Latent Diffusion Models, Kernel Density Estimation

## TL;DR

This paper proposes a training-free image attribution method based on the ratio of autoencoder double-reconstruction losses. By incorporating image uniformity calibration to eliminate texture complexity bias, the method achieves an average accuracy of 95.1% across 8 mainstream diffusion models, surpassing the strongest baseline by 24.7%, while being approximately 100× faster.

## Background & Motivation

### State of the Field

**Background**: With the widespread adoption of powerful latent diffusion models (LDMs) such as Stable Diffusion and FLUX, anyone can easily generate photorealistic images, raising serious security concerns — malicious actors may misappropriate others' model outputs or exploit commercial models for illicit gain. Accurately tracing the source model of a generated image (origin attribution) has therefore become critically important.

Existing passive attribution methods (reconstruction-based approaches such as RONAN and LatentTracer) rely on gradient-guided reconstruction and compare absolute reconstruction losses to determine attribution. However, these methods suffer from two critical problems:
1. **Failure on next-generation high-quality models**: The autoencoders of models such as FLUX achieve extremely high reconstruction quality, causing the reconstruction losses of both belonging and non-belonging images to collapse to similarly low values (~10⁻⁵), making their distributions indistinguishable.
2. **High computational cost**: The gradient optimization process requires numerous iterations, taking 30–160 seconds per image.

### Starting Point

**Goal**: How to design an accurate and efficient image attribution method — without additional training or modification of generative models — that remains robust on the latest diffusion models including FLUX?

## Method

### Overall Architecture

AEDR consists of three modules:
1. **Double Reconstruction**: The target model's autoencoder performs two successive encode-decode reconstructions on the test image.
2. **Uniformity Calibration**: A Gray-Level Co-occurrence Matrix (GLCM)-based uniformity metric is computed to calibrate the attribution signal.
3. **Threshold Determination**: Kernel density estimation (KDE) is used to adaptively determine the decision threshold.

### Key Designs

**Core Observation of Double Reconstruction**:
- Images belonging to the target model lie within the autoencoder's learned distribution; the losses of two successive reconstructions are nearly identical, yielding a ratio $t = \mathcal{L}_1 / \mathcal{L}_2 \approx 1$.
- Non-belonging images initially lie outside the distribution; the first reconstruction projects them into the distribution, causing the second reconstruction loss to decrease substantially, resulting in $t \gg 1$.

The key insight is that using the **ratio** of losses rather than their **absolute values** inherently eliminates the scale discrepancy arising from differences in reconstruction quality across models.

**Uniformity Calibration**:
The GLCM-based uniformity measure is defined as:
$$\mathcal{H} = \sum_{i=0}^{\ell-1}\sum_{j=0}^{\ell-1} \frac{P(i,j)}{1+|i-j|}$$

Images with simple textures exhibit high uniformity but small variation in the double-reconstruction ratio, whereas complex-texture images behave oppositely. Calibrating via $t' = t \times \mathcal{H}$ effectively mitigates the interference of image complexity on the attribution signal.

**Adaptive Thresholding**: KDE is applied to estimate the distribution of the calibrated signal $t'$, and the $1-\alpha$ quantile of the CDF is taken as the decision threshold $\tau$. The hyperparameter $\alpha$ is model-specific and selected on a validation set.

### Loss & Training

AEDR is entirely training-free. Reconstruction loss is computed using MSE (ablation studies confirm it outperforms MAE, SSIM, and LPIPS). Threshold determination requires only forward-pass statistics over 500 belonging images.

## Key Experimental Results

- **8 models evaluated**: SD1.5, SD2base, SD2.1, SDXL, SD3.5, FLUX, VQDiffusion, Kandinsky 2.1
- **Attribution accuracy** (Table 1, belonging vs. images from other models): AEDR average **95.1%** vs. LatentTracer 70.4% vs. RONAN 50.3%
- **Distinguishing belonging vs. real images** (Table 2): AEDR average **96.9%** vs. LatentTracer 66.7% vs. RONAN 52.2%
- **Runtime efficiency** (Table 3): AEDR 0.27–1.25 s/image vs. LatentTracer 12–163 s/image, approximately **100× speedup**
- **Generalization** (Table 4): Accuracy >96% on VAE, 90.85% on VQ-VAE, 82.93% on MoVQ

### Ablation Study

1. **Reconstruction loss metric**: MSE (99.4%) > SSIM (99.1%) > MAE (97.2%) > LPIPS (90.7%)
2. **Effect of uniformity calibration**: Improvements of 0.18%–9.09% across most models; slight degradation on FLUX and VQDM
3. **Quantile selection**: Optimal $\alpha$ varies substantially across models (0.003–0.085), validating the necessity of KDE-based adaptive selection

## Highlights & Insights

1. **Elegant and principled insight**: Using the double-reconstruction loss ratio as the attribution signal is conceptually clean — belonging images are approximate fixed points of the autoencoder, while non-belonging images are "pulled into" the distribution by the first reconstruction.
2. **Truly training-free**: Only forward passes through the autoencoder are required; no gradient computation or classifier training is needed.
3. **Effective on state-of-the-art models**: Addresses the complete failure of existing methods on high-quality autoencoders such as FLUX.
4. **High efficiency**: Over 100× speedup makes practical deployment feasible.

## Limitations & Future Work

1. **Performance degradation on quantized autoencoders**: Accuracy drops to 82.93% on MoVQ; the discrete quantization in VQ-VAE degrades reconstruction fidelity, undermining the fixed-point assumption. Designing analogous attribution signals for discrete latent spaces remains an open problem.
2. **White-box assumption**: Access to the target model's autoencoder is required, making the method inapplicable to fully black-box commercial APIs (e.g., DALL·E 3).
3. **Limited to LDM-family models**: Coverage does not extend to generators with different architectures such as GANs or autoregressive models (e.g., the DALL·E series).
4. **Robustness not fully validated**: The paper does not evaluate attribution robustness under common post-processing operations such as JPEG compression, resizing, or cropping.
5. **Model-specific threshold calibration**: Each model requires 500 belonging samples for threshold calibration, which may introduce overhead as the number of models grows.

## Related Work & Insights

| Method | Type | Avg. Accuracy | Speed | Training-Free | Supports FLUX |
|--------|------|--------------|-------|---------------|---------------|
| RONAN | Gradient reconstruction | 50.3% | Very slow | Yes | ✗ |
| LatentTracer | Gradient reconstruction | 70.4% | Slow (12–163 s) | Yes | ✗ |
| AEROBLADE | AE reconstruction | — | Fast | Yes | Detection only |
| **AEDR** | **AE double reconstruction** | **95.1%** | **Fast (0.06–1.25 s)** | **Yes** | **✓** |

RONAN and LatentTracer rely on the absolute value of single-pass reconstruction loss, which collapses when facing high-quality autoencoders. AEROBLADE also employs AE reconstruction but only performs detection (real vs. fake) rather than source attribution. AEDR resolves this limitation through the ratio-plus-calibration formulation.

## Related Work & Insights

1. **Transferability of the fixed-point intuition**: The core idea that "in-distribution samples are approximate fixed points of the autoencoder" may generalize to other detection and attribution scenarios, such as identifying images processed by a specific image editing model.
2. **Connection to idea `20260316_semantic_watermark_provenance`**: AEDR provides a purely passive attribution path that is complementary to the retrieval-based attribution approach of LIDA. Its training-free nature makes it a candidate rapid pre-screening module for large-scale attribution systems.
3. **Generality of uniformity calibration**: GLCM-based image complexity calibration is a broadly applicable technique that can be borrowed for other tasks relying on reconstruction error, such as anomaly detection and image quality assessment.
4. **Failure on quantized autoencoders** leaves open research space for attribution in VQ-based models, potentially requiring attribution signals designed at the level of quantization codebooks rather than pixels.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The double-reconstruction ratio attribution approach is concise and original)
- **Technical Contribution**: ⭐⭐⭐⭐ (A complete attribution framework with calibration and adaptive thresholding)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (8 models, multiple AE types, comprehensive ablations)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear motivation, intuitive figures and tables)
- **Practical Impact**: ⭐⭐⭐⭐ (Training-free + 100× speedup enables real-world deployment)
- **Overall Recommendation**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution](../../CVPR2026/image_generation/attribution_as_retrieval_modelagnostic_aigenerated.md)
- [\[AAAI 2026\] Infinite-Story: A Training-Free Consistent Text-to-Image Generation](infinite-story_a_training-free_consistent_text-to-image_gene.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[AAAI 2026\] Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)

</div>

<!-- RELATED:END -->
