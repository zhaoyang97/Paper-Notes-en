---
title: >-
  [Paper Note] WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images
description: >-
  [AAAI 2026][Medical Imaging][Microaneurysm Detection] This paper proposes WDT-MD, a framework that addresses three fundamental challenges in fundus image microaneurysm (MA) detection—identity mapping…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Microaneurysm Detection"
  - "Diffusion Models"
  - "Wavelet Transform"
  - "Anomaly Detection"
  - "Diabetic Retinopathy"
date: 2026-05-08
content_hash: d8009e6c6a2de19f
---

# WDT-MD: Wavelet Diffusion Transformers for Microaneurysm Detection in Fundus Images

**Conference**: AAAI 2026  
**arXiv**: [2511.08987](https://arxiv.org/abs/2511.08987)  
**Code**: [GitHub](https://github.com/diaoquesang/WDT-MD)  
**Area**: Medical Imaging  
**Keywords**: Microaneurysm Detection, Diffusion Models, Wavelet Transform, Anomaly Detection, Diabetic Retinopathy

## TL;DR

This paper proposes WDT-MD, a framework that addresses three fundamental challenges in fundus image microaneurysm (MA) detection—identity mapping, high false positives, and poor normal-feature reconstruction quality—through noise-encoded image conditioning, pseudo-normal pattern synthesis, and a wavelet diffusion Transformer architecture.

## Background & Motivation

1. **Background**: Microaneurysms (MAs) are the earliest pathological hallmarks of diabetic retinopathy (DR), with diameters of only 15–60 μm (approximately 6 pixels in fundus images), exhibiting large variations in brightness, contrast, and morphology. Manual screening is time-consuming and error-prone.
2. **Limitations of Prior Work**: Discriminative models (segmentation-based) suffer from annotation difficulty and severe class imbalance (positive pixels <1%). Reconstruction-based generative anomaly detection methods (AE, GAN, diffusion models) reduce annotation dependency but introduce three core problems.
3. **Key Challenge**:
    - **Identity mapping**: Diffusion models directly copy the input (including anomalous regions), causing missed detections.
    - **High false positives**: Without pixel-level supervision, the model cannot distinguish MAs from other anomalies (artifacts, merged lesions).
    - **Poor reconstruction quality**: Normal vascular textures cannot be reconstructed accurately, introducing spurious reconstruction errors.
4. **Goal**: Simultaneously address the above three problems to achieve high-precision pixel-level MA segmentation and image-level classification.
5. **Key Insight**: Transfer Diffusion Transformers (DiT) to the wavelet domain, and introduce noise-perturbed image conditioning and pseudo-normal label synthesis during training.
6. **Core Idea**: Perform anomaly detection with DiT in the wavelet domain; perturb the conditioning image with noise during training to prevent identity mapping; and leverage inpainting-based pseudo-normal label synthesis to provide pixel-level supervision.

## Method

### Overall Architecture

WDT-MD is a supervised, DiT-based anomaly detection framework operating in the wavelet domain. The input fundus image is first converted to the V channel of HSV, then decomposed via DWT into four subbands ($V_{LL}, V_{LH}, V_{HL}, V_{HH}$). These are concatenated and iteratively denoised by a diffusion model in the wavelet domain to reconstruct a pseudo-normal image. Pixel-level and image-level predictions are subsequently derived from the residual map (input − reconstruction).

### Key Designs

1. **Wavelet Diffusion Transformer Architecture (Wavelet DiT)**:
    - **Function**: Conducts the diffusion process in the wavelet domain, replacing the pixel domain or AE latent space.
    - **Mechanism**: DWT decomposes the V channel into low-frequency ($V_{LL}$) and high-frequency ($V_{LH}, V_{HL}, V_{HH}$) subbands, which are concatenated as the diffusion model input. The forward process follows $z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$; the reverse process uses DiT ($N=12$ blocks) as the noise estimation network.
    - **Design Motivation**: Compared to AE-based tokenizers (e.g., VQGAN), DWT enables near-lossless reconstruction at low computational cost. The V channel is selected because the H/S channels carry substantial noise but limited useful information. DiT's global modeling capability is well-suited for capturing the spatial distribution context of MAs.

2. **Noise-Encoded Image Conditioning**:
    - **Function**: Perturbs the conditioning image with random noise during training to prevent identity mapping.
    - **Mechanism**: The image condition $\tilde{z}$ is perturbed to $\widetilde{z_\delta} = \sqrt{\bar{\alpha}_\delta} \tilde{z} + \sqrt{1-\bar{\alpha}_\delta} \epsilon$, where $\delta \in \{1, ..., \delta_{max}\}$ is randomly sampled. The modified diffusion loss is $\mathcal{L}(\epsilon_\theta) = \sum_{t=1}^{T} \mathbb{E}_{z_0, \delta, \epsilon} [\| \epsilon_\theta(z_t, t, \widetilde{z_\delta}) - \epsilon \|_2^2]$.
    - **Design Motivation**: Existing methods that add and remove noise at inference time face a frequency-resolution conflict—MA details and vascular textures occupy overlapping high-frequency bands yet require opposite noise treatment. Dynamically perturbing the condition during training forces the model to learn normal patterns rather than directly copying the input.

3. **Pseudo-Normal Pattern Synthesis**:
    - **Function**: Introduces pixel-level supervision signals to reduce false positives.
    - **Mechanism**: The Telea inpainting algorithm is applied to restore anomalous regions to normal appearance based on the MA mask $M$: $V_{pn} = (1-M) \odot V + M \odot \mathcal{I}(V, M)$, which serves as the training ground truth.
    - **Design Motivation**: Unlike approaches that synthesize anomalies on normal images, this method infers unknown regions from known normal pixels, ensuring accurate spatial distribution of pixel-level supervision and enabling the model to distinguish MAs from other anomalies.

### Loss & Training

- Diffusion loss (Eq. 9) with noise-encoded conditioning.
- AdamW optimizer, initial learning rate $10^{-4}$, with dynamic learning rate scheduling.
- Noise schedule $\beta_t$: 0.00085 → 0.012, $T=1000$ steps.
- Inference sampling steps $T_s = 50$ (LCM sampler).
- Daubechies-6 wavelet basis; inpainting radius $r=3$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|--------|----------|------|
| IDRiD | Pixel AUC | **82.80%** | 81.82% (Dif-fuse) | +0.98% |
| IDRiD | Pixel F1 | **74.43%** | 69.55% (Dif-fuse) | +4.88% |
| IDRiD | Image AUC | **85.95%** | 77.45% (CPC) | +8.50% |
| IDRiD | Image F1 | **82.35%** | 70.59% (CPC) | +11.76% |
| e-ophtha MA | Pixel AUC | **81.08%** | 80.82% (Dif-fuse) | +0.26% |
| e-ophtha MA | Pixel F1 | **57.70%** | 42.99% (DTU-Net) | +14.71% |
| e-ophtha MA | Image AUC | **70.83%** | 65.42% (CPC) | +5.41% |

### Ablation Study

| Configuration | IDRiD Pixel AUC | IDRiD Image AUC | Note |
|------|----------------|-----------------|------|
| w/o noise encoding + w/o pixel supervision | 73.17% | 48.37% | Baseline |
| Noise encoding only (τ) | 67.20% | 42.16% | Identity mapping only |
| Pixel supervision only (ψ) | 49.98% | 42.81% | False positive reduction only |
| Both (full model) | **82.80%** | **85.95%** | Complementary; neither alone suffices |

### Key Findings

- The DWT tokenizer outperforms all learned tokenizers (AE-KL, VQ-VAE, VQGAN) while requiring the fewest parameters (35.04M) and lowest FLOPs (119.76G).
- The DiT backbone ($N=12$) reduces parameters by 90.85% and FLOPs by 38.10% compared to Attention U-Net.
- The optimal noise encoding timestep is $\delta_{max} = 10$: moderate perturbation is effective, while excessive noise hinders convergence.
- Noise encoding and pixel supervision are mutually indispensable; each component used in isolation performs worse than the baseline.

## Highlights & Insights

- The paper precisely identifies three core failure modes of diffusion-based MA detection and proposes targeted solutions for each.
- The combination of wavelet domain and DiT is particularly elegant: DWT naturally separates high- and low-frequency information, facilitating discrimination between micro-lesions and normal textures.
- Using the V channel of HSV as input is a simple yet effective engineering decision that substantially reduces computational overhead.
- The pseudo-normal synthesis strategy is conceptually novel: rather than synthesizing anomalies on normal images, it restores anomalous regions back to normal appearance.

## Limitations & Future Work

- Validation is limited to two datasets (IDRiD and e-ophtha MA) with relatively small sample sizes (249 and 381 samples, respectively).
- Downsampling to 300×200 discards some fine-grained detail.
- Pseudo-normal synthesis relies on MA masks, requiring annotated anomalous images.
- Processing only the V channel may discard weakly informative signals present in the H/S channels.
- No comparison with recent foundation model-based methods is provided.

## Related Work & Insights

- **vs. Dif-fuse (TMI24)**: Dif-fuse applies noise-then-denoise at inference, facing a frequency-resolution conflict; WDT-MD resolves this issue during training.
- **vs. AnoDDPM (CVPR22)**: AnoDDPM employs simplex noise but lacks pixel-level supervision, resulting in high false positives.
- **vs. HACDR-Net (AAAI24)**: A U-Net-based segmentation method severely impacted by class imbalance (F1 of only 4.03%).
- **vs. Img-Cond (Baugh 2024)**: The absence of conditioning image processing propagates identity mapping artifacts.

## Rating

- Novelty: ⭐⭐⭐⭐ Each component exhibits thoughtful design, though the core ideas (perturbed conditioning + pseudo labels) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations (components, tokenizers, backbones, hyperparameters), but evaluation is limited to two small-scale datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, methodological derivations are rigorous, and figures/tables are of high quality.
- Value: ⭐⭐⭐⭐ Clinically valuable for early DR screening, but larger-scale validation is needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards Effective and Efficient Context-aware Nucleus Detection in Histopathology Whole Slide Images](towards_effective_and_efficient_context-aware_nucleus_detection_in_histopatholog.md)
- [\[AAAI 2026\] Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs](training-free_policy_violation_detection_via_activation-space_whitening_in_llms.md)
- [\[AAAI 2026\] TAlignDiff: Automatic Tooth Alignment assisted by Diffusion-based Transformation Learning](taligndiff_automatic_tooth_alignment_assisted_by_diffusion-based_transformation_.md)
- [\[AAAI 2026\] Personality-guided Public-Private Domain Disentangled Hypergraph-Former Network for Multimodal Depression Detection](personality-guided_public-private_domain_disentangled_hypergraph-former_network_.md)
- [\[ICML 2026\] Scaling Vision Transformers for Functional MRI with Flat Maps](../../ICML2026/medical_imaging/scaling_vision_transformers_for_functional_mri_with_flat_maps.md)

</div>

<!-- RELATED:END -->
