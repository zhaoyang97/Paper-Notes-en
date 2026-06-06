---
title: >-
  [Paper Note] Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Latent Diffusion Models] This paper identifies that β-VAE tokenizers in latent diffusion models suffer from variance collapse…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Latent Diffusion Models"
  - "Variance Expansion Loss"
  - "Sampling Robustness"
  - "Variance Collapse"
  - "VAE Tokenizer"
date: 2026-05-08
content_hash: 8805530752b4a3a6
---

# Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2603.21085](https://arxiv.org/abs/2603.21085)  
**Code**: [https://github.com/CVL-UESTC/VE-Loss](https://github.com/CVL-UESTC/VE-Loss)  
**Area**: Diffusion Models
**Keywords**: Latent Diffusion Models, Variance Expansion Loss, Sampling Robustness, Variance Collapse, VAE Tokenizer

## TL;DR
This paper identifies that β-VAE tokenizers in latent diffusion models suffer from variance collapse, producing an overly compact latent space that is highly sensitive to diffusion sampling perturbations. The proposed Variance Expansion (VE) Loss achieves adaptive latent variance regulation through an adversarial balance between reconstruction and variance expansion objectives, consistently improving generation quality (FID 1.18) across multiple diffusion architectures.

## Background & Motivation

1. **Background**: Latent diffusion models (LDMs) have become the dominant paradigm for high-quality image generation. The core pipeline encodes images into a latent space via an autoencoder (typically a β-VAE), followed by training a diffusion or flow model in that space. Recent works such as VA-VAE, MAETok, and DC-AE 1.5 improve the semantic structure of the latent space by aligning semantic priors or incorporating self-supervised objectives.
2. **Limitations of Prior Work**: Beyond reconstruction fidelity and semantic alignment, a critical yet overlooked factor is the robustness of the latent space to diffusion sampling perturbations. A counterintuitive phenomenon is observed: tokenizers with better reconstruction and lower diffusion loss can yield worse generation quality. This is because the KL term weight in standard β-VAE is extremely small (e.g., $10^{-6}$), driving the latent variance $\sigma^2$ toward zero and producing an overly compact latent manifold. Under such conditions, minor stochastic perturbations during sampling can displace samples outside the manifold, causing decoding failures.
3. **Key Challenge**: The reconstruction loss inherently drives variance collapse ($\partial \mathcal{L}_{rec}/\partial \sigma \approx 2\sigma T(\mu)$, always pushing $\sigma$ toward zero), while the stochastic nature of diffusion sampling demands sufficient latent space robustness to accommodate perturbations. Although traditional KL regularization can increase variance, it severely damages reconstruction quality by enforcing alignment with a standard Gaussian prior.
4. **Goal**: To construct a latent space that simultaneously maintains high reconstruction fidelity and robustness against diffusion sampling perturbations.
5. **Key Insight**: The gradient of the reconstruction loss with respect to variance is analyzed via first-order Taylor expansion to derive a theoretical mechanism for variance collapse, which then motivates the design of a counter-gradient to oppose the collapse.
6. **Core Idea**: VE Loss ($\mathcal{L}_{var} = 1/(\sigma^2+\delta)$) provides a strong counter-gradient to oppose the variance-collapsing tendency of the reconstruction loss, achieving adaptive variance equilibrium through their natural adversarial interaction.

## Method

### Overall Architecture
In standard VAE tokenizer training, VE Loss replaces the KL divergence term. The encoder outputs a Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$; samples drawn via reparameterization are passed to the decoder for reconstruction. The training objective is: reconstruction loss + $\lambda_1 \cdot$ variance expansion loss + $\lambda_2 \cdot$ regularization loss. The trained tokenizer is compatible with arbitrary diffusion models (DiT/LightningDiT/SiT).

### Key Designs

1. **Theoretical Analysis of Variance Collapse**:

    - **Function**: Provides a theoretical explanation for why the latent space of standard β-VAE is unsuitable for diffusion sampling.
    - **Mechanism**: A first-order Taylor expansion of the decoder gives $\mathcal{D}(\mu+\sigma\epsilon) \approx \mathcal{D}(\mu) + J(\mu)\sigma\epsilon$. Substituting into the expected reconstruction error yields $\mathcal{L}_{rec}(\mu,\sigma) \approx \|\mathbf{X}_0 - \mathcal{D}(\mu)\|^2 + \sigma^2 \cdot \text{Tr}(J(\mu)J(\mu)^\top)$. The gradient of the reconstruction loss with respect to $\sigma$ is $2\sigma T(\mu)$, which always pushes $\sigma$ toward zero. When the KL weight is extremely small, $\sigma^2$ approaches $10^{-8}$, causing the latent manifold to degenerate into an extremely thin "needle-like" structure where diffusion sampling perturbations easily exceed the manifold boundary.
    - **Design Motivation**: Provides a first-principles explanation for variance collapse and establishes the theoretical foundation for VE Loss design.

2. **Variance Expansion (VE) Loss**:

    - **Function**: Counteracts variance collapse and maintains a healthy latent space variance.
    - **Mechanism**: The inverse-variance loss $\mathcal{L}_{var}(\sigma) = 1/(\sigma^2 + \delta)$ has gradient $-2\lambda/\sigma^3$ with respect to $\sigma$, providing a strong counter-push when $\sigma$ is small. Balancing against the reconstruction gradient yields an equilibrium $\sigma = (\lambda/T(\mu))^{1/4}$, meaning variance adapts inversely to the fourth root of the decoder's local sensitivity $T(\mu)$. Three candidate formulations are compared: (i) negative variance $-\alpha\sigma^2$: gradient vanishes as $\sigma \to 0$, offering no protection; (ii) log entropy $\log\sigma^2$: equilibrium $\sigma^2 \propto 1/T(\mu)$ is theoretically sound but provides insufficient protection in the small-$\sigma$ regime; (iii) inverse variance $1/(\sigma^2+\delta)$ (selected): provides the strongest protection as $\sigma \to 0$.
    - **Design Motivation**: Unlike KL regularization, which enforces alignment with a fixed Gaussian prior and degrades reconstruction, VE Loss only prevents variance from becoming too small. The natural adversarial interaction with the reconstruction loss achieves adaptive equilibrium—maintaining small variance in decoder-sensitive regions (for precise reconstruction) while permitting larger variance in insensitive regions (for enhanced robustness).

3. **Regularization Term**:

    - **Function**: Prevents latent variable magnitudes from growing excessively due to variance expansion.
    - **Mechanism**: $\mathcal{L}_{reg} = e^{|z|-\tau}$ imposes exponential penalty when $|z|$ exceeds threshold $\tau$, with $\tau=1$ and $\lambda_2=10^{-6}$.
    - **Design Motivation**: VE Loss increases variance and may cause the absolute values of latent variables to grow; a mild constraint is needed to keep them within a reasonable range.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{var} + \lambda_2 \mathcal{L}_{reg}$
- $\lambda_1 = 0.1$, $\lambda_2 = 10^{-6}$, $\tau = 1$
- Tokenizer architecture and training strategy follow VA-VAE with a downsampling factor of 16
- Ablation tokenizers are trained for 16 epochs; the SOTA variant fine-tunes VA-VAE for 5 epochs
- Diffusion models use DiT/LightningDiT/SiT with flow matching objective
- SOTA configuration: LightningDiT-XL (675M) + Muon optimizer (to address training instability from DINOv2 alignment)

## Key Experimental Results

### Main Results — SOTA Comparison (ImageNet 256×256, CFG)

| Method | Params | Training Epochs | gFID↓ | IS↑ | Recall↑ |
|--------|--------|-----------------|-------|-----|---------|
| DiT | 675M | 1400 | 2.27 | 278.2 | 0.57 |
| MDTv2 | 675M | 1080 | 1.58 | 314.7 | 0.65 |
| REPA | 675M | 800 | 1.42 | 305.7 | 0.65 |
| VA-VAE | 675M | 800 | 1.35 | 295.3 | 0.65 |
| RAE | 675M | 800 | 1.41 | 309.4 | 0.63 |
| **VE Loss (Ours)** | **675M** | **530** | **1.18** | 289.8 | **0.66** |

### Ablation Study — VE Loss Effect Across Different Tokenizers

| Tokenizer | Training Epochs | rFID↓ | PSNR↑ | FID-10K (DiT-B)↓ | FID-10K (LightningDiT-B)↓ |
|-----------|-----------------|-------|-------|-------------------|--------------------------|
| LDM | 10 | 0.55 | 26.05 | 31.93 | 22.25 |
| LDM + VE | 10 | 0.60 | 25.23 | **29.03** | **19.70** |
| VAVAE | 16 | 0.35 | 27.43 | 22.27 | 19.85 |
| VAVAE + VE | 16 | 0.45 | 26.54 | **19.42** | **15.50** |

### Limitations of KL Regularization

| KL weight β | Latent variance $\sigma^2$ | rFID↓ | PSNR↑ | FID-10K↓ |
|-------------|---------------------------|-------|-------|----------|
| $10^{-6}$ | $10^{-8}$ | 0.39 | 27.12 | 23.12 |
| $10^{-2}$ | $10^{-5}$ | 0.44 | 26.71 | 22.87 |
| 1 | 0.07 | 0.61 | 25.45 | 23.18 |
| 8 | 0.94 | 2.36 | 22.29 | 27.54 |
| **VE Loss** | **0.06** | **0.46** | **26.31** | **18.90** |

### Key Findings
- **Consistent improvement from VE Loss**: Significant FID reductions are observed on both the vanilla LDM and VA-VAE tokenizers, indicating that VE Loss is a universally applicable latent space improvement rather than a tokenizer-specific technique.
- **Dilemma of KL regularization**: Increasing the KL weight raises variance but severely degrades reconstruction (rFID increases from 0.39 to 2.36 at β=8), ultimately worsening FID. VE Loss achieves the best FID of 18.90 at a moderate variance of $\sigma^2=0.06$.
- **Training efficiency**: The SOTA configuration surpasses VA-VAE using only 530 epochs compared to 800 (FID 1.18 vs. 1.35), with the tokenizer fine-tuned for only 5 epochs rather than trained from scratch for 50 epochs.
- **Effectiveness of fine-tuning**: Applying 10 epochs of VE Loss fine-tuning to an already-trained VA-VAE improves both reconstruction and generation (rFID 0.28→0.26, PSNR 27.71→28.31).

## Highlights & Insights
- **A neglected third dimension**: Beyond reconstruction fidelity and semantic alignment, "sampling robustness" constitutes a third critical dimension in latent space design. This finding carries significant implications for the broader LDM community—a good tokenizer must not only reconstruct accurately and encode rich semantics, but also tolerate perturbations.
- **Adversarial adaptive equilibrium**: VE Loss and the reconstruction loss naturally form an adversarial pair, eliminating the need for manual per-location variance tuning—decoder-sensitive regions automatically maintain small variance while insensitive regions automatically expand it. This approach of leveraging the inherent conflict between loss terms for adaptive regulation is particularly elegant.
- **Complete and instructive theoretical analysis**: From the first-order analysis of variance collapse to the derivation of the equilibrium point, every step carries clear physical intuition. The comparison of three candidate VE formulations is also highly instructive.

## Limitations & Future Work
- Validation is currently limited to ImageNet 256×256; effectiveness at higher resolutions (512/1024) and in video generation remains unknown.
- VE Loss introduces three hyperparameters ($\lambda_1$, $\lambda_2$, $\tau$) which, while relatively stable, lack an automatic tuning mechanism.
- The theoretical analysis relies on a first-order Taylor expansion (valid when $\sigma$ is small); the influence of higher-order terms when $\sigma$ is substantially increased is not discussed.
- The combination of VE Loss with tokenizers using self-supervised objectives, such as MAETok and DC-AE 1.5, has not been explored.

## Related Work & Insights
- **vs. KL regularization**: KL enforces alignment with a fixed Gaussian prior, imposing a rigid reconstruction-variance trade-off; VE Loss only prevents variance collapse and achieves a flexible adaptive balance through natural adversarial interaction.
- **vs. RAE (σ-VAE)**: RAE enhances robustness by injecting noise with fixed variance, but this fixed variance is global and non-adaptive; VE Loss learns variance that varies with decoder sensitivity.
- **vs. GIVT**: GIVT increases variance by raising the KL weight, but remains constrained by the reconstruction-penalizing nature of KL; VE Loss operates outside the KL framework, offering greater flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐ — Identifies "sampling robustness" as an overlooked dimension; VE Loss design is theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations across multiple tokenizers and diffusion architectures; comprehensive SOTA comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical analysis is clear and elegant; toy example visualizations are intuitive and compelling.
- Value: ⭐⭐⭐⭐⭐ — Achieves SOTA FID of 1.18 with a simple and general method; significant implications for the LDM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)
- [\[CVPR 2026\] Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)
- [\[ICML 2026\] Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems](../../ICML2026/image_generation/zeroth-order_non-log-concave_sampling_with_variance_reduction_and_applications_t.md)
- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)

</div>

<!-- RELATED:END -->
