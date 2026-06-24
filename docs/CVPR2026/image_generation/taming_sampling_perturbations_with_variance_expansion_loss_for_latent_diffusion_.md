---
title: >-
  [Paper Note] Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Latent Diffusion Models] This paper reveals that the $\beta$-VAE tokenizer in Latent Diffusion Models (LDMs) suffers from an overly compact latent space due to variance collapse, making it highly sensitive to diffusion sampling perturbations. It proposes Variance Expansion (VE) Loss to adaptively learn a robust latent space variance through an adversarial balance between reconstruction and variance expansion, consistently improving generation qua…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Latent Diffusion Models"
  - "Variance Expansion Loss"
  - "Sampling Robustness"
  - "Variance Collapse"
  - "VAE tokenizer"
date: 2026-05-08
content_hash: 9a2fd98af1cfbf4e
---

# Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models

**Conference**: CVPR 2026  
**arXiv**: [2603.21085](https://arxiv.org/abs/2603.21085)  
**Code**: [https://github.com/CVL-UESTC/VE-Loss](https://github.com/CVL-UESTC/VE-Loss)  
**Area**: Diffusion Models  
**Keywords**: Latent Diffusion Models, Variance Expansion Loss, Sampling Robustness, Variance Collapse, VAE tokenizer

## TL;DR
This paper reveals that the $\beta$-VAE tokenizer in Latent Diffusion Models (LDMs) suffers from an overly compact latent space due to variance collapse, making it highly sensitive to diffusion sampling perturbations. It proposes Variance Expansion (VE) Loss to adaptively learn a robust latent space variance through an adversarial balance between reconstruction and variance expansion, consistently improving generation quality (FID 1.18) across multiple diffusion architectures.

## Background & Motivation

1. **Background**: Latent Diffusion Models (LDMs) have become the mainstream paradigm for high-quality image generation. Their core involves using an autoencoder (typically a $\beta$-VAE) to encode images into a latent space, where a diffusion/flow model is then trained. Recent works like VA-VAE, MAETok, and DC-AE 1.5 improve the semantic structure of the latent space by aligning semantic priors or introducing self-supervised objectives.
2. **Limitations of Prior Work**: Beyond reconstruction accuracy and semantic alignment, a critical but overlooked factor is the robustness of the latent space to diffusion sampling perturbations. A counter-intuitive phenomenon is observed: tokenizers with better reconstruction and lower diffusion loss can actually produce worse generation quality. This is because the KL term weight in standard $\beta$-VAEs is extremely small (e.g., $10^{-6}$), causing the latent space variance $\sigma^2$ to approach zero. This results in an overly compact latent manifold where even minor random perturbations during sampling can push samples outside the manifold, leading to decoding failure.
3. **Key Challenge**: Reconstruction loss naturally drives variance collapse ($\partial \mathcal{L}_{rec}/\partial \sigma \approx 2\sigma T(\mu)$, constantly pushing $\sigma$ toward 0), while the inherent randomness of the diffusion sampling process requires the latent space to have sufficient robustness to accommodate perturbations. Although traditional KL regularization can increase variance, it severely degrades reconstruction quality by forcing alignment with a standard Gaussian prior.
4. **Goal**: Construct a latent space that maintains high reconstruction fidelity while being robust to diffusion sampling perturbations.
5. **Key Insight**: Analyze the gradient of reconstruction loss with respect to variance via a first-order Taylor expansion to derive the theoretical mechanism of variance collapse, and then design a reverse gradient to counteract this collapse.
6. **Core Idea**: Use the strong reverse gradient of VE Loss ($\mathcal{L}_{var} = 1/(\sigma^2+\delta)$) to oppose the variance collapse trend of the reconstruction loss. Adaptive variance balance is achieved through the natural antagonism between the two.

## Method

### Overall Architecture
This paper addresses the "fragility" of the latent space: when $\beta$-VAE compresses images into the latent space, the variance is squeezed nearly to zero, causing samples to fall off the manifold and decoding to fail under sampling noise. The approach is lightweight—it replaces the original KL divergence term in standard VAE tokenizer training with a new Variance Expansion (VE) loss, keeping the rest unchanged. The encoder outputs a Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$, which is reparameterized and fed into the decoder for reconstruction. The training objective becomes "reconstruction loss + $\lambda_1 \cdot$ VE Loss + $\lambda_2 \cdot$ regularization." The trained tokenizer can be directly paired with any diffusion model (DiT / LightningDiT / SiT) without modifying the diffusion side.

### Key Designs

**1. Theoretical Analysis of Variance Collapse: Explaining why the latent space collapses into a "needle"**

To counteract a phenomenon, its origin must be clarified. The authors apply a first-order Taylor expansion to the decoder at $\mu$: $\mathcal{D}(\mu+\sigma\epsilon) \approx \mathcal{D}(\mu) + J(\mu)\sigma\epsilon$. Substituting this into the expected reconstruction error yields:

$$\mathcal{L}_{rec}(\mu,\sigma) \approx \|\mathbf{X}_0 - \mathcal{D}(\mu)\|^2 + \sigma^2 \cdot \text{Tr}\big(J(\mu)J(\mu)^\top\big)$$

The second term increases monotonically with $\sigma^2$. Consequently, the gradient of the reconstruction loss with respect to $\sigma$ is always $2\sigma T(\mu)$ (where $T(\mu)=\text{Tr}(JJ^\top)$ is the local sensitivity of the decoder at that point), which always pushes $\sigma$ toward zero. In standard $\beta$-VAEs, the KL weight is minimal (e.g., $10^{-6}$), providing no resistance. This causes $\sigma^2$ to collapse to the magnitude of $10^{-8}$, compressing the latent manifold into an extremely thin "needle-like" structure. While this is fine for reconstruction, the inherent random noise in the diffusion sampling process easily pushes samples beyond the boundaries of this "needle." This explains the counter-intuitive phenomenon where tokenizers with better reconstruction yield worse generation quality—the latent space is too "thin."

**2. Variance Expansion (VE) Loss: Using a strong reverse gradient to pull variance back from zero**

Since reconstruction loss naturally pushes $\sigma$ toward zero, the most direct solution is to add a reverse gradient that becomes stronger as $\sigma$ decreases. The authors design the inverse variance loss $\mathcal{L}_{var}(\sigma) = 1/(\sigma^2 + \delta)$, whose gradient with respect to $\sigma$ is $-2\lambda/\sigma^3$—the smaller $\sigma$ is, the stronger the push. At equilibrium between the two forces, the solution is $\sigma = (\lambda/T(\mu))^{1/4}$. That is, the variance adaptively scales inversely with the fourth root of the decoder's local sensitivity $T(\mu)$. In regions where the decoder is sensitive (large $T$), $\sigma$ automatically decreases to maintain reconstruction accuracy; in insensitive regions (small $T$), $\sigma$ automatically increases to resist perturbations. This is the fundamental difference from KL regularization, which forces every position to align with a fixed standard Gaussian prior, inevitably damaging reconstruction. VE Loss only ensures the variance is "not too small," establishing a flexible pointwise balance through natural competition with the reconstruction loss. The authors compared three candidates and chose the inverse variance because it provides the strongest protection as $\sigma \to 0$.

**3. Magnitude Regularization: Preventing latent variables from exploding while expanding variance**

Expanding the variance has a side effect: the absolute value of $z$ might also expand, disrupting the numerical range of the latent space. The authors add a mild magnitude constraint $\mathcal{L}_{reg} = e^{|z|-\tau}$, which applies exponential punishment only when $|z|$ exceeds a threshold $\tau$ (set to 1). Its weight $\lambda_2=10^{-6}$ is very small, serving as a safeguard at the boundaries without interfering with the primary interaction between VE Loss and reconstruction loss.

### Loss & Training
- Total Loss: $\mathcal{L} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{var} + \lambda_2 \mathcal{L}_{reg}$
- $\lambda_1 = 0.1$, $\lambda_2 = 10^{-6}$, $\tau = 1$
- Tokenizer architecture and training strategy follow VA-VAE with a downsampling factor of 16.
- Ablation study tokenizers were trained for 16 epochs; the SOTA version was fine-tuned for 5 epochs on top of VA-VAE.
- Diffusion models use DiT/LightningDiT/SiT with a flow matching objective.
- SOTA Configuration: LightningDiT-XL (675M) + Muon optimizer (to resolve training instability from DINOv2 alignment).

## Key Experimental Results

### Main Results - SOTA Comparison (ImageNet 256×256, CFG)

| Method | Params | Train Epochs | gFID↓ | IS↑ | Recall↑ |
|------|--------|-----------|-------|-----|---------|
| DiT | 675M | 1400 | 2.27 | 278.2 | 0.57 |
| MDTv2 | 675M | 1080 | 1.58 | 314.7 | 0.65 |
| REPA | 675M | 800 | 1.42 | 305.7 | 0.65 |
| VA-VAE | 675M | 800 | 1.35 | 295.3 | 0.65 |
| RAE | 675M | 800 | 1.41 | 309.4 | 0.63 |
| **Ours (VE Loss)** | **675M** | **530** | **1.18** | 289.8 | **0.66** |

### Ablation Study - VE Loss effect on different tokenizers

| Tokenizer | Train Epochs | rFID↓ | PSNR↑ | FID-10K (DiT-B)↓ | FID-10K (LightningDiT-B)↓ |
|-----------|-----------|-------|-------|-------------------|--------------------------|
| LDM | 10 | 0.55 | 26.05 | 31.93 | 22.25 |
| LDM + VE | 10 | 0.60 | 25.23 | **29.03** | **19.70** |
| VAVAE | 16 | 0.35 | 27.43 | 22.27 | 19.85 |
| VAVAE + VE | 16 | 0.45 | 26.54 | **19.42** | **15.50** |

### Limitations of KL Regularization

| KL Weight $\beta$ | Latent Variance $\sigma^2$ | rFID↓ | PSNR↑ | FID-10K↓ |
|---------|-----------------|-------|-------|----------|
| $10^{-6}$ | $10^{-8}$ | 0.39 | 27.12 | 23.12 |
| $10^{-2}$ | $10^{-5}$ | 0.44 | 26.71 | 22.87 |
| 1 | 0.07 | 0.61 | 25.45 | 23.18 |
| 8 | 0.94 | 2.36 | 22.29 | 27.54 |
| **VE Loss** | **0.06** | **0.46** | **26.31** | **18.90** |

### Key Findings
- **Consistency of VE Loss**: Significant FID reductions on both vanilla LDM and VA-VAE tokenizers indicate that this is a universal latent space improvement rather than a tokenizer-specific trick.
- **Dilemma of KL Regularization**: Increasing KL weight increases variance but severely damages reconstruction (rFID jumps from 0.39 to 2.36 at $\beta=8$), causing FID to actually rise. VE Loss achieves optimal performance (FID 18.90) with a moderate variance of $\sigma^2=0.06$.
- **Training Efficiency**: The SOTA configuration surpasses VA-VAE's 800 epochs in only 530 epochs (FID 1.18 vs 1.35). The tokenizer requires only 5 epochs of fine-tuning compared to 50 epochs of training from scratch.
- **Effectiveness of Fine-tuning**: Applying 10 epochs of VE Loss fine-tuning to a pre-trained VA-VAE improved both reconstruction and generation (rFID 0.28 $\to$ 0.26, PSNR 27.71 $\to$ 28.31).

## Highlights & Insights
- **The Overlooked Third Dimension**: Beyond reconstruction precision and semantic alignment, "sampling robustness" is the third key dimension of latent space design. This discovery provides significant insights for the LDM community—a good tokenizer must not only reconstruct well and have good semantics but also be "perturbation-resistant."
- **Adversarial Adaptive Balance**: VE Loss and reconstruction loss naturally form an adversarial relationship, eliminating the need for manual variance adjustment at each position. Sensitive regions of the decoder automatically maintain small variance, while insensitive regions expand it. This approach of using inherent loss conflicts to achieve adaptivity is elegant.
- **Complete and Guided Theoretical Analysis**: From the first-order analysis of variance collapse to the derivation of the equilibrium point, every step has clear physical intuition. The comparison of the three candidate VE forms is also very instructive.

## Limitations & Future Work
- Currently only validated on ImageNet 256×256; performance on higher resolutions (512/1024) and video generation is unknown.
- VE Loss introduces 3 hyperparameters ($\lambda_1$, $\lambda_2$, $\tau$). While they are relatively stable, an automatic adjustment mechanism is lacking.
- Theoretical analysis is based on first-order Taylor expansion (valid for small $\sigma$); the impact of higher-order terms when $\sigma$ is significantly expanded is not discussed.
- Integration with tokenizers using self-supervised objectives (e.g., MAETok, DC-AE 1.5) has not been explored.

## Related Work & Insights
- **vs KL Regularization**: KL forces alignment with a fixed Gaussian prior, leading to a rigid reconstruction-variance trade-off; VE Loss only prevents variance collapse, achieving a flexible adaptive balance through natural competition.
- **vs RAE ($\sigma$-VAE)**: RAE enhances robustness by injecting noise with a fixed variance, but this fixed variance is global and non-adaptive; the variance learned by VE Loss varies with decoder sensitivity.
- **vs GIVT**: GIVT increases variance by increasing KL weight but is limited by the reconstruction penalty inherent in KL; VE Loss moves beyond the KL framework and is more flexible.

## Rating
- Novelty: ⭐⭐⭐⭐ Identifies the overlooked dimension of "sampling robustness"; VE Loss design has a solid theoretical foundation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive ablation across various tokenizers and diffusion architectures; strong SOTA comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear and elegant theoretical analysis; visualization of toy examples is intuitive and effective.
- Value: ⭐⭐⭐⭐⭐ Refreshing SOTA with FID 1.18; methods are simple and universal, offering major insights for the LDM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] VFM-VAE: Vision Foundation Models Can Be Good Tokenizers for Latent Diffusion Models](vfm-vae_vision_foundation_models_can_be_good_tokenizers_for_latent_diffusion_mod.md)
- [\[CVPR 2026\] IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)
- [\[CVPR 2026\] Your Latent Mask is Wrong: Pixel-Equivalent Latent Compositing for Diffusion Models](your_latent_mask_is_wrong_pixel-equivalent_latent_compositing_for_diffusion_mode.md)
- [\[CVPR 2026\] Taming Generative Diffusion Model for Task-Oriented Infrared Imaging](taming_generative_diffusion_model_for_task-oriented_infrared_imaging.md)

</div>

<!-- RELATED:END -->
