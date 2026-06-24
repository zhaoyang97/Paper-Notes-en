---
title: >-
  [Paper Note] Improving the Diffusability of Autoencoders
description: >-
  [ICML 2025][Image Generation][latent diffusion] Through 2D DCT spectral analysis, this study reveals excessively strong high-frequency components in the latent space of autoencoders that do not match the RGB space. A Scale Equivariance regularization is proposed to align the frequency distributions of both. Finetuning for only 10-20K steps reduces ImageNet FID by 19% and Kinetics FVD by over 44%.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "latent diffusion"
  - "autoencoder"
  - "spectral analysis"
  - "diffusability"
  - "scale equivariance"
  - "VAE"
date: 2026-05-08
content_hash: ff5b9f7f89444af9
---

# Improving the Diffusability of Autoencoders

**Conference**: ICML 2025  
**arXiv**: [2502.14831](https://arxiv.org/abs/2502.14831)  
**Code**: [GitHub](https://github.com/snap-research/diffusability)  
**Area**: Image Generation  
**Keywords**: latent diffusion, autoencoder, spectral analysis, diffusability, scale equivariance, VAE

## TL;DR

Through 2D DCT spectral analysis, this study reveals excessively strong high-frequency components in the latent space of autoencoders that do not match the RGB space. A Scale Equivariance regularization is proposed to align the frequency distributions of both. Finetuning for only 10-20K steps reduces ImageNet FID by 19% and Kinetics FVD by over 44%.

## Background & Motivation

**Background**: Latent Diffusion Models (LDMs) consist of two components: an autoencoder (AE) and a diffusion backbone. Recent breakthroughs primarily stem from scaling up the diffusion backbone and improving the reconstruction quality or compression ratio of the AE.

**Limitations of Prior Work**: The interaction between the AE and the diffusion backbone is heavily ignored. The effectiveness of an AE should be determined by three factors: reconstruction quality, compression efficiency, and **Diffusability** (how easily the latent space can be modeled by diffusion models). This third dimension is virtually unexplored.

**Key Challenge**: Diffusion models possess a natural coarse-to-fine characteristic—generating low frequencies first, followed by high frequencies. The decaying power spectrum of natural images naturally supports this implicit spectral autoregression. However, the latent space spectrum of an AE is flatter with pronounced high frequencies, which breaks this spectral autoregression and forces diffusion models to spend extra capacity modeling unnecessary high frequencies.

**Goal**: (1) Diagnose the spectral issues in the AE latent space; (2) Propose a low-cost solution to improve diffusability.

**Key Insight**: Systematically analyzing the frequency distribution of different AE latent spaces using 2D DCT reveals that the high-frequency issue escalates with an increasing number of channels, and KL regularization actually worsens the spectrum.

**Core Idea**: By enforcing scale equivariance on the decoder (decoding downsampled latents $\approx$ downsampling decoded results), the latent space is aligned with the RGB space across different frequencies.

## Method

### Overall Architecture

Two steps: (1) Block-wise 2D DCT is used to analyze the frequency distribution of the AE latent space versus RGB to locate spectral mismatches; (2) Scale Equivariance (SE) regularization is added to finetune the AE decoder, constraining the decoded downsampled latent to be consistent with the downsampled original decoded result. This requires only 10-20K steps and minimal code changes.

### Key Designs

1. **DCT Spectral Analysis Diagnosis**:

    - **Function**: Quantitatively reveal the frequency pathologies of the AE latent space.
    - **Mechanism**: Segment the latent into $B \times B$ blocks to perform 2D DCT, calculate the frequency distribution of the normalized magnitude $A_{uv} = |D_{uv}/D_{0,0}|$, and sort them in a zigzag order to obtain the frequency curve.
    - **Key Findings**: The RGB spectrum exhibits a clear decay; the AE latent spectrum is flat with prominent high frequencies; **more channels lead to stronger high frequencies** (larger bottlenecks encode more high-frequency information but in an uncontrolled distribution); **KL regularization unexpectedly increases high frequencies** (noise injection in VAEs introduces uniform high-frequency energy).
    - **Design Motivation**: Accurately diagnose "what went wrong" first.

2. **Scale Equivariance Regularization (SE)**:

    - **Function**: Align the frequency correspondence between the latent space and the RGB space.
    - **Mechanism**: The core constraint is $\text{Dec}(\text{downsample}_s(z)) \approx \text{downsample}_s(\text{Dec}(z))$—decoding a downsampled latent (retaining low frequencies) should be approximately equal to downsampling the decoded result (low-frequency RGB). This forces low-frequency latent $\to$ low-frequency RGB, and high-frequency latent $\to$ high-frequency RGB. MSE constraints are applied across multiple downsampling scales $s$.
    - **Design Motivation**: Diffusion models rely on the spectral autoregression of "generating low frequencies first"—if low-frequency latents do not correspond to low-frequency RGB, early-stage generations of diffusion may manifest as high-frequency artifacts in the RGB space.

3. **The Double-Edged Sword of KL Regularization**:

    - **Function**: Explain why KL regularization in existing VAEs is insufficient.
    - **Mechanism**: KL encourages latents to approach a standard Gaussian prior (reducing the workload of diffusion), but the reparameterization trick injects white noise with a flat spectrum, thereby enhancing high frequencies. The conflict between the benefit (prior matching) and the drawback (high-frequency injection) is particularly prominent in AEs with larger channel counts.
    - **Design Motivation**: SE coverage offsets the blind spot of KL—KL governs the distribution shape, while SE governs frequency alignment.

### Loss & Training

Total loss = Original reconstruction loss + $\lambda$ $\times$ SE constraint loss. Only the AE decoder is finetuned for 10-20K steps.

## Key Experimental Results

### Image Generation: ImageNet-1K $256^2$ (DiT-XL/2)

| Autoencoder | FID↓ | Improvement |
|---------|------|------|
| FluxAE (Original) | 3.07 | — |
| FluxAE + SE (10K-step finetuning) | 2.49 | **19%↓** |

### Video Generation: Kinetics-700 $17\times256^2$

| Autoencoder | Improvement |
|---------|---------|
| CogVideoX-AE + SE | **FVD ≥44%↓** |
| LTX-AE + SE | Significant ↓ |
| CosmosTokenizer + SE | Significant ↓ |

### Spectral Alignment Verification

| Configuration | Spectral Characteristics |
|------|---------|
| RGB | Strong decay, low-frequency dominated |
| FluxAE Original | Flat, prominent high frequencies |
| FluxAE + SE (Weak $\lambda$) | High frequencies somewhat suppressed |
| FluxAE + SE (Strong $\lambda$) | Significantly closer to the RGB distribution |

### Key Findings

- After SE finetuning, DiT achieves not only a better final FID but also significantly faster convergence—spectral alignment eases the difficulty of diffusion learning.
- Consistent improvements across 4 AEs and 2 tasks demonstrate that spectral mismatch is a universal issue in AEs.
- AEs with larger channel counts benefit more (as their high-frequency issue is more severe).
- SE has almost no impact on the reconstruction quality of the AE (with minimal change in PSNR/SSIM), incurring negligible cost.
- Significant improvements can be achieved by only finetuning the AE without retraining the diffusion backbone.

## Highlights & Insights

- The concept of "Diffusability" fills the gap as the third dimension of AE design—a crucial factor alongside reconstruction quality and compression efficiency.
- The 2D DCT spectral analysis methodology is reusable—suitable for diagnosing the frequency domain behavior of any LDM component.
- The solution is extremely simple and elegant: a single downsampling consistency constraint, 10K steps of finetuning, less than 10 lines of code changes, and universal improvements across various architectures and tasks.
- The discovery of the double-edged sword effect of KL regularization holds independent value—challenging the intuition that "stronger KL brings better latents."
- It offers a friendly upgrade path for deployed LDMs—only the AE decoder is finetuned, accounting for approximately 0.1% of the total training cost.

## Limitations & Future Work

- Lack of deep theoretical analysis regarding why SE works; the causal mechanism relies heavily on hypotheses and experimental validation.
- Lack of systematic guidance for choosing the SE regularization strength $\lambda$.
- Only continuous AEs are analyzed; the applicability to discrete AEs like VQ-VAE remains unexplored.
- Direct comparison with EQ-VAE (contemporaneous work) is missing.
- DiT-XL/2 performance on ImageNet $256^2$ is close to saturation in the experiments; larger-scale verification would be more convincing.

## Related Work & Insights

- **vs EQ-VAE (Kouzelis et al., 2025)**: A contemporaneous independent work that proposes scale/shift equivariance, but the motivation lies in spatial transformation equivariance. The current paper starts from spectral analysis, providing a deeper analysis.
- **vs AF-LDM (Zhou et al., 2025)**: Another contemporaneous work enforcing shift equivariance in both the AE and LDM. The current paper focuses specifically on scale equivariance on the AE side.
- **vs Rissanen et al. (2023)**: Explored implicit spectral autoregression in diffusion models for the first time. The current work instantiates this insight into an actionable AE improvement scheme.
- **Insight**: AE design should treat diffusability as a first-class citizen; dedicated architectures optimized specifically for this might emerge in the future.

## Rating

- Novelty: ⭐⭐⭐⭐ Diagnosing the AE-Diffusion interaction from a spectral perspective is an original contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic validation across 4 AEs and 2 tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain leading from diagnosis to cause, solution, and validation.
- Value: ⭐⭐⭐⭐⭐ Brings significant improvements at extremely low costs, with the concept of diffusability promising a long-lasting impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Diffusion Transformers with Representation Autoencoders](../../ICLR2026/image_generation/diffusion_transformers_with_representation_autoencoders.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](../../ICCV2025/image_generation/latent_diffusion_models_with_masked_autoencoders.md)
- [\[CVPR 2026\] MeanFlow Transformers with Representation Autoencoders](../../CVPR2026/image_generation/meanflow_transformers_with_representation_autoencoders.md)
- [\[NeurIPS 2025\] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](../../NeurIPS2025/image_generation/learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)
- [\[NeurIPS 2025\] Exploring Variational Graph Autoencoders for Distribution Grid Data Generation](../../NeurIPS2025/image_generation/exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)

</div>

<!-- RELATED:END -->
