---
title: >-
  [Paper Note] Unlocking the Potential of Diffusion Priors in Blind Face Restoration
description: >-
  [ICCV 2025][Image Generation][Blind face restoration] This paper proposes FLIPNET, a unified framework built upon a T2I diffusion model that switches between a restoration mode (BoostHub selectively fuses LQ features + BFR-oriented facial embeddings) and a degradation mode (learns from real degradation datasets and synthesizes degraded images) by simply flipping the inputs, simultaneously addressing two key challenges: the HQ/LQ distribution gap and the synthetic/real degradation gap.
tags:
  - ICCV 2025
  - Image Generation
  - Blind face restoration
  - diffusion priors
  - dual-mode network
  - degradation modeling
  - LoRA
date: 2026-05-08
content_hash: d0a4809a59d1743d
---

# Unlocking the Potential of Diffusion Priors in Blind Face Restoration

**Conference**: ICCV 2025
**arXiv**: [2508.08556](https://arxiv.org/abs/2508.08556)
**Code**: Not available
**Area**: Image Generation
**Keywords**: Blind face restoration, diffusion priors, dual-mode network, degradation modeling, LoRA

## TL;DR

This paper proposes FLIPNET, a unified framework built upon a T2I diffusion model that switches between a restoration mode (BoostHub selectively fuses LQ features + BFR-oriented facial embeddings) and a degradation mode (learns from real degradation datasets and synthesizes degraded images) by simply flipping the inputs, simultaneously addressing two key challenges: the HQ/LQ distribution gap and the synthetic/real degradation gap.

## Background & Motivation

**Two fundamental gaps in Blind Face Restoration (BFR)**:
- **HQ vs. LQ gap**: Diffusion models are trained on high-quality images, yet BFR must handle moderately to severely degraded inputs.
- **Synthetic vs. real degradation gap**: LQ training images are synthesized via simple degradation pipelines (blur + noise + compression), which fail to capture the complexity of real-world degradations.

**Limitations of Prior Work**:
- **Pre-processing + diffusion** (DiffBIR, PMRF): Pre-processing modules (e.g., SwinIR) remove degradations but also erase facial details such as wrinkles and freckles.
- **Conditional concatenation** (WaveFace): Directly concatenating LQ images as input suffers from limited degradation modeling, yielding poor real-world performance.
- **Real-ESRGAN degradation model**: Repeatedly applies classical degradations, resulting in limited diversity and mode coverage.

**Goal**: Address both gaps using a single T2I model with minimal trainable parameters, without relying on any pre-processing module.

## Method

### Overall Architecture (FLIPNET)

Built upon Stable Diffusion 2.1, the model takes (HQ, LQ) image pairs as input. **Flipping the inputs switches the operating mode**:
- **Restoration mode**: HQ as target, LQ as condition → learns to recover HQ from LQ.
- **Degradation mode**: LQ as target, HQ as condition → learns the HQ→LQ degradation mapping.

Only LoRA (rank=64) weights, BoostHub, and Adapter parameters are updated during training; the base model remains fully frozen.

### Key Design 1: BoostHub

Operates in parallel with the UNet self-attention layers and selectively fuses LQ features via cross-attention:

$$F_{ro} = W_O \cdot \text{Softmax}(Q_xK_y^T/\sqrt{d})V_y$$
$$F_{joint} = \text{Self-Attn}(F_x) + \phi \cdot F_{ro}$$

- $Q_x$ is derived from HQ features; $K_y, V_y$ are derived from LQ features.
- The boosting weight $\phi$ controls the degree of LQ feature injection: $\phi=0$ yields pure generation, $\phi=2$ makes LQ features dominant, and $\phi=1$ achieves the optimal balance.
- The output projection matrix $W_O$ is initialized to zero to prevent harmful interference at the start of training.

### Key Design 2: BFR-Oriented Facial Embeddings

Face recognition embeddings such as ArcFace are found to be ill-suited for BFR — recognition models are trained to discriminate identities rather than preserve fine details, and their attention concentrates on discriminative regions (e.g., eyes) only.

**Two-stage training**:
1. **Reconstruction stage**: HQ and LQ autoencoders are trained independently to reconstruct their inputs from a latent space.
   - Loss: $\mathcal{L}_{ae} = \mathcal{L}_1 + \lambda_{ap}\mathcal{L}_p + \lambda_{adv}\mathcal{L}_{adv}$
2. **Association stage**: Cross-entropy loss is used to align the HQ and LQ latent spaces.
   - A similarity matrix $\mathcal{M} \in \mathbb{R}^{N \times N}$ is computed over HQ/LQ patch features.
   - Diagonal similarity is maximized: $\mathcal{L}^{H(L)}_{ce} = -\frac{1}{N}\sum_i\sum_j y_{i,j}\log(z_{x(y)}^{i,j})$

The trained LQ encoder extracts facial embeddings $\mathbf{z}_y$, which are projected to the text token dimension via Adapter $\tau$ and integrated through cross-attention.

### Key Design 3: Degradation Mode

By flipping the inputs, the model learns degradation distributions from real degradation datasets (Dense-Haze, LOL, SIDD, RealBlur; ~2,500 pairs), generating diverse and realistic degraded images:
- Each HQ face image can produce 5 distinct degradation variants.
- Degradation severity is entirely unknown, consistent with the "blind" restoration setting.
- Only $\mathcal{L}_{ldm}$ is used during training (no image-level constraints, to avoid introducing noise artifacts).

### Loss & Training

Restoration mode: $\mathcal{L}_{rm} = \mathcal{L}_{ldm} + \lambda_{mse}\mathcal{L}_{mse} + \lambda_p\mathcal{L}_p$

where $\mathcal{L}_{mse}$ and $\mathcal{L}_p$ impose image-level constraints on the per-step denoising prediction $\hat{x}_0$.

## Key Experimental Results

### Quantitative Comparison on CelebA-Test

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | Deg.↓ |
|--------|-------|-------|--------|------|-------|
| DiffBIR | Baseline | - | - | - | - |
| WaveFace | Baseline | - | - | - | - |
| RestoreFormer++ | Baseline | - | - | - | - |
| **FLIPNET (+O)** | Improved | Improved | Improved | Improved | Improved |
| **FLIPNET (+O/F)** | **Best** | **Best** | **Best** | **Best** | **Best** |

### Degradation Modeling Comparison

| Method | Distribution Coverage |
|--------|----------------------|
| Real-ESRGAN | Clustered in limited regions; low diversity |
| **FLIPNET Degradation Mode** | Broadly distributed across LFW, WIDER, and WebPhoto real-world datasets |

### Key Findings

- BoostHub boosting weight $\phi=1$ achieves the optimal balance between PSNR and Identity Angle.
- BFR-oriented embeddings vs. ArcFace embeddings: the former exhibits more uniform attention distribution (covering the entire face rather than only the eyes), yielding better detail preservation.
- The association stage that aligns HQ/LQ embedding spaces is critical — omitting this step causes the model to over-rely on LQ features, degrading restoration quality.
- Combining online and offline degradation (from FLIPNET's degradation mode) yields the best training performance.
- The combination of Real-ESRGAN and FLIPNET-generated degradations provides broader degradation distribution coverage.

## Highlights & Insights

1. The **input-flipping mode-switching** design is particularly elegant — a single architecture simultaneously addresses restoration and degradation modeling.
2. The observation motivating **BFR-oriented embeddings** is insightful: the discriminative objective of recognition embeddings is fundamentally misaligned with the fidelity objective of restoration.
3. **No pre-processing module** is required: the fully end-to-end pipeline avoids the detail erasure caused by pre-processing stages.

## Limitations & Future Work

- The degradation mode relies on non-face datasets (dehazing, denoising), and the domain gap may affect the quality of synthesized facial degradations.
- The training pipeline involving LoRA + BoostHub + Adapter is relatively complex.
- Inference speed is constrained by the number of diffusion sampling steps.

## Related Work & Insights

- **Diffusion-prior BFR**: DiffBIR, PMRF, WaveFace, PGDiff
- **Generative-prior BFR**: GFP-GAN, GPEN, DAEFR
- **Degradation modeling**: Real-ESRGAN, BSRGAN

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Input-flipping dual-mode design + BFR-oriented embeddings
- Technical Depth: ⭐⭐⭐⭐⭐ — BoostHub + embedding alignment + multi-level degradation mode design
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated on both synthetic and multiple real-world datasets
- Value: ⭐⭐⭐⭐ — Strong performance on real-world degradation restoration

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BVINet: Unlocking Blind Video Inpainting with Zero Annotations](bvinet_unlocking_blind_video_inpainting_with_zero_annotations.md)
- [\[ICCV 2025\] FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning](feddifrc_unlocking_the_potential_of_text-to-image_diffusion_models_in_heterogene.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)
- [\[ICCV 2025\] Less-to-More Generalization: Unlocking More Controllability by In-Context Generation](less-to-more_generalization_unlocking_more_controllability_by_in-context_generat.md)

</div>

<!-- RELATED:END -->
