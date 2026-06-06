---
title: >-
  [Paper Note] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers
description: >-
  [ICCV 2025][Image Generation][End-to-end training] This paper proposes REPA-E, which enables joint end-to-end training of VAE and latent diffusion Transformers via representation alignment (REPA) loss…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "End-to-end training"
  - "VAE"
  - "latent diffusion models"
  - "representation alignment"
  - "training acceleration"
date: 2026-05-08
content_hash: a91d50be0c87dd72
---

# REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers

**Conference**: ICCV 2025
**arXiv**: [2504.10483](https://arxiv.org/abs/2504.10483)  
**Code**: [https://end2end-diffusion.github.io](https://end2end-diffusion.github.io)  
**Area**: Image Generation
**Keywords**: End-to-end training, VAE, latent diffusion models, representation alignment, training acceleration

## TL;DR

This paper proposes REPA-E, which enables joint end-to-end training of VAE and latent diffusion Transformers via representation alignment (REPA) loss, achieving 17× and 45× training speedups over REPA and vanilla training respectively, and setting a new state of the art of FID 1.12 on ImageNet 256×256.

## Background & Motivation

### Two-Stage Training Paradigm of Latent Diffusion Models

The standard training pipeline for latent diffusion models (LDMs) consists of two fully decoupled stages: the first stage trains a VAE (variational autoencoder) with reconstruction loss to compress images into a latent space; the second stage freezes the VAE and trains the diffusion model in that latent space. This decoupling raises a fundamental question: **how can one ensure that the VAE representations learned in the first stage are optimal for the generative performance of the second stage?**

### Limitations of Prior Work

**VAE latent spaces may be suboptimal for diffusion models**: Prior work has found that mainstream VAEs (e.g., SD-VAE) exhibit high-frequency noise components in their latent spaces, while others (e.g., f16d32 VAE trained on ImageNet) suffer from over-smoothing. Neither constitutes a generation-optimal representation.

**Difficulty of empirical tuning**: The optimal compatibility between a VAE and a diffusion model depends on the architectures and training configurations of both components, making it infeasible to resolve through one-time empirical analysis.

**Naive end-to-end training is ineffective**: Directly backpropagating the diffusion loss through the VAE causes latent space collapse — the diffusion loss drives the VAE to learn a simpler latent structure (reduced variance along spatial dimensions), which lowers denoising difficulty but degrades generation quality.

### Core Insight and Starting Point

The authors identify three key observations: (1) naive end-to-end diffusion loss "hacks" the latent space, making denoising easier at the cost of generation quality; (2) higher representation alignment scores (CKNNA) correlate positively with better generation performance, serving as a proxy metric; (3) the maximum achievable alignment score under standard REPA is bottlenecked by the VAE features. This leads to the central conclusion: **using representation alignment loss rather than diffusion loss for end-to-end training can simultaneously improve both the VAE and the diffusion model**.

## Method

### Overall Architecture

REPA-E extends standard REPA by unlocking backpropagation into the VAE encoder. Training jointly updates three sets of parameters: the VAE encoder $\mathcal{V}_\phi$, the diffusion model $\mathcal{D}_\theta$, and the REPA projection layer $h_\omega$. The total loss is:

$$\mathcal{L}(\theta, \phi, \omega) = \mathcal{L}_{\text{DIFF}}(\theta) + \lambda \mathcal{L}_{\text{REPA}}(\theta, \phi, \omega) + \eta \mathcal{L}_{\text{REG}}(\phi)$$

The diffusion loss updates only the diffusion model parameters (via stop-gradient), the REPA loss updates all three parameter sets, and the regularization loss updates only the VAE.

### Key Designs

1. **Batch-Norm Layer for Latent Space Normalization**:

    - Function: A Batch-Norm layer is inserted between the VAE and the diffusion model.
    - Mechanism: Standard LDM training normalizes VAE outputs using precomputed dataset statistics (e.g., std = 1/0.1825 for SD-VAE). During end-to-end training, the VAE is continuously updated, making it prohibitively expensive to recompute global statistics from scratch. The exponential moving average maintained by Batch-Norm serves as a proxy for global statistics, enabling differentiable normalization without repeated recomputation.
    - Design Motivation: The affine transformation in BN is disabled (no learnable scale/bias), using only running mean and std to ensure normalization purity.

2. **End-to-End Representation Alignment Loss**:

    - Function: The REPA loss is backpropagated into the VAE encoder.
    - Mechanism: Features from a pretrained vision model (e.g., DINOv2) serve as targets for aligning the hidden states of intermediate layers in the diffusion Transformer:
    $\mathcal{L}_{\text{REPA}}(\theta, \phi, \omega) = -\mathbb{E}_{\mathbf{x}, \epsilon, t}\left[\frac{1}{N}\sum_{n=1}^{N}\text{sim}(\mathbf{y}^{[n]}, h_\omega(\mathbf{h}_t^{[n]}))\right]$
      where $\mathbf{y} = f(\mathbf{x})$ denotes DINOv2 features and $\mathbf{h}_t$ denotes the hidden states at the 8th layer of the diffusion Transformer.
    - Design Motivation: Standard REPA freezes the VAE, causing the maximum achievable CKNNA score to saturate at approximately 0.42. Backpropagating the REPA loss into the VAE breaks this bottleneck, as the VAE can actively restructure its latent space to better support alignment.

3. **Stop-Gradient on Diffusion Loss**:

    - Function: The diffusion loss $\mathcal{L}_{\text{DIFF}}$ updates only the diffusion model parameters $\theta$ and does not propagate gradients to the VAE.
    - Mechanism: A stop-gradient operator is inserted at the VAE output, preventing diffusion loss gradients from flowing into the VAE.
    - Design Motivation: Experiments demonstrate that directly backpropagating the diffusion loss causes the VAE to learn a simpler but inferior latent space (reduced spatial variance, easier denoising, but degraded generation quality). Without stop-gradient, gFID degrades from 16.3 to 444.1.

4. **VAE Regularization Loss**:

    - Function: Prevents end-to-end training from impairing the VAE's reconstruction capability.
    - Mechanism: $\mathcal{L}_{\text{REG}} = \mathcal{L}_{\text{KL}} + \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{LPIPS}} + \mathcal{L}_{\text{GAN}}$
    - Design Motivation: Without regularization constraints, the VAE may overfit to the diffusion model and lose its reconstruction ability.

### Loss & Training

- Optimizer: AdamW with a fixed learning rate of $1 \times 10^{-4}$ and global batch size 256.
- Different REPA loss coefficients are applied to the diffusion model and the VAE: $\lambda_{\text{REPA}_g} = 0.5$, $\lambda_{\text{REPA}_v} = 1.5$.
- Gradient clipping and EMA are applied to the diffusion model.
- Training is conducted on 8 × NVIDIA H100 GPUs.

## Key Experimental Results

### Main Results

Generation performance on ImageNet 256×256 without CFG (SiT-XL + SD-VAE):

| Method | Epochs | gFID↓ | sFID↓ | IS↑ |
|--------|--------|-------|-------|-----|
| DiT | 1400 | 9.62 | 6.85 | 121.5 |
| SiT | 1400 | 8.61 | 6.32 | 131.7 |
| MaskDiT | 1600 | 5.69 | 10.34 | 177.9 |
| REPA | 20 | 19.40 | 6.06 | 67.4 |
| REPA | 80 | 7.90 | 5.06 | 122.6 |
| REPA | 800 | 5.90 | 5.73 | 157.8 |
| **REPA-E** | **20** | **12.83** | **5.04** | **88.8** |
| **REPA-E** | **80** | **4.07** | **4.60** | **161.8** |

REPA-E with only 80 epochs surpasses REPA's final result at 800 epochs (4.07 vs. 5.90), achieving more than 17× training speedup.

### Ablation Study

| Configuration | gFID↓ | sFID↓ | IS↑ | Note |
|---------------|-------|-------|-----|------|
| w/o stop-grad | 444.1 | 460.3 | 1.49 | Diffusion loss backpropagated to VAE → latent collapse |
| w/o batch-norm | 18.1 | 5.32 | 72.4 | No BN, normalization non-adaptive |
| w/o $\mathcal{L}_{\text{GAN}}$ | 19.2 | 6.47 | 68.2 | GAN regularization removed |
| **REPA-E (full)** | **16.3** | **5.69** | **75.0** | All components enabled |
| REPA-E (scratch) 400K steps | 4.34 | 4.44 | 154.3 | VAE trained from scratch |
| REPA-E (VAE init.) 400K steps | 4.07 | 4.60 | 161.8 | VAE initialized from pretrained weights |

Cross-model scale validation (100K steps, no CFG):

| Diffusion Model | REPA gFID | +REPA-E gFID | Gain% |
|-----------------|-----------|--------------|-------|
| SiT-B (130M) | 49.5 | 34.8 | 29.6% |
| SiT-L (458M) | 24.1 | 16.3 | 32.3% |
| SiT-XL (675M) | 19.4 | 12.8 | 34.0% |

### Key Findings

- Stop-gradient is critical to the method's success: removing it causes gFID to deteriorate from 16.3 to 444.1.
- The benefit of REPA-E increases with model scale (29.6% → 34.0%), demonstrating favorable scaling behavior.
- End-to-end training automatically improves the VAE's latent structure: high-frequency noise in SD-VAE is smoothed, and the over-smoothed IN-VAE learns finer details.
- The end-to-end fine-tuned VAE can directly replace the original VAE in other training settings, consistently improving generation performance.
- Even when the VAE is trained from scratch (without pretrained weights), REPA-E still substantially outperforms standard REPA.

## Highlights & Insights

- The paper addresses a seemingly straightforward yet long-overlooked question: why are LDMs not trained end-to-end? The authors precisely diagnose the root cause of latent space collapse under naive end-to-end training (the diffusion loss hacks the denoising objective).
- Using CKNNA as a proxy metric for generation performance is an elegant design choice, providing a path toward end-to-end training that bypasses the diffusion loss entirely.
- The technique of using Batch-Norm to substitute global statistics is simple and practical, and is broadly applicable to other end-to-end training scenarios.
- The experimental design is highly comprehensive, validating across different model scales, VAE variants, alignment encoders, and alignment depths.

## Limitations & Future Work

- The method requires an additional pretrained vision model (DINOv2) for the REPA loss, introducing extra training dependencies and computational overhead.
- VAE regularization involves multiple auxiliary objectives including GAN loss, and hyperparameter tuning may be non-trivial.
- Validation is currently limited to ImageNet 256×256; extension to higher resolutions or text-conditional generation remains unexplored.
- End-to-end training requires maintaining both the VAE and diffusion model in memory simultaneously, increasing GPU memory pressure.

## Related Work & Insights

- REPA (Yu et al.) introduced representation alignment to accelerate diffusion training; the present work extends this to end-to-end training, representing a natural and high-value extension.
- LSGM explored joint training of score-based models but relied on variational lower bounds and entropy terms to prevent collapse, resulting in slower convergence than REPA-E.
- The end-to-end philosophy parallels the evolution of the RCNN family (from RCNN to Faster RCNN), where progressive integration of pipeline stages yields compounding improvements.
- Similar end-to-end training strategies are worth exploring for other two-stage frameworks, such as VQ-VAE combined with autoregressive models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](end-to-end_multi-modal_diffusion_mamba.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](../../NeurIPS2025/image_generation/lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](../../CVPR2026/image_generation/deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[ICML 2026\] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer](../../ICML2026/image_generation/end-to-end_autoregressive_image_generation_with_1d_semantic_tokenizer.md)

</div>

<!-- RELATED:END -->
