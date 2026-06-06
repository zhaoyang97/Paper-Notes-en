---
title: >-
  [Paper Note] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization
description: >-
  [LLM Pretraining] This paper proposes FlowMo, a Transformer-based diffusion autoencoder trained in two stages (mode-matching pretraining + mode-seeking post-training)…
tags:
  - "LLM Pretraining"
date: 2026-05-08
content_hash: 5cc90d9fd4a02302
---

# FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2503.11056](https://arxiv.org/abs/2503.11056)
- **Code**: [kylesargent.github.io/flowmo](https://kylesargent.github.io/flowmo)
- **Area**: LLM Pretraining
- **Keywords**: diffusion autoencoder, image tokenization, rectified flow, mode-seeking, discrete tokenizer
- **Authors**: Kyle Sargent, Kyle Hsu, Justin Johnson, Li Fei-Fei, Jiajun Wu (Stanford, UMich)

## TL;DR

This paper proposes FlowMo, a Transformer-based diffusion autoencoder trained in two stages (mode-matching pretraining + mode-seeking post-training), achieving state-of-the-art performance on ImageNet-1K discrete image tokenization for the first time among diffusion autoencoders — without convolutions, adversarial losses, 2D spatially-aligned latents, or distillation from other tokenizers.

## Background & Motivation

Modern visual generation systems are predominantly two-stage: a tokenizer first compresses pixel data into a discrete latent space, followed by a generative model trained in that space. Since VQGAN, state-of-the-art tokenizers have typically combined CNN autoencoders, adversarial losses, perceptual losses, and 2D spatially-aligned latent codes.

Diffusion autoencoders represent an alternative paradigm: using a diffusion model as the decoder to learn perception-oriented image compression end-to-end. However, no prior diffusion autoencoder had achieved state-of-the-art performance on the competitive ImageNet-1K reconstruction benchmark.

The core insight of FlowMo is that for perceptual reconstruction, **sampling a mode of the reconstruction distribution that is perceptually close to the original image (mode-seeking)** is superior to attempting to match all modes (mode-matching). This intuition motivates the two-stage training scheme.

## Method

### Overall Architecture

FlowMo is a diffusion autoencoder consisting of:
- **Encoder** $e_\theta$: maps image $x$ to quantized latent $c$ (a 1D sequence, not 2D spatially aligned)
- **Decoder** $d_\theta$: a conditional diffusion model learning the conditional distribution $p(x|c)$
- Architecture based on MMDiT (the backbone of Stable Diffusion 3), fully Transformer-based with no convolutions
- Encoder and decoder are structurally symmetric but differ in size (decoder is larger and deeper)
- $\mu P$ parameterization is adopted to facilitate hyperparameter transfer

### Quantization

Lookup-Free Quantization (LFQ) is employed:
$$c = q(\hat{c}) = 2 \cdot \mathbb{1}[\hat{c} \geq 0] - 1$$

Element-wise binarization eliminates the complexity of conventional codebook lookups.

### Stage 1A: Mode-Matching Pretraining

Objective: train the encoder and decoder end-to-end so that $p_\theta(x|c)$ matches the true distribution.

Rectified flow loss:
$$\mathcal{L}_{\text{flow}} = \mathbb{E}\left[ \| x - z - d_\theta(x_t, q(e_\theta(x)), t) \|_2^2 \right]$$

where $x_t = tz + (1-t)x$.

Auxiliary losses include:
- **Perceptual loss** $\mathcal{L}_{\text{perc}}$: LPIPS-VGG supervision on the one-step denoising prediction
- **Entropy loss** $\mathcal{L}_{\text{ent}}$: LFQ codebook utilization
- **Commitment loss** $\mathcal{L}_{\text{commit}}$: $\| \hat{c} - q(\hat{c}) \|_2^2$

Total loss: $\mathcal{L}_{\text{flow}} + \lambda_{\text{perc}} \mathcal{L}_{\text{perc}} + \lambda_{\text{commit}} \mathcal{L}_{\text{commit}} + \lambda_{\text{ent}} \mathcal{L}_{\text{ent}}$

A thick-tailed logit-normal noise schedule is used (with 10% of timesteps sampled from a uniform distribution) to prevent color shift caused by zero probability at $t=1$.

### Stage 1B: Mode-Seeking Post-Training (Key Innovation)

The encoder is frozen, and the decoder is optimized by backpropagating through the entire sampling chain (8-step ODE integration):

$$\mathcal{L}_{\text{sample}} = \mathbb{E}\left[ d_{\text{perc}}(x, d_{t_n} \circ d_{t_{n-1}} \circ \cdots \circ d_{t_1}(z)) \right]$$

Total loss: $\mathcal{L}_{\text{flow}} + \lambda_{\text{sample}} \mathcal{L}_{\text{sample}}$

Key implementation details:
- $\lambda_{\text{sample}} = 0.01$; larger values cause reward hacking or training divergence
- Stage 1A uses LPIPS-VGG; Stage 1B uses a ResNet as the perceptual network
- Gradient checkpointing and gradient accumulation are used to reduce memory overhead
- Training runs for approximately 1 epoch with early stopping

### Sampler Design

A shifted sampler is used with timestep intervals:
$$t_i = \left(\frac{n-i+1}{n}\right)^\rho, \quad \rho = 4$$

Setting $\rho > 1$ concentrates sampling in the low-noise regime, biasing samples toward the mean of $p(x|c)$, while preserving sampling FLOPs at low noise levels.

## Key Experimental Results

### Main Results: ImageNet-1K Reconstruction (Table 1)

| Model | BPP | rFID ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|-----|--------|--------|--------|---------|
| OpenMagViT-V2 | 0.070 | 1.17 | 21.63 | 0.640 | 0.111 |
| **FlowMo-Lo** | **0.070** | **0.95** | **22.07** | **0.649** | 0.113 |
| LlamaGen-32 | 0.219 | 0.59 | 24.44 | 0.768 | 0.064 |
| **FlowMo-Hi** | **0.219** | **0.56** | **24.93** | **0.785** | 0.073 |

FlowMo achieves state-of-the-art rFID, PSNR, and SSIM at both BPP levels. The only disadvantage is on the LPIPS metric.

### Ablation Study

**Stage 1A Ablation (Table 4)**:

| Variant | rFID ↓ | PSNR ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| FlowMo (fewer params) | 2.87 | 20.71 | 0.15 |
| 2× patch size | 6.39 | 19.94 | 0.17 |
| MSE-trained encoder | 3.82 | 21.40 | 0.15 |
| No perceptual loss | 13.86 | 22.11 | 0.21 |
| FSQ quantization | 3.14 | 21.31 | 0.14 |
| Logit-normal noise | 4.08 | 16.45 | 0.21 |
| No shifted sampler | 3.42 | 20.25 | 0.16 |
| No guidance | 3.28 | 20.67 | 0.16 |

**Stage 1B Ablation (Table 5 — most critical)**:

| Model | rFID ↓ | PSNR ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| FlowMo-Lo (w/o post-training) | 1.10 | 21.38 | 0.134 |
| **FlowMo-Lo (w/ post-training)** | **0.95** | **22.07** | **0.113** |
| FlowMo-Hi (w/o post-training) | 0.73 | 24.02 | 0.086 |
| **FlowMo-Hi (w/ post-training)** | **0.56** | **24.93** | **0.073** |

Post-training improves all metrics, reducing rFID by 14–23%.

### Generation Quality (Table 2)

| Tokenizer | FID ↓ | IS ↑ | sFID ↓ | Prec. ↑ | Rec. ↑ |
|-----------|-------|------|--------|---------|--------|
| OpenMagViT-V2 | 3.73 | 241 | 10.66 | 0.80 | 0.51 |
| FlowMo-Lo | 4.30 | 274 | 10.31 | 0.86 | 0.47 |

A better tokenizer does not necessarily lead directly to better generation quality, indicating a complex interaction between tokenizer quality and generative model performance.

### Key Findings

1. **Mode-seeking post-training is critical for SOTA**: a simple one-step denoising approach with ResNet perceptual loss cannot substitute for backpropagation through the full sampling chain.
2. **End-to-end training is essential**: pre-training the encoder with MSE before attaching a diffusion decoder leads to significantly worse rFID.
3. **Thick-tailed noise schedule is indispensable**: standard logit-normal scheduling causes color shift.
4. **Reconstructions remain multimodal after post-training**: variance concentrates in perceptually insensitive regions such as backgrounds.

## Highlights & Insights

- **"Better, not broader" philosophy**: rather than matching all reconstruction modes, the model selectively retains the perceptually best mode.
- **Pure Transformer + 1D latent**: challenges the dominance of CNN + 2D latent designs.
- **No adversarial loss**: the diffusion model's inherent multimodal modeling capacity combined with mode-seeking post-training effectively replaces the GAN discriminator.
- **Connection to RLHF-style post-training**: Stage 1B is conceptually analogous to post-training and alignment paradigms for diffusion models.

## Limitations & Future Work

- **Slow inference**: requires multi-step ODE integration ($n=25$ steps), significantly slower than a single forward pass through a CNN decoder.
- **LPIPS not competitive**: slightly inferior to conventional methods on the LPIPS metric.
- **Generation quality does not surpass conventional tokenizers**: FID is marginally worse than OpenMagViT-V2.
- Post-training is computationally expensive due to backpropagation through the full sampling chain and the need for gradient checkpointing.

## Related Work & Insights

- **Diffusion autoencoders enter the SOTA tier**: demonstrates the viability of the diffusion autoencoder approach for competitive image tokenization.
- **Connection to alignment techniques**: mode-seeking post-training shares technical lineage with diffusion model alignment methods such as DDPO and AlignProp.
- **Implications for tokenizer design**: 1D latent + Transformer architectures may offer advantages at scale.

## Rating ⭐⭐⭐⭐⭐

A pioneering contribution. FlowMo is the first diffusion autoencoder to achieve state-of-the-art results on the highly competitive ImageNet tokenization benchmark. The two-stage mode-matching + mode-seeking training strategy is both elegant and principled, and the ablation study is exceptionally thorough. This work has significant implications for the future design of image tokenizers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization](../../ICLR2026/llm_pretraining/understanding_and_improving_shampoo_and_soap_via_kullback-leibler_minimization.md)
- [\[ICLR 2026\] Reducing Class-Wise Performance Disparity via Margin Regularization](../../ICLR2026/llm_pretraining/reducing_class-wise_performance_disparity_via_margin_regularization.md)
- [\[ICML 2026\] Annotations Mitigate Post-Training Mode Collapse](../../ICML2026/llm_pretraining/annotations_mitigate_post-training_mode_collapse.md)
- [\[ICML 2026\] Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability](../../ICML2026/llm_pretraining/beyond_structural_symmetries_linear_mode_connectivity_via_neuron_identifiability.md)
- [\[ICCV 2025\] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution](image_intrinsic_scale_assessment_bridging_the_gap_between_quality_and_resolution.md)

</div>

<!-- RELATED:END -->
