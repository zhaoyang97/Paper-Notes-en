---
title: >-
  [Paper Note] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer
description: >-
  [ICML 2026][Image Generation][1D Tokenizer] EOSTok jointly trains a 1D ViT tokenizer and an autoregressive model in a single-stage end-to-end pipeline. The newly proposed APR (Autoregressive Prediction Reconstruction) lo…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "1D Tokenizer"
  - "Autoregressive Image Generation"
  - "APR loss"
  - "VFM Implicit Alignment"
  - "ImageNet FID"
date: 2026-05-08
content_hash: 7334ea88a4ebac29
---

# End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer

**Conference**: ICML 2026  
**arXiv**: [2605.00503](https://arxiv.org/abs/2605.00503)  
**Code**: None  
**Area**: Image Generation / Autoregressive Visual Tokenizer / Representation Alignment  
**Keywords**: 1D Tokenizer, Autoregressive Image Generation, APR loss, VFM Implicit Alignment, ImageNet FID

## TL;DR
EOSTok jointly trains a 1D ViT tokenizer and an autoregressive model in a single-stage end-to-end pipeline. The newly proposed APR (Autoregressive Prediction Reconstruction) loss enables gradients from "next-token prediction" to flow back to the pixel space, preventing codebook collapse. "Implicit alignment" injects DINOv2 semantics into the 1D latent space without disrupting the 1D autoregressive structure. Ultimately, EOSTok achieves an FID of 1.48 on ImageNet 256 without guidance (SOTA).

## Background & Motivation

**Background**: Autoregressive image generation (VQGAN, LLaMaGen, MAR, VAR, etc.) aims to replicate the success of LLMs, but most approaches use a 2D grid tokenizer: compressing a 256×256 image into 16×16 patch tokens, decoded in raster-scan order. Recent works like TiTok, FlexTok, and Semanticist use learnable queries to compress images into 1D sequences, mainly pursuing high compression rates (e.g., 32 tokens).

**Limitations of Prior Work**: (1) 2D grid tokens naturally have bidirectional dependencies (a patch is interpreted by its neighbors), which conflicts with the unidirectional factorization of raster-scan AR modeling, leading to "directional misalignment." (2) Existing 1D tokenizers sacrifice reconstruction quality for high compression and are trained in two stages—tokenizer is first trained for reconstruction, then frozen for AR model training, so AR gradients never reach the tokenizer. (3) Directly aligning VFM representations to the 1D latent space causes degeneration into a raster-ordered patch sequence, reintroducing 2D priors.

**Key Challenge**: The three intertwined trade-offs are "reconstruction quality vs. AR-friendliness," "2D semantic priors vs. 1D sequence structure," and "next-token loss vs. pixel generation quality." Single-stage joint training is vulnerable to NTP loss exploitation—tokenizer learns to use very few tokens to minimize NTP loss (hacking), causing codebook utilization to collapse from 99.8% to 51.8%.

**Goal**: (1) Design a 1D tokenizer without forcing extreme compression; (2) Allow the tokenizer to directly receive generation gradients from pixel space; (3) Inject VFM semantics into the 1D path without disrupting the 1D structure.

**Key Insight**: The essence of the 2D limitation is the "conflict between token arrangement and causal factorization direction." Once the 2D prior is removed, a 1D tokenizer can natively support vanilla AR modeling—no need for random masking or next-scale prediction.

**Core Idea**: Use APR loss to decode AR-predicted tokens back to pixels and align with ground truth, forming end-to-end generative supervision. Simultaneously, use "implicit alignment" to align VFM representations to the encoder's 2D hidden patch embeddings rather than the 1D latent space, allowing the 1D latent to absorb semantics indirectly.

## Method

### Overall Architecture
The input is a 256×256 image $x$, processed by a ViT encoder into 2D patch embeddings $h_\text{Enc}$ and 1D latent tokens $z$ (extracted using $L$ learnable queries); only $z$ is retained and quantized via IBQ to obtain $z_q$. The AR model $\mathcal{G}_\theta$ performs next-token prediction on $z_q$. The newly designed embedding layer multiplies the probability matrix with the codebook instead of using look-up, allowing NTP gradients to backpropagate to the encoder and codebook. The decoder $\mathcal{D}_\psi$ decodes both $z_q$ and the AR model's prediction $\hat z_q = \mathcal{G}_\theta(z_q)$ (concatenated along the batch dimension) back to pixels. The total loss is $\mathcal{L}_\text{VQVAE} + \lambda_\text{NTP}\mathcal{L}_\text{NTP} + \lambda_\text{APR}\mathcal{L}_\text{APR} + \lambda_\text{align}\mathcal{L}_\text{align}$.

### Key Designs

1. **APR loss: Backpropagating AR generation gradients to pixel space**:

    - **Function**: Prevents the tokenizer from being exploited by NTP loss and collapsing to a few tokens.
    - **Mechanism**: Defines $\mathcal{L}_\text{APR}(\phi, \psi, \theta) = \|x - \mathcal{D}_\psi(\mathcal{G}_\theta(z_q))\|_2^2$ (plus LPIPS). Under teacher forcing, the AR model predicts $\hat z_q$ in one step, which is decoded back to pixels and aligned with the original image. During training, $\hat z_q$ and $z_q$ are concatenated along the batch dimension and decoded together, enabling simultaneous computation of reconstruction and APR losses in one forward pass.
    - **Design Motivation**: Vanilla end-to-end training with only NTP loss increases AR accuracy from 11.8% to 30.2%, but codebook utilization collapses to 51.8% and gFID soars to 8.01—because NTP loss only considers the discrete token space, not the final pixels. APR shifts the constraint from "token matching" to "pixel matching after decoding," aligning the constraint with the true generation objective. Codebook utilization recovers to 99.7%, and gFID drops to 3.32.

2. **Implicit alignment: Injecting VFM into the 1D encoder without disrupting 1D structure**:

    - **Function**: Embeds DINOv2 and other VFM semantics into the 1D tokenizer without leaking 2D spatial priors into the 1D latent space.
    - **Mechanism**: The authors compare three injection methods—(a) Direct alignment: aligns 1D latent $z$ to VFM features $f(x)$, causing $z$ to degenerate into a raster-ordered sequence; (b) Direct substitution: replaces original patch embeddings with VFM; (c) Implicit alignment: defines $\mathcal{L}_\text{implicit} = -\frac{1}{N}\sum_n \text{sim}(h_\omega(h_\text{Enc}^{[n]}), y^{[n]})$, aligning VFM to the encoder's intermediate 2D hidden patch embeddings, allowing the 1D latent $z$ to absorb semantics via cross-attention without being forced to retain 2D order.
    - **Design Motivation**: Direct alignment pulls the 1D space "back to 2D," nullifying the benefits of AR generation (gFID increases from 12.27 to non-convergence); implicit alignment allows the 1D latent to remain freely ordered, reducing gFID from 12.27 to 3.32 and increasing AR accuracy from 7.8% to 11.9%.

3. **Differentiable embedding enables NTP gradients to flow to the tokenizer**:

    - **Function**: Makes true end-to-end joint training feasible.
    - **Mechanism**: Standard LLM embeddings use discrete index look-up, which is non-differentiable for the tokenizer. EOSTok changes the AR input to the probability output of IBQ, $\text{Ind} \in \mathbb{R}^{L \times K}$, and computes $h = \text{Ind}^\top \text{Embed}$ as a weighted sum. With IBQ's straight-through trick ($\text{Ind} = \text{onehot}(\arg\max p) + [p - \text{stopgrad}(p)]$), gradients can flow continuously from AR loss back to the encoder and codebook.
    - **Design Motivation**: Without this pipeline, NTP loss can only update the AR model; the tokenizer never learns which token sequences are easier for AR prediction.

### Loss & Training
The overall objective is $\mathcal{L}_\text{E2E} = \mathcal{L}_\text{VQVAE} + \lambda_\text{NTP}\mathcal{L}_\text{NTP} + \lambda_\text{APR}\mathcal{L}_\text{APR} + \lambda_\text{align}\mathcal{L}_\text{align}$; $\mathcal{L}_\text{recon}$ includes L1/L2 + LPIPS + GAN, and $\mathcal{L}_\text{reg}$ includes commitment + entropy. The decoder also performs REPA-style alignment—aligning the hidden features of the $k$-th layer mask token to VFM, accelerating 1D decoder convergence (the paper likens the 1D decoder to "conditional generation" rather than "reconstruction").

## Key Experimental Results

### Main Results

| Model | Tokenizer | #Tokens | rFID ↓ | gFID (no guidance) ↓ | gFID (with guidance) ↓ |
|-------|-----------|---------|--------|----------------------|-----------------------|
| LDM-4 | SD-VAE (2D) | 64×64 | 0.27 | 10.56 | 3.60 |
| DiT-XL/2 | SD-VAE | 32×32 | 0.62 | 9.62 | 2.27 |
| MAR-L | SD-VAE | 16×16 | 0.87 | 2.60 | 1.78 |
| Lightning-DiT | VA-VAE | 32×32 | 0.28 | 2.17 | 1.35 |
| **EOSTok-H** | **1D + VFM Implicit Alignment** | 256 query | — | **1.48** | — |

### Ablation Study

| Configuration | rFID ↓ | gFID ↓ | AR Acc. ↑ | Codebook Utilization |
|---------------|--------|--------|-----------|----------------------|
| Two-stage baseline | 1.09 | 3.82 | 11.8% | 99.8% |
| Vanilla E2E (NTP only) | 4.92 | 8.01 | 30.2% | 51.8% |
| **+ APR loss** | 1.02 | 3.32 | 11.9% | 99.7% |
| + Decoder VFM alignment | 1.12 | 5.68 | 8.2% | — |
| + Encoder Direct alignment | 0.98 | 5.98 | 8.5% | — |
| + Direct substitution | 1.05 | 4.89 | 12.1% | — |
| **+ Implicit alignment (Ours)** | 1.02 | 3.32 | 11.9% | — |

### Key Findings
- **Vanilla E2E as a negative example**: Adding only NTP supervision leads to artificially high AR accuracy (30.2%) but poor generation quality (gFID 8.01) and severe codebook collapse—an example of "alignment on the wrong dimension." The authors visualize this collapse by projecting the codebook onto a 3D sphere using PCA.
- **APR loss as a crucial fix**: Adding a single pixel-level loss restores codebook utilization from 51.8% to 99.7%, and rFID/gFID fully recover—demonstrating the importance of directly supervising the true target.
- **2D spatial priors are detrimental to 1D AR**: Direct alignment of VFM to the 1D latent space increases gFID from 12.27 to 5.98 and fails to converge—showing that 1D approaches cannot mix in 2D order assumptions.
- **Scaling-friendly**: On EOSTok-S/L/H models, gFID decreases monotonically, and codebook size increases from 4096 to 16384 with continued improvement; larger models reduce the gap between different codebook configurations.

## Highlights & Insights
- **Paradigm value of "joint training + end-to-end supervision"**: The paper demonstrates that as long as the supervision signal targets the true generation objective (pixel MSE) rather than an intermediate proxy (NTP), single-stage training can preserve reconstruction and improve generation. This challenges the "freeze encoder, then train generator" paradigm.
- **Subtlety of VFM injection**: Choices like "align to latent vs. intermediate hidden" and "direct replacement vs. implicit distillation" determine whether 1D AR is viable. The paper provides a counterexample: adding VFM is not always beneficial; misaligned injection can be worse.
- **Differentiable codebook embedding trick**: Replacing look-up with `Ind^T Embed` is a seemingly engineering but actually crucial change, closing the loop for end-to-end joint training and transferable to any VQ + downstream joint optimization scenario.

## Limitations & Future Work
- Experiments are only on class-conditional ImageNet-256; whether SOTA can be replicated in text-to-image, video, or more complex scenarios remains to be seen.
- The number of 1D tokens is fixed at 256 for comparison with 2D sequence length; adaptive token numbers (e.g., FlexTok's nested dropout) are not explored.
- APR loss requires the AR model to decode to pixels at each step, making training more expensive than two-stage approaches; wall-clock comparisons are not provided.
- The quantizer is fixed to IBQ; the behavior of other codebook designs (e.g., FSQ, LFQ) in this end-to-end framework is unknown.

## Related Work & Insights
- **vs TiTok / FlexTok / Semanticist**: These works use 1D tokenizers but still rely on two-stage training; EOSTok is the first to truly achieve end-to-end 1D + AR joint training.
- **vs VAR / MAR**: VAR uses next-scale prediction to circumvent 2D directionality, MAR uses random masking; EOSTok advocates that "vanilla AR suffices once 2D priors are dropped," aligning more with the simplicity of LLMs.
- **vs VA-VAE / REPA / RAE**: These works use VFM alignment for diffusion models; EOSTok systematically compares three injection methods and concludes that "implicit alignment is essential for the 1D approach."
- **vs LLaMaGen / RQ-VAE**: Traditional 2D AR models start with gFID 8-15 without guidance; EOSTok-H pushes 1D AR to 1.48, nearly matching the best diffusion model (VA-VAE) at 1.35.

## Rating
- Novelty: ⭐⭐⭐⭐ End-to-end 1D+AR joint training + APR loss + implicit alignment—each is not disruptive alone, but together they achieve SOTA.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablations on joint training, injection methods, scaling, codebook size, and convergence curves.
- Writing Quality: ⭐⭐⭐⭐ Clearly explains each failure case (NTP hacking, Direct alignment degeneration), enabling readers to understand the "why."
- Value: ⭐⭐⭐⭐⭐ Injects new vitality into AR visual generation, potentially changing the community's perception that 1D tokenizers are "only for high compression."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spectral Image Tokenizer](../../ICCV2025/image_generation/spectral_image_tokenizer.md)
- [\[ICML 2025\] FlexTok: Resampling Images into 1D Token Sequences of Flexible Length](../../ICML2025/image_generation/flextok_resampling_images_into_1d_token_sequences_of_flexible_length.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](../../ICCV2025/image_generation/holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[CVPR 2025\] SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer](../../CVPR2025/image_generation/softvq-vae_efficient_1-dimensional_continuous_tokenizer.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](../../CVPR2025/image_generation/tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
