---
title: >-
  [Paper Note] End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer
description: >-
  [ICML 2026][Image Generation][1D Tokenizer] EOSTok adopts a single-stage end-to-end pipeline to jointly train a 1D ViT tokenizer and an autoregressive model. By utilizing the proposed APR (Autoregressive Prediction Recon…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "1D Tokenizer"
  - "Autoregressive Image Generation"
  - "APR loss"
  - "VFM Implicit Alignment"
  - "ImageNet FID"
date: 2026-05-08
content_hash: 52f5787b438c3955
---

# End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.00503](https://arxiv.org/abs/2605.00503)  
**Code**: None  
**Area**: Image Generation / Autoregressive Visual Tokenizer / Representation Alignment  
**Keywords**: 1D Tokenizer, Autoregressive Image Generation, APR loss, VFM Implicit Alignment, ImageNet FID

## TL;DR
EOSTok adopts a single-stage end-to-end pipeline to jointly train a 1D ViT tokenizer and an autoregressive model. By utilizing the proposed APR (Autoregressive Prediction Reconstruction) loss, gradients from next-token predictions are backpropagated to the pixel space to prevent codebook collapse. Simultaneously, "implicit alignment" is employed to inject DINOv2 semantics into the 1D latent space without compromising its 1D structure, ultimately achieving a SOTA FID of 1.48 on ImageNet 256 without guidance.

## Background & Motivation

**Background**: Autoregressive image generation (e.g., VQGAN, LLaMaGen, MAR, VAR) seeks to replicate the success of LLMs. However, most extant methods utilize 2D grid tokenizers, compressing 256×256 images into 16×16 patch tokens decoded via raster-scan. Recent works like TiTok, FlexTok, and Semanticist employ learnable queries to compress images into 1D sequences, primarily targeting high compression ratios (e.g., 32 tokens).

**Limitations of Prior Work**: (1) 2D grid tokens naturally possess bidirectional dependencies (a patch is interpreted by its neighbors in all directions), which conflicts with the unidirectional factorization of raster scanning, causing "directional misalignment" in AR modeling. (2) Existing 1D tokenizers sacrifice reconstruction quality for extreme compression and typically follow a two-stage training paradigm—training the tokenizer via reconstruction before freezing it for the AR model—meaning AR gradients cannot update the tokenizer. (3) Directly aligning VFM representations to the 1D latent space causes them to degenerate into raster-ordered sequences, inadvertently reintroducing 2D priors.

**Key Challenge**: There is a tangled contradiction between "reconstruction quality vs. AR friendliness," "2D semantic priors vs. 1D sequence structure," and "next-token loss vs. pixel generation quality." Single-stage joint training is prone to "hacking" by the NTP loss—the tokenizer learns to use a minimal subset of tokens to minimize NTP loss, causing codebook utilization to collapse from 99.8% to 51.8%.

**Goal**: (1) Design a 1D tokenizer that does not prioritize extreme compression; (2) Enable the tokenizer to receive generation gradients directly from pixel space; (3) Inject VFM semantics into the 1D path without disrupting the 1D structure.

**Key Insight**: The authors observe that the root of 2D limitations lies in the conflict between token arrangement and the direction of causal factorization. Once the 2D spatial prior is removed, a 1D tokenizer can natively support vanilla AR modeling without requiring random masking or next-scale prediction.

**Core Idea**: Utilize APR loss to decode AR-predicted tokens back to pixels for alignment with ground truth, establishing end-to-end generation supervision. Simultaneously, apply "implicit alignment" to align VFM representations with the encoder's intermediate 2D hidden patch embeddings rather than the 1D latent space, allowing 1D latents to absorb semantics indirectly.

## Method

### Overall Architecture
Input 256×256 images $x$ are processed by a ViT encoder into 2D patch embeddings $h_\text{Enc}$ and 1D latent tokens $z$ (extracted via $L$ learnable queries); only $z$ undergoes IBQ quantization to obtain $z_q$. An AR model $\mathcal{G}_\theta$ performs next-token prediction on $z_q$. A redesigned embedding layer uses a probability matrix multiplication with the codebook instead of look-ups, enabling NTP gradients to backpropagate to the encoder and codebook. The decoder $\mathcal{D}_\psi$ concatenates $z_q$ and the AR-predicted tokens $\hat z_q = \mathcal{G}_\theta(z_q)$ along the batch dimension for pixel decoding. The total loss is $\mathcal{L}_\text{VQVAE} + \lambda_\text{NTP}\mathcal{L}_\text{NTP} + \lambda_\text{APR}\mathcal{L}_\text{APR} + \lambda_\text{align}\mathcal{L}_\text{align}$.

### Key Designs

1.  **APR loss: Propagating AR Generation Gradients to Pixel Space**:
    *   **Function**: Prevents the tokenizer from exploiting the NTP loss and collapsing into a few tokens.
    *   **Mechanism**: Defined as $\mathcal{L}_\text{APR}(\phi, \psi, \theta) = \|x - \mathcal{D}_\psi(\mathcal{G}_\theta(z_q))\|_2^2$ (augmented with LPIPS). Under teacher forcing, the AR model predicts $\hat z_q$ in a single step, which is decoded back to pixels and aligned with the original image. During training, $\hat z_q$ and $z_q$ are concatenated to pass through the decoder, enabling simultaneous computation of reconstruction and APR losses in one forward pass.
    *   **Design Motivation**: Vanilla end-to-end training increases AR accuracy (11.8% to 30.2%) but causes codebook utilization to drop to 51.8% and gFID to rise to 8.01, as NTP loss focuses only on discrete token space. APR shifts the constraint from "token alignment" to "pixel alignment after decoding," matching the true generation objective. Consequently, codebook utilization restores to 99.7%, and gFID improves to 3.32.

2.  **Implicit Alignment: Injecting VFM into the 1D Encoder without Breaking 1D Structure**:
    *   **Function**: Implants semantics from VFMs (e.g., DINOv2) into the 1D tokenizer without leaking 2D spatial priors into the 1D latent space.
    *   **Mechanism**: The authors compare three methods: (a) Direct alignment: aligning 1D latent $z$ to VFM features $f(x)$, resulting in $z$ degenerating into a raster-ordered sequence; (b) Direct substitution: replacing original patch embeddings with VFM; (c) Implicit alignment: defined as $\mathcal{L}_\text{implicit} = -\frac{1}{N}\sum_n \text{sim}(h_\omega(h_\text{Enc}^{[n]}), y^{[n]})$. VFM is aligned with the encoder's internal 2D hidden patch embeddings, allowing the 1D latent $z$ to absorb semantics via cross-attention while maintaining flexibility in its own arrangement.
    *   **Design Motivation**: Direct alignment re-imposes 2D constraints, negating 1D AR advantages (gFID increases from 12.27 to non-convergence). Implicit alignment grants 1D latents freedom for permutation, reducing gFID from 12.27 to 3.32 and increasing AR accuracy.

3.  **Differentiable Embedding for NTP Gradient Backpropagation**:
    *   **Function**: Facilitates true end-to-end joint training.
    *   **Mechanism**: Standard LLM embeddings use discrete index look-ups, which are non-differentiable for the tokenizer. EOSTok formulates AR input as the probability matrix $\text{Ind} \in \mathbb{R}^{L \times K}$ from the IBQ output, using $h = \text{Ind}^\top \text{Embed}$ for the weighted sum. Combined with the straight-through trick in IBQ ($\text{Ind} = \text{onehot}(\arg\max p) + [p - \text{stopgrad}(p)]$), gradients flow continuously from AR loss back to the encoder.
    *   **Design Motivation**: Without this pipeline, NTP loss only updates the AR model; the tokenizer never learns which token sequences are inherently more predictable for the AR model.

### Loss & Training
The objective is $\mathcal{L}_\text{E2E} = \mathcal{L}_\text{VQVAE} + \lambda_\text{NTP}\mathcal{L}_\text{NTP} + \lambda_\text{APR}\mathcal{L}_\text{APR} + \lambda_\text{align}\mathcal{L}_\text{align}$. Here, $\mathcal{L}_\text{recon}$ consists of L1/L2 + LPIPS + GAN, and $\mathcal{L}_\text{reg}$ includes commitment and entropy losses. The decoder also undergoes REPA-style alignment—aligning hidden features of mask tokens at the $k$-th layer to VFM to accelerate 1D decoder convergence (the paper analogizes the 1D decoder to "conditional generation" rather than simple "reconstruction").

## Key Experimental Results

### Main Results

| Model | Tokenizer | #Tokens | rFID ↓ | gFID (w/o guidance) ↓ | gFID (w/ guidance) ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LDM-4 | SD-VAE (2D) | 64×64 | 0.27 | 10.56 | 3.60 |
| DiT-XL/2 | SD-VAE | 32×32 | 0.62 | 9.62 | 2.27 |
| MAR-L | SD-VAE | 16×16 | 0.87 | 2.60 | 1.78 |
| Lightning-DiT| VA-VAE | 32×32 | 0.28 | 2.17 | 1.35 |
| **EOSTok-H** | **1D + Implicit VFM** | 256 query | — | **1.48** | — |

### Ablation Study

| Configuration | rFID ↓ | gFID ↓ | AR Acc. ↑ | Codebook Util. |
| :--- | :--- | :--- | :--- | :--- |
| Two-stage Baseline | 1.09 | 3.82 | 11.8% | 99.8% |
| Vanilla E2E (NTP only) | 4.92 | 8.01 | 30.2% | 51.8% |
| **+ APR loss** | 1.02 | 3.32 | 11.9% | 99.7% |
| + Decoder VFM Align | 1.12 | 5.68 | 8.2% | — |
| + Encoder Direct Align | 0.98 | 5.98 | 8.5% | — |
| + Direct substitution | 1.05 | 4.89 | 12.1% | — |
| **+ Implicit alignment (Ours)**| 1.02 | 3.32 | 11.9% | — |

### Key Findings
*   **Vanilla E2E is a cautionary tale**: Using only NTP supervision leads to inflated AR accuracy (30.2%) but catastrophic generation quality (gFID 8.01) and codebook collapse. This is a classic example of "aligning in the wrong dimension," visualized via PCA on a 3D sphere.
*   **APR loss is the critical fix**: Adding a pixel-level loss restores codebook utilization from 51.8% to 99.7% and recovers rFID/gFID performance. It represents a victory for "directly supervising the metrics of interest."
*   **2D spatial priors are poison for 1D AR**: Direct alignment forces VFM constraints on the 1D latent space, increasing gFID and hindering convergence, proving that 1D routes should not implicitly incorporate 2D sequential assumptions.
*   **Scaling friendliness**: gFID decreases monotonically across EOSTok-S/L/H, and performance improves as codebook sizes increase from 4096 to 16384.

## Highlights & Insights
*   **The Paradigm Value of Joint Training**: This work demonstrates that single-stage training can preserve reconstruction and enhance generation as long as supervision is directed toward the generation target (pixel MSE) rather than intermediate proxies (NTP). This challenges the "freeze encoder, then train generator" paradigm.
*   **The Subtlety of VFM Injection**: The decision between "aligning latent vs. hidden" and "direct replacement vs. implicit distillation" is crucial. It provides a counter-example: adding VFM is not universally beneficial; improper alignment can degrade performance.
*   **The Differentiable Codebook Embedding Trick**: Replacing look-ups with `Ind^T Embed` is a simple yet critical engineering modification that closes the end-to-end loop, applicable to any joint optimization of VQ and downstream models.

## Limitations & Future Work
*   Evaluation is limited to ImageNet-256 class-conditional generation; its effectiveness in text-to-image or video remains to be explored.
*   The number of 1D tokens is fixed at 256; while this allows for comparison with 2D sequences, adaptive token counts (e.g., nested dropout in FlexTok) have not been investigated.
*   APR loss requires the AR model to decode to pixels during each training step, leading to higher computational costs than two-stage methods; wall-clock comparisons were not provided.
*   The behavior of other codebook designs (e.g., FSQ, LFQ) within this end-to-end framework is unknown.

## Related Work & Insights
*   **vs. TiTok / FlexTok / Semanticist**: While they use 1D tokenizers, they utilize two-stage training. EOSTok is the first to achieve end-to-end 1D + AR integration.
*   **vs. VAR / MAR**: VAR uses next-scale prediction to bypass 2D directionality, while MAR uses random masking. EOSTok suggests that by discarding 2D priors, vanilla AR is sufficient, aligning closely with LLM simplicity.
*   **vs. VA-VAE / REPA / RAE**: These apply VFM alignment to diffusion models. EOSTok systematically compares three injection methods, establishing that 1D routes require implicit alignment.
*   **vs. LLaMaGen / RQ-VAE**: Traditional 2D AR models typically yield gFID scores between 8-15 without guidance. EOSTok-H pushes 1D AR to 1.48, rivaling the best diffusion models like VA-VAE (1.35).

## Rating
*   Novelty: ⭐⭐⭐⭐ The combination of end-to-end 1D+AR training, APR loss, and implicit alignment achieves SOTA results.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablations across joint training, injection methods, scaling, codebook size, and convergence.
*   Writing Quality: ⭐⭐⭐⭐ Effectively explains failure cases (NTP hacking, Direct alignment degradation), providing "how" and "why."
*   Value: ⭐⭐⭐⭐⭐ Revitalizes the AR visual generation route and may shift community perception of 1D tokenizers beyond just "high compression."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CLEAR: Context-Aware Learning with End-to-End Mask-Free Inference for Adaptive Video Subtitle Removal](clear_context-aware_learning_with_end-to-end_mask-free_inference_for_adaptive_vi.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](../../CVPR2026/image_generation/deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](../../ICCV2025/image_generation/end-to-end_multi-modal_diffusion_mamba.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](../../ICCV2025/image_generation/holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICML 2026\] Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)

</div>

<!-- RELATED:END -->
