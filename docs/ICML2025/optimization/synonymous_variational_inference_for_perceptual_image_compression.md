---
title: >-
  [Paper Note] Synonymous Variational Inference for Perceptual Image Compression
description: >-
  [ICML2025][Optimization][Synonymous Variational Inference] Based on the perspective of synonymity in semantic information theory, this work proposes the Synonymous Variational Inference (SVI) method to theoretically prove that the optimization direction for perceptual image compression is a rate-distortion-perception three-way trade-off. It also designs a progressive Synonymous Image Compression (SIC) codec, allowing a single model to cover multiple bitrates and perceptual qu…
tags:
  - "ICML2025"
  - "Optimization"
  - "Synonymous Variational Inference"
  - "Perceptual Image Compression"
  - "Semantic Information Theory"
  - "Rate-Distortion-Perception Trade-off"
  - "Progressive Codec"
date: 2026-05-08
content_hash: a29357bbc72b74fa
---

# Synonymous Variational Inference for Perceptual Image Compression

**Conference**: ICML2025  
**arXiv**: [2505.22438](https://arxiv.org/abs/2505.22438)  
**Code**: TBD  
**Area**: Image Compression / Perceptual Quality Optimization  
**Keywords**: Synonymous Variational Inference, Perceptual Image Compression, Semantic Information Theory, Rate-Distortion-Perception Trade-off, Progressive Codec

## TL;DR

Based on the perspective of synonymity in semantic information theory, this work proposes the Synonymous Variational Inference (SVI) method to theoretically prove that the optimization direction for perceptual image compression is a rate-distortion-perception three-way trade-off. It also designs a progressive Synonymous Image Compression (SIC) codec, allowing a single model to cover multiple bitrates and perceptual quality levels.

## Background & Motivation

### Background

Classical lossy image compression has evolved along the rate-distortion (R-D) framework: traditional methods such as JPEG and BPG, as well as VAE-based learned image compression (LIC), are optimized between bitrate and PSNR/MS-SSIM. However, **low distortion $\neq$ high perceptual quality**—Blau & Michaeli (2018, 2019) revealed the distortion-perception trade-off, expanding the optimization objective to a rate-distortion-perception (R-D-P) three-way framework.

### Limitations of Prior Work

Existing perceptual compression schemes design perceptual losses in their own ways: HiFiC uses GAN adversarial loss, while MS-ILLM mixes LPIPS and adversarial loss. These methods are diverse but lack a unified theoretical explanation. Specifically:

- Why is a distribution divergence term required in the loss function? There is a lack of essential mathematical explanation.
- The relationship between different perceptual metrics (KL divergence, Wasserstein distance, LPIPS, DISTS) remains unclear.
- There is no unified variational inference framework to guide the design of perceptual compression schemes.

### Motivation of Ours

The authors approach the problem from the perspective of **synonymity** in semantic information theory (Niu & Zhang, 2024): a single semantic meaning can have multiple syntactic expressions, where perceptual similarity equals a synonymy relation. Accordingly, the concept of a Synonymous Set (Synset) is established, and partial semantic KL divergence is used to drive variational inference, mathematically deriving the necessity of the R-D-P three-way trade-off.

## Method

### Core Concepts: Synset and Semantic Variable

- **Synset** $\mathcal{X}$: The set of all images perceptually similar to the original image $\boldsymbol{x}$.
- **Semantic Variable** $\mathring{X}$: Corresponds to various possible synsets, whose semantic entropy satisfies $H_s(\mathring{U}) \leq H(U)$ (semantic uncertainty $\leq$ syntactic uncertainty).
- **Partial Semantic KL Divergence**: Measures the distance between the syntactic distribution $q$ and the semantic distribution $p_s$:

$$D_{\text{KL},s}[q \| p_s] = \sum_{i_s} \sum_{u_i \in \mathcal{U}_{i_s}} q(u_i) \log \frac{q(u_i)}{p(\mathcal{U}_{i_s})}$$

### Synonymous Variational Inference (SVI)

The latent representation $\tilde{\boldsymbol{y}}$ is decomposed into a **synonymous representation** $\tilde{\boldsymbol{y}}_s$ (encoding shared semantics) and a **detail representation** $\tilde{\boldsymbol{y}}_\epsilon$ (encoding individual differences). By minimizing the partial semantic KL divergence, the posterior of the ideal synset is approximated:

$$\min \mathbb{E}_{\boldsymbol{x} \sim p(\boldsymbol{x})} D_{\text{KL},s}[q \| p_{\tilde{\boldsymbol{y}}_s | \mathcal{X}}]$$

After expansion, the SVI objective decomposes into three terms:

1. **First term** $\log q(\tilde{\boldsymbol{y}} | \boldsymbol{x})$: Becomes 0 under the uniform noise assumption.
2. **Second term** $-\log p_{\mathcal{X} | \tilde{\boldsymbol{y}}_s}$: Synonymous likelihood term $\rightarrow$ equivalent to weighted distortion + expected KL divergence (perceptual term).
3. **Third term** $-\log p_{\tilde{\boldsymbol{y}}_s}$: Rate term.

### Theorem 3.3 (Synonymous Rate-Distortion-Perception Trade-off)

The minimum achievable bitrate for perceptual compression is:

$$R(\mathcal{X}) = \min_{p(\hat{\mathcal{X}} | \boldsymbol{x})} I(\boldsymbol{X}; \hat{\mathring{\boldsymbol{X}}})$$

Under the constraints of expected distortion $\leq D$ and expected KL divergence $\leq P$, the final training loss function is:

$$\mathcal{L}_{\mathcal{X}} = \lambda_r \cdot \text{Rate} + \lambda_d \cdot \text{E-MSE} + \lambda_p \cdot \text{E-KLD}$$

### Synonymous Image Compression (SIC) Framework

**General Framework**: The encoder extracts $\hat{\boldsymbol{y}}_s$ (synonymous representation) and encodes only this part; the decoder generates multiple reconstructed images satisfying the synonymy relationship by sampling different $\hat{\boldsymbol{y}}_{\epsilon,j}$ multiple times.

**Progressive Framework**: The $C=512$ channels of the latent feature $\hat{\boldsymbol{y}}$ are evenly divided into $L=16$ levels. The first $l$ levels serve as the synonymous representation $\hat{\boldsymbol{y}}_s^{(l)}$, while subsequent levels serve as detail representations. A single codec supports 16 bitrate levels.

### Loss & Training

The loss for each level $l$ is:

$$\mathcal{L}^{(l)} = \alpha \mathcal{L}_{\mathcal{X}}^{(l)} + (1-\alpha) \mathcal{L}_{\mathcal{X}}^{(L)} + \mathcal{L}_c^{(l)}$$

where the loss of each level is:

$$\mathcal{L}_{\mathcal{X}}^{(l)} = \mathbb{E}\left[-\lambda_r^{(l)} \log p_{\hat{\boldsymbol{y}}_s^{(l)}} + \frac{1}{M}\sum_{i=1}^{M}\left(\lambda_d^{(l)} \cdot \text{MSE} + \lambda_p^{(l)} \cdot \text{LPIPS}\right)\right]$$

In practice, LPIPS replaces the theoretical KL divergence, and $M$ represents the number of reconstruction samples.

## Key Experimental Results

### Experimental Setup

| Item | Setting |
|------|------|
| Backbone Network | Swin Transformer (Analysis/Synthesis Transform) + CNN (Entropy Model) |
| Latent Channels | $C = 512$, $L = 16$ levels (32 channels per level) |
| Training Data | OpenImages V6, 100k images, $256 \times 256$ |
| Training Volume | $10^6$ iterations, batch 16, lr $10^{-4}$, AdamW |
| Test Sets | CLIC2020 / DIV2K val / Kodak |

### Main Results (DISTS Perceptual Quality)

| Method | Number of Models | DISTS Performance | PSNR Trend |
|------|---------|-----------|----------|
| BPG / VTM | Multiple bitrate points | Baseline (distortion-optimized) | Best |
| HiFiC | One model per point | Perceptually best (GAN-driven) | Lower |
| MS-ILLM | One model per point | Perceptually second-best (GAN-driven) | Medium |
| MS-ILLM No-GAN | One model per point | Medium (LPIPS-driven) | Medium |
| **SIC (M=1)** | **Single model for 16 bitrates** | **Surpasses No-GAN by a large margin** | Close to / surpasses No-GAN |
| **SIC (M=5)** | **Single model for 16 bitrates** | **Slightly better than M=1 at low-to-medium bitrates** | Comparable |

### Supplementary Experiments on GAN Fine-tuning

After fine-tuning the base model with non-saturating adversarial loss for $2 \times 10^5$ steps:

- DISTS and FID improve, gradually approaching the level of MS-ILLM (with GAN).
- LPIPS remains largely unchanged.
- PSNR decreases (validating the distortion-perception trade-off).
- Numeric improvements are limited at low bitrates (bpp < 0.10), but visual quality is significantly enhanced.

## Highlights & Insights

1. **Solid Theoretical Contribution**: This work is the first to demonstrate the fundamental reason for the existence of the divergence term in perceptual compression from the perspective of semantic information theory, clarifying empirical R-D-P losses within the SVI framework.
2. **Single Model, Multiple Bitrates**: The progressive SIC covers 16 bitrate points using a single codec, whereas HiFiC/MS-ILLM requires training an independent model for each bitrate point.
3. **Synset Sampling**: The decoder can generate multiple reconstructed images that are perceptually similar but differ in detail for the same encoding result, inherently offering diversity.
4. **Theoretical Compatibility**: It is proven that existing R-D-P frameworks are special cases of the SVI derivation (degenerating when the reconstruction set contains only one image).
5. **Advantage in DISTS Metric**: Outperforms the LPIPS-trained No-GAN scheme on the DISTS metric, which is closer to human perception.

## Limitations & Future Work

1. **Imperfect Perceptual Loss Substitution**: Replacing the theoretical KL divergence with LPIPS is an approximation, leaving a performance gap of ours compared to GAN-based schemes (HiFiC/MS-ILLM).
2. **Limited Performance at Low Bitrates**: The performance gain is limited when bpp < 0.10, indicating that the uniform channel-splitting hierarchy strategy might not be optimal.
3. **Limited FID Improvement**: The sampling mechanism insufficiently optimizes distribution alignment, and multiple samplings fail to significantly improve FID.
4. **Inadequate GAN Fine-tuning**: Mutual gaming between multiple levels makes adversarial training convergence difficult.
5. **Training Overhead**: Alternating training progressively for $L=16$ levels results in high training complexity.

## Related Work & Insights

- **Blau & Michaeli (2018, 2019)**: R-D-P three-way trade-off (theoretical foundation of this work)
- **Niu & Zhang (2024)**: Synonymity principle in semantic information theory (core source of inspiration)
- **HiFiC (Mentzer et al., 2020)**: GAN-driven perceptual compression (baseline)
- **MS-ILLM (Muckley et al., 2023)**: Multi-scale LPIPS + GAN (primary comparison scheme)
- **Ballé et al. (2017, 2018)**: Variational inference framework for learned compression (methodological starting point)
- Insight: Introducing the "set-element" relationship from semantic information theory into the compression field provides new theoretical tools for future semantic communication and compression.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First theoretical link between semantic information theory and perceptual compression, with strong originality in the SVI method.
- Experimental Thoroughness: ⭐⭐⭐ — Single-model multi-bitrate is convincing, but the gap with SOTA is not fully closed, and GAN fine-tuning is not deeply explored.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivation is clear and rigorous, though the complex notation system (synset/semantic variable/partial semantic KL) increases the reading burden.
- Value: ⭐⭐⭐⭐ — Provides a unified theoretical perspective for perceptual compression and a practical progressive single-model approach, though practical performance still needs engineering refinement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Revisiting Unbiased Implicit Variational Inference](revisiting_unbiased_implicit_variational_inference.md)
- [\[NeurIPS 2025\] Least Squares Variational Inference](../../NeurIPS2025/optimization/least_squares_variational_inference.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](../../NeurIPS2025/optimization/viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](../../NeurIPS2025/optimization/neuro-symbolic_entity_alignment_via_variational_inference.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](../../NeurIPS2025/optimization/vera_variational_inference_framework_for_jailbreaking_large_language_models.md)

</div>

<!-- RELATED:END -->
