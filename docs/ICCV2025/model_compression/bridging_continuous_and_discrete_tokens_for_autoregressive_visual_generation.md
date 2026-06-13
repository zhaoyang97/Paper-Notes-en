---
title: >-
  [Paper Note] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation
description: >-
  [ICCV 2025][Model Compression][Autoregressive generation] This paper proposes TokenBridge, which converts continuous tokens into discrete tokens by applying post-training dimension-wise quantization to pre-trained contin…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Autoregressive generation"
  - "visual tokenizer"
  - "post-training quantization"
  - "discrete-continuous bridging"
  - "dimension-wise autoregression"
date: 2026-05-08
content_hash: 59d83c96ae4618c5
---

# Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation

**Conference**: ICCV 2025
**arXiv**: [2503.16430](https://arxiv.org/abs/2503.16430)  
**Code**: [https://yuqingwang1029.github.io/TokenBridge](https://yuqingwang1029.github.io/TokenBridge)  
**Area**: Visual Generation / Model Compression
**Keywords**: Autoregressive generation, visual tokenizer, post-training quantization, discrete-continuous bridging, dimension-wise autoregression

## TL;DR

This paper proposes TokenBridge, which converts continuous tokens into discrete tokens by applying post-training dimension-wise quantization to pre-trained continuous VAE features. The approach preserves the high-fidelity representation capability of continuous tokens while enabling straightforward autoregressive modeling with standard cross-entropy loss, achieving generation quality on ImageNet 256×256 comparable to continuous methods.

## Background & Motivation

Autoregressive visual generation models face a fundamental dilemma: **discrete tokens vs. continuous tokens**. Discrete tokens (e.g., VQ, LFQ) can be directly modeled with cross-entropy loss, but the quantization process introduces gradient approximations that cause training instability, and limited codebook sizes constrain representation capacity. Continuous tokens (e.g., VAE latents) better preserve visual details, but require complex distribution modeling techniques (e.g., diffusion heads, GMM), increasing the complexity of the generation pipeline.

The authors observe: can one **simultaneously enjoy the high-quality representation of continuous tokens and the modeling simplicity of discrete tokens**? The key insight is to decouple quantization from tokenizer training — first train a continuous VAE to convergence, then obtain discrete tokens via post-training quantization.

## Method

### Overall Architecture

TokenBridge consists of two core components: (1) a post-training dimension-wise quantization strategy that converts pre-trained VAE continuous features into discrete tokens; and (2) an efficient dimension-wise autoregressive prediction mechanism that handles the exponentially large token space.

### Key Designs

1. **Post-Training Dimension-wise Quantization**:

    - Rather than introducing quantization during tokenizer training, quantization is applied channel-independently to continuous VAE features $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$ after the VAE is fully trained.
    - Two key properties of VAE features are exploited: (a) KL regularization bounds the value range; (b) the approximately Gaussian distribution enables non-uniform quantization.
    - Quantization procedure: each dimension is first normalized to $[-r, r]$ ($r=3$), then the distribution is partitioned into $B$ equiprobable intervals based on the standard normal CDF, with each interval represented by its conditional expectation $\gamma_i = \mathbb{E}[\xi | b_i \leq \xi < b_{i+1}]$.
    - During dequantization, discrete indices are mapped back to continuous values and fed directly into the pre-trained VAE decoder.
    - Design Motivation: avoids codebook collapse and gradient approximation issues of conventional VQ, with no additional trainable parameters.

2. **Dimension-wise Autoregressive Head**:

    - Dimension-wise quantization produces an exponentially large token space ($B^C$ combinations), making direct softmax classification infeasible.
    - Independently predicting each dimension in parallel ignores critical inter-dimensional dependencies (experiments show FID degrades from 1.94 to 15.7).
    - Solution: a lightweight autoregressive head is introduced at each spatial position, factorizing the joint distribution as: $p(\mathbf{q}) = \prod_{c=1}^{C} p(q^c | \mathbf{q}^{<c}, \mathbf{z})$
    - Each step reduces to a $B$-class classification problem, making computation tractable.
    - Design Motivation: transforms modeling over a large vocabulary space into a series of small classification problems while preserving critical inter-channel dependencies.

3. **Frequency-based Dimension Ordering**:

    - FFT is used to analyze the spectral characteristics of each dimension, with dimensions sorted by their low-frequency energy ratio.
    - Dimensions carrying more low-frequency (structural) information are generated first, followed by high-frequency (detail) dimensions.
    - Design Motivation: a structure-first, detail-later generation order improves overall coherence of generated images.

### Loss & Training

- Standard cross-entropy loss is used during training, without requiring complex distribution modeling.
- At inference, spatial autoregression first generates context features $\mathbf{z}$ for each position; the dimension-wise autoregressive head then predicts discrete indices channel by channel. Upon completing all channels for one spatial position, the token is immediately dequantized into a continuous feature and used as the conditional input for the next position.
- Temperature sampling and classifier-free guidance are employed.

## Key Experimental Results

### Main Results

| Method | Token Type | Loss | Params | FID↓ | IS↑ | Recall↑ |
|--------|-----------|------|--------|------|-----|---------|
| LlamaGen | Train-quantized discrete | CE | 3.1B | 2.18 | 263.3 | 0.58 |
| VAR | Train-quantized discrete | CE | 2.0B | 1.73 | 350.2 | 0.60 |
| MAR-L | Continuous | Diff. | 479M | 1.78 | 296.0 | 0.60 |
| MAR-H | Continuous | Diff. | 943M | 1.55 | 303.7 | 0.62 |
| **Ours-L** | Post-train quantized discrete | CE | 486M | **1.76** | 294.8 | 0.63 |
| **Ours-H** | Post-train quantized discrete | CE | 910M | **1.55** | **313.3** | **0.65** |

### Ablation Study

| Configuration | gFID↓ | IS↑ | Note |
|--------------|-------|-----|------|
| Parallel prediction | 15.7 | 158.5 | Ignores inter-dimensional dependencies; severely degraded quality |
| Autoregressive prediction | 1.94 | 306.1 | Captures dependencies; 8× FID improvement |
| $B=16$ quantization | 2.03 | 295.0 | Coarse-grained quantization |
| $B=64$ quantization | 1.94 | 306.1 | Fine-grained quantization is optimal |
| Default dimension order | 1.94 | 306.1 | Baseline |
| Frequency-based ordering | 1.89 | 307.3 | Marginal improvement |
| AR head 3M params | 2.88 | 277.3 | Minimal configuration still functional |
| AR head 94M params | 1.94 | 306.1 | Increasing capacity yields consistent gains |

### Key Findings

- Post-training quantization at $B=64$ achieves reconstruction quality (rFID=1.11) that fully matches the continuous VAE baseline.
- Dimension-wise autoregressive prediction yields approximately 8× FID improvement over parallel prediction, demonstrating that inter-channel dependencies are critical.
- Discrete tokens natively support confidence-guided generation, enabling images with clear foregrounds and clean backgrounds — an advantage not available to continuous methods.

## Highlights & Insights

- **Paradigm Inversion**: Conventional methods quantize during tokenizer training; this work reverses the approach — training a continuous tokenizer first and applying post-training quantization afterwards, cleanly decoupling the two objectives.
- **Elegant Handling of Exponential Space**: The enormous $64^{16}$ token space is decomposed into 16 independent 64-class classification problems via dimension-wise autoregression.
- The work demonstrates that **standard cross-entropy loss can match the generation quality of diffusion heads and GMMs**, substantially simplifying the autoregressive visual generation pipeline.

## Limitations & Future Work

- Dimension-wise autoregression increases the number of inference steps per spatial token (by a factor of $C$), which may impact generation speed.
- Validation is currently limited to ImageNet 256×256; extension to higher resolutions and text-conditioned generation remains unexplored.
- The gain from frequency-based ordering is marginal, suggesting that more optimal dimension ordering strategies may exist.

## Related Work & Insights

- The direct comparison with MAR (continuous tokens + diffusion head) is the most compelling: TokenBridge achieves comparable FID at equivalent parameter counts while being significantly simpler to train.
- The post-training quantization philosophy is analogous to PTQ techniques in LLMs (e.g., GPTQ); transferring this idea to visual tokenizers represents an interesting cross-domain innovation.
- The approach provides a viable path toward unified multimodal frameworks where visual and language tokens share a common cross-entropy modeling objective.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of post-training quantization and dimension-wise autoregression is novel; the paradigm-inversion design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablations are comprehensive, covering quantization granularity, prediction strategy, and AR head capacity.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear and figures are intuitive.
- Value: ⭐⭐⭐⭐ Offers a clean and efficient new paradigm for autoregressive visual generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](../../ICLR2026/model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)
- [\[NeurIPS 2025\] When Worse is Better: Navigating the Compression-Generation Trade-off in Visual Tokenization](../../NeurIPS2025/model_compression/when_worse_is_better_navigating_the_compression-generation_tradeoff_in_visual_to.md)
- [\[ICCV 2025\] B-VLLM: A Vision Large Language Model with Balanced Spatio-Temporal Tokens](b-vllm_a_vision_large_language_model_with_balanced_spatio-temporal_tokens.md)
- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](../../CVPR2026/model_compression/planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)

</div>

<!-- RELATED:END -->
