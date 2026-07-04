---
title: >-
  [Paper Note] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation
description: >-
  [CVPR 2025][Model Compression][Learned Image Compression] An end-to-end learned image compression framework, ARCHE, is proposed, which integrates hierarchical hyperprior, masked spatial autoregressive context, channel conditioning, and SE-excited channel recalibration into a unified probabilistic architecture. Without requiring Transformers or recurrent components, ARCHE reduces BD-Rate on Kodak by approximately 48% compared to the Ballé baseline and by about 5.6% compared to…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Learned Image Compression"
  - "Autoregressive Entropy Models"
  - "Hyperprior"
  - "Squeeze-and-Excitation"
  - "Residual Prediction"
date: 2026-05-08
content_hash: 86ad8937382765ef
---

# ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation

**Conference**: CVPR 2025  
**arXiv**: [2603.10188](https://arxiv.org/abs/2603.10188)  
**Code**: [sof-il/ARCHE](https://github.com/sof-il/ARCHE)  
**Area**: Model Compression / Learned Image Compression  
**Keywords**: Learned Image Compression, Autoregressive Entropy Models, Hyperprior, Squeeze-and-Excitation, Residual Prediction

## TL;DR

An end-to-end learned image compression framework, ARCHE, is proposed, which integrates hierarchical hyperprior, masked spatial autoregressive context, channel conditioning, and SE-excited channel recalibration into a unified probabilistic architecture. Without requiring Transformers or recurrent components, ARCHE reduces BD-Rate on Kodak by approximately 48% compared to the Ballé baseline and by about 5.6% compared to VVC Intra, with only 95M parameters and a 222ms decoding time.

## Background & Motivation

### Background
Learned image compression has evolved from fixed-transform algorithms to end-to-end trainable architectures. Under the variational autoencoder (VAE) framework, analysis transform, entropy model, and rate-distortion trade-off can be learned jointly. Significant milestones include the Hyperprior model by Ballé et al., the joint autoregressive + hyperprior model by Minnen et al., and the channel-wise autoregressive model by Minnen & Singh.

### Limitations of Prior Work
- **Trade-off between computational cost and performance**: Attention- or Transformer-based frameworks show excellent visual results but are difficult to deploy and suffer from slow inference.
- **Sequential decoding bottleneck**: Serialized entropy models (such as ConvLSTM) restrict parallel processing, resulting in slow inference speeds.
- **Limitations of single modeling approaches**: Pure spatial or pure channel autoregression both have limitations, necessitating a hybrid scheme.

### Key Challenge
Balancing model capacity, parameter efficiency, and practical feasibility is difficult. Complex models (Transformers, recurrent networks) perform well but suffer from slow inference; simple models are fast but lack compression efficiency.

### Goal
To enhance compression performance by better modeling dependencies while maintaining the efficiency of convolutional architectures, enabling fully convolutional networks to achieve rate-distortion performance close to state-of-the-art.

### Key Insight
Instead of pursuing larger and more complex models, this work deepens the interaction among entropy estimation, context dependency capturing, and adaptive feature recalibration.

### Core Idea
To unify hierarchical, spatial, and channel priors within a single probabilistic framework, combined with SE-excitation and residual refinement to enhance the quality of latent representations—"not by increasing architecture depth, but by better understanding dependency modeling".

## Method

### Overall Architecture
Based on the VAE structure: Analysis transform $g_a$ → Quantization → Entropy coding (Hyperprior + Masked Context + Channel Conditioning) → Arithmetic decoding → Synthesis transform $g_s$. The latent representation is divided into 10 slices and decoded sequentially, with each slice containing SE-excitation and LRP residual prediction.

### Key Designs

#### Key Design 1: Key Design 1: Masked Autoregressive Context Model (Masked PixelCNN)

- **Function**: Models fine-grained local spatial correlations in the latent space.
- **Mechanism**: Uses masked convolutions of PixelCNN to replace ConvLSTM. Type A masks exclude the current pixel and subsequent positions, while Type B masks include the current pixel. Multi-layer stacking expands the receptive field, and the sigmoid non-linearity preserves the causal structure.
- **Design Motivation**: ConvLSTM requires maintaining hidden states to achieve causality, which is computationally expensive and unstable to train. Masked convolutions directly guarantee causality through masking, allowing parallel computation across the spatial dimensions, which significantly speeds up inference.

#### Key Design 2: Channel Conditioning + Squeeze-and-Excitation

- **Function**: Captures residual correlations among channels and adaptively recalibrates channel responses.
- **Mechanism**:
    - Channel conditioning: When decoding the $c$-th channel, features from the previous $(c-1)$ channels are used for conditioning.
    - SE block: The squeeze step aggregates per-channel statistics through global average pooling: $s = \frac{1}{HW}\sum Y_{i,j}$; the excitation step learns channel attention weights: $w = \text{sig}(W_2 \cdot \text{relu}(W_1 \cdot s))$.
- **Design Motivation**: The channels output by the analysis transform are not statistically independent; different channels encode complementary structural or textural info. Channel-wise dependencies are typically smoother than spatial dependencies, allowing the module to be extremely lightweight.

#### Key Design 3: Latent Residual Prediction (LRP)

- **Function**: Estimates and compensates for quantization residual errors.
- **Mechanism**: Predicts the correction terms for the $m$-th slice as $\hat{y}'_m = \hat{y}_m + \lambda_{LRP} \cdot \text{softsign}(r_m)$, where $\lambda_{LRP}$ is a learnable scaling factor.
- **Design Motivation**: Quantization inevitably introduces noise that hyperprior and context models cannot fully correct. Using softsign instead of tanh provides smoother gradients and bounded outputs, yielding more stable training.

### Loss & Training
- Rate-distortion loss: $L = R + \lambda D$, where $D$ represents MSE.
- Trained separately for 8 lambda values: {0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1}.
- Adam optimizer, learning rate $10^{-4}$, 400 epochs, batch size 8.
- Training data: CLIC dataset, randomly cropped to 256×256.
- Latent representation depth of 320 split into 10 slices, hyperprior depth of 192, SE reduction ratio of 16.

## Key Experimental Results

### Main Results: Kodak BD-Rate (PSNR)

| Method | BD-Rate vs Ballé | BD-Rate vs VVC |
|------|-------------------|----------------|
| Minnen et al. | -8.00% | +90.61% |
| Minnen & Singh | -16.28% | +63.55% |
| WeConvene | -6.92% | +92.47% |
| Iliopoulou et al. (prior work) | -24.22% | +30.19% |
| **ARCHE (Ours)** | **-48.01%** | **-5.61%** |

### Tecnick BD-Rate (PSNR)

| Method | BD-Rate vs Ballé | BD-Rate vs VVC |
|------|-------------------|----------------|
| Minnen et al. | -8.81% | +79.04% |
| Minnen & Singh | -13.99% | +50.32% |
| **ARCHE (Ours)** | **-44.89%** | **-10.28%** |

### Computational Complexity Comparison

| Method | Parameters | Decoding Time / Image |
|------|--------|-------------|
| Ballé et al. | 11.7M | 25 ms |
| Minnen et al. | 95.8M | 591 ms |
| Minnen & Singh | 121.7M | 249 ms |
| Iliopoulou et al. | 124.3M | 265 ms |
| **ARCHE** | **95.4M** | **222 ms** |

### Ablation Study

| Variant | Effect |
|------|------|
| Number of Slices = 2 | BD-Rate savings of ~5% |
| Number of Slices = 10 | BD-Rate savings of over 11% (selected as final configuration) |
| Without Masked Context (no MCM) | Significant performance drop |
| Without Autoregressive Prior (no AR) | Largest performance drop (degenerates to pure Hyperprior) |
| Without SE Module (no SE) | Moderate quality drop at low bitrates |

### Key Findings
1. **Outperforming VVC Intra**: ARCHE achieves a BD-Rate reduction of 5.61% on Kodak and 10.28% on Tecnick compared to VVC Intra.
2. **Improvement over prior work**: Compared to Iliopoulou et al., Kodak BD-Rate improved from -24.22% to -48.01%, mainly attributed to replacing ConvLSTM with the masked context model + SE modules.
3. **Parameter efficiency advantage**: 95.4M parameters is 22% fewer than Minnen & Singh (121.7M), with faster decoding.
4. **Complementary modules**: Ablations confirm that the masked context, autoregressive prior, and SE blocks each contribute unique and complementary improvements.
5. **10 slices is the optimal trade-off**: Gains diminish beyond 10 slices.

## Highlights & Insights

1. **"Outperforming VVC without Transformers"**: A pure convolutional architecture achieves SOTA performance through meticulously designed entropy modeling combinations, challenging the notion that "larger and more complex is always better".
2. **Replacing ConvLSTM with masked convolutions**: Substantially improves parallelism and inference speed while preserving causality, serving as a key improvement over prior work.
3. **Systematic integration of multi-level priors**: Hyperprior (global) + masked context (local spatial) + channel conditioning (inter-channel) + SE (adaptive channel importance) forms a complete probabilistic modeling chain.
4. **The tip of using softsign instead of tanh**: Using softsign in LRP provides smoother gradients, contributing to training stability.
5. **Open-source code**: The GitHub repository is provided, facilitating reproducible research.

## Limitations & Future Work

1. **Optimized only for MSE**: Perceptual loss (such as LPIPS) is not utilized, leaving potential room for improvement in perceptual quality at low bitrates.
2. **Sequential decoding still exists**: Although masked convolution is faster than ConvLSTM, it is still not fully parallel; future work can explore block-level parallel context prediction.
3. **Small training set**: Models were trained solely on the CLIC dataset without verification on large-scale diverse datasets.
4. **Single GPU training**: Training on an RTX 3080 11GB might have constrained further optimization of model scale and training efficiency.
5. **No task-specific optimization**: Lacks joint compression-analysis optimization for downstream tasks like classification or segmentation.
6. **Delay introduced by slice dependencies**: Sequential decoding of 10 slices remains a bottleneck for inference speed.

## Related Work & Insights

- **Relationship with Ballé Hyperprior**: On top of Ballé, spatial autoregression, channel conditioning, and SE excitations are added, achieving a relative BD-Rate improvement of 48%.
- **Difference from Minnen & Singh channel-wise AR**: ARCHE employs both spatial and channel autoregression, whereas Minnen & Singh only utilize channel-dimension AR to gain parallelizability.
- **Positioning against Transformer schemes**: CNN-Transformer hybrid codecs (e.g., Liu et al.) are powerful but computationally expensive, whereas ARCHE demonstrates that pure convolutional schemes remain highly competitive.
- **Comparison with WeConvene wavelet method**: WeConvene shows only a -6.92% BD-Rate on Kodak relative to Ballé, which is far below ARCHE's -48.01%.
- **Insights for practical deployment**: Proves that under efficiency constraints, carefully crafted probabilistic modeling is more crucial than stacking parameters.

## Rating
- Novelty: ⭐⭐⭐ (Though individual components are not first of their kind, the integration method and combination of SE + masked convolution are novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Evaluation on both Kodak and Tecnick datasets, comprehensive ablations, computational complexity analysis, and visual comparisons)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, comprehensive related work, and complete mathematical derivations)
- Value: ⭐⭐⭐⭐ (Demonstrates the compression potential of convolutional architectures, marking a key milestone by outperforming VVC Intra)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Content-Adaptive Hierarchical Hyperprior for Neural Video Coding](../../CVPR2026/model_compression/content-adaptive_hierarchical_hyperprior_for_neural_video_coding.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[ICML 2026\] RQ-MoE: Residual Quantization via Mixture of Experts for Efficient Input-Dependent Vector Compression](../../ICML2026/model_compression/rq-moe_residual_quantization_via_mixture_of_experts_for_efficient_input-dependen.md)
- [\[ICCV 2025\] Bridging Continuous and Discrete Tokens for Autoregressive Visual Generation](../../ICCV2025/model_compression/bridging_continuous_and_discrete_tokens_for_autoregressive_visual_generation.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)

</div>

<!-- RELATED:END -->
