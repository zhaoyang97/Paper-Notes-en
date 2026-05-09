---
title: >-
  [Paper Note] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation
description: >-
  [CVPR 2026][Model Compression][Autoregressive entropy model] A fully convolutional architecture that unifies hierarchical hyperprior, Masked PixelCNN spatial autoregression, channel-conditional modeling, and SE channel excitation — without relying on Transformers or recurrent components. With 95M parameters and a 222ms decoding time, it achieves a 48% BD-Rate reduction over the Ballé baseline and outperforms VVC Intra by 5.6%.
tags:
  - CVPR 2026
  - Model Compression
  - Autoregressive entropy model
  - hyperprior
  - Squeeze-and-Excitation
  - residual prediction
  - rate-distortion optimization
date: 2026-05-08
content_hash: ec1e6247c053db33
---

# ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation

**Conference**: CVPR 2026
**arXiv**: [2603.10188](https://arxiv.org/abs/2603.10188)
**Code**: [GitHub](https://github.com/sof-il/ARCHE)
**Area**: Learned Image Compression
**Keywords**: Autoregressive entropy model, hyperprior, Squeeze-and-Excitation, residual prediction, rate-distortion optimization

## TL;DR

A fully convolutional architecture that unifies hierarchical hyperprior, Masked PixelCNN spatial autoregression, channel-conditional modeling, and SE channel excitation — without relying on Transformers or recurrent components. With 95M parameters and a 222ms decoding time, it achieves a 48% BD-Rate reduction over the Ballé baseline and outperforms VVC Intra by 5.6%.

## Background & Motivation

**Background**: End-to-end learned image compression has surpassed traditional codecs (JPEG/JPEG2000) by jointly optimizing analysis transforms, quantization, and entropy models for superior rate-distortion trade-offs. Transformer/attention architectures and hybrid entropy models have continually pushed the performance frontier.

**Limitations of Prior Work**: (1) Transformer/attention-based compression models incur heavy computation and slow inference, making deployment difficult; (2) ConvLSTM context models require maintaining hidden states across large regions and suffer from high latency due to strict serial decoding; (3) pure channel autoregression (Minnen & Singh) discards spatial local correlations, while pure spatial autoregression introduces decoding bottlenecks.

**Key Challenge**: The trade-off between modeling precision and computational efficiency — more expressive models better estimate the probability distribution of latent representations, but at the cost of escalating inference overhead and parameter counts.

**Goal**: Achieve state-of-the-art rate-distortion efficiency using a purely convolutional architecture, without relying on Transformers or recurrent components, while maintaining manageable parameter counts and inference speed.

**Key Insight**: Unify four complementary probabilistic modeling components — hierarchical hyperprior, masked spatial autoregression, channel-conditional modeling, and SE excitation — into a single VAE framework, where each component serves a distinct role rather than replacing the others.

**Core Idea**: Rather than pursuing larger and deeper models, the paper proposes more refined modeling of global, spatial, and channel-wise dependencies within a convolutional framework.

## Method

### Overall Architecture

ARCHE is built on a VAE framework: the analysis transform $g_a$ maps the input to a latent representation $y$, and the synthesis transform $g_s$ reconstructs the image from the quantized representation $\hat{y}$. The key contribution lies in the hierarchical design of the entropy model: the hyperprior provides global statistics → Masked PixelCNN context refines local probabilities → channel conditioning captures cross-channel dependencies → SE excitation adaptively weights channels → LRP corrects quantization error. The latent representation $y$ is split into 10 slices along the channel dimension for sequential decoding, each with dedicated conditional transforms and LRP submodules.

### Key Designs

1. **Autoregressive Hyperprior + Masked PixelCNN Context**
   - The hyper-analysis transform $h_a$ maps $y$ to side information $z$, which is quantized and transmitted; the hyper-synthesis transform $h_s$ reconstructs conditional prior parameters (mean $\mu$ and scale $\sigma$) from $\hat{z}$
   - The spatial autoregressive prior uses Masked PixelCNN to model $p(\hat{y}_i|\hat{y}_{<i}, \hat{z})$ in raster-scan order; Type A masks exclude the center and subsequent positions to ensure causality, while Type B masks include the center position
   - Stacked masked convolutional layers with sigmoid nonlinearities expand the receptive field, achieving more stable training and partially parallelizable inference compared to ConvLSTM
   - Hyperprior and context features are concatenated and passed through a parameter network (pointwise convolutions + small kernels + nonlinear activations) to produce the final Gaussian parameters

2. **Channel Conditioning + SE Excitation**
   - When decoding channel $c$, features from the preceding $c-1$ channels are used via lightweight convolutions to model the joint probability $p(\hat{y}_{i,c}|\hat{y}_{<i,c}, \hat{y}_{<c}, \hat{z})$, extending the dependency space from purely spatial to spatial + channel
   - Squeeze-and-Excitation blocks are embedded within slice transforms: global average pooling produces channel descriptors → FC (reduction ratio 16) → ReLU → FC → sigmoid gating, adaptively amplifying informative channels and suppressing redundant ones
   - Cross-channel dependencies are typically smoother than spatial dependencies, allowing the channel-conditional module to remain lightweight without sacrificing performance

3. **Latent Residual Prediction (LRP)**
   - A correction is predicted for each quantized slice: $\hat{y}'_m = \hat{y}_m + \lambda_{LRP} \cdot \text{softsign}(r_m)$
   - Softsign replaces tanh to provide smoother gradients and bounded outputs; $\lambda_{LRP}$ is a learnable scaling factor
   - LRP explicitly compensates for quantization noise that the hyperprior and context model cannot fully eliminate

### Loss & Training

$L = R + \lambda D$, where $R$ is the cross-entropy rate (including the prior contribution from $z$ and the conditional contribution from $y|z$), and $D$ is MSE. Models are trained on the CLIC dataset with random $256 \times 256$ crops normalized to $[0,1]$. Eight $\lambda$ values $\in \{0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1\}$ cover different rate points. Optimizer: Adam with $lr=10^{-4}$, 400 epochs, batch size 8. During training, quantization is approximated with additive uniform noise to maintain gradient flow. Latent depth: 320; 10 slices; hyperprior depth: 192; SE reduction ratio: 16. Implemented with TensorFlow 2.11 + TFC on an RTX 3080.

## Key Experimental Results

### Main Results

| Method | BD-Rate vs Ballé (Kodak) | BD-Rate vs VVC (Kodak) | Params | Decode Time |
|--------|--------------------------|------------------------|--------|-------------|
| Minnen et al. | -8.00% | +90.61% | 95.8M | 591ms |
| Minnen & Singh | -16.28% | +63.55% | 121.7M | 249ms |
| WeConvene | -6.92% | +92.47% | — | — |
| Iliopoulou et al. | -24.22% | +30.19% | 124.3M | 265ms |
| **ARCHE** | **-48.01%** | **-5.61%** | **95.4M** | **222ms** |

On the Tecnick dataset: ARCHE achieves -44.89% vs. Ballé and -10.28% vs. VVC Intra, with consistent trends.

### Ablation Study

| Variant | Effect |
|---------|--------|
| Remove all AR components | Degrades to pure hyperprior; largest performance drop |
| Remove Masked Context Model | Significant degradation; spatial context is critical for local probability estimation |
| Remove SE | Moderate drop at low bitrates; channel weighting is important for preserving fine-grained structure |
| Slices 2→10 | BD-Rate gain increases from ~5% to >11%; marginal returns beyond 10 |
| GMM vs. single Gaussian | No significant improvement; conditional modeling already captures latent statistics sufficiently |
| Checkerboard vs. PixelCNN | 58% faster training but worse rate-distortion (especially at low bitrates); inference is actually 15% slower |

### Key Findings

- Component contributions are complementary rather than redundant; removing all simultaneously causes cumulative performance degradation
- 10 slices is the optimal balance: further splitting yields diminishing returns while computational cost grows linearly
- ARCHE consistently outperforms VVC Intra on both Kodak and Tecnick datasets

## Highlights & Insights

- A purely convolutional architecture surpasses VVC Intra with better parameter efficiency and speed than most learned methods, providing strong evidence that carefully designed CNNs remain competitive
- Each component collaborates within a unified probabilistic framework (confirmed by ablations), embodying a design philosophy of "specialized modeling" over "single complex modules"
- Visual comparisons at low bitrates show sharper edges, more natural color transitions, and superior texture detail retention compared to VVC
- Replacing LSTM context with Masked PixelCNN simultaneously improves training stability and inference speed

## Limitations & Future Work

- A 222ms decoding time remains insufficient for real-time video; block-wise semi-parallel decoding strategies could be explored
- Optimization is limited to MSE; incorporating perceptual metrics (LPIPS/DISTS, etc.) could further improve visual fidelity
- Task-oriented compression (e.g., using compressed representations directly for classification or segmentation) is not explored
- Scalability and memory efficiency on higher-resolution images are not verified
- Evaluation is limited to natural image datasets; generalization to medical imaging, remote sensing, and other domains remains unknown

## Related Work & Insights

- **vs. Iliopoulou et al. [2025]**: ARCHE replaces LSTM context with Masked PixelCNN and adds SE excitation, achieving an additional ~24pp BD-Rate reduction, 29M fewer parameters, and 43ms faster decoding
- **vs. Minnen et al. [2018]**: Building on joint AR + hyperprior, ARCHE adds channel conditioning, SE, and LRP, reducing BD-Rate by an additional ~40pp while cutting decoding time from 591ms to 222ms
- **vs. WeConvene [ECCV24]**: The wavelet-domain approach performs more weakly (-6.92% vs. Ballé); ARCHE's spatial-domain joint modeling proves more effective
- **Design Philosophy**: The principle of "better dependency modeling over larger models" and the complementary multi-level prior combination (global/spatial/channel) are transferable to other probabilistic modeling tasks

## Rating

- Novelty: ⭐⭐⭐ — Individual components have prior precedents; the contribution lies in careful integration and engineering optimization
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Dual datasets + 6 baselines + complete ablations + visual comparisons + computational analysis + appendix variant analysis
- Writing Quality: ⭐⭐⭐⭐ — Detailed method derivation, rich tables and figures, transparent appendix
- Value: ⭐⭐⭐ — Demonstrates that carefully designed CNN-based compression remains competitive; practically valuable for real-world deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](rdvq_differentiable_vq_image_compression.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)

</div>

<!-- RELATED:END -->
