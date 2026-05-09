---
title: >-
  [Paper Note] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation
description: >-
  [CVPR 2026][Model Compression][Learned Image Compression] This paper proposes ARCHE, an end-to-end image compression framework that, within a purely convolutional architecture free of Transformers and recurrent modules, integrates five complementary components — hierarchical hyperprior, Masked PixelCNN spatial autoregressive context, channel conditioning, SE channel recalibration, and latent residual prediction — achieving a 48% BD-Rate reduction over the Ballé baseline and −5.6% over VVC Intra on Kodak, with only 95M parameters and 222ms decoding time.
tags:
  - CVPR 2026
  - Model Compression
  - Learned Image Compression
  - Autoregressive Prior
  - Hyperprior
  - Squeeze-and-Excitation
  - Latent Residual Prediction
date: 2026-05-08
content_hash: d94529b682853654
---

# ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation

**Conference**: CVPR 2026
**arXiv**: [2603.10188](https://arxiv.org/abs/2603.10188)
**Code**: [https://github.com/sof-il/ARCHE](https://github.com/sof-il/ARCHE)
**Area**: Model Compression
**Keywords**: Learned Image Compression, Autoregressive Prior, Hyperprior, Squeeze-and-Excitation, Latent Residual Prediction

## TL;DR

This paper proposes ARCHE, an end-to-end image compression framework that, within a purely convolutional architecture free of Transformers and recurrent modules, integrates five complementary components — hierarchical hyperprior, Masked PixelCNN spatial autoregressive context, channel conditioning, SE channel recalibration, and latent residual prediction — achieving a 48% BD-Rate reduction over the Ballé baseline and −5.6% over VVC Intra on Kodak, with only 95M parameters and 222ms decoding time.

## Background & Motivation

Learned image compression has recently surpassed traditional coding standards (JPEG, JPEG 2000) through joint end-to-end optimization of the analysis transform, quantization, and entropy model. Current state-of-the-art methods face a fundamental tension: **the trade-off between model representational capacity and computational efficiency**.

- Transformer- and attention-based methods offer strong global modeling capability but are large, slow to infer, and difficult to deploy.
- Spatial autoregressive models based on ConvLSTM can precisely capture local dependencies but suffer from severe sequential bottlenecks due to element-wise decoding order.
- Pure channel autoregressive methods (Minnen & Singh) improve parallelism but sacrifice fine-grained spatial dependency modeling.

**Key Insight**: Rather than pursuing architectural complexity, ARCHE deepens the combined modeling of multiple statistical dependencies within a purely convolutional framework. The **Core Idea** is that "the synergy of complementary dependency modeling outperforms any single complex architecture."

## Method

### Overall Architecture

ARCHE adopts a variational autoencoder (VAE) structure: the analysis transform $g_a$ encodes image $x$ into a latent representation $y$, which is quantized and transmitted via entropy coding; the synthesis transform $g_s$ reconstructs $\hat{x}$ from the quantized representation $\hat{y}$. The entropy model follows a hierarchical design — the hyperprior provides global statistics, channel conditioning and Masked PixelCNN progressively refine probability estimates, and latent residual prediction compensates for quantization noise. The optimization objective is the rate-distortion loss $L = R + \lambda D$.

### Key Designs

1. **Autoregressive Hyperprior**:

    - **Function**: Captures global statistical variation in the latent space.
    - **Mechanism**: The hyper-analysis transform $h_a(y; \phi_h)$ maps $y$ to a second-level latent variable $z$, which is transmitted as side information; the hyper-synthesis transform outputs conditional prior parameters. A spatial autoregressive prior is introduced, modeling dependencies via masked convolutions: $p(\hat{y}|\hat{z}) = \prod_i p(\hat{y}_i | \hat{y}_{<i}, \hat{z})$
    - **Design Motivation**: A factorized prior assumes conditional independence among latent elements and fails to capture spatial correlations arising from the limited receptive field of convolutions.

2. **Masked PixelCNN Context Model**:

    - **Function**: Refines entropy estimation by exploiting the spatial local structure of the latent representation.
    - **Mechanism**: Causal convolutions (Type A/B masks) based on PixelCNN utilize only the upper and left neighborhoods in raster-scan order to predict conditional distribution parameters at each position. Multiple masked convolution layers are stacked to enlarge the receptive field.
    - **Design Motivation**: Compared to ConvLSTM, masked convolutions enable parallel computation in a single forward pass, **significantly reducing computational overhead and training instability**.

3. **Channel Conditioning (CC)**:

    - **Function**: Models statistical co-occurrence relationships across channels.
    - **Mechanism**: The latent tensor is divided into $C$ channel slices decoded in causal order; already-decoded channel features are processed through a lightweight convolutional stack to extract cross-channel statistical patterns.
    - **Design Motivation**: Cross-channel dependencies are typically low-frequency and smooth, making them effectively capturable with a lightweight network.

4. **Squeeze-and-Excitation Channel Recalibration**:

    - **Function**: Adaptively re-weights channel responses within slice transforms.
    - **Mechanism**: Squeeze obtains channel descriptors via global average pooling; Excitation learns channel-wise attention weights through two FC layers: $w = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot s))$
    - **Design Motivation**: SE enables the network to concentrate capacity on more informative channels with negligible parameter overhead.

5. **Latent Residual Prediction (LRP)**:

    - **Function**: Compensates for irreversible noise introduced by quantization.
    - **Mechanism**: A residual correction term $r_m$ is predicted and applied with bounded correction via softsign activation: $\hat{y}'_m = \hat{y}_m + \lambda_{LRP} \cdot \text{softsign}(r_m)$
    - **Design Motivation**: Softsign yields smoother gradients than tanh, leading to more stable training.

### Loss & Training

- Rate-distortion loss $L = R + \lambda D$, with MSE as the distortion metric $D$.
- Eight $\lambda$ values spanning near-lossless to high-compression operating points.
- Trained on the CLIC dataset with random 256×256 crops, 400 epochs, batch size 8, Adam optimizer with lr=1e-4.
- Latent depth of 320 divided into 10 slices; hyperprior depth of 192; SE reduction ratio of 16.

## Key Experimental Results

### Main Results: Kodak BD-Rate Comparison (PSNR)

| Method | BD-Rate vs. Ballé (%) | BD-Rate vs. VVC (%) |
|---|---|---|
| Minnen et al. | −8.00 | +90.61 |
| Minnen & Singh | −16.28 | +63.55 |
| WeConvene | −6.92 | +92.47 |
| Iliopoulou et al. (prior work) | −24.22 | +30.19 |
| **ARCHE** | **−48.01** | **−5.61** |

### Ablation Study: Contribution of Each Component

| Configuration | BD-Rate Change | Notes |
|---|---|---|
| Full ARCHE | Optimal baseline | All 5 components combined |
| w/o AR + MCM | Largest performance drop | Degenerates to pure hyperprior model |
| w/o MCM | Significant drop | Spatial context modeling is critical |
| w/o SE | Moderate drop at low bitrate | Channel recalibration benefits fine-grained structure |
| 10 slices vs. 1 slice | ~11% BD-Rate improvement | More slices help, with diminishing returns |

### Key Findings

- ARCHE is the first learned codec to surpass VVC Intra within a purely convolutional framework (BD-Rate −5.61%).
- With 95M parameters and 222ms decoding time, it is lighter than Minnen & Singh (121.7M, 249ms) and the prior work (124.3M, 265ms).
- At low bitrates, ARCHE preserves sharper textures and more natural color transitions.
- Replacing ConvLSTM with Masked PixelCNN improves decoding speed and training stability.

## Highlights & Insights

- **Design philosophy of "complementarity over complexity"**: The five components each address distinct levels of statistical redundancy; their combined effect far exceeds that of any single powerful module.
- **Masked PixelCNN vs. ConvLSTM**: Maintains causality while enabling parallel training.
- **SE: low cost, high return**: Plug-and-play and transferable to other compression frameworks.

## Limitations & Future Work

- Optimization is limited to MSE; perceptual losses are not employed.
- Performance on high-resolution images (4K) has not been evaluated.
- Implementation is restricted to TF 2.11 / TFC library.
- No direct comparison with recent Transformer-hybrid methods is provided.

## Related Work & Insights

- The Ballé hyperprior model serves as the foundation of learned compression; ARCHE validates a "progressive enhancement" strategy by stacking four complementary components on top of it.
- WeConvene's wavelet-domain autoregression is complementary to ARCHE's spatial-domain autoregression.

## Rating

- **Novelty**: ⭐⭐⭐ — Individual components are not entirely novel, but their combination and complementarity analysis offer meaningful contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Dual-dataset evaluation (Kodak + Tecnick) with BD-Rate, visual quality, ablation, and computational cost analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with thorough motivation and derivation for each component.
- **Value**: ⭐⭐⭐⭐ — Demonstrates that purely convolutional architectures remain competitive with state-of-the-art methods while being more practical.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] Markovian Scale Prediction: A New Era of Visual Autoregressive Generation](markovian_scale_prediction_a_new_era_of_visual_autoregressive_generation.md)
- [\[CVPR 2026\] UniComp: Rethinking Video Compression Through Informational Uniqueness](unicomp_rethinking_video_compression_through_informational_uniqueness.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)

<!-- RELATED:END -->
