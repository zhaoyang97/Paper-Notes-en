---
title: >-
  [Paper Note] FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies
description: >-
  [NeurIPS 2025][Image Generation][Synthetic image detection] Grounded in Markov Random Field (MRF) theory, this paper proposes a Local Pixel Dependency (LPD) feature representation that exposes textural inconsistencies in generated images via median-filter reconstruction. Combined with FerretNet, a lightweight convolutional network with only 1.1M parameters, the approach achieves an average detection accuracy of 97.1% across 22 generative models while being trained exclusively on 4 categories of ProGAN data.
tags:
  - NeurIPS 2025
  - Image Generation
  - Synthetic image detection
  - local pixel dependency
  - Markov random field
  - lightweight network
  - cross-model generalization
date: 2026-05-08
content_hash: 105b7f39ba4adefd
---

# FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies

**Conference**: NeurIPS 2025
**arXiv**: [2509.20890](https://arxiv.org/abs/2509.20890)
**Code**: [https://github.com/xigua7105/FerretNet](https://github.com/xigua7105/FerretNet)
**Area**: Image Generation
**Keywords**: Synthetic image detection, local pixel dependency, Markov random field, lightweight network, cross-model generalization

## TL;DR

Grounded in Markov Random Field (MRF) theory, this paper proposes a Local Pixel Dependency (LPD) feature representation that exposes textural inconsistencies in generated images via median-filter reconstruction. Combined with FerretNet, a lightweight convolutional network with only 1.1M parameters, the approach achieves an average detection accuracy of 97.1% across 22 generative models while being trained exclusively on 4 categories of ProGAN data.

## Background & Motivation

As the capabilities of generative models such as VAEs, GANs, and LDMs continue to advance, distinguishing synthetic images from real ones becomes increasingly difficult. Existing detection methods face two key challenges:

(1) **Insufficient generalization**: Many methods rely on artifact features specific to particular models. Frequency-domain approaches (e.g., F3Net, FrePGAN) perform well on known models but struggle to generalize to unseen architectures; DIRE uses diffusion-based reconstruction for detection but performs poorly on GAN-generated content.

(2) **Low computational efficiency**: Methods based on large pretrained models (e.g., Ojha with frozen CLIP, FatFormer adapting CLIP) incur large parameter counts and slow inference speeds, making deployment in resource-constrained settings difficult.

The core insight of FerretNet is that despite architectural diversity across generative models, they share two unified artifact sources — latent variable distribution shift and the smoothing effect of the decoding process. These artifacts manifest as disruptions in local pixel dependency relationships and can be uniformly modeled under the MRF framework.

## Method

### Overall Architecture

The FerretNet pipeline consists of two stages: (1) LPD feature extraction — the image is reconstructed via median filtering, and a difference map is computed between the original and reconstructed images; (2) lightweight classification — the FerretNet network processes the LPD feature map and outputs a real/fake prediction. The entire pipeline totals only 1.1M parameters.

### Key Designs

1. **MRF-based Local Pixel Dependency (LPD) features**: Under the MRF assumption, the distribution of each pixel in a natural image depends only on its local neighborhood. For an $n \times n$ window, the center pixel is first set to zero (zero-masking, to prevent the generated pixel from contaminating the median computation), and the neighborhood median is then computed as the reconstructed value. The LPD feature map is the difference between the original image and its median-reconstructed counterpart: $\text{LPD} = I - I'$. Natural images exhibit strong local statistical consistency and thus yield LPD values close to zero, whereas generated images display significant anomalous patterns in the LPD at textural edges and color transitions. The physical intuition behind this design is that the pixel correlations in real images arise from optical physical processes (illumination, material interaction) that generative models cannot perfectly replicate at the level of underlying statistics.

2. **Ferret Block dual-path architecture**: The core of the network consists of 4 cascaded Ferret Blocks, each featuring a parallel dual-path structure. The main path employs $3\times3$ dilated group convolutions (dilation=2), expanding the effective receptive field to $5\times5$; the auxiliary path uses standard $3\times3$ group convolutions to capture fine-grained local patterns. The outputs of both paths are fused via $1\times1$ convolutions. Group convolutions combined with depthwise separable designs substantially reduce parameter count, while residual connections ensure stable gradient propagation. The overall design philosophy is to emulate the behavior of deep networks within a shallow architecture.

3. **Zero-masking median filtering strategy**: Conventional median filtering introduces ambiguity with even window sizes, and including the center pixel in the computation introduces autocorrelation. Zero-masking sets the center pixel to zero before computing the median, ensuring that the LPD purely reflects the neighborhood's predictive capability rather than the pixel's own information. This small design choice is critical to robustness.

### Loss & Training

- BCEWithLogitsLoss (binary cross-entropy with logits)
- Adam optimizer, lr=$2 \times 10^{-4}$, betas=(0.937, 0.999), weight decay=$5 \times 10^{-4}$
- Trained from scratch for 100 epochs, batch size 32
- Training data: only 4 categories of ProGAN-generated images (car, cat, chair, horse), paired with an equal number of real LSUN images
- Data augmentation: random crop to 224×224, random horizontal flip
- Test-time: center crop to 256×256

## Key Experimental Results

### Main Results

ForenSynths test set (8 GAN models), trained on only 4 ProGAN categories:

| Method | Params | ProGAN | StyleGAN | StyleGAN2 | BigGAN | Mean ACC/AP |
|--------|--------|--------|----------|-----------|--------|-------------|
| Wang et al. | - | 91.4/99.4 | 63.8/91.4 | 76.4/97.5 | 52.9/73.3 | 67.1/86.9 |
| Ojha (CLIP-based) | ~150M | 99.7/100 | 89.0/98.7 | 83.9/98.4 | 90.5/99.1 | 89.1/98.3 |
| FatFormer (CLIP-adapted) | ~150M | 99.9/100 | 97.2/99.8 | 98.8/99.9 | 99.5/100 | 98.4/99.7 |
| NPR | - | 99.8/100 | 96.3/99.8 | 97.3/100 | 87.5/94.5 | 92.5/96.1 |
| **FerretNet (1.1M)** | **1.1M** | 99.9/100 | 98.0/100 | 98.5/100 | 92.6/98.5 | **95.9/99.3** |

Diffusion-6-cls (6 diffusion model variants):

| Method | Mean ACC/AP |
|--------|-------------|
| FatFormer | 95.0/98.8 |
| SAFE | 94.5/99.1 |
| **FerretNet** | **96.9/99.6** |

Synthetic-Pop (6 recent high-fidelity models, including SDXL-Turbo and SD-3.5-Medium):

| Method | Openjourney | RealVisXL | SD-3.5-Medium | SDXL-Turbo | Mean |
|--------|------------|-----------|---------------|------------|------|
| FreqNet | 56.3/63.6 | 59.4/66.6 | 78.5/86.8 | 77.5/86.0 | 65.0/71.4 |
| NPR | 78.8/83.5 | 78.1/82.0 | 80.4/84.1 | 78.2/82.9 | 77.9/81.9 |
| FatFormer | 97.3/99.7 | 99.3/100 | 99.2/100 | 98.5/100 | 98.8/99.9 |
| **FerretNet** | 96.7/99.5 | 98.9/100 | 98.0/99.9 | 97.9/100 | **97.1/99.6** |

### Ablation Study

Throughput comparison (RTX 4090, batch=128, Synthetic-Aesthetic test set):

| Method | Params | Throughput (img/s) | Mean ACC |
|--------|--------|--------------------|----------|
| Ojha (CLIP) | ~150M | Low | 82.5 |
| FatFormer | ~150M | Low | 93.1 |
| **FerretNet** | **1.1M** | **High** | **91.5** |

FerretNet's parameter count is approximately 1/136 that of CLIP-based methods, while matching or surpassing their performance on most benchmarks.

### Key Findings

- LPD features exhibit strong cross-model generalization — trained on only 4 ProGAN categories, the model successfully detects 22 architectures spanning VAE, GAN, and LDM families.
- Detection accuracy remains above 97% on the latest high-fidelity diffusion models (SD 3.5, SDXL-Turbo, RealVisXL).
- FerretNet's 1.1M parameters offer substantially higher inference efficiency compared to CLIP-based methods with orders-of-magnitude more parameters such as FatFormer.
- LPD visualization maps intuitively illustrate the difference between natural and synthetic images: real images exhibit uniform and consistent textures, while synthetic images show clear structural residuals in fine-detail regions.

## Highlights & Insights

- **Unified theoretical perspective**: Grounded in MRF theory, the paper reveals a common vulnerability shared by all generative models — the inability to perfectly reproduce local pixel dependency relationships — providing an elegant and interpretable detection principle.
- **Extreme lightweight design**: 1.1M parameters achieves performance comparable to CLIP-based methods with 100× more parameters, making it suitable for edge deployment.
- The paper introduces Synthetic-Pop, a new benchmark comprising 6 recent generative models and 60,000 images, filling a gap in evaluation against the latest high-fidelity models.
- The combination of median filtering and difference maps — a "classical technique" — proves effective on a new task, serving as a reminder not to overlook traditional signal processing methods in the deep learning era.

## Limitations & Future Work

- Performance is relatively weaker on the BigGAN category (92.6% vs. FatFormer's 99.5%), possibly because BigGAN's latent space structure is more regular.
- CO-SPY uses different training data, making the comparison not entirely fair.
- Only image-level detection is validated; the method has not been extended to video or local forgery detection.
- The LPD window size $n$ is a fixed hyperparameter; an adaptive window selection strategy may further improve performance.
- Robustness to high-quality post-processing (e.g., JPEG compression, degradation from social media sharing) has not been thoroughly evaluated.

## Related Work & Insights

- **NPR (Neighbor Pixel Relations)**: Conceptually similar (neighborhood pixel relationships), but NPR focuses on upsampling patterns, whereas FerretNet's MRF-based median deviation has stronger theoretical grounding.
- **DIRE**: Detects synthetic images via diffusion-based reconstruction differences, but incurs extremely high computational cost and performs poorly on GAN-generated content; FerretNet's LPD extraction introduces virtually no computational overhead.
- **Frequency-domain methods (F3Net, BiHPF, FreqNet)**: Capture specific frequency characteristics but exhibit limited cross-model generalization.
- Key insight: The critical factor in detection is not larger models but better feature representations — LPD demonstrates that modeling physical priors is more valuable than brute-force fitting.

## Rating

⭐⭐⭐⭐ — Theoretically clear (MRF→LPD), extremely lightweight (1.1M), and outstanding generalization across 22 models. Highly notable for practical deployment value. The primary weaknesses are the performance gap on BigGAN and the unverified robustness against post-processing.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] PixPerfect: Seamless Latent Diffusion Local Editing with Discriminative Pixel-Space Refinement](pixperfect_seamless_latent_diffusion_local_editing_with_discriminative_pixel-spa.md)
- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [\[AAAI 2026\] Beyond Semantic Features: Pixel-Level Mapping for Generalized AI-Generated Image Detection](../../AAAI2026/image_generation/beyond_semantic_features_pixel-level_mapping_for_generalized_ai-generated_image_.md)
- [\[ICCV 2025\] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis](../../ICCV2025/image_generation/deepshield_fortifying_deepfake_video_detection_with_local_and_global_forgery_ana.md)
- [\[CVPR 2026\] Layer Consistency Matters: Elegant Latent Transition Discrepancy for Generalizable Synthetic Image Detection](../../CVPR2026/image_generation/layer_consistency_matters_elegant_latent_transition_discrepancy_for_generalizabl.md)

<!-- RELATED:END -->
