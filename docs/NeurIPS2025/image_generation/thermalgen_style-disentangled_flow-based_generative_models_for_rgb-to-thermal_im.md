---
title: >-
  [Paper Note] ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation
description: >-
  [NeurIPS 2025][Image Generation][RGB-to-Thermal Translation] This paper proposes ThermalGen, an adaptive flow-based generative model that achieves, for the first time…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "RGB-to-Thermal Translation"
  - "Flow-Based Generative Models"
  - "Style Disentanglement"
  - "Multi-Dataset Joint Training"
  - "Thermal Imaging"
date: 2026-05-08
content_hash: fec1e66d183093bc
---

# ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation

**Conference**: NeurIPS 2025
**arXiv**: [2509.24878](https://arxiv.org/abs/2509.24878)  
**Code**: [Project Page](https://xjh19971.github.io/ThermalGen)  
**Area**: Image Generation / Cross-Modal Translation
**Keywords**: RGB-to-Thermal Translation, Flow-Based Generative Models, Style Disentanglement, Multi-Dataset Joint Training, Thermal Imaging

## TL;DR

This paper proposes ThermalGen, an adaptive flow-based generative model that achieves, for the first time, high-fidelity RGB-to-Thermal image translation across diverse viewpoints, sensors, and environmental conditions via an RGB-conditioned architecture and a style disentanglement mechanism. Three new large-scale satellite–aerial RGB-T paired datasets are also released.

## Background & Motivation

Visual–thermal sensor fusion is critical under challenging conditions such as low illumination and adverse weather; however, the scarcity of paired RGB-T data severely constrains related research. RGB-to-Thermal image translation can synthesize thermal images from abundant RGB data, offering three key advantages:

**Perfect Alignment**: Synthesized thermal images share pixel-level correspondence with source RGB images, making them suitable for fine-grained tasks such as dense feature matching.

**Scalable Data**: Massive publicly available RGB data can be leveraged, far exceeding the scale achievable through hardware-based RGB-T collection.

**Diversity Simulation**: Varying thermal characteristics and environmental conditions can be simulated from a single RGB input, improving the robustness of downstream models.

Nevertheless, existing methods face severe challenges:

- **Narrow Training Data**: GAN-based methods are typically trained on small, single-dataset settings with poor generalization.
- **Lack of Thermal Cues in RGB**: Models must infer thermal information from semantic content alone.
- **Large Domain Gaps**: Significant distribution shifts exist across different thermal sensors, camera viewpoints, and environmental conditions.

The key innovation of ThermalGen lies in a **style disentanglement mechanism** that encodes dataset-specific RGB-T mapping relationships as learnable style embeddings, enabling a single model to handle multiple RGB-T styles simultaneously.

## Method

### Overall Architecture

ThermalGen is built upon the SiT (Scalable Interpolant Transformer) architecture and performs flow-based generation in a latent space. Given an RGB image and a dataset-specific style embedding, the model predicts the velocity field of the thermal image latent variable, reconstructing the thermal image through a thermal image decoder after $T=50$ denoising steps.

### Key Designs

1. **Thermal Image Encoder–Decoder**

   A latent diffusion framework is adopted: the thermal image encoder $E_T$ compresses thermal images into latent variables $\mathbf{z}_T \in \mathbb{R}^{\frac{H}{f}\times\frac{W}{f}\times C}$, and the decoder $D_T$ reconstructs the thermal image. The RGB encoder uses a pretrained KL-VAE encoder $E_{\text{RGB}}$ to extract RGB latent representations $\mathbf{z}_{\text{RGB}}$.

   Flow-based generation operates in the latent space:

   $$\mathbf{z}_t = \alpha_t \mathbf{z}_0 + \sigma_t \boldsymbol{\epsilon}, \quad \alpha_t = 1-t, \quad \sigma_t = t$$

   The training objective for the velocity function:

   $$\mathcal{L}_{\text{flow}} = \mathbb{E}_{\mathbf{z}_t, t}\left[\|v_\theta(\mathbf{z}_t, t) - v(\mathbf{z}_t, t)\|^2\right]$$

2. **Style-Disentangled Mechanism**

   A set of learnable style embeddings $Y = \{\mathbf{y}_0, \mathbf{y}_1, \ldots, \mathbf{y}_n, \mathbf{y}_{\text{un}}\}$ is defined, where $n$ is the number of datasets and $\mathbf{y}_{\text{un}}$ is an unconditional style embedding (dimension 1024).

   Style embeddings are injected into the model via **adaLN-Zero conditioning**: given style embedding $\mathbf{y}_i$ and timestep $t$, a conditional embedding $\mathbf{c}_{\mathbf{y}_i, t}$ is produced to modulate the scale and shift parameters of adaptive layer normalization.

   During training, either a dataset-specific or unconditional embedding is randomly selected, enabling Classifier-Free Guidance (CFG). New datasets require only the addition of new style embeddings without retraining.

   **Design Motivation**: Inspired by AdaIN—modifying normalization parameters effectively enables style transfer. RGB-T mapping relationships differ substantially across datasets (sensor, viewpoint, time), and disentangling this "style" from model parameters allows a single model to adapt to diverse scenarios.

3. **RGB Image Conditioning Architecture**

   Two variants are explored:
   - **Multi-Head Cross-Attention (Cross-Attn)**: $\mathbf{z}_{\text{RGB}}$ serves as the query and $\hat{\mathbf{z}}_{t,T}$ as the key and value.
   - **Concatenation**: $\hat{\mathbf{z}}_{t,T}$ and $\mathbf{z}_{\text{RGB}}$ are directly concatenated as the SiT input.

   Experiments demonstrate that concatenation yields consistently better FID and facilitates fine-tuning from pretrained SiT weights.

### Loss & Training

- Standard flow matching loss is used for training.
- During joint training, batches are randomly sampled from all training sets.
- Training images are randomly cropped to 256×256; evaluation images are resized to 256×256.
- Approximately 200K training samples are drawn from 11+ datasets in total.
- Inference uses 50 denoising steps.

## Key Experimental Results

### Main Results (Satellite–Aerial Datasets)

| Method | Type | Boson-night FID↓ | Bosonplus-day FID↓ | Bosonplus-night FID↓ |
|------|------|-----------------|-------------------|---------------------|
| pix2pix | GAN | 149.55 | 170.45 | 137.74 |
| pix2pixHD | GAN | 106.33 | 157.65 | 89.26 |
| VQGAN | GAN | 207.12 | 185.41 | 286.74 |
| DiffV2IR | Diffusion | 150.11 | 215.20 | 96.42 |
| **ThermalGen-L/2** | **Flow** | **161.22** | **76.91** | **75.80** |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| SiT-B vs SiT-L vs SiT-XL | FID decreasing | Larger Transformer yields better generation quality |
| Patch size 2 vs 4 vs 8 | patch=2 optimal | Finer patch granularity improves image quality |
| Cross-Attn vs Concatenation | Concatenation achieves lower FID | Concatenation consistently outperforms |
| Unconditional vs Conditional vs CFG | CFG optimal (distinctive-style datasets) | Style embeddings significantly benefit datasets with unique styles |
| Boson-night CFG scale 1→8 | FID: 161.22→116.46 | CFG tuning substantially improves low-contrast scenes |
| FLIR CFG scale 1→4 | FID: 70.09→63.43 | CFG also effective under extreme lighting conditions |

### Key Findings

1. ThermalGen achieves state-of-the-art or near-state-of-the-art perceptual quality (FID, LPIPS) on most datasets, with particularly substantial gains on Bosonplus and NII-CU.
2. Style embeddings yield significant improvements on datasets with distinctive RGB-T styles; gains are smaller on general datasets (M3FD, MSRS), possibly because such styles are already encoded in model parameters.
3. GAN-based methods commonly produce distortions or grid artifacts; DiffV2IR tends to generate overly sharp boundaries.
4. The DDIM baseline tends to generate random samples close to the training distribution rather than conditioned outputs, highlighting ThermalGen's effective RGB conditioning.
5. Suboptimal performance on datasets such as LLVIP is primarily attributable to training/test distribution mismatch (verified via t-SNE), which can be resolved by expanding the training data.

## Highlights & Insights

- **The first generalizable RGB-T translation model** spanning cross-viewpoint, cross-sensor, and cross-environment settings, covering satellite–aerial, aerial, and ground-level categories.
- The style disentanglement design is elegant and practical—new datasets require only the addition of embeddings rather than full model retraining.
- Three new datasets (DJI-day, Bosonplus-day, Bosonplus-night) broaden the data foundation for RGB-T research.
- CFG scale serves as an inference-time control mechanism that can effectively mitigate performance issues on specific datasets.

## Limitations & Future Work

- Performance is suboptimal on Boson-night (low contrast), LLVIP (distribution shift), and FLIR (extreme lighting).
- The model assumes spatial resolution consistency between RGB and thermal images; resolution mismatch scenarios are not addressed.
- Style embeddings operate at the dataset level rather than the scene level, leaving intra-dataset style variation unmodeled.
- Evaluation is conducted only at 256×256 resolution; high-resolution scenarios remain unvalidated.

## Related Work & Insights

- adaLN-Zero style conditioning is generalizable to other multi-domain image translation tasks.
- The RGB-T translation paradigm is extensible to other cross-modal tasks (e.g., RGB-to-depth, RGB-to-SAR).
- Joint training on large-scale heterogeneous datasets is a key strategy for improving generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Style disentanglement combined with flow-based RGB-T translation is a novel combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 11+ datasets with thorough ablation studies and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with detailed quantitative results.
- **Value**: ⭐⭐⭐⭐ Provides a practical and scalable solution for RGB-T cross-modal translation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](../../ICCV2025/image_generation/scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[NeurIPS 2025\] Gradient Variance Reveals Failure Modes in Flow-Based Generative Models](gradient_variance_reveals_failure_modes_in_flow-based_generative_models.md)
- [\[NeurIPS 2025\] EditInfinity: Image Editing with Binary-Quantized Generative Models](editinfinity_image_editing_with_binary-quantized_generative_models.md)
- [\[NeurIPS 2025\] BitMark: Watermarking Bitwise Autoregressive Image Generative Models](bitmark_watermarking_bitwise_autoregressive_image_generative_models.md)
- [\[NeurIPS 2025\] Towards General Modality Translation with Contrastive and Predictive Latent Diffusion Bridge](towards_general_modality_translation_with_contrastive_and_predictive_latent_diff.md)

</div>

<!-- RELATED:END -->
