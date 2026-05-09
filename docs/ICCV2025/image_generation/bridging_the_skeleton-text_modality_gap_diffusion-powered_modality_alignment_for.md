---
title: >-
  [Paper Note] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition
description: >-
  [ICCV 2025][Image Generation][Zero-shot skeleton-based action recognition] This paper proposes TDSM (Triplet Diffusion for Skeleton-Text Matching), the first work to apply diffusion models to zero-shot skeleton-based action recognition (ZSAR). TDSM achieves implicit alignment between skeleton features and text prompts through the reverse diffusion process, and introduces a triplet diffusion loss to enhance discriminability. It substantially outperforms state-of-the-art methods on NTU-60/120 and PKU-MMD, with improvements ranging from 2.36% to 13.05%.
tags:
  - ICCV 2025
  - Image Generation
  - Zero-shot skeleton-based action recognition
  - diffusion model
  - cross-modal alignment
  - Triplet Loss
  - skeleton-text matching
date: 2026-05-08
content_hash: 8eda1e2d5f0a1ea6
---

# Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition

**Conference**: ICCV 2025  
**arXiv**: [2411.10745](https://arxiv.org/abs/2411.10745)  
**Code**: [https://kaist-viclab.github.io/TDSM_site](https://kaist-viclab.github.io/TDSM_site)  
**Area**: Image Generation / Action Recognition  
**Keywords**: Zero-shot skeleton-based action recognition, diffusion model, cross-modal alignment, Triplet Loss, skeleton-text matching

## TL;DR

This paper proposes TDSM (Triplet Diffusion for Skeleton-Text Matching), the first work to apply diffusion models to zero-shot skeleton-based action recognition (ZSAR). TDSM achieves implicit alignment between skeleton features and text prompts through the reverse diffusion process, and introduces a triplet diffusion loss to enhance discriminability. It substantially outperforms state-of-the-art methods on NTU-60/120 and PKU-MMD, with improvements ranging from 2.36% to 13.05%.

## Background & Motivation

- **State of the Field**: The core challenge in ZSAR lies in the **modality gap between skeleton and text**. Skeleton data captures spatiotemporal motion patterns, while text descriptions encode high-level semantic information. The divergence between their feature spaces makes alignment difficult, severely limiting generalization to unseen actions.

- **Limitations of Prior Work**: Prior methods fall into two categories: (1) VAE-based methods (CADA-VAE, SynSE, etc.) that align skeleton and text latent spaces via VAEs; and (2) contrastive learning methods (SMIE, PURLS, STAR, etc.) that perform alignment through positive/negative sample pairs. However, all these approaches attempt to **directly align** skeleton and text features within their respective independent latent spaces, and the modality gap constrains generalization.

- **Root Cause**: The modality gap between skeleton and text feature spaces limits the effectiveness of direct alignment strategies used by prior methods.

- **Paper Goals**: To leverage the conditional denoising mechanism of diffusion models—rather than their generative capacity—to achieve cross-modal alignment between skeleton and text representations.

- **Starting Point**: Diffusion models have demonstrated powerful cross-modal alignment capabilities in image-text generation, realizing precise cross-modal correspondence by incorporating text conditions into the reverse denoising process. The key insight is whether this **condition-guided denoising alignment mechanism** can be repurposed to address the skeleton-text alignment problem.

## Method

### Overall Architecture

TDSM consists of three stages: (1) a pretrained skeleton encoder and CLIP text encoder extract skeleton and text features respectively; (2) during the reverse diffusion process, noisy skeleton features are denoised conditioned on text features, establishing a unified skeleton-text latent space; (3) a triplet diffusion loss enhances alignment for correct pairings while pushing away incorrect ones.

### Key Designs

1. **Skeleton and Text Embeddings**:

    - The skeleton encoder $\mathcal{E}_x$ (Shift-GCN or ST-GCN) is first pretrained with cross-entropy on labeled data and then frozen, extracting skeleton features $\mathbf{z}_x \in \mathbb{R}^{M_x \times C}$.
    - The text encoder $\mathcal{E}_d$ uses CLIP, extracting global features $\mathbf{z}_g \in \mathbb{R}^{1 \times C}$ and local features $\mathbf{z}_l \in \mathbb{R}^{M_l \times C}$.
    - For each sample, both positive (ground-truth label) and negative (randomly incorrect label) text features are prepared.
    - **Design Motivation**: Leveraging the strong representational capacity of pretrained models concentrates TDSM's learning burden on the alignment task.

2. **Conditional Diffusion Alignment**:

    - Forward process: Gaussian noise is added to skeleton features as $\mathbf{z}_{x,t} = \sqrt{\bar{\alpha}_t} \mathbf{z}_x + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$.
    - Reverse process: The Diffusion Transformer $\mathcal{T}_{\text{diff}}$ predicts noise conditioned on global and local text features: $\hat{\boldsymbol{\epsilon}} = \mathcal{T}_{\text{diff}}(\mathbf{z}_{x,t}, t; \mathbf{z}_g, \mathbf{z}_l)$.
    - Key point: Rather than generation, the conditional dependency arising during denoising is exploited to **implicitly align** skeleton and text features.
    - $\mathcal{T}_{\text{diff}}$ is based on the DiT architecture, with reduced blocks and channels tailored to the small-scale nature of skeleton data.
    - **Design Motivation**: Conditional denoising in diffusion models naturally establishes fine-grained correspondences between the conditioning signal (text) and the target (skeleton).

3. **Triplet Diffusion (TD) Loss**:

    - Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{TD}}$
    - Standard diffusion loss: $\mathcal{L}_{\text{diff}} = \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2$, ensuring denoising accuracy for correct pairings.
    - Triplet diffusion loss: $\mathcal{L}_{\text{TD}} = \max(\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2 - \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_n\|_2 + \tau, 0)$
    - The model is encouraged to denoise accurately for correct skeleton-text pairings ($\hat{\boldsymbol{\epsilon}}_p$ close to $\boldsymbol{\epsilon}$) while failing to denoise for incorrect pairings ($\hat{\boldsymbol{\epsilon}}_n$ far from $\boldsymbol{\epsilon}$).
    - **Design Motivation**: Introducing a discriminative learning signal into the diffusion framework converts "denoising error" into a measure of matching quality.

### Inference Strategy

At inference, **single-step inference** is used with fixed noise $\boldsymbol{\epsilon}_{\text{test}}$ and timestep $t_{\text{test}}=25$:
- For an unseen skeleton sequence and all candidate text labels, noise $\hat{\boldsymbol{\epsilon}}_k$ is predicted for each candidate.
- The predicted label is $\hat{y}^u = \arg\min_k \|\boldsymbol{\epsilon}_{\text{test}} - \hat{\boldsymbol{\epsilon}}_k\|_2$.
- The candidate with the smallest denoising error is selected, indicating the best alignment with the skeleton sequence.

## Key Experimental Results

### Main Results (SynSE and PURLS Benchmarks — NTU-60/NTU-120)

| Method | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|--------|-------------|--------------|----------------|---------------|
| CADA-VAE | 76.84 | 28.96 | 59.53 | 35.77 |
| PURLS | 79.23 | 40.99 | 71.95 | 52.01 |
| SA-DVAE | 82.37 | 41.38 | 68.77 | 46.12 |
| STAR | 81.40 | 45.10 | 63.30 | 44.30 |
| **TDSM** | **86.49** | **56.03** | **74.15** | **65.06** |
| Gain | +4.12 | +9.93(!!!) | +2.20 | +13.05(!!!) |

### Ablation Study

| Configuration | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|---------------|-------------|--------------|----------------|---------------|
| $\mathcal{L}_{\text{diff}}$ only | 79.87 | 53.03 | 72.44 | 57.65 |
| $\mathcal{L}_{\text{TD}}$ only | 80.90 | 54.36 | 70.73 | 60.95 |
| $\mathcal{L}_{\text{diff}} + \mathcal{L}_{\text{TD}}$ | **86.49** | **56.03** | **74.15** | **65.06** |
| Global text $\mathbf{z}_g$ only | 83.41 | 51.50 | 70.14 | 61.90 |
| Local text $\mathbf{z}_l$ only | 83.33 | 52.63 | 69.95 | 62.10 |
| $\mathbf{z}_g + \mathbf{z}_l$ | **86.49** | **56.03** | **74.15** | **65.06** |

### Key Findings

- TDSM substantially outperforms state-of-the-art methods across all benchmark splits, with particularly pronounced advantages under extreme settings where the proportion of unseen classes is large (30/30, 60/60).
- Both loss components are indispensable: the diffusion loss alone lacks discriminability, the triplet loss alone lacks denoising precision, and their combination is complementary.
- Combining global and local text features yields the best results: global features provide holistic semantics while local features capture word-level details.
- The stochastic noise in the diffusion process acts as a natural regularizer, preventing overfitting and improving generalization.
- The optimal inference timestep is $t_{\text{test}}=25$ (midpoint of 50 total steps); performance degrades when the value is too small (denoising task too trivial) or too large (noise too strong).

## Highlights & Insights

- **Novel Perspective**: This is the first work to apply diffusion models to ZSAR, exploiting not their generative capacity but the cross-modal alignment capability inherent in the conditional denoising process.
- **Elegant Triplet Diffusion Loss Design**: The classical triplet loss is seamlessly integrated into the diffusion framework, with denoising error serving as a proxy for matching quality.
- **Efficient Single-Step Inference**: No iterative denoising is required; a single forward pass suffices for matching, yielding high inference efficiency.
- The **13.05% improvement** on the NTU-120 96/24 split is remarkable, demonstrating that diffusion-based alignment substantially surpasses conventional approaches.

## Limitations & Future Work

- Inference requires one forward pass per candidate label, leading to non-trivial computational cost when the candidate label set is large.
- The skeleton encoder requires pretraining on seen classes, which may introduce bias.
- Fixed inference noise introduces randomness (±2.5% variance), necessitating averaging over multiple runs.
- Integration with large-scale skeleton-text pretrained models remains unexplored.

## Related Work & Insights

- The use of diffusion models for discrimination rather than generation is consistent with the direction of works such as DiffSeg and DiffCut.
- Triplet loss variants are common in metric learning, but their integration into a diffusion framework is attempted here for the first time.
- **Inspiration**: Other zero-shot tasks requiring cross-modal alignment—such as zero-shot video understanding and audio-text matching—could benefit from this diffusion alignment paradigm.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of diffusion models to ZSAR; the triplet diffusion loss is both novel and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, multiple split settings, comprehensive ablation analysis, and variance analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Highly significant performance gains (up to 13%+), with strong practical value and broad inspirational impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GAP: Gaussianize Any Point Clouds with Text Guidance](gap_gaussianize_any_point_clouds_with_text_guidance.md)
- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [\[ICCV 2025\] DIIP: Diffusion Image Prior](diffusion_image_prior.md)
- [\[ICCV 2025\] SMGDiff: Soccer Motion Generation using Diffusion Probabilistic Models](smgdiff_soccer_motion_generation_using_diffusion_probabilistic_models.md)
- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)

<!-- RELATED:END -->
