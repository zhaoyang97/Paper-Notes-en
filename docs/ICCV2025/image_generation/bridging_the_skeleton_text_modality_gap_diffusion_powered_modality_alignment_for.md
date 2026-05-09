---
title: >-
  [Paper Note] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition
description: >-
  [ICCV 2025][Image Generation][Zero-shot skeleton-based action recognition] This paper proposes TDSM (Triplet Diffusion for Skeleton-Text Matching), the first work to apply diffusion models to zero-shot skeleton-based action recognition (ZSAR). It achieves implicit alignment between skeleton features and text prompts through the reverse diffusion process, and introduces a triplet diffusion loss to enhance discriminability. TDSM substantially outperforms state-of-the-art methods on NTU-60/120 and PKU-MMD by margins ranging from 2.36% to 13.05%.
tags:
  - ICCV 2025
  - Image Generation
  - Zero-shot skeleton-based action recognition
  - diffusion models
  - cross-modal alignment
  - Triplet Loss
  - skeleton-text matching
date: 2026-05-08
content_hash: 6391710d7e37d902
---

# Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition

**Conference**: ICCV 2025  
**arXiv**: [2411.10745](https://arxiv.org/abs/2411.10745)  
**Code**: [https://kaist-viclab.github.io/TDSM_site](https://kaist-viclab.github.io/TDSM_site)  
**Area**: Image Generation / Action Recognition  
**Keywords**: Zero-shot skeleton-based action recognition, diffusion models, cross-modal alignment, Triplet Loss, skeleton-text matching

## TL;DR

This paper proposes TDSM (Triplet Diffusion for Skeleton-Text Matching), the first work to apply diffusion models to zero-shot skeleton-based action recognition (ZSAR). It achieves implicit alignment between skeleton features and text prompts through the reverse diffusion process, and introduces a triplet diffusion loss to enhance discriminability. TDSM substantially outperforms state-of-the-art methods on NTU-60/120 and PKU-MMD by margins ranging from 2.36% to 13.05%.

## Background & Motivation

The core challenge in ZSAR lies in the **modality gap between skeleton and text**. Skeleton data captures spatiotemporal motion patterns, while text descriptions encode high-level semantic information; the disparity between their feature spaces makes alignment difficult and severely limits generalization to unseen action classes.

Prior methods fall into two main categories: (1) VAE-based approaches (e.g., CADA-VAE, SynSE) that align skeleton and text latent spaces via VAEs; and (2) contrastive learning approaches (e.g., SMIE, PURLS, STAR) that align features through positive/negative sample pairs. However, all of these attempt to **directly align** skeleton and text features within their respective independent latent spaces, and the modality gap constrains generalization.

The authors' key insight is that diffusion models have demonstrated powerful cross-modal alignment capability in image-text generation — achieving precise cross-modal correspondence by incorporating text conditioning into the reverse denoising process. This raises the question: can this **conditional denoising alignment mechanism** (rather than generative capability) be repurposed to address skeleton-text alignment?

## Method

### Overall Architecture

TDSM consists of three stages: (1) extracting skeleton and text features using a pretrained skeleton encoder and CLIP text encoder, respectively; (2) conditioning on text features to denoise noisy skeleton features during the reverse diffusion process, establishing a unified skeleton-text latent space; and (3) applying a triplet diffusion loss to enhance alignment of correctly matched pairs while repelling mismatched pairs.

### Key Designs

1. **Skeleton and Text Embeddings**:
    - The skeleton encoder $\mathcal{E}_x$ (Shift-GCN or ST-GCN) is pretrained with cross-entropy on labeled data and then frozen, extracting skeleton features $\mathbf{z}_x \in \mathbb{R}^{M_x \times C}$.
    - The text encoder $\mathcal{E}_d$ uses CLIP, extracting global features $\mathbf{z}_g \in \mathbb{R}^{1 \times C}$ and local features $\mathbf{z}_l \in \mathbb{R}^{M_l \times C}$.
    - For each sample, both positive (ground-truth label) and negative (randomly incorrect label) text features are prepared.
    - Design Motivation: Leveraging the strong representational capacity of pretrained models concentrates TDSM's learning on the alignment task itself.

2. **Conditional Diffusion Alignment**:
    - Forward process: Gaussian noise is added to skeleton features: $\mathbf{z}_{x,t} = \sqrt{\bar{\alpha}_t} \mathbf{z}_x + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$.
    - Reverse process: A Diffusion Transformer $\mathcal{T}_{\text{diff}}$ predicts noise conditioned on global and local text features: $\hat{\boldsymbol{\epsilon}} = \mathcal{T}_{\text{diff}}(\mathbf{z}_{x,t}, t; \mathbf{z}_g, \mathbf{z}_l)$.
    - Key point: Rather than generating content, the conditional dependencies formed during denoising are exploited to **implicitly align** skeleton and text features.
    - $\mathcal{T}_{\text{diff}}$ is based on the DiT architecture, with reduced blocks and channels to match the small-scale nature of skeleton data.
    - Design Motivation: The conditional denoising in diffusion models naturally establishes fine-grained correspondence between the conditioning signal (text) and the target (skeleton).

3. **Triplet Diffusion (TD) Loss**:
    - Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \mathcal{L}_{\text{TD}}$
    - Standard diffusion loss: $\mathcal{L}_{\text{diff}} = \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2$, ensuring denoising accuracy for correctly matched pairs.
    - Triplet diffusion loss: $\mathcal{L}_{\text{TD}} = \max(\|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_p\|_2 - \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}_n\|_2 + \tau, 0)$
    - The model is encouraged to denoise correctly matched skeleton-text pairs accurately ($\hat{\boldsymbol{\epsilon}}_p$ close to $\boldsymbol{\epsilon}$) while failing to denoise mismatched pairs ($\hat{\boldsymbol{\epsilon}}_n$ far from $\boldsymbol{\epsilon}$).
    - Design Motivation: Introducing a discriminative learning signal into the diffusion framework, converting "denoising error" into a measure of matching quality.

### Inference Strategy

Inference uses **single-step inference** with fixed noise $\boldsymbol{\epsilon}_{\text{test}}$ and timestep $t_{\text{test}}=25$:
- For an unseen skeleton sequence and all candidate text labels, noise $\hat{\boldsymbol{\epsilon}}_k$ is predicted for each candidate.
- Predicted label: $\hat{y}^u = \arg\min_k \|\boldsymbol{\epsilon}_{\text{test}} - \hat{\boldsymbol{\epsilon}}_k\|_2$
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

- TDSM substantially outperforms state-of-the-art methods across all benchmark splits, with particularly pronounced advantages under extreme settings where unseen class proportions are large (30/30, 60/60).
- Both loss components are indispensable: the diffusion loss alone lacks discriminability, the triplet loss alone lacks denoising precision, and their combination is mutually complementary.
- The combination of global and local text features yields the best performance: global features provide holistic semantics while local features capture word-level details.
- Random noise in the diffusion process acts as a natural regularizer, preventing overfitting and improving generalization.
- The optimal inference timestep is $t_{\text{test}}=25$ (the midpoint of the 50-step schedule); performance degrades when the timestep is too small (denoising task is trivial) or too large (noise is too strong).

## Highlights & Insights

- **Novel perspective**: This is the first work to apply diffusion models to ZSAR, exploiting not their generative capacity but rather the cross-modal alignment capability inherent in the conditional denoising process.
- **Elegant Triplet Diffusion loss design**: The classical triplet loss concept is seamlessly integrated into the diffusion framework, using denoising error as a proxy for matching quality.
- **Efficient single-step inference**: No iterative denoising is required; a single forward pass suffices for matching, making inference computationally efficient.
- The **13.05% improvement** on NTU-120 96/24 is remarkably significant, demonstrating that diffusion-based alignment substantially surpasses traditional approaches.

## Limitations & Future Work

- Inference requires one forward pass per candidate label, leading to increased computational cost when the candidate label set is large.
- The skeleton encoder requires pretraining (albeit only on seen classes), which may introduce bias.
- Fixed inference noise introduces stochasticity (±2.5% variance), necessitating multi-run averaging.
- Integration with large-scale skeleton-text pretrained models remains unexplored.

## Related Work & Insights

- The idea of using diffusion models for discrimination rather than generation aligns with the directions of DiffSeg, DiffCut, and related works.
- Triplet loss variants are common in metric learning, but their integration into a diffusion framework is attempted here for the first time.
- Inspiration: Other zero-shot tasks requiring cross-modal alignment — such as zero-shot video understanding and audio-text matching — can potentially adopt this diffusion alignment paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of diffusion models to ZSAR; the triplet diffusion loss is both novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, multiple split settings, comprehensive ablation analysis, and variance analysis included.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich illustrations.
- Value: ⭐⭐⭐⭐⭐ Highly significant performance gains (up to 13%+), with strong practical value and broad inspirational potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[ICCV 2025\] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)

</div>

<!-- RELATED:END -->
