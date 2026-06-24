---
title: >-
  [Paper Note] Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation
description: >-
  [NeurIPS 2025][Image Generation][Latent diffusion models] This paper proposes Discriminative Vicinity Diffusion (DVD), which for the first time employs latent diffusion models for discriminative knowledge transfer. By training a diffusion model within the vicinity latent space of source-domain features to generate source-style cues, DVD enables domain adaptation without access to source data, surpassing state-of-the-art methods on standard SFDA benchmarks.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Latent diffusion models"
  - "source-free domain adaptation"
  - "privacy preservation"
  - "discriminative transfer"
  - "vicinity guidance"
date: 2026-05-08
content_hash: d52b790d159e3e87
---

# Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation

**Conference**: NeurIPS 2025
**arXiv**: [2510.00478](https://arxiv.org/abs/2510.00478)  
**Code**: None  
**Area**: Domain Adaptation / Image Generation
**Keywords**: Latent diffusion models, source-free domain adaptation, privacy preservation, discriminative transfer, vicinity guidance

## TL;DR

This paper proposes Discriminative Vicinity Diffusion (DVD), which for the first time employs latent diffusion models for discriminative knowledge transfer. By training a diffusion model within the vicinity latent space of source-domain features to generate source-style cues, DVD enables domain adaptation without access to source data, surpassing state-of-the-art methods on standard SFDA benchmarks.

## Background & Motivation

Source-Free Domain Adaptation (SFDA) is increasingly important due to privacy protection requirements — source data is inaccessible, and only a pre-trained classifier is available. Limitations of existing methods:

**Implicit knowledge transfer**: Most methods exploit source knowledge only indirectly via pseudo-labels or consistency regularization.

**Lack of explicit decision boundary transfer**: Source-domain decision boundary information is lost during adaptation.

**Untapped potential of diffusion models**: LDMs are predominantly used for generative tasks; their application to discriminative transfer remains largely unexplored.

Core innovation: leveraging diffusion models as a privacy-preserving bridge to explicitly transfer source-domain decision boundaries to the target domain.

## Method

### Overall Architecture

The DVD framework consists of three stages:
1. **Source-domain training**: Train a classifier together with an auxiliary diffusion module on source data.
2. **Diffusion module release**: Release only the pre-trained diffusion module (no raw data is exposed).
3. **Target-domain adaptation**: Use the frozen diffusion module to generate source-style cues and align the target encoder.

### Key Designs

1. **Vicinity Encoding**:

    - For each source feature, identify its $k$-nearest neighbors.
    - Fit a Gaussian prior $\mathcal{N}(\mu_k, \Sigma_k)$ over the neighborhood.
    - The diffusion network learns to drift noisy samples back to label-consistent representations.

2. **Label-Guided Diffusion**:

    - Label information is encoded into the latent vicinity region of each feature.
    - The diffusion process maintains label consistency.
    - Key constraint: drifted samples should remain close to the class centroid of the original label.

3. **Target-Domain Adaptation**:

    - Sample from the vicinity region of target features.
    - Generate source-style cues through the frozen diffusion module.
    - Align the target encoder with diffusion-generated cues via InfoNCE loss.
    - Explicitly transfer decision boundaries.

### Loss & Training

- Source-domain diffusion training: standard DDPM loss + label consistency regularization.
- Target-domain adaptation:
$$\mathcal{L}_{\text{adapt}} = \mathcal{L}_{\text{InfoNCE}}(f_T(x_T), g_\theta(z_T)) + \lambda \mathcal{L}_{\text{pseudo}}$$

## Key Experimental Results

### Main Results (SFDA Benchmarks)

| Method | Office-Home Avg ↑ | VisDA-C ↑ | DomainNet Avg ↑ | Source Data Required |
|------|------------------|----------|----------------|---------|
| SHOT | 71.8 | 82.9 | 43.5 | No |
| NRC | 72.5 | 83.5 | 44.2 | No |
| AaD | 73.8 | 84.8 | 45.8 | No |
| CoWA | 74.2 | 85.1 | 46.5 | No |
| PLUE | 75.1 | 85.8 | 47.2 | No |
| **DVD (Ours)** | **77.5** | **87.8** | **49.8** | **No** |
| Oracle (w/ source data) | 79.2 | 89.5 | 52.3 | Yes |

### Additional Capability Evaluation

| Scenario | Baseline Acc. ↑ | +DVD Acc. ↑ | Gain |
|---------|-----------|------------|-----|
| Source classifier enhancement | 85.2 | 87.5 | +2.3 |
| Supervised classification | 78.5 | 80.8 | +2.3 |
| Domain generalization | 72.3 | 75.1 | +2.8 |

### Ablation Study

| Component | Office-Home ↑ | VisDA-C ↑ |
|------|-------------|---------|
| DVD (full) | 77.5 | 87.8 |
| w/o vicinity encoding | 73.8 | 84.2 |
| w/o label guidance | 74.5 | 85.1 |
| w/o InfoNCE alignment | 72.1 | 83.5 |
| GAN replacing diffusion | 74.2 | 84.8 |
| Random sampling (non-vicinity) | 71.5 | 82.5 |

### Key Findings

1. DVD narrows the gap between SFDA and source-data methods to 1.7% on Office-Home.
2. Diffusion models outperform GANs in discriminative transfer (+3.3% on Office-Home).
3. Vicinity encoding is the most critical design; removing it causes a 3.7% performance drop.
4. The auxiliary diffusion module also enhances the source-domain classifier itself.

## Highlights & Insights

- **Novel perspective**: Extends LDMs from generative applications to discriminative knowledge transfer.
- **Privacy preservation**: No source data samples are exposed, complying with GDPR and similar regulations.
- **Versatility**: The same diffusion module is applicable to SFDA, domain generalization, and source-domain augmentation.
- **Near-supervised performance**: Achieves results approaching oracle performance without source data.

## Limitations & Future Work

1. Training the auxiliary diffusion module on source data incurs additional source-side training cost.
2. Although the diffusion module does not expose raw data, it may theoretically leak partial distributional information.
3. Efficiency needs improvement for settings with a large number of categories (e.g., 345 classes in DomainNet).
4. Integration with text-guided diffusion models may further improve performance.

## Related Work & Insights

- **SHOT (Liang et al., 2020)**: A representative SFDA method.
- **NRC**: Neighbor consistency-based SFDA.
- **CoWA**: Covariance-weighted alignment.
- **Diffusion Models for DA**: A new direction pioneered by this paper.

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 5 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| Overall Recommendation | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [\[NeurIPS 2025\] PixPerfect: Seamless Latent Diffusion Local Editing with Discriminative Pixel-Space Refinement](pixperfect_seamless_latent_diffusion_local_editing_with_discriminative_pixel-spa.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](../../ICCV2025/image_generation/whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)
- [\[NeurIPS 2025\] ObCLIP: Oblivious Cloud-Device Hybrid Image Generation with Privacy Preservation](obclip_oblivious_cloud-device_hybrid_image_generation_with_privacy_preservation.md)
- [\[CVPR 2025\] Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment](../../CVPR2025/image_generation/everything_to_the_synthetic_diffusion-driven_test-time_adaptation_via_synthetic-.md)

</div>

<!-- RELATED:END -->
