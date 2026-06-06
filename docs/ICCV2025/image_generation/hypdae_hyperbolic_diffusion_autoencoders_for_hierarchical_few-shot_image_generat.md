---
title: >-
  [Paper Note] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation
description: >-
  [ICCV 2025][Image Generation][Hyperbolic Space] This work combines the hierarchical representation learning capacity of hyperbolic space with the high-quality generative capability of diffusion autoencoders. By manipulat…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Hyperbolic Space"
  - "Diffusion Autoencoders"
  - "Hierarchical Representation"
  - "Few-shot Generation"
  - "Poincaré Disk"
date: 2026-05-08
content_hash: d737d978e9951567
---

# HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation

**Conference**: ICCV 2025
**arXiv**: [2411.17784](https://arxiv.org/abs/2411.17784)  
**Code**: [https://github.com/lingxiao-li/HypDAE](https://github.com/lingxiao-li/HypDAE)  
**Area**: Diffusion Models / Few-shot Image Generation
**Keywords**: Hyperbolic Space, Diffusion Autoencoders, Hierarchical Representation, Few-shot Generation, Poincaré Disk

## TL;DR

This work combines the hierarchical representation learning capacity of hyperbolic space with the high-quality generative capability of diffusion autoencoders. By manipulating the radius and direction of latent codes within the Poincaré disk, it achieves controllable, diverse, and class-consistent few-shot image generation.

## Background & Motivation

Few-shot Image Generation aims to generate diverse, high-quality images for unseen categories using only a handful of samples. The core challenge is the **trade-off between class consistency and image diversity**.

**Three major bottlenecks of existing methods**:

**Limited generation quality of GAN-based methods**: Transfer-, fusion-, and transformation-based GAN approaches struggle to produce realistic images under insufficient training data.

**Insufficient diversity**: One-to-one mappings (latent code → image) lose high-frequency details when latent codes are undertrained, leading to homogeneous outputs.

**Dependence on annotated data**: Learning hierarchical latent representations typically requires class labels, which are difficult to obtain in practice.

**Why hyperbolic space?** Images exhibit semantic hierarchical structure: high-level identity-related attributes (e.g., gender, ethnicity) define the category core, while low-level identity-irrelevant attributes (e.g., expression, hairstyle) introduce intra-class variation. Hyperbolic space (negative curvature) naturally encodes tree-like hierarchies via exponentially growing radii — disk boundaries correspond to fine-grained features, and the disk center corresponds to abstract/shared features.

**Why integrate diffusion models?** Diffusion models yield higher generation quality than GANs under limited data, and pretrained foundation models (SD, CLIP) provide strong priors that support adaptation with few samples.

## Method

### Overall Architecture

HypDAE consists of two stages:

**Stage I — Diffusion Autoencoder**:
- **Semantic encoder**: A CLIP image encoder extracts high-level semantic codes $\boldsymbol{c}$ (class token only, 512→1024 dimensions), aligned to SD's text feature space via an MLP.
- **Stochastic encoder**: The pretrained SD model encodes images into stochastic subcodes $\boldsymbol{z}_T$ via DDIM inversion, capturing low-level details not covered by the semantic code.
- Both encoders cooperate to achieve high-fidelity reconstruction: $(\boldsymbol{c}, \boldsymbol{z}_T) \to x'$.
- **Anti-copying techniques**: (1) Strong data augmentation (flipping, rotation, blurring, elastic transforms); (2) content bottleneck (using only the class token to compress information).

**Stage II — Hyperbolic Encoder-Decoder**:
- A 5-layer single-head Transformer encoder reduces the Euclidean latent code $\boldsymbol{c}$ to 512 dimensions.
- The exponential map $\exp_\mathbf{0}^c$ projects Euclidean vectors onto the Poincaré disk $\mathbb{D}^n$.
- A Möbius linear layer produces the hyperbolic representation $\boldsymbol{c}_h = f^{\otimes_c}(\exp_\mathbf{0}^c(\text{E}(\boldsymbol{c})))$.
- A 30-layer single-head Transformer decoder reconstructs $\boldsymbol{c}' = \text{D}(\log_\mathbf{0}^c(\boldsymbol{c}_h))$ via the logarithmic map $\log_\mathbf{0}^c$.

### Key Designs

1. **Hyperbolic Hierarchical Representation**:

    - In the Poincaré disk, the distance formula $d_\mathbb{D}(\mathbf{x}, \mathbf{y}) = \text{arccosh}(1 + \frac{2\|\mathbf{x}-\mathbf{y}\|^2}{(1-\|\mathbf{x}\|^2)(1-\|\mathbf{y}\|^2)})$ causes distances between boundary points to grow exponentially.
    - A classification loss (extended MLR) pushes fine-grained image embeddings toward the disk boundary (maximizing inter-class distance), while shared features are embedded near the center.
    - The radius $r_\mathbb{D}$ directly corresponds to attribute hierarchy: $r_\mathbb{D} > 5.0$ reflects low-level identity-irrelevant attribute variation, while $r_\mathbb{D} < 2.0$ leads to changes in identity attributes.

2. **Hyperbolic Latent Code Editing (for diverse generation)**:

    - **Stochastic subcode variation**: The semantic code $\boldsymbol{c}$ is frozen while different random seeds yield different $\boldsymbol{z}_T$, altering low-level features such as texture and background.
    - **Semantic code perturbation**: $\boldsymbol{c}_h$ is perturbed randomly along geodesic directions at a fixed radius $r_\mathbb{D}$, modifying intra-class variation features.
    - **Hierarchical interpolation**: Interpolation between two embeddings along geodesics enables smooth attribute transitions.

3. **Pseudo-label Training**: Pretrained CLIP's zero-shot classification is used to generate pseudo-labels for images, eliminating the need for manual annotation.

### Loss & Training

**Stage I** (only the MLP is trained):
$$\mathcal{L}_{align} = \mathbb{E}_{\boldsymbol{z}_0, t, \boldsymbol{c}, \epsilon \sim \mathcal{N}(0,1)}[\|\epsilon - \epsilon_\theta(\boldsymbol{z}_t, t, \boldsymbol{c})\|_2^2]$$

**Stage II**:
$$\mathcal{L} = \mathcal{L}_{hyper} + \lambda \cdot \mathcal{L}_{rec}$$

- $\mathcal{L}_{hyper} = -\frac{1}{N}\sum_{n=1}^N \log(p_n)$: Hyperbolic MLR classification loss that promotes hierarchical structure formation.
- $\mathcal{L}_{rec}(\boldsymbol{c}, \boldsymbol{c}') = \|\boldsymbol{c} - \boldsymbol{c}'\|_2 + 1 - \cos(\boldsymbol{c}, \boldsymbol{c}')$: L2 + cosine similarity reconstruction loss.

## Key Experimental Results

### Main Results (1-shot Generation)

| Method | Setting | Flowers FID↓ | Flowers LPIPS↑ | AnimalFaces FID↓ | AnimalFaces LPIPS↑ | VGGFaces FID↓ | NABirds FID↓ |
|------|------|-------------|---------------|-----------------|-------------------|--------------|-------------|
| DeltaGAN | 1-shot | 109.78 | 0.391 | 89.81 | 0.442 | 80.12 | 96.79 |
| SAGE | 1-shot | 43.52 | 0.439 | 27.43 | 0.545 | 34.97 | 19.45 |
| HAE | 1-shot | 50.10 | 0.474 | 26.33 | 0.564 | 35.93 | 21.85 |
| **HypDAE (Real)** | 1-shot | **23.96** | **0.760** | **14.31** | **0.742** | **6.25** | **7.64** |
| **HypDAE (Pseudo)** | 1-shot | 24.43 | **0.763** | **13.14** | **0.743** | **5.96** | **7.57** |

HypDAE achieves substantial improvements across all datasets: AnimalFaces FID drops from 26.33 (HAE) to 13.14 (a 50% improvement), and LPIPS improves from 0.564 to 0.743.

### Ablation Study

**Effect of Hyperbolic Radius $r_\mathbb{D}$**:

| Radius $r_\mathbb{D}$ | 6.2 (boundary) | 5.5 | 4.5 | 3.0 (center) |
|---------------------|----------|-----|-----|----------|
| FID↓ | 15.18 | **14.31** | 14.71 | 20.65 |
| LPIPS↑ | 0.704 | 0.742 | 0.794 | **0.896** |
| CLIP-S (identity preservation) | 77.37 | 75.15 | 71.45 | 67.89 |
| CLIP-P (perturbation similarity) | 69.62 | 72.35 | 74.25 | 77.00 |

$r_\mathbb{D} = 5.5$ achieves the best FID–LPIPS trade-off; smaller radii increase diversity at the cost of identity preservation.

**Euclidean vs. Hyperbolic Space**:

| Method | AnimalFaces FID↓ | AnimalFaces LPIPS↑ |
|------|-----------------|-------------------|
| HypDAE (Euclidean) | 20.72 | 0.729 |
| **HypDAE (Hyperbolic)** | **14.31** | **0.742** |

Hyperbolic space representation reduces FID by 30.9%.

### Key Findings

- **Pseudo-labels outperform real labels**: HypDAE (Pseudo) slightly surpasses HypDAE (Real) on most benchmarks, suggesting that noise in manual annotations interferes with hierarchical representation learning. Despite pseudo-label accuracy of only 39–79%, it suffices to learn useful hierarchical structures.
- The stochastic encoding strength controls the similarity–diversity trade-off between generated and reference images: higher strength drives $\boldsymbol{z}_T$ closer to pure noise, yielding more diverse outputs.
- Qualitative comparisons (Fig. 9) show that HypDAE generates fine details such as intricate feather textures that HAE/WaveGAN fail to reproduce.
- In a user study (Table 4), HypDAE ranks first in quality (3.45/4), fidelity (3.58/4), and diversity (3.86/4) by a wide margin.

## Highlights & Insights

- **First combination of hyperbolic space and diffusion models** for few-shot generation: the hierarchical structure of hyperbolic space endows the diffusion model with interpretable and controllable semantic editing.
- **$r_\mathbb{D}$ as a diversity knob** is intuitively elegant: smaller radius → more abstract → more diverse; larger radius → more specific → more faithful.
- The **two-stage design** elegantly avoids the difficulties of joint end-to-end training and eliminates the need for large-scale annotated data.
- The combination of pretrained SD and CLIP enables high-quality generation even with extremely limited data.

## Limitations & Future Work

- The Stage II Transformer decoder (30 layers) is relatively large; lighter mapping networks are worth exploring.
- The low resolution (64×64) of VGGFaces leads to anomalous FID values; switching to FFHQ resolves this but introduces domain shift.
- The current framework only supports 1-shot and 3-shot settings; fusion strategies for more reference images remain to be designed.
- Extension to video generation or 3D generation has not been explored.

## Related Work & Insights

- This work transfers DiffAE's dual-code (semantic + stochastic) architecture from Euclidean to hyperbolic space, demonstrating the value of non-Euclidean geometry in generative modeling.
- HAE first explored hyperbolic space for few-shot generation but was limited to GANs; HypDAE overcomes GAN quality bottlenecks through diffusion models.
- The success of pseudo-label training suggests that for hierarchical representation learning, "approximately correct" labels are more effective than precise but noisy manual annotations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First method combining hyperbolic space and diffusion models for few-shot generation; conceptually novel
- **Technical Depth**: ⭐⭐⭐⭐ — Solid hyperbolic geometry foundations; well-motivated two-stage design
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, rich ablations, user study, and comprehensive visualizations
- **Value**: ⭐⭐⭐⭐ — No manual labels required; diverse images generated from a single reference

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[AAAI 2026\] Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval](../../AAAI2026/image_generation/hyperbolic_hierarchical_alignment_reasoning_network_for_text-3d_retrieval.md)
- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](../../CVPR2026/image_generation/uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[ICML 2026\] Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](../../ICML2026/image_generation/envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)
- [\[ICLR 2026\] HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation](../../ICLR2026/image_generation/hierloc_hyperbolic_entity_embeddings_for_hierarchical_visual_geolocation.md)

</div>

<!-- RELATED:END -->
