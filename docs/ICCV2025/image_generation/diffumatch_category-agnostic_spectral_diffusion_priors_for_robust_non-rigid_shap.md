---
title: >-
  [Paper Note] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching
description: >-
  [ICCV 2025][Image Generation][Functional Maps] This paper proposes training an unconditional diffusion model in the spectral domain of Functional Maps, and replacing hand-crafted axiomatic regularizers (e.g.…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Functional Maps"
  - "Spectral Diffusion Priors"
  - "Non-rigid Shape Matching"
  - "Score Distillation"
  - "Zero-shot Generalization"
date: 2026-05-08
content_hash: 9c1299fbe4ad17fe
---

# DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching

**Conference**: ICCV 2025
**arXiv**: [2507.23715](https://arxiv.org/abs/2507.23715)  
**Code**: [https://github.com/daidedou/diffumatch/](https://github.com/daidedou/diffumatch/)  
**Area**: Diffusion Models / 3D Shape Matching
**Keywords**: Functional Maps, Spectral Diffusion Priors, Non-rigid Shape Matching, Score Distillation, Zero-shot Generalization

## TL;DR

This paper proposes training an unconditional diffusion model in the spectral domain of Functional Maps, and replacing hand-crafted axiomatic regularizers (e.g., Laplacian commutativity, orthogonality) with distilled structural priors, enabling zero-shot non-rigid shape matching across categories.

## Background & Motivation

Deep Functional Maps have demonstrated strong performance on non-rigid shape correspondence tasks, yet two fundamental limitations persist:

**Over-reliance on axiomatic modeling**: Existing methods confine the learned component to feature extraction, while the regularization and training losses for functional maps still depend on hand-crafted axiomatic constraints (e.g., near-isometry assumption, area preservation). When these assumptions break down—as in cross-category matching—generalization degrades sharply.

**Category specificity**: Learning-based methods (e.g., 3D-CODED, Neural Jacobian Fields) trained on a specific category struggle to generalize to new ones; models trained on humans fail to match animal shapes effectively.

**Core Insight**: The functional map matrix $C \in \mathbb{R}^{k \times k}$ exhibits image-like structural properties in the spectral domain (e.g., approximate diagonality), making its distribution learnable by generative models. If structural priors can be learned from large collections of high-quality functional maps, data-driven regularization can supplant axiomatic constraints.

## Method

### Overall Architecture

The pipeline consists of two stages: (1) training a spectral diffusion model on a large-scale dataset of functional maps computed from registered human shapes; (2) performing zero-shot optimization at test time for novel shape pairs, using distilled diffusion priors to derive a regularization mask that guides functional map estimation.

### Key Designs

1. **Spectral Diffusion Model Training**:

    - Approximately 40,000 template-to-shape functional maps ($30 \times 30$) from the D-FAUST dataset are used, with absolute values $|C_{gt}|$ as training inputs to handle sign ambiguity.
    - The architecture is DiT-S (Diffusion Transformer) with patch size 5, trained under the EDM framework for 1,000 epochs.
    - The denoiser $D(C_\sigma, \sigma)$ learns the structural distribution of functional maps at varying noise levels.

2. **Mask Distillation (Core Contribution)**:

    - Traditional methods derive sparse masks $M_{reg}$ via Laplacian commutativity; this paper directly distills masks from the score function of the diffusion model.
    - Assuming the functional map likelihood is $p(C_\sigma;\sigma) \propto \exp(-\|M_\sigma \cdot C_\sigma\|^2)$, its score is $s(C_\sigma;\sigma) = -2M_\sigma^2 \cdot C_\sigma$.
    - Combined with the diffusion model's score estimate $(D(C_\sigma;\sigma) - C_\sigma)/\sigma^2$, the mask is computed as:
   $$M_\sigma^2 = \mathbb{E}_{n_\sigma \sim \mathcal{N}(0,\sigma^2 I), n_\sigma > 0}\left[\frac{|C|_\sigma - D(|C|_\sigma;\sigma)}{2\sigma^2 |C|_\sigma}\right]$$
   - Only positive noise samples $n_\sigma > 0$ are used to avoid division-by-zero instability.

3. **Zero-shot Matching Pipeline**:

    - Given a novel shape pair, point features are extracted via DiffusionNet;
    - A "raw" functional map $C_{raw}$ is estimated through an FMReg layer ($\alpha=0$);
    - The diffusion model distills a mask $M_\sigma$ from $C_{raw}$ ($\sigma=1$, 100 noise samples);
    - $M_\sigma$ regularizes a refined solution $C_{reg}$, which is further processed by Zoomout to obtain the final proper map.

### Loss & Training

The total loss consists of two terms:

$$\mathcal{L}_{total}(C_{raw}) = \mathcal{L}_{proper}(C_{raw}) + \mathcal{L}_{SDS}(|C_{raw}|)$$

- $\mathcal{L}_{proper} = \|C_{raw} - C_{proper}\|^2$: encourages the raw map to remain close to the proper map (the valid map obtained after Zoomout).
- $\mathcal{L}_{SDS}$: Score Distillation Sampling loss, continuously injecting diffusion prior knowledge into the optimization.
- Gradients from both losses are not backpropagated through the mask or denoiser; only the feature extractor parameters $\theta$ are updated.

## Key Experimental Results

### Main Results

| Dataset | Type | DiffuMatch | SNK | Simplified Fmaps | 3D-CODED |
|---------|------|-----------|-----|------------------|----------|
| FAUST | Human | **1.9** | 1.8 | 1.7 | 7.5 |
| SCAPE | Human | **4.4** | 4.7 | 2.3 | 17.2 |
| SHREC19 | Human | **3.9** | 5.8 | 3.4 | 13.4 |
| DT4D-Intra | Human-like | **1.8** | 2.0 | 2.0 | 45.0 |
| DT4D-Inter | Human-like | **8.6** | 9.0 | 8.9 | 61.4 |
| SMAL | Animal | **10.1** | 9.1 | 42.1 | 54.6 |
| TOSCA | Animal | **2.9** | 3.6 | 5.1 | 32.8 |

Metric: geodesic error (lower is better). DiffuMatch significantly outperforms learning-based methods on cross-category benchmarks such as animals.

### Ablation Study

| Configuration | SHREC Geodesic Error | Notes |
|--------------|----------------------|-------|
| Vanilla SDS | 57.3 | Severe mismatches due to sign ambiguity |
| Mask + Zoomout | 8.3 | Distilled mask alone |
| $\mathcal{L}_{proper}$ | 7.7 | Proper loss alone |
| Mask + $\mathcal{L}_{SDS}$ | 7.1 | Mask + SDS |
| Mask + $\mathcal{L}_{proper}$ | 6.7 | Mask + proper loss |
| **Mask + $\mathcal{L}_{proper}$ + $\mathcal{L}_{SDS}$** | **4.4** | Full method |
| Ours + Axiomatic | 4.3 | Adding axiomatic loss yields negligible gain |

### Key Findings

- Diffusion priors trained on human shapes generalize directly to human-like and animal shapes, demonstrating category-agnostic transferability.
- Distilled masks yield higher quality than traditional Laplacian/Resolvent masks, as evidenced by improved Zoomout accuracy after initialization.
- Adding axiomatic constraints yields nearly no improvement (4.3 vs. 4.4), indicating that the learned priors already subsume the information encoded by axiomatic regularizers.

## Highlights & Insights

- **First fully data-driven replacement of axiomatic regularization**: In the deep functional map pipeline, both the mask regularizer and training loss are derived from diffusion priors rather than hand-crafted constraints.
- **Elegant handling of sign ambiguity**: Modeling the absolute value $|C|$ circumvents the inherent sign ambiguity of functional maps.
- **Strong cross-category generalization**: Training exclusively on human shapes enables matching of 3D shapes from entirely different categories, such as animals.

## Limitations & Future Work

- Limited capability for handling highly non-isometric or partial shapes, a general weakness of functional map methods.
- The diffusion model is trained solely on human shapes, limiting diversity; training on richer registered data is expected to yield further improvements.
- Jointly learning basis functions and spectral regularization may offer a promising direction for partial shape matching.

## Related Work & Insights

- This work transfers Score Distillation Sampling (SDS) from 3D generation (e.g., DreamFusion) to spectral-domain regularization, demonstrating the applicability of diffusion priors in non-generative tasks.
- It represents a first step toward a "functional map foundation model," where pre-trained priors in the spectral domain are universally applicable across shape categories.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Replacing axiomatic regularizers with spectral diffusion priors is a genuinely novel concept
- **Technical Depth**: ⭐⭐⭐⭐ — The mask distillation derivation is rigorous, though the diffusion model itself is standard
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers diverse benchmarks spanning humans, human-like shapes, and animals
- **Value**: ⭐⭐⭐⭐ — Zero-shot cross-category matching has broad practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spectral Image Tokenizer](spectral_image_tokenizer.md)
- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[ICCV 2025\] Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion](dual_recursive_feedback_on_generation_and_appearance_latents_for_pose-robust_tex.md)
- [\[ICCV 2025\] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis](infgen_a_resolution-agnostic_paradigm_for_scalable_image_synthesis.md)

</div>

<!-- RELATED:END -->
