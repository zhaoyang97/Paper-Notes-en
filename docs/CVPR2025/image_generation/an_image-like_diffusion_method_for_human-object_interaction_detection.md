---
title: >-
  [Paper Note] "HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection"
description: >-
  [CVPR2025][Image Generation][Diffusion Model] Academic paper note for "HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection".
tags:
  - CVPR2025
  - Image Generation
  - Diffusion Model
  - HOI Image
date: 2026-05-08
content_hash: a340a22ceab8696d
---
# HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection

**Conference**: CVPR 2025  
**Institution**: Lancaster University  
**Keywords**: Human-Object Interaction Detection, Diffusion Models, Multinomial Diffusion, HOI Image  

## Background & Motivation

Human-Object Interaction Detection (HOI Detection) is a core task in scene understanding, aiming to detect <human, interaction, object> triplets from images. For example, "a person riding a bicycle" requires identifying the human, the bicycle, and the interaction relationship "riding".

Traditional HOI detection methods (such as QPIC, CDN) are typically based on Transformer decoders, utilizing a set of learnable queries to predict HOI triplets. The issue with this approach is that the number of queries is fixed, making it difficult to handle scenes with highly variable numbers of interactions; moreover, one-step prediction lacks the capability for iterative optimization.

Recently, diffusion models have demonstrated outstanding iterative denoising capabilities in generative tasks. Can diffusion models be applied to HOI detection? Existing attempts (such as DiffHOI) directly perform diffusion on bounding box coordinates, but achieve limited effectiveness because:
1. The core of HOI lies not in precise box coordinates, but in **who is doing what with whom**.
2. Continuous Gaussian diffusion is unnatural for categorical outputs (such as interaction types).
3. There is a lack of mechanism to utilize detector priors.

The core innovation of HOI-IDiff is: **re-encoding HOI triplets into an "image" and then applying a specially designed multinomial diffusion process on this image.**

## Method

### Core Innovation 1: HOI Image Construction

Encode all HOI relationships in each scene into an $H 	imes W 	imes 2$ probability image:

$$I_{	ext{HOI}}[h, w, :] = v_{	ext{obj}}(h) \otimes m_{	ext{int}}(w)$$

where:
- $H$ = number of human-object pairs in the scene
- $W$ = number of interaction categories (117 classes for HICO-DET)
- Channel 0: Object class probability $v_{	ext{obj}} \in \Delta^{|\mathcal{O}|}$ (probability distribution on a simplex)
- Channel 1: Interaction type probability $m_{	ext{int}} \in \{0, 1\}^{|\mathcal{A}|}$ (multi-label binary indicator)

**Intuitive Understanding**: Each row of the HOI Image corresponds to a human-object pair, and each column corresponds to an interaction type. The pixel value represents the probability of that interaction occurring. This representation transforms structural prediction into an image generation problem.

### Core Innovation 2: Multinomial Diffusion

Standard Gaussian diffusion adds Gaussian noise to continuous data, but each pixel in the HOI Image is a probability value (summing to 1), and Gaussian noise would violate this constraint.

The forward process of **Multinomial Diffusion**:

$$q(x_t | x_{t-1}) = 	ext{Cat}(x_t; (1 - eta_t) x_{t-1} + eta_t / K)$$

where $K$ is the number of categories. Key differences:
- The coefficient is $(1-eta_k)$ instead of $\sqrt{1-eta_k}$.
- The noise term is a uniform distribution $1/K$ instead of a Gaussian distribution.
- The probability sum is always maintained as 1.

| Property | Gaussian Diffusion | Multinomial Diffusion |
|------|---------|----------|
| Data type | Continuous value | Probability distribution |
| Noise type | Gaussian $\mathcal{N}(0,1)$ | Uniform $1/K$ |
| Forward coefficient | $\sqrt{1-eta_t}$ | $(1-eta_t)$ |
| Probability constraint | None | Always satisfies $\sum=1$ |
| Final state | $\mathcal{N}(0,I)$ | Uniform distribution |

### Core Innovation 3: Slice Patchification

Traditional ViTs partition images into local patches (such as 16×16), but the semantic structure of the HOI Image is different—each row represents complete human-object pair information, and each column represents complete interaction type information. Local patches would break this row-column semantics.

**Slice Patchification** proposes slice-based partitioning:
- **Horizontal Slices**: $H$ row vectors of width $W$ (each slice is a complete human-object pair)
- **Vertical Slices**: $W$ column vectors of height $H$ (each slice is a complete interaction type)

The two sets of slices are processed by Transformers separately and then fused. This guarantees global dependencies within rows and within columns, while establishing relations between rows and columns through cross-attention.

### Core Innovation 4: Detector Prior Initialization

Standard diffusion starts denoising from pure noise, but HOI detection can leverage the output of an object detector (such as DETR) as a prior:

$$x_T = (1 - lpha) \cdot 	ext{Uniform} + lpha \cdot 	ext{DetectorPrior}$$

The detector prior provides initial human-object pairing guesses, significantly reducing the number of denoising steps.

## Experimental Results

### HICO-DET

| Method | Full mAP | Rare mAP | Non-Rare mAP |
|------|---------|---------|-------------|
| QPIC | 29.07 | 21.85 | 31.23 |
| CDN | 32.07 | 27.19 | 33.53 |
| GEN-VLKT | 33.75 | 29.25 | 35.10 |
| HOICLIP | 34.69 | 31.12 | 35.74 |
| Standard diffusion baseline | 42.50 | 40.12 | 43.21 |
| **HOI-IDiff** | **47.71** | **48.36** | **47.52** |

### V-COCO

| Method | Scenario 1 | Scenario 2 |
|------|-----------|-----------|
| QPIC | 58.8 | 61.0 |
| CDN | 63.9 | 65.9 |
| HOICLIP | 66.2 | 68.5 |
| **HOI-IDiff** | **73.4** | **76.1** |

### Ablation Study

| Configuration | HICO-DET Full mAP |
|------|------------------|
| Standard Gaussian diffusion | 42.50 |
| + Multinomial diffusion | 44.23 |
| + Slice Patchification | 45.89 |
| + Detector prior | 46.84 |
| + All optimizations | **47.71** |

The step-by-step improvement from 42.50 to 47.71 validates the contribution of each component.

## Method Analysis

### Why is Slice Patchification effective?

Traditional patches disrupt the row-column semantic structure of the HOI Image. For example, a 16×16 patch contains partial interaction information for 16 human-object pairs—neither completely representing any single human-object pair nor completely representing any single interaction type. Slices guarantee the completeness of semantic units.

### Why is multinomial diffusion better than Gaussian diffusion?

The pixels of the HOI Image are probability distributions; Gaussian noise would produce negative and unnormalized values, requiring extra normalization steps. Multinomial diffusion maintains the probability constraints throughout the entire process, generating intermediate results that are all valid probability distributions.

## Limitations & Future Work

- The size of the HOI Image varies with the number of human-object pairs in the scene, which requires padding for batch processing.
- The number of denoising steps in multinomial diffusion is still relatively large (typically 100 steps).
- The efficiency in dense interaction scenes (>50 human-object pairs) needs to be optimized.

## Summary

HOI-IDiff cleverly leverages the iterative optimization capability of diffusion models by redefining HOI detection as a "probability image generation" problem. The three major innovations—multinomial diffusion, Slice Patchification, and detector prior—work synergistically to achieve a new SOTA on both HICO-DET and V-COCO. This concept of "transforming structural prediction into image generation" has broad inspiring significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](../../ICCV2025/image_generation/scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[CVPR 2026\] OneHOI: Unifying Human-Object Interaction Generation and Editing](../../CVPR2026/image_generation/onehoi_unifying_human-object_interaction_generation_and_editing.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](../../CVPR2026/image_generation/vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[CVPR 2025\] BootPlace: Bootstrapped Object Placement with Detection Transformers](bootplace_bootstrapped_object_placement_with_detection_transformers.md)

</div>

<!-- RELATED:END -->
