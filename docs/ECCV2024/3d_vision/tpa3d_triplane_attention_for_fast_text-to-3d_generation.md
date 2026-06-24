---
title: >-
  [Paper Note] TPA3D: Triplane Attention for Fast Text-to-3D Generation
description: >-
  [ECCV 2024][3D Vision] Proposes TPA3D, a GAN-based text-guided 3D generation framework that performs layer-wise refinement of sentence-level and word-level text features through a Triplane Attention (TPA) module, achieving fast and fine-grained text-to-3D textured mesh generation.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: 1324ba69ec8790ed
---

# TPA3D: Triplane Attention for Fast Text-to-3D Generation

**Conference**: ECCV 2024  
**arXiv**: [2312.02647](https://arxiv.org/abs/2312.02647)  
**Code**: None  
**Area**: 3D Vision

## TL;DR

Proposes TPA3D, a GAN-based text-guided 3D generation framework that performs layer-wise refinement of sentence-level and word-level text features through a Triplane Attention (TPA) module, achieving fast and fine-grained text-to-3D textured mesh generation.

## Background & Motivation

- Existing text-to-3D methods primarily rely on 2D diffusion models for SDS optimization (such as DreamFusion, Magic3D), resulting in inference times of tens of minutes or even hours.
- GAN-based methods utilizing vision-language models (such as TAPS3D) only use global CLIP sentence features, failing to capture fine-grained descriptions in the text.
- The lack of large-scale paired text-3D data necessitates text-guided 3D generation under an unsupervised setup.
- **Core Problem**: Existing GAN methods solely utilize global semantic features, leading to highly similar shapes and textures generated from different fine-grained text descriptions.

## Method

### Overall Architecture

TPA3D is built upon GET3D, consisting of two core modules:
1. **Sentence-level Triplane Generator $G$**: Concatenates CLIP sentence features with random noise to generate sentence-level geometry/texture triplanes via modulated convolutions.
2. **Triplane Attention Block (TPA)**: Performs word-level refinement on the sentence-level triplanes to generate detailed triplanes encoding both 3D spatial and word-level information.

InstructBLIP is used to automatically generate pseudo-captions for rendered images, eliminating the need for human-annotated text-3D paired data.

### Key Designs

**The TPA module comprises three attention mechanisms**:
- **In-plane Self-Attention**: Performs self-attention independently on each plane feature to maintain consistency within each plane.
- **Cross-plane Attention**: Fuses the triplane features as key/value, and uses the feature content of each plane as query, establishing 3D spatial connectivity.
- **Cross-word Attention**: Uses the self-refined features as query, and CLIP word features as key/value, injecting fine-grained word-level information.

Texture TPA additionally incorporates geometry triplanes (with a weight of $\alpha=0.5$) as input, ensuring correspondence between texture and geometry. The discriminator uses sentence features and camera poses as conditions, introducing a mismatch objective function to enhance sensitivity to mismatched text.

### Loss & Training

Total Loss = RGB adversarial loss + Mask adversarial loss + Mismatch loss + CLIP similarity loss

The mismatch loss utilizes mismatched sentence features to construct negative samples, enhancing the discriminator's capability. The CLIP loss computes the similarity between the generated image and text to stabilize the training process.

## Key Experimental Results

### Main Results

| Method | Car(FID↓) | Chair(FID↓) | Motorbike(FID↓) | Vehicle(FID↓) | Acc.(FID↓) |
|------|-----------|-------------|-----------------|---------------|------------|
| GET3D | 11.50 | 22.75 | 49.98 | 98.15 | 145.66 |
| TAPS3D | 26.37 | 44.70 | 84.83 | 152.34 | 172.14 |
| **TPA3D** | **18.50** | **38.11** | **77.69** | **68.80** | **83.31** |

| Method | Car(R-Prec@5↑) | Chair | Motorbike | Vehicle | Acc. |
|------|----------------|-------|-----------|---------|------|
| TAPS3D | 12.55 | 7.52 | 5.00 | 9.47 | 6.67 |
| **TPA3D** | **80.94** | **38.58** | **24.76** | **65.26** | **64.44** |

### Ablation Study

| Method | Device | Output Type | Inference Time |
|------|------|----------|----------|
| DreamFusion | TPUv4 | Rendering | 90 min |
| Magic3D | A100 x8 | Rendering | 40 min |
| TAPS3D | V100-32G | Mesh | 1.03 sec |
| **TPA3D** | V100-32G | Mesh | 2.87 sec |
| **TPA3D** | V100-32G | Rendering | 0.09 sec |

### Key Findings

- TPA3D significantly outperforms TAPS3D in the CLIP R-Precision@5 metric (Car: 80.94% vs 12.55%), demonstrating the high effectiveness of word-level refinement.
- Under the high-diversity OmniObject3D dataset, its FID outperforms the unconditional GET3D, suggesting that text guidance benefits multi-category generation.
- The inference speed is comparable to GAN-based methods (rendering in milliseconds), which is 3 to 4 orders of magnitude faster than SDS-based methods.
- Fixing the random seed and modifying only the text allows incremental manipulation of 3D object details.

## Highlights & Insights

- Introduces word-level triplane attention in a GAN framework for the first time, breaking through the information bottleneck of global features.
- Eliminates dependency on human-labeled text-3D paired data by employing InstructBLIP.
- Compared to SDS methods, it facilitates more accurate attribution of colors to distinct parts under complex multi-attribute texts (e.g., "red wheels + blue seats").

## Limitations & Future Work

- Scaling to open-world scenarios remains challenging, limited by the category diversity of ShapeNet/OmniObject3D.
- Texture details still exhibit a gap compared to real 3D scans.
- The resolution of the triplane representation constrains the upper bound of structural geometry and textures.

## Rating

- Novelty: ⭐⭐⭐⭐
- Effectiveness: ⭐⭐⭐⭐
- Practicality: ⭐⭐⭐⭐⭐
- Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)

</div>

<!-- RELATED:END -->
