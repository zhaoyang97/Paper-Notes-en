---
title: >-
  [Paper Note] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?
description: >-
  [ICCV 2025][Segmentation][Remote Sensing Foundation Models] This paper proposes SatDiFuser, a framework that repurposes a generative geospatial diffusion model (DiffusionSat) as a discriminative remote sensing foundation…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Remote Sensing Foundation Models"
  - "Diffusion Models"
  - "Feature Fusion"
  - "Self-Supervised Learning"
  - "Satellite Imagery"
date: 2026-05-08
content_hash: 34e1fca86dd83fa2
---

# Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?

**Conference**: ICCV 2025
**arXiv**: [2503.07890](https://arxiv.org/abs/2503.07890)
**Code**: [https://github.com/yurujaja/SatDiFuser](https://github.com/yurujaja/SatDiFuser)
**Area**: Semantic Segmentation / Remote Sensing
**Keywords**: Remote Sensing Foundation Models, Diffusion Models, Feature Fusion, Self-Supervised Learning, Satellite Imagery

## TL;DR

This paper proposes SatDiFuser, a framework that repurposes a generative geospatial diffusion model (DiffusionSat) as a discriminative remote sensing foundation model. Through systematic analysis of multi-stage, multi-timestep diffusion features and three designed fusion strategies (Global Weighted, Localized Weighted, and MoE Joint Fusion), SatDiFuser outperforms existing state-of-the-art geospatial foundation models (GFMs) on semantic segmentation and classification tasks, achieving gains of up to +5.7% mIoU and +7.9% F1.

## Background & Motivation

GFMs in remote sensing have predominantly relied on self-supervised paradigms such as contrastive learning (e.g., CROMA) or masked image modeling (e.g., SatMAE). However, these approaches exhibit inherent limitations: contrastive learning depends on constructing positive/negative pairs and applies global supervision that neglects spatial detail, while the patch-level reconstruction objective of MIM provides insufficient learning signal for remote sensing scenes dominated by homogeneous regions.

The paper raises a core question: **Can generative diffusion models also serve as effective discriminative geospatial foundation models?**

A key observation motivating this work is that diffusion models, through their iterative denoising process, naturally capture both global semantic structure and local detail simultaneously—a property well-suited to the multi-scale target characteristics of remote sensing imagery. By visualizing self-attention maps of a pretrained diffusion model, the authors find that semantically similar objects (e.g., cattle, trees, farmland) exhibit strong mutual attention even across large scale differences, demonstrating that diffusion models have learned rich multi-granularity semantic representations.

## Method

### Overall Architecture

SatDiFuser builds upon DiffusionSat, a latent diffusion model (LDM) pretrained on large-scale satellite imagery. The generative backbone is frozen, and multi-scale, multi-timestep features are extracted from the diffusion process. These features are aggregated via three learnable fusion strategies and subsequently fed into task-specific decoders (a linear head for classification and UPerNet for segmentation).

### Key Designs

1. **Diffusion Feature Extraction**:

    - Input images are encoded into a latent representation $\mathbf{z}$ via VAE; DDIM inversion is applied to obtain noisy latent variables.
    - The denoising UNet extracts three types of features at each of $S=4$ scales:
        - Self-attention output $\mathbf{A}_{t,s}$: captures contextual dependencies.
        - Cross-attention output $\mathbf{C}_{t,s}$: encodes text–image interactions.
        - ResNet residual output $\mathbf{R}_{t,s}$: captures local spatial information.
    - Features are extracted at multiple timesteps $t \in \{1, 100, 200\}$, forming a multi-scale, multi-timestep feature collection.
    - Key finding: ResNet and self-attention outputs contribute most significantly; cross-attention is nearly uninformative, as it encodes task-irrelevant text information.
    - Performance peaks within the earliest 20% of timesteps; later timesteps introduce excessive noise.

2. **Global Weighted Fusion**:

    - A scalar weight $w_{l,t}$ is learned for each feature block–timestep pair.
    - Aggregation formula: $\mathbf{X}_s = \sum_{t} \sum_{l} w_{l,t} \cdot \Phi_s^l(\mathbf{F}_{t,s}^l)$
    - Original multi-scale resolutions are preserved (no resizing to a unified size), forming a feature pyramid.
    - Design Motivation: Simple and efficient global importance weighting with minimal computational overhead.

3. **Localized Weighted Fusion**:

    - A lightweight gating network generates pixel-level weights $\mathbf{W}_{t,s}^l(u,v)$ for each spatial location.
    - Aggregation formula: $\mathbf{X}_s(u,v) = \sum_{t} \sum_{l} \mathbf{W}_{t,s}^l(u,v) \cdot \Phi_s^l(\mathbf{F}_{t,s}^l)(u,v)$
    - Allows different spatial positions to dynamically emphasize different features.
    - Design Motivation: Pixel-level weights better preserve spatial detail for remote sensing targets with complex boundaries or heterogeneous textures.

4. **Mixture-of-Experts (MoE) Joint Fusion**:

    - At each scale and timestep, features from all modules are concatenated along the channel dimension to form $\mathbf{X}_{t,s}$.
    - Processed through a shared MoE layer: $\mathbf{Y}_{t,s} = \sum_{e=1}^{E} \gamma_e(\mathbf{X}_{t,s}) f_e(\mathbf{X}_{t,s})$
    - Expert sub-networks each learn distinct patterns; the gating function determines which experts are activated.
    - Design Motivation: Jointly models complex interactions across modules and timesteps, offering greater flexibility than simple scalar or pixel-level weighting.

### Loss & Training

- The DiffusionSat backbone is frozen; only the fusion layers and decoder are trained.
- AdamW optimizer with cosine decay and a 5-epoch warmup is used.
- Text conditioning is uniformly set to "A satellite image" to prevent information leakage.
- Only RGB channels are used (even when datasets provide multispectral inputs), yet the method still outperforms approaches using full spectral information.

## Key Experimental Results

### Main Results — Semantic Segmentation (mIoU %)

| Method | pv-s | nz-c | neon | cashew | sa-c | ches |
|--------|------|------|------|--------|------|------|
| Satlas (supervised pretraining) | 92.3 | 83.1 | 52.0 | 49.1 | 31.6 | 52.2 |
| CROMA (contrastive learning) | 92.5 | 83.4 | 56.3 | 62.2 | 32.3 | 63.6 |
| DOFA (MIM) | 94.8 | 82.8 | 58.1 | 53.9 | 26.6 | 65.7 |
| **SatDiFuser-MoE** | **95.3** | **83.7** | **63.4** | **66.1** | 31.9 | **71.6** |
| Gain | +0.5 | −0.4 | +5.7 | +4.3 | +0.3 | +5.9 |

### Ablation Study — Feature Fusion Strategy Comparison

| Configuration | s2s (cls.) | es (cls.) | cashew (seg.) | pv-s (seg.) |
|---------------|-----------|----------|--------------|------------|
| Single timestep $t=1$, SA | 53.6 | 94.3 | 55.3 | 92.5 |
| Single timestep $t=100$, R | 50.5 | 92.4 | 57.9 | 92.6 |
| Simple concatenation | 55.4 | 94.5 | 59.1 | 92.9 |
| **Global Weighted Fusion** | **59.3** | **97.7** | **66.5** | **95.1** |
| **Localized Weighted Fusion** | 58.9 | 96.8 | 64.8 | 95.0 |
| **MoE Joint Fusion** | 58.8 | 97.3 | 66.1 | 95.3 |

### Key Findings

- SatDiFuser using only RGB channels surpasses GFMs that leverage full multispectral inputs, demonstrating the exceptional representational capacity of diffusion features.
- On m-forestnet and m-so2sat classification tasks, the method even outperforms fully supervised baselines, which is particularly noteworthy.
- Simple concatenation of multi-timestep features yields limited gains; learnable fusion strategies are essential.
- Each fusion strategy offers distinct advantages: Global Weighted Fusion is the most stable; Localized Weighted Fusion is better suited to detail-rich tasks; MoE Joint Fusion achieves the best results in complex scenarios.
- Cross-attention features are nearly uninformative, as they encode interactions with a generic text prompt rather than visual semantics.

## Highlights & Insights

- **The paper raises an important question and provides an affirmative answer**: generative diffusion models can indeed serve as GFMs and outperform existing discriminative approaches.
- The three fusion strategies form a coherent framework spanning simple-to-complex and global-to-local designs, providing a practical toolkit for utilizing diffusion features.
- The finding that RGB-only inputs surpass multispectral methods suggests that the semantic representations learned by diffusion models are of exceptionally high quality, compensating for the absence of spectral information.

## Limitations & Future Work

- Validation is currently limited to a single diffusion model (DiffusionSat); generalizability across additional diffusion architectures remains to be confirmed.
- The computational cost of DDIM inversion combined with multi-timestep feature extraction is substantial, potentially limiting practical deployment.
- The text prompt is fixed as "A satellite image," leaving the classification-guidance potential of text conditioning largely unexplored.
- Both the segmentation decoder (UPerNet) and classification head (linear) are relatively simple designs; stronger decoders may yield further performance gains.

## Related Work & Insights

- Diffusion Hyperfeatures first proposed aggregating multi-timestep diffusion features for keypoint correspondence; this paper extends that idea to multi-task remote sensing settings.
- Unsupervised segmentation works leveraging diffusion attention maps, such as DiffSeg and DiffCut, have validated the discriminative potential of diffusion features.
- The systematic experiments in this paper—covering combinations of timesteps, module types, and fusion strategies—provide valuable empirical guidance for the practical use of diffusion features.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic application of a large-scale remote sensing diffusion model to discriminative tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 datasets (6 segmentation + 6 classification), 8 GFM baselines, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; self-attention visualizations are particularly compelling.
- Value: ⭐⭐⭐⭐ Introduces a novel pretraining paradigm for remote sensing foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](../../NeurIPS2025/segmentation/roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](../../NeurIPS2025/segmentation/mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)
- [\[CVPR 2026\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](../../CVPR2026/segmentation/crossearthsar_a_sarcentric_and_billionscale_geospa.md)
- [\[NeurIPS 2025\] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](../../NeurIPS2025/segmentation/fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)

</div>

<!-- RELATED:END -->
