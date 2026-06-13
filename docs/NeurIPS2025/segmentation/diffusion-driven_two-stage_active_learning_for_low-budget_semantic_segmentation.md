---
title: >-
  [Paper Note] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation
description: >-
  [NeurIPS 2025][Segmentation][Active Learning] A two-stage active learning pipeline (coverage → uncertainty) is proposed, leveraging multi-scale features from pretrained diffusion models to achieve efficient semantic segm…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Active Learning"
  - "Semantic Segmentation"
  - "Diffusion Models"
  - "Uncertainty Sampling"
  - "Low-Budget Annotation"
date: 2026-05-08
content_hash: 4598b1c37910a9a8
---

# Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2510.22229](https://arxiv.org/abs/2510.22229)  
**Code**: [Available](https://github.com/jn-kim/two-stage-edald)  
**Area**: Semantic Segmentation / Active Learning
**Keywords**: Active Learning, Semantic Segmentation, Diffusion Models, Uncertainty Sampling, Low-Budget Annotation

## TL;DR

A two-stage active learning pipeline (coverage → uncertainty) is proposed, leveraging multi-scale features from pretrained diffusion models to achieve efficient semantic segmentation under extremely low annotation budgets.

## Background & Motivation

Semantic segmentation requires dense pixel-level annotations, making the labeling cost prohibitively high. Active learning (AL) reduces this cost by strategically selecting the most informative samples for annotation; however, existing methods perform poorly in extremely low-budget scenarios:

- **Uncertainty-based methods** (Entropy, Margin, etc.): tend to select redundant pixels, as high-uncertainty pixels cluster near object boundaries
- **Representativeness-based methods** (Core-set, etc.): avoid redundancy but frequently miss informative boundary pixels
- **Existing pixel-level AL methods** (PixelPick, etc.): exhibit limited effectiveness under very low budgets

This paper defines an extreme low-budget setting: only $b = 0.1N$ pixels are annotated per round (where $N$ is the number of images), resulting in a total of approximately $N$ annotated pixels after 10 rounds—equivalent to an average of one pixel label per image.

**Key observation**: The reverse process of diffusion models generates features ranging from global structure to local detail across different timesteps. These multi-timestep features exhibit ensemble-like properties that can be exploited for uncertainty estimation.

## Method

### Overall Architecture

A two-stage pixel selection pipeline is proposed:
1. **Stage 1 (Coverage)**: Hierarchical candidate pixel selection based on representation (MaxHerding) to construct a diverse candidate pool.
2. **Stage 2 (Uncertainty)**: Entropy-enhanced disagreement scoring (eDALD) applied to the candidate pool to select final pixels for annotation.

### Key Designs

**1. Diffusion Feature Extraction**

Multi-scale features are extracted using an ImageNet-pretrained diffusion model:
- $T=3$ timesteps are sampled ($t_1=50, t_2=150, t_3=250$)
- $L=4$ feature layers are extracted per timestep ($l_1=5, l_2=8, l_3=12, l_4=17$)
- Features are upsampled and concatenated to form a rich per-pixel representation $z_x$

**2. Stage 1: Hierarchical MaxHerding**

- **Intra-image selection**: MaxHerding is applied to all pixels within each image to select $K=50$ representative pixels.
- **Cross-image refinement**: Candidate pixels from all images are merged, and MaxHerding is applied again to obtain a globally diverse candidate pool $\mathcal{M}$.

This local-to-global two-step strategy ensures that the candidate pool is both representative within each image and diverse at the global level.

**3. Stage 2: eDALD Uncertainty Scoring**

Core Idea: The stochastic noise injection of the diffusion model generates multiple sets of features, and mutual information is computed to measure epistemic uncertainty.

**DALD (Diffusion Active Learning Disagreement)** is grounded in the BALD framework:
$$I(\hat{Y}; Z | x) = H(\hat{Y} | x) - \mathbb{E}_{z}[H(\hat{Y} | Z=z, x)]$$

- Unconditional entropy $H(\hat{Y}|x)$: entropy of the averaged predictions over $M$ noisy samples
- Conditional entropy: mean of per-sample prediction entropies

**eDALD (entropy-enhanced DALD)** adds a single-sample entropy term:
$$\text{eDALD}(x) = I(\hat{Y}; Z | x) + H(\hat{Y} | z^{(0)}, x)$$

The additional entropy term captures predictive confidence, compensating for pure disagreement methods' insensitivity to low-confidence regions.

### Loss & Training

The segmentation head (MLP) is trained with standard cross-entropy loss:
$$\theta^* = \arg\min_\theta -\frac{1}{|\mathcal{L}|}\sum_{(x,y)\in\mathcal{L}} \log \hat{p}_\theta(y | x, f)$$

Training details: Adam optimizer, learning rate $10^{-3}$, batch size 5, with early stopping if loss shows no improvement for 50 iterations and pixel accuracy exceeds 95%.

## Key Experimental Results

### Main Results

**Table 2: mIoU (%) Comparison of Low-Budget AL Methods (after 10 rounds)**

| Backbone | Method | CamVid | ADE-Bed | Cityscapes | Pascal-C | Avg. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepLabV3 | PixelPick | 29.93 | 8.35 | 26.82 | 26.28 | 22.85 |
| DeepLabV3 | Didari et al. | 22.47 | 8.66 | 19.85 | 28.15 | 19.78 |
| DDPM | Random | 25.91 | 17.83 | 27.13 | 41.70 | 28.14 |
| DDPM | Margin | 31.27 | 30.03 | 32.23 | 45.11 | 34.66 |
| DDPM | eDALD (single-stage) | 25.14 | 23.06 | 29.44 | 43.05 | 30.17 |
| DDPM | **2-Stage eDALD** | **36.12** | **31.12** | **33.34** | **47.98** | **37.14** |

Key findings:
- The DDPM backbone outperforms DeepLabV3 by an average of **11.81** mIoU
- Two-stage eDALD surpasses the strongest single-stage method (Margin) by **2.48** mIoU on average
- Two-stage eDALD outperforms PixelPick by **14.29** mIoU on average

**Table 1: Effect of Coverage Filtering on Uncertainty Sampling (CamVid)**

| Uncertainty Method | Single-Stage | Herding → Uncertainty | Gain |
|:---:|:---:|:---:|:---:|
| Entropy | 25.26 | 30.77 | +5.51 (+21.8%) |
| eBALD | 25.96 | 32.12 | +6.16 (+23.7%) |
| eDALD | 25.14 | **36.12** | **+10.98 (+43.7%)** |
| BALD | 24.59 | 22.79 | -1.80 (-7.3%) |
| DALD | 23.81 | 21.05 | -2.76 (-11.6%) |

### Ablation Study

Core ablation: two-stage vs. single-stage
- Pure disagreement methods (BALD, DALD) degrade after incorporating MaxHerding, suggesting that pure disagreement over-emphasizes noisy regions once diversity is already guaranteed.
- Entropy-enhanced variants (eBALD, eDALD) achieve substantial gains, demonstrating strong complementarity between disagreement and confidence signals.
- The two-stage gain of eDALD (+43.7%) far exceeds that of other methods, confirming the strongest complementarity between diffusion feature disagreement and representativeness-based filtering.

### Key Findings

1. **Near fully-supervised performance**: Two-stage eDALD achieves 90% of fully-supervised mIoU within 21–47 rounds using only 0.003%–0.007% of pixel annotations.
2. **Diffusion backbone substantially outperforms conventional backbones**: Using DDPM features instead of DeepLabV3 features yields an average mIoU improvement of 11.81.
3. **Decoupled coverage → uncertainty outperforms hybrid strategies**: Ensuring diversity first and then applying uncertainty filtering is markedly superior to the reverse order or a mixed approach.

## Highlights & Insights

1. **Practical significance of the extreme low-budget setting**: An average of one pixel label per image more closely reflects real-world annotation budget constraints.
2. **Novel use of diffusion models**: Rather than being used for generation, diffusion models are leveraged for uncertainty estimation via their multi-timestep features—a conceptually novel approach.
3. **Elegant two-stage decoupled design**: Coverage and uncertainty play distinct, complementary roles, yielding a synergistic effect greater than the sum of its parts.
4. **Role of the additional entropy term in eDALD**: Only a single independent noise sample is required, incurring negligible computational overhead while yielding significant performance gains.

## Limitations & Future Work

1. DALD relies on stochastic noise injection from a diffusion backbone and is therefore not applicable to general model architectures.
2. The MaxHerding stage requires computing pairwise similarities over all pixels in each image, resulting in high computational cost.
3. Experiments are limited to a resolution of 256×256; applicability to high-resolution scenarios remains unverified.
4. Evaluation is restricted to pixel-level AL; no efficiency–accuracy trade-off comparisons with region-level or image-level methods are provided.

## Related Work & Insights

- **PixelPick** [Shin et al., 2021]: A pioneering pixel-level AL method using Margin sampling with DeepLabV3; serves as the primary baseline in this work.
- **MaxHerding** [Bae et al., 2024]: A generalized coverage method adopted in Stage 1 of the proposed pipeline.
- **BALD** [Houlsby et al., 2011]: Bayesian Active Learning by Disagreement; provides the theoretical foundation for eDALD.
- **Baranchuk et al., 2022**: Demonstrates that multi-timestep features from diffusion models are beneficial for semi-supervised segmentation.

## Rating

- **Novelty**: ★★★★☆ — The two-stage pipeline and eDALD design are original; leveraging diffusion features for AL is a distinctly novel idea.
- **Technical Depth**: ★★★★☆ — The information-theoretic framework is well-founded, and each design choice is theoretically motivated.
- **Experimental Thoroughness**: ★★★★★ — Comprehensive evaluation across four datasets with detailed ablation studies and rich qualitative analysis.
- **Writing Quality**: ★★★★☆ — Clear structure with high-quality figures and tables.
- **Practical Value**: ★★★★☆ — Code is open-sourced; the extreme low-budget setting has tangible real-world application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation](omnisegmentor_a_flexible_multi-modal_learning_framework_for_semantic_segmentatio.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](../../ICCV2025/segmentation/exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICCV 2025\] DDB: Diffusion Driven Balancing to Address Spurious Correlations](../../ICCV2025/segmentation/ddb_diffusion_driven_balancing_to_address_spurious_correlations.md)
- [\[AAAI 2026\] A²LC: Active and Automated Label Correction for Semantic Segmentation](../../AAAI2026/segmentation/a2lc_active_and_automated_label_correction_for_semantic_segm.md)

</div>

<!-- RELATED:END -->
