---
title: >-
  [Paper Note] Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation
description: >-
  [AAAI 2026][Segmentation][Domain generalized semantic segmentation] This paper proposes FLEX-Seg, a framework that reframes the inherent **boundary misalignment** between images and semantic masks in diffusion-model-synt…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "Domain generalized semantic segmentation"
  - "boundary misalignment"
  - "diffusion model synthetic data"
  - "adaptive prototypes"
  - "uncertainty weighting"
date: 2026-05-08
content_hash: c7ea5bd7ee72bc22
---

# Do We Need Perfect Data? Leveraging Noise for Domain Generalized Segmentation

**Conference**: AAAI 2026
**arXiv**: [2511.22948](https://arxiv.org/abs/2511.22948)  
**Code**: [Available](https://github.com/VisualScienceLab-KHU/FLEX-Seg)  
**Area**: Segmentation
**Keywords**: Domain generalized semantic segmentation, boundary misalignment, diffusion model synthetic data, adaptive prototypes, uncertainty weighting

## TL;DR

This paper proposes FLEX-Seg, a framework that reframes the inherent **boundary misalignment** between images and semantic masks in diffusion-model-synthesized data as an opportunity to learn robust representations. Through three modules—Granular Adaptive Prototypes (GAP), Uncertainty Boundary Emphasis (UBE), and Hardness-Aware Sampling (HAS)—FLEX-Seg achieves state-of-the-art performance on domain generalized semantic segmentation.

## Background & Motivation

Domain Generalized Semantic Segmentation (DGSS) aims to train models exclusively on source-domain data such that they generalize to unseen target domains (e.g., varying weather and lighting conditions). Recent diffusion-model-based data generation methods (e.g., DGInStyle) enhance generalization by producing diverse synthetic images, but face a fundamental challenge:

**Boundary misalignment between generated images and semantic masks.** Unlike real datasets where annotations are derived from real images, synthetic data pipelines generate images from semantic masks. This inverse process inherently introduces pixel-level spatial misalignment, particularly at object boundaries.

Key observations by the authors:
- Error rates in boundary regions are substantially higher than in interior regions even under normal conditions.
- This gap is further amplified under adverse conditions such as fog, rain, snow, and nighttime.
- Existing boundary-aware methods assume perfect image–mask alignment and cannot handle the misalignment inherent in synthetic data.

Core insight: **Rather than forcing perfect alignment, the misalignment itself can be exploited to learn more robust representations.**

## Method

### Overall Architecture

FLEX-Seg (**FL**exible **E**dge e**X**ploitation for **Seg**mentation) comprises three synergistic components:

1. **GAP (Granular Adaptive Prototypes)**: Multi-granularity boundary prototype learning.
2. **UBE (Uncertainty Boundary Emphasis)**: Dynamic boundary weighting based on predictive entropy.
3. **HAS (Hardness-Aware Sampling)**: Progressive hard-sample mining.

Training pipeline: synthetic data $\mathcal{D}_G$ is generated from source domain $\mathcal{D}_S$ via a diffusion model and merged into a unified training set → GAP learns cross-domain invariant boundary representations → UBE adaptively emphasizes uncertain regions → HAS progressively focuses on hard samples.

### Key Designs

#### GAP: Granular Adaptive Prototypes

**Problem analysis**: Semantic boundaries exhibit inherent scale variation—distant small objects present thin boundaries while nearby large objects yield thick boundary regions. Boundary pixels simultaneously exhibit geometric variation (thickness) and style variation (appearance under different environmental conditions).

**Class-shape token coordinate system**: Each boundary pixel $p_i$ is represented as $(c_i, g_i)$, where $c_i$ denotes class semantics and $g_i$ encodes geometric attributes (boundary thickness).

**Multi-granularity boundary extraction**: Three granularity levels of boundary masks (thin, medium, thick) are generated via morphological operations:
$$B_g = \text{Dilate}(M_d, k_g) \ominus \text{Erode}(M_d, k_g)$$

**Prototype bank construction**: A $C \times 3$ prototype bank $\mathcal{P} = \{p_{c,g}\}$ ($C$ classes × 3 granularities) is maintained via momentum update:
$$p_{c,g} \leftarrow m \cdot p_{c,g} + (1-m) \cdot f_{c,g}$$

**Contrastive learning**: An InfoNCE loss with imbalance-aware weights is employed:
$$\mathcal{L}_{GAP} = -\frac{1}{N} \sum_{i=1}^{N} w_{c_i,g_i} \cdot \log \frac{e^{\langle f_i, p_{c_i,g_i} \rangle / \tau}}{\sum_{(c',g') \in \mathcal{P}} e^{\langle f_i, p_{c',g'} \rangle / \tau}}$$

Weights $w_{c,g}$ are adaptively adjusted based on prototype update frequency, assigning higher weights to low-frequency class–granularity combinations.

#### UBE: Uncertainty Boundary Emphasis

A dynamic weighting mechanism based on predictive entropy that adapts across domains without manual tuning:

1. Compute per-pixel predictive entropy: $H_{x,y} = -\sum_{c=1}^{C} p_c(x,y) \log p_c(x,y)$
2. Apply adaptive weights exclusively to boundary regions (interior pixels retain weight 1):
$$w(x,y) = 1 + \alpha \cdot \text{sigmoid}\left(\frac{H_{x,y} - \mu_H}{\sigma_H + \epsilon}\right), \quad \text{if } (x,y) \in B$$
3. Apply weighted cross-entropy: $\mathcal{L}_{UBE} = \frac{1}{N} \sum_{(x,y)} w(x,y) \cdot \mathcal{L}_{CE}(x,y)$

Pixels with high entropy (typically at misaligned boundaries or ambiguous regions) receive larger weights, directing the model's attention toward difficult areas.

#### HAS: Hardness-Aware Sampling

A progressive curriculum that transitions from random sampling to loss-based sampling via a sigmoid decay schedule:

- A per-image difficulty score $h_i(t)$ is maintained and updated via EMA.
- Threshold function: $\text{threshold}(t) = \frac{1}{1 + e^{k(t-m)}}$
- At each iteration: if random value $r > \text{threshold}(t)$, loss-based sampling is applied; otherwise, random sampling is used.
- Sampling probability is proportional to difficulty scores (controlled via softmax with a temperature parameter).

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{UBE} + \lambda_{gap} \cdot \mathcal{L}_{GAP}$

- GAP ensures cross-domain consistent boundary representations via contrastive learning.
- UBE adaptively adjusts the learning focus based on prediction confidence.
- Hyperparameters: $\tau = 0.07$, $\alpha = 3.0$, $\lambda_{gap} = 0.5$, $k = 0.05$, $\tau_{HAS} = 1.0$

## Key Experimental Results

### Main Results

Source domain GTA → 5 real driving datasets, using MiT-B5 backbone + HRDA:

| Method | ACDC | DZ | CS | BDD | MV | Avg |
|------|------|------|------|------|------|------|
| HRDA + DGInStyle | 46.07 | 25.53 | 58.63 | 52.25 | 62.47 | 48.99 |
| **HRDA + FLEX-Seg** | **48.51** | **28.16** | **59.49** | **52.48** | 61.71 | **50.07** |
| DAFormer + DGInStyle | 44.04 | 25.58 | 55.31 | 50.82 | 56.62 | 46.47 |
| **DAFormer + FLEX-Seg** | **46.56** | **29.51** | **56.84** | **52.06** | **57.93** | **48.58** |

Improvements are particularly pronounced on adverse-condition domains: ACDC +2.44%, Dark Zurich +2.63% (HRDA); Dark Zurich +3.93% (DAFormer).

### Ablation Study

Module contributions (DAFormer + MiT-B5; Avg2 = mean of ACDC + DZ):

| GAP | UBE | HAS | Avg2 | Avg3 | Avg5 |
|-----|-----|-----|------|------|------|
| ✗ | ✗ | ✗ | 34.81 | 54.25 | 46.47 |
| ✓ | ✗ | ✗ | 36.33 (+1.52) | 55.26 | 47.69 |
| ✗ | ✓ | ✗ | 35.05 (+0.24) | 55.33 | 47.21 |
| ✓ | ✓ | ✗ | 36.15 | 56.07 | 48.10 |
| ✓ | ✓ | ✓ | **38.04** | 55.61 | **48.58** |

Synthetic data volume ablation: 10,000 images is optimal; performance slightly degrades with more data.

Sigmoid decay vs. linear decay vs. no decay: Sigmoid 38.04% > no decay 36.82% > linear 36.28%.

### Key Findings

- **GAP is the core contribution**: Introducing GAP alone yields +1.52% Avg2 improvement; multi-granularity boundary prototypes are critical for domain-invariant representations.
- **Trade-off effect of HAS**: Adding HAS slightly decreases standard-domain Avg3 (−0.46%) but substantially improves adverse-domain Avg2 (+1.89%), reflecting its hard-sample-focused strategy.
- **Strong framework generalizability**: Consistent improvements (+1.44% Avg2) are observed with synthetic data generated by ALDM.

## Highlights & Insights

1. **Counterintuitive reframing**: The inherent defect of synthetic data (boundary misalignment) is transformed into an opportunity to learn robust representations rather than being eliminated.
2. **Class–granularity 2D prototype bank**: Decomposing boundary features into semantic and geometric dimensions enables fine-grained cross-domain alignment.
3. **Adaptive learning focus**: UBE automatically identifies difficult regions via predictive entropy, eliminating the need for manual boundary weight tuning.
4. **Progressive curriculum learning**: The sigmoid decay schedule of HAS ensures sufficient exploration in early training and hard-sample focus in later stages.

## Limitations & Future Work

- The framework depends on pretrained diffusion models (e.g., DGInStyle/ALDM) for synthetic data generation and cannot be applied without synthetic data.
- HAS incurs a slight performance drop on standard domains; the trade-off between hard-sample focus and full-domain balance warrants further optimization.
- The prototype bank size $C \times 3 \times 256$ grows linearly with the number of classes, raising efficiency concerns for large-vocabulary scenarios.
- Validation is limited to GTA → real driving scenes; other synthetic-to-real domain gaps (e.g., indoor scenes, satellite imagery) remain unexplored.

## Related Work & Insights

- **DGInStyle** (2024): Leverages latent diffusion models to synthesize diverse images; serves as the primary synthetic data source in this work.
- **FAMix**: Employs CLIP-pretrained ResNet-50 for domain generalization but underperforms the proposed method under adverse conditions.
- **HRDA** (2023): A multi-resolution domain adaptation framework; FLEX-Seg is stacked on top of it to achieve further improvements.
- **Boundary-aware methods** (BAPA, InverseForm, etc.): Assume perfect alignment and cannot handle the misalignment present in synthetic data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "exploit noise rather than eliminate it" perspective is novel; the three-component design is well-targeted.
- **Technical Depth**: ⭐⭐⭐⭐ — Multi-granularity prototype contrastive learning + entropy-guided weighting + curriculum sampling constitute a principled combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five target domains, two backbones, detailed hyperparameter ablations, and cross-generator validation.
- **Writing Quality**: ⭐⭐⭐⭐ — In-depth motivation analysis; error rate visualizations are convincing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Causal-Tune: Mining Causal Factors from Vision Foundation Models for Domain Generalized Semantic Segmentation](causal-tune_mining_causal_factors_from_vision_foundation_mod.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](../../CVPR2026/segmentation/phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[ICML 2026\] What Makes Synthetic Data Effective in Image Segmentation](../../ICML2026/segmentation/what_makes_synthetic_data_effective_in_image_segmentation.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)

</div>

<!-- RELATED:END -->
