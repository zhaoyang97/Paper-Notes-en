---
title: >-
  [Paper Note] Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution
description: >-
  [ICCV 2025][LLM Pretraining][Image Quality Assessment] This paper introduces Image Intrinsic Scale (IIS)—the maximum scaling factor at which an image exhibits its highest perceptual quality—and proposes the IISA task…
tags:
  - "ICCV 2025"
  - "LLM Pretraining"
  - "Image Quality Assessment"
  - "Intrinsic Scale"
  - "Weak Labels"
  - "Multi-Scale Perception"
  - "Subjective Annotation"
date: 2026-05-08
content_hash: 3b32b57e44ef7180
---

# Image Intrinsic Scale Assessment: Bridging the Gap Between Quality and Resolution

**Conference**: ICCV 2025
**arXiv**: [2502.06476](https://arxiv.org/abs/2502.06476)
**Code**: [GitHub](https://github.com/SonyResearch/IISA)
**Area**: LLM Pretraining
**Keywords**: Image Quality Assessment, Intrinsic Scale, Weak Labels, Multi-Scale Perception, Subjective Annotation

## TL;DR

This paper introduces Image Intrinsic Scale (IIS)—the maximum scaling factor at which an image exhibits its highest perceptual quality—and proposes the IISA task, constructs a dataset of 785 images with expert annotations, and presents a weak-label training strategy (WIISA) that consistently improves IIS prediction across multiple NR-IQA methods.

## Background & Motivation

Image Quality Assessment (IQA) is a core task in computer vision, yet a critically overlooked problem is that **the relationship between image quality and spatial resolution (scaling factor) has never been systematically quantified**.

In practice, a paradoxical phenomenon is commonly observed:
- **Downscaling an image** can reduce noise grain and make blur less noticeable, potentially improving perceptual quality.
- **Excessive downscaling** causes loss of high-frequency details (e.g., bird feather texture), degrading quality again.
- Therefore, an **optimal scaling factor** exists that strikes the best balance between distortion removal and detail preservation.

This motivates a fundamental question: given an image, what is its optimal display scale?

Limitations of prior work:
1. Traditional camera performance metrics (e.g., P-MP, MTF) measure resolving power only under ideal laboratory conditions and are not applicable to real-world photographs.
2. NR-IQA methods assess quality at a fixed resolution and cannot answer the practical question of "at what size should this image be displayed."
3. The KonX dataset, while providing multi-resolution annotations for the first time, covers only 3 discrete scales, which is far too coarse.

## Method

### Overall Architecture

This work comprises three mutually reinforcing contributions:
1. **Concept Definition**: Image Intrinsic Scale (IIS), defined as the maximum scaling factor at which an image exhibits its highest perceptual quality.
2. **Dataset Construction**: IISA-DB, an expert-annotated dataset of 785 images.
3. **Weak-Label Strategy**: WIISA, which derives multiple weak-label training samples from a single annotation.

### Key Designs

1. **Formal Definition of IIS**: Let $I^s$ denote image $I$ scaled to factor $s$, and $Q(I^s)$ its perceptual quality. IIS is defined as:

$$\Omega(I) = \max\left(\operatorname*{argmax}_{s_{lb} \leq s \leq 1} Q(I^s)\right)$$

where the lower bound is $s_{lb} = 0.05$, since quality assessment at extremely small sizes is unreliable. The $\max$ operator selects the most informative scale when multiple equally optimal scales exist.

2. **WIISA Weak-Label Generation Strategy**: This is the core methodological contribution. The key insight is that once the IIS $\Omega(I)$ of the original image is known, the IIS of its downscaled versions can be derived via a piecewise function:

$$\overline{\Omega}(I^s) = \begin{cases} 1 & s_{lb} \leq s \leq \Omega(I) \\ \frac{\Omega(I)}{s} & \Omega(I) < s \end{cases}$$

Intuitive explanation:
- When $s \leq \Omega(I)$, the image is already at its optimal state and needs no further downscaling, so IIS = 1.
- When $s > \Omega(I)$, the image must be downscaled to $\Omega(I)/s$ to reach the optimum.

Based on this, $n_{wl}=2$ scales greater than $\Omega(I)$ are randomly sampled from each annotated pair $(I, \Omega(I))$ to generate weak-label pairs $(I^{s_i}, \overline{\Omega}(I^{s_i}))$. This strategy is applied online during training, adding $B \cdot n_{wl}$ weak-label samples per batch.

3. **Subjective Annotation Methodology**: A web-based annotation tool called ZOVI (Zoom Viewer) was developed, in which annotators use a slider to reduce scale from $s=1$ to $s_{lb}=0.05$ and identify the maximum scale beyond which no further quality improvement is perceived. Each annotator labels each image twice (with a gap of several days), and batches with SRCC < 0.5 are re-annotated. Final IIS values are aggregated using the geometric mean (due to the nonlinear nature of scale space), referred to as MOIS.

### Loss & Training

- The original loss functions of each NR-IQA method remain unchanged (e.g., TOPIQ uses MSE + ranking loss).
- WIISA modifies only the data sampling: downscaled image versions and their weak IIS labels are generated automatically each batch.
- Lanczos interpolation is used for rescaling, consistent with the annotation phase.
- 10-fold cross-validation is applied, with median test performance reported.

## Key Experimental Results

### Main Results

| Method | Training | SRCC ↑ | PLCC ↑ | RMSE ↓ | MAE ↓ |
|--------|----------|--------|--------|--------|-------|
| DBCNN | Base | 0.755 | 0.761 | 0.093 | 0.074 |
| DBCNN | +WIISA | 0.776 | 0.780 | 0.090 | 0.069 |
| TOPIQ | Base | 0.764 | 0.762 | 0.098 | 0.078 |
| TOPIQ | +WIISA | **0.808** | **0.805** | **0.088** | **0.069** |
| CONTRIQUE | Base | 0.618 | 0.635 | 0.114 | 0.090 |
| CONTRIQUE | +WIISA | 0.631 | 0.651 | 0.106 | 0.083 |
| ARNIQA | Base | 0.651 | 0.650 | 0.105 | 0.082 |
| ARNIQA | +WIISA | 0.687 | 0.672 | 0.103 | 0.079 |

WIISA consistently improves performance across all 6 methods, with a maximum relative gain of 5%.

### Ablation Study

| Configuration | SRCC | PLCC | RMSE | MAE | Note |
|---------------|------|------|------|-----|------|
| Base (no weak labels) | 0.764 | 0.762 | 0.098 | 0.078 | Baseline |
| $n_{wl}=1$ | 0.803 | 0.801 | 0.090 | 0.072 | 1 weak label |
| $n_{wl}=2$ (WIISA) | **0.808** | **0.805** | **0.088** | **0.069** | Optimal |
| $n_{wl}=3$ | 0.788 | 0.785 | 0.096 | 0.077 | Redundancy from too many weak labels |
| $\delta=0.50$ | 0.795 | 0.780 | 0.097 | 0.076 | Threshold too low |
| $\delta=0.80$ | 0.802 | 0.800 | 0.089 | 0.069 | Threshold too high |
| Bilinear interpolation | 0.799 | 0.796 | 0.089 | 0.070 | Minor effect of interpolation method |

### Key Findings

- **Zero-shot transfer fails**: Pretrained NR-IQA models applied directly to IIS prediction perform poorly (TOPIQ on SPAQ achieves only SRCC 0.475), indicating that IIS and conventional quality scores represent distinct tasks.
- **IISA-DB annotation reliability**: The average confidence interval is 0.057, comparable to the high-reliability NR-IQA dataset KonX (0.046).
- **Concavity assumption validated**: 90% (378/420) of image triplets in KonX satisfy the concave quality-scale function assumption.
- **WIISA method-agnostic**: All methods ranging from supervised learning to self-supervised and VLM-based approaches benefit from WIISA.

## Highlights & Insights

- **New task definition**: IIS addresses an eminently practical yet neglected question—"at what display size does this image look best?"
- **Bootstrapped data augmentation**: WIISA elegantly exploits the mathematical properties of IIS to derive multiple training samples from a single annotation.
- **Cross-domain application potential**: Applicable to image storage optimization, print size selection, super-resolution evaluation, and dataset construction.
- **Annotation methodology contribution**: The ZOVI tool, combined with double annotation and geometric mean aggregation, constitutes a reusable subjective experiment paradigm.

## Limitations & Future Work

- The dataset scale is limited (785 images), which may be insufficient for training large models.
- The concavity assumption does not hold for all images (10% exceptions), and handling of edge cases warrants further investigation.
- Weak labels can only be generated for scales $s > \Omega(I)$, leaving the low-scale regime uncovered.
- Display device variation is not considered (IIS may shift with different PPI values).
- Only Lanczos interpolation is evaluated; the effect of learned super-resolution methods on IIS remains unexplored.

## Related Work & Insights

- The distinction from effective resolution is noteworthy: effective resolution concerns information preservation, whereas IIS targets perceptual quality maximization.
- The weak-label approach is generalizable to other vision tasks requiring continuous-valued annotations (e.g., depth estimation, saliency detection).
- IISA exhibits greater sensitivity than conventional NR-IQA and may be used to evaluate subtle image processing differences.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introduces a novel and practical task with a clear and concise formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six methods with comprehensive ablations and reliability analysis, though dataset scale is modest.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured with a strong balance between mathematical rigor and intuitive explanation.
- **Value**: ⭐⭐⭐⭐ Opens a new research direction on quality–resolution interaction with broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Memory Mosaics at Scale](../../NeurIPS2025/llm_pretraining/memory_mosaics_at_scale.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](../../NeurIPS2025/llm_pretraining/predict_training_data_quality_via_its_geometry_in_metric_space.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[ICLR 2026\] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](../../ICLR2026/llm_pretraining/semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)

</div>

<!-- RELATED:END -->
