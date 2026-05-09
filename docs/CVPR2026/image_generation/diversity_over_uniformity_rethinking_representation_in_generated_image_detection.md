---
title: >-
  [Paper Note] Diversity over Uniformity: Rethinking Representation in Generated Image Detection
description: >-
  [CVPR 2026][Image Generation][Generated image detection] This paper proposes an Anti-Feature-Collapse Learning (AFCL) framework that filters task-irrelevant features via an information bottleneck and suppresses excessive overlap among heterogeneous forgery cues, thereby preserving diversity and complementarity in discriminative representations. The method achieves significant improvements in cross-model generated image detection.
tags:
  - CVPR 2026
  - Image Generation
  - Generated image detection
  - feature collapse
  - representation diversity
  - information bottleneck
  - CLIP
date: 2026-05-08
content_hash: 1affe9d39cdf7c2f
---

# Diversity over Uniformity: Rethinking Representation in Generated Image Detection

**Conference**: CVPR 2026
**arXiv**: [2603.00717](https://arxiv.org/abs/2603.00717)
**Code**: [GitHub](https://github.com/Yanmou-Hui/DoU)
**Area**: Image Forensics / AI-Generated Image Detection
**Keywords**: Generated image detection, feature collapse, representation diversity, information bottleneck, CLIP

## TL;DR

This paper proposes an Anti-Feature-Collapse Learning (AFCL) framework that filters task-irrelevant features via an information bottleneck and suppresses excessive overlap among heterogeneous forgery cues, thereby preserving diversity and complementarity in discriminative representations. The method achieves significant improvements in cross-model generated image detection.

## Background & Motivation

The core problem with existing generated image detectors is not insufficient features but **representation homogenization**: during training, models tend to compress multi-source information into a small number of salient discriminative patterns, forming shortcut-based decision rules. This feature collapse phenomenon leads to:

- Discriminative capacity concentrated in a few principal component directions (effective rank of only 1–2)
- Detection accuracy saturating after a small number of principal components, with additional features left unexploited
- Severe performance degradation when the generative method or image distribution shifts

The authors validate this hypothesis through UMAP visualization and principal component analysis: the feature space of pre-trained models inherently contains rich discriminative cues, yet after fine-tuning these are compressed into an extremely low-rank subspace. Upon introducing representation heterogeneity constraints, the detector effectively leverages more principal components and exhibits substantially improved generalization.

## Method

### Overall Architecture

Built upon a frozen CLIP ViT-L/14 image encoder, the framework extracts multi-stage CLS features. Features at each stage are first filtered by a CIB module, then decorrelated by an AFCL module, and finally aggregated via weighted pooling with class-specific prompt learning for real/fake classification.

### Key Designs

1. **Cue Information Bottleneck (CIB)**: Applies information bottleneck filtering to each stage's features, maximizing mutual information with label $y$ while minimizing mutual information with input $x$:

    $\max_{\{\mathrm{CIB}_i\}} \sum_{i=1}^{N} [I(\tilde{v}_i; y) - \beta I(\tilde{v}_i; x)]$

   A tractable CIB loss is derived via a variational lower bound:

    $\mathcal{L}_{\mathrm{CIB}} = \sum_{i=1}^{N} D_{\mathrm{KL}}[p(y|\tilde{\mathcal{V}}) \| p(y|\tilde{\mathcal{V}} \setminus \tilde{v}_i)]$

   This ensures each cue carries indispensable discriminative information, yielding a purified and complementary feature set.

2. **Anti-Feature-Collapse Learning (AFCL)**: Employs the Hilbert-Schmidt Independence Criterion (HSIC) as a kernel-based dependency measure to enforce statistical independence among features across different stages:

    $\mathcal{L}_{\mathrm{AFCL}} = \frac{1}{N(N-1)} \sum_{i \neq j} \mathrm{HSIC}(\tilde{v}_i, \tilde{v}_j)$

   where $\mathrm{HSIC}(\tilde{v}_i, \tilde{v}_j) = \frac{1}{(B-1)^2} \mathrm{Tr}(K_i H K_j H)$. A weight uniformity regularizer $\mathcal{L}_{\mathrm{reg}} = (\sum_{i=1}^{N} \alpha_i^2 - 1/N)^2$ is further introduced to prevent aggregation weight collapse.

3. **Class-Specific Prompt Learning (CSP)**: Learns a set of trainable context vectors for each of the "real" and "fake" classes, aligning the final visual representation with text prototypes via cosine similarity:

    $s_c = \frac{\tilde{v}_{\mathrm{final}} \cdot e_c}{\|\tilde{v}_{\mathrm{final}}\| \|e_c\|}$

   Optimized using a cross-entropy loss $\mathcal{L}_{\mathrm{CSP}}$.

### Loss & Training

The total loss jointly optimizes four terms:

$$\mathcal{L} = \mathcal{L}_{\mathrm{CSP}} + \lambda_1 \mathcal{L}_{\mathrm{CIB}} + \lambda_2 \mathcal{L}_{\mathrm{AFCL}} + \lambda_3 \mathcal{L}_{\mathrm{reg}}$$

- Backbone: CLIP ViT-L/14 (frozen)
- Optimizer: Adam, learning rate $1 \times 10^{-4}$, batch size 512
- Training data: SD v1.4 subset of GenImage
- Early stopping applied to prevent overfitting

## Key Experimental Results

### Main Results

| Dataset / Metric | Ours (AFCL) | VIB-Net | CLIPping | Gain (vs. VIB-Net) |
|------------------|-------------|---------|----------|--------------------|
| Mean AP | 99.52% | 96.13% | 87.32% | +3.39% |
| Mean ACC | 92.81% | 87.13% | 87.81% | +5.68% |
| Cross-model ACC | 90.02% | — | — | — |

Evaluation covers 21 generative models (11 from UniversalFakeDetect + 7 from GenImage + 6 from AIGI-Holmes), spanning both GAN and diffusion model paradigms.

### Ablation Study

| Configuration | Mean ACC | Cross-model ACC | Mean AP | Note |
|---------------|----------|-----------------|---------|------|
| Baseline | 87.81% | 85.56% | 87.32% | No CIB / AFCL / reg |
| +CIB | 89.72% | 85.99% | 99.38% | Information bottleneck filtering |
| +AFCL+reg | 91.60% | 91.15% | 99.40% | Decorrelation + regularization |
| Full model | **92.81%** | **90.02%** | **99.52%** | All components combined |

### Key Findings

- The proposed method achieves an effective rank of **67.38**, far exceeding CNNDet (1.37) and VIB-Net (1.92)
- The number of principal components required to explain 90% of the variance is only 26 fewer than the pre-trained backbone, whereas other detectors require hundreds fewer
- With only 0.1% of training data (320 samples), the method achieves 80.98% ACC / 90.81% AP
- Maintains state-of-the-art robustness under JPEG compression and Gaussian blur perturbations

## Highlights & Insights

- **Precise problem diagnosis**: Through effective rank and principal component analysis, the paper quantitatively identifies feature collapse as the root cause of generalization failure, rather than insufficient information capacity
- **HSIC-based decorrelation**: Using kernel methods to measure feature independence is more flexible than simple orthogonality constraints, capturing nonlinear dependencies
- **Dual strategy of purification and diversity**: CIB denoises and AFCL decorrelates — the two are complementary and neither is dispensable

## Limitations & Future Work

- Training is conducted solely on SD v1.4; whether the advantages persist under larger-scale multi-source training remains unverified
- The effect of the choice of the number of multi-stage features $N$ on performance is not thoroughly discussed
- Detection capability for video generation models (e.g., Sora) is not addressed

## Related Work & Insights

- Complementary to the variational information bottleneck mechanism in VIB-Net: VIB-Net compresses redundancy but does not address cross-layer collapse
- DRCT/Bias-Free methods mitigate the issue indirectly through debiasing operations; the proposed approach targets the representational structure directly, offering a more fundamental solution
- Inspiration: the anti-collapse idea underlying AFCL can be transferred to other discriminative tasks requiring generalization, such as deepfake video detection and cross-domain classification

## Rating

- Novelty: ⭐⭐⭐⭐ Re-examines the detection problem from the perspective of representation collapse — a novel and theoretically grounded angle
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 21 generative models with complete ablations and thorough robustness/few-shot experiments
- Writing Quality: ⭐⭐⭐⭐ Motivation analysis and visualizations are convincing; presentation is clear and well-structured
- Value: ⭐⭐⭐⭐ Provides important insights for the field of generated image detection; the AFCL idea is broadly transferable

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods](training-free_detection_of_generated_videos_via_spatial-temporal_likelihoods.md)
- [\[AAAI 2026\] CausalCLIP: Causally-Informed Feature Disentanglement and Filtering for Generalizable Detection of Generated Images](../../AAAI2026/image_generation/causalclip_causally-informed_feature_disentanglement_and_filtering_for_generaliz.md)
- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](../../NeurIPS2025/image_generation/epistemic_uncertainty_for_generated_image_detection.md)
- [\[AAAI 2026\] Aggregating Diverse Cue Experts for AI-Generated Image Detection](../../AAAI2026/image_generation/aggregating_diverse_cue_experts_for_ai-generated_image_detec.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)

<!-- RELATED:END -->
