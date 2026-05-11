---
title: >-
  [Paper Note] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation
description: >-
  [AAAI 2026][Segmentation][Cross-domain few-shot segmentation] This paper proposes the HSL framework, which addresses the **segmentation granularity gap** between source and target domains in cross-domain few-shot segment…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "Cross-domain few-shot segmentation"
  - "hierarchical semantic learning"
  - "style randomization"
  - "superpixel"
  - "prototype confidence"
date: 2026-05-08
content_hash: bffe429f7d69b43d
---

# Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation

**Conference**: AAAI 2026
**arXiv**: [2511.12200](https://arxiv.org/abs/2511.12200)
**Code**: [Available](https://github.com/Sparkling-Water/HSLNet)
**Area**: Segmentation
**Keywords**: Cross-domain few-shot segmentation, hierarchical semantic learning, style randomization, superpixel, prototype confidence

## TL;DR

This paper proposes the HSL framework, which addresses the **segmentation granularity gap** between source and target domains in cross-domain few-shot segmentation (CD-FSS) via three modules — Dual Style Randomization (DSR), Hierarchical Semantic Mining (HSM), and Prototype Confidence-modulated Thresholding (PCMT) — achieving state-of-the-art performance across four target-domain datasets.

## Background & Motivation

Cross-domain few-shot segmentation (CD-FSS) aims to segment novel categories from unseen target domains given only a few annotated samples. Existing methods primarily address the **style gap** (e.g., color, texture) between source and target domains, while overlooking a fundamental issue: the **segmentation granularity gap**.

Specifically, the distinction between foreground (e.g., birds) and background in the source domain is typically coarse-grained (i.e., visually prominent), whereas in the target domain, foreground and background may be highly similar (e.g., lesion regions versus normal skin), resembling the fine-grained intra-foreground variations in the source domain (e.g., subtle color differences among feathers). Since models are trained exclusively on the source domain, they tend to treat the foreground as a monolithic entity and fail to capture fine-grained foreground–background semantic distinctions in the target domain.

The core mechanism is to extract **hierarchical semantic features** that endow the model with intra-class compactness and inter-class discriminability at multiple granularities.

## Method

### Overall Architecture

The HSL framework consists of three core modules:

1. **DSR (Dual Style Randomization)**: Applies foreground-level and global-level style randomization to training data.
2. **HSM (Hierarchical Semantic Mining)**: Mines hierarchical semantic features using multi-scale superpixel masks.
3. **PCMT (Prototype Confidence-modulated Thresholding)**: Mitigates segmentation ambiguity caused by high foreground–background similarity at test time.

The pipeline proceeds as follows: multi-scale superpixel masks are obtained via a superpixel segmentation model → DSR augmentation → feature extraction via an image encoder → feature enhancement via HSM → prototype computation via the SSP module → final prediction generation via PCMT.

### Key Designs

#### DSR: Dual Style Randomization

**Foreground Style Randomization**:
- A local region image $\mathbf{I}^{local}$ is randomly sampled from the coarsest superpixel mask.
- Both the foreground image $\mathbf{I}^{fg}$ and the local region image are decomposed into amplitude and phase spectra via FFT.
- Their amplitude spectra are fused: $\mathbf{A}^{fusion} = \omega \mathbf{A}^{local} + (1-\omega) \mathbf{A}^{fg}$, where $\omega \sim N(0, \sigma_f^2)$.
- The foreground phase spectrum is preserved, and the fused amplitude is used to reconstruct the image via IFFT, simulating varying degrees of foreground–background dissimilarity.

**Global Style Randomization**:
- A random convolution (RC) layer perturbs local textures of the foreground-randomized image.
- Similarly FFT-based: the phase spectrum of the original image is retained while the amplitude spectrum of the RC output is used for reconstruction.
- This prevents the RC layer from directly corrupting content details.

#### HSM: Hierarchical Semantic Mining

The core idea is that multi-scale superpixel masks naturally partition an image into local regions at varying granularities, approximating semantic regions at different scales.

Procedure:
1. For each superpixel scale, binary masks are generated for each region.
2. Shallow low-level features $\mathbf{F}^l$ and deep high-level features $\mathbf{F}^h$ are extracted.
3. Low-level features are downsampled, and Masked Average Pooling (MAP) is applied to obtain low-level and high-level prototypes for each region.
4. Low-level prototypes are enhanced via two layers of multi-head self-attention (MSA) and then fused with high-level prototypes: $\mathbf{p}_{ij} = \alpha \tilde{\mathbf{p}}_{ij}^l + (1-\alpha) \mathbf{p}_{ij}^h$.
5. Region prototypes are mapped back to feature maps via RMAP, and all scale feature maps are aggregated onto the high-level features.

This allows each pixel to be influenced by multi-scale region prototypes, enhancing intra-class compactness and inter-class discriminability across granularities.

#### PCMT: Prototype Confidence-modulated Thresholding

To address segmentation ambiguity arising from high foreground–background similarity at test time:

1. A foreground confidence map is computed: $\mathbf{M}_q^{conf} = \mathbf{M}_q^{fg} - \mathbf{M}_q^{bg}$.
2. An adaptive threshold $t$ is computed via OTSU.
3. A **prototype confidence** score $C$ is introduced to quantify the probability of segmentation ambiguity (based on cross-view prototype similarity).
4. The final threshold is: $\frac{1}{1+e^{\beta(C+\gamma)}} t$
    - High prototype confidence → threshold approaches 0 (equivalent to conventional similarity comparison).
    - Low prototype confidence → adaptive threshold $t$ is applied (mitigating ambiguity).

### Loss & Training

- BCE loss is used for training.
- DSR is applied only during training; PCMT is applied only during inference.
- A meta-learning paradigm is adopted, with each episode comprising a support set and a query set.
- Four superpixel scales are used: $\{5^2, 10^2, 15^2, 20^2\}$.
- SGD optimization is applied for 5 epochs with a learning rate of 1e-3.

## Key Experimental Results

### Main Results

Models are trained on PASCAL VOC 2012 + SBD and evaluated on four target-domain datasets:

| Method | Backbone | Deepglobe 1/5-shot | ISIC 1/5-shot | Chest X-ray 1/5-shot | FSS-1000 1/5-shot | Avg. 1/5-shot |
|------|----------|-------------------|---------------|---------------------|-------------------|--------------|
| DRA | Res-50 | 41.29/50.12 | 40.77/48.87 | 82.35/82.31 | 79.05/80.40 | 60.86/65.42 |
| LoEC | ViT-base | 42.12/51.48 | 52.91/62.43 | 83.94/84.12 | 81.05/83.69 | 65.01/70.43 |
| **HSL (Ours)** | **Res-50** | **46.13/53.80** | **48.01/55.56** | **84.57/85.34** | 78.22/80.36 | **64.23/68.77** |
| **HSL (Ours)** | **ViT-base** | **45.77/54.56** | **59.36/64.62** | **85.95/86.25** | **81.89/83.84** | **68.24/72.32** |

With ViT-base, the proposed method surpasses the previous SOTA LoEC by **+3.23%** and **+1.89%** on 1-shot and 5-shot settings, respectively.

### Ablation Study

| DSR | HSM | PCMT | Res-50 mIoU | ViT-base mIoU |
|-----|-----|------|-------------|---------------|
| ✗ | ✗ | ✗ | 57.82 | 62.24 |
| ✓ | ✗ | ✗ | 60.44 (+2.62) | 64.55 (+2.31) |
| ✗ | ✓ | ✗ | 60.92 (+3.10) | 65.29 (+3.05) |
| ✓ | ✓ | ✗ | 62.97 | 67.05 |
| ✓ | ✓ | ✓ | **64.23** | **68.24** |

Ablation over multi-scale superpixel masks: removing any single scale consistently degrades performance, with the 4-scale configuration being optimal (e.g., removing the $5\times5$ scale reduces performance from 67.05 to 66.34).

Ablation over thresholding strategies (ViT-base, 1-shot):

| Strategy | Deepglobe | ISIC | Chest | FSS | Avg. |
|------|-----------|------|-------|-----|------|
| Fixed threshold 0 | 44.54 | 55.79 | 85.93 | 81.93 | 67.05 |
| OTSU | 45.57 | 59.98 | 85.81 | 80.43 | 67.95 |
| PCMT (Ours) | 45.77 | 59.36 | 85.95 | 81.89 | **68.24** |

### Key Findings

- **HSM contributes most**: Introducing HSM alone yields a larger gain than DSR alone (3.10% vs. 2.62%), indicating that hierarchical semantic mining is the primary driver for addressing the granularity gap.
- **PCMT achieves flexible balance**: OTSU performs well on ambiguity-prone ISIC but degrades on FSS-1000 (which is closer to the source domain); PCMT adaptively adjusts the threshold on a per-sample basis.
- **All scales are complementary**: Removing any single scale results in performance degradation, confirming the complementary nature of fine-grained and coarse-grained information.

## Highlights & Insights

1. **First work to focus on the segmentation granularity gap**: Unlike prior CD-FSS methods that target style discrepancy, this paper identifies and addresses the overlooked issue of granularity gap.
2. **Elegant use of FFT frequency-domain operations**: Foreground style randomization alters appearance while preserving content by fusing amplitude spectra.
3. **Superpixels as hierarchical semantic priors**: Multi-scale superpixels naturally provide semantic partitions at varying granularities in a simple yet effective manner.
4. **Adaptive mechanism in PCMT**: Avoids a one-size-fits-all thresholding strategy by smoothly transitioning between conventional similarity comparison and adaptive thresholding based on prototype confidence.

## Limitations & Future Work

- The quality of the superpixel segmentation model directly affects HSM performance; for structurally simple target domains (e.g., medical images), superpixels may not constitute the optimal prior.
- The four superpixel scales and various hyperparameters ($\sigma_f$, $\sigma_g$, $K$, $\alpha$, $\beta$, $\gamma$) require careful tuning.
- PCMT relies on the OTSU algorithm, which may not be robust for confidence maps with multimodal distributions.
- Validation is limited to the PASCAL VOC → 4 target domain setting; generalization to larger-scale source domains remains unexplored.

## Related Work & Insights

- **PATNet** (ECCV 2022): The pioneering CD-FSS work, proposing to transform features into a domain-agnostic space.
- **DRA** (CVPR 2024): Enhances generalization through domain randomization.
- **LoEC** (CVPR 2025): Applies style perturbation at the feature level.
- **Broad applicability of frequency-domain methods**: FFT is increasingly important in domain adaptation and generalization; the amplitude-swap paradigm merits broader adoption across tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — The granularity gap perspective is novel; the DSR+HSM+PCMT three-module design is well-motivated and coherent.
- Technical Depth: ⭐⭐⭐⭐ — FFT-based frequency augmentation, multi-scale superpixel mining, and adaptive threshold modulation are technically solid.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations covering two backbones and four target domains.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated with intuitive illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV](otter_mitigating_background_distractions_of_wide-angle_few-shot_action_recogniti.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](../../NeurIPS2025/segmentation/towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)
- [\[AAAI 2026\] InfoCLIP: Bridging Vision-Language Pretraining and Open-Vocabulary Semantic Segmentation via Information-Theoretic Alignment Transfer](infoclip_bridging_vision-language_pretraining_and_open-vocab.md)
- [\[CVPR 2026\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](../../CVPR2026/segmentation/scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)
- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)

</div>

<!-- RELATED:END -->
