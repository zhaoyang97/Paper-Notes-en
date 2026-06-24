---
title: >-
  [Paper Note] Towards Generalizable Scene Change Detection
description: >-
  [CVPR 2025][Segmentation][Scene Change Detection] Proposes GeSCF, the first zero-shot scene change detection framework, which leverages internal features of SAM to achieve cross-domain generalization and temporally consistent change mask generation, while defining a generalized SCD benchmark.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Scene Change Detection"
  - "Zero-Shot"
  - "SAM"
  - "Generalization Ability"
  - "Temporal Consistency"
date: 2026-05-08
content_hash: 39682fcf314b6995
---

# Towards Generalizable Scene Change Detection

**Conference**: CVPR 2025  
**arXiv**: [2409.06214](https://arxiv.org/abs/2409.06214)  
**Code**: Yes (mentioned in paper)  
**Area**: Image Segmentation / Change Detection  
**Keywords**: Scene Change Detection, Zero-Shot, SAM, Generalization Ability, Temporal Consistency

## TL;DR

Proposes GeSCF, the first zero-shot scene change detection framework, which leverages internal features of SAM to achieve cross-domain generalization and temporally consistent change mask generation, while defining a generalized SCD benchmark.

## Background & Motivation

Scene Change Detection (SCD) is crucial in fields such as visual surveillance, anomaly detection, and autonomous driving. However, existing SOTA methods heavily rely on target training datasets—performance drops drastically from 77.6% to 8.0% in unseen environments, and down to 4.6% under different temporal conditions. There are two core problems: (1) Poor generalization: Existing methods are trained and evaluated on specific datasets, failing to adapt to new scenes. (2) Temporal inconsistency: Inverting the input order yields different change masks. In an era where zero-shot generalization capabilities of 'anything' models like SAM are increasingly powerful, enabling 'anything' capability for SCD has become a key challenge. However, SAM is designed for single-image interactive vision segmentation, and direct application to bi-temporal change detection face fundamental differences. This work ingeniously bridges this gap through two innovative steps: initial pseudo-mask generation and geometric-semantic matching.

## Method

### Overall Architecture

The training-free GeSCF pipeline comprises two main steps: (1) Initial pseudo-mask generation: Intercepting key facet features from the intermediate layers of the SAM encoder, computing multi-head feature correlation of bi-temporal images to obtain a similarity map, and converting it into a binary pseudo-mask via an adaptive threshold function. (2) Geometric-semantic mask matching: Utilizing class-agnostic mask proposals from SAM for Geometric Intersection Matching (GIM), followed by refining the final change mask using Semantic Similarity Matching (SSM) of mask embeddings.

### Key Designs

1. **Multi-head Feature Correlation + Adaptive Thresholding**: Intercepting key facet features $\mathbf{F}_{l,n}$ from the intermediate layers of the SAM ViT encoder, and calculating the permutation-invariant feature correlation of the bi-temporal images $\bar{\mathbf{S}}_{l,n}^{t_0 \leftrightarrow t_1} = \mathbf{F}_{l,n}^{t_0} \cdot (\mathbf{F}_{l,n}^{t_1})^\top$. This operation is naturally permutation-invariant (ensuring temporal consistency), yielding a similarity map after multi-head aggregation. The crucial innovation is the skewness-based adaptive threshold $\mathbf{F}(\gamma) = b_\gamma + c \cdot \text{sign}(\gamma) \cdot s_\gamma \cdot \gamma$, which dynamically adjusts the threshold based on the shape of the similarity distribution, addressing the limitation of fixed thresholds across diverse scenes.

2. **Geometric Intersection Matching (GIM)**: Computing the intersection ratio $\alpha$ between each SAM mask and the pseudo-mask, keeping those exceeding a threshold $\alpha_t$, and upgrading pixel-level analysis to object-level change detection. Operating separately on both temporal images to preserve permutation invariance.

3. **Semantic Similarity Matching (SSM)**: GIM might include unchanged regions caused by pseudo-mask noise. By extracting bi-temporal mask embeddings $\mathcal{M}_{l,o}^{t_0}$ and $\mathcal{M}_{l,o}^{t_1}$, the cosine similarity is computed as a change confidence score. Low similarity confirms true changes, while high similarity filters out false positives. The final layer's embeddings are used, as semantic discrepancies are most prominent at this level.

### Loss & Training

- **Completely zero-training**, with no learnable parameters.
- Temporal consistency metric TC = $\frac{\mathbf{Y}_{\text{pred}}^{t0 \to t1} \cap \mathbf{Y}_{\text{pred}}^{t1 \to t0}}{\mathbf{Y}_{\text{pred}}^{t0 \to t1} \cup \mathbf{Y}_{\text{pred}}^{t1 \to t0}}$
- Proposed the GeSCD evaluation protocol: conducting comprehensive cross-domain evaluation across three standard datasets and the ChangeVPR dataset.

## Key Experimental Results

### Main Results

| Method | VL-CMU-CD IoU | TSUNAMI IoU | ChangeSim IoU | ChangeVPR IoU | TC |
|------|--------------|------------|-------------|-------------|-----|
| CSCDNet (In-domain) | 77.4 | - | - | - | Low |
| CSCDNet (Cross-domain) | - | 5.6 | 25.5 | Extremely low | 0.02 |
| **GeSCF** | **Competitive** | **Significant improvement** | **Significant improvement** | **Highest** | **1.0** |

### Ablation Study

| Component | Performance Impact |
|------|---------|
| Key vs Value facet | Key facet distinguishes change/no-change more clearly |
| Intermediate layers vs first/last layers | Intermediate layers achieve optimal performance |
| Fixed threshold vs adaptive threshold | Adaptive threshold yielding significant improvement |
| Without SSM | Increase in false positives |

### Key Findings

- GeSCF achieves an average improvement of 19.2% on existing SCD datasets, and 30.0% on ChangeVPR.
- Perfect temporal consistency (TC = 1.0) is achieved, owing to the naturally permutation-invariant feature correlation.
- Existing SOTA methods collapse in cross-domain scenarios, underscoring the urgent need for generalized SCD research.
- The key facet features of SAM's intermediate layers are highly sensitive to semantic changes, while being most robust against illumination and seasonal variations.

## Highlights & Insights

- Exploits the internal byproducts of SAM (facet features, class-agnostic masks, mask embeddings) to the extreme, realizing change detection with zero training cost.
- The permutation-invariance of feature correlation elegantly solves the temporal consistency issue—guaranteed naturally without external constraints.
- The skewness-based design of the adaptive threshold accounts for the relativity of "changes", which is highly intuitive.
- The GeSCD benchmark and the ChangeVPR dataset provide much-needed generalization evaluation standards for the field.

## Limitations & Future Work

- Zero-shot approaches may be less precise than supervised methods when in-domain training data is abundant.
- The high computational cost of the SAM encoder may limit its real-time application.
- Handling capabilities for extreme viewpoint variations or completely unaligned scene contents have not been fully validated.
- Fine-tuning strategies could be integrated with the zero-shot baseline to further enhance performance in specific domains.

## Related Work & Insights

- **vs CSCDNet/CDResNet**: Supervised methods perform exceptionally well on specific datasets but collapse in cross-domain settings; GeSCF sacrifices some in-domain precision to achieve comprehensive generalization ability.
- **vs SAM in Remote Sensing CD**: Remote sensing CD usually fine-tunes SAM adapters; GeSCF is the first work to utilize SAM in a zero-shot manner for scene change detection (SCD) in natural scenes.
- **vs Symmetric SCD Architectures**: Current symmetric designs rely on domain-specific inductive biases; GeSCF's permutation invariance is mathematically guaranteed.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first zero-shot SCD framework, with highly creative utilization of SAM features.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Introduces a complete benchmark, new dataset, and new evaluation metrics.
- **Writing Quality**: ⭐⭐⭐⭐ — Strong problem motivation with a clear framework.
- **Value**: ⭐⭐⭐⭐⭐ — A training-free generalization solution, deployable directly to any scene.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene](learning_4d_panoptic_scene_graph_generation_from_rich_2d_visual_scene.md)
- [\[CVPR 2025\] Scene-Centric Unsupervised Panoptic Segmentation](scene-centric_unsupervised_panoptic_segmentation.md)
- [\[AAAI 2026\] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts](../../AAAI2026/segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)
- [\[ICLR 2026\] S3OD: Towards Generalizable Salient Object Detection with Synthetic Data](../../ICLR2026/segmentation/s3od_towards_generalizable_salient_object_detection_with_synthetic_data.md)
- [\[ICML 2026\] Beyond Detection: A Structure-Aware Framework for Scene Text Tracking](../../ICML2026/segmentation/beyond_detection_a_structure-aware_framework_for_scene_text_tracking.md)

</div>

<!-- RELATED:END -->
