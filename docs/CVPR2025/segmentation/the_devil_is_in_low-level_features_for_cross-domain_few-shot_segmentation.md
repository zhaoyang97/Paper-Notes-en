---
title: >-
  [Paper Note] The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation
description: >-
  [CVPR 2025][Segmentation][Cross-Domain Few-Shot Segmentation] This paper deeply analyzes the phenomenon in CDFSS (Cross-Domain Few-Shot Segmentation) where "performance peaks in early training and then drops sharply". It finds that the culprit is the vulnerability of low-level features to domain shift, which leads to a sharp loss landscape. Consequently, two plug-and-play modules are proposed: LEM (for sharpness-aware minimization of low-level features via random convolution…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Cross-Domain Few-Shot Segmentation"
  - "Low-Level Features"
  - "Loss Landscape Flattening"
  - "Domain Generalization"
  - "Sharpness-Aware Minimization"
date: 2026-05-08
content_hash: 401de119af9c35cf
---

# The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2503.21150](https://arxiv.org/abs/2503.21150)  
**Code**: None  
**Area**: Segmentation / Few-Shot Learning  
**Keywords**: Cross-Domain Few-Shot Segmentation, Low-Level Features, Loss Landscape Flattening, Domain Generalization, Sharpness-Aware Minimization

## TL;DR

This paper deeply analyzes the phenomenon in CDFSS (Cross-Domain Few-Shot Segmentation) where "performance peaks in early training and then drops sharply". It finds that the culprit is the vulnerability of low-level features to domain shift, which leads to a sharp loss landscape. Consequently, two plug-and-play modules are proposed: LEM (for sharpness-aware minimization of low-level features via random convolution + FFT during training) and LCM (for directly calibrating segmentation results using low-level query features during testing), outperforming SOTA by an average of 3.71%/5.34% mIoU on four target domains.

## Background & Motivation

**Background**: Cross-Domain Few-Shot Segmentation (CDFSS) aims to transfer segmentation capability learned on a source domain (e.g., PASCAL VOC) to completely different target domains (e.g., medical images, remote sensing). Existing methods like PATNet, DR-Adapter, and APSeg have made progress, but all face a widely observed yet unresolved problem.

**Limitations of Prior Work**: On target domains (especially those with large domain gaps), model performance peaks extremely early in training (even within the 1st epoch) and then drops sharply as source-domain training continues. This implies that source training not only fails to help target generalization but instead "harms" it. Although early stopping can mitigate this, it requires parameter tuning for specific target domains, which violates the "one model fits all target domains" goal of CDFSS.

**Key Challenge**: While low-level features (edges, textures) are traditionally considered more general and transferable, experiments reveal that it is precisely these low-level features (shallow network layers) that absorb excessive domain-specific information from the source domain during training, leading to a sharp decline in generalization on the target domain.

**Goal**: (1) Disclose the root cause of the "early stopping phenomenon"; (2) Design targeted solutions to enable continuous performance improvement rather than degradation during sustained training.

**Key Insight**: The authors approach the problem from the perspective of loss landscape flatness, as prior domain generalization studies suggest that a flatter loss landscape corresponds to better domain generalization. Through layer-wise analysis, they find that the shallow layers suffer the largest performance drop under the same perturbation, indicating that the shallow layers lead to the sharpest loss landscape.

**Core Idea**: The issue resides in the shallow low-level features. During training, random convolution + FFT are utilized as shape-preserving domain perturbations to flatten the loss landscape of low-level features. During testing, low-level features of the target domain are directly leveraged to calibrate segmentation results.

## Method

### Overall Architecture

A meta-learning episodic paradigm is adopted, where each episode contains support and query sets. Support and query images extract features via a weight-shared backbone, generating coarse segmentation masks through a comparison module. During training, LEM applies domain transformation perturbations to the low-level features of the support. During testing, LCM utilizes low-level features of the query to calibrate the coarse segmentation results.

### Key Designs

1. **LEM (Low-level Enhancement Module, during training)**:

    - **Function**: Perturbs domain information of low-level features without changing semantic content, achieving sharpness-aware minimization specifically for low-level features.
    - **Mechanism**: A two-step operation. Step 1: Apply a random convolution to the shallow support features $F_s$ as $F_s' = F_s * \Theta$ ($\Theta$ sampled from $\mathcal{N}(0, \sigma^2)$) to generate a domain transformation effect. Step 2: Combine the phase spectrum of the original feature (which preserves shape/edges) with the amplitude spectrum of the perturbed feature (which alters domain/texture) via FFT: $F_s^t = \text{IFFT}(\mathcal{A}' e^{i\mathcal{P}})$, ensuring that shape information is not lost after the domain transformation.
    - **Design Motivation**: Random convolution can preserve shape while transforming texture/domain information, which is highly suitable for segmentation tasks (where segmentation depends on shape rather than texture). The phase-amplitude separation of FFT further ensures semantic consistency. The overall effect is equivalent to performing domain-aware SAM on low-level features.

2. **LCM (Low-level Calibration Module, during testing)**:

    - **Function**: Directly utilizes low-level query features of the target domain to supplement collapsed low-level information, calibrating the coarse segmentation result.
    - **Mechanism**: (1) Compute a confidence map from the coarse segmentation score map as $C_{i,j} = S_{i,j,1} - S_{i,j,0}$; (2) Crop the confidence map into patches and select the top-$K$ highest confidence patches as reliable foreground anchors; (3) Locate the corresponding patches in the low-level query feature map and calculate their cosine similarity with all patches; (4) Calibrate the foreground score with weighted similarities: $S'_{i,j,1} = S_{i,j,1} + \sum_k w_k(Sim_k - \beta_k)$.
    - **Design Motivation**: Since low-level features may completely collapse after domain migration, relying solely on high-level feature matching is unreliable. LCM bypasses the high-level features contaminated by training and directly leverages the low-level features (e.g., color and texture similarities) of target domain images to supplement segmentation cues.

3. **Hierarchical Diagnostic Analysis Framework**:

    - **Function**: Systematically validates the causal relationship of "low-level features causing performance degradation".
    - **Mechanism**: Multi-dimensional validation chain: (1) Visualizing feature maps across stages, revealing that shallow layers are completely inactive on the target domain; (2) Applying pixel disturbances to different layers to measure performance drops, showing the largest decline in shallow layers $\rightarrow$ sharpest loss landscape; (3) Comparing frozen vs. trained shallow layers, showing frozen shallow layers yield better performance; (4) Using CKA similarity to measure source-target feature distance, demonstrating that LEM increases cross-domain feature similarity.
    - **Design Motivation**: Establishing a solid causal understanding prior to proposing solutions, ensuring the method's design is well-founded.

### Loss & Training

Training utilizes the standard BCE loss. LEM is applied as data augmentation to shallow features and does not introduce additional loss terms. The random convolution kernel size is $3 \times 3$ with standard deviation $\sigma=0.1$. LCM is used only during testing, with hyperparameters $K=3$, $w=0.6$, and $\beta=0.7$. It supports both ResNet-50 and ViT-B/16 backbones.

## Key Experimental Results

### Main Results

| Method | Backbone | FSS-1000 | Deepglobe | ISIC | Chest X-ray | Mean (1-shot) | Mean (5-shot) |
|------|----------|----------|-----------|------|-------------|-------------|-------------|
| APSeg (CVPR24) | ViT-base | 79.71 | 35.94 | 45.43 | 84.10 | 61.30 | 65.09 |
| DRA (CVPR24) | Res-50 | 79.05 | 41.29 | 40.77 | 82.35 | 60.86 | 65.42 |
| LoEC (Ours) | ViT-base | **81.05** | **42.12** | **52.91** | **83.94** | **65.01** | **70.43** |
| LoEC (Ours) | Res-50 | 78.51 | 44.10 | 38.21 | 81.02 | 60.46 | 65.01 |

### Ablation Study

| LEM | LCM | ResNet Mean | ViT Mean |
|-----|-----|-----------|---------|
| ✗ | ✗ | 57.21 | 62.17 |
| ✓ | ✗ | 58.35 (+1.14) | 63.06 (+0.89) |
| ✗ | ✓ | 59.78 (+2.57) | 64.39 (+2.22) |
| ✓ | ✓ | **60.46** (+3.25) | **65.01** (+2.84) |

Frozen Shallow vs. Trained Shallow:

| Configuration | FSS-1000 | Deepglobe | ISIC | Chest | Mean |
|------|----------|-----------|------|-------|------|
| Train stage 1,2,3,4 | 78.86 | 39.44 | 35.76 | 72.49 | 56.64 |
| Freeze stage 1 | 78.88 | 39.90 | 37.00 | 72.12 | 56.98 |
| Freeze stage 1,2 | 78.91 | 40.00 | 35.49 | 74.44 | **57.21** |

### Key Findings

- LCM's contribution (+2.57/+2.22) is greater than LEM's (+1.14/+0.89), suggesting that directly supplementing target domain information during testing is more effective than enhancing robustness during training.
- The two modules are complementary: the improvement from LEM+LCM is greater than the sum of their individual improvements, showing they tackle the problem from different angles.
- The comparison between frozen shallow vs. trained shallow layers directly validates the conclusion that "training shallow layers is harmful".
- The performance gain is largest on the ISIC medical dataset, which is farthest from the source domain (52.91 vs 45.43 = +7.48), verifying the efficacy of the method for large domain gaps.
- CKA analysis shows that LEM indeed reduces the feature distance between the source and target domains.

## Highlights & Insights

- **Extremely In-depth Problem Diagnosis**: A complete causal chain is built from observation of phenomena $\rightarrow$ feature visualization $\rightarrow$ loss landscape analysis $\rightarrow$ layer-freezing experiments $\rightarrow$ CKA measurement. This "understand the problem before solving it" research paradigm is highly exemplary.
- **Ingenious Combination of Random Convolution + FFT**: Random convolution alters the domain but might lose shape, which is perfectly compensated by the phase-amplitude separation of FFT. As long as the domain is altered (amplitude spectrum changes) and the shape remains intact (phase spectrum is preserved), an ideal domain perturbation is achieved.
- **Novel Design of LCM**: Instead of trying to recover collapsed low-level features, it bypasses them during testing and directly utilizes the low-level similarities of active target domain images as segmentation cues. This pragmatic approach of "bypassing what cannot be easily repaired" is highly valuable.

## Limitations & Future Work

- The selection of top-$K$ patches in LCM relies on the quality of coarse segmentation results; if the coarse segmentation is entirely incorrect, the selected anchors will also be unreliable.
- Hyperparameters ($K$, $w$, $\beta$) are manually set and may not be optimal for different target domains.
- Current analysis focuses on ResNet/ViT architectures; whether the same phenomenon persists in other architectures (such as ConvNeXt) remains unverified.
- Future work can explore automatically detecting and quantifying the degree of domain-specific representation absorbed by shallow layers during training to achieve adaptive LEM intensity control.

## Related Work & Insights

- **vs. PATNet**: PATNet handles the domain gap through a domain-invariant feature transformation module, but the transformation is applied to all layers rather than focusing on shallow layers. LoEC's analysis reveals that the problem concentrates on shallow layers, making it more targeted.
- **vs. SSP (self-support prototype)**: LCM is similar to the self-support idea of SSP, but SSP uses query prototypes to match query features (which causes information loss), whereas LCM directly leverages low-level features for calibration.
- **vs. SAM (Sharpness-Aware Minimization)**: Standard SAM applies perturbations to the entire parameter space, while LoEC's LEM acts as a "domain-oriented SAM" in the low-level feature space, which is more targeted.
- The analytical framework of this paper can inspire similar problem diagnostics in other cross-domain tasks (e.g., cross-domain detection, cross-domain classification).

## Rating

- Novelty: ⭐⭐⭐⭐ The insight of identifying and validating "low-level features as the culprit" is highly valuable, and the solution is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed diagnostic experiments (visualization, perturbation analysis, layer-freezing, CKA), with a comprehensive ablation study.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem-oriented narrative structure is exceptionally clear, seamlessly progressing from phenomena to causes to solutions.
- Value: ⭐⭐⭐⭐ The plug-and-play design is easy to apply, and the diagnostic analysis framework is highly inspiring for the cross-domain learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [\[ICML 2025\] Self-Disentanglement and Re-Composition for Cross-Domain Few-Shot Segmentation](../../ICML2025/segmentation/self-disentanglement_and_re-composition_for_cross-domain_few-shot_segmentation.md)
- [\[ICML 2025\] Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](../../ICML2025/segmentation/adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)
- [\[CVPR 2026\] Cross-Domain Few-Shot Segmentation via Multi-view Progressive Adaptation](../../CVPR2026/segmentation/cross-domain_few-shot_segmentation_via_multi-view_progressive_adaptation.md)
- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)

</div>

<!-- RELATED:END -->
