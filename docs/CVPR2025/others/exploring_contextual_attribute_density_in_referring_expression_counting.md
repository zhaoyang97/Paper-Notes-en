---
title: >-
  [Paper Note] Exploring Contextual Attribute Density in Referring Expression Counting (CAD-GD)
description: >-
  [CVPR 2025][Referring Expression Counting] The concept of Contextual Attribute Density (CAD) is proposed to enhance referring expression counting. By incorporating three modules—a U-shape density estimator, CAD attention, and dynamic query initialization—the approach reduces counting errors on the REC-8K dataset by approximately 30% compared to GroundingREC (MAE decreases from 6.80 to 5.43).
tags:
  - "CVPR 2025"
  - "Referring Expression Counting"
  - "Contextual Attribute Density"
  - "Open-World Detection"
  - "GroundingDINO"
  - "Density Map"
date: 2026-05-08
content_hash: 909a2ea77db135c5
---

# Exploring Contextual Attribute Density in Referring Expression Counting (CAD-GD)

**Conference**: CVPR 2025  
**arXiv**: [2503.12460](https://arxiv.org/abs/2503.12460)  
**Code**: [github.com/Xu3XiWang/CAD-GD](https://github.com/Xu3XiWang/CAD-GD)  
**Area**: Other  
**Keywords**: Referring Expression Counting, Contextual Attribute Density, Open-World Detection, GroundingDINO, Density Map

## TL;DR

The concept of Contextual Attribute Density (CAD) is proposed to enhance referring expression counting. By incorporating three modules—a U-shape density estimator, CAD attention, and dynamic query initialization—the approach reduces counting errors on the REC-8K dataset by approximately 30% compared to GroundingREC (MAE decreases from 6.80 to 5.43).

## Background & Motivation

**Background**: Referring Expression Counting (REC) is an emerging counting task that requires counting objects of specific attributes based on fine-grained textual descriptions (e.g., "walking person" instead of a simple "person"). GroundingREC is the first REC baseline based on GroundingDINO.

**Limitations of Prior Work**: GroundingREC suffers from two types of errors when handling fine-grained attributes: (1) over-counting, where it over-focuses on category information while ignoring fine-grained attributes and thus includes objects with incorrect attributes; and (2) under-counting, where objects with the specified attributes are missed due to occlusions and scale variations.

**Key Challenge**: REC is inherently a detection-by-counting pipeline (one-to-one matching) and lacks awareness of spatial density distribution. While traditional counting methods have demonstrated that "visual density" is crucial for scale-robust spatial distribution modeling, existing open-world models ignore this capability.

**Key Insight**: Analogous to the concept of "visual density", "Contextual Attribute Density" (CAD) is defined to measure the information intensity of a specific fine-grained attribute across visual regions of different scales. Modeling CAD guides the model to more accurately align attribute information with visual patterns.

**Core Idea**: Introduce attribute-level density map supervision to open-world detectors, enabling them to perceive the spatial distribution of attributes corresponding to fine-grained textual descriptions.

## Method

### Overall Architecture

The CAD-GD framework is built upon GroundingDINO: after extracting image and text features via their respective backbones, multi-scale visual features $\{F_{vi}\}_{i=1}^{4}$ and text features $F_t$ are obtained through the Feature Enhancer. Then, CAD information is injected through three main modules: the CAD generation module produces density features, the CAD attention module enhances visual features, and the CAD dynamic query module initializes decoder queries.

### Key Designs

1. **CAD Generation Module (U-shape CADE)**: It first projects visual features into the text space to calculate similarity $S_i = \text{Proj}(F_{vi}) \cdot F_t$, and then feeds the similarity features along with visual features into a U-shape estimator to generate multi-scale CAD features $\{D_i\}_{i=1}^{4}$. Finally, it outputs density maps supervised by an $\ell_2$ loss (where ground-truth density maps are generated via Gaussian kernels).

2. **CAD Attention Module**: This operates in two steps: spatial attention uses channel pooling (max+avg) on CAD features to generate spatial weight maps that enhance foreground regions; channel attention applies channel-level weighting via a shared MLP to the features enhanced by spatial attention, distinguishing different attributes across scales.

3. **CAD Dynamic Query Initialization**: It first dynamically initializes query content using text features (Text Init, $\dot{Q} = (Q \times (F_t \times M)^\top) \times F_t$), and then further refines it using CAD features via cross-attention (Density Init). This makes queries for different referring expressions more distinguishable in the feature space.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{loc}} + \alpha \cdot \mathcal{L}_{\text{density}}$$

Where $\mathcal{L}_{\text{loc}}$ is the standard localization loss (Hungarian matching + L1/GIoU), and $\mathcal{L}_{\text{density}} = \|D_{\text{pred}} - D_{\text{gt}}\|_2^2$ is the $\ell_2$ regression loss of the density map. During training, the visual and text backbones are frozen. The model is trained using AdamW with a learning rate of 1e-5 for 20 epochs, decaying by a factor of 10 at the 10th epoch.

## Key Experimental Results

| Method | Backbone | Val MAE↓ | Val RMSE↓ | Val F1↑ | Test MAE↓ | Test RMSE↓ | Test F1↑ |
|------|----------|----------|-----------|---------|-----------|------------|----------|
| GroundingDINO | Swin-T | 9.03 | 21.98 | 0.65 | 8.88 | 21.95 | 0.66 |
| GroundingREC | Swin-T | 6.80 | 18.13 | 0.68 | 6.50 | 19.79 | 0.69 |
| **CAD-GD** | Swin-T | **5.43** | **15.01** | **0.70** | **5.29** | **17.08** | **0.72** |
| GroundingREC* | Swin-B | 5.66 | 15.24 | 0.71 | 5.42 | 18.47 | 0.70 |
| **CAD-GD** | Swin-B | **4.83** | **13.52** | **0.75** | **4.94** | **14.65** | **0.76** |

### Ablation Study

| Module Combination | Val MAE | Val RMSE | Val F1 |
|---------|---------|----------|--------|
| Baseline (w/o CAD) | 6.52 | 17.72 | 0.665 |
| +CAD Generation | 6.17 | 16.38 | 0.673 |
| +Spatial Attention | 5.88 | 16.43 | 0.691 |
| +Channel Attention | 5.61 | 16.28 | 0.690 |
| +Text Init | 5.67 | 14.43 | 0.690 |
| +Density Init | 5.43 | 15.01 | 0.700 |
| +Density Inference Strategy | **4.83** | **13.52** | 0.695 |

### Key Findings

- The density map inference strategy (substituting thresholding with density map estimation for counting) yields an additional 11% reduction in MAE.
- The CAD density map can distinguish spatial distributions of different attributes within the same category (e.g., "bluish pen" vs "greenish pen").
- It also outperforms GroundingREC in zero-shot counting on FSC-147 (MAE of 9.30 vs 10.06).

## Highlights & Insights

- **Conceptual Innovation**: Introduces density estimation into cross-modal referring expression counting for the first time, defining the new concept of "Contextual Attribute Density".
- **Convincing Query Visualization**: t-SNE visualization clearly demonstrates that queries for different attributes can be effectively separated after CAD initialization.
- **Plug-and-Play**: The CAD module can enhance any open-world detector based on DETR-like structures.

## Limitations & Future Work

- Ground-truth density maps use a fixed-size Gaussian kernel ($\sigma=15$), which does not adapt to target scales.
- The density inference strategy improves counting but slightly degrades localization metrics, showing a mismatch between the two.
- Validation is only performed on REC-8K (~8000 images), which has a relatively small dataset scale.
- There is still room for improvement regarding complex semantic combinations of unrelated attributes (e.g., negative expressions like "not in a bus").

### Zero-Shot Counting Generalization (FSC-147)

| Method | Val MAE | Val RMSE | Test MAE | Test RMSE |
|------|---------|----------|----------|-----------|
| GroundingREC | 10.06 | 58.62 | 10.12 | 107.19 |
| CountGD | 12.14 | 47.51 | 12.98 | 98.35 |
| **CAD-GD** | **9.30** | **40.96** | **10.35** | **86.88** |

## Related Work & Insights

- **GroundingDINO / GroundingREC**: Open-world detection $\rightarrow$ REC baseline.
- **Density Estimation Counting**: CounTR, LOCA, CACViT, DAVE, etc. — proving the value of density features for counting tasks.
- **Text-Guided Counting**: CLIP-Count, CounTX, CountGD, VLCounter — cross-modal counting.
- **Density Modeling in Other Tasks**: DQ-DETR (small object detection), Cholakka (instance segmentation) — illustrating the universality of density priors.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of CAD is novel, and the fusion of density + detection is inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies, including zero-shot generalization validation.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured, with rich visualizations.
- Value: ⭐⭐⭐⭐ Provides a fresh perspective for open-world counting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](../../NeurIPS2025/others/contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[ACL 2025\] Entropy-UID: A Method for Optimizing Information Density](../../ACL2025/others/entropy-uid_a_method_for_optimizing_information_density.md)
- [\[CVPR 2026\] Neural Mixture Density Processes](../../CVPR2026/others/neural_mixture_density_processes.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](../../ACL2025/others/contextual_experience_replay_for_self-improvement_of_language_agents.md)

</div>

<!-- RELATED:END -->
