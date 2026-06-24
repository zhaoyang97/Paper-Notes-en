---
title: >-
  [Paper Note] ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations
description: >-
  [ECCV2024][Remote Sensing][cross-view geo-localization] This paper proposes ConGeo, a model-agnostic single-view + cross-view contrastive learning framework. By enforcing feature consistency across different ground view variations at the same location, it enables a single model to achieve robust cross-view geo-localization under arbitrary orientations and fields of view (FoV).
tags:
  - "ECCV2024"
  - "Remote Sensing"
  - "cross-view geo-localization"
  - "contrastive learning"
  - "orientation invariance"
  - "field of view"
  - "image retrieval"
date: 2026-05-08
content_hash: 70de7994d1fde1bd
---

# ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations

**Conference**: ECCV2024  
**arXiv**: [2403.13965](https://arxiv.org/abs/2403.13965)  
**Code**: [eceo-epfl/ConGeo](https://eceo-epfl.github.io/ConGeo/)  
**Area**: Remote Sensing  
**Keywords**: cross-view geo-localization, contrastive learning, orientation invariance, field of view, image retrieval

## TL;DR

This paper proposes ConGeo, a model-agnostic single-view + cross-view contrastive learning framework. By enforcing feature consistency across different ground view variations at the same location, it enables a single model to achieve robust cross-view geo-localization under arbitrary orientations and fields of view (FoV).

## Background & Motivation

The goal of Cross-View Geo-Localization (CVGL) is to match ground-level images with georeferenced aerial images to determine locations. In real-world scenarios, ground images captured by users have arbitrary orientations and varying fields of view (FoVs) (e.g., mobile phones or vehicle cameras only cover 70°–180°). However, existing methods suffer from severe limitations:

1. **Orientation/FoV-Specific Training**: Existing methods such as DSM and SAIG-D require training separate models for each orientation configuration and FoV, failing to generalize to unseen view variations.
2. **Over-reliance on Spatial Correspondence**: Geometric features such as road directions in north-aligned training data act as "shortcuts" for models. Once the orientation changes, model performance drops precipitously (e.g., the R@1 of Sample4Geo drops from 98.7% to 16.3% under unknown orientations).
3. **Requirement of FoV Priors**: Some methods (e.g., DSM) require known FoV information during both training and testing, which is often unavailable in real-world scenarios.

## Core Problem

How to maintain robust cross-view retrieval performance under different orientations and FoVs using a **single model**? The key challenge lies in enabling the model to learn orientation-invariant and FoV-resilient feature representations.

## Method

### Overall Architecture

ConGeo is based on the classic Siamese network architecture (ground encoder + aerial encoder) and introduces a three-way input: the original north-aligned ground image $I_q$, the transformed ground image $I_q^*$ (with random orientation shift + FoV cropping), and the aerial reference image $I_r$. ConGeo is a model-agnostic learning objective that can be integrated into any baseline CVGL model.

### Key Designs

#### Ground View Transformation

A transformation $T$ is applied to the north-aligned panorama: first, a random angle $\theta$ is used for horizontal cyclic shifting to simulate unknown orientation, and then a FoV angle $\alpha$ is used for cropping to simulate limited field of view: $I_q^* = T_q(I_q | \theta, \alpha)$. During training, $\theta \in [0°, 360°)$ and $\alpha = 180°$.

#### Single-view Contrastive Learning

- **Ground view contrastive loss** $\mathcal{L}_{\text{single-q}}$: InfoNCE loss is applied between the transformed ground feature $q^*$ and the original ground feature set $Q$ to force different view variations of the same location to be close in the feature space.
- **Aerial view contrastive loss** $\mathcal{L}_{\text{single-r}}$: Contrastive loss is applied to two random data augmentation versions of the same aerial image to enhance the robustness of aerial features.

#### Cross-view Contrastive Learning

- **Baseline loss** $\mathcal{L}_{\text{vanilla}}$: The cross-view alignment loss between the original ground image and the corresponding aerial image (reusing the baseline method, such as InfoNCE in Sample4Geo).
- **Cross-view contrastive loss** $\mathcal{L}_{\text{cross}}$: Contrastive alignment between the transformed ground image $q^*$ and the aerial reference $R$ breaks geometric shortcuts and forces the model to focus on semantically consistent features.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{vanilla}} + w_1 \mathcal{L}_{\text{single-q}} + w_2 \mathcal{L}_{\text{single-r}} + w_3 \mathcal{L}_{\text{cross}}$$

Where $w_1 = 0.5$, $w_2 = 0.5$, and $w_3 = 0.25$. Single-view contrastive weights are set higher than cross-view contrastive weights. All temperature parameters are learnable.

## Key Experimental Results

### Main Results: CVUSA Dataset (Single Model, No FoV-Specific Training)

| Method | FoV=360° R@1 | FoV=180° R@1 | FoV=90° R@1 | FoV=70° R@1 | Average R@1 |
|------|-------------|-------------|------------|------------|---------|
| Sample4Geo (Baseline) | 16.3% | 4.1% | 2.5% | 1.5% | 6.1% |
| Sample4Geo + DA | 93.2% | 84.6% | 45.1% | 28.4% | 62.8% |
| **ConGeo** | **85.2%** | **92.3%** | **55.9%** | **37.1%** | **67.6%** |

### Stronger in FoV-Specific Training

| Method | FoV=360° R@1 | FoV=180° R@1 | FoV=90° R@1 | FoV=70° R@1 |
|------|-------------|-------------|------------|------------|
| SAIG-D | 72.0% | 52.5% | 26.7% | 20.9% |
| Sample4Geo | 93.3% | 84.6% | 55.1% | 40.9% |
| **ConGeo** | **96.6%** | **92.3%** | **55.5%** | **49.1%** |

### Maintaining Competitiveness in North-Aligned Settings

ConGeo achieves R@1=98.3% on CVUSA (vs Sample4Geo 98.7%) and R@1=71.7% on CVACT Test (vs Sample4Geo 71.5%), showing almost no performance degradation.

### Cross-Area Robustness (VIGOR Cross-Area)

| Method | FoV=360° R@1 | FoV=90° R@1 |
|------|-------------|------------|
| Sample4Geo | 9.0% | 0.5% |
| **ConGeo** | **16.2%** | **3.9%** |

### Generalization to Unseen Transformations (CVUSA)

ConGeo significantly outperforms data augmentation methods on unseen transformations like Random Zooming (68.7% vs baseline 48.2%) and Gaussian Noise (45.8% vs 0.2% for DA), demonstrating the transferability of the invariance brought by contrastive learning.

### Ablation Study

- The ground view contrastive loss ($\mathcal{L}_{\text{single-q}}$) is the most critical component, significantly improving performance across all FoVs.
- The cross-view contrastive loss ($\mathcal{L}_{\text{cross}}$) further improves performance for FoV=90° by about 35% on top of the single-view loss.
- Pure data augmentation leads to a significant performance drop in the north-aligned setting (98.7% -> 88.9%), whereas ConGeo only drops to 98.3%.

## Highlights & Insights

1. **Single Model for All Settings**: No need to train separate models for each orientation and FoV; a single ConGeo model can handle all scenarios from 70° to 360°.
2. **Model-Agnostic Plug-and-Play Design**: Successfully applied to both CNN (Sample4Geo) and ViT (TransGeo) architectures, as well as hybrid architectures (SAIG-D), bringing significant improvements to all.
3. **In-Depth Interpretability Analysis**: Through orientation-invariance curves and Grad-CAM activation maps, the paper clearly demonstrates how ConGeo shifts from relying on spatial shortcuts to leveraging semantically consistent features (e.g., from focusing on road directions to focusing on semantic objects like trees).
4. **Contrastive Learning > Data Augmentation**: Systematically demonstrates that the contrastive objective is superior to simple data augmentation, especially in terms of generalization capability on unseen transformations, which far exceeds DA.
5. **High Training Efficiency**: Requires only a single RTX 4090 to complete training in 60 epochs.

## Limitations & Future Work

1. **Slight Performance Drop in North-Aligned Settings**: ConGeo's R@1 under the north-aligned setting drops from 98.7% to 98.3%, which is the cost of breaking spatial shortcuts. However, in practical applications, the orientation is usually unknown.
2. **Sensitivity to Training FoV Selection**: The training fixes $\alpha = 180°$. Although it performs well across various FoVs, adaptive FoV sampling strategies remain unexplored.
3. **Exploration Limited to the Contrastive Learning Paradigm**: The paper mentions that other modality alignment methods, such as redundancy reduction, are also worth exploring.
4. **Challenges Remain in Extremely Low FoVs**: While performance is significantly improved at FoV=70°, the absolute value is still low (37.1%), and localization under scarce semantic information remains an open problem.
5. **Temporal Information Unexploited**: Real-world scenarios can utilize consecutive frames to provide additional constraints.

## Related Work & Insights

| Dimension | DSM | SAIG-D | Sample4Geo | ConGeo |
|------|-----|--------|-----------|--------|
| FoV Prior Required | Yes | Yes | No | **No** |
| Single Model Multi-FoV | No | No | No | **Yes** |
| Unknown Orientation R@1 (CVUSA) | 78.1% | 72.0% | 93.3% | **96.6%** |
| FoV=90° R@1 (CVUSA) | 16.2% | 26.7% | 55.1% | **55.9%** |
| Unseen Transformation Gen. | Poor | Poor | Medium | **Strong** |

The core advantage of ConGeo lies in extending contrastive learning from traditional cross-view alignment to alignment between different variations within the same view. This aligns with the philosophy of self-supervised methods like SimCLR/BYOL but is specifically tailored for geo-localization scenarios.

## Related Work & Insights

1. **New Application Paradigm of Contrastive Learning in Cross-Modal Retrieval**: It aligns not only two different modalities but also different variations within the same modality. This paradigm can be generalized to other cross-modal tasks (e.g., contrasting paraphrased variants of the same query in text-image retrieval).
2. **General Concept of "Breaking Shortcuts"**: Models relying on spatial correspondence shortcuts in training data is a widespread issue. ConGeo's approach of explicitly breaking such shortcuts via contrastive learning is worth emulating in other tasks.
3. **Connection to the Remote Sensing Field**: Cross-view localization is of direct value to autonomous driving navigation and UAV localization. ConGeo improves the practicality of these methods.
4. **Extensibility to Video-Based Localization**: Combining temporal information with ConGeo's view-invariance shows potential for video-level geo-localization.

## Rating
- Novelty: ⭐⭐⭐⭐ — The approach is clear and intuitive, and the contrastive learning framework is well-designed, although the core component (InfoNCE) is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Very comprehensive experiments with four datasets, three baseline models, and extensive ablation and interpretability analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-explained problem motivation, and the analysis section (Sec 6) is particularly outstanding.
- Value: ⭐⭐⭐⭐ — High practicality. The plug-and-play design lowers the application threshold, and a single model handling multiple configurations is highly attractive for engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SinGeo: Unlock Single Model's Potential for Robust Cross-View Geo-Localization](../../CVPR2026/remote_sensing/singeo_unlock_single_models_potential_for_robust_cross-view_geo-localization.md)
- [\[CVPR 2026\] RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](../../CVPR2026/remote_sensing/rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)
- [\[ECCV 2024\] Adapting Fine-Grained Cross-View Localization to Areas without Fine Ground Truth](adapting_fine-grained_cross-view_localization_to_areas_without_fine_ground_truth.md)
- [\[AAAI 2026\] UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization](../../AAAI2026/remote_sensing/uniabg_unified_adversarial_view_bridging_and_graph_correspondence_for_unsupervis.md)
- [\[CVPR 2026\] Geo2: Geometry-Guided Cross-view Geo-Localization and Image Synthesis](../../CVPR2026/remote_sensing/geo2_geometry-guided_cross-view_geo-localization_and_image_synthesis.md)

</div>

<!-- RELATED:END -->
