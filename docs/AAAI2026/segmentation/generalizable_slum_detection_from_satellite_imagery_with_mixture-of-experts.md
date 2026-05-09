---
title: >-
  [Paper Note] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts
description: >-
  [AAAI 2026][Segmentation][slum detection] This paper proposes GRAM (Generalized Region-Aware Mixture-of-Experts), a two-stage test-time adaptation framework. In the first stage, a MoE architecture is used to train region-specialized experts on million-scale satellite imagery from 12 cities. In the second stage, reliable pseudo-labels are selected via cross-region prediction consistency for self-training, enabling generalization of slum segmentation to unseen African cities.
tags:
  - AAAI 2026
  - Segmentation
  - slum detection
  - satellite image segmentation
  - mixture-of-experts
  - test-time adaptation
  - domain generalization
date: 2026-05-08
content_hash: 79b6d0669f84c44b
---

# Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts

**Conference**: AAAI 2026
**arXiv**: [2511.10300](https://arxiv.org/abs/2511.10300)
**Code**: [GitHub](https://github.com/DS4H-GIS/GRAM)
**Area**: Segmentation
**Keywords**: slum detection, satellite image segmentation, mixture-of-experts, test-time adaptation, domain generalization

## TL;DR

This paper proposes GRAM (Generalized Region-Aware Mixture-of-Experts), a two-stage test-time adaptation framework. In the first stage, a MoE architecture is used to train region-specialized experts on million-scale satellite imagery from 12 cities. In the second stage, reliable pseudo-labels are selected via cross-region prediction consistency for self-training, enabling generalization of slum segmentation to unseen African cities.

## Background & Motivation

Accurate detection of slums (informal settlements) is critical for urban planning and sustainable development. However:

**Extreme morphological heterogeneity**: Slums across different countries and cities vary greatly in architectural style, roofing materials, and spatial organization (e.g., gray square rooftops in coastal cities vs. brown rectangular rooftops in inland cities), making it difficult for a single model to generalize across regions.

**Scarcity of annotated data**: Pixel-level annotation is costly, particularly in low-income regions, where traditional survey methods are constrained by political sensitivity and logistical challenges.

**Poor transferability of existing methods**: Models trained in one country cannot reliably be applied to other regions, especially when there is a large distribution gap between source and target domains.

Core objective: achieve cross-region slum segmentation generalization **without any annotated data** in the target region.

## Method

### Overall Architecture

GRAM consists of two stages:

**Step 1 – Source Domain Training**: A segmentation model is trained on a multi-region dataset from 12 cities using a MoE architecture. MoE layers are integrated into the Transformer encoder of SegFormer, with each region having a dedicated gating network that routes tokens to the most relevant experts.

**Step 2 – Target Adaptation**: For unlabeled target images, an external region classifier identifies the most similar source-domain region. The corresponding experts generate pseudo-labels, cross-region prediction consistency is used to filter unreliable samples, and self-training fine-tuning is performed on the high-confidence subset.

### Key Designs

#### MoE Architecture and Adaptive Routing

Lightweight MoE blocks $\mathcal{F}$ are integrated into $L$ intermediate layers of the Transformer encoder. Each block contains $E$ MLP expert adapters:

- **Region-specific gating**: Each source-domain region $d$ has an independent gating network $g_d$ that computes the relevance of each expert for token features $z$.
- **Noisy Top-k routing**: Gaussian noise is added to prevent over-concentration of expert selection; softmax-normalized top-k expert weights are computed as:

$$\text{MoE}(z) = \sum_{e \in \text{top-}k(\tilde{g}_d(z))} \alpha_e \cdot \mathcal{E}_e(z)$$

- The shared backbone learns cross-region general features, while MoE experts learn region-specialized features.

#### Region-Aware Regularization

Two key regularization terms ensure expert diversity and region specialization:

1. **Mutual information regularization** $\mathcal{L}_{MI}$: Maximizes mutual information between regions and expert selections, ensuring different regions activate different expert sets and preventing mode collapse:

$$I^l(d;e) = \sum_{d=1}^{D}\sum_{e=1}^{E} P^l(d,e) \log\frac{P^l(d,e)}{P^l(d)P^l(e)}$$

2. **Region classification loss** $\mathcal{L}_{dom}$: An auxiliary region classifier predicts region labels from intermediate features of the shared backbone, enhancing routing quality.

Overall training objective: $\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_{MI} \cdot \mathcal{L}_{MI} + \lambda_{dom} \cdot \mathcal{L}_{dom}$

#### Pseudo-Label Filtering via Cross-Region Prediction Consistency

The core technique of the target adaptation stage:

1. An external region classifier $h_\psi$ predicts the most similar source-domain region $d_t$ for each target image, and generates pseudo-labels $\bar{y}_{d_t}$ via the corresponding routing.
2. A **stability score** is computed by routing the same image through all source-domain regions and evaluating mIoU consistency across different routing predictions:

$$s(x) = \sum_{d \neq d_t} \text{mIoU}(\bar{y}_{d_t}, \bar{y}_d)$$

3. The top $\rho_s$ fraction of samples by stability score forms a reliable dataset $\bar{\mathcal{D}}_t$, on which self-training fine-tuning is performed.

### Loss & Training

- Source domain training: pixel-level cross-entropy + MI regularization + region classification loss
- Target adaptation: fine-tuning on filtered high-confidence pseudo-labels using pixel-level cross-entropy
- Backbone: SegFormer; SGD optimizer (lr = 0.0001, momentum = 0.99)
- Hyperparameters: $\rho_s = 0.5$ (top 50% high-confidence samples), $E = 12$ experts, $k = 2$ top-k routing

## Key Experimental Results

### Main Results

**mIoU on three African test cities**:

| Method | Dar es Salaam | Kampala | Maputo |
|--------|--------------|---------|--------|
| Vanilla Source | 0.681 | 0.716 | 0.800 |
| MoE Source (w/o TTA) | 0.806 | 0.800 | 0.900 |
| TENT | 0.691 | 0.716 | 0.802 |
| CoTTA | 0.762 | 0.821 | 0.821 |
| BeCoTTA | 0.741 | 0.844 | 0.904 |
| **GRAM** | **0.859** | **0.870** | **0.907** |

GRAM outperforms all baselines across all cities, with the most significant improvement in Dar es Salaam (+5.3% vs. MoE Source, +11.8% vs. CoTTA).

**Slum class (minority class) IoU comparison (Dar es Salaam)**:

| Method | Slum IoU | Slum F1 |
|--------|----------|---------|
| Vanilla Source | 0.476 | 0.645 |
| BeCoTTA | 0.540 | 0.702 |
| **GRAM** | **0.752** | **0.859** |

### Ablation Study

**Ablation on Dar es Salaam**:

| Configuration | mIoU | F1 |
|---------------|------|-----|
| w/o $\mathcal{L}_{dom}$ | 0.836 | 0.906 |
| w/o $\mathcal{L}_{MI}$ | 0.734 | 0.823 |
| No Filtering | 0.818 | 0.893 |
| Confidence Filtering | 0.463 | 0.501 |
| Temporal Consistency | 0.837 | 0.907 |
| **Full GRAM** | **0.859** | **0.921** |

Key findings: $\mathcal{L}_{MI}$ is the most critical component (removing it causes a 12.5% drop in mIoU). Confidence-based filtering completely fails under domain shift (mIoU = 0.463), whereas cross-region consistency filtering proves far more reliable.

### Key Findings

- MoE Source vs. Vanilla Source: adding MoE alone substantially improves generalization (+12.5% mIoU on Dar es Salaam), demonstrating the value of region specialization.
- Region classifier predictions are highly consistent with geographic image similarity: coastal cities (Cape Town ↔ Dar es Salaam ↔ Maputo) and inland cities (Nairobi ↔ Kampala) naturally cluster together.
- Temporal tracking application: the slum rate in Kampala increased from 8.4% (2015) to 8.6% (2023); Maputo from 35.3% to 41.2%; Dar es Salaam decreased from 17.3% to 12.6%.

## Highlights & Insights

1. **Million-scale dataset contribution**: A large-scale slum segmentation dataset comprising 2.7M+ image patches across 12 cities is constructed, representing the largest benchmark in this domain.
2. **Elegant combination of MoE and TTA**: MoE naturally handles multi-region heterogeneity, and cross-expert consistency naturally provides a quality measure for pseudo-labels.
3. **Failure of confidence-based filtering**: The paper clearly demonstrates the fragility of entropy-based methods under domain shift, providing an important lesson for the TTA community.
4. **Societal value**: The framework supports slum monitoring in regions lacking official statistical data, with direct practical implications for urban policy-making.

## Limitations & Future Work

- Target domain evaluation is limited to three African cities; assessment on unseen cities in Asia or South America is absent.
- The quality of the region classifier $h_\psi$ directly affects routing accuracy, yet the robustness of the classifier itself is not thoroughly analyzed.
- The filtering ratio $\rho_s = 0.5$ is uniformly applied across all cities; adaptive adjustment could be considered.
- The stability score relies solely on mIoU; finer-grained pixel-level uncertainty estimation could be explored.
- Dataset annotations are partially derived from semi-supervised pseudo-labels, which may introduce annotation noise.

## Related Work & Insights

- **MoE in segmentation**: This work represents the first successful application of MoE + TTA to remote sensing segmentation, demonstrating the value of region-specialized experts for cross-domain generalization.
- **Relationship to BeCoTTA**: BeCoTTA also employs MoE adapters but lacks effective pseudo-label filtering; GRAM's cross-region consistency mechanism is the key differentiator.
- **Two-stage semi-supervised + fully supervised pipeline**: The strategy of first generating pseudo-labels via ST++ and then training a fully supervised model provides a practical paradigm for large-scale annotation-scarce scenarios.
- Inspiration: the cross-region MoE + consistency filtering approach is transferable to other remote sensing tasks (e.g., building detection, land use classification).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of MoE and cross-expert consistency TTA is novel; the stability score design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Large-scale dataset, comparison with multiple baselines, thorough ablation, and temporal tracking demonstration.
- Writing Quality: ⭐⭐⭐⭐ — Method motivation is clearly articulated; geographic visualizations are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Open-source dataset and code; substantial real-world societal impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](../../CVPR2026/segmentation/spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)
- [\[ICCV 2025\] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [\[CVPR 2026\] Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](../../CVPR2026/segmentation/generalizable_knowledge_distillation_from_vision_foundation_models_for_semantic_.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [\[AAAI 2026\] From Attribution to Action: Jointly ALIGNing Predictions and Explanations](from_attribution_to_action_jointly_aligning_predictions_and_explanations.md)

</div>

<!-- RELATED:END -->
