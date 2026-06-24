---
title: >-
  [Paper Note] Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction
description: >-
  [CVPR 2025][Autonomous Driving][HD Map Construction] Ours proposes UIGenMap, which obtains explicit structural features through an uncertainty-aware perspective view (PV) detection branch, constructs PV prompts based on uncertainty weights to inject into the BEV map decoder, and incorporates Mimic Query distillation for real-time inference, achieving a +5.7 mAP generalization performance improvement on geographically disjoint data splits.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "HD Map Construction"
  - "Uncertainty Estimation"
  - "Perspective View Structure Injection"
  - "Generalization Ability"
  - "Distillation"
date: 2026-05-08
content_hash: d78d0e61030a0d5a
---

# Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction

**Conference**: CVPR 2025  
**arXiv**: [2503.23109](https://arxiv.org/abs/2503.23109)  
**Code**: [https://github.com/xiaolul2/UIGenMap](https://github.com/xiaolul2/UIGenMap)  
**Area**: Autonomous Driving  
**Keywords**: HD Map Construction, Uncertainty Estimation, Perspective View Structure Injection, Generalization Ability, Distillation

## TL;DR

Ours proposes UIGenMap, which obtains explicit structural features through an uncertainty-aware perspective view (PV) detection branch, constructs PV prompts based on uncertainty weights to inject into the BEV map decoder, and incorporates Mimic Query distillation for real-time inference, achieving a +5.7 mAP generalization performance improvement on geographically disjoint data splits.

## Background & Motivation

**Background**: Online HD map vectorization has become a crucial direction in autonomous driving perception. Mainstream methods utilize Transformers to transform perspective view (PV) image features into the bird's-eye view (BEV) space, followed by a decoder to predict map elements. Methods such as MapTR, MapTRv2, GeMap, and StreamMapNet have achieved continuous performance improvements on standard benchmarks.

**Limitations of Prior Work**: Existing public datasets (e.g., nuScenes) contain significant geographical overlap between training and validation sets, causing models to memorize similar scenes rather than genuinely learn road structures. Under geographically disjoint (geo-based) data splits, the performance of existing methods degrades significantly. Furthermore, learning-based PV-to-BEV transformation inevitably introduces geometric errors and the loss of texture details.

**Key Challenge**: The models' excessive reliance on training data distribution leads to insufficient generalization capability; implicit PV-to-BEV transformation loses valuable structural information. There is an urgent need for a method that can both adapt to changes in feature distributions across different driving scenarios and compensate for the explicit structural information lost during BEV transformation.

**Goal**: (1) Utilize uncertainty modeling to enhance the model's dynamic adaptability to different scenarios; (2) Introduce PV explicit structural information to compensate for BEV map predictions; (3) Ensure real-time performance during inference.

**Key Insight**: Uncertainty estimation can learn statistical means and variances to achieve dynamic resampling based on probability distributions, endowing the model with dynamic adaptability in unfamiliar environments. Meanwhile, 2D perspective view detection captures more intuitive semantic and angular structural information, which can serve as a reliable compensation for BEV predictions.

**Core Idea**: An Uncertainty-Guided perspective view structure injection strategy (UIGenMap) is proposed. It designs an Uncertainty-Aware Decoder (UA-Decoder) in both PV and BEV spaces, constructs PV prompts based on uncertainty weighting to compensate for BEV map predictions via a hybrid injection mechanism, and finally eliminates the extra overhead of the PV branch during inference through lightweight Mimic Query distillation.

## Method

### Overall Architecture

The input consists of multi-camera surround-view images, and the output is vectorized map elements (category labels + ordered point sequences) in the BEV space. The architecture includes: (1) an image backbone to extract PV features; (2) BEV features constructed via cross-attention between PV features and learnable BEV queries; (3) a PV detection branch using UA-Decoder to obtain PV instance coordinates and uncertainty; (4) a UI2DPrompt module to construct PV prompts; (5) a hybrid injection module that integrates PV prompts into BEV features and map queries; (6) a BEV UA-Decoder to predict the final map; and (7) an MQ-Distillation module to distill the PV prompt knowledge during training. At inference time, only the distilled Mimic Query is used to replace the PV branch.

### Key Designs

1. **Uncertainty-Aware Decoder (UA-Decoder)**:

    - **Function**: Injects probability modeling at the instance and point levels, endowing the model with dynamic self-adaptation capabilities.
    - **Mechanism**: Design UA-Attention at the feature level—changing the deterministic attention weights $\alpha_i$ in deformable attention to Gaussian distribution sampling $\alpha_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$, where the mean and variance are predicted from queries via MLPs using the reparameterization trick. Design UA-Head at the output level—each point predicts not only the coordinates $(\hat{p}_x^i, \hat{p}_y^i)$ but also the uncertainty $(\sigma_x^i, \sigma_y^i)$, modeled as a Laplace distribution. The network is trained with a joint NLL loss and point regression loss.
    - **Design Motivation**: In highly diverse driving scenarios, deterministic attention weights fail to adapt to challenging environments. Probabilistic sampling offers dynamic regulation capabilities, and uncertainty outputs provide reliable confidence indicators for subsequent feature selection.

2. **UI2DPrompt (Uncertainty-Guided 2D Prompt Construction)**:

    - **Function**: Constructs reliable structural prompts from PV detection results to compensate for BEV predictions.
    - **Mechanism**: First, high-confidence PV instances are filtered based on classification scores, and PV coordinates are projected onto the BEV coordinate system using IPM (Inverse Perspective Mapping). The projected coordinates and uncertainty parameters are encoded and concatenated respectively into point-level embeddings $e_{pv}^i$. The uncertainty serves as a weight: $\omega_{pv}^i = \exp((\|\sigma_{pv}^i\|_2)^{-1} / \sum(\|\sigma_{pv}^i\|_2)^{-1})$, where lower uncertainty yields higher weight. The enhanced PV prompt is finally defined as $\tilde{e}_{pv}^i = \omega_{pv}^i \cdot e_{pv}^i + e_m^i$ (where $e_m^i$ is the Mimic Query).
    - **Design Motivation**: Directly using PV detection results may introduce errors. Uncertainty-guided weighting can amplify reliable information and suppress unreliable predictions.

3. **Hybrid Injection and Mimic Query Distillation**:

    - **Function**: Efficiently injects PV prompts into the BEV prediction pipeline and eliminates additional inference computation through distillation.
    - **Mechanism**: Hybrid injection includes P2BEV (point-level PV prompt incorporated into BEV features via cross-attention) and P2Q (instance-level PV prompt injected into map queries via cross-attention). MQ-Distillation defines learnable Mimic Queries $e_m^i$ and an MLP learner $h(\cdot)$ to distill the structural features of the PV prompt using an MSE distillation loss $\mathcal{L}_{distill} = \|e_{pv}^i - h(e_m^i)\|^2$. During inference, the PV branch is substituted entirely with the Mimic Queries.
    - **Design Motivation**: The PV branch increases computational overhead. Distilling it into lightweight queries preserves real-time inference capabilities (UIGenMap-d achieves 12.2 FPS vs. 8.2 FPS for the full version).

### Loss & Training

- **Total Loss**: $\mathcal{L}_{map} = \lambda_1 \mathcal{L}_{pts} + \lambda_2 \mathcal{L}_{cls} + \mathcal{L}_{nll} + \mathcal{L}_{distill}$
- $\mathcal{L}_{pts}$: Manhattan distance loss for point regression.
- $\mathcal{L}_{cls}$: Focal loss for map classification.
- $\mathcal{L}_{nll}$: Negative log-likelihood loss for uncertainty training (Laplace distribution).
- $\mathcal{L}_{distill}$: MSE distillation loss for Mimic Queries.
- During inference, only the Mimic Queries are used, and the learned uncertainty facilitates dynamic sampling.

## Key Experimental Results

### Main Results (nuScenes Region-Based / City-Based)

| Method | Backbone | Region mAP | City mAP | FPS |
|------|----------|-----------|---------|-----|
| MapTR | R50 | 20.9 | 15.0 | 15.8 |
| MapTRv2 | R50 | 28.9 | 21.8 | 12.9 |
| StreamMapNet | R50 | 34.1 | 19.3 | 13.3 |
| GeMap | R50 | 27.3 | 18.6 | 11.6 |
| **UIGenMap-d** | R50 | **39.3 (+5.2)** | **22.7 (+3.4)** | 12.2 |
| **UIGenMap** | R50 | **39.8 (+5.7)** | **23.6 (+4.3)** | 8.2 |

### Ablation Study

| Component | Region mAP | Description |
|------|-----------|------|
| Baseline (StreamMapNet) | 34.1 | — |
| + UA-Decoder | ~36 | Uncertainty modeling improves adaptability |
| + PV Branch + UI2DPrompt | ~38 | Significant improvement from PV structural compensation |
| + Hybrid Injection (P2BEV+P2Q) | ~39 | Dual-path injection outperforms single-path |
| + MQ-Distillation | 39.3 | Distilled version achieves accuracy close to the full version |

### Key Findings

- The performance gain is most significant under geographically disjoint splits (+5.7 mAP), validating the clear effectiveness of the method in improving generalization capability.
- UIGenMap-d (distilled version) loses only 0.5 mAP on region-based evaluation but improves FPS from 8.2 to 12.2, demonstrating high practicality.
- The pedestrian crossing (Pedestrian) category shows the largest improvement (from 32.2 to 40.3), demonstrating that PV structural compensation is highly effective for fine-grained elements.
- Using a SwinT backbone can further boost performance to 40.6 mAP.
- Consistent performance improvements are also achieved on the Argoverse2 dataset.

## Highlights & Insights

- **Generalization-oriented experimental design**: Unlike most HD mapping works that evaluate on standard splits, this work focuses on geographically disjoint splits, which possess higher practical significance.
- **Dual utility of uncertainty**: It is used both for dynamic attention resampling to enhance adaptability and for confidence-weighted selection of PV prompts.
- **Highly practical distillation strategy**: UIGenMap-d does not require the PV branch during inference, maintaining an FPS comparable to the baseline, which is suitable for deployment.
- Explicit structural information from the PV space indeed compensates for information loss in BEV transformations, which is a convincing approach.

## Limitations & Future Work

- IPM assumes a flat ground, which might be inaccurate in scenarios such as ramps.
- The PV detection branch increases training time and GPU memory footprint.
- The accuracy of uncertainty estimation relies on the adequacy of training data.
- In the future, stronger PV-to-BEV transformation schemes can be explored to replace IPM.
- Integrating temporal information with uncertainty estimation can be considered to further enhance generalization.

## Related Work & Insights

- **StreamMapNet**: The baseline of this paper; UIGenMap introduces the PV branch and uncertainty modeling on top of it.
- **BEVFormerv2 / SimMoD**: The paradigm of using PV detection to assist BEV perception has been applied in 3D object detection; this work extends it to HD map construction.
- **MapQR / GeMap**: These optimize map construction from the perspectives of decoder design and geometric relations, complementing the uncertainty perspective of this work.
- Insight: For generalization issues, explicit structural priors combined with uncertainty estimation represent an effective joint strategy.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 7 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 7 |
| Practical Value | 8 |
| Overall Rating | 7.6 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] SafeMap: Robust HD Map Construction from Incomplete Observations](../../ICML2025/autonomous_driving/safemap_robust_hd_map_construction_from_incomplete_observations.md)
- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [\[ICCV 2025\] DAMap: Distance-aware MapNet for High Quality HD Map Construction](../../ICCV2025/autonomous_driving/damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)

</div>

<!-- RELATED:END -->
