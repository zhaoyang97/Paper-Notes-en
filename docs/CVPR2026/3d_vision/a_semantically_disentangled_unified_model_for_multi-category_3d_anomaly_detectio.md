---
title: >-
  [Paper Note] A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection
description: >-
  [CVPR 2026][3D Vision][3D anomaly detection] This paper proposes SeDiR, a framework for semantically disentangled unified 3D anomaly detection, comprising three modules: Coarse-to-Fine Global Tokenization (CFGT)…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D anomaly detection"
  - "unified model"
  - "semantic disentanglement"
  - "inter-category entanglement"
  - "contrastive learning"
date: 2026-05-08
content_hash: 1e00b4e7cfb53d4c
---

# A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection

**Conference**: CVPR 2026
**arXiv**: [2603.25159](https://arxiv.org/abs/2603.25159)  
**Code**: [Project Page](https://spoiuy3.github.io/SeDiR/)  
**Area**: 3D Vision / Anomaly Detection
**Keywords**: 3D anomaly detection, unified model, semantic disentanglement, inter-category entanglement, contrastive learning

## TL;DR
This paper proposes SeDiR, a framework for semantically disentangled unified 3D anomaly detection, comprising three modules: Coarse-to-Fine Global Tokenization (CFGT), Category-Conditioned Contrastive Learning (C3L), and Geometry-Guided Decoder (GGD). SeDiR addresses the Inter-Category Entanglement (ICE) problem and outperforms the state of the art by 2.8% and 9.1% AUROC on Real3D-AD and Anomaly-ShapeNet, respectively.

## Background & Motivation
**Background**: 3D anomaly detection (3D-AD) aims to detect defects in 3D point clouds by training exclusively on normal data. Conventional approaches train a separate model per category, which incurs prohibitive maintenance costs in multi-category industrial settings.

**Necessity of Unified Models**: A single model covering multiple categories reduces system redundancy and improves deployment efficiency. Methods such as MC3D-AD have explored this direction but with limited performance.

**Core Problem — Inter-Category Entanglement (ICE)**:
- In unified models, latent features from different categories overlap in the embedding space (e.g., chicken/duck/gemstone are severely intermixed in t-SNE visualizations).
- This causes the model to reconstruct objects under incorrect category priors (e.g., parts of a chair reconstructed with table geometry).
- This failure is not one of anomaly detection per se, but rather one of establishing object identity.

**Key Insight**: Reconstruction failure occurs not because an object is anomalous, but because the model has not determined *what it is reconstructing* prior to reconstruction.

**Core Idea**: Understand before reconstruct — reframing unified 3D-AD as a problem of semantically conditioned reconstruction.

## Method

### Overall Architecture
Input point cloud → Multi-resolution neighborhood encoding (PointMAE) → CFGT generates category-aware global tokens → C3L disentangles category semantics → GGD reconstructs under disentangled semantics and geometric guidance → Reconstruction error serves as anomaly score.

### Key Designs
1. **Coarse-to-Fine Global Tokenization (CFGT)**:

    - **Multi-resolution neighborhood encoding**: Neighborhoods are constructed at symmetric resolutions $\mathcal{R} = \{k/2, k, 2k\}$ over shared center points and encoded with pretrained PointMAE, capturing multi-scale geometry from fine detail to global structure.
    - **Adaptive Context Token (ACT)**: A learnable token $\mathbf{t}_{\text{act}}$ is prepended to the reference-resolution sequence and aggregates global context after transformer encoding.
    - **Global representation**: Global average pooling at each resolution is concatenated with the ACT token: $\mathbf{f}_{\text{global}} = \text{concat}([\mathbf{g}^{(k)}, \mathbf{g}^{(2k)}, \mathbf{g}^{(k/2)}, \mathbf{t}^{\text{enc}}_{\text{act}}])$
    - **Cross-scale alignment loss**: $\mathcal{L}_{\text{cos}} = \frac{1}{g}\sum_{m=1}^{g}\sum_{r}[1 - \cos(\tilde{\mathbf{f}}_m^{(k)}, \tilde{\mathbf{f}}_m^{(r)})]$
    - **Auxiliary classification loss**: $\mathcal{L}_{\text{cls}} = \text{CrossEntropy}(\hat{\mathbf{y}}, \mathbf{y})$
    - **Design Motivation**: Local features are insufficient to distinguish category identity; multi-scale global aggregation is necessary to form instance-level semantic representations.

2. **Category-Conditioned Contrastive Learning (C3L)**:
   A dynamic buffer $\mathcal{B}$ of size 64 is maintained, and supervised contrastive learning is applied to global tokens $\mathbf{z}$:
    $$\mathcal{L}_{\text{scl}}(i) = \frac{1}{|\mathcal{P}(i)|}\sum_{\mathbf{z}_{\text{pos}} \in \mathcal{P}(i)} -\log \frac{\exp(\mathbf{z}_i^\top \mathbf{z}_{\text{pos}} / \tau)}{\sum_{\mathbf{z}_a \in \mathcal{A}(i)} \exp(\mathbf{z}_i^\top \mathbf{z}_a / \tau)}$$
    - Positives: same category; negatives: different categories.
    - Overall C3L objective: $\mathcal{L}_{\text{C3L}} = \lambda_{\text{scl}}\mathcal{L}_{\text{scl}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{cos}}\mathcal{L}_{\text{cos}}$
    - **Design Motivation**: Explicitly enforces intra-class compactness and inter-class separation, directly addressing the ICE problem.

3. **Geometry-Guided Decoder (GGD)**:
   The semantic prior $\mathbf{z}$ serves as query, with encoded feature sequences as key/value, augmented with a geometric bias:
    $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \beta \mathbf{B}_{\text{geo}}\right)\mathbf{V}$$
   where $\mathbf{B}_{\text{geo}}$ encodes local surface normals and curvature variations.
    - **Design Motivation**: Faithful reconstruction requires not only a correct semantic prior but also geometric evidence to guide attention.

### Loss & Training
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{C3L}} + \mathcal{L}_{\text{rec}}$$
- Reconstruction loss: $\mathcal{L}_{\text{rec}} = \frac{1}{g}\sum_j \|\hat{\mathbf{f}}_j^{(k)} - \mathbf{f}_j^{(k)}\|_2^2$
- At inference: reconstruction error + Gaussian pooling + normalization = anomaly score.

## Key Experimental Results

### Main Results (Real3D-AD, Object-level AUROC %)

| Method | Type | Airplane | Car | Duck | Fish | Gemstone | Mean |
|--------|------|----------|-----|------|------|----------|------|
| Group3AD | Category-specific | 74.4 | 72.8 | 67.9 | 97.6 | 53.9 | 75.1 |
| ISMP | Category-specific | 85.8 | 73.1 | 71.2 | 94.5 | 46.8 | 76.7 |
| MC3D-AD | Unified | 85.0 | 74.9 | 83.1 | 86.5 | 56.0 | 78.2 |
| **SeDiR** | **Unified** | **86.0** | **78.3** | **86.2** | **93.8** | **62.7** | **81.0** |

### Ablation Study

| Configuration | Key Metric (AUROC) | Note |
|---------------|--------------------|------|
| Baseline (w/o CFGT/C3L/GGD) | ~78.2 | Comparable to MC3D-AD |
| + CFGT | Improved | Global semantic representation is effective |
| + C3L | Further improved | t-SNE shows clear category separation |
| + GGD | **81.0** | Geometric guidance ensures reconstruction consistency |
| Correlation between classification accuracy and reconstruction error | Low classification score → High reconstruction error | Quantitatively validates the ICE problem |

### Key Findings
- The unified model surpasses all category-specific models: 81.0 vs. 76.7 (best category-specific).
- Improvements are most pronounced on visually similar categories (chicken, duck, gemstone) — precisely where ICE is most severe.
- t-SNE visualizations: severe intermixing of chicken/duck/gemstone in MC3D-AD → clear separation in SeDiR.
- Classification scores strongly correlate with reconstruction errors, validating the necessity of the understand-before-reconstruct paradigm.

## Highlights & Insights
- **The identification and characterization of the ICE problem** constitutes a key contribution: it reframes the fundamental bottleneck of unified 3D-AD from "how to reconstruct anomalies" to "how to establish object identity."
- The **understand-before-reconstruct paradigm** is intuitive and effective, aligning with human inspection practices.
- The combination of **multi-resolution encoding, global tokens, and contrastive learning** provides complete coverage of the pipeline from feature extraction to spatial separation to conditioned reconstruction.
- A unified model outperforming category-specific models suggests that cross-category learning itself is beneficial through shared generalization knowledge.

## Limitations & Future Work
- Category labels are required for contrastive learning, limiting applicability in label-free settings.
- With a very large number of categories, the dynamic buffer in C3L may be insufficient to cover all negative samples.
- The current method operates on point clouds only; RGB-D or multi-modal fusion may yield further improvements.
- Generalization to extremely rare or entirely unseen categories remains unanalyzed.

## Related Work & Insights
- This work transfers supervised contrastive learning ideas from 2D anomaly detection (e.g., SupCon) into the 3D domain.
- The observation of ICE generalizes to other cross-category unified models (e.g., unified object detection, unified segmentation).
- The understand-before-reconstruct paradigm may be applicable to other reconstruction-based methods.

## Rating
- Novelty: ⭐⭐⭐⭐ The identification of ICE is valuable; the method combination is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed comparison across 12 categories on Real3D-AD, with ablations and visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is exceptionally clear; figures and tables are polished.
- Value: ⭐⭐⭐⭐ Direct practical relevance to industrial 3D quality inspection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[ICCV 2025\] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](../../ICCV2025/3d_vision/unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)
- [\[CVPR 2026\] Changes in Real Time: Online Scene Change Detection with Multi-View Fusion](changes_in_real_time_online_scene_change_detection_with_multi-view_fusion.md)
- [\[CVPR 2026\] VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection](vggt-det_mining_vggt_internal_priors_for_sensor-geometry-free_multi-view_indoor_.md)

</div>

<!-- RELATED:END -->
