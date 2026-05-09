---
title: >-
  [Paper Note] GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration
description: >-
  [CVPR 2026][Self-Supervised Learning][Geochemical anomaly detection] This paper introduces GeoChemAD, the first open-source multi-region multi-element geochemical anomaly detection benchmark (8 subsets covering three sampling media—sediment/rockchip/soil—and four target elements—Au/Cu/Ni/W), and proposes GeoChemFormer, a two-stage Transformer framework that first learns spatial context and then models inter-element dependencies, achieving a mean AUC of 0.7712 that surpasses all baselines.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Geochemical anomaly detection
  - mineral exploration
  - benchmark dataset
  - self-supervised pretraining
  - Transformer
date: 2026-05-08
content_hash: c937af2686a9990c
---

# GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration

**Conference**: CVPR 2026
**arXiv**: [2603.13068](https://arxiv.org/abs/2603.13068)
**Code**: [github.com/yihaoding/geochemad](https://github.com/yihaoding/geochemad)
**Area**: Anomaly Detection / Geoscience / Self-Supervised Learning
**Keywords**: Geochemical anomaly detection, mineral exploration, benchmark dataset, self-supervised pretraining, Transformer

## TL;DR
This paper introduces GeoChemAD, the first open-source multi-region multi-element geochemical anomaly detection benchmark (8 subsets covering three sampling media—sediment/rockchip/soil—and four target elements—Au/Cu/Ni/W), and proposes GeoChemFormer, a two-stage Transformer framework that first learns spatial context and then models inter-element dependencies, achieving a mean AUC of 0.7712 that surpasses all baselines.

## Background & Motivation
Geochemical anomaly detection is a core step in mineral exploration: surface-sample concentrations deviating from the regional baseline may indicate mineralization. Existing research is constrained by two major issues: (1) the vast majority of methods are evaluated on a single-region setting (typically gold-deposit data from the China Geological Survey), making model generalizability impossible to assess; (2) nearly all datasets are closed-source or proprietary, preventing reproducibility and fair comparison. Furthermore, although unsupervised methods offer advantages in generalizability, the anomalies they detect may be unrelated to the target mineralization elements.

## Core Problem
How to construct a standardized and reproducible benchmark for unsupervised geochemical anomaly detection? How to design an unsupervised detection framework that is both spatially context-aware and capable of distinguishing anomalies relevant to target mineralization elements?

## Method

### Overall Architecture
GeoChemFormer operates in two stages: Stage 1 (Spatial Context Learning, SCL) learns latent representations of local geochemical co-variation patterns; Stage 2 (Element Dependency Modeling) detects anomalies conditioned on the learned spatial context.

### Key Designs
1. **GeoChemAD Dataset**: Sourced from publicly available government data released by the Geological Survey of Western Australia (GSWA), comprising 8 subsets—2 sediment (sed1/sed2), 3 rockchip (rock1/2/3), and 3 soil (soil1/2/3)—covering areas ranging from 6 km² to 8,500 km², with sample counts from 224 to 21,040, corresponding to four target mineralization elements: Au/Cu/Ni/W. Each subset provides geochemical concentration CSVs and known mineralization site CSVs.
2. **Stage 1 — Spatial Context Learning**: For each query location $p_i$, its $K$ nearest neighbors are retrieved via a KD-tree. A token sequence $\mathcal{S}=[e, q_i, t_1,...,t_K]$ (target-element token + query-location token + neighborhood tokens) is constructed and encoded by an $L$-layer Transformer encoder to produce the spatial context representation $q_i'$. The self-supervised objective is to predict the target-element concentration at the query location from its neighborhood alone, with loss $\mathcal{L}_{sc} = \frac{1}{N}\sum({\hat{y}_i - y_i})^2$.
3. **Stage 2 — Element Dependency Modeling**: The input sequence is $\mathcal{S}'=[W_g q_i', u_1,...,u_c]$ (geo-context token + $c$ element tokens). Each element token is formed by concatenating an element identity embedding with the scalar concentration value and projecting the result. A Transformer encoder learns inter-element dependencies conditioned on spatial context. The anomaly score is the reconstruction MSE over all elements: $s_i = \frac{1}{C}\sum_{c}(x_{i,c}-\hat{x}_{i,c})^2$.
4. **Data Preprocessing Module**: Handles missing values (e.g., −9999), the closure problem (CLR/ILR transforms), feature selection (manual/PCA/causal discovery/LLM-assisted), and spatial interpolation (IDW/Kriging).

### Loss & Training
- Stage 1: MSE regression loss (predicting target-element concentration)
- Stage 2: MSE reconstruction loss (reconstructing all element concentrations)
- Implemented in PyTorch; trained on a single NVIDIA RTX A6000 (48 GB)
- Evaluation metric: AUC (averaged over 20 random background samplings)

## Key Experimental Results

| Method Category | Representative Method | Mean AUC |
|---|---|---|
| Statistical | Z-score / Mahalanobis / KNN | 0.50–0.58 |
| Classical ML | Isolation Forest / One-Class SVM | ~0.58–0.62 |
| Deep Generative | AE / VAE / VAE-GAN | 0.70–0.73 |
| Transformer | Vanilla Transformer (T1) | 0.7147 |
| **GeoChemFormer (T2)** | **Ours** | **0.7712** |

GeoChemFormer achieves the best performance on 6 of 8 subsets. On sed2, IF and OSVM perform better (AUC 0.77–0.80 vs. 0.73); on rock2, AE outperforms GeoChemFormer (0.92 vs. 0.83). This indicates that the optimal method varies across geological settings.

### Ablation Study
- **SCL pretraining epochs**: rock2 peaks at 20 epochs (0.919), sed1 at 40 epochs (0.743), and soil2 requires 60 epochs (0.821).
- **Neighborhood size $K$**: sed1 improves monotonically as $K$ increases (16→256, 0.591→0.720), while rock2/soil2 peak at $K$=32/16 and then decline—indicating that sediment anomalies benefit from broad spatial context, whereas rockchip and soil anomalies require compact local neighborhoods.
- **Closure transform**: ILR achieves the best mean performance (0.6788), followed by CLR (0.6771), with raw features performing worst (0.6406).
- **Feature selection**: LLM-assisted selection yields the best overall results (0.7412 vs. 0.6419 for manual), while PCA achieves the highest score specifically on GeoChemFormer (0.9427).
- **Interpolation method**: Performance is primarily driven by data characteristics rather than model architecture.

## Highlights & Insights
- GeoChemAD is the first open-source geochemical anomaly detection benchmark spanning multiple regions and sampling media, filling a critical gap in standardized evaluation for this field.
- The two-stage Transformer design is elegant: SCL enables the model to capture geographic environmental context, while element dependency modeling realizes target-element-aware anomaly detection.
- The paper systematically evaluates the impact of diverse preprocessing strategies (closure transforms, feature selection, interpolation methods), providing practical guidance for real-world applications.
- The effectiveness of LLM-assisted feature selection in the geoscience domain is a noteworthy and interesting finding.

## Limitations & Future Work
- The dataset originates from a single geographic source (Western Australia); despite covering multiple sub-regions, geological background bias remains.
- The spatial sparsity and irregular sampling inherent to geochemical data constrain the performance of deep learning methods.
- On several subsets, simpler methods (IF/AE) outperform GeoChemFormer, indicating that its general advantage is not overwhelming.
- Evaluation relies solely on AUC; in-depth analysis of spatial localization accuracy is absent (distance-based metrics are only demonstrated in a case study).

## Related Work & Insights
- **Yu et al. (NRR 2024)**: Applies Transformers to geochemical anomaly identification but lacks self-supervised pretraining and a standardized benchmark.
- **Luo & Zuo (Math Geosci 2025)**: Uses causal discovery algorithms for gold-deposit anomaly detection, but relies on closed-source, single-region data.
- **Scheidt et al. (NRR 2025)**: Employs masked autoregressive flow for Li-Cs-Ta exploration with publicly available data, though the method is complex.

## Rating
- Novelty: ⭐⭐⭐⭐ (first standardized open-source benchmark + target-element-aware two-stage framework)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (systematic comparison across 12 baselines × 8 subsets × multiple preprocessing strategies)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, thorough dataset description)
- Value: ⭐⭐⭐ (cross-domain application; limited direct value to the CV community, but the dataset contribution is significant)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)
- [\[CVPR 2026\] OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism](omnigcd_abstracting_generalized_category_discovery_for_modality_agnosticism.md)
- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)

</div>

<!-- RELATED:END -->
