---
title: >-
  [Paper Note] AstroLoc: Robust Space to Ground Image Localizer
description: >-
  [ICCV 2025][Remote Sensing][astronaut photo localization] This paper proposes AstroLoc, the first space-to-ground localization model trained on 300K manually annotated astronaut photographs. Through a query-satellite pai…
tags:
  - "ICCV 2025"
  - "Remote Sensing"
  - "astronaut photo localization"
  - "cross-domain image retrieval"
  - "contrastive learning"
  - "unsupervised mining"
  - "space-to-ground observation"
date: 2026-05-08
content_hash: ba5cee4257936b98
---

# AstroLoc: Robust Space to Ground Image Localizer

**Conference**: ICCV 2025
**arXiv**: [2502.07003](https://arxiv.org/abs/2502.07003)  
**Code**: None (dataset available at [https://eol.jsc.nasa.gov/](https://eol.jsc.nasa.gov/))  
**Area**: Remote Sensing / Image Retrieval / Geo-Localization
**Keywords**: astronaut photo localization, cross-domain image retrieval, contrastive learning, unsupervised mining, space-to-ground observation

## TL;DR
This paper proposes AstroLoc, the first space-to-ground localization model trained on 300K manually annotated astronaut photographs. Through a query-satellite pairwise loss and unsupervised mining technique, the model learns robust representations of Earth's surface, achieving an average improvement of 35% in Recall@1, consistently exceeding 99% in Recall@100, and has already localized over 500K photographs in real-world deployment.

## Background & Motivation
Astronauts aboard the International Space Station (ISS) capture large numbers of Earth photographs daily using handheld cameras. Since 2000, more than 5 million such images have been accumulated. These photographs are uniquely valuable: they offer spatial resolutions up to 2 meters per pixel, support oblique viewing angles, cover diverse illumination conditions, and constitute the highest-resolution open-source Earth observation data. They find important applications in climate science, atmospheric research, urban planning, and disaster response.

Unlike satellite imagery, however, astronaut photographs lack automatic geo-tagging. Astronauts may point their cameras in any direction within a field of view spanning approximately 20 million km², while a single photograph may cover only 100 km²—equivalent to searching for a needle in 0.0005% of the visible area. NASA has described manual localization as "critically important yet extremely time-consuming," with 300K photographs requiring hundreds of thousands of person-hours.

Existing astronaut photo localization (APL) methods (e.g., EarthLoc) train solely on satellite imagery and never leverage the 300K annotated astronaut photographs—a critical limitation, given that the target task is to localize astronaut images the model has never seen. The core innovation of AstroLoc lies in introducing these photographs into the training pipeline for the first time, making full use of cross-domain data through two complementary training techniques: pairwise loss and unsupervised mining.

## Method

### Overall Architecture
The AstroLoc training pipeline consists of two parallel branches:
1. **Upper branch**: Paired astronaut–satellite images are fed into the **pairwise loss** to directly learn cross-domain correspondences.
2. **Lower branch**: Satellite images are clustered, and training batches are constructed via weighted sampling according to the distribution of astronaut photographs across clusters (**unsupervised mining**), which are then passed into a Multi-Similarity loss.

At inference time, given an astronaut photograph as a query, the model retrieves the most similar satellite tile from a global database via nearest-neighbor search to estimate geographic location.

### Data Preparation — Precise Annotation of 300K Weakly Labeled Photographs

This constitutes a significant preprocessing contribution of the paper. The 300K photographs carry only weak annotations (approximate center-point coordinates), which are insufficient for training that requires precise footprints (four-corner coordinates). The solution proceeds as follows:

1. For each query, 80 candidate satellite tiles are searched across 5 zoom levels × 4 rotations × 4 covering tiles.
2. SuperPoint + LightGlue + EarthMatch are used for feature matching to estimate precise footprints.
3. 221K photographs are successfully annotated; the remainder fail due to labeling errors, cloud occlusion, or horizon-dominated images.
4. Each query is paired with all satellite tiles with IoU > 0.2, yielding 865K query–database training pairs.

### Key Designs

1. **Query-Satellite Pairwise Loss**:

    - **Function**: Directly learns cross-domain feature correspondences between astronaut photographs and satellite images.
    - **Mechanism**: A batch $\mathcal{P} = \{(q_i, d_i)\}_{i=1}^B$ is constructed such that each pair has sufficient IoU and no geographic overlap exists within the batch. The loss comprises an attraction term and a repulsion term:
        - Attraction: $\mathcal{L}_{pos} = \frac{1}{\alpha_1 B}\sum_{i=1}^B \log[1 + e^{-\alpha_1 \times \mathcal{S}(q_i, d_i)}]$
        - The repulsion term covers four combinations—query–query, query–database, database–query, and database–database: $\varphi(x, y, \mathcal{Z}) = \log(1 + \sum_j e^{x \times \mathcal{S}(y, \mathcal{Z}_j)})$
    - Total loss: $\mathcal{L}_{pairs} = \mathcal{L}_{pos} + \mathcal{L}_{neg}$
    - **Design Motivation**: Cross-domain contrastive learning directly addresses the domain gap between astronaut photographs and satellite imagery.

2. **Unsupervised Mining (MUM)**:

    - **Function**: Exploits 5.5 million global satellite images for training while biasing the training distribution toward regions more frequently photographed by astronauts.
    - **Mechanism** — three progressive formulations:
        - **Scheme 1 (Naïve Sampling)**: Random quadruple sampling; lacks hard negatives, yielding a less robust model.
        - **Scheme 2 (Database Clustering)**: K-means clustering into $K$ visually similar groups (forests, deserts, etc.); each batch is sampled from the same cluster to obtain hard negatives. Limitation: uninformative clusters (deserts, oceans) waste training capacity.
        - **Scheme 3 (Full MUM)**: Query features are assigned to $K$ satellite clusters, and clusters are sampled with probability weighted by query count $b_k$: $Pr(k) = \frac{b_k}{\sum_{i=1}^K b_i}$. Regions frequently photographed by astronauts (volcanoes, glaciers, lakes) are sampled more often; uninformative regions (deserts) are sampled less.
    - Loss: Multi-Similarity Loss applied to the sampled quadruples.
    - **Key Properties**: (1) The first mining method to guide the sampling of one distribution (database) using another (queries); (2) requires no query labels, potentially enabling use of all 5 million unannotated photographs.
    - **Design Motivation**: Satellite images are distributed globally and uniformly, whereas astronaut photographs are unevenly distributed (biased toward visually salient regions); the two distributions must be aligned.

3. **Model Architecture**:

    - Backbone: DINOv2-base + SALAD descriptor + linear dimensionality reduction layer (8448→2048 dimensions).
    - More than 10× lighter than AnyLoc (DINOv2-base vs. DINOv2-giant).
    - Total loss: $\mathcal{L} = \lambda_1 \mathcal{L}_{pairs} + \lambda_2 \mathcal{L}_{MUM}$

### Loss & Training
- Hyperparameters: $t_{iou}=0.2,\ \alpha_1=\alpha_2=1,\ \beta_1=\beta_2=50,\ \lambda_1=\lambda_2=1,\ K=50$
- Batch size 48, learning rate 5e-5, Adam optimizer.
- Trained for 30K iterations; cluster features recomputed every 5,000 iterations.
- Four 90° rotation augmentations applied per image at evaluation time.
- The Texas dataset is used as the validation set.

## Key Experimental Results

### Main Results (Original Test Sets, Recall@N)

| Method | Texas R@1 | Alps R@1 | California R@1 | Gobi R@1 | Amazon R@1 | Toshka R@1 |
|--------|-----------|----------|----------------|----------|------------|------------|
| AnyLoc | 44.1 | 40.7 | 48.7 | 28.7 | 38.6 | 63.7 |
| EarthLoc | 55.9 | 58.4 | 58.0 | 51.1 | 47.2 | 72.2 |
| EarthLoc++ | 80.0 | 80.6 | 82.9 | 67.6 | 73.6 | 90.1 |
| **AstroLoc** | **96.1** | **98.1** | **97.4** | **94.6** | **93.0** | **99.0** |

On the extended test sets (more challenging, containing all queries), Recall@100 exceeds 96% across all splits.

### Ablation Study

| Pairwise Loss | Scheme 1 | Scheme 2 | Scheme 3 (MUM) | Texas-L R@1 | Alps-L R@1 |
|---------------|----------|----------|----------------|-------------|------------|
| ✓ | | | | 83.6 | 87.2 |
| | | | ✓ | 82.2 | 86.9 |
| ✓ | | | ✓ | **91.1** | **94.6** |
| | ✓ | | | 67.6 | 76.5 |
| | | ✓ | | 72.4 | 79.4 |

### Key Findings
- AstroLoc achieves Recall@100 above 99% on all original test sets, saturating the existing benchmarks.
- On the more challenging extended test sets (L-variants), Recall@100 still exceeds 96%.
- The pairwise loss and MUM loss are orthogonal—their combination substantially outperforms either component alone.
- Unsupervised mining (Scheme 3) markedly outperforms both naïve sampling and pure clustering.
- Zero-shot transfer to the "Lost in Space" problem yields Recall@1 of 52.7%, surpassing other methods by 45%.
- Zero-shot transfer to historical Space Shuttle photographs (film images from 40 years ago) yields Recall@1 of 82.0%.
- Global search (880K-image database): Recall@100 reaches 96.8%.
- Over 500K photographs have been localized in real-world deployment; the ISS localization backlog is expected to be cleared within months.

## Highlights & Insights
- An exemplary case of data engineering: 300K weakly annotated photographs are precisely annotated through an automated pipeline, creating a valuable training resource.
- The unsupervised mining design is elegant: one distribution guides the sampling of another, requiring no query labels whatsoever.
- The system delivers exceptional real-world value — it is not an academic artifact but a deployed system that has already localized hundreds of thousands of photographs for NASA.
- The model's generalization capability is remarkable: it performs excellently on domains never seen during training, including 40-year-old film photographs and micro/nano-satellite imagery.
- The lightweight design is 10× smaller than AnyLoc yet achieves substantially superior performance.

## Limitations & Future Work
- Nighttime photographs and heavily cloud-covered images remain challenging.
- "Earth-limb" photographs with extreme oblique angles cannot be processed (footprints become invalid).
- The current system performs only coarse retrieval; a learned fine-grained re-ranking stage is absent.
- Scaling to higher-resolution zoom levels may require a larger database and more efficient retrieval.
- Timestamp bit-flips caused by cosmic rays require additional handling.

## Related Work & Insights
- **vs. EarthLoc**: Under the same architecture (EarthLoc++), AstroLoc still leads by 16–27% in Recall@1, demonstrating the critical role of training data and loss design.
- **vs. AnyLoc**: AnyLoc features have dimension 49,152 (requiring 235 GB storage); AstroLoc uses only 2,048 dimensions (9 GB), is 20× faster, and achieves substantially higher performance.
- **vs. UAV Localization**: Although related in problem formulation, astronaut photographs face far more extreme domain gaps (oblique angles, wide field of view, ISS hardware occlusion, etc.).

## Rating
- Novelty: ⭐⭐⭐⭐ — First use of astronaut photographs to train an APL model; the unsupervised mining technique is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Six original and six extended test sets, three cross-domain transfer tasks, comprehensive ablations, and global-scale retrieval experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Compelling motivation, forceful argumentation for problem significance, and clear figures and tables.
- Value: ⭐⭐⭐⭐⭐ — A deployed system addressing a real NASA need, with both strong academic contributions and high practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] WildSAT: Learning Satellite Image Representations from Wildlife Observations](wildsat_learning_satellite_image_representations_from_wildlife_observations.md)
- [\[NeurIPS 2025\] Scaling Image Geo-Localization to Continent Level](../../NeurIPS2025/remote_sensing/scaling_image_geo-localization_to_continent_level.md)
- [\[CVPR 2026\] RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](../../CVPR2026/remote_sensing/rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)
- [\[AAAI 2026\] Debiasing Machine Learning Predictions for Causal Inference Without Additional Ground Truth Data](../../AAAI2026/remote_sensing/debiasing_machine_learning_predictions_for_causal_inference_without_additional_g.md)
- [\[ICLR 2026\] TAMMs: Change Understanding and Forecasting in Satellite Image Time Series with Temporal-Aware Multimodal Models](../../ICLR2026/remote_sensing/tamms_change_understanding_and_forecasting_in_satellite_image_time_series_with_t.md)

</div>

<!-- RELATED:END -->
