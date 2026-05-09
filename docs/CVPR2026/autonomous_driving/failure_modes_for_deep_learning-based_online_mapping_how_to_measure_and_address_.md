---
title: >-
  [Paper Note] Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them
description: >-
  [CVPR 2026][Autonomous Driving][Online mapping] This paper systematically defines and quantifies two failure modes of deep learning-based online mapping models — localization overfitting and map geometry overfitting — proposes a Fréchet distance-based performance metric and a minimum spanning tree (MST)-based training set sparsification strategy, and validates on nuScenes and Argoverse 2 that geometrically diverse and balanced training sets improve model generalization.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Online mapping
  - overfitting analysis
  - generalization evaluation
  - dataset bias
  - map geometric diversity
date: 2026-05-08
content_hash: cd3f0bf425abad6a
---

# Failure Modes for Deep Learning-Based Online Mapping: How to Measure and Address Them

**Conference**: CVPR 2026
**arXiv**: [2603.19852](https://arxiv.org/abs/2603.19852)
**Code**: Available (GitHub Page)
**Area**: Autonomous Driving
**Keywords**: Online mapping, overfitting analysis, generalization evaluation, dataset bias, map geometric diversity

## TL;DR

This paper systematically defines and quantifies two failure modes of deep learning-based online mapping models — localization overfitting and map geometry overfitting — proposes a Fréchet distance-based performance metric and a minimum spanning tree (MST)-based training set sparsification strategy, and validates on nuScenes and Argoverse 2 that geometrically diverse and balanced training sets improve model generalization.

## Background & Motivation

1. **State of the Field**: Deep learning-based online mapping (e.g., MapTR, MapTRv2) has become a core perception task in autonomous driving, where models directly generate vectorized HD map elements from sensor data (cameras, LiDAR).
2. **Limitations of Prior Work**:
   - Performance is inflated on geographically overlapping train/validation splits and drops sharply when switching to geographically disjoint splits (e.g., mAP on nuScenes drops from 60.95 to ~25–29);
   - Models memorize location-specific input features rather than learning generalizable representations;
   - Dataset geometric bias (repetitive map geometry) has not been sufficiently studied.
3. **Root Cause**: Existing evaluations do not distinguish between "models memorizing location features" and "models overfitting to map geometry," making it impossible to address either failure mode in a targeted manner.
4. **Paper Goals**: (1) How to decouple and quantify the two overfitting modes? (2) How to evaluate the geometric diversity of a dataset? (3) How to design better training sets for improved generalization?
5. **Starting Point**: Two orthogonal dimensions — geographic distance and geometric similarity — are introduced to stratify the validation set; Fréchet distance replaces Chamfer distance as a more robust performance metric.
6. **Core Idea**: Decoupling the two overfitting modes by controlling geographic distance and geometric similarity, and eliminating redundant geometric structures in the training set via MST sparsification to improve generalization.

## Method

### Overall Architecture

The paper consists of two major components: (1) a model failure mode analysis framework — proposing evaluation split methods, performance metrics, and overfitting scores; and (2) dataset bias analysis and correction — proposing geometric diversity metrics, geometric similarity metrics, and an MST sparsification strategy.

### Key Designs

1. **Evaluation Set Decoupling and Overfitting Scores**:
   - Function: Stratify the validation set by geographic distance and geometric similarity to independently quantify each overfitting mode.
   - Mechanism: The validation set is first partitioned into $V_{\text{close}}$ (geographically near) and $V_{\text{far}}$ (geographically distant) using a geographic distance threshold $T_{\text{dist}}$. To quantify localization overfitting, the geometric similarity distributions of $V_{\text{close}}$ and $V_{\text{far}}$ are matched via bilateral matching (filtered by KL divergence < 0.01) to obtain $V_{\text{close*}}$ and $V_{\text{far*}}$. The localization overfitting score is defined as $\mathcal{O}_{\text{loc}} = \frac{M_{\text{far*}} - M_{\text{close*}}}{M_{\text{close*}}}$. Geometric overfitting is quantified by binning $V_{\text{far}}$ by geometric similarity and fitting a linear regression slope $\mathcal{O}_{\text{geom}}$.
   - Design Motivation: Geographic distance and geometric similarity are strongly correlated (Pearson r = 0.724); distribution matching is therefore necessary to decouple the two effects. Relying solely on geographic distance would confound the two overfitting modes.

2. **Fréchet Distance-Based Performance Metric**:
   - Function: Provides a more robust assessment of map reconstruction quality than Chamfer distance.
   - Mechanism: The discrete Fréchet distance (accounting for all point orderings) is computed between predicted and ground-truth map elements. Bilateral matching is used to collect a distribution $D$ of matching costs, with the median $M$ and interquartile range $IQR$ serving as performance metrics. For polygons, all cyclic permutations and both orientations are considered.
   - Design Motivation: Chamfer distance is permutation-invariant and cannot detect point ordering errors (e.g., crossing cases in Fig. 3(b)), and is insufficiently robust on small sample sets. Fréchet distance preserves point ordering information and more accurately evaluates shape fidelity.

3. **MST Sparsification Strategy**:
   - Function: Improve geometric diversity and training balance by removing geometrically redundant samples from the training set.
   - Mechanism: A fully connected weighted graph of training samples is constructed (with edge weights $\text{sim}(s_i, s_j)$), and the minimum spanning tree is extracted. A similarity threshold is used to cluster nodes with MST edge weights below the threshold, with the most representative sample (lowest average neighbor weight) selected from each cluster. Set-level geometric diversity is defined as $\text{geomdiv}(D) = \sum_{(i,j) \in \mathcal{E}(\mathcal{T}_{\text{sim}})} \text{sim}(s_i, s_j)$.
   - Design Motivation: At thresholds 0.1–1, removing a large number of samples causes almost no change in geometric diversity yet yields performance gains, indicating that redundant similar samples cause training imbalance. Controlled experiments with random subsampling confirm that MST sparsification significantly outperforms random deletion.

### Loss & Training

No new training loss is introduced. The analysis uses publicly available code and configurations for four models — MapTR, MapTRv2, MapQR, and MGMap.

## Key Experimental Results

### Main Results (MapTRv2 across Different Splits)

| Dataset/Split | geomdiv(T) | geomsim(T,V) | Geographic overlap <5m | mAP↑ | M±IQR↓ | $\mathcal{O}_{\text{loc}}$↓ | $\mathcal{O}_{\text{geom}}$↓ |
|---|---|---|---|---|---|---|---|
| nuScenes original | 96.8 km | 8.32 m | 79.47% | 60.95 | 1.94±3.05 | 24.73 | 21.22 |
| nuScenes geo.[24] | 80.6 km | 14.66 m | 0.95% | 24.96 | 4.07±6.14 | n.a. | 9.75 |
| nuScenes geo.[42] | 90.2 km | 13.85 m | 0% | 28.53 | 3.24±5.50 | n.a. | 13.84 |
| nuScenes geometric | 91.3 km | 21.08 m | 8.53% | 28.37 | 4.17±6.08 | 4.40 | 10.49 |
| Argoverse2 original | 91.0 km | 8.98 m | 44.89% | 63.97 | 1.77±2.99 | 7.29 | 11.17 |

### Multi-Model Overfitting Comparison (nuScenes original)

| Model | $\mathcal{O}_{\text{loc}}$↓ | $\mathcal{O}_{\text{geom}}$ (original)↓ |
|---|---|---|
| MapTR | 24.42 | 18.66 |
| MapTRv2 | 24.73 | 21.22 |
| MapQR | 57.07 | 21.03 |
| MGMap | 33.19 | 24.12 |

### Key Findings

- All models exhibit positive overfitting scores across all splits, indicating that overfitting is a systemic problem.
- MapQR shows the most severe localization overfitting (57.07), possibly related to its query design.
- Performance correlates more strongly with geometric similarity $s(v)$ (Pearson r = 0.568) than with geographic distance $d(v)$ (r = 0.379), suggesting that geometric overfitting may be more critical than localization overfitting.
- MST sparsification at thresholds 0.1–1 reduces sample count while improving performance, outperforming random subsampling.
- geo.[42] outperforms geo.[24] due to higher training set geometric diversity (90.2 km vs. 80.6 km).

## Highlights & Insights

- **Fine-grained decoupling of overfitting**: This is the first work to decompose generalization failures in online mapping into two orthogonal dimensions — "memorizing input features" and "overfitting to map geometry." This framework is generalizable beyond online mapping to any spatial perception task.
- **MST geometric diversity metric**: Quantifying dataset geometric diversity via the sum of MST edge weights is both intuitive and actionable. The finding that removing a small number of redundant samples actually improves performance has direct implications for dataset curation.
- **Fréchet distance over Chamfer distance**: Preserving point ordering information yields more accurate assessment of map element shape fidelity, particularly for crossed or distorted predictions.

## Limitations & Future Work

- Computing geometric similarity (pairwise Fréchet distance + matching) is computationally expensive and difficult to scale to larger datasets.
- The analysis is limited to BEV-view map geometry and does not account for 3D geometric structures (e.g., elevation).
- MST sparsification is a post-processing strategy; dynamically adjusting sampling weights during training remains unexplored.
- Validation is conducted only on nuScenes and Argoverse 2; additional datasets (e.g., Waymo) are needed.
- No training-time methods for directly mitigating overfitting (e.g., geometry-aware data augmentation or specialized loss functions) are proposed.

## Related Work & Insights

- **vs. Lilja et al.**: That work first revealed the geographic memorization effect; this paper builds upon it by further decomposing it into two independent overfitting modes.
- **vs. geographically disjoint splits [24, 42]**: These works address geographic overlap only; this paper additionally analyzes geometric bias and proposes a geometric split.
- **vs. MapTR/MapTRv2**: As the subjects of analysis, both models are found to suffer from severe overfitting, and the proposed framework provides diagnostic tools for future model design.

## Rating

- Novelty: ⭐⭐⭐⭐ — The overfitting decoupling framework and MST sparsification strategy are insightful contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive analysis across four models × multiple splits with thorough ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear, mathematical definitions are rigorous, and visualizations are intuitive.
- Value: ⭐⭐⭐⭐ — Provides important diagnostic tools and dataset design guidelines for the online mapping community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](../../NeurIPS2025/autonomous_driving/how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [\[AAAI 2026\] PriorDrive: Enhancing Online HD Map Construction with Unified Vector Priors](../../AAAI2026/autonomous_driving/priordrive_enhancing_online_hd_mapping_with_unified_vector_p.md)
- [\[CVPR 2026\] ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)
- [\[ICLR 2026\] NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](../../ICLR2026/autonomous_driving/nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)

<!-- RELATED:END -->
