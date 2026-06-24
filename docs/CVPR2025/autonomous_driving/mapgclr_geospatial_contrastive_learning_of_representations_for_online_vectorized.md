---
title: >-
  [Paper Note] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction
description: >-
  [CVPR 2025][Autonomous Driving][Online HD Maps] MapGCLR proposes a geospatial contrastive learning method that improves the BEV encoder for online vectorized HD map construction by enforcing BEV feature consistency in geospatially overlapping regions across multiple drives, achieving a 42% relative mAP improvement with only 5% labeled data.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Online HD Maps"
  - "Contrastive Learning"
  - "Semi-Supervised Learning"
  - "BEV Features"
  - "Geospatial Consistency"
date: 2026-05-08
content_hash: 80c577785ab7e754
---

# MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction

**Conference**: CVPR 2025  
**arXiv**: [2603.10688](https://arxiv.org/abs/2603.10688)  
**Code**: None  
**Area**: Autonomous Driving / HD Map Construction  
**Keywords**: Online HD Maps, Contrastive Learning, Semi-Supervised Learning, BEV Features, Geospatial Consistency

## TL;DR
MapGCLR proposes a geospatial contrastive learning method that improves the BEV encoder for online vectorized HD map construction by enforcing BEV feature consistency in geospatially overlapping regions across multiple drives, achieving a 42% relative mAP improvement with only 5% labeled data.

## Background & Motivation
**Background**: Online HD map construction (MapTR, MapTRv2, MapTracker) predicts vectorized map elements in real time through 360° visual inputs, requiring annotations only during training, which significantly reduces the maintenance costs of traditional HD maps.

**Limitations of Prior Work**: These learning-based methods still require a large amount of well-distributed training data and annotations; labeling HD maps is expensive, and the lack of annotations is the primary bottleneck for scalability.

**Key Challenge**: A vast amount of driving data is easily accessible but expensive to annotate—how can the geospatial consistency of unlabeled data be leveraged to enhance the model? Multiple drives through the same location provide a natural "data augmentation."

**Goal**: How to leverage unlabeled multi-drive data through self-supervised learning to improve the quality of BEV feature representations for online HD map construction?

**Key Insight**: BEV features of the same geographic location at different times and under different perspectives should remain consistent—this is a natural self-supervised signal that can be enforced using contrastive learning.

**Core Idea**: Treating the geospatial overlap of multiple drives as natural data augmentation, and enforcing geospatial consistency in the BEV feature space using the InfoNCE contrastive loss.

## Method

### Overall Architecture
Semi-supervised learning pipeline: (1) analyze the geospatial overlap of driving trajectories in the dataset and classify them as single-drive/multi-drive; (2) pass a small amount of labeled data to the supervised branch (standard MapTRv2 training); (3) pass a large amount of unlabeled multi-drive data to the self-supervised branch (geospatial contrastive learning). Both branches are jointly trained.

### Key Designs

1. **Geospatial Multi-Drive Analysis**

    - **Function**: Automatically classify driving logs in the dataset into single/multi-drive runs to ensure sufficient overlapping samples for contrastive learning.
    - **Mechanism**: Transform all poses to the global coordinate system $\rightarrow$ calculate the perception range bbox for each pose $\rightarrow$ merge them into trajectory polygons $\rightarrow$ determine if trajectories intersect. Trajectories with insufficient overlap are categorized as single-drive (lacking diversity).
    - **Design Motivation**: To establish the foundation for contrastive learning—sufficient geospatial overlap must be confirmed to construct valid positive and negative sample pairs.

2. **Geospatial Contrastive Learning of BEV Features**

    - **Function**: Enforce consistency among BEV features of the same geographic location across different drives.
    - **Mechanism**: Given BEV grids of a reference pose R and an adjacent pose A $\rightarrow$ transform them to the global coordinate system $\rightarrow$ BEV cells at the same geographic location act as positive pairs, while different locations serve as negative pairs $\rightarrow$ map to the contrastive space $\mathbf{z} \in \mathcal{Z}$ using a projection head $h$ $\rightarrow$ optimize with InfoNCE loss.
    - **Design Motivation**: Geospatial overlap provides natural "data augmentation"—which is physically more plausible than image distortions and decouples the learning domain from the application domain (via the projection head).

3. **Semi-Supervised Training Regime**

    - **Function**: Mix supervised and self-supervised samples in a single training loop.
    - **Mechanism**: Each batch contains $n$ labeled samples + $2m$ unlabeled samples ($m$ reference-adjacent pairs) $\rightarrow$ labeled samples pass through the full MapTRv2 pipeline to compute $\mathcal{L}_\text{sup}$ $\rightarrow$ unlabeled samples pass only through the encoder to compute $\mathcal{L}_\text{GCLR}$ $\rightarrow$ the total loss is $\mathcal{L}_\text{semi} = \lambda_\text{sup}\mathcal{L}_\text{sup} + \lambda_\text{GCLR}\mathcal{L}_\text{GCLR}$.
    - **Design Motivation**: Single-stage mixed training is more efficient than pre-training followed by fine-tuning, avoiding the problem of feature misalignment between the pre-training phase and downstream tasks.

### Loss & Training
$\mathcal{L}_\text{GCLR} = -\log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+)/\tau)}{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_i^+)/\tau) + \sum_k \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k^-)/\tau)}$, based on the SimCLR framework.
The sampling strategy constrains the IoU within the range of $[\text{IoU}_\min, \text{IoU}_\max]$ to ensure that the overlapping regions are "sufficiently related but not identical."
The projection head $h$ maps the BEV cell features to the contrastive learning space, decoupling the learning domain from the application domain.
Negative samples are randomly sampled from both the reference and adjacent BEV grids, while explicitly excluding the anchor and positive cells.
The weight factors $\lambda_\text{sup}$ and $\lambda_\text{GCLR}$ simultaneously balance the scale and relative influence of both loss terms.

## Key Experimental Results

### Main Results (Argoverse 2)

| Supervision Ratio | SSL | mAP | Absolute Gain | Relative Gain |
|------------|-----|-----|---------|---------|
| 2.5% | ✗ | 6.5 | — | — |
| 2.5% | ✓ | 8.5 | +2.0 | +31% |
| 5% | ✗ | 13.3 | — | — |
| 5% | ✓ | 18.9 | +5.6 | **+42%**|
| 10% | ✗ | 22.0 | — | — |
| 10% | ✓ | 27.3 | +5.3 | +24% |
| 20% | ✗ | 31.0 | — | — |
| 20% | ✓ | 34.9 | +3.9 | +13% |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| 5% SSL | 18.9 mAP | $\approx$ 10% fully supervised (22.0), nearly doubling annotation efficiency |
| 10% SSL | 27.3 mAP | $\approx$ 20% fully supervised (31.0), doubling annotation efficiency |

### Key Findings
- The smaller the labeled dataset, the larger the self-supervised gain—achieving a 42% relative improvement with 5% of data, which drops to 13% at 20% of data.
- The performance of 5% SSL is almost equivalent to 10% fully supervised training—meaning contrastive learning effectively doubles annotation efficiency.
- PCA visualization shows that BEV features trained with SSL present clearer separation at road boundaries, and eliminates abnormal feature clusters that persistently appear at fixed grid coordinates in the baseline.
- Gains are most significant for road boundaries and pedestrian crossings.

## Highlights & Insights
- **Geospatial Consistency as a Self-Supervised Signal**: Cleverly leverages natural overlaps from multiple drives—physically more meaningful than image augmentations, without requiring extra data collection.
- **Contrastive Learning in the BEV Feature Space**: Performs contrastive learning directly in the BEV feature space instead of on raw images or point clouds—bringing it closer to downstream task representations.
- **Performance Equivalent to Doubled Annotation**: 5% SSL $\approx$ 10% fully supervised, practically offering a way to halve annotation costs.
- **Eliminating Positional Bias Artifacts**: PCA visualization reveals that the baseline suffers from abnormal feature clusters at fixed grid coordinates—geospatial consistency constraints inherently eliminate this positional bias.

## Limitations & Future Work
- Requires high-precision (relative) localization—the localization accuracy of some large-scale datasets (such as nuScenes) might be insufficient.
- Self-supervision is only applied at the encoder/BEV level and is not extended to the Transformer decoder—extending contrastive learning to the decoder could yield further improvements.
- Currently a single-frame method, lacking the use of temporal information; integration with MapTracker's tracking mechanism might lead to greater gains.
- The proportion of multi-drive runs in Argoverse 2 is high (well-suited for this method), but its efficacy on datasets with fewer multi-drive sequences remains to be validated.

## Related Work & Insights
- **vs PseudoMapTrainer**: PseudoMapTrainer generates pseudo-labels from sensors for semi-supervised learning, whereas this work uses geospatial consistency for self-supervised learning—making it independent of pseudo-label quality.
- **vs Lilja et al.**: Employs a teacher-student framework + temporal pseudo-label fusion for semi-supervised learning, which also depends on pseudo-labels; this work's contrastive loss directly constrains the feature space.
- **vs HRMapNet**: HRMapNet utilizes cached BEV features/rasterized polylines from multiple drives as global priors during inference; this work leverages geospatial consistency to improve the encoder during training.
- **vs BEVCon**: BEVCon uses contrastive learning in BEV segmentation but constructs positive pairs through image augmentations; MapGCLR uses geospatial overlap as a natural augmentation, which is physically more robust.
- **vs original SimCLR framework**: MapGCLR replaces SimCLR's augmentation strategy from image transformations with geospatial multi-drive runs—representing a tailored application of contrastive learning in autonomous driving.
- **Contrastive Loss for Pose Refinement**: When relative localization is imprecise, the gradient direction of the contrastive loss could potentially be used to fine-tune pose estimation, mitigating the bottleneck of limited localization accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ Treats geospatial overlap as natural augmentation for contrastive learning—a simple yet powerful concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic evaluations across multiple annotation ratios + qualitative analysis via PCA visualization.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear, and the analysis of the dataset is thorough.
- Value: ⭐⭐⭐⭐ Provides a practical and efficient semi-supervised solution for online map construction in annotation-scarce scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] InteractionMap: Improving Online Vectorized HDMap Construction with Interaction](interactionmap_improving_online_vectorized_hdmap_construction_with_interaction.md)
- [\[CVPR 2025\] Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [\[CVPR 2025\] Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map](driving_by_the_rules_a_benchmark_for_integrating_traffic_sign_regulations_into_v.md)
- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)

</div>

<!-- RELATED:END -->
