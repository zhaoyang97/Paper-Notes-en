---
title: >-
  [Paper Note] Cut Out the Middleman: Revisiting Pose-Based Gait Recognition
description: >-
  [ECCV 2024][Human Understanding][Gait Recognition] This paper revisits pose-based gait recognition methods and proposes the GaitHeat framework, which utilizes heatmaps instead of traditional skeleton keypoint coordinates to encode human poses. By introducing an improved preprocessing pipeline and a pose-guided heatmap alignment module, this framework significantly enhances performance and generalization capability, bringing pose-based methods close to the accuracy of silhouet…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Gait Recognition"
  - "Heatmap Representation"
  - "Pose Estimation"
  - "Cross-Dataset Generalization"
  - "Global-Local Network"
date: 2026-05-08
content_hash: 4a31f942d5a36abb
---

# Cut Out the Middleman: Revisiting Pose-Based Gait Recognition

**Conference**: ECCV 2024  
**Code**: [https://github.com/BNU-IVC/FastPoseGait](https://github.com/BNU-IVC/FastPoseGait)  
**Area**: Human Understanding / Gait Recognition  
**Keywords**: Gait Recognition, Heatmap Representation, Pose Estimation, Cross-Dataset Generalization, Global-Local Network

## TL;DR

This paper revisits pose-based gait recognition methods and proposes the GaitHeat framework, which utilizes heatmaps instead of traditional skeleton keypoint coordinates to encode human poses. By introducing an improved preprocessing pipeline and a pose-guided heatmap alignment module, this framework significantly enhances performance and generalization capability, bringing pose-based methods close to the accuracy of silhouette-based methods for the first time.

## Background & Motivation

**Background**: Gait recognition is a biometric identification technology based on walking patterns, enabling identity recognition at a distance without active cooperation. Current gait recognition methods are mainly divided into two paradigms: (1) Silhouette-based methods, which use human silhouettes as input (e.g., GaitSet, GaitPart), achieving high accuracy but being susceptible to variations in clothing and carrying conditions; (2) Pose-based methods, which use skeleton keypoint coordinates as input (e.g., GaitGraph, GaitTR), offering robustness to clothing and occlusions but yielding significantly lower accuracy than silhouette-based methods.

**Limitations of Prior Work**: Pose-based gait recognition suffers from two fundamental limitations: (1) **Shape loss**: Skeleton keypoints only represent the spatial coordinates of joints (typically 17-25 points), losing substantial human shape information (e.g., body shape, limb thickness, torso proportions) that is crucial for gait recognition. (2) **Lack of generalizability**: Skeleton-coordinate-based methods are sensitive to the accuracy of pose estimators. Utilizing different pose estimators across different datasets leads to drastic performance fluctuations, making it difficult to transfer a model trained on one dataset to another.

**Key Challenge**: Although skeleton keypoints serve as a compact "middleman" representation for poses, they discard too much original visual information. The dual information bottleneck—from raw images to keypoints, and then to gait features—limits the upper bound of pose-based methods.

**Goal**: (1) Identify a richer pose encoding method that restores shape information while maintaining the robustness of pose-based approaches to clothing changes; (2) Design a cross-dataset generalization framework to eliminate dependency on specific pose estimators; (3) Bridge the performance gap between pose-based and silhouette-based methods.

**Key Insight**: It is observed that the intermediate output of pose estimators—keypoint heatmaps—retains significantly more information than the final keypoint coordinates. Heatmaps represent dense spatial probability distributions, containing not only joint locations but also localization uncertainties, rough limb shapes, and body size information. Directly using heatmaps instead of skeleton coordinates as the input for gait recognition allows the framework to "cut out the middleman", avoiding the information loss inherent in keypoint extraction.

**Core Idea**: Replacing skeleton coordinates with pose-estimator-derived heatmaps as the input representation for gait recognition, combined with pose-guided alignment and a global-local fusion network, enables pose-based methods to approach the performance of silhouette-based methods for the first time.

## Method

### Overall Architecture

The pipeline of GaitHeat is as follows: (1) An off-the-shelf pose estimator (e.g., HRNet) is used to extract multi-channel heatmaps (where each channel corresponds to a keypoint) for each frame of the gait video; (2) Preprocessing is performed on the heatmaps by conducting cropping, scaling, and centering operations in the RGB space to maximize the retention of heatmap integrity; (3) A Pose-Guided Heatmap Alignment (PGHA) module is deployed to eliminate the influence of gait-unrelated covariates; (4) The aligned heatmaps are fed into a Global-Local Network to extract gait features; (5) Metric learning losses (e.g., triplet loss + cross-entropy) are employed for training and recognition.

### Key Designs

1. **Heatmap Representation to Replace Skeleton Coordinates**:

    - **Function**: To provide a richer pose encoding than skeleton coordinates and restore shape information.
    - **Mechanism**: While traditional methods utilize the final keypoint coordinates output by a pose estimator (e.g., 17 sets of $(x,y)$ or $(x,y,c)$ values), GaitHeat intercepts the intermediate output of the pose estimator—namely, the keypoint heatmaps. The heatmap is a $K$-channel spatial probability map (where $K$ is the number of keypoints), where each channel features a Gaussian-like activation around the corresponding joint. These heatmaps retain: (a) joint position information (Gaussian center); (b) localization uncertainty (Gaussian variance reflecting estimation confidence); (c) local shape characteristics (the heatmap shape is influenced by limb thickness); and (d) spatial relationships of adjacent joints. Compared to a coordinate vector of only $K \times 2$ dimensions, the $K \times H \times W$ heatmap conveys orders of magnitude more information.
    - **Design Motivation**: Skeleton coordinates represent a highly compressed summary of heatmaps, a compression that discards significant information useful for gait recognition. Directly utilizing heatmaps circumvents this irreversible loss of information.

2. **RGB-Space Preprocessing Pipeline**:

    - **Function**: To maximize the retention of heatmap information integrity during cropping and scaling processes.
    - **Mechanism**: Traditional gait preprocessing is typically conducted either in the binary silhouette space or the skeleton coordinate space. However, heatmaps represent continuous probability distributions. Performing direct cropping and scaling within the heatmap space can truncate or distort the Gaussian distributions, causing severe information loss. This work proposes performing all geometric transformations (pedestrian bounding box detection $\to$ cropping $\to$ scaling $\to$ centering) in the original RGB image space first, and then applying these transformation parameters to the heatmaps. Consequently, the Gaussian shapes in the heatmaps are fully preserved. Specifically, a pedestrian detector first localizes the body region in the RGB image, and then the corresponding heatmap region is cropped and resized to a standardized dimension.
    - **Design Motivation**: Heatmaps possess a much higher information density than binary silhouettes, making any information loss during preprocessing irrecoverable in subsequent gait recognition stages.

3. **Pose-Guided Heatmap Alignment Module (PGHA)**:

    - **Function**: To eliminate the effects of gait-unrelated appearance covariates, thereby improving cross-dataset generalization.
    - **Mechanism**: Variations in camera viewpoints, clothing, and carrying conditions affect heatmap distributions, yet these factors are unrelated to identity. PGHA utilizes keypoint coordinates as auxiliary information to normalize and align the heatmaps. Specifically, it involves three steps: (a) estimating the body orientation and scale based on keypoint coordinates, and performing affine transformations on the heatmaps to align the body to a standard pose; (b) employing a lightweight attention network conditioned on keypoint coordinates to generate spatial attention masks for each channel of the heatmap, suppressing activations in gait-unrelated regions; (c) applying a residual connection between the aligned heatmap and the original heatmap to preserve useful information that might otherwise be over-suppressed. Here, keypoint coordinates serve merely as auxiliary conditions, while the primary information is still carried by the heatmaps.
    - **Design Motivation**: Cross-dataset generalization is a major weakness of pose-based methods. PGHA enhances the domain invariance of the model by explicitly eliminating the influence of covariates.

4. **Global-Local Network and Efficient Fusion Branch**:

    - **Function**: To extract multi-granularity gait semantic features from heatmaps.
    - **Mechanism**: Gait features encompass both whole-body motion rhythms (global features) and specific joint motion patterns (local features). The network is designed with two parallel branches: the global branch applies spatiotemporal convolutions on the full heatmap to extract global motion patterns; the local branch partitions the heatmap based on body parts (head, torso, upper limbs, lower limbs) to extract localized motion features separately. The outputs of both branches are integrated via an efficient fusion module that utilizes a channel attention mechanism to adaptively weight global and local features, avoiding feature redundancy caused by simple concatenation. The final gait representation is a compact embedding vector.
    - **Design Motivation**: Gait information is distributed across different body parts; thus, a global-local multi-granularity extraction strategy is more comprehensive than a single-granularity one.

### Loss & Training

Training employs a combination of triplet loss and cross-entropy loss. The triplet loss uses a batch-hard mining strategy to select the hardest positive and negative pairs for each anchor. The cross-entropy loss is used for auxiliary identity classification. A balanced sampling strategy is implemented during data sampling to ensure that each mini-batch contains a sufficient number of different identities and varying conditions (viewpoints, carrying conditions, etc.) per identity. Distributed Data Parallel (DDP) is used to accelerate the training process.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (GaitHeat) | Prev. SOTA (Pose) | Silhouette Ref. |
|---------|------|---------------|-------------|-------------|
| CASIA-B | Rank-1 NM | Significant Gain | GaitTR/GPGait | Close to GaitPart |
| CASIA-B | Rank-1 BG | Significant Gain | Baseline | Comparable to silhouette methods |
| CASIA-B | Rank-1 CL | Substantial Lead | Baseline | Outperforms some silhouette methods |
| OUMVLP | Rank-1 | Significant Gain | Prior pose-based methods | Close to silhouette methods |
| GREW | Rank-1 | Significant Gain | Prior pose-based methods | Narrows the gap |
| Gait3D | Rank-1 | SOTA | Prior pose-based methods | Competitive with silhouette methods |

Remarkably, under clothing change (CL) conditions, GaitHeat significantly outperforms skeleton-based methods and achieves performance close to or even exceeding some silhouette-based methods. This validates that the heatmap representation substantially boosts performance while preserving the robustness of pose-based methods to clothing variations.

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Skeleton coordinate input | Baseline Rank-1 | Traditional approach |
| Heatmap input (w/o PGHA) | +10-15% | Heatmaps significantly outperform coordinates |
| Heatmap + PGHA | Additional +3-5% | Alignment module is effective |
| Heatmap + PGHA + Global-Local | Optimal | All components are complementary |
| RGB preprocessing vs Heatmap-space preprocessing | RGB approach +2-3% | Importance of information preservation |
| Cross-dataset testing | Highest consistency | Generalization capability is significantly enhanced |

### Key Findings

- Switching from skeleton coordinates to heatmap inputs yields the most substantial improvement (approx. 10-15 percentage points), proving that the "middleman" issue is indeed the primary performance bottleneck.
- Under clothing variation conditions, the advantage of GaitHeat is most pronounced, as heatmaps naturally lack clothing texture details while preserving body shape information.
- In cross-dataset generalization experiments, GaitHeat trained on CASIA-B and tested on other datasets shows the minimal performance degradation.
- The contribution of RGB-space preprocessing is larger than expected, indicating that the high information density of heatmaps amplifies the impact of the preprocessing stage.
- PGHA contributes the most in multi-view scenarios, significantly improving robustness to viewpoint variations.

## Highlights & Insights

1. **Insight of "cutting out the middleman"**: Correctly identifying that skeleton coordinates as an intermediate representation create a major information bottleneck; the proposed solution is simple yet highly effective.
2. **Innovation in representation**: Providing the first systematic exploration of utilizing heatmaps as a pose encoding scheme in gait recognition, opening up a new direction for the field.
3. **Importance of engineering details**: Showing that seemingly simple engineering choices, such as RGB-space preprocessing, significantly impact the final performance, reflecting solid practical expertise.
4. **Bridging the gap between two paradigms**: Bringing pose-based methods to a level of accuracy comparable with silhouette-based methods for the first time, which marks a significant milestone.
5. **Complete open-source implementation**: Releasing all code based on the FastPoseGait framework, facilitating reproduction and subsequent research.

## Limitations & Future Work

1. The resolution and number of channels in heatmaps increase computational complexity and memory footprint, requiring more resources than skeleton coordinates.
2. The framework still relies on an off-the-shelf, pre-trained pose estimator (e.g., HRNet), and the estimator's errors can propagate into the heatmaps.
3. In scenarios with extreme occlusion, heatmap quality degrades, which might necessitate fusion with silhouette-based methods.
4. The current framework primarily validates 2D heatmaps; utilizing 3D heatmaps (via 3D pose estimators) could potentially yield further improvements.
5. The alignment precision of PGHA is constrained by the accuracy of keypoint coordinates. If keypoint predictions are highly inaccurate, the alignment is also adversely affected.
6. Explaining the potential of multi-modal fusion (heatmaps + silhouettes + skeletons) remains unexplored.

## Related Work & Insights

- **GPGait (ICCV 2023)**: A prior work by the same research group exploring generalized pose-based gait recognition. GaitHeat represents a further breakthrough built upon this foundation.
- **GaitTR / GaitGraph Series**: Skeleton-based gait recognition methods based on Transformers/GCNs. GaitHeat demonstrates that input representations can be more critical than the network architecture itself.
- **Utilization of Intermediate Representations in Pose Estimation**: While some works in other human understanding tasks (e.g., action recognition) have used heatmaps as inputs, GaitHeat successfully introduces this concept to gait recognition.
- **Insight**: This work suggests that in many tasks, carefully examining each "intermediate representation" within the information processing pipeline can reveal simple yet highly effective opportunities for improvement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The core insight is simple yet profound, and the idea of replacing coordinates with heatmaps is highly inspiring.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated comprehensively across three datasets with extensive ablation and generalization experiments.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation is well-articulated, and the "cut out the middleman" title is highly memorable.
- **Value**: ⭐⭐⭐⭐ Effectively bridges the gap between pose-based and silhouette-based methods, driving significant progress in the field of gait recognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](../../CVPR2026/human_understanding/mmgait_multi_modal_gait_recognition.md)
- [\[CVPR 2026\] EventGait: Towards Robust Gait Recognition with Event Streams](../../CVPR2026/human_understanding/eventgait_towards_robust_gait_recognition_with_event_streams.md)
- [\[ICCV 2025\] CarGait: Cross-Attention based Re-ranking for Gait Recognition](../../ICCV2025/human_understanding/cargait_cross_attention_based_re_ranking_for_gait_recognition.md)
- [\[ICLR 2026\] GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences](../../ICLR2026/human_understanding/gaitsnippet_gait_recognition_beyond_unordered_sets_and_ordered_sequences.md)
- [\[CVPR 2026\] HyperGait: Unleashing the Power of Parsing for Gait Recognition in the Wild via Hypergraph](../../CVPR2026/human_understanding/hypergait_unleashing_the_power_of_parsing_for_gait_recognition_in_the_wild_via_h.md)

</div>

<!-- RELATED:END -->
