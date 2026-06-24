---
title: >-
  [Paper Note] ADMap: Anti-disturbance Framework for Vectorized HD Map Construction
description: >-
  [ECCV 2024][HD Map Construction] This paper proposes the ADMap framework, which cascadedly monitors the point sequence prediction process from both inter-instance and intra-instance levels using three modules: Multi-Scale Perception Neck (MPN), Instance Interactive Attention (IIA), and Vector Direction Difference Loss (VDDL). This effectively alleviates the point sequence jitter/jaggedness issues in vectorized HD map construction and achieves SOTA performance on nuScenes and…
tags:
  - "ECCV 2024"
  - "HD Map Construction"
  - "Vectorized Map"
  - "Point Sequence Jitter"
  - "Multi-scale Perception"
  - "Direction Difference Loss"
date: 2026-05-08
content_hash: 3dd25a1dc0efa547
---

# ADMap: Anti-disturbance Framework for Vectorized HD Map Construction

**Conference**: ECCV 2024  
**arXiv**: [2401.13172](https://arxiv.org/abs/2401.13172)  
**Code**: [https://github.com/hht1996ok/ADMap](https://github.com/hht1996ok/ADMap)  
**Area**: Other  
**Keywords**: HD Map Construction, Vectorized Map, Point Sequence Jitter, Multi-scale Perception, Direction Difference Loss

## TL;DR

This paper proposes the ADMap framework, which cascadedly monitors the point sequence prediction process from both inter-instance and intra-instance levels using three modules: Multi-Scale Perception Neck (MPN), Instance Interactive Attention (IIA), and Vector Direction Difference Loss (VDDL). This effectively alleviates the point sequence jitter/jaggedness issues in vectorized HD map construction and achieves SOTA performance on nuScenes and Argoverse2.

## Background & Motivation

**Background**: In the field of autonomous driving, online high-definition (HD) map construction is crucial for planning tasks. Recently, vectorized map construction methods represented by MapTR have made significant progress—these methods represent map elements (lane lines, pedestrian crossings, curbs, etc.) as ordered point sequences (i.e., vectors) and predict them end-to-end from onboard sensor data. Compared with traditional rasterized representations, vectorized representations are more compact, precise, and user-friendly for downstream planning modules.

**Limitations of Prior Work**: Although existing vectorized map construction models perform well on overall performance metrics, the generated point sequences exhibit obvious jittery or jagged phenomena. Specifically, this manifests as: (1) intrinsically smooth lane lines showing irregular jagged edges; (2) sudden direction changes between adjacent points, resulting in less fluent vectors; (3) lack of coordination in point sequences between different instances. Although such jitter may not severely degrade evaluation metrics such as mAP, it has a significant impact on downstream planning tasks, which rely on precise and smooth map information to generate safe driving trajectories.

**Key Challenge**: Existing methods primarily supervise point sequence prediction through regression losses of endpoint coordinates. This point-by-point independent supervision ignores the relative relationship between points—neither considering direction consistency between adjacent points within the same instance, nor considering spatial coordination among different instances. Consequently, prediction errors accumulate into visible jitter and jaggedness.

**Goal**: (1) How to ensure direction consistency and sequence smoothness between adjacent points inside an instance? (2) How to maintain the coordination of spatial relations among different instances? (3) How to improve the geometric quality of map elements without sacrificing detection speed?

**Key Insight**: The authors approach this from the perspective of the "sequential order" of point sequences, arguing that the sequential relationships of points should be explored in a cascaded manner at both inter-instance and intra-instance levels. By focusing on direction changes between adjacent points (rather than just absolute positions), the geometric smoothness of predictions can be directly constrained.

**Core Idea**: By exploring the sequential relationship of point sequences cascadedly at both inter-instance and intra-instance levels, and directly constraining geometric smoothness with direction difference loss, the jitter issue in vectorized maps is fundamentally alleviated.

## Method

### Overall Architecture

ADMap is built upon the MapTR framework. The inputs are multi-view camera images (and optional LiDAR data) from the vehicle. Features are extracted by a backbone (such as ResNet-50) and SECOND (the optional LiDAR backbone) to obtain multi-modal features, which are then transformed into bird's-eye view (BEV) features through BEV transformation. The innovation consists of three modules: (1) Multi-Scale Perception Neck (MPN) for multi-scale enhancement of BEV features; (2) Instance Interactive Attention (IIA) for modeling spatial relationships among different map instances in the decoder; (3) Vector Direction Difference Loss (VDDL) for constraining direction smoothness of point sequences during training. These three modules form a cascaded pipeline of "feature enhancement $\rightarrow$ inter-instance relations $\rightarrow$ intra-instance constraints".

### Key Designs

1. **Multi-Scale Perception Neck (MPN)**:

    - **Function**: Performs multi-scale enhancement on BEV features, enabling the model to simultaneously perceive map elements of different scales (such as short pedestrian crossings and long lane lines).
    - **Mechanism**: Designs a multi-scale feature fusion module, which applies pooling and upsampling on multiple scales to the BEV features outputted by the backbone. It then fuses information across scales through a feature pyramid-style module. Specifically, it uses convolutional kernels of different sizes (or dilated convolutions with different dilation rates) to extract multi-scale context, and adaptively selects the most useful scale information through a channel attention mechanism.
    - **Design Motivation**: Different types of map elements possess contrasting spatial scales—pedestrian crossings are typically short and localized, while lane lines can span across the entire perception range. Single-scale features struggle to simultaneously capture the geometric details of both classes precisely.

2. **Instance Interactive Attention (IIA)**:

    - **Function**: Models the spatial relations among different map instances inside the Transformer decoder to ensure inter-instance coordination.
    - **Mechanism**: In the self-attention layer of the standard DETR-style decoder, instance-level interactive attention is introduced. Each instance query not only attends to its own point sequence information but also interacts with queries of other instances via cross-attention. In this way, adjacent lane lines can "perceive" each other's positions, avoiding unreasonable crossings or inconsistent intervals. A cascaded attention structure is designed: first interacting between instances, and then refining point sequences within each instance.
    - **Design Motivation**: Strong spatial constraints exist among map elements in road scenes—parallel lane lines should remain roughly equidistant, and adjacent elements should not overlap. Predicting each instance completely independently misses these global consistency constraints.

3. **Vector Direction Difference Loss (VDDL)**:

    - **Function**: Directly constrains direction consistency between adjacent points within point sequences, eliminating jitter from the loss function level.
    - **Mechanism**: For a point sequence $\{p_1, p_2, ..., p_n\}$, the direction vectors formed by adjacent points are calculated as $v_i = p_{i+1} - p_i$. The angular variation between adjacent direction vectors is then constrained to be as small as possible. Specifically, the loss is defined as the $L1$ distance between the direction differences of predicted and ground truth sequences: $L_{VDDL} = \sum_i \| (v_{i+1}^{pred} - v_i^{pred}) - (v_{i+1}^{gt} - v_i^{gt}) \|_1$. This loss directly penalizes sudden changes in directions; if jagged patterns occur in predicted point sequences, the differences between adjacent direction vectors will be huge, resulting in a larger loss.
    - **Design Motivation**: Traditional coordinate regression losses (like L1/L2) only constrain the absolute position of each point, leaving "relative relationships between points" unconstrained. Even if coordinate errors of individual points are small, accumulated direction inconsistencies lead to visible jitter. VDDL optimizes geometric smoothness directly by explicitly modeling direction variations.

### Loss & Training

The total loss is a weighted sum of classification loss, point regression loss, and VDDL: $L = L_{cls} + \lambda_1 L_{pts} + \lambda_2 L_{VDDL}$, where $L_{pts}$ is the standard point-wise L1 regression loss. Training follows MapTR's strategy, utilizing Hungarian matching for ground truth allocation and end-to-end training.

## Key Experimental Results

### Main Results

| Dataset | Backbone | Metric (mAP) | Ours (ADMap) | MapTR | Gain |
|--------|----------|-----------|------------|-------|------|
| nuScenes val | R50 | mAP | 54.5 | 50.3 | +4.2 |
| nuScenes val | R50+SECOND | mAP | 68.0 | 62.5 | +5.5 |
| Argoverse2 val | R50 | mAP | 64.7 | 61.3 | +3.4 |
| Argoverse2 val | R50+SECOND | mAP | 68.7 | 67.4(MapTRv2) | +1.3 |

Detailed sub-class performance (nuScenes val, R50):

| Method | Pedestrian Crossing | Divider | Curb | mAP |
|------|---------|-----------|------|-----|
| MapTR | 51.5 | 46.3 | 53.1 | 50.3 |
| ADMap | 56.2 | 49.4 | 57.9 | 54.5 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| MapTR baseline | 50.3 mAP | Baseline method |
| + MPN | mAP gain | Multi-scale perception enhances BEV features |
| + IIA | Further mAP gain | Inter-instance interaction improves global consistency |
| + VDDL | Additional mAP gain | Direction constraint eliminates jitter |
| Full (MPN+IIA+VDDL) | 54.5 mAP | Optimal performance through three modules coordination |

### Key Findings

- ADMap achieves an average improvement of about 4.2 mAP over MapTR on nuScenes, with consistent gains across all three map element classes.
- Under the R50+SECOND (fusing LiDAR) configuration, ADMap reaches 68.0 mAP, surpassing MapTRv2's 69.0 mAP (the latter requiring more complex designs).
- VDDL shows the most prominent effect in qualitative visualization—predicted lane lines are visibly much smoother, and jagged artifacts are significantly reduced.
- FPS is virtually unaffected (14.8 vs 15.1), indicating minimal added computational overhead.
- Consistent improvements are achieved on Argoverse2, a larger and more challenging dataset.

## Highlights & Insights

- Precisely targets the "point sequence smoothness" problem in vectorized map construction, which is often neglected but critical for downstream tasks.
- VDDL is a simple yet effective loss design—constraining geometric smoothness via direction differences, making it computationally lightweight but highly effective.
- The "inter-instance $\rightarrow$ intra-instance" cascaded relationship modeling pipeline is highly inspiring, reflecting a design philosophy from global to local.
- As an enhancement module for MapTR, the method is highly backward-compatible with almost no drop in FPS.

## Limitations & Future Work

- Built upon the MapTR framework, its generalizability to other map construction frameworks (e.g., VectorMapNet, PivotNet) needs further validation.
- VDDL assumes that map elements should be smooth. However, sharp directional changes are reasonable in tight curves or intersection areas; the loss design needs special handling for these cases.
- Evaluated only on two datasets (nuScenes and Argoverse2), without verification in more scenarios like dense urban roads or highways.
- Although real-time performance does not significantly degrade, deployment efficiency on embedded computing platforms requires further evaluation.
- Although the code is public, the authors mention it is an "extra replication version", which might differ from the complete implementation in the paper.

## Related Work & Insights

- MapTR and MapTRv2 are representative works in vectorized map construction; ADMap focuses on improving geometric quality on top of them.
- The idea of direction difference loss can be extended to other tasks requiring geometric smoothness, such as trajectory prediction and curve detection.
- The design of inter-instance interaction shares similar ideas with object query interactions in DETR, but emphasizes spatial relationship constraints more.

## Rating
- Novelty: ⭐⭐⭐ Precise problem targeting and simple, effective VDDL design, though the overall framework innovation is moderate.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively verified on two mainstream datasets with clear ablation studies and intuitive visualizations.
- Writing Quality: ⭐⭐⭐ Clear problem definition and method descriptions.
- Value: ⭐⭐⭐⭐ Solves an important problem in practical deployment; code is open-source and results are reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](../../ACL2025/others/mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)
- [\[ECCV 2024\] An Incremental Unified Framework for Small Defect Inspection](an_incremental_unified_framework_for_small_defect_inspection.md)
- [\[ECCV 2024\] A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation](a_framework_for_efficient_model_evaluation_through_stratific.md)
- [\[ECCV 2024\] HiEI: A Universal Framework for Generating High-quality Emerging Images from Natural Images](hiei_a_universal_framework_for_generating_high-quality_emerging_images_from_natu.md)
- [\[ACL 2025\] Autalic: A Dataset for Anti-Autistic Ableist Language In Context](../../ACL2025/others/autalic_a_dataset_for_anti-autistic_ableist_language_in_context.md)

</div>

<!-- RELATED:END -->
