---
title: >-
  [Paper Note] Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation
description: >-
  [ECCV 2024][Object Detection][Oriented Object Detection] This paper proposes a Point-Axis representation method that decouples the position (point set) and orientation (axis encoding) of oriented objects. Facilitated by Max-Projection Loss and Cross-Axis Loss, this method achieves optimization without requiring extra annotations. Based on this, the Oriented DETR model is designed to resolve the loss discontinuity issue inherent in traditional oriented bounding box representat…
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Oriented Object Detection"
  - "Point-Axis Representation"
  - "DETR"
  - "Aerial Imagery"
  - "Loss Inconsistency"
date: 2026-05-08
content_hash: 34fd20ef73569f09
---

# Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation

**Conference**: ECCV 2024  
**arXiv**: [2407.08489](https://arxiv.org/abs/2407.08489)  
**Code**: [https://PointAxis.github.io/](https://PointAxis.github.io/)  
**Area**: Object Detection  
**Keywords**: Oriented Object Detection, Point-Axis Representation, DETR, Aerial Imagery, Loss Inconsistency

## TL;DR

This paper proposes a Point-Axis representation method that decouples the position (point set) and orientation (axis encoding) of oriented objects. Facilitated by Max-Projection Loss and Cross-Axis Loss, this method achieves optimization without requiring extra annotations. Based on this, the Oriented DETR model is designed to resolve the loss discontinuity issue inherent in traditional oriented bounding box representations.

## Background & Motivation

Oriented object detection is an important task in computer vision, widely applied in aerial image analysis. Existing methods primarily use rotated bounding boxes to represent targets, but they face the **loss discontinuity** issue:

**Angle-based methods** $(x, y, w, h, \theta)$: When the aspect ratio is close to 1, the angle $\theta$ jump-changes between $\theta$ and $\theta \pm 90°$, leading to discontinuous loss.

**Quadrilateral-based methods** $(x, y, w, h, l_1, l_2, l_3, l_4)$: When the object is close to horizontal, vertex offsets change abruptly, which also causes discontinuities.

**Other variants** (bbox boundary vectors, middle lines, Gaussian distributions): All face similar boundary issues (e.g., the square problem).

Although recent **point-set-based** methods (e.g., Oriented Reppoints) avoid boundary-definition jumps, they lack the capability to describe orientation information—when points are distributed in a near-circular shape, the computed minimum bounding box may fail to enclose the object accurately.

**Key Challenge**: Can a representation method be designed to describe both position and orientation while avoiding loss discontinuity?

## Method

### Overall Architecture

The Point-Axis representation defines each oriented object $i$ as:
- **Point set** $\mathcal{P}_i = \{p_i^j\}_{j=1,...,K}$: A set of $K$ points describing the spatial extent and contour of the object, where the $K$-th point is the center point.
- **Axis representation** $\mathcal{A}_i$: Discretizes the orientation into bins and applies Gaussian smoothing to generate a four-peak label encoding, representing the principal direction of the object.

The core advantages of this design include:
- **Decoupling of Position and Rotation**: Prevents boundary jumps caused by coupling in oriented bounding box definitions.
- **Axis-Order Invariance**: Does not differentiate between the long and short axes, naturally resolving boundary issues for square/circular targets.
- **Cyclic Labels**: Ensures consistency between labels at 0° and 360°, guaranteeing continuity at angular periodic boundaries.

### Key Designs

1. **Max-Projection Loss**

   This loss is used to supervise point-set learning without requiring explicit keypoint annotations. For the predicted point set $\hat{\mathcal{P}}_i$, it is first converted into center-relative vectors $\hat{\mathcal{V}}_i$. Each vector is then projected onto the ground-truth (GT) boundary vectors $\mathcal{V}_i$, and the maximum projection value is selected for optimization:

    $\text{minimize} \sum_{j=1}^{4} \left| \max_{m=1,...,K-1} \frac{(\hat{v}_i^m - v_i^j) \cdot v_i^j}{\|v_i^j\|} \right| + \|\hat{v}_i^K\|$

   Design Motivation: Only constraining the maximum projection value in each direction while leaving non-maximum values unconstrained reduces optimization ambiguity and enhances the flexibility of the point-set description. Experiments show that adding extra penalties or top-k constraints actually decreases performance.

2. **Cross-Axis Loss**

   Used for axis representation learning. The orientation is discretized into $N_{bins} = 360$ bins and supervised using binary cross-entropy loss:

    $\text{minimize} \frac{1}{N_{bins}} \sum_{j=1}^{N_{bins}} [\mathcal{A}_i^j \log \hat{\mathcal{A}}_i^j + (1-\mathcal{A}_i^j) \log(1-\hat{\mathcal{A}}_i^j)]$

   During inference, the argmax is taken to obtain the principal orientation, which is then expanded into four directions at 90° intervals. For targets lacking well-defined orientations (e.g., certain swimming pools), the model can still learn a distribution that covers all possible directions.

3. **Oriented DETR Architecture**

   An end-to-end detection model based on the DETR framework, consisting of three core modules:

    - **Object-to-Point Query Conversion**: Converts each object query $Q_o^i$ into $K$ point queries. The center point query predicts the offset relative to the reference point via an MLP, while the boundary point queries predict distances in various directions via a polar coordinate system.
    - **Points Detection Decoder**: Comprises Point-to-Point Attention (intra-group self-attention for interaction among the $K$ points of the same instance) and Object-to-Object Attention (extracting center points of each instance for cross-instance interaction), preventing ambiguous interactions among point queries from different instances.
    - **Prediction Head**: Each point query is mapped to 2D coordinates, while the category and axis representation are predicted from all conditioned point queries.

### Loss & Training

The overall loss function is a weighted combination of the Max-Projection Loss and Cross-Axis Loss:

$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (\lambda_1 \mathcal{L}_{proj} + \lambda_2 \mathcal{L}_{ca})$$

- Employs the AdamW optimizer with an initial learning rate of 1e-4 and a weight decay of 1e-4.
- Trained for 36 epochs (DOTA/DIOR/COCO) and 50 epochs (HRSC2016).
- Utilizing 4 × RTX 4090 GPUs with a batch size of 8.

## Key Experimental Results

### Main Results

**DOTA Dataset (single-scale training and testing) comparison:**

| Method | Type | Backbone | mAP50 |
|------|------|----------|-------|
| Orient-Rep | Point Set | Swin-T | 77.6 |
| AO2-DETR | DETR | R-50 | 77.7 |
| ARS-DETR | DETR | Swin-T | 75.5 |
| EMO2-DETR | DETR | Swin-T | 72.3 |
| **Oriented DETR** | **DETR** | **R-50** | **79.1** |
| **Oriented DETR** | **DETR** | **Swin-T** | **79.8** |

**DOTA Dataset (multi-scale) comparison:**

| Method | Backbone | mAP50 |
|------|----------|-------|
| LSKNet | - | 81.85 |
| AO2-DETR | - | 79.22 |
| **Oriented DETR** | **Swin-L** | **82.26** |

**DIOR-R Dataset:**

| Method | Backbone | mAP50 |
|------|----------|-------|
| Prev. SOTA | R-50 | 63.91 |
| **Oriented DETR** | **R-50** | **66.80** (+2.89) |
| Prev. SOTA | Swin-T | 71.05 |
| **Oriented DETR** | **Swin-T** | **74.26** (+3.21) |

### Ablation Study

**Contribution of individual components in the Points Detection Decoder (DOTA val):**

| Configuration | mAP50 | mAP75 | Description |
|------|-------|-------|------|
| Baseline (w/o point queries) | 72.80 | 45.25 | Two-stage Deformable DETR |
| + Point Queries | 70.98 | 44.06 | Point queries + standard self-attention (with cross-instance ambiguity) |
| + Group Self-Attention | 74.21 | 48.30 | Intra-group self-attention eliminates ambiguity |
| + Decouple Cross-Attention | **75.35** (+2.55) | **50.14** (+4.89) | All components |

**Comparison of point-constraint losses:**

| Loss Design | mAP50 | mAP75 | Description |
|-----------|-------|-------|------|
| Max-Projection | **75.35** | **50.14** | Max projection value only |
| + penalty | 75.20 | 50.02 | Add outer-point penalty |
| top-2 | 74.77 | 49.36 | Constrain top-2 projection values |
| top-3 | 73.20 | 47.88 | Constrain top-3 projection values, lower flexibility |

### Key Findings

- Increasing the number of points from K=5 to K=13 yields only a 0.49% increase in mAP50, indicating that the axis representation effectively compensates for the missing orientation information in small point sets.
- Group Self-Attention (intra-group self-attention) is the most critical component of the decoder; standard self-attention allows point queries from different instances to interfere with one another, deteriorating performance.
- For targets with ambiguous orientation definitions (e.g., certain swimming pools), the model can learn distributions covering all directions, demonstrating robust performance.
- The improvement in mAP75 (+4.89) is much more pronounced than that in mAP50 (+2.55), proving that the Point-Axis representation significantly enhances localization accuracy.

## Highlights & Insights

1. **Representation Novelty**: First to propose a decoupled position-orientation representation in oriented object detection, radically resolving the loss discontinuity issue at the representation level.
2. **Elegant Max-Projection Loss**: Only constrains the maximum projection values, leaving room for free exploration by the point set and avoiding optimization ambiguity caused by over-constraint.
3. **Axis-Order Invariance**: Does not differentiate between long and short axes, naturally handling square/circular targets where traditional methods frequently fail.
4. **Natural Integration with DETR**: The point query mechanism is highly compatible with DETR's query-based architecture, featuring an elegant Group Self-Attention design.
5. **Comprehensive Evaluation**: Extensive experiments on four datasets (DOTA, DIOR-R, HRSC2016, COCO) in both single-scale and multi-scale setups.

## Limitations & Future Work

1. **Diminishing Returns with More Points**: Increasing K from 5 to 13 improves mAP50 by only 0.49%; whether the computational overhead of many points is worthwhile requires further analysis.
2. **Limited Advantage on HRSC2016**: This dataset features simple shapes with high aspect ratios, where traditional formulations do not suffer dramatically from boundary problems.
3. **Fixed 360-Bins Design in Cross-Axis Loss**: Whether coarser or finer granularities affect performance under different scenarios remains unexplored.
4. **No Instance Segmentation Considerations**: The Point-Axis representation describes oriented bounding boxes; its extensibility to dense pixel-level tasks remains to be verified.
5. **Lack of Detailed Inference Speed Comparisons**: As a DETR model, whether its end-to-end inference efficiency meets real-time requirements lacks detailed verification.

## Related Work & Insights

- **vs Oriented Reppoints**: Both are point-set-based methods, but this work incorporates an axis representation to compensate for missing orientation information, outperforming it by 2.2% mAP50 using ResNet-50.
- **vs CSL (Circular Smooth Label)**: CSL converts angle regression into a classification task to address periodicity but fails to resolve boundary-definition issues. The axis encoding in this paper is more thorough.
- **vs AO2-DETR / ARS-DETR**: All are oriented detectors in the DETR family, but those methods still iteratively update oriented bounding box queries, ignoring the issue where horizontal box alignment hypotheses fail in oriented scenarios.
- **Insights**: The decoupling philosophy can be extended to 3D object detection (decoupling position and pose) and other tasks requiring orientation representation, such as scene text detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The decoupled Point-Axis representation and Max-Projection Loss are highly novel, representing a major advancement in oriented object representation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers four datasets with detailed ablation studies, though it lacks an inference speed comparison.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear illustrations (especially Figures 2 and 3) with highly intuitive concept explanations.
- **Value**: ⭐⭐⭐⭐ — Achieves clear SOTA accuracy on aerial remote sensing object detection, and the end-to-end DETR design favors real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TAPTR: Tracking Any Point with Transformers as Detection](taptr_tracking_any_point_with_transformers_as_detection.md)
- [\[ECCV 2024\] Shifted Autoencoders for Point Annotation Restoration in Object Counting](shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)
- [\[CVPR 2026\] Partial Weakly-Supervised Oriented Object Detection](../../CVPR2026/object_detection/partial_weakly-supervised_oriented_object_detection.md)
- [\[ECCV 2024\] GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention](gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)
- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)

</div>

<!-- RELATED:END -->
