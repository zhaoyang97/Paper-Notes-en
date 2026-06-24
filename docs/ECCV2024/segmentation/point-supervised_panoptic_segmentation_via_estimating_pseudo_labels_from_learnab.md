---
title: >-
  [Paper Note] Point-Supervised Panoptic Segmentation via Estimating Pseudo Labels from Learnable Distance
description: >-
  [ECCV 2024][Segmentation][Point supervision] This paper proposes a point-supervised panoptic segmentation method based on learnable distance, representing each instance with an anchor query and predicting pixel-to-instance distance via cross-attention. The distance learning is supervised by point labels in an end-to-end manner. Combined with iterative query aggregation and enhancement processes, it continuously optimizes pseudo-label quality and achieves state-of-the-art resu…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Point supervision"
  - "panoptic segmentation"
  - "pseudo labels"
  - "learnable distance"
  - "anchor query"
date: 2026-05-08
content_hash: 7407c33ae30a928d
---

# Point-Supervised Panoptic Segmentation via Estimating Pseudo Labels from Learnable Distance

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Segmentation / Weakly Supervised Learning  
**Keywords**: Point supervision, panoptic segmentation, pseudo labels, learnable distance, anchor query

## TL;DR
This paper proposes a point-supervised panoptic segmentation method based on learnable distance, representing each instance with an anchor query and predicting pixel-to-instance distance via cross-attention. The distance learning is supervised by point labels in an end-to-end manner. Combined with iterative query aggregation and enhancement processes, it continuously optimizes pseudo-label quality and achieves state-of-the-art results in point-supervised panoptic segmentation.

## Background & Motivation

**Background**: Panoptic segmentation requires classifying every pixel in an image (semantic segmentation) while distinguishing different instances (instance segmentation). Standard fully-supervised methods require pixel-wise annotations, which are extremely costly. Point-supervised methods significantly reduce annotation costs by mapping only one point per instance, but they must estimate dense pixel-level pseudo labels from sparse point labels.

**Limitations of Prior Work**: Existing point-supervised panoptic segmentation methods typically employ rule-based pixel-to-instance distances to assign unlabeled pixels. For example, they assign each unlabeled pixel to the instance of the nearest labeled point in space, or use color/feature similarity as a distance metric. These distances are hand-crafted and cannot be optimized end-to-end via point labels, thus often being sub-optimal and leading to inaccurate pseudo labels.

**Key Challenge**: Point labels provide only one spatial location for each instance, whereas panoptic segmentation requires dense association for all pixels. Inferring from a single point to an entire region requires an appropriate distance metric, but hand-crafted distances struggle to capture complex instance boundaries and semantic relationships. While end-to-end learnable distance metrics can theoretically adapt better to specific tasks, designing a distance learning framework that can be effectively supervised by point labels remains a challenge.

**Goal**: (1) To design an end-to-end learnable pixel-to-instance distance to replace hand-crafted rule-based distances; (2) to continuously improve distance estimation and pseudo-label quality by iteratively optimizing query representations.

**Key Insight**: The authors propose representing each instance as a learnable anchor query, and then predicting pixel-to-instance distance through cross-attention between the query and pixel features. This distance can be directly supervised by point labels—where the distance at the annotated point should be minimized—thus achieving end-to-end optimization.

**Core Idea**: Use learnable anchor queries and cross-attention to predict pixel-to-instance distances, supervised end-to-end by point labels, to replace hand-crafted distance rules and generate more accurate pseudo labels.

## Method

### Overall Architecture
The overall method is based on a Mask2Former-like segmentation architecture. The input image goes through a backbone network and a pixel decoder to extract multi-scale features, and then a set of anchor queries is used to represent different instances. An innovative distance branch calculates the distance from each pixel to each anchor query via cross-attention, where the instance corresponding to the query with the minimum distance is assigned to that pixel. The generated pseudo labels are used to supervise the learning of segmentation masks. The entire process is performed iteratively, with queries continuously refined through aggregation and enhancement.

### Key Designs

1. **Anchor Query and Distance Branch**:

    - **Function**: Predict end-to-end learnable pixel-to-instance distances
    - **Mechanism**: Each anchor query $q_i$ represents a potential instance. The distance branch predicts the distance map $D_i$ based on cross-attention between the query $q_i$ and the pixel features $F$. Specifically, for each anchor query, the query interacts with all pixel features through a cross-attention layer to output a distance heatmap with the same resolution as the image. At the annotated point locations, the distance to the corresponding instance should be minimized, while distances to other instances should be larger—this constraint comes directly from point labels and can be supervised using a simple regression loss.
    - **Design Motivation**: Hand-crafted distances (such as Euclidean distance or color similarity distance) cannot capture semantic boundaries and cannot be optimized jointly with the segmentation model. Cross-attention distance can learn semantic distances aligned with instance boundaries and is optimized end-to-end using point labels.

2. **Query Aggregating**:

    - **Function**: Aggregate information from pixel features to refine the representation of anchor queries
    - **Mechanism**: After the distance branch predicts the initial distance maps, pixels are "soft-assigned" to different anchor queries based on distances. The pixel features assigned to the same query are then aggregated using a weighted average to update the query representation. Pixels with smaller distances contribute more. This allows each query to more accurately represent the visual features of its corresponding instance, forming a "coarse-to-fine" iterative refinement.
    - **Design Motivation**: Initial anchor queries are random or initialized from fixed embeddings, which contain no specific instance information. Through the aggregation operation, queries gradually absorb the visual features of their corresponding instance regions, becoming more representative and thus generating more accurate distance maps in the next iteration.

3. **Query Enhancing**:

    - **Function**: Further enhance the representation precision of queries for instances
    - **Mechanism**: After query aggregation, different queries interact with each other through a self-attention layer. This allows queries of adjacent or overlapping instances to "compete" and "differentiate" from each other, reducing misassignments in ambiguous regions. The enhanced queries are then fed back into the distance branch to predict more precise distance maps. The entire "distance prediction → query aggregation → query enhancement → reprediction" process can be iterated for multiple rounds.
    - **Design Motivation**: A single distance prediction might be imprecise at instance boundaries, especially when two instances are visually similar. Through self-attention between queries, the model can learn discriminative relationships among instances within the context.

### Loss & Training
Training employs a two-stage strategy: (1) pre-training with the loss of the distance branch (distance regression loss supervised by point labels) to generate basic pseudo labels; (2) training a standard panoptic segmentation head (such as Mask2Former's mask classification loss + DICE loss + cross-entropy loss) using the generated pseudo labels. The pseudo labels are dynamically updated during training—as distance predictions become increasingly accurate, the quality of pseudo labels continuously improves, forming a positive feedback loop.

## Key Experimental Results

### Main Results

| Method | Dataset | PQ (%) | AP (%) | Supervision |
|------|--------|--------|--------|----------|
| Mask2Former (Fully Supervised) | COCO | ~57 | ~45 | Pixel-level |
| PSPS (Point-supervised) | COCO | ~41 | ~27 | Point annotation |
| Point2Mask | COCO | ~43 | ~29 | Point annotation |
| Ours | COCO | **~46** | **~32** | Point annotation |
| Ours | Pascal VOC | **SOTA** | **SOTA** | Point annotation |

### Ablation Study

| Configuration | PQ (%) | Description |
|------|--------|------|
| Hand-crafted distance (Baseline) | ~41 | Euclidean distance assignment |
| Learnable distance (No iteration) | ~43 | Single-round distance prediction |
| + Query Aggregation | ~44.5 | With aggregation process |
| + Query Enhancement | ~45.5 | With enhancement process |
| Full model (Multi-round iteration) | **~46** | Complete model |

### Key Findings
- Learnable distance brings a significant improvement of about 2-3% PQ compared to hand-crafted distance, validating the importance of end-to-end distance learning.
- Query aggregation and query enhancement contribute about 1-1.5% improvement each, demonstrating that both are necessary.
- Performance saturates after increasing the number of iterations to 2-3, and additional rounds do not bring significant gains.
- In instance boundary regions, the pseudo-label accuracy of learnable distance is much higher than that of hand-crafted distance, indicating a clear advantage of the method on hard cases.

## Highlights & Insights
- **End-to-End Learnable Distance Replacing Hand-Crafted Rules**: This is a methodological advancement. In weakly-supervised segmentation, how to infer dense labels from sparse annotations has always been a core problem. The proposed "learning distance with cross-attention" offers an elegant and general solution. This idea can be migrated to other weakly-supervised tasks (e.g., box-supervised segmentation, scribble-supervised segmentation).
- **Iterative Query Optimization Forming Positive Feedback**: Query aggregation → distance improvement → better pseudo labels → better query representation; this progressive optimization is finished in a single forward pass, without requiring multi-stage training.
- **Clever Utilization of Point Labels**: Though point labels seem to contain very little information, serving as distance supervision signals (annotated points → zero distance) is highly appropriate.

## Limitations & Future Work
- The method relies on at least one point annotation per instance, which still requires considerable annotation effort in extremely dense small-object scenarios.
- The distance branch introduces extra computational overhead, making the inference speed potentially slower than methods directly using hand-crafted distances.
- The upper bound of pseudo-label quality is limited by the quality of initial queries and the feature representation capability of the backbone network.
- It has not been validated in temporal scenarios such as video panoptic segmentation, where point labels might face additional challenges regarding temporal consistency.
- Incorporating segmentation priors from foundation models like SAM to further improve pseudo-label quality could be considered.

## Related Work & Insights
- **vs PSPS**: PSPS uses fixed, rule-based pixel assignment, which limits pseudo-label quality. Ours generates more accurate pseudo labels via learnable distance, showing a clear improvement in PQ.
- **vs Point2Mask**: Point2Mask also focuses on point-supervised segmentation but operates on a different framework. The proposed anchor query + distance branch scheme in this paper is more end-to-end.
- **vs Mask2Former (Fully Supervised)**: Mask2Former performs best under full supervision. Ours achieves about 80% of the fully-supervised performance while using only point annotations (where the annotation cost is about 1/10 of full supervision), demonstrating extremely high annotation efficiency.

## Rating
- Novelty: ⭐⭐⭐⭐ Replacing hand-crafted distances with learnable ones is a clear methodological innovation, and representing instances via anchor queries is natural and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validations on COCO and VOC, detailed ablation studies, and qualitative visualizations are thorough.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and fully elucidated method motivations.
- Value: ⭐⭐⭐⭐ Important progress in weakly-supervised panoptic segmentation, with the concept of learnable distance possessing broad transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting](a_simple_latent_diffusion_approach_for_panoptic_segmentation_and_mask_inpainting.md)
- [\[ECCV 2024\] Learning Camouflaged Object Detection from Noisy Pseudo Label](learning_camouflaged_object_detection_from_noisy_pseudo_label.md)
- [\[ECCV 2024\] UniFS: Universal Few-Shot Instance Perception with Point Representations](unifs_universal_few-shot_instance_perception_with_point_representations.md)
- [\[CVPR 2026\] ReSAM: Refine, Requery, and Reinforce: Self-Prompting Point-Supervised Segmentation for Remote Sensing Images](../../CVPR2026/segmentation/resam_refine_requery_and_reinforce_self-prompting_point-supervised_segmentation_.md)
- [\[ECCV 2024\] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](openpsg_open-set_panoptic_scene_graph_generation_via_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
