---
title: >-
  [Paper Note] Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures
description: >-
  [ECCV 2024][Segmentation][Tubular structure segmentation] Using fractal theory, fractal dimension (FD) is extended from the image level to the pixel level to generate Fractal Feature Maps (FFM) as additional inputs and loss weights for deep learning models. A Multi-Decoder Network (MD-Net) containing a boundary decoder and a skeleton decoder is designed, significantly improving segmentation performance across five tubular structure datasets.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Tubular structure segmentation"
  - "fractal feature map"
  - "topological self-similarity"
  - "multi-decoder network"
  - "boundary and skeleton"
date: 2026-05-08
content_hash: 3f57e786e147ea56
---

# Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures

**Conference**: ECCV 2024  
**arXiv**: [2407.14754](https://arxiv.org/abs/2407.14754)  
**Code**: Yes ([https://github.com/cbmi-group/FFM-Multi-Decoder-Network](https://github.com/cbmi-group/FFM-Multi-Decoder-Network))  
**Area**: Segmentation  
**Keywords**: Tubular structure segmentation, fractal feature map, topological self-similarity, multi-decoder network, boundary and skeleton

## TL;DR

Using fractal theory, fractal dimension (FD) is extended from the image level to the pixel level to generate Fractal Feature Maps (FFM) as additional inputs and loss weights for deep learning models. A Multi-Decoder Network (MD-Net) containing a boundary decoder and a skeleton decoder is designed, significantly improving segmentation performance across five tubular structure datasets.

## Background & Motivation

Accurate segmentation of tubular structures is crucial in multiple fields:

- **Biology**: Endoplasmic reticulum (ER) segmentation for human disease mechanism research
- **Medicine**: Blood vessel segmentation for early diagnosis of retinopathy and stroke
- **Remote Sensing**: Road extraction for navigation and route planning

Key challenges in tubular structure segmentation:
1. Complex morphology and geometry
2. Low signal-to-noise ratio and low contrast

**The disconnection of interconnected structures**—frequent disconnections of tubular structures in segmentation results.

Existing deep learning methods mainly improve in three aspects: convolutional kernel design (e.g., deformable convolution), model architecture design, and loss function design. However, these methods primarily focus on the optimization of the network itself and do not provide **additional structural prior information** to the model.

Core observation of this paper: Tubular structures exhibit **topological self-similarity**—large, complex tubular structures present similar topological patterns at different scales (the basic component of "one node connecting multiple edges" recurs at different scales). This property can be quantified using fractal theory.

## Method

### Overall Architecture

The contributions of this work consist of two independent but complementary modules:

1. **Fractal Feature Map (FFM)**: Extends fractal dimension to the pixel level, serving as model inputs and loss weights.
2. **Multi-Decoder Network (MD-Net)**: Adds a boundary decoder and a skeleton decoder based on U-Net.

### Key Designs

**Pixel-level Extension of Fractal Dimension**:

Traditional fractal dimension calculates a single scalar value for the entire image. This paper extends it to each pixel using a **sliding window** technique:

1. Slide a 5×5 window over the image (with a step size of 1).
2. Calculate the local FD within each window using the box-counting method.
3. The FD values at all pixel positions form the FFM.

**Box-counting Method**: Models the grayscale image as a 3D space $(x,y,z)$, where $z$ represents the grayscale value. The space is partitioned into cubes of size $k \times k \times h$ at different scales $k$, and the minimum number of boxes $N_r$ required to cover the overall grayscale surface is counted. FD is obtained by least-squares linear fitting of $\log N_r$ vs. $\log(1/r)$.

**Two Uses of FFM**:

| FFM Type | Source | Usage |
|----------|------|------|
| $FFM_{image}$ | Original image | As an additional input channel for the model |
| $FFM_{label}$ | Annotation mask | As pixel-level weights for the loss function |

$FFM_{image}$ helps the model perceive texture complexity and self-similar structures, while $FFM_{label}$ forces the model to assign larger loss weights to more complex regions (higher FD).

**Multi-Decoder Network (MD-Net)**:

Based on the encoder-decoder structure of U-Net, two parallel decoders are added:

- **Target Decoder**: Predicts the segmentation mask.
- **Boundary Decoder**: Predicts the boundary of the tubular structures.
- **Skeleton Decoder**: Predicts the skeleton of the tubular structures.

The three decoders share the encoder's features and obtain multi-scale information through skip connections. Only the output of the target decoder is used during inference.

### Loss & Training

**Global Loss**:

$$\mathcal{L}_{global} = \alpha\mathcal{L}_{object} + \beta\mathcal{L}_{edge} + \gamma\mathcal{L}_{skeleton}$$

- Soft IoU Loss is used for target segmentation.
- BCE Loss is used for boundaries and skeletons.
- Default weights: $\alpha=1.0, \beta=0.5, \gamma=0.5$.

**Fractal Constrained Loss**: Using $FFM_{label}$ as pixel-level weights for the target segmentation loss:

$$\mathcal{L}_{constrained} = \alpha\mathcal{L}_{object} \cdot FFM_{label} + \beta\mathcal{L}_{edge} + \gamma\mathcal{L}_{skeleton}$$

Training details: SGD optimizer, initial learning rate of 0.05, a fixed batch size of 32 for all datasets, trained for 50 epochs.

## Key Experimental Results

### Main Results

Segmentation performance on ER (Endoplasmic Reticulum) and MITO (Mitochondria) datasets:

| Model | Loss | ER IoU↑ | ER clDice↑ | ER β Error↓ | MITO IoU↑ | MITO clDice↑ |
|------|------|---------|------------|-------------|-----------|--------------|
| U-Net | $\mathcal{L}_{iou}$ | 75.44 | 94.63 | 28.72 | 79.77 | 96.91 |
| U-Net++ | $\mathcal{L}_{iou}$ | 75.02 | 94.67 | 26.02 | 79.70 | 97.30 |
| DSC-Net | $\mathcal{L}_{iou}$ | 75.51 | 94.44 | 34.51 | 80.32 | 97.16 |
| U-Net* (+ FFM) | $\mathcal{L}_{iou}$ | 76.59 | 95.43 | 20.78 | 80.71 | 97.42 |
| HR-Net* (+ FFM) | $\mathcal{L}_{iou}$ | 76.43 | 95.47 | 20.52 | 80.62 | 97.29 |
| **MD-Net*** | $\mathcal{L}_{constrained}$ | **77.09** | 95.74 | 19.52 | **81.18** | 97.61 |

### Ablation Study

Effect of FFM as a plug-in module (IoU improvement):

| Base Model | Original IoU (ER) | + FFM IoU (ER) | Gain |
|----------|-------------|----------------|------|
| U-Net | 75.44 | 76.59 | +1.15 |
| HR-Net | 75.83 | 76.43 | +0.60 |
| MD-Net | 77.01 | 77.09 | +0.08 |

FFM + constrained loss vs. FFM input only: After adopting the constrained loss on MD-Net, the HD (Hausdorff Distance) of ER decreases from 6.77 to 6.72, and ACC increases from 92.06 to 92.14.

### Key Findings

1. **FFM is effective as a general plug-in**: Adding $FFM_{image}$ improves performance for both U-Net and HR-Net, validating its generalizability.
2. **Multi-decoder design outperforms single-decoder**: MD-Net exceeds baseline U-Net on all metrics.
3. **Fractal constrained loss provides additional gains**: The loss weighted by $FFM_{label}$ further improves boundary accuracy (reduced HD).
4. **Topological errors are significantly reduced**: On the ER dataset, the $\beta$ Error drops from 28.72 (U-Net) to 19.52 (MD-Net*) (a 32% reduction).
5. FFM is also effective on retinal vessel datasets like ROSE and STARE, and remote sensing road datasets like ROAD.

## Highlights & Insights

1. **Innovative integration of fractal theory and deep learning**: Pixel-level fractal features are introduced to deep learning segmentation models for the first time, providing a new perspective for "structural prior injection".
2. **Plug-and-play design**: FFM does not rely on specific network architectures and can be used as a plug-in module for any segmentation model.
3. **Dual utilization of FFM**: It serves both as an input channel to enhance feature representation and as loss weights to guide the optimization direction.
4. **Auxiliary boundary + skeleton decoders**: Although not used during inference, multi-task learning during training effectively improves the segmentation quality of the main decoder.
5. Using standard deviation instead of grayscale value to calculate FD enhances robustness to image noise.

## Limitations & Future Work

1. The sliding window calculation of FFM introduces some overhead (though it is pre-computed offline and does not affect inference speed).
2. The sliding window size (5×5) and the scale parameters of the box-counting method are manually set; adaptive schemes could be explored.
3. The improvement of FFM on non-tubular structures (such as elliptical nuclei in the NUCLEUS dataset) is limited, indicating that its advantage is mainly reflected in topologically self-similar structures.
4. Validated only on 2D images, the method can be extended to 3D tubular structure segmentation (such as 3D blood vessels).
5. The ground truths of skeletons and boundaries are automatically extracted by algorithms, which may introduce noise.

## Related Work & Insights

- **clDice**: A skeleton-intersection-based segmentation metric and loss function, which is complementary to the skeleton decoder in this paper.
- **DSC-Net**: Captures tubular features using dynamic snake convolutions, focusing on micro-level kernel designs.
- **Dconn-Net**: A segmentation network focusing on connectivity.
- Insights: Providing **structural prior information** (such as fractal features and topological constraints) to segmentation models is an effective strategy to enhance the segmentation of complex structures.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| **Overall** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[ECCV 2024\] Eliminating Feature Ambiguity for Few-Shot Segmentation](eliminating_feature_ambiguity_for_few-shot_segmentation.md)
- [\[ECCV 2024\] FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)
- [\[ECCV 2024\] LiFT: A Surprisingly Simple Lightweight Feature Transform for Dense ViT Descriptors](lift_a_surprisingly_simple_lightweight_feature_transform_for_dense_vit_descripto.md)
- [\[ICCV 2025\] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](../../ICCV2025/segmentation/topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)

</div>

<!-- RELATED:END -->
