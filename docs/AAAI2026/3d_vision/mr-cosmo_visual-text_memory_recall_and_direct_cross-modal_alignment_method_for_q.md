---
title: >-
  [Paper Note] MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation
description: >-
  [AAAI 2026][3D Vision][3D Segmentation] MR-CoSMo is proposed, a coarse-to-fine query-driven 3D segmentation model. It establishes explicit alignment between 3D point clouds and text/2D images via a Direct Cross-Modal Alignment (DCMA) module, and integrates a visual-text memory module (Memory Module) to store high-confidence feature pairs to enhance cross-scene segmentation consistency. It achieves state-of-the-art (SOTA) performance across three tasks: 3D instruction segmenta…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Segmentation"
  - "Cross-Modal Alignment"
  - "Visual-Text Memory"
  - "Point Cloud Segmentation"
  - "Query-Driven Segmentation"
date: 2026-05-08
content_hash: 69c68c3c8220cf45
---

# MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation

**Conference**: AAAI 2026  
**arXiv**: [2506.20991](https://arxiv.org/abs/2506.20991)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Segmentation, Cross-Modal Alignment, Visual-Text Memory, Point Cloud Segmentation, Query-Driven Segmentation

## TL;DR

MR-CoSMo is proposed, a coarse-to-fine query-driven 3D segmentation model. It establishes explicit alignment between 3D point clouds and text/2D images via a Direct Cross-Modal Alignment (DCMA) module, and integrates a visual-text memory module (Memory Module) to store high-confidence feature pairs to enhance cross-scene segmentation consistency. It achieves state-of-the-art (SOTA) performance across three tasks: 3D instruction segmentation, referring segmentation, and semantic segmentation.

## Background & Motivation

Text-guided 3D segmentation aims to segment 3D objects or scenes based on natural language inputs, serving as a critical capability for autonomous driving and embodied AI. Existing methods exhibit core limitations:

**Indirect Alignment Strategy**: Methods like PointCLIP and Seal use 2D images as intermediaries between 3D point clouds and text. This indirect strategy heavily relies on the accuracy of camera intrinsic and extrinsic parameters, making it highly susceptible to parameter calculation errors and pixel-point alignment artifacts.

**Insufficient Linking of Local Features and Textual Context**: Fine-grained segmentation requires identifying subtle structural changes within objects, demanding a profound understanding of 3D geometry and the capabilities to capture associations between local details and textual context. Existing methods fail to establish stable and accurate coordinate correspondences among 3D point clouds, 2D images, and text.

**Imbalanced Class Samples**: Inherent intra-class variations in texture and contextual features of identical classes within datasets lead to misclassifications of objects and low accuracy in few-shot classes.

## Method

### Overall Architecture

The architecture adopts a coarse-to-fine scheme:
1. **Coarse Stage**: Multimodal feature extraction $\to$ DCMA cross-modal alignment $\to$ Multi-layer Transformer update $\to$ Detection head generates 3D bounding boxes.
2. **Fine Stage**: Point features within bounding boxes + text features $\to$ Memory module enhancement $\to$ Binary classifier iteratively generates segmentation masks.

Input: Point cloud + corresponding 2D images + text query

Feature Extraction:
- Point cloud: MLP extracts point-wise features $f_{point}$ + 4-layer 3D window-shifted Transformer extracts voxel features $f_{voxel}$.
- Image: Pre-trained ResNet-50 extracts visual features $f_{image}$.
- Text: Frozen LLaMA2-7B extracts text features $f_{txt}$.

### Key Designs

#### 1. **Direct Cross-Modal Alignment Module (DCMA)**

DCMA consists of two sub-modules:

**Alignment Constraints Block**: Prior to 3D feature alignment, contrastive learning is used to constrain the relationship between 2D image features and text features. Independent encoders map the image and text features, employing symmetric cross-entropy loss to force matching pairs to converge in the embedding space and non-matching pairs to diverge. This establishes a proper cross-modal semantic foundation for subsequent 3D alignment.

**Bidirectional Direct Alignment Block**:

Core Innovation—Utilizes **bidirectional Mamba attention** to implement direct alignment between 3D features and text/image features, rather than indirect alignment via 2D projections.

Modality Pairing Strategy:
- **Text $\leftrightarrow$ Point-wise Features**: Avoids pixel-point misalignment caused by 2D projections.
- **Image $\leftrightarrow$ Voxel Features**: Leverages regular voxel structures to minimize geometric distortion.

For text-point alignment, a 3-element sequence $X = [\phi_{txt}, \phi_{points}, \phi_{txt}^{copy}]$ is constructed and processed by a bidirectional state space model:

Forward $(\phi_{txt} \to \phi_{points} \to \phi_{txt}^{copy})$:

$$h_t^f = \tilde{A}_f h_{t-1}^f + \tilde{B}_f X_t$$

$$\psi_t^f = \tilde{C}_f h_t^f + \tilde{D}_f X_t$$

Backward processing is conducted in reverse, resulting in the final aligned features:

$$\psi_{point}^* = \text{LayerNorm}(\psi_3^f + \psi_1^b)$$

By placing text features (original + duplicate) at both ends of the sequence, both the forward and backward passes capture the transmission of text semantics to point features, as well as the refined representation of text after cross-modal interaction.

#### 2. **Memory Module**

Addresses the issue of segmentation inconsistency caused by imbalanced class samples and intra-class variations.

**Feature Pair Storage**: Stores the text features $f_{txt}^i$ and 3D point features $f_{box}^i$ within the bounding boxes into dedicated text/visual memory banks, and concatenates them to form a feature pair memory bank:

$$\mathcal{M}_p = \{[f_{txt}^i; f_{box}^i] | i=1,...,N\}$$

**Confidence Weighting**: Calculates initial weights based on BCE loss as $w_i^{(\text{init})} = \frac{1}{\mathcal{L}_{BCE_i} + \tau}$ (lower loss $\to$ higher confidence $\to$ larger weight), and normalizes within the same category:

$$w_i = \frac{1}{\mathcal{L}_{BCE_i} + \tau} \cdot \frac{1}{\sum_{j \in C} \frac{1}{\mathcal{L}_{BCE_j} + \tau}}$$

**Three-Step Attention Retrieval**: When processing a new scene:
1. Text self-attention: Current text queries the text memory bank $\mathcal{M}_t$.
2. Feature pair self-attention: Current concatenated text and point query the feature pair memory bank $\mathcal{M}_p$.
3. Cross-attention: Aligns the text attention results with the feature pair attention results.

Retrieval results are fed into a binary classifier to generate segmentation masks, and its BCE loss is then used to update the current feature pair weights, forming a dynamic weight optimization loop.

### Loss & Training

Total Loss: $\mathcal{L}_{all} = \mathcal{L}_{task} + \mathcal{L}_{DCMA}$

$$\mathcal{L}_{task} = \alpha\mathcal{L}_{det} + \beta\mathcal{L}_{seg} = \alpha(\mathcal{L}_{smoothL1} + \mathcal{L}_{WCE}) + \beta\mathcal{L}_{BCE}$$

$$\mathcal{L}_{DCMA} = \mathcal{L}_{SCE} = \gamma(-\sum y_i\log p_i) + \delta(-\sum p_i\log y_i)$$

Training Details:
- 4$\times$ Nvidia V100 (32G), AdamW optimizer, cosine scheduler.
- Initial learning rate for indoor/outdoor: 0.005/0.002, trained for 500/100 epochs.
- Frozen LLaMA2-7B, with only vectorized text input.
- Memory module features: float32 $\to$ float16, memory footprint < 50MB.
- $\tau = 0.05$ (optimal), random seeds 42/888/2026, each experiment run $\ge 3$ times, standard deviation < 0.2%.

## Key Experimental Results

### Main Results

**3D Instruction Segmentation** (Instruct3D/ScanNet++):

| Method | Acc | mIoU |
|------|-----|------|
| **MR-CoSMo** | **33.8** | **28.5** |
| MR-CoSMo (w/o Memory) | 31.9 | 27.4 |
| SegPoint | 31.6 | 27.5 |
| M3DRef | 18.1 | 12.8 |
| EDA | 16.6 | 12.1 |

**3D Referring Segmentation** (ScanRefer/ScanNet):

| Method | mIoU |
|------|------|
| **MR-CoSMo** | **45.6** |
| RefMask3D | 44.8 |
| SegPoint | 41.7 |
| 3D-STMN | 39.5 |

**3D Semantic Segmentation**:

| Method | S3DIS Area5 mIoU | SemanticKITTI val mIoU |
|------|-----------------|----------------------|
| **MR-CoSMo** | **75.6** | **73.4** |
| PTv3+PPT | 74.7 | 72.3 |
| PTv2 | 72.6 | 70.3 |

### Ablation Study

Module Ablation on Instruct3D:

| Configuration | mIoU | $\Delta$mIoU | Description |
|------|------|-------|------|
| Baseline | 26.4 | +0.0 | Baseline |
| + DCMA | 27.4 | +1.0 | Direct alignment is effective |
| + Memory Module | 27.5 | +1.1 | Memory module contributes independently |
| Only Voxel Encoder | 27.7 | +1.3 | Dual encoders outperform single encoder |
| Reversed matching | 27.9 | +1.5 | Validates modality pairing strategy |
| w/o Alignment Constraints | 28.0 | +1.6 | Contrastive learning constraints are effective |
| w/o Loss on BBox | 28.4 | +2.0 | BBox constraint has minor impact |
| Full model | **28.5** | +2.1 | Full model is optimal |

Ablation on Backbone Alternatives:

| Configuration | Speed(fps) | GPU(GB) | mIoU |
|------|-----------|---------|------|
| Replacing Mamba with Transformer | 2.21 | 30.4 | 28.4 |
| Replacing ResNet50 with ViT | 1.89 | 33.7 | 28.7 |
| Replacing LLaMA2-7B with 13B | 2.35 | 33.0 | 28.6 |
| Replacing LLaMA2-7B with 2B | 2.74 | 27.6 | 28.2 |
| **Default Configuration** | **2.66** | **28.9** | **28.5** |

### Key Findings

- **The combination of DCMA and the Memory Module boosts mIoU by 2.1%**, while each module independently yields a ~1% improvement.
- Compared to Transformer, DCMA with Mamba attention maintains performance while increasing speed (2.66 vs 2.21 fps) and reducing GPU memory usage (28.9 vs 30.4 GB).
- The Memory Module increases Accuracy from 31.9% to 33.8% (+1.9%), primarily helping with individual distinction when handling multiple similar objects.
- It outperforms PTv3+PPT in semantic segmentation (trained on a single dataset vs. multiple datasets), demonstrating that class-aware priors benefit segmentation.
- The temperature parameter $\tau=0.05$ is optimal, and sensitivity analysis on the contrastive learning effect confirms the importance of parameter tuning.
- The default backbone configuration achieves the best balance between performance and efficiency: LLaMA2-7B vs. 13B shows only a 0.1% mIoU difference but is 22% faster.

## Highlights & Insights

1. **Direct Alignment Replacing Indirect Alignment**: Bypasses error accumulation from 2D projections, establishing a direct connection between 3D features and text/images.
2. **Sequence Construction with Bidirectional Mamba**: By placing text feature duplicates at both ends, the forward and backward passes achieve "text-guided $\to$ point feature enhancement" and "point feature-guided $\to$ text refinement" respectively.
3. **Dynamic Weight Updates in Memory Module**: Elegantly addresses sample imbalance through loss-based confidence weighting and intra-class normalization.
4. **Generality**: Resolves three distinct 3D segmentation tasks (instruction, referring, and semantic) within a unified framework.

## Limitations & Future Work

- Inference speed (2.66 fps) is slightly lower compared to 3D-STMN (3.53 fps).
- Computational overhead for Memory Module storage and retrieval scales with the number of training samples.
- The LLaMA2 model is frozen; fine-tuning it might yield further improvements.
- Outdoor scenes (SemanticKITTI) are trained for only 100 epochs, which might result in under-training.
- Open-vocabulary scenarios remain unexplored.

## Related Work & Insights

- SegPoint is the most direct baseline/comparison method (also utilizing an LLM to comprehend text).
- RefMask3D serves as a strong baseline for referring segmentation (44.8 $\to$ 45.6 mIoU).
- PTv3 acts as the foundational backbone for semantic segmentation.
- Insights: The memory module concept can be extended to other 3D tasks demanding cross-scene consistency (e.g., 3D object detection, 3D instance segmentation).
- The efficiency of the Mamba architecture in 3D processing is worthy of further investigation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel combination of direct cross-modal alignment and the Memory Module.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three tasks, four datasets, comprehensive ablations, and backbone analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with detailed formulations.
- Value: ⭐⭐⭐⭐ — Strong practicality by addressing multiple 3D segmentation tasks in a unified framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MORE-STEM: Long-Short MemOry REcall and Spatio-TEmporal Consistency Model for Query-Driven 3D/4D Point Cloud Segmentation](../../CVPR2026/3d_vision/more-stem_long-short_memory_recall_and_spatio-temporal_consistency_model_for_que.md)
- [\[CVPR 2026\] Geometry-Aware Cross-Modal Graph Alignment for Referring Segmentation in 3D Gaussian Splatting](../../CVPR2026/3d_vision/geometry-aware_cross-modal_graph_alignment_for_referring_segmentation_in_3d_gaus.md)
- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)
- [\[CVPR 2025\] CrossOver: 3D Scene Cross-Modal Alignment](../../CVPR2025/3d_vision/crossover_3d_scene_cross-modal_alignment.md)
- [\[CVPR 2026\] GeoFree-CoSeg: Unsupervised Point Cloud-Image Cross-Modal Co-Segmentation Without Geometric Alignment](../../CVPR2026/3d_vision/geofree-coseg_unsupervised_point_cloud-image_cross-modal_co-segmentation_without.md)

</div>

<!-- RELATED:END -->
