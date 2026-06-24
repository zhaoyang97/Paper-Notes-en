---
title: >-
  [Paper Note] TOD³Cap: Towards 3D Dense Captioning in Outdoor Scenes
description: >-
  [ECCV 2024][Autonomous Driving][3D dense captioning] This work pioneeringly proposes the task of outdoor 3D dense captioning, constructs the million-scale dataset TOD3Cap (2.3M captions across 850 scenes), and designs an end-to-end network based on BEV features + Relation Q-Former + LLaMA-Adapter, outperforming adapted indoor-based methods by +9.6 CIDEr@0.5IoU.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "3D dense captioning"
  - "outdoor scenes"
  - "BEV"
  - "Relation Q-Former"
  - "LLaMA-Adapter"
date: 2026-05-08
content_hash: a9f72230ce9780c9
---

# TOD³Cap: Towards 3D Dense Captioning in Outdoor Scenes

**Conference**: ECCV 2024  
**arXiv**: [2403.19589](https://arxiv.org/abs/2403.19589)  
**Code**: [https://github.com/jxbbb/TOD3Cap](https://github.com/jxbbb/TOD3Cap)  
**Area**: Autonomous Driving / 3D Vision-Language  
**Keywords**: 3D dense captioning, outdoor scenes, BEV, Relation Q-Former, LLaMA-Adapter

## TL;DR
This work pioneeringly proposes the task of outdoor 3D dense captioning, constructs the million-scale dataset TOD3Cap (2.3M captions across 850 scenes), and designs an end-to-end network based on BEV features + Relation Q-Former + LLaMA-Adapter, outperforming adapted indoor-based methods by +9.6 CIDEr@0.5IoU.

## Background & Motivation
**Background**: Significant progress has been made in indoor 3D dense captioning (e.g., Scan2Cap, Vote2Cap-DETR). However, these methods focus exclusively on indoor environments, leaving outdoor scenes unexplored.

**Limitations of Prior Work**: There are fundamental domain discrepancies between indoor and outdoor scenes:
   - Outdoor objects are dynamic (possessing velocity and motion states), whereas indoor objects are static.
   - Outdoor environments rely on sparse LiDAR point clouds (with spatially uneven density), while indoor scenes use dense scans.
   - Outdoor cameras are fixed in six directions (causing severe self-occlusion), whereas indoor cameras can move freely.
   - Outdoor scenes cover much larger geographic areas.

**Key Challenge**: Indoor methods cannot be directly adapted to outdoor settings due to detector failure, lack of temporal modeling, and lack of support for multi-modal fusion. Furthermore, there is a lack of annotated dataset containing outdoor box-caption pairs.

**Key Insight**: (a) Designing a detection-captioning pipeline tailored for outdoor BEV representation and temporal fusion; (b) constructing a large-scale outdoor dense captioning dataset.

**Core Idea**: Utilizing a unified BEV representation, extracting relations with a Relation Q-Former, and generating descriptions via LLaMA-Adapter, without retraining the LLM.

## Method

### Overall Architecture
The TOD3Cap network consists of three stages: (1) a BEV detector extracts unified BEV features from LiDAR point clouds and multi-view images to generate object proposals; (2) a Relation Q-Former captures relationships between objects and scene context; (3) a LLaMA-Adapter transforms object features into prompts for the LLM, keeping the LLM frozen to generate dense captions.

### Key Designs

#### 1. **BEV-based Detector**
    - **Function**: Mergind multi-view images and LiDAR point clouds into a unified BEV space and generating object proposals.
    - **Mechanism**:
      - **Image Branch**: A learnable BEV query $Q_c \in \mathbb{R}^{H_b \times W_b \times C}$ aggregates multi-view image features via spatial cross-attention: $F_c = \text{Spatial-Cross-Attention}(Q_c, \text{Backbone}(I))$.
      - **Temporal Fusion**: The BEV query interacts with the previous BEV features $F_c^p$ via temporal self-attention: $Q_c' = \text{Temporal-Self-Attention}(Q_c, F_c^p)$, which is designed to model object motion.
      - **LiDAR Branch**: Voxelization $\rightarrow$ backbone network $\rightarrow$ flattening along the height dimension to obtain $F_l \in \mathbb{R}^{H_b \times W_b \times C}$.
      - **Fusion**: A convolutional fusion module merges the BEV features of both modalities to obtain $F_b$.
      - **Proposal Generation**: A DETR-style query-based detection head generates $K$ object proposals $\hat{B} = \{\hat{B}_i\}_{i=1}^K \in \mathbb{R}^{K \times D}$.
    - **Design Motivation**: BEV representation has proven effective in outdoor 3D detection (e.g., BEVFormer, BEVFusion); temporal fusion is essential for modeling dynamic outdoor scenes.

#### 2. **Relation Q-Former**
    - **Function**: Extracting context-aware features for each object and modeling relationships between objects.
    - **Mechanism**:
      - Object proposals $\hat{B}$ are encoded into features of the same dimension as $F_b$ via a learnable MLP.
      - Concatenating the object features and BEV features, and feeding them into the Relation Q-Former (composed of multi-layer self-attentions) for feature interaction.
      - $Q_B = \text{Relation Q-Former}(\text{MLP}(\hat{B}), F_b)$.
    - **Design Motivation**: Outdoor dense captioning requires understanding the relative spatial relationships between objects (e.g., "this car is next to the white truck"). Simple relation graphs or Transformer decoders fail to exploit the global context in BEV.

#### 3. **LLaMA-Adapter Captioning Decoder**
    - **Function**: Translating object query features into natural language descriptions.
    - **Mechanism**:
      - Dimension alignment via MLP: $Q_B' = \text{MLP}(Q_B)$.
      - Modality alignment via Adapter: $\mathcal{V} = \text{Adapter}(Q_B')$, projecting object features into visual prompts understandable by the LLM.
      - Frozen LLM caption generation: $\hat{\mathcal{C}} = \text{LLM}(\mathcal{T}, \mathcal{V})$, where $\mathcal{T}$ denotes the system prompt.
      - Captioning loss: $\mathcal{L}_{cap} = -\sum_{i=1}^M \log \hat{p}(w_i | w_{[1:i-1]}, \mathcal{T}, \mathcal{V}, \theta_{\text{LLM}})$.
    - **Design Motivation**: Freezing the LLM avoids catastrophic forgetting and exploits the pre-trained commonsense reasoning abilities of LLMs. The Adapter bridges the domain gap between BEV and textual features.

### Loss & Training
- Total loss: $\mathcal{L} = \alpha \mathcal{L}_{obj} + \beta \mathcal{L}_{cap}$, with $\alpha=10, \beta=1$.
- $\mathcal{L}_{obj}$: L1 regression loss supervising the 3D bounding boxes.
- Three-stage training: (1) pre-training the BEV detector (24 epochs, lr=2e-4); (2) freezing the detector to train the caption generator (10 epochs, lr=2e-4); (3) full model fine-tuning (10 epochs, lr=2e-5).
- Hungarian matching selection and random subset sampling are utilized during training to reduce GPU memory footprint and optimization difficulty, while NMS is employed during inference.

### TOD3Cap Dataset
- Based on nuScenes with 850 scenes and 34.1K frames.
- Four description dimensions: Appearance (69.7% vocabulary ratio), Motion (2.6%), Environment (7.1%), and Relationship (20.6%), where the relationship category features the longest average length (11.2 words).
- Semi-automatic annotation pipeline: 3D Box $\rightarrow$ 2D projection cropping $\rightarrow$ initial descriptions from LLaMA-Adapter $\rightarrow$ manual correction $\rightarrow$ GPT-4 summarization $\rightarrow$ verification by three annotators.
- A total of 2.3M captions, annotated by 10 annotators with approximately 2,000 hours of effort.

## Key Experimental Results

### Main Results (2D+3D Inputs)

| Method | C@0.25 | B-4@0.25 | C@0.5 | B-4@0.5 |
|------|--------|----------|-------|---------|
| Scan2Cap* | 60.6 | 41.5 | 62.5 | 39.2 |
| X-Trans2Cap* | 99.8 | 45.9 | 92.2 | 43.3 |
| Vote2Cap-DETR* | 110.1 | 48.0 | 98.4 | 46.1 |
| **TOD3Cap** | **120.3** | **51.5** | **108.0** | **50.2** |

Outperforms Vote2Cap-DETR by +9.6 CIDEr@0.5 (+9.76%).

### Ablation Study: Relationship Modeling

| Relationship Module | C@0.25 | C@0.5 |
|---------|--------|-------|
| Relational Graph | 88.8 | 82.7 |
| Transformer Decoder | 94.9 | 90.0 |
| **Relation Q-Former** | **96.2** | **94.1** |

### Ablation Study: Language Decoder

| Decoder | C@0.25 | C@0.5 |
|--------|--------|-------|
| S&T | 81.2 | 78.6 |
| GPT2 | 89.4 | 85.6 |
| **LLaMA** | **96.2** | **94.1** |

### Ablation Study: Training Strategy

| Detector Pre-training | Captioner Pre-training | Full Model Fine-tuning | C@0.25 | C@0.5 |
|------------|-----------|---------|--------|-------|
| ✗ | ✓ | ✓ | 74.2 | 69.5 |
| ✓ | ✗ | ✓ | 87.4 | 85.3 |
| ✓ | ✓ | ✓ | **96.2** | **94.1** |

### Model Scale Comparison

| Configuration | Trainable Parameters | Inference Time | C@0.5 |
|------|----------|---------|-------|
| TOD3Cap-Tiny | 90.5M | 316.1min | 87.3 |
| TOD3Cap-Small | 115.4M | 331.7min | 87.5 |
| **TOD3Cap** | 124.5M | 350.4min | **94.1** |

### Key Findings
- Multi-modal input (2D+3D) significantly outperforms single-modal input: LiDAR provides distance information, whereas cameras offer visual attributes, making them complementary.
- Relation Q-Former outperforms relational graphs and Transformer decoders, primarily because it leverages the global context of BEV.
- Every step of the three-stage training is indispensable; omitting the captioner pre-training stage drops performance by 8.8 CIDEr.
- LLaMA significantly outperforms GPT2 and S&T as a language generator, indicating that the network design fully unleashes the generation capabilities of large foundation models.

## Highlights & Insights
- **Pioneers the new task of outdoor 3D dense captioning**: It clearly defines domain discrepancies between indoor and outdoor environments (dynamic, sparse, fixed camera views, large range) and orchestrates solutions accordingly. This task definition holds substantial practical value for the explainability of autonomous driving and human-robot interaction.
- **Dataset construction methodology**: It structures annotations across four dimensions (appearance, motion, environment, and relationships). The pipeline of semi-automatic annotation followed by multi-round human validation strikes a balance between scale and quality. The resulting 2.3M captions make it the largest 3D dense captioning dataset to date.
- **Engineering wisdom of freezing LLM + Adapter**: Sidestepping the retraining of LLM cuts computational costs while retaining the pre-trained commonsense reasoning abilities of the foundation model, which is especially valuable for long-tail outdoor scenarios (e.g., rare objects).

## Limitations & Future Work
- Detecting and captioning small and distant objects remains challenging.
- BEV resolution significantly impacts performance (a gap of 6.8 CIDEr between $50\times50$ and $200\times200$), and higher resolutions incur substantial computational overhead.
- Motion terms account for a very low percentage of the dataset (2.6%); future work needs to enhance the diversity of dynamic descriptions.
- Only 23 object categories from nuScenes are supported, failing to cover finer-grained outdoor object categories.

## Related Work & Insights
- **vs Scan2Cap**: An indoor detection-captioning pipeline utilizing VoteNet + graph relations; direct adaptation to outdoor scenes yields poor performance (43.3 vs 108.0 C@0.5).
- **vs Vote2Cap-DETR**: The state-of-the-art indoor baseline built on a one-stage set-to-set framework; it still lags behind by 9.6 CIDEr after adaptation, demonstrating the necessity of specialized designs for outdoor domain discrepancies.
- **vs BEVFormer**: It adopts its spatial-temporal BEV encoding concepts, providing support for temporal-dynamic feature representation in caption generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define the outdoor 3D dense captioning task, supplemented by a million-scale dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-dimensional ablations, though lacking cross-dataset evaluation.
- Writing Quality: ⭐⭐⭐⭐ Thorough domain discrepancy analysis and highly informative figures.
- Value: ⭐⭐⭐⭐⭐ Fills the void in outdoor 3D dense captioning, strongly promoting explainability in autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 4D Contrastive Superflows are Dense 3D Representation Learners](4d_contrastive_superflows_are_dense_3d_representation_learners.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](../../ICCV2025/autonomous_driving/controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [\[ECCV 2024\] Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [\[CVPR 2026\] MTA: Multimodal Task Alignment for BEV Perception and Captioning](../../CVPR2026/autonomous_driving/mta_multimodal_task_alignment_for_bev_perception_and_captioning.md)
- [\[ECCV 2024\] Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes](random_walk_on_pixel_manifolds_for_anomaly_segmentation_of_complex_driving_scene.md)

</div>

<!-- RELATED:END -->
