---
title: >-
  [Paper Note] Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning
description: >-
  [CVPR 2025][Segmentation][Unsupervised Object Discovery] COCA-Net proposes a hierarchical clustering attention layer based on physical compactness, discovering object centers via a bottom-up hierarchical merging strategy. It resolves the inherent limitations of Slot Attention—such as initialization sensitivity, the requirement of preset slot quantities, and poor background segmentation—achieving state-of-the-art performance on six unsupervised object discovery datasets.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Unsupervised Object Discovery"
  - "Object-Centric Learning"
  - "Hierarchical Clustering"
  - "Compactness"
  - "Slot Attention"
date: 2026-05-08
content_hash: c595031674b68338
---

# Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning

**Conference**: CVPR 2025  
**arXiv**: [2505.02071](https://arxiv.org/abs/2505.02071)  
**Code**: None  
**Area**: Segmentation / Unsupervised Learning  
**Keywords**: Unsupervised Object Discovery, Object-Centric Learning, Hierarchical Clustering, Compactness, Slot Attention

## TL;DR

COCA-Net proposes a hierarchical clustering attention layer based on physical compactness, discovering object centers via a bottom-up hierarchical merging strategy. It resolves the inherent limitations of Slot Attention—such as initialization sensitivity, the requirement of preset slot quantities, and poor background segmentation—achieving state-of-the-art performance on six unsupervised object discovery datasets.

## Background & Motivation

**Background**: Object-Centric Learning (OCL) aims to decompose scenes into independent object representations (slots) in an unsupervised manner, with Slot Attention (SA) and its variants being the dominant approaches. SA assigns pixels to slots using an iterative attention mechanism similar to soft K-Means.

**Limitations of Prior Work**: (1) **Initialization sensitivity**: SA initializes all slots from the same latent distribution, causing routing problems where multiple slots bind to the same object. (2) **Preset slot count**: Incorrectly predefining the number of slots leads to over- or under-segmentation. (3) **Poor background segmentation**: SA struggles to effectively handle sprawling, non-compact background regions.

**Key Challenge**: SA inherits the fundamental limitations of K-Means—sensitivity to initialization, requirement of preset cluster counts, and assumptions on cluster shapes and sizes. These restrictions are particularly severe in complex scenes.

**Key Insight**: Hierarchical Agglomerative Clustering (HAC) naturally offers benefits such as flexible cluster counts, robustness to noise, and hierarchical relationship capture, yet remains under-explored in the OCL field. Physical compactness metrics can serve as spatial inductive biases to distinguish between foreground (compact) and background (dispersed).

**Core Idea**: Incorporate physical compactness metrics into a hierarchical agglomerative clustering framework to build a bottom-up object discovery network that does not require a preset number of slots.

## Method

### Overall Architecture

COCA-Net adopts an encoder-decoder architecture. The encoder consists of multiple cascaded COCA layers that cluster pixel features into object slots in a bottom-up manner layer by layer. The decoder reconstructs the image from the slots using a Spatial Broadcast Decoder (SBD). The training objective is the MSE loss of pixel reconstruction.

### Key Designs

1. **Compactness Scoring**:

    - Function: Calculates a compactness score for each candidate object mask, serving as a spatial inductive bias to guide clustering.
    - Mechanism: A mass-normalized compactness metric based on the Moment of Inertia (MI). For the affinity mask $\boldsymbol{\Lambda}_i$ of each node $i$, it computes $\mathcal{C}^i(\boldsymbol{\Lambda}_i) = \frac{I^\mu(\Theta_\mu)}{I^\mu(\boldsymbol{\Lambda})}$, which is the ratio of the MI of the mask to the MI of a circle with the equal area. A circle (the most compact shape) scores 1, while dispersed shapes score close to 0.
    - Design Motivation: Foreground objects typically have compact, convex shapes, whereas background elements tend to be dispersed and contain holes. The MI compactness metric is additive and scale-invariant. Furthermore, when the axis is set at the centroid of the shape, the MI is minimized (meaning compactness is maximized), naturally locating the object centers.

2. **Sequential Object Centroid Discovery**:

    - Function: Iteratively discovers and isolates object clusters within each window without requiring a preset number of clusters.
    - Mechanism: Based on a variant of Stick-Breaking Clustering (SBC). It initializes a scope tensor $\mathbf{Z}$ with all ones. In each iteration, the node $\omega_m$ with the highest compactness is selected as the anchor. Its affinity mask, masked by the scope, becomes the output cluster $\Pi_m$. The scope is then updated to exclude clustered nodes. The compactness score only needs to be calculated once before the clustering loop.
    - Design Motivation: Leverages the property of MI compactness—where compactness peaks at the object centroid—to naturally separate different object centers. The compactness of all affinity masks is computed in parallel, which is more efficient than Sequential Slot methods.

3. **Hierarchical Pool, Aggregate & Skip Connect**:

    - Function: Transmits features and clustering structures across levels to construct a hierarchical tree of object representations.
    - Mechanism: Each COCA layer partitions input nodes into non-overlapping windows, performing clustering and pooling in parallel within each window. Skip connections are implemented between layers by merging the clustering masks of adjacent layers, allowing features to transfer directly across alternate layers. The clusters in the final layer constitute the object slots, and the entire hierarchical clustering structure forms a dendrogram tree.
    - Design Motivation: Non-overlapping window partitioning enables parallel clustering execution. Skip connections facilitate unsupervised feature learning in deep networks. The dendrogram allows evaluation of segmentation quality directly from the encoder side.

### Loss & Training

- The sole optimization objective is the MSE loss of pixel reconstruction, trained from scratch in an unsupervised manner.
- The pixel feature encoder is a simple point-wise backbone network (each pixel independently encodes appearance and positional information).
- Maintains five physical properties for each pixel in the hierarchy: area, mass, density, moment of inertia, and mean position.
- All methods uniformly utilize the SBD decoder, with the number of slots set to the maximum number of objects in the dataset.

## Key Experimental Results

### Main Results

Decoder-side foreground object segmentation on six datasets (ARI/mSC, mean ± standard deviation over 3 seeds):

| Dataset | Method | ARI↑ | mSC↑ |
|-------|------|------|------|
| Multi-dSprites | BOQSA | 0.91±0.01 | 0.89±0.01 |
| Multi-dSprites | **COCA-Net** | **0.93±0.01** | **0.91±0.01** |
| ObjectsRoom | INVSA | 0.88±0.00 | 0.80±0.01 |
| ObjectsRoom | **COCA-Net** | **0.88±0.00** | **0.82±0.01** |
| ShapeStacks | BOQSA | 0.83±0.09 | 0.80±0.09 |
| ShapeStacks | **COCA-Net** | **0.91±0.01** | **0.85±0.01** |
| CLEVR6 | INVSA | 0.96±0.01 | 0.87±0.03 |
| CLEVR6 | **COCA-Net** | **0.98±0.00** | **0.92±0.01** |

### Key Results with Background Segmented

| Dataset | Method | ARI↑ (with BG) | mSC↑ (with BG) |
|-------|------|------------|------------|
| ObjectsRoom | INVSA | 0.66±0.12 | 0.68±0.07 |
| ObjectsRoom | **COCA-Net** | **0.95±0.01** | **0.87±0.02** |
| Multi-dSprites | BOQSA | 0.34±0.06 | 0.56±0.02 |
| Multi-dSprites | **COCA-Net** | **0.98±0.00** | **0.96±0.00** |

### Encoder-Side Evaluation

COCA-Net's encoder-side segmentation also significantly outperforms competing methods, such as achieving an ENC-FG ARI of 0.82 on ShapeStacks compared to BOQSA's 0.49 (a gain of approximately 67%).

### Key Findings

- COCA-Net outperforms or is on par with the state-of-the-art across almost all datasets and metrics, displaying extremely low variance (high robustness).
- The performance is particularly outstanding in segmentation that includes backgrounds, showing an approximate 30% ARI improvement over the runner-up on ObjectsRoom.
- High-quality segmentation masks can be produced directly from the encoder side, demonstrating that the COCA-Net encoder can independently serve as an object-centric feature extractor.
- The sole weakness is the ARI with background on ShapeStacks (0.31), which occurs because this dataset groups all backgrounds into a single ground truth mask, whereas COCA-Net correctly segments multiple distinct background regions.

## Highlights & Insights

1. **Elegant integration of physical intuition and deep learning**: Compactness is an intuitively clear physical quantity. Embedding it as a differentiable operation within the network offers greater interpretability than purely data-driven methods.
2. **Paradigm shift from K-Means to HAC**: Moving away from the K-Means framework of the Slot Attention family, the use of hierarchical agglomerative clustering avoids fundamental issues such as initialization sensitivity and preset cluster counts.
3. **High-quality encoder-side segmentation**: While existing methods almost exclusively evaluate decoder-side masks, the dendrogram from the COCA-Net encoder inherently delivers high-quality results, opening up possibilities for its use as a feature extractor in downstream tasks.

## Limitations & Future Work

- Currently validated only on synthetic datasets, with its performance in real-world complex scenes yet to be demonstrated.
- The hyperparameters of the hierarchical structure (e.g., window size, number of layers) require dataset-specific tuning.
- The compactness assumption may not hold for concave or ring-shaped objects.
- Future work could explore combining the COCA encoder with more powerful decoders (e.g., diffusion models).

## Related Work & Insights

- **Slot Attention**: The direct baseline of comparison; COCA resolves its routing problem.
- **GENv2 / BOQSA**: Sequential Slot models and improved SA, both acting as primary baselines.
- **TokenCut**: This method utilizes graph-clustering-based unsupervised object discovery with pre-trained backbones.
- Insights: The physical inductive bias of compactness may serve as a valuable reference for other computer vision tasks requiring spatial priors.

## Rating

⭐⭐⭐⭐ — Proposes a novel and effective paradigm shift in unsupervised object-centric learning. The integration of physical intuition and network design is elegant, with thorough experiments, though it lacks real-world validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Scene-Centric Unsupervised Panoptic Segmentation](scene-centric_unsupervised_panoptic_segmentation.md)
- [\[CVPR 2025\] Style-Editor: Text-driven Object-Centric Style Editing](style-editor_text-driven_object-centric_style_editing.md)
- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](../../ECCV2024/segmentation/part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[ICCV 2025\] Ensemble Foreground Management for Unsupervised Object Discovery](../../ICCV2025/segmentation/ensemble_foreground_management_for_unsupervised_object_discovery.md)
- [\[ICML 2026\] Unsupervised Hierarchical Skill Discovery](../../ICML2026/segmentation/unsupervised_hierarchical_skill_discovery.md)

</div>

<!-- RELATED:END -->
