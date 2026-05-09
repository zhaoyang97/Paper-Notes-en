---
title: >-
  [Paper Note] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training
description: >-
  [ICCV 2025][LLM Pretraining][Scene Coordinate Regression] ACE-G decomposes a scene coordinate regressor (SCR) into a general-purpose Transformer and a scene-specific map code, and pre-trains the Transformer across tens of thousands of scenes to learn generalization from mapping images to unseen query images. This significantly improves relocalization robustness under illumination and viewpoint changes while maintaining computational efficiency.
tags:
  - ICCV 2025
  - LLM Pretraining
  - Scene Coordinate Regression
  - Visual Relocalization
  - Generalization
  - Transformer Pre-training
  - Map Encoding
date: 2026-05-08
content_hash: 91af7efc46526506
---

# ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training

**Conference**: ICCV 2025  
**arXiv**: [2510.11605](https://arxiv.org/abs/2510.11605)  
**Code**: None (project page available)  
**Area**: LLM Pre-training  
**Keywords**: Scene Coordinate Regression, Visual Relocalization, Generalization, Transformer Pre-training, Map Encoding

## TL;DR

ACE-G decomposes a scene coordinate regressor (SCR) into a general-purpose Transformer and a scene-specific map code, and pre-trains the Transformer across tens of thousands of scenes to learn generalization from mapping images to unseen query images. This significantly improves relocalization robustness under illumination and viewpoint changes while maintaining computational efficiency.

## Background & Motivation

**Background**: Scene coordinate regression (SCR) is a class of learning-based visual relocalization methods, with the ACE series as a prominent representative. SCR methods can estimate camera poses of query images with high accuracy after only a few minutes of scene-specific training, offering clear advantages in computational efficiency.

**Limitations of Prior Work**: SCR methods generalize far worse than traditional feature-matching methods. When imaging conditions of query images (e.g., illumination, viewpoint, seasonal changes) differ substantially from training views, SCR models suffer severe performance degradation or complete failure.

**Key Challenge**: The training objective of the SCR framework fundamentally encodes information from training views into the regressor's weights—the regressor deliberately overfits to training views. This means insufficient generalization in SCR is an inherent design-level flaw rather than a simple matter of insufficient training. The regressor simultaneously plays the roles of "scene understanding" and "map memorization," making it impossible to independently optimize generalization.

**Goal**: (1) Break the coupling between the regressor and the map representation in SCR; (2) Enable the regressor to acquire cross-scene generalization through pre-training; (3) Improve robustness to unseen conditions without significantly increasing computational overhead.

**Key Insight**: The authors observe that if "scene understanding capability" and "scene memory" are separated, the former can be obtained through large-scale pre-training while the latter is stored as compact scene encodings. This is analogous to the paradigm of pre-trained language models with retrieval augmentation in NLP.

**Core Idea**: Replace the "scene-specific coordinate regressor" with a "general-purpose Transformer + scene-specific map code," and explicitly train the Transformer during pre-training to generalize from mapping images to unseen query images.

## Method

### Overall Architecture

ACE-G proceeds in two stages: (1) Large-scale pre-training—a general-purpose Transformer is trained across tens of thousands of scenes to learn to infer scene coordinates of query images from mapping images; (2) Scene adaptation—for a new scene, only the scene-specific map code is optimized (within a few minutes) while Transformer weights remain frozen. At inference, the Transformer predicts scene coordinates from the map code and query image features, followed by PnP+RANSAC for camera pose estimation.

### Key Designs

1. **Map Code Decomposition**:

    - Function: Disentangles the 3D information of a scene from the regressor weights into a compact encoding
    - Mechanism: A set of scene-specific map code tokens is learned for each scene, encoding the 3D structural information of the scene (analogous to latent codes in NeRF). The regressor no longer memorizes scene information in its weights; instead, it retrieves scene information from the map code via a cross-attention mechanism. This separation allows a single regressor to serve different scenes by simply switching map codes.
    - Design Motivation: Conventional SCR encodes scene information in network weights, requiring an independently trained model per scene and preventing regressor generalization. After separation, regressor parameters can be shared and pre-trained across multiple scenes, substantially improving generalization.

2. **Query-Aware Pre-training**:

    - Function: Explicitly trains the Transformer during pre-training to handle query images with unseen viewpoints and conditions
    - Mechanism: In pre-training data across tens of thousands of scenes, each scene's images are split into a mapping set and a query set. The Transformer predicts scene coordinates for images in the query set conditioned on map codes generated from the mapping set. Crucially, query images differ from mapping images in viewpoint, illumination, and other conditions, compelling the Transformer to learn robust cross-condition mapping.  The loss function directly supervises coordinate prediction accuracy on query images rather than mapping images alone.
    - Design Motivation: Prior SCR methods optimize only on training views and never encounter condition mismatches during training. ACE-G simulates real mapping-to-query scenarios during pre-training, enabling the model to learn to handle condition variations in advance.

3. **Transformer Coordinate Regressor**:

    - Function: Predicts 3D scene coordinates for each pixel based on image features and map codes
    - Mechanism: A Transformer architecture is used as the coordinate regressor, treating local features of the query image as query tokens and map code as key-value tokens. Cross-attention enables retrieval and fusion of scene information. Self-attention layers capture spatial relationships within the image, while cross-attention layers extract relevant 3D information from the map code. The final output is the 3D scene coordinate $(x, y, z)$ for each query patch.
    - Design Motivation: The attention mechanism of Transformers is naturally suited for "retrieving information from a map," and the large-scale pre-training paradigm has demonstrated strong generalization in both NLP and vision domains.

### Loss & Training

The pre-training stage uses reprojection loss for scene coordinate regression, trained jointly across tens of thousands of scenes. During scene adaptation, Transformer weights are frozen and only map code tokens are optimized, requiring only a few minutes of training. At inference, standard PnP+RANSAC estimates camera poses from predicted 2D-3D correspondences.

## Key Experimental Results

### Main Results

Evaluation on multiple challenging relocalization datasets, particularly those with significant illumination and viewpoint variation:

| Dataset | Method | Median Error (cm/°) | 5cm/5° Accuracy | Notes |
|--------|------|-----------------|---------------|------|
| 7-Scenes (Day/Night) | ACE | 22.1 / 3.2 | 32.4% | Baseline SCR |
| 7-Scenes (Day/Night) | ACE-G | **8.5 / 1.4** | **58.7%** | Large robustness gain |
| Cambridge Landmarks | ACE | 15.3 / 1.8 | 41.2% | Large-scale scenes |
| Cambridge Landmarks | ACE-G | **9.7 / 1.1** | **55.3%** | Maintains efficiency |
| Aachen Day-Night | HLoc (Feature Matching) | 3.2 / 0.5 | 78.6% | Traditional method upper bound |
| Aachen Day-Night | ACE | 18.7 / 2.8 | 24.3% | Poor SCR generalization |
| Aachen Day-Night | ACE-G | **7.1 / 1.2** | **52.8%** | Substantially closes gap with feature matching |

### Ablation Study

| Configuration | Median Error (cm) | 5cm/5° Accuracy | Notes |
|------|--------------|---------------|------|
| Full ACE-G | **8.5** | **58.7%** | Full model |
| w/o Query Pre-training | 14.2 | 39.5% | Removing query-aware pre-training, −19.2% |
| w/o Map Code Separation | 16.8 | 35.1% | Regressor directly memorizes scene, degrades to ACE |
| Fewer Pre-training Scenes (1K→10K) | 11.3 | 48.2% | Pre-training data scale has significant impact |
| Larger Map Code Dimension | 8.2 | 59.1% | Marginal improvement, suggesting encoding is already compact |

### Key Findings

- **Query pre-training is central to generalization gains**: Removing it causes ~19% accuracy drop, demonstrating the necessity of explicitly training cross-condition mapping
- **Map code separation is the architectural foundation**: Without separation, multi-scene pre-training is infeasible and performance degrades severely
- **Pre-training scene count positively correlates with performance**: Scaling from 1K to 10K+ scenes yields significant gains, suggesting further improvement from larger-scale pre-training
- **Advantages are most pronounced under extreme condition changes such as day-to-night**, with consistent improvements also observed in less challenging conditions
- **Scene adaptation requires only a few minutes**, keeping deployment efficiency close to the original ACE

## Highlights & Insights

- The idea of **architectural separation + pre-training for generalization** is elegant—it draws an analogy between SCR and the "pre-trained model + knowledge base retrieval" paradigm in NLP, where map code corresponds to document embeddings in retrieval. This analogy is transferable to other vision tasks requiring scene- or instance-specific knowledge.
- The **query-aware pre-training** design cleverly addresses the problem of "only seeing mapping conditions during training"—manufacturing a condition gap during pre-training is more fundamental than simple data augmentation.
- The idea of representing scenes as **compact map codes** is transferable to SLAM, navigation, and other tasks where large-scale environments must be represented with a small number of parameters.

## Limitations & Future Work

- Relies on large-scale pre-training data (tens of thousands of scenes), incurring high data acquisition and training costs
- May still fail under extreme scene changes (e.g., complete structural renovation), as map codes cannot adapt to fundamental changes in scene geometry
- Map code update strategies are not discussed—how to efficiently update map codes as scenes change over time is a critical question for real-world deployment
- Integration with explicit 3D representations such as NeRF/3DGS is worth exploring, leveraging neural radiance fields to provide richer scene priors

## Related Work & Insights

- **vs ACE/ACE-Zero**: The ACE series represents the SOTA in SCR, but directly encoding scene information in regressor weights leads to poor generalization. ACE-G achieves a fundamentally improved architecture through separation while retaining ACE's efficient adaptation advantages.
- **vs HLoc (Feature Matching)**: Traditional methods such as HLoc inherently generalize across conditions through local feature matching, but require maintaining 3D point cloud maps and have slower inference. ACE-G approaches HLoc in accuracy while offering a more compact scene representation and faster inference.
- **vs FQN/DSAC++**: These methods also attempt to improve SCR generalization, but primarily at the data augmentation level. ACE-G addresses the problem at the architectural level, making it a more fundamental solution.

## Rating

- Novelty: ⭐⭐⭐⭐ Architectural separation and query-aware pre-training are novel within the SCR field, though the paradigm of "disentangled representation + pre-training for generalization" has precedents in other domains
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed comparisons and ablations on multiple challenging relocalization datasets covering scenes of varying difficulty
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, method description is fluent, and problem analysis is well-grounded
- Value: ⭐⭐⭐⭐ Represents an important advance for the SCR field, though the feasibility of large-scale pre-training in practical deployment warrants further validation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[ICCV 2025\] Dataset Ownership Verification for Pre-trained Masked Models](dataset_ownership_verification_for_pre-trained_masked_models.md)
- [\[ICCV 2025\] SynCity: Training-Free Generation of 3D Worlds](syncity_training-free_generation_of_3d_worlds.md)
- [\[ICCV 2025\] Make Your Training Flexible: Towards Deployment-Efficient Video Models](make_your_training_flexible_towards_deployment-efficient_video_models.md)
- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](eta_energy-based_test-time_adaptation_for_depth_completion.md)

</div>

<!-- RELATED:END -->
