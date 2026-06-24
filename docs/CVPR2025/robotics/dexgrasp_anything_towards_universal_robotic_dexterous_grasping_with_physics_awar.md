---
title: >-
  [Paper Note] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness
description: >-
  [CVPR 2025][Robotics][Dexterous Grasping] This paper proposes DexGrasp Anything, which integrates three physical constraint forces into the training and sampling phases of diffusion models to achieve SOTA dexterous grasp pose generation on almost all open datasets. Additionally, it constructs the largest-scale dexterous grasping dataset containing over 15K objects and more than 3.4 million grasping poses.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Dexterous Grasping"
  - "Diffusion Models"
  - "Physics Constraints"
  - "Large-scale Dataset"
  - "Universal Grasping"
date: 2026-05-08
content_hash: 24400e674b0d9ca4
---

# DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness

**Conference**: CVPR 2025  
**arXiv**: [2503.08257](https://arxiv.org/abs/2503.08257)  
**Code**: [https://github.com/4DVLab/DexGrasp-Anything](https://github.com/4DVLab/DexGrasp-Anything)  
**Area**: Image Generation  
**Keywords**: Dexterous Grasping, Diffusion Models, Physics Constraints, Large-scale Dataset, Universal Grasping

## TL;DR
This paper proposes DexGrasp Anything, which integrates three physical constraint forces into the training and sampling phases of diffusion models to achieve SOTA dexterous grasp pose generation on almost all open datasets. Additionally, it constructs the largest-scale dexterous grasping dataset containing over 15K objects and more than 3.4 million grasping poses.

## Background & Motivation

**Background**: Dexterous grasping is a core capability of robotic manipulation. The ShadowHand has 24 joint parameters, presenting an extremely large search space. Diffusion models have become the mainstream choice for modeling grasp distribution due to their ability to generate diverse and high-quality samples.

**Limitations of Prior Work**: Existing diffusion-based methods lack physical constraints during training and sampling, often leading to hand-object interpenetration or insufficient contact in the generated grasp poses. Furthermore, existing datasets are limited in scale and have narrow object categories.

**Key Challenge**: The standard MSE training objective of diffusion models only focuses on noise prediction accuracy, without containing explicit supervision of physical feasibility.

**Goal**: (1) Embed physical priors into the training and sampling of diffusion models; (2) Construct a large-scale, high-quality dexterous grasping dataset.

**Key Insight**: Leveraging Tweedie's formula to infer the clean sample $\hat{h}_0$ from noise prediction to compute physical constraint losses, and continuously imposing physical guidance via posterior sampling during the sampling phase.

**Core Idea**: Three physical constraint forces (surface pulling, external-penetration repulsion, self-penetration repulsion) + dual injection in training and sampling + LLM-augmented object representations.

## Method

### Overall Architecture
Given an object 3D point cloud $O$, a conditional diffusion model is trained to generate dexterous hand grasp poses $h = (\theta, R, t) \in \mathbb{R}^{33}$. Object features are encoded by Point Transformer and fused with LLM semantic priors.

### Key Designs

1. **Three Physical Constraint Forces**:

    - **Function**: Ensure physical feasibility of generated grasp poses.
    - **Mechanism**: (a) **Surface Pulling Force (SPF)**: Computes the nearest-neighbor distance to the object for points on the inner surface of the hand, applying pulling forces to close points to bring fingers closer; (b) **External-Penetration Repulsion Force (ERF)**: Uses signed distances to detect penetrations and apply repulsion forces; (c) **Self-Penetration Repulsion Force (SRF)**: Applies repulsion forces when pairwise distances between fingers fall below a threshold to prevent self-penetration.
    - **Design Motivation**: SPF ensures "stable grasping", ERF ensures "no penetration", and SRF ensures "no self-collision"—covering the three core physical requirements of dexterous grasping.

2. **Physics-Aware Training + Physics-Guided Sampling**:

    - **Function**: Embed physical constraints simultaneously into both training and inference.
    - **Mechanism**: During training, Tweedie's formula is used to infer the clean sample $\hat{h}_0$ from noise predictions, applying $L_{PADG} = L_{simple} + \sum \alpha_i L_{PA_i}(\hat{h}_0)$. During sampling, classifier guidance-style posterior sampling is utilized to map physical constraint gradients to posterior mean offsets, with spherical Gaussian constraints applied to mitigate estimation bias.
    - **Design Motivation**: The training phase provides "sparse" physical supervision, while the sampling phase iteratively refines and guides the distribution toward physically feasible regions, serving as complementary mechanisms.

3. **LLM-Augmented Object Representation**:

    - **Function**: Complement point cloud geometric features with semantic priors.
    - **Mechanism**: LLMs generate object descriptions, which are encoded into semantic vectors by BERT. These are concatenated with Point Transformer geometric features and integrated into the diffusion backbone via cross-attention.
    - **Design Motivation**: Pure point clouds struggle to distinguish objects with similar functions but different shapes; semantic priors provide high-level knowledge of "how to grasp".

### Loss & Training
$L_{PADG} = L_{simple} + \alpha_1 L_{SPF} + \alpha_2 L_{ERF} + \alpha_3 L_{SRF}$. A "model-in-the-loop" strategy is adopted: the trained model is used to generate new poses, which are added to the dataset after validation in IsaacGym.

## Key Experimental Results

### Main Results
Evaluation across 5 datasets (Suc. 6 = success rate under a 6N force):

| Method | DexGraspNet | UniDexGrasp | MultiDex | RealDex | DexGRAB |
|------|-------------|-------------|----------|---------|---------|
| UniDexGrasp | 33.9 | 23.7 | 21.6 | 27.1 | 20.8 |
| GraspTTA | 18.6 | 21.0 | 30.3 | 13.3 | 14.4 |
| DexGrasp Anything | **Ours (Best)** | **Ours (Best)** | **Ours (Best)** | **Ours (Best)** | **Ours (Best)** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Base Diffusion | Baseline | No physical constraints |
| + SPF + ERF + SRF | Significant Gain | Joint force combination |
| + LLM Semantics | Additional Gain | Semantic priors are beneficial |
| DGA Dataset | All methods improved | Data quality is crucial |

### Key Findings
- Physical constraints contribute independently during both training and sampling phases, with the combination yielding the best performance.
- The DGA dataset (3.4M grasps) significantly boosts existing methods—improving performance solely through data substitution.
- The "model-in-the-loop" strategy allows for continuous dataset expansion.

## Highlights & Insights
- **Paradigm of Injecting Physical Constraints into Diffusion Models**: Constraints are applied by mapping to the data space via Tweedie's formula, which can be transferred to fields like molecular design.
- **Data Flywheel**: Drawing inspiration from SAM's model-in-the-loop approach to expand the dataset represents a replicable growth model.
- **The tripartite force-balancing design is both complete and highly intuitive.**

## Limitations & Future Work
- Validated only on ShadowHand; transferability to other dexterous hands remains unexplored.
- Open-loop method, without considering contact feedback during execution.
- The sim-to-real gap between synthetic objects and the real world remains to be addressed.

## Related Work & Insights
- **vs UniDexGrasp**: Ours integrates physical constraints into both diffusion model training and sampling based on their work.
- **vs SceneDiffuser**: General scene interaction diffusion, lacking specialized constraints for dexterous grasping.
- The dataset construction methodology can serve as a reference for other manipulation domains.

## Rating
- Novelty: ⭐⭐⭐⭐ Complete solution for injecting physical constraints into diffusion training and sampling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 datasets with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology and complete formulations.
- Value: ⭐⭐⭐⭐ Both the dataset and methodology make significant contributions to the dexterous grasping community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DemoGrasp: Universal Dexterous Grasping from a Single Demonstration](../../ICLR2026/robotics/demograsp_universal_dexterous_grasping_from_a_single_demonstration.md)
- [\[CVPR 2026\] GeoDexGrasp: Geometry-aware Generation for Data-efficient and Physics-plausible Dexterous Grasping](../../CVPR2026/robotics/geodexgrasp_geometry-aware_generation_for_data-efficient_and_physics-plausible_d.md)
- [\[CVPR 2025\] Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References](learning_physics-based_full-body_human_reaching_and_grasping_from_brief_walking_.md)
- [\[CVPR 2025\] ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping](manivideo_generating_hand-object_manipulation_video_with_dexterous_and_generaliz.md)
- [\[CVPR 2025\] ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping](zerograsp_zero-shot_shape_reconstruction_enabled_robotic_grasping.md)

</div>

<!-- RELATED:END -->
