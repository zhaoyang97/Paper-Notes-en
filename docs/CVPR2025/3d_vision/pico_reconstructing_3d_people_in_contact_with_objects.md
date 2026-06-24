---
title: >-
  [Paper Note] PICO: Reconstructing 3D People In Contact with Objects
description: >-
  [CVPR 2025][3D Vision][Human-Object Interaction Reconstruction] PICO proposes a comprehensive framework comprising a dataset (PICO-db) and a fitting method (PICO-fit). By establishing dense bijective contact correspondences between humans and objects, it reconstructs realistic 3D human-object interaction scenes from a single in-the-wild image, supporting arbitrary object categories.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human-Object Interaction Reconstruction"
  - "Contact Estimation"
  - "3D Object Retrieval"
  - "Human Pose"
  - "Optimization Fitting"
date: 2026-05-08
content_hash: 438407a636351c63
---

# PICO: Reconstructing 3D People In Contact with Objects

**Conference**: CVPR 2025  
**arXiv**: [2504.17695](https://arxiv.org/abs/2504.17695)  
**Code**: [https://pico.is.tue.mpg.de](https://pico.is.tue.mpg.de)  
**Area**: 3D Vision  
**Keywords**: Human-Object Interaction Reconstruction, Contact Estimation, 3D Object Retrieval, Human Pose, Optimization Fitting

## TL;DR

PICO proposes a comprehensive framework comprising a dataset (PICO-db) and a fitting method (PICO-fit). By establishing dense bijective contact correspondences between humans and objects, it reconstructs realistic 3D human-object interaction scenes from a single in-the-wild image, supporting arbitrary object categories.

## Background & Motivation

**Background**: Reconstructing 3D human-object interaction (HOI) from a single image requires inferring human pose and shape, object pose and shape, and their spatial layout. Existing methods such as PHOSA utilize hand-designed category-specific contact constraints, while CONTHO and HDM employ regression-based approaches but are limited to predefined object categories.

**Limitations of Prior Work**: On one hand, the lack of a unified statistical model for object shapes (unlike SMPL for humans) makes recovering 3D object shapes from a single image extremely challenging. On the other hand, existing contact estimation methods either perform inference only in 2D, estimate 3D contacts solely on the human body while ignoring the object, or are trained on synthetic data, failing to generalize to real-world images.

**Key Challenge**: To achieve robust 3D HOI reconstruction from in-the-wild images, one must simultaneously address the diversity of object shapes and the precise annotation of human-object contact correspondences. These two coupled tasks currently lack adequate data support.

**Goal**: (1) Construct a dataset containing in-the-wild images with annotated bijective contact correspondences between the human body and the object; (2) Develop a contact-guided optimization fitting method capable of handling arbitrary object categories.

**Key Insight**: The authors observe that body contact forms continuous 'patches', from which contact axes can be automatically generated via PCA. This allows projecting body contact onto objects with minimal human annotation (only 2 clicks per patch). Meanwhile, the joint latent space of the OpenShape foundation model is leveraged to achieve category-agnostic 3D object shape retrieval.

**Core Idea**: Construct a dense human-object contact correspondence dataset using a contact transfer method with minimal human effort, and then leverage these contact correspondences as constraints to optimize 3D human-object interaction reconstruction.

## Method

### Overall Architecture

The PICO framework comprises two core components. **PICO-db** is a dataset containing 4,123 in-the-wild images across 44 object categories and 627 object instances, with each image annotated with dense 3D contact correspondences between the human body and the object. **PICO-fit** is a three-stage analysis-by-synthesis optimization method that takes a single natural image and outputs the 3D human mesh, object mesh, and their spatial layout.

### Key Designs

1. **OpenShape-Based Object Shape Retrieval**:

    - **Function**: Automatically retrieve a matching 3D object mesh from a database given an input image.
    - **Mechanism**: Leverage OpenShape to embed both images and 3D shapes into a joint latent space. All meshes in the Objaverse-LVIS database are embedded offline; during online inference, the input image is embedded, and the nearest neighbor is identified via cosine similarity. The retrieved object meshes preserve 3D details and demonstrate robustness to occlusions.
    - **Design Motivation**: Traditional methods require prior knowledge of object categories or rely on diffusion models (which demand full object visibility), whereas this retrieval-based approach scales to arbitrary unseen categories and naturally improves as the database expands.

2. **Minimal-Effort Contact Transfer Mechanism**:

    - **Function**: Project contact patches annotated only on the human body in the DAMON dataset onto the object mesh, establishing bijective correspondences.
    - **Mechanism**: The contact patch on the body automatically generates a contact axis (the direction of the first principal component) via PCA. Annotators only need to click twice on the object (defining the start and direction of the axis) to complete the transfer. For non-convex regions like fingers, a "webbed hand" proxy mesh is created via convex hulls to bypass geodesic tracking difficulties.
    - **Design Motivation**: While methods like ContactEdit are theoretically feasible, they require professional 3D expertise. PICO democratizes this process through automated axis generation and simplified annotations, enabling large-scale data collection via AMT crowdsourcing.

3. **Three-Stage Analysis-by-Synthesis Optimization Fitting**:

    - **Function**: Progressively optimize the 3D human and object poses, shapes, and spatial layout starting from initial estimates.
    - **Mechanism**: Stage 1 fixes the human body and optimizes the object's rotation and translation $R_o, t_o$ via an L2 contact correspondence loss. Stage 2 incorporates an object mask IoU loss $\mathcal{L}_o^m$, an SDF-based penetration loss $\mathcal{L}_p$, and a scale loss $\mathcal{L}_o^s$ to align the object with the image. Stage 3 optimizes only the contacting limbs using local pose parameters $\theta_C$ from the kinematic chain (from torso to contact joints), paired with a human mask loss and pose regularization.
    - **Design Motivation**: To avoid the "chicken-and-egg" issue inherent in joint optimization, the staged pipeline provides explicit constraints for each step. Optimizing only the contacting limb chain instead of the entire body prevents distortion caused by depth ambiguities.

### Loss & Training

PICO-fit is an optimization-based (non-learning) method, utilizing different loss combinations across the three stages:

- **Stage 1**: $L_1 = \mathcal{L}_c$, consisting only of the contact correspondence distance loss.
- **Stage 2**: $L_2 = \lambda_c \mathcal{L}_c + \lambda_p \mathcal{L}_p + \lambda_o^m \mathcal{L}_o^m + \lambda_o^s \mathcal{L}_o^s$, combining contact, penetration, object mask, and scale losses.
- **Stage 3**: $L_3 = \lambda_c \mathcal{L}_c + \lambda_p \mathcal{L}_p + \lambda_h^m \mathcal{L}_h^m + \lambda_{\theta_C} \mathcal{L}_{\theta_C}$, combining contact, penetration, human mask, and pose regularization losses.

Contact initialization utilizes DECO to infer body contacts combined with GPT-4V verification to reduce false positives, while the object scale is initialized via GPT-4V.

## Key Experimental Results

### Main Results

| Method | Type | PA-CDh↓ | PA-CDo↓ | PA-CDh+o↓ | Perceptual Preference Rate |
|------|------|---------|---------|-----------|-----------|
| HDM | Regression | 17.34 | 14.12 | 13.60 | 20.1% vs 79.9% |
| CONTHO* | Regression+GT | 8.16 | 23.26 | 12.81 | 24.7% vs 75.3% |
| PHOSA* | Optimization+GT | 10.12 | 20.91 | 13.28 | 32.0% vs 68.0% |
| PICO-fit | Optimization | 7.43 | 21.85 | 10.33 | 37.3% vs 62.7% |
| **PICO-fit*** | **Optimization+GT** | **6.66** | **13.34** | **8.36** | - |

### Ablation Study

| Stage | PA-CDh↓ | PA-CDo↓ | PA-CDh+o↓ |
|------|---------|---------|-----------|
| Stage 1 only | 7.25 | 24.51 | 11.47 |
| Stage 1+2 | 6.65 | 13.67 | 8.40 |
| Stage 1+2+3 (Full) | 6.66 | 13.34 | 8.36 |

### Key Findings

- Even without using GT contacts, PICO-fit outperforms CONTHO* and PHOSA* which use GT contacts (PA-CDh+o: 10.33 vs 12.81/13.28).
- In perceptual studies, PICO-fit* is perceived as more realistic than all baselines, achieving an average preference rate of 74.4%.
- Stage 2 yields the most significant improvement in object metrics (PA-CDo: 24.51 $\rightarrow$ 13.67), while Stage 3 brings minor overall improvements.
- PICO-fit successfully handles object categories like sofas, bananas, and frisbees for the first time, which were challenging for prior methods.

## Highlights & Insights

- Establishing high-quality bijective human-object contact correspondences with minimal human annotation (2 clicks per patch) offers an elegant crowdsourcing paradigm.
- The retrieval-based (rather than generative) object shape acquisition strategy is simple, effective, and inherently scalable.
- Decomposing the complex joint optimization problem into three well-defined sub-problems clarifies the optimization variables and constraints at each stage.
- Utilizing GPT-4V for contact verification and scale initialization represents an interesting application of multimodal foundation models in geometric vision.

## Limitations & Future Work

- The method relies on DECO for contact detection, which is imperfect (especially with severe false positives on feet).
- Object shape retrieval depends on database coverage; unseen or rare object types may still cause failures.
- The authors plan to use PICO-fit to automatically generate pseudo-GT to train feed-forward contact regressors, replacing the current nearest-neighbor search.
- Future directions include exploring vision-language models to transcend the constraints of limited datasets.

## Related Work & Insights

- DAMON [77] provides human contact annotations but ignores objects; PICO elegantly extends this dataset.
- The contact axis concept from ContactEdit [44] is redesigned into a more user-friendly crowdsourcing scheme.
- OpenShape's [53] joint embedding space serves as a powerful tool for cross-modal retrieval.
- Insight: Contact correspondences serve as a key bridge for understanding 3D human-object interaction. Although slower, optimization methods are more robust in OOD scenarios compared to pure regression.

## Rating

- **Novelty**: 8/10 — The contact transfer and bijective correspondence data collection approach is novel, and the three-stage optimization design is sound.
- **Experimental Thoroughness**: 8/10 — Extensive evaluations including InterCap cross-distribution evaluation, AMT perceptual studies, and ablation studies are provided, though efficiency analysis is lacking.
- **Writing Quality**: 8/10 — Clear motivation, highly detailed method descriptions, and rich illustrations.
- **Value**: 8/10 — Both the dataset and code are open-sourced, providing an extensible foundation for in-the-wild HOI reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reconstructing People, Places, and Cameras](reconstructing_people_places_and_cameras.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [\[CVPR 2025\] UnCommon Objects in 3D](uncommon_objects_in_3d.md)
- [\[CVPR 2025\] Reconstructing Humans with a Biomechanically Accurate Skeleton](reconstructing_humans_with_a_biomechanically_accurate_skeleton.md)

</div>

<!-- RELATED:END -->
