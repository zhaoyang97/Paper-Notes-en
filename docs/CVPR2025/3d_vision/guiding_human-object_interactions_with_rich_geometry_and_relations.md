---
title: >-
  [Paper Note] Guiding Human-Object Interactions with Rich Geometry and Relations
description: >-
  [CVPR 2025][3D Vision][Human-Object Interaction] This paper proposes the ROG framework, which constructs an Interactive Distance Field (IDF) by sampling geometry-rich keypoints on object meshes. It utilizes a diffusion-based relation model to guide the motion generation model during inference to produce relation-aware and semantically-aligned human-object interactions, significantly outperforming the state-of-the-art on the FullBodyManipulation dataset.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human-Object Interaction"
  - "Interactive Distance Field"
  - "Diffusion Models"
  - "Geometric Representation"
  - "Motion Generation"
date: 2026-05-08
content_hash: 92cb738197789c9e
---

# Guiding Human-Object Interactions with Rich Geometry and Relations

**Conference**: CVPR 2025  
**arXiv**: [2503.20172](https://arxiv.org/abs/2503.20172)  
**Code**: [https://lalalfhdh.github.io/rog_page/](https://lalalfhdh.github.io/rog_page/)  
**Area**: 3D Vision / Motion Generation  
**Keywords**: Human-Object Interaction, Interactive Distance Field, Diffusion Models, Geometric Representation, Motion Generation

## TL;DR

This paper proposes the ROG framework, which constructs an Interactive Distance Field (IDF) by sampling geometry-rich keypoints on object meshes. It utilizes a diffusion-based relation model to guide the motion generation model during inference to produce relation-aware and semantically-aligned human-object interactions, significantly outperforming the state-of-the-art on the FullBodyManipulation dataset.

## Background & Motivation

**Background**: Human-Object Interaction (HOI) synthesis is a core technology in fields such as virtual reality, animation, and robotics. Recently, the success of diffusion models has driven numerous studies to apply them to HOI generation, including incorporating prior information like textual descriptions, physical forces, hand joint positions, and contact maps, or utilizing post-optimization strategies to improve interaction quality.

**Limitations of Prior Work**: Existing methods suffer from severe oversimplification when representing object geometry and modeling human-object spatial relations. (1) Oversimplified object representation: Many approaches represent the object using only its centroid or a single closest point to the human body, neglecting the overall geometric complexity of the object. (2) Insufficient relation modeling: Contact-based methods rely on fixed contact points, which perform poorly in dynamic, multi-stage interactions; distance-based methods only consider a single-perspective relation between human joints and the closest object vertex, failing to comprehensively capture mutual spatial relations.

**Key Challenge**: High-fidelity HOI generation requires precise geometric representations and complex spatio-temporal relation modeling. However, the complexity of high-dimensional dynamic interactions makes direct use of all surface points computationally infeasible, while oversimplification leads to the loss of critical geometric information.

**Goal**: (1) How to efficiently represent object geometry without losing crucial details? (2) How to effectively model and utilize complex spatio-temporal relations between human and objects to guide motion generation?

**Key Insight**: Starting from the perspectives of object sampling and spatial distance fields, the authors efficiently represent object geometry using 24 carefully selected keypoints, construct a complete distance matrix (IDF) between human joints and object keypoints, and then train a dedicated relation model to learn the distribution of this distance field, ultimately using the relation model to guide motion generation during inference.

**Core Idea**: Represent object geometry with 24 keypoints sampled via boundary points and Poisson Disk Sampling, construct an Interactive Distance Field (IDF) to encode spatio-temporal relations, and guide motion generation with a diffusion relation model.

## Method

### Overall Architecture

ROG is a two-stage diffusion framework. Given an object mesh, a human skeleton, and a text prompt: (1) A motion generation model generates an initial motion sequence of the human and object based on MDM. (2) A relation model takes the IDF matrix computed from the initial motion as input and outputs a refined IDF. (3) The refined IDF optimizes the initial motion via gradient guidance, rendering the final output interactions more realistic and semantically aligned. The two models are trained separately and collaborate during inference through a guidance mechanism.

### Key Designs

1. **Object Key Points Sampling**:

    - **Function**: Compactly and comprehensively represent the 3D geometry of the object using 24 keypoints.
    - **Mechanism**: First, the Axis-Aligned Bounding Box (AABB) of the object is calculated to find 8 boundary points on the object's surface closest to the 8 vertices of the bounding box, capturing the overall silhouette and extreme positions of the object. Then, Poisson Disk Sampling (PDS) is employed to uniformly sample an additional 16 points on the object surface; PDS ensures a uniform distribution via minimum distance constraints, capturing fine-grained shape variations. This yields a total of 24 keypoints $\mathbf{P} = \{\mathbf{p}_1, ..., \mathbf{p}_{24}\}$, establishing a one-to-one correspondence with the 24 joint points of the human skeleton.
    - **Design Motivation**: Directly utilizing all surface vertices incurs excessive computational cost and redundancy, whereas minimal representations like the centroid lose geometric details. The combination of boundary points and PDS points covers the object silhouette (coarse-grained) while retaining surface details (fine-grained), achieving a balance between accuracy and efficiency.

2. **Interactive Distance Field (IDF)**:

    - **Function**: Comprehensively encode the spatio-temporal distance relationships between human and object throughout the entire interaction sequence in a matrix form.
    - **Mechanism**: For an HOI sequence of $N$ frames, a 3D distance matrix $\mathbf{D} \in \mathbb{R}^{24 \times 24 \times N}$ is constructed, where the $(i,j,n)$-th element $\mathbf{D}_{i,j,n} = \|\mathbf{q}_{i,n} - \mathbf{p}_{j,n}\|_2^2$ represents the squared Euclidean distance between the $i$-th human joint and the $j$-th object keypoint in the $n$-th frame. During training, an IDF Loss $\mathcal{L}_{IDF} = \|\mathbf{D}_{pr} - \mathbf{D}_{gt}\|_2^2$ is introduced to directly supervise the motion generation model in learning correct spatial relationships, with a weight of $\lambda_{IDF}=5.0$.
    - **Design Motivation**: Existing methods either focus on binary contact/non-contact relations or calculate a single distance based on centroids. The IDF matrix provides a complete distance mapping between all human joints and all object keypoints, precisely describing the dynamic changes in joint proximity, contact, and separation during interaction.

3. **Relation Model & Guidance**:

    - **Function**: Learn the prior of realistic IDF distributions and guide the motion generation model to produce more realistic interactions during inference.
    - **Mechanism**: The relation model is a diffusion model based on the Video Diffusion Transformer. It takes a noisy IDF matrix as input and aims to denoise it to recover the ground-truth IDF. The model incorporates spatial self-attention (capturing dependencies between different body parts and object parts on a reduced $4 \times 4$ spatial grid) and temporal self-attention (capturing the dynamic evolution of interactions along the temporal dimension $N$). During inference, the motion generation model first produces an initial motion $\tilde{\mathbf{m}}_0$, from which the IDF matrix $\mathbf{D}$ is calculated and fed into the relation model to obtain a refined $\tilde{\mathbf{D}}$. Then, we propagate gradients to the initial motion via the guidance loss $L_{guidance} = \|\mathbf{D} - \tilde{\mathbf{D}}\|_2^2$ using the L-BFGS optimizer. Guidance is executed only during the final 10 timesteps of the denoising process.
    - **Design Motivation**: A standalone motion generation model may yield unrealistic contacts and dynamic behaviors. By training a separate relation model to learn the real distribution of the IDF, its output can be used to "correct" the output of the motion generation model during inference.

### Loss & Training

- **Motion Generation Model**: $\mathcal{L}_m = \mathcal{L}_{rec} + \lambda_{IDF}\mathcal{L}_{IDF}$, reconstruction loss + IDF loss, $\lambda_{IDF}=5.0$
- **Relation Model**: $\mathcal{L}_D = \|\mathbf{D}_0 - \tilde{\mathbf{D}}_0\|_2^2$, standard diffusion denoising target
- The two models are trained independently. Motion generation uses an 8-layer Transformer + batch_size=64, and the relation model uses the VDT configuration + batch_size=8.
- The optimizer is AdamW with lr=$1 \times 10^{-4}$, 1000 diffusion steps, and DDPM sampling for both models.

## Key Experimental Results

### Main Results

| Method | R-Precision Top-1↑ | R-Precision Top-3↑ | FID↓ | Contact%↑ | Collision% | MDev↓ |
|------|-------|-------|------|-----------|------------|-------|
| Real motions | 0.651 | 0.917 | 0.001 | 0.623 | 0.157 | 4.846 |
| InterGen | 0.490 | 0.685 | 19.038 | 0.179 | 0.156 | 39.795 |
| MDM | 0.495 | 0.681 | 9.775 | 0.349 | 0.210 | 9.549 |
| HOI-Diff | 0.534 | 0.722 | 11.875 | 0.372 | 0.175 | 58.728 |
| CHOIS | 0.630 | 0.844 | 5.227 | 0.444 | 0.208 | 13.408 |
| **ROG (Ours)** | **0.706** | **0.902** | **5.119** | **0.466** | **0.200** | **5.815** |

### Ablation Study

| Component | obj-kp | IDF loss | Guidance | Top-1↑ | FID↓ | Contact% | MDev↓ |
|------|--------|----------|----------|--------|------|----------|-------|
| Baseline(MDM) | ✗ | ✗ | ✗ | 0.495 | 9.775 | 0.349 | 9.549 |
| +obj-kp | ✓ | ✗ | ✗ | 0.547 | 7.514 | 0.374 | 9.227 |
| +IDF loss | ✓ | ✓ | ✗ | 0.666 | 5.726 | 0.424 | 7.020 |
| +Guidance(C) | ✓ | ✓ | Centroid | 0.668 | 5.902 | 0.364 | 9.936 |
| +Guidance(D) | ✓ | ✓ | Full IDF | **0.706** | **5.119** | **0.466** | **5.815** |

### Key Findings

- Introducing object keypoint representation (obj-kp) reduces the FID from 9.775 to 7.514, demonstrating that geometric information directly improves motion realism.
- The IDF Loss is the largest source of performance gain, significantly boosting R-Precision Top-1 from 0.547 to 0.666.
- Guidance using the full distance matrix (D) performs substantially better than guidance relying solely on centroid distance (C).

## Highlights & Insights

- **Exquisite geometric representation design**: The combination of boundary points and PDS sampling ensures both complete coverage of the object silhouette and uniform sampling of surface details.
- **Inspiring IDF concept**: Unifying discrete contact/distance relationships into a continuous 3D distance field provides an elegant mathematical framework for modeling human-object interactions.
- **Clear division of labor in the two stages**: The motion generation model is responsible for "what action to perform," while the relation model is responsible for "whether the action is reasonable."

## Limitations & Future Work

- Dataset limitations: Articulated objects are excluded, and finger motions are not modeled.
- L-BFGS optimization increases computational overhead during inference.
- The method can be extended to multi-person and multi-object interaction scenarios in the future.

## Related Work & Insights

- **MDM**: The foundational architecture for the motion generation model.
- **CHOIS**: The strongest baseline, but requires additional control signals.
- Insights: The concept of IDF can be extended to other generative tasks requiring spatial relation modeling.

## Rating

- Novelty: ⭐⭐⭐⭐ (The IDF concept is novel, and the relation model guidance mechanism is cleverly designed)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive quantitative, qualitative, and ablation experiments)
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐ (Provides significant momentum to the HOI generation field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[ICCV 2025\] SceneMI: Motion In-betweening for Modeling Human-Scene Interactions](../../ICCV2025/3d_vision/scenemi_motion_in-betweening_for_modeling_human-scene_interaction.md)
- [\[ICCV 2025\] Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints](../../ICCV2025/3d_vision/guiding_diffusion-based_articulated_object_generation_by_partial_point_cloud_ali.md)

</div>

<!-- RELATED:END -->
