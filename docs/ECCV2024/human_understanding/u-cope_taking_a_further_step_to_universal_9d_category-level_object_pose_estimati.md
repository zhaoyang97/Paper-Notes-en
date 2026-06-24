---
title: >-
  [Paper Note] U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation
description: >-
  [ECCV 2024][Human Understanding][Category-level pose estimation] This paper proposes U-COPE, the first category-level 9D pose estimation framework that unifiedly handles both rigid and articulated objects. By redefining rigid objects as single-part articulated objects, this work unifies the problem definition, independently extracts features for each part using Point Pair Features (PPF), and predicts key pose parameters via a universal voting strategy…
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Category-level pose estimation"
  - "9D pose"
  - "rigid objects"
  - "articulated objects"
  - "point pair features"
date: 2026-05-08
content_hash: 003a481bafd9a699
---

# U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Human/Object Pose Understanding  
**Keywords**: Category-level pose estimation, 9D pose, rigid objects, articulated objects, point pair features

## TL;DR
This paper proposes U-COPE, the first category-level 9D pose estimation framework that unifiedly handles both rigid and articulated objects. By redefining rigid objects as single-part articulated objects, this work unifies the problem definition, independently extracts features for each part using Point Pair Features (PPF), and predicts key pose parameters via a universal voting strategy, achieving state-of-the-art (SOTA) performance on both synthetic and real-world datasets.

## Background & Motivation

**Background**: Object pose estimation is a fundamental task in computer vision. For rigid objects (e.g., mugs, bowls), existing methods typically estimate 6D poses (3D rotation + 3D translation). For articulated objects (e.g., laptops, scissors), they need to estimate 9D poses (6D pose for each part + joint parameters). Historically, these two types of estimation have been studied separately using different model architectures and training pipelines.

**Limitations of Prior Work**: Independent development of pose estimation methods for rigid and articulated objects yields redundant deployment pipelines with multiple models. Critically, existing articulated-object methods usually directly regress joint parameters, a strategy that generalizes poorly under large intra-category shape variations. Furthermore, most existing methods rely on specific kinematic models or templates, rendering them difficult to generalize to unseen object instances.

**Key Challenge**: Mathematically, pose estimation for both rigid and articulated objects is fundamentally the same—inferring $SE(3)$ transformations from observational data. However, existing approaches artificially separate them due to different treatments of joint degrees of freedom. The lack of a unified framework not only increases system complexity but also limits the adaptability of these methods to more general environments.

**Goal**: (1) How to unify the pose representations of rigid and articulated objects? (2) How to design a universal pose estimation strategy capable of handling both rigid and articulated bodies? (3) How to achieve cross-instance generalization?

**Key Insight**: The authors present a key observation: rigid objects can be interpreted as articulated objects with only one part (and zero joint degrees of freedom). Based on this, all object pose estimation challenges can be cast as "part-wise estimation" problems. Furthermore, the authors adopt geometry-driven Point Pair Features (PPF) to represent the local shape of each part, eliminating the dependency on global templates.

**Core Idea**: Redefining rigid objects as single-part articulated objects and replacing direct regression with a PPF-based universal voting strategy to enable unified 9D pose estimation for both rigid and articulated objects.

## Method

### Overall Architecture
U-COPE takes a 3D point cloud or RGB-D image as input and outputs the 9D pose (including 3D rotation, 3D translation, and 3D scale) for each part of the object. The overall pipeline is: (1) extracting the point cloud of each object part from the input; (2) computing Point Pair Features independently for each part; (3) learning PPF representation via an end-to-end network; and (4) deriving key pose parameters from PPF features using a universal voting strategy. The network jointly optimizes three objectives: joint parameter prediction, part segmentation, and parameter voting-based 9D pose estimation.

### Key Designs

1. **Unified Representation**:
    - **Function**: Unifies pose estimation of rigid and articulated objects into a single mathematical framework.
    - **Mechanism**: For articulated objects, each part has an independent 6D pose $(R_i, t_i)$ and 3D scale $s_i$, connected via joint constraints. For rigid objects, the entire object is treated as an articulated object with only one part, where the joint constraint degenerates into an identity transformation. Consequently, pose estimation for all objects is unified into: independently estimating $(R_i, t_i, s_i)$ for each part and then combining them through joint constraints. The network architecture and loss functions are fully shared.
    - **Design Motivation**: A unified representation avoids redundancy in designing separate models for different object types, while leveraging the divide-and-conquer philosophy of part-level pose estimation—local estimations are generally more stable than global ones.

2. **Part-wise PPF Feature Extraction**:
    - **Function**: Extracts rotation-invariant geometric features for each object part.
    - **Mechanism**: Point Pair Features are composed of a pair of points $(p_i, p_j)$ and their corresponding normals $(n_i, n_j)$, encoded as a 4D vector $(\|d\|, \angle(n_i, d), \angle(n_j, d), \angle(n_i, n_j))$ where $d = p_j - p_i$. PPFs are inherently rotation- and translation-invariant, making them well-suited for describing local geometric structures. For each part, PPFs of internal point pairs are independently computed and then encoded via a PointNet++ type network to obtain part-level feature representations.
    - **Design Motivation**: Unlike approaches directly processing Cartesian coordinates, PPF eliminates the influence of the global coordinate system, allowing the learned features to directly transfer to unseen object instances. Processing parts independently also avoids interference between different parts.

3. **Universal Voting Strategy**:
    - **Function**: Robustly derives pose parameters from PPF features.
    - **Mechanism**: Instead of directly regressing pose parameters, U-COPE forces the network to vote on key geometric targets (e.g., rotation axis direction, centroid location, and scale ratios). Each PPF point-pair yields a voting result, which is subsequently aggregated via weighted averaging to obtain the final pose parameters. The voting strategies include: (a) rotation voting—each pair votes for a rotation residual; (b) translation voting—each point votes for its offset to the part centroid; (c) scale voting—each point votes for the normalized scale of the part. All votes are averaged using learned confidence weights.
    - **Design Motivation**: Voting is more robust than direct regression; individual outliers do not dominate the final result, and the effects of occlusions and noises are naturally diluted. This strategy is universally applicable to both rigid and articulated objects, as the transition between them is merely the number of voting parts.

### Loss & Training
The network is trained end-to-end to jointly optimize three objectives: (1) joint parameter loss—cross-entropy/regression losses for predicting joint type and joint parameters; (2) part segmentation loss—cross-entropy loss for predicting which part each point belongs to; and (3) 9D pose voting loss—$L_1$/$L_2$ regression losses between the voting results and ground-truth pose parameters. The three losses are combined in a weighted sum, with weights tuned on a validation set.

## Key Experimental Results

### Main Results

| Dataset | Metric | U-COPE | Prev. SOTA | Gain |
|---|---|---|---|---|
| Synthetic Rigid Objects | $5^\circ$ 5cm | SOTA | NPCS/SPD | Significant |
| Synthetic Articulated Objects | $5^\circ$ 5cm | SOTA | ANCSH | Significant |
| REAL275 (Real) | IoU75 mAP | SOTA | NOCS, etc. | Highly Competitive |
| Real Articulated Objects | Joint Acc. | SOTA | A-NCSH | Significant |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Full model (PPF + Voting) | Optimal | Complete model |
| w/o PPF (Coordinates directly) | Drop obvious | Loses rotation invariance |
| w/o Voting (Direct regression) | Drop significant | Sensitive to occlusions |
| w/o Joint Loss | Performance drop on articulated | Joint constraints improve accuracy |
| Rigid + Articulated Joint Training vs. Separate | Joint is superior | Unified framework has regularization effect |

### Key Findings
- The rotation invariance of PPF features is key to generalizing to unseen instances—removing PPFs severely degrades cross-instance generalization capability.
- The voting strategy shows the most profound advantage in occluded scenarios: with 40% occlusion, the voting model retains $> 70\%$ accuracy, whereas direct regression drops below $50\%$.
- The performance of the unified framework on both rigid and articulated objects is at least comparable to specialized approaches, validating the rationality of the "rigid = single-part articulated" design.
- Evaluation on real-world datasets demonstrates robust synthetic-to-real transfer capability.

## Highlights & Insights
- **The proposal of a unified perspective** is highly inspiring—treating rigid objects as single-part articulated objects. This simple redefinition breaks down the barrier between the two separate domains without requiring extra architectural designs. This "generalization-reduction" paradigm can be applied to other classification paradigms.
- **The comparison of Voting vs. Regression** reveals an important design principle: for problems requiring the inference of global properties from partial observations, voting is significantly more robust than direct regression. This conclusion is transferable to other 3D understanding tasks (e.g., 3D detection and point cloud registration).
- The utilization of PPF features is a classic yet effective choice—while many concurrent works pursue end-to-end learned features, handcrafted geometric priors remain irreplaceable when data is sparse.

## Limitations & Future Work
- The source code is not publicly released, presenting potential reproducibility challenges.
- The current framework only supports known joint types (rotational or translational); extensions are needed to support more complex joint types (e.g., ball joints, multi-DoF joints).
- Robustness under extreme occlusion ($>60\%$) has not been systematically evaluated.
- The computational complexity of PPF scales quadratically with the number of points, indicating potential efficiency hurdles on high-density point clouds.
- Lacks comparison with recent Transformer- or NeRF-based pose estimation methods.

## Related Work & Insights
- **vs. NOCS**: NOCS uses a Normalized Object Coordinate Space (NOCS) and requires training coordinate regressors for each individual object category. In contrast, U-COPE offers template-free feature extraction via PPF, providing better generalization capability.
- **vs. A-NCSH**: A-NCSH is designed specifically for articulated objects and requires predefined joint structures, whereas U-COPE predicts joint parameters through the network, proving more flexible.
- **vs. GPV-Pose**: Also uses a voting strategy but is limited to rigid objects. U-COPE extends voting to all individual parts of articulated objects.
- The core idea of this unified framework can be further extended to deformable objects (e.g., clothes, ropes), though this would require redefining what constitutes a "part".

## Rating
- Novelty: ⭐⭐⭐⭐ Redefining rigid/articulated pose estimation is an elegant problem abstraction.
- Experimental Thoroughness: ⭐⭐⭐ Thorough on synthetic data, though real-world evaluations remain limited in scale.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems, with solid motivation for the unified framework.
- Value: ⭐⭐⭐⭐ Provides a unified perspective for object pose estimation with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)
- [\[ECCV 2024\] LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation](lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](../../CVPR2025/human_understanding/gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[ECCV 2024\] FoundPose: Unseen Object Pose Estimation with Foundation Features](foundpose_unseen_object_pose_estimation_with_foundation_features.md)

</div>

<!-- RELATED:END -->
