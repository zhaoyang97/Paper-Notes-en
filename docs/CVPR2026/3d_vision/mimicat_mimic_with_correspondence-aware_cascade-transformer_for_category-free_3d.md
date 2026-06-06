---
title: >-
  [Paper Note] MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer
description: >-
  [CVPR 2026][3D Vision][3D pose transfer] This paper proposes MimiCAT, a cascade Transformer framework that learns flexible many-to-many soft correspondences via semantic keypoint labels. Combined with the million-scale m…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D pose transfer"
  - "cross-category transfer"
  - "soft correspondence"
  - "cascade Transformer"
  - "large-scale motion dataset"
date: 2026-05-08
content_hash: 60e03778b7ba01e0
---

# MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer

**Conference**: CVPR 2026
**arXiv**: [2511.18370](https://arxiv.org/abs/2511.18370)  
**Code**: [https://mimicat3d.github.io/](https://mimicat3d.github.io/) (Project Page)  
**Area**: 3D Vision
**Keywords**: 3D pose transfer, cross-category transfer, soft correspondence, cascade Transformer, large-scale motion dataset

## TL;DR
This paper proposes MimiCAT, a cascade Transformer framework that learns flexible many-to-many soft correspondences via semantic keypoint labels. Combined with the million-scale multi-category motion dataset PokeAnimDB, it achieves, for the first time, high-quality cross-category 3D pose transfer (e.g., humanoid to quadruped/bird).

## Background & Motivation

1. **Background**: 3D pose transfer aims to apply the pose of a source character to a target character while preserving the target's geometric identity and the source's pose information. Existing methods are mostly limited to structurally similar characters (e.g., humanoid to robot), achieving transfer through learned one-to-one correspondences at the keypoint or vertex level.

2. **Limitations of Prior Work**: When the body structures of source and target characters differ drastically (e.g., humanoid to bird), one-to-one mappings fail entirely. How should two arms correspond to two wings? Furthermore, existing methods predominantly rely on human motion datasets (e.g., AMASS), producing out-of-distribution and unnatural deformations on non-humanoid characters.

3. **Key Challenge**: Characters across different categories have fundamentally different skeletal structures, keypoint counts, and rotation patterns, making traditional one-to-one keypoint mappings incapable of expressing the complex many-to-many correspondences involved. There is also a lack of large-scale datasets encompassing multi-type character animations.

4. **Goal**: (a) How to establish flexible correspondences between characters with vastly different structures? (b) How to obtain sufficiently diverse cross-category motion data for training? (c) How to ensure that generated pose transformations are physically plausible?

5. **Key Insight**: The authors observe that skeletal keypoints typically carry semantic labels (e.g., "limbs" can correspond to human "arms" and bird "wings"). Leveraging this semantic information circumvents the need for manual correspondence annotation; CLIP-encoded text labels can be used to generate many-to-many soft correspondence pseudo-labels.

6. **Core Idea**: Cross-category 3D pose transfer is achieved by combining semantic keypoint label-driven soft correspondence learning, a shape-aware cascade Transformer, and the million-scale multi-category motion dataset PokeAnimDB.

## Method

### Overall Architecture
MimiCAT takes the posed mesh of a source character and the rest-pose mesh of a target character as input, and outputs a deformed mesh of the target character in the source pose. The pipeline consists of two stages: Stage I trains a correspondence Transformer $\mathcal{G}$ to learn a soft correspondence matrix between source and target keypoints; Stage II freezes $\mathcal{G}$ and trains a pose transfer Transformer $\mathcal{H}$ to generate the final transformation parameters for the target character via a cycle-consistency objective, followed by Linear Blend Skinning (LBS) to obtain the final mesh.

### Key Designs

1. **PokeAnimDB: Large-Scale Multi-Category Motion Dataset**
    - **Function**: Provides cross-category training data.
    - **Mechanism**: Collects 28,809 high-quality artist-designed motions from 975 characters (including humanoids, quadrupeds, birds, reptiles, fish, insects, etc.) sourced from the web, totaling approximately 4.4 million frames. Each character is unified to a 5,000-face mesh, skeletal animations are stored in `.bvh` format, and bone semantic names are recorded.
    - **Design Motivation**: Existing datasets (e.g., Mixamo, AMASS) are either limited to humanoid characters or contain only a small variety of character types. Cross-category transfer necessitates data covering diverse morphologies.

2. **Correspondence Transformer $\mathcal{G}$ (Soft Correspondence Learning)**
    - **Function**: Estimates many-to-many soft correspondences between keypoint sets of different lengths.
    - **Mechanism**: Keypoint coordinates are encoded via an MLP to produce keypoint tokens $g_{\mathbf{C}}$; a pretrained 3D shape encoder extracts geometric features to generate shape tokens $g_{\mathbf{M}}$. Both are concatenated and fed into Transformer blocks to learn shape-aware representations $\mathbf{g}^{\text{src}}$ and $\mathbf{g}^{\text{tgt}}$. A learnable affine matrix $\mathbf{A}$ is used to compute the similarity $\mathbf{S} = \exp(\mathbf{g}^{\text{src}\top}\mathbf{A}\mathbf{g}^{\text{tgt}})$, which is normalized via the Sinkhorn algorithm into a doubly stochastic matrix $\mathbf{M}$, where each $\mathbf{M}_{i,j}$ represents the soft matching probability between source keypoint $i$ and target keypoint $j$.
    - **Design Motivation**: GNNs are abandoned (as they rely on skeletal connectivity priors that limit generalization) in favor of direct coordinate encoding. Shape features are incorporated to improve body-part discriminability. The doubly stochastic matrix produced by Sinkhorn naturally supports many-to-many matching, offering greater flexibility than the one-to-one Hungarian algorithm.

3. **Correspondence-Based Transformation Initialization (with Quaternion Weighted Averaging)**
    - **Function**: Maps source transformations to initial transformations for target keypoints using the soft correspondence matrix.
    - **Mechanism**: For each target keypoint $j$, source translations and positions are aggregated via weighted averaging using $\mathbf{M}$. However, naive quaternion averaging yields non-unit rotations and sign ambiguities. A Frobenius-norm minimization-based rotation averaging is therefore adopted, whose solution is the eigenvector corresponding to the largest eigenvalue of the weighted covariance matrix $\sum_i \mathbf{M}_{i,j}\mathbf{q}_i\mathbf{q}_i^\top$.
    - **Design Motivation**: Naive quaternion averaging produces distortions and flips (confirmed experimentally). Frobenius rotation averaging is mathematically rigorous and guarantees valid rotations.

4. **Pose Transfer Transformer $\mathcal{H}$ (Shape-Aware Pose Transfer)**
    - **Function**: Refines the initialized transformations into final target transformations.
    - **Mechanism**: Geometric features are injected into the target representation via cross-attention ($\delta_\mathbf{f} = \mathbf{f}_{\mathbf{V}^{\text{src}}} - \mathbf{f}_{\bar{\mathbf{V}}^{\text{src}}}$ encodes source deformation information), fused with target geometric features to produce shape-conditioned tokens. Keypoint tokens are formed by concatenating target keypoint positions, query positions, and initialized transformations, then projected via an MLP. Both token types are concatenated and fed into Transformer blocks; a final MLP decodes the transformation parameters for each keypoint, and LBS produces the final mesh.
    - **Design Motivation**: Correspondence-based initialization alone is insufficient; the specific geometric constraints of the target character must be considered to refine the transformations.

5. **Text-Guided Pseudo Ground-Truth Correspondences**
    - **Function**: Provides training supervision for correspondence learning.
    - **Mechanism**: CLIP encodes the semantic names of keypoints (e.g., "left_arm", "right_wing") to compute a cosine similarity matrix $\mathbf{S}_{\cos}$. The Hungarian algorithm yields a one-to-one hard matching $\mathbf{M}_{\text{hung}}$, while Sinkhorn yields a many-to-many soft matching $\mathbf{M}_{\text{sink}}$.
    - **Design Motivation**: Manual annotation of correspondences is prohibitively expensive. The semantic bone names assigned by artists serve as a natural cross-category bridge.

### Loss & Training

**Stage I**: Trains the correspondence Transformer $\mathcal{G}$ with joint optimization of the Frobenius loss $\mathcal{L}_{\text{forb}} = \|\mathbf{S} - \mathbf{S}_{\cos}\|_2^2 + \|\mathbf{M} - \mathbf{M}_{\text{sink}}\|_2^2 + \|\mathbf{M} - \mathbf{M}_{\text{hung}}\|_2^2$.

**Stage II**: Freezes $\mathcal{G}$ and trains $\mathcal{H}$ with cycle-consistency. The reconstruction loss is $\mathcal{L}_{\text{rec}} = \|\hat{\mathbf{V}}^{\text{src}} - \mathbf{V}^{\text{src}}\|_2^2$; a pose prior regularization $\mathcal{L}_{\text{reg}}$ constrains rotation plausibility using a pretrained matrix-Fisher distribution model; a feature consistency loss $\mathcal{L}_{\text{feat}}$ enforces high-level geometric feature consistency of the reconstructed mesh.

At inference, ARAP optimization is additionally applied to enhance mesh smoothness.

## Key Experimental Results

### Main Results

| Setting | Method | PMD↓ (×100) | ELS↑ |
|---------|--------|-------------|------|
| H2H (Human to Human) | NPT | 6.334 | 0.842 |
| H2H | CGT | 5.687 | 0.887 |
| H2H | SFPT | 3.616 | 0.888 |
| H2H | TapMo | 5.078 | 0.877 |
| H2H | **MimiCAT** | **3.570** | **0.923** |
| CCT (Cross-Category) | NPT | 9.889 | 0.260 |
| CCT | CGT | 6.314 | 0.744 |
| CCT | SFPT | 4.312 | 0.913 |
| CCT | TapMo | 4.883 | 0.922 |
| CCT | **MimiCAT** | **4.264** | **0.927** |

### Ablation Study

| Configuration | PMD↓ (H2H) | PMD↓ (CCT) | Notes |
|---------------|-----------|-----------|-------|
| Full MimiCAT | 3.570 | 4.264 | Complete model |
| A1: w/o rotation averaging (Eq. 4) | 4.439 | 4.524 | Naive equal-weight averaging causes directional ambiguity |
| A2: w/o pose prior (Eq. 8) | 4.161 | 4.655 | Absence of prior regularization leads to unnatural deformations |
| A3: w/o text supervision (Eq. 5) | 4.268 | 4.612 | Replaced by hierarchical correspondence; inaccurate mappings |

### Key Findings
- Rotation initialization (Eq. 4) is critical for cross-category transfer; removing it increases CCT PMD from 4.264 to 4.524.
- Pose prior regularization (Eq. 8) prevents joint twisting and self-intersection; its removal yields the largest CCT PMD increase (4.655).
- Text-guided semantic correspondences outperform heuristic hierarchical correspondence algorithms, which are prone to erroneous matchings (e.g., mapping a dog's hind legs to human arms).
- The model can be zero-shot integrated into existing text-to-motion generation systems (e.g., MLD, T2M-GPT) to animate arbitrary characters.

## Highlights & Insights
- **Semantic Label-Driven Soft Correspondence**: The method cleverly leverages the textual semantic names of bones combined with CLIP encoding to establish cross-category correspondences, requiring no manual annotation and supporting many-to-many matching. This paradigm is transferable to other tasks requiring cross-domain correspondence.
- **Frobenius Rotation Averaging**: This technique resolves the mathematical ill-posedness of quaternion weighted averaging and is broadly applicable to any 3D task involving rotation aggregation.
- **Million-Scale Diverse Dataset**: PokeAnimDB covers 4.4 million frames of animation across 975 character types, making it the largest multi-category 3D character motion dataset to date.

## Limitations & Future Work
- The method depends on the quality of a pretrained skeleton prediction model (RigNet), which may produce inaccurate skeletons for highly non-standard characters.
- The "plausibility" of cross-category transfer lacks a precise definition; the evaluation metric (cycle-consistency) is a proxy measure.
- ARAP optimization is still required at inference to ensure mesh quality, incurring additional computational overhead.
- The copyright and licensing issues of the dataset sources are not sufficiently discussed.

## Related Work & Insights
- **vs. SFPT**: SFPT uses a fixed number of handle points for one-to-one mapping and cannot handle cross-category scenarios with differing keypoint counts. MimiCAT's soft correspondences naturally support variable-length keypoints.
- **vs. TapMo**: TapMo also adopts a handle-based approach and is constrained by the one-to-one correspondence assumption. MimiCAT significantly outperforms both in the cross-category setting.
- **vs. NPT/CGT**: These methods are designed for similar topologies and degrade severely in cross-category scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic treatment of category-free 3D pose transfer; the soft correspondence + cascade Transformer design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluations cover both same-category and cross-category settings with complete ablations and downstream application demonstrations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, detailed method descriptions, and rich illustrations.
- **Value**: ⭐⭐⭐⭐ The new dataset and method offer significant contributions to the fields of 3D animation and character transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] Global-Aware Edge Prioritization for Pose Graph Initialization](global-aware_edge_prioritization_for_pose_graph_initialization.md)
- [\[CVPR 2026\] FreeScale: Scaling 3D Scenes via Certainty-Aware Free-View Generation](freescale_scaling_3d_scenes.md)
- [\[CVPR 2026\] MARCO: Navigating the Unseen Space of Semantic Correspondence](marco_semantic_correspondence.md)

</div>

<!-- RELATED:END -->
