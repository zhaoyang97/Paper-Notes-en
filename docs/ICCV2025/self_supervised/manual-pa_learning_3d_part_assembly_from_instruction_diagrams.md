---
title: >-
  [Paper Note] Manual-PA: Learning 3D Part Assembly from Instruction Diagrams
description: >-
  [ICCV 2025][Self-Supervised Learning][3D Part Assembly] This paper proposes Manual-PA, a Transformer-based instruction-guided 3D part assembly framework that infers assembly order by aligning 3D parts with instruction st…
tags:
  - "ICCV 2025"
  - "Self-Supervised Learning"
  - "3D Part Assembly"
  - "Instruction Diagram Assembly"
  - "Transformer"
  - "Contrastive Learning"
  - "Permutation Learning"
date: 2026-05-08
content_hash: a2a4cc9bef141a76
---

# Manual-PA: Learning 3D Part Assembly from Instruction Diagrams

**Conference**: ICCV 2025
**arXiv**: [2411.18011](https://arxiv.org/abs/2411.18011)  
**Code**: None  
**Area**: Self-Supervised
**Keywords**: 3D Part Assembly, Instruction Diagram Assembly, Transformer, Contrastive Learning, Permutation Learning

## TL;DR

This paper proposes Manual-PA, a Transformer-based instruction-guided 3D part assembly framework that infers assembly order by aligning 3D parts with instruction step diagrams via contrastive learning, then uses the learned order as soft guidance through positional encoding for 6DoF pose prediction, significantly outperforming existing methods on PartNet.

## Background & Motivation

**Background**: The 3D part assembly task aims to predict the 6DoF pose of a set of unordered 3D parts and assemble them into a complete object. Existing methods fall into two categories: (1) geometry-based generative methods (e.g., 3DHPA, SPAFormer) that leverage part shape relationships but may produce unstable results; and (2) guidance-based methods (e.g., MEPNet for LEGO) that typically assume parts are provided one per step.

**Limitations of Prior Work**:
   - **Enormous and sparse solution space**: The combinatorial explosion problem — the number of permutations for $N$ parts is $N!$, combined with continuous 6DoF pose parameters per part, leaves very few feasible and stable assembly sequences.
   - **Guidance-free generative methods**: These do not exploit user-accessible auxiliary information (e.g., instruction manuals) and suffer sharp performance degradation as the number of parts increases.
   - **Limitations of LEGO-style methods**: They assume parts are provided step-by-step and rely on standardized "stud" interfaces, making them inapplicable to general assembly scenarios such as furniture — where instruction manuals do not explicitly indicate which part to use at each step.
   - **Error accumulation**: Directly assembling parts in an autoregressive manner according to predicted order tends to propagate errors from earlier steps to later ones.

**Core Problem**: Humans rely on step-by-step diagrams in instruction manuals when assembling furniture — leveraging such visual information can reduce the search space. The key challenges are: (1) how to align 2D line-drawing diagrams with 3D parts to determine assembly order; and (2) how to use order information as "soft guidance" rather than a "hard constraint" to facilitate assembly.

**Core Idea**: Contrastive learning aligns 3D parts with step diagrams to derive assembly order; permutation-aware positional encoding then guides a Transformer to predict poses.

## Method

### Overall Architecture

**Input**: $N$ 3D part point clouds $\{\mathcal{P}_i\}_{i=1}^N$ and an $N$-step instruction image sequence $(\mathcal{I}_1, ..., \mathcal{I}_N)$.

The pipeline consists of three stages:
1. **Feature Extraction**: PointNet encodes 3D parts → $\mathbf{f}^P \in \mathbb{R}^{N \times D}$; DINOv2 encodes adjacent-step difference images → $\mathbf{f}^I \in \mathbb{R}^{N \times K \times D}$
2. **Permutation Learning**: A similarity matrix is computed → Hungarian matching yields permutation matrix $\mathbf{P}$
3. **Pose Prediction**: Permutation order is used to set positional encodings → a Transformer decoder predicts rotation and translation for each part

### Key Designs

1. **Difference Image Feature Extraction**:

    - **Function**: Extracts information about "which part is newly added at each step" from instruction step images.
    - **Mechanism**: Adjacent step images are differenced as $|\mathcal{I}_j - \mathcal{I}_{j+1}|$ to obtain the newly added part region; the difference image is patchified and fed into a DINOv2 encoder, then projected to a unified dimension $D$ via a linear layer.
    - **Design Motivation**: The incremental nature of instruction manuals implies that step differences directly correspond to newly introduced part information.

2. **Contrastive Learning-Driven Permutation Learning**:

    - **Function**: Learns the correspondence between 3D parts and instruction manual steps.
    - **Mechanism**:
        - A similarity matrix is constructed as $\mathbf{S}_{ij} = \text{sim}(\mathbf{f}_i^P, \mathbf{g}_j^I)$, where $\mathbf{g}^I$ denotes step features obtained by max-pooling over the patch dimension.
        - Hungarian matching is applied on $\mathbf{C} = -\mathbf{S}$ to solve the optimal bipartite matching and obtain permutation matrix $\mathbf{P}$.
        - Training uses an InfoNCE contrastive loss: $\mathcal{L}_{\text{order}} = -\frac{1}{B}\sum_i \log\frac{\exp(\text{sim}(\mathbf{f}^P_{\sigma(i)}, \mathbf{g}^I_i)/\tau)}{\sum_j \exp(\text{sim}(\mathbf{f}^P_{\sigma(i)}, \mathbf{g}^I_j)/\tau)}$
    - **Design Motivation**: Contrastive learning is naturally suited for cross-modal alignment, while Hungarian matching enforces the one-to-one permutation constraint.

3. **Permutation-Aware Positional Encoding-Guided Pose Prediction**:

    - **Function**: Injects the learned assembly order as soft guidance into the pose prediction process.
    - **Mechanism**:
        - Sinusoidal positional encodings $\Phi \in \mathbb{R}^{N \times D}$ represent step order.
        - Step images directly use $\mathbf{p}^I = \Phi$; part positional encodings are reordered via the permutation matrix as $\mathbf{p}^P = \mathbf{P}^T \Phi$.
        - Ground-truth order is used during training; predicted $\hat{\mathbf{P}}$ is used during inference.
        - After adding positional encodings to features, the result is fed into an $L$-layer Transformer decoder: self-attention (inter-part interaction) → cross-attention (information injection from step images to parts).
        - The pose prediction head outputs a quaternion rotation $\hat{q}_i$ and 3D translation $\hat{t}_i$ per part. RoPE replaces standard sinusoidal encoding for improved performance.
    - **Design Motivation**: Positional encoding acts as "soft guidance" — through attention scores it naturally encourages each part to attend more to its corresponding step image, without imposing hard constraints, thereby avoiding error accumulation.

4. **Geometrically Equivalent Group Handling**:

    - **Function**: Handles geometrically identical parts (e.g., four table legs).
    - **Mechanism**: Equivalent groups are identified by AABB dimensions; within each group, Hungarian matching with Chamfer distance as the cost function determines the optimal correspondence before loss computation.
    - **Design Motivation**: Prevents arbitrary labeling of symmetric parts from introducing noisy training signals.

### Loss & Training

- **Permutation Learning**: InfoNCE contrastive loss $\mathcal{L}_{\text{order}}$
- **Pose Estimation**: Weighted sum of four terms $\mathcal{L}_{\text{pose}} = \lambda_T \mathcal{L}_T + \lambda_C \mathcal{L}_C + \lambda_E \mathcal{L}_E + \lambda_S \mathcal{L}_S$
    - $\mathcal{L}_T$: $\ell_2$ distance for translation
    - $\mathcal{L}_C$: Chamfer distance for rotation (handling intrinsic symmetry)
    - $\mathcal{L}_E$: $\ell_2$ distance for rotation (regularization for non-perfectly symmetric parts)
    - $\mathcal{L}_S$: Chamfer distance for the overall assembled shape
- Two-stage training: permutation learning is trained to convergence first, followed by pose estimation training using the predicted order from the permutation model.

## Key Experimental Results

### Main Results

**PartNet Test Set (Level-3, 3 categories)**:

Comparison with existing methods on Chair / Table / Storage categories, evaluated using Shape Chamfer Distance (SCD↓), Part Accuracy (PA↑), Connectivity Accuracy (CA↑), and Success Rate (SR↑):

- Manual-PA achieves the highest Success Rate (SR) on the **Chair** category, significantly outperforming guidance-free methods such as SPAFormer and 3DHPA.
- On the **Table** category, it achieves the lowest Shape Chamfer Distance and highest assembly accuracy.
- Compared to Image-PA (which uses RGB image guidance), Manual-PA using line-drawing diagrams still achieves superior results, indicating that step order information is more critical than visual appearance.

**Zero-Shot Generalization on IKEA-Manual**:

- Zero-shot evaluation is conducted on a real IKEA furniture dataset (trained only on PartNet).
- Strong generalization is demonstrated on both Chair and Table categories.
- This confirms that the method does not rely on dataset-specific distributional properties.

### Ablation Study

| Component | SCD↓ | PA↑ | SR↑ |
|-----------|------|-----|-----|
| No instruction guidance (baseline) | High | Low | Low |
| + Permutation learning | Medium | Medium | Medium |
| + Order-guided positional encoding | Low | High | High |
| + RoPE | **Lowest** | **Highest** | **Highest** |

- The assembly order provided by permutation learning is the key factor driving performance improvement.
- Using order as soft guidance (positional encoding) is more robust than hard constraints (autoregressive decoding).
- Accurate permutation prediction is critical for downstream pose estimation — higher permutation accuracy consistently yields higher assembly success rates.

## Personal Reflections

- **Highlights**: The problem formulation is novel — this is the first work to incorporate assembly instruction manuals into 3D part assembly. The "soft guidance" design is elegant, naturally embedding discrete order information into continuous pose prediction via positional encoding. The combination of contrastive learning for cross-modal alignment and Hungarian matching is concise and effective.
- **Limitations**: The method assumes only one part is added per step and requires pre-rendered difference images. Real-world instruction manuals typically contain complex elements such as text and arrows, whereas the current method only handles line drawings.
- **Insights**: Leveraging structured human knowledge (manuals/diagrams) to constrain combinatorial optimization problems is a promising direction, with potential extensions to robotic manipulation, architectural construction, and related domains.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction](wir3d_visually-informed_and_geometry-aware_3d_shape_abstraction.md)
- [\[CVPR 2026\] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](../../CVPR2026/self_supervised/las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](../../CVPR2026/self_supervised/talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](scaling_languagefree_visual_representation_learning.md)
- [\[ICCV 2025\] MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning](mosic_optimal-transport_motion_trajectory_for_dense_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
