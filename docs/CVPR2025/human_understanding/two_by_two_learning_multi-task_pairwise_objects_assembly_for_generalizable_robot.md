---
title: >-
  [Paper Note] Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation
description: >-
  [CVPR 2025][Human Understanding][3D Assembly] This paper proposes the 2BY2 dataset—the first large-scale dataset for daily pairwise object assembly (18 task classes, 517 object pairs)—and designs a two-step SE(3) pose estimation network that leverages equivariant features to achieve multi-task pairwise object assembly, achieving state-of-the-art (SOTA) performance across all tasks and demonstrating generalization capabilities through real-world robot experiments.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "3D Assembly"
  - "Pairwise Object Assembly"
  - "SE(3) Pose Estimation"
  - "Equivariant Features"
  - "Robotic Manipulation"
date: 2026-05-08
content_hash: 935426c8690144fd
---

# Two by Two: Learning Multi-Task Pairwise Objects Assembly for Generalizable Robot Manipulation

**Conference**: CVPR 2025  
**arXiv**: [2504.06961](https://arxiv.org/abs/2504.06961)  
**Code**: [https://tea-lab.github.io/TwoByTwo/](https://tea-lab.github.io/TwoByTwo/)  
**Area**: Human/Object Understanding  
**Keywords**: 3D Assembly, Pairwise Object Assembly, SE(3) Pose Estimation, Equivariant Features, Robotic Manipulation

## TL;DR

This paper proposes the 2BY2 dataset—the first large-scale dataset for daily pairwise object assembly (18 task classes, 517 object pairs)—and designs a two-step SE(3) pose estimation network that leverages equivariant features to achieve multi-task pairwise object assembly, achieving state-of-the-art (SOTA) performance across all tasks and demonstrating generalization capabilities through real-world robot experiments.

## Background & Motivation

**Background**: 3D assembly tasks (e.g., furniture assembly, part mating) are ubiquitous in daily life and represent a core capability for future household robots. Existing benchmarks and datasets primarily focus on geometric fracture reassembly (e.g., Breaking Bad) or industrial part assembly (e.g., Factory), leaving daily object interaction scenarios underrepresented.

**Limitations of Prior Work**: While existing methods perform well in geometric fracture matching, they struggle in daily assembly scenarios (e.g., inserting flowers into a vase, placing bread into a toaster). Such tasks require not only geometric alignment but also an understanding of the functional relationships and spatial semantics between objects. Furthermore, current datasets are limited in scale and task variety, and lack symmetry annotations, failing to meet the diverse requirements of daily assembly.

**Key Challenge**: Geometric matching methods focus solely on local shape alignment while ignoring semantic constraints and functional relationships in daily assembly (e.g., "a bottle cap must cover the bottle" instead of arbitrary mating), resulting in poor cross-scenario generalization.

**Goal**: (1) Construct the first large-scale pairwise object dataset covering 18 daily assembly tasks; (2) design an SE(3) pose estimation method capable of handling multiple assembly tasks simultaneously with strong generalization ability.

**Key Insight**: The authors observe that humans typically assemble objects in a stepwise manner—first positioning the base/container (e.g., a vase) and then placing the accessory (e.g., a flower). Thus, they propose a network architecture that mimics this two-step assembly logic.

**Core Idea**: Predict the SE(3) poses of the base and accessory sequentially using a two-step network architecture, combining SE(3) equivariant feature extraction and cross-object feature fusion to achieve precise pairwise object assembly.

## Method

### Overall Architecture

The inputs are two point clouds $\mathcal{P}_A$ (accessory, e.g., a flower) and $\mathcal{P}_B$ (base, e.g., a vase), each containing 1024 3D points. Both point clouds are augmented with random SO(3) rotations and centroid translations. The network predicts poses in two steps: Branch B first predicts the canonical pose (rotation and translation) of the base to transform it into the canonical space; then, Branch A predicts the canonical pose of the accessory based on the transformed base in conjunction with the accessory's information. The network ultimately outputs two SE(3) transformations to assemble the objects into a predefined canonical space.

### Key Designs

1. **Two-step Pairwise Network**:

    - **Function**: Mimics human assembly logic by first locating the base and then the accessory, sequentially predicting the 6DoF pose of each object.
    - **Mechanism**: Branch B uses a Two-scale VN DGCNN encoder to extract the SE(3) equivariant feature $\mathcal{E}_B$ of $\mathcal{P}_B$, outputting rotation and translation through an MLP prediction head. The transformed $\mathcal{P}_B$, along with the original $\mathcal{P}_A$, is then fed into Branch A to extract the equivariant feature $\mathcal{E}_A$ of $\mathcal{P}_A$ and the SO(3) invariant feature $\mathcal{I}_B$ of $\mathcal{P}_B$. The accessory pose is predicted following feature fusion.
    - **Design Motivation**: Jointly predicting the poses of both objects can lead to error interference; stepwise prediction isolates error propagation, ensuring the accuracy of the base pose first before predicting the accessory pose based on it.

2. **Two-scale VN DGCNN**:

    - **Function**: Extracts features from point clouds that maintain SE(3) equivariance while capturing multi-scale geometric information.
    - **Mechanism**: Extended from the Vector Neuron DGCNN, it utilizes two KNN branches with different K values for feature extraction to capture local details and global shape information, respectively. These features are then concatenated and fused through an extra VN convolutional layer. T(3) translation equivariance is achieved by subtracting the centroid of the point cloud, i.e., $f(\mathcal{P} + \mathcal{T}) = f(\mathcal{P}) + \mathcal{T}$.
    - **Design Motivation**: A single-scale KNN graph cannot simultaneously capture fine geometric details and global shapes; the pyramid structure acquires both global and local information, enhancing feature representation capability.

3. **Cross Object Fusion**:

    - **Function**: Injects the geometric information of the base into the feature representation of the accessory, enabling Branch A to predict the pose by synthesizing information from both objects.
    - **Mechanism**: Fuses $\mathcal{I}_B$ (SO(3) invariant feature) and $\mathcal{E}_A$ (SE(3) equivariant feature) via element-wise multiplication. This guarantees that the fused features maintain rotational equivariance with respect to $\mathcal{P}_A$, i.e., $f(R \cdot (\mathcal{I}_B * \mathcal{E}_A)) = R \cdot f(\mathcal{I}_B * \mathcal{E}_A)$.
    - **Design Motivation**: Multiplying invariant and equivariant features introduces the base shape information as context without breaking the rotational equivariance of the accessory features.

### Loss & Training

The loss function is a weighted sum of the rotation and translation losses: $\mathcal{L} = \lambda_{rot}\mathcal{L}_{rot} + \lambda_{trans}\mathcal{L}_{trans}$. An L1 loss is used for translation, and Geodesic Distance is used for rotation: $\mathcal{L}_{rot} = \arccos\left(\frac{\text{tr}(\mathcal{R}_{gt}\mathcal{R}_{pred}^T) - 1}{2}\right)$. Geodesic distance measures the shortest path between two rotations on the rotation manifold, providing a smooth, bounded angular error and stable gradients.

Regarding training strategy, Branch A and Branch B are trained independently. During the training of Branch A, the ground truth (i.e., $\mathcal{P}_B$ under the canonical pose) is used, whereas at test time, the predicted results from Branch B are employed for cascaded inference. Separate MLP prediction heads are used for rotation and translation to avoid interference caused by differing convergence speeds.

## Key Experimental Results

### Main Results

| Method | Translation RMSE(T) ↓ | Rotation RMSE(R) ↓ |
|------|--------------|--------------|
| Jigsaw | 0.360 | 53.34 |
| Puzzlefusion++ | 0.342 | 58.23 |
| NSM | 0.284 | 70.30 |
| SE(3)-Assembly | 0.233 | 52.34 |
| **Ours** | **0.110** | **41.44** |

It outperforms the baselines on all 18 tasks, reducing the average translation RMSE by 0.046 and the rotation RMSE by 8.97.

### Ablation Study

| Encoder | Translation RMSE(T) | Rotation RMSE(R) |
|--------|------------|------------|
| PointNet | 0.264 | 75.38 |
| DGCNN | 0.277 | 72.46 |
| VN DGCNN (Single-scale) | 0.123 | 44.67 |
| w/o Two-step | 0.139 | 45.20 |
| **Ours (Two-scale VN DGCNN + Two-step)** | **0.110** | **41.44** |

### Key Findings

- Two-scale VN DGCNN achieves improvements over the single-scale VN DGCNN across all tasks, validating the effectiveness of multi-scale feature extraction.
- The two-step network design shows significant improvements over end-to-end joint prediction (w/o Two-step), confirming that sequential prediction effectively isolates errors.
- In real-world robot experiments, the proposed method achieves an overall success rate of 77.5% (compared to only 22.5% for the baseline SE(3) Assembly) and reaches a 100% success rate on the flower-arrangement task.

## Highlights & Insights

- **Prominent Dataset Contribution**: 2BY2 is the first large-scale pairwise assembly dataset tailored for daily scenarios, covering 18 diverse tasks (from flower arrangement to USB insertion), filling a critical gap in this field.
- **Human-like Assembly Logic**: The two-step architecture ingeniously mimics the natural assembly sequence of humans, which is both intuitive and computationally effective.
- **Meticulous Equivariance Design**: Equivariance and invariance are carefully maintained during feature fusion to ensure that mathematical properties are not compromised, reflecting a deep understanding of symmetry.

## Limitations & Future Work

- The current method assumes clean point cloud inputs; its performance under noisy point clouds or partial occlusions remains unknown.
- Symmetry annotations in the dataset require manual effort, leading to high labor costs when scaling up to new tasks.
- Real-world robot experiments are tested on only 4 tasks; generalization capabilities in more complex settings (e.g., multi-step assembly sequences) need further validation.
- Future work can integrate vision foundation models to automatically acquire semantic information or introduce diffusion-based pose-refinement strategies.

## Related Work & Insights

- Reassembly methods like Breaking Bad and Neural Shape Mating lay the foundation for shape matching, but 2BY2 generalizes the problem to daily functional assembly.
- SE(3) equivariant networks (e.g., VNNs) demonstrate strong advantages in tasks requiring arbitrary pose generalization, and are worth promoting in a wider range of robotic manipulation tasks.
- The two-step prediction paradigm can be extended to more complex, multi-body assembly sequences.

## Rating

- **Novelty**: 8/10 — The combination of the dataset and the two-step equivariant network architecture represents a novel contribution.
- **Experimental Thoroughness**: 8/10 — Comprehensive evaluation across 18 tasks plus real-world robot experiments, though ablation analysis could be more detailed.
- **Writing Quality**: 7/10 — The structure is clear, but descriptions of some details could be more concise.
- **Value**: 8/10 — Both the dataset and the method hold high value for the robotic manipulation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)
- [\[ICCV 2025\] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning](../../ICCV2025/human_understanding/ar-vrm_imitating_human_motions_for_visual_robot_manipulation_with_analogical_rea.md)
- [\[CVPR 2025\] FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](fsfm_a_generalizable_face_security_foundation_model_via_self-supervised_facial_r.md)
- [\[ICLR 2026\] Text2Interact: High-Fidelity and Diverse Text-to-Two-Person Interaction Generation](../../ICLR2026/human_understanding/text2interact_high-fidelity_and_diverse_text-to-two-person_interaction_generatio.md)
- [\[CVPR 2025\] FreeUV: Ground-Truth-Free Realistic Facial UV Texture Recovery via Cross-Assembly](freeuv_ground-truth-free_realistic_facial_uv_texture_recovery_via_cross-assembly.md)

</div>

<!-- RELATED:END -->
