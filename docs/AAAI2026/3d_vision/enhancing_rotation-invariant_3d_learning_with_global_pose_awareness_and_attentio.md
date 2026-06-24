---
title: >-
  [Paper Note] Enhancing Rotation-Invariant 3D Learning with Global Pose Awareness and Attention Mechanisms
description: >-
  [AAAI 2026][3D Vision][Rotation Invariance] This paper proposes the Shadow-informed Pose Feature (SiPF) and the RIAttnConv operator, which enhance the global pose awareness of local rotation-invariant features by introducing global "shadow" reference points learned via Bingham distribution. This addresses the "Wing-tip Feature Collapse" issue where symmetric structures (such as left and right wings of an airplane) cannot be distinguished, achieving SOTA results on ModelNet40…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Rotation Invariance"
  - "Point Cloud Classification"
  - "Part Segmentation"
  - "Attention Mechanism"
  - "Global Pose Awareness"
date: 2026-05-08
content_hash: f9c4759639d0e8d7
---

# Enhancing Rotation-Invariant 3D Learning with Global Pose Awareness and Attention Mechanisms

**Conference**: AAAI 2026  
**arXiv**: [2511.08833](https://arxiv.org/abs/2511.08833)  
**Code**: [GitHub](https://github.com/jiaxunguo/EnRI-GAM)  
**Area**: 3D Vision  
**Keywords**: Rotation Invariance, Point Cloud Classification, Part Segmentation, Attention Mechanism, Global Pose Awareness

## TL;DR
This paper proposes the Shadow-informed Pose Feature (SiPF) and the RIAttnConv operator, which enhance the global pose awareness of local rotation-invariant features by introducing global "shadow" reference points learned via Bingham distribution. This addresses the "Wing-tip Feature Collapse" issue where symmetric structures (such as left and right wings of an airplane) cannot be distinguished, achieving SOTA results on ModelNet40 classification and ShapeNetPart segmentation.

## Background & Motivation

**Background**: The mainstream approach in rotation-invariant (RI) 3D point cloud learning is to replace original coordinates with hand-crafted local geometric features (e.g., PPF, RI tensors) to guarantee feature invariance under arbitrary rotations. Representative methods include PaRI-Conv, RISurConv, and PaRot.

**Limitations of Prior Work**: These methods ensure rotation invariance by discarding absolute coordinate information, which inevitably discards the global pose context. Consequently, components with similar geometric structures but different spatial arrangements (such as the left and right wings of an airplane) generate identical feature representations.

**Key Challenge**: The authors formalize the "Wing-tip Feature Collapse" phenomenon: for symmetric points $p_{\text{left}}$ and $p_{\text{right}}$, their local neighborhoods satisfy $\Omega(p_{\text{right}}) = \Omega(p_{\text{left}}) R_{\text{sym}}$, yielding $f(p_{\text{left}}) = f(p_{\text{right}})$ under any RI functions. This is a fundamental limitation of local receptive fields.

**Goal**: To inject global pose information while maintaining rotation invariance, enabling the model to distinguish structures that are geometrically similar but spatially distinct.

**Key Insight**: Introduce a "shadow" reference point for each point by projecting it to a new position using a learned shared rotation matrix, utilizing this globally consistent anchor to encode relative position information.

**Core Idea**: Generate "shadow points" using global rotation learned via a Bingham distribution, encode them into local PPF features to construct SiPF, and combine them with an attention-based convolutional operator to achieve global-pose-aware rotation-invariant learning.

## Method

### Overall Architecture
The input is a 3D point cloud, and the output is a classification label or point-wise part segmentation. The pipeline consists of three core components:
1. **Task-adaptive Shadow Locating**: Learns a global rotation $R_g$ to generate shadow points.
2. **SiPF Feature Extraction**: Constructs an 8D descriptor incorporating both local geometry and global pose information.
3. **RIAttnConv**: An attention-based rotation-invariant convolutional operator that uses SiPF to guide feature aggregation.

### Key Designs

1. **Shadow-informed Pose Feature (SiPF)**:

    - **Function**: Encodes global pose information into local rotation-invariant features.
    - **Mechanism**: For a reference point $p_r$, a shadow point is generated through a shared rotation $R_g$ as $p_r' = p_r R_g$. Based on standard PPF (4D: distance + 3 angles), SiPPF is additionally computed as the difference in PPF of the reference point and neighboring points relative to the shadow point: $\text{SiPPF}(p_r, p_r', p_j) = \frac{\text{PPF}(p_r, p_r') - \text{PPF}(p_j, p_r')}{\|\text{PPF}(p_r, p_r') - \text{PPF}(p_j, p_r')\|_2}$
    - The final SiPF is an 8D vector: $\mathcal{P}_r^j = (\text{PPF}(p_r, p_j), \text{SiPPF}(p_r, p_r', p_j))$
    - **Design Motivation**: Standard PPF yields identical values for neighboring points symmetrically distributed along the LRF's main axis, losing positional information. Shadow points provide a globally consistent reference direction, thus breaking this symmetry.

2. **Task-adaptive Shadow Locating**:

    - **Function**: Adaptively learns the optimal global rotation $R_g$ to generate shadow points.
    - **Mechanism**: Modeling the rotation uncertainty on the unit quaternion sphere $S^3$ using the Bingham distribution: $\mathcal{B}(q | \mathbf{V}, \mathbf{\Lambda}) = \frac{1}{F(\mathbf{\Lambda})} \exp(q^\top \mathbf{V} \mathbf{\Lambda} \mathbf{V}^\top q)$
    - Extracts the mode vector from $\mathbf{V}$ as the optimal rotation candidate for the current epoch.
    - Joint loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \delta \cdot \sqrt{(\mathcal{L}_{\text{bingham}} - 0.1 \cdot \mathcal{L}_{\text{task}})^2}$
    - **Design Motivation**: An arbitrarily selected $R_g$ might fail under certain geometric configurations (e.g., degenerating into standard PPF when the shadow point aligns with the LRF main axis). By combining end-to-end learning with Bingham distribution-based uncertainty modeling, the optimal rotation that avoids degeneration is automatically discovered.

3. **RIAttnConv Operator**:

    - **Function**: Aggregates neighbor features based on an attention mechanism, using SiPF to guide the weights.
    - **Mechanism**: An MLP maps the SiPF $\mathcal{P}_r^j$ to adaptive kernel weights $W_j^r$, followed by scaled dot-product attention: $Q = \mathbf{W}_r, K = \mathbf{X}_r, V = \mathbf{W}_r \cdot \mathbf{X}_r$
    - Integrated with Reversed EdgeConv: first aggregates neighbor features to obtain $\hat{x}_r$, which is then fused with the reference point features $x_r$: $x_r' = g((\hat{x}_r - x_r) \oplus x_r)$
    - **Design Motivation**: In traditional methods, kernel weights depend solely on local relative poses, making them identical when the local geometry is the same. The global information introduced by SiPF differentiates kernel weights at different global positions, thereby distinguishing symmetric structures.

### Loss & Training
The classification task uses cross-entropy loss along with Bingham regularization. It employs an SGD optimizer with an initial learning rate of 0.1, cosine annealed to 0.001 over 300 training epochs. The batch size is 32 for classification and 16 for segmentation, with a dropout of 0.5.

## Key Experimental Results

### Main Results

**ModelNet40 Shape Classification (%)**:

| Method | Input | z/z | z/SO(3) | SO(3)/SO(3) |
|------|------|-----|---------|-------------|
| DGCNN | pc | 92.2 | 20.6 | 81.1 |
| PaRI-Conv | pc+n | - | - | 83.3 |
| PaRot | pc | 90.9 | 91.0 | 90.8 |
| **Ours** | pc | **91.8** | **91.8** | **91.8** |
| **Ours** | pc+n | **92.6** | **92.6** | **92.6** |

**ShapeNetPart Part Segmentation (z/SO(3))**:

| Method | C. mIoU | I. mIoU |
|------|---------|---------|
| PaRI-Conv (pc+n) | - | 84.6 |
| LocoTrans (pc) | 80.1 | 84.0 |
| **Ours** (pc) | **81.7** | **84.4** |
| **Ours** (pc+n) | **82.9** | **85.0** |

### Ablation Study

| RI Representation | Dimension | C. mIoU | I. mIoU |
|---------|------|---------|---------|
| PPF | 4 | 81.1 | 84.1 |
| Aug. PPF | 8 | 81.8 | 84.2 |
| SiPF-w/o Direction | 5 | 82.4 | 84.5 |
| **SiPF** | 8 | **82.9** | **85.0** |

### Key Findings
- Achieves 91.8% classification accuracy using coordinates only (without normals), outperforming several methods that require normal inputs.
- SiPF achieves a 1.8% C. mIoU improvement over standard PPF on the segmentation task.
- RIAttnConv delivers superior segmentation performance while maintaining comparable parameter counts (3.01M) and FLOPs (4795M) to PaRI-Conv.
- Remains optimal on the real-world ScanObjectNN dataset (84.0% z/SO(3)), demonstrating robustness against noise and occlusions.

## Highlights & Insights
- **Slick formal analysis of "Wing-tip Feature Collapse"**: Rigorously analyzes the fundamental limitations of RI methods with finite receptive fields via patch-swapping transformation, providing a theoretical foundation for incorporating global information.
- **Intuitive "shadow point" concept**: Projecting each point to a new position using a learned global rotation as a reference anchor preserves rotation invariance (as the rotation is shared) while successfully injecting global positional information.
- **Modeling rotation uncertainty with Bingham distribution**: Instead of fixing a rotation matrix, a probability distribution adaptively learns the optimal rotation, preventing potential degeneracy.

## Limitations & Future Work
- Validated only on object-level datasets; evaluating scene-level point clouds (e.g., S3DIS, ScanNet) is a crucial future direction.
- Learning the Bingham distribution may be unstable during early training phases, and the paper does not elaborate on convergence behavior.
- The effectiveness of shadow points relies on globally asymmetric geometric structures, which may fail for perfectly symmetric objects (e.g., spheres).
- The large neighborhood of k=40 in the segmentation task may incur high computational costs on large-scale point clouds.

## Related Work & Insights
- **vs PaRI-Conv**: PaRI-Conv enhances local descriptors using 8D Aug. PPF but remains limited to local information. SiPF naturally extends the PPF series by introducing global information via shadow points.
- **vs VN-DGCNN**: VN-DGCNN preserves pose information via equivariant networks, but is constrained by linear combinations. The proposed SiPF method is more flexible.
- **vs LocoTrans**: LocoTrans utilizes an equivariant backbone to enhance local features, demanding heavy computations (6.72M params, 7998M FLOPs); our method is significantly more efficient.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizing wing-tip collapse + shadow points + Bingham distribution is elegant and theoretically sound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three benchmarks with extensive ablation studies, though lacking scene-level assessments.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, clear diagrams, and smooth logical flow.
- Value: ⭐⭐⭐⭐ Provides an elegant solution for injecting global information into RI point cloud learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation](../../ECCV2024/3d_vision/risurconv_rotation_invariant_surface_attention-augmented_convolutions_for_3d_poi.md)
- [\[AAAI 2026\] Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning](hierarchical_direction_perception_via_atomic_dot-product_operators_for_rotation-.md)
- [\[CVPR 2026\] 4D Local Modeling Toward Dynamic Global Perception for Ambiguity-free Rotation-Invariant Point Cloud Analysis](../../CVPR2026/3d_vision/4d_local_modeling_toward_dynamic_global_perception_for_ambiguity-free_rotation-i.md)
- [\[CVPR 2026\] RINO: Rotation-Invariant Non-Rigid Correspondences](../../CVPR2026/3d_vision/rino_rotation-invariant_non-rigid_correspondences.md)
- [\[CVPR 2026\] AVGGT: Rethinking Global Attention for Accelerating VGGT](../../CVPR2026/3d_vision/avggt_rethinking_global_attention_for_accelerating_vggt.md)

</div>

<!-- RELATED:END -->
