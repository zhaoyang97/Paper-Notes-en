---
title: >-
  [Paper Note] ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis
description: >-
  [CVPR 2026][3D Vision][Point cloud analysis] This paper proposes ECKConv, which defines convolutional kernels on the double coset space $\text{SO(2)}\backslash\text{SE(3)}/\text{SO(2)}$ within the intertwiner framework a…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Point cloud analysis"
  - "SE(3) equivariance"
  - "group convolution"
  - "double coset space"
  - "coordinate network"
  - "intertwiner framework"
date: 2026-05-08
content_hash: 3f24b572b9d791eb
---

# ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis

**Conference**: CVPR 2026
**arXiv**: [2603.17538](https://arxiv.org/abs/2603.17538)  
**Code**: N/A  
**Area**: 3D Vision
**Keywords**: Point cloud analysis, SE(3) equivariance, group convolution, double coset space, coordinate network, intertwiner framework

## TL;DR

This paper proposes ECKConv, which defines convolutional kernels on the double coset space $\text{SO(2)}\backslash\text{SE(3)}/\text{SO(2)}$ within the intertwiner framework and explicitly parameterizes kernel functions via coordinate networks. This is the first approach to simultaneously achieve continuous SE(3) equivariance and large-scale scalability, validated comprehensively across four tasks: classification, registration, and segmentation.

## Background & Motivation

In 3D point cloud deep learning, model symmetry with respect to rigid-body transformations (rotation + translation) is critical for efficient learning. Group convolution is a representative method for extracting equivariant features, yet existing implementations face a fundamental trade-off between **strict symmetry** and **scalability**:

**Background**: Group convolution methods fall into two paradigms — discrete group methods (EPN, E2PN) expand kernel parameters directly over discretized rotation groups, while steerable convolution methods (TFN, SE(3)-Transformer) guarantee continuous equivariance via irreducible decomposition.

**Limitations of Prior Work**:
   - Discrete group methods discretize the continuous rotation group (e.g., approximating SO(3) with a finite set of rotations), introducing a discrepancy between model symmetry and the continuity of the group, thereby failing to strictly guarantee continuous SE(3) equivariance.
   - Steerable convolution methods, while theoretically elegant, require decomposing features and kernels into irreducible representations, incurring prohibitive computational cost (TFN achieves only 62.28% classification accuracy) and preventing scalability to large-scale 3D scenes.

**Key Challenge**: Strict continuous equivariance vs. memory/computational scalability — an inherent tension.

**Prev. Attempts**: The intertwiner framework (Cohen et al.) proposes replacing the domain of group convolution from the group space to a quotient space, which is theoretically more principled. However, the prior work CSEConv only achieves SO(3) symmetry (lacking translation equivariance) and uses implicit kernels that incur large memory consumption, preventing scalability.

**Key Insight**: The authors observe that when the reference point lies on the SO(2) subgroup (Z-axis), topologically equivalent point distributions reside on disjoint orbits (double cosets), each uniquely characterized by three parameters. This implies that equivariant operations can be constructed by relying solely on these SE(3)-invariant parameters.

**Core Idea**: Explicitly parameterize convolutional kernels on the double coset space via coordinate networks, simultaneously achieving continuous SE(3) equivariance and memory scalability.

## Method

### Overall Architecture

ECKConv takes as input a point cloud $(x, n) \in \mathbb{R}^3 \times S^2$ (coordinates + normals) and produces equivariant features through multiple ECKConv blocks. The core operation within each block is: for each center point's ball-query neighborhood, extract SE(3)-invariant double coset parameters, compute kernel weights via a coordinate network, and aggregate weighted neighborhood features to produce the output. The overall architecture adopts a multi-resolution design similar to PointNet++ (FPS downsampling + ball query); segmentation tasks employ a U-Net architecture with feature interpolation for upsampling.

### Key Designs

1. **Continuous SE(3) Equivariant Convolution (Double Coset Space Construction)**:

    - Function: Maps the kernel function domain from group space to the double coset space $\text{SO(2)}\backslash\text{SE(3)}/\text{SO(2)}$.
    - Mechanism: Let $G=\text{SE(3)}$, $H_1=H_2=\text{SO(2)}$, with invariant representations $\rho_1=\rho_2=\text{id}$. Each element of the double coset space is uniquely characterized by three parameters $[\beta_g, r_g, z_g]$, and the kernel function $\kappa$ depends only on these parameters: $(f*\kappa)(x) = \sum_{x_i \in \mathcal{N}(x)} \kappa(s(x)^{-1}x_i) f(x_i)$, where $\kappa: \text{SE(3)/SO(2)} \to \mathbb{R}^{C_{out} \times C_{in}}$.
    - Design Motivation: The double coset parameters are inherently invariant to SE(3) transformations — Z-axis rotations are absorbed by the left and right SO(2) subgroup actions. Consequently, the kernel needs only three scalar parameters to guarantee continuous SE(3) equivariance, avoiding discretization or irreducible decomposition.

2. **Double Coset Encoding**:

    - Function: Extracts SE(3)-invariant three-dimensional parameters from local neighborhoods.
    - Mechanism: The domain space is factored as $\mathbb{R}^3 \times S^2$. For each point $x_i$ in the ball-query neighborhood, after aligning to the local coordinate frame via the inverse section map, the following are extracted: $\bar{\beta}_i = \arccos(\mathbf{n}^\top \cdot \mathbf{n}_i)$ (angle between normals), $\bar{z}_i$ (displacement component parallel to the normal, normalized by radius), and $\bar{r}_i$ (displacement component perpendicular to the normal, normalized by radius). These three parameters encode the normal relationship and two orthogonal components of spatial displacement.
    - Design Motivation: Normalizing by the ball-query radius confines each ECKConv layer to scale-normalized local geometric information. When normals are unavailable, Algorithm 1 provides a heuristic alternative (K-NN mean difference vectors or PCA estimation).

3. **Explicit Kernel via Coordinate-based Networks**:

    - Function: Computes kernel values as a weighted sum of learnable basis matrices via a coordinate network.
    - Mechanism: Double coset parameters are mapped to $[0,1]$, encoded via Gaussian embeddings $\text{Gau}(\cdot) \in \mathbb{R}^{3d}$ (where $\psi(x,y) = \exp(-(x-y)^2/2\sigma^2)$), and passed through a neural network $F_\theta$ to produce coefficient vectors $\omega(\bar{x};\theta) \in \mathbb{R}^A$. The kernel value is: $\kappa(s(x)^{-1}x_i) = \sum_{j=1}^{A} \omega_j(\bar{x}_i;\theta) \mathbf{W}_j$.
    - Key difference from CSEConv: CSEConv uses an implicit kernel based on random Fourier features (the network directly outputs a $C_{out} \times C_{in}$ matrix), whereas ECKConv decomposes the kernel into a "coefficient network + basis matrices", with $\omega$ outputting only $A$-dimensional scalar coefficients.

4. **Reordering for Scalability (Proposition 4.1)**:

    - Function: Reduces backpropagation memory complexity via computation reordering.
    - Mechanism: By exploiting matrix multiplication associativity, $\sum_i (\sum_j \omega_j \mathbf{W}_j) f(x_i)$ is reordered as $\sum_j \mathbf{W}_j \sum_i \omega_j f(x_i)$. The former has gradient complexity $\mathcal{O}(AKC_{in}C_{out})$; the latter reduces it to $\mathcal{O}(A(KC_{in} + C_{in}C_{out}))$.
    - Practical impact: ECKConv-mini requires only 0.72 GB vs. CSEConv's 2.95 GB; in pose registration, 5.37 GB vs. 39.09 GB.

### Loss & Training

- Classification: Cross-entropy loss (label smoothing = 0.2), Adam, lr 1e-4 with cosine annealing to 1e-6, 200 epochs.
- Registration: $\mathcal{L} = \|\mathbf{R}_{pred}\mathbf{R}_{gt}^\top - \mathbf{I}\|_2^2 + \|\mathbf{t}_{pred} - \mathbf{t}_{gt}\|_2^2$, combined with DCP, training poses sampled uniformly from SE(3).
- Segmentation: U-Net + feature interpolation upsampling (ShapeNet K=1, S3DIS K=3 nearest-neighbor interpolation).
- S3DIS: Training on $4\text{m}^2$ crops of 4096 points, with XY scaling and X-axis flip augmentation; class-weighted cross-entropy; inference with 1-meter stride and 3-vote aggregation. All experiments conducted on a single RTX 3090.

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | ECKConv | Prev. SOTA | Gain |
|---|---|---|---|---|
| ModelNet40 (SO(3)/SO(3)) | Classification Acc. | 90.92% (Normal) | 88.58% (E2PN) | +2.34% |
| ModelNet40 Pose Registration | Mean Angular Error | 0.63° | 1.62° (EPN) | 2.6× |
| ModelNet40 Pose Registration | Max Angular Error | 8.57° | 178.95° (CSEConv) | Qualitative leap |
| ShapeNet (SO(3)/SO(3)) | Part Segmentation mIoU | 83.68% | 81.76% (VN) | +1.92% |
| S3DIS Area5 | Semantic Segmentation mIoU | 61.80% | 60.3% (RI-MAE) | +1.50% |

### Ablation Study

| Configuration | Accuracy / Memory | Notes |
|---|---|---|
| ECKConv-mini (explicit kernel) | 87.36% / 0.72 GB | Same-architecture CSEConv: 83.75% / 2.95 GB; +3.6% accuracy, 4.1× memory reduction |
| ECKConv (Algorithm 1) | 90.19% | Without normals; heuristic replacement |
| ECKConv-Normal | 90.92% | With ground-truth normals; best performance |
| A = 12 → 22 (anchor bases) | 89.22 → 90.19% | Performance robust to hyperparameter variation |
| Registration: ECKConv vs. CSEConv | 5.37 GB vs. 39.09 GB | 7.3× memory reduction |
| Segmentation: ECKConv vs. CSEConv | 10.26 GB vs. 66.46 GB | 6.5× memory reduction |

### Key Findings

- ECKConv maintains consistent accuracy across all training/testing rotation combinations, validating continuous SE(3) equivariance.
- Pose registration most clearly demonstrates the value of continuous equivariance: CSEConv (SO(3)-only symmetry) achieves a maximum angular error of 178.95°, indicating that the absence of translation equivariance leads to catastrophic failure.
- On large-scale S3DIS semantic segmentation, ECKConv surpasses both data augmentation methods (KPConv + SO(3) Aug: 57.42%) and rotation-invariant methods (RI-MAE: 60.3%), demonstrating the practical value of SE(3) equivariance in large-scale scenes.
- Hyperparameter ablations confirm that ECKConv is robust to variations in anchor count $A$ and embedding dimension $\Psi$, and that residual connections play an important role in maintaining performance.

## Highlights & Insights

- **Unification of theoretical elegance and practicality**: Framing the SE(3) equivariance problem as kernel function design on the double coset space provides theoretically rigorous continuous symmetry guarantees while achieving engineering scalability via explicit kernels.
- **Explicit kernel reordering**: By simply changing the summation order (Proposition 4.1), backpropagation complexity is reduced from $O(AKC_{in}C_{out})$ to $O(A(KC_{in}+C_{in}C_{out}))$ without altering the forward computation result — a zero-cost scalability gain with remarkable practical impact (7× memory reduction).
- **Appropriate choice of Gaussian embeddings**: Mapping double coset parameters to a Gaussian kernel space leverages its bounded-rank property to balance memorization and generalization, and is transferable to other settings involving continuous geometric parameters.
- **Pose registration experiment**: This experiment vividly illustrates the fundamental distinction between continuous and discrete equivariance — CSEConv's maximum angular error of 178.95° stems from SO(3)-only symmetry failing to handle translations, making this a highly compelling experimental design.

## Limitations & Future Work

- **Isotropic constraint**: Due to the use of scalar-type features and double coset elements, the kernel function is isotropic with respect to SE(3) actions, making it unsuitable for tasks requiring directional expressiveness (normal estimation, molecular structure prediction, n-body problems).
- **Normal vector dependency**: ECKConv-Normal requires ground-truth normals for peak performance; the Algorithm 1 replacement incurs an ~0.7% performance drop.
- **Gap with non-equivariant SOTA**: PTv3 (92.54%) still leads by ~1.6% on classification, suggesting that equivariance constraints may limit model capacity.
- **Color information unused**: S3DIS experiments use only coordinates and normals; incorporating color features may yield further improvements.
- **Attention mechanisms unexplored**: Integration with Transformer architectures has not been investigated, which may limit global context modeling.

## Related Work & Insights

- **vs. CSEConv**: The most direct predecessor. ECKConv extends to continuous SE(3) (CSEConv covers only SO(3)) and achieves scalability through explicit kernels (4–7× memory reduction), with substantially improved performance.
- **vs. E2PN**: Achieves scalability via discrete SE(3) and attention pooling, but discretization results in non-strict rotation equivariance (88.58% vs. 90.92% under SO(3)/SO(3) evaluation).
- **vs. Vector Neurons**: A model-agnostic framework for SO(3) equivariance; VN (81.76%) underperforms ECKConv (83.68%) on segmentation, indicating that local SE(3) equivariance is superior to global SO(3) equivariance.
- **vs. TFN / SE(3)-Transformer**: Representative steerable convolution methods; theoretically equivariant but computationally prohibitive (62–73% classification accuracy), illustrating the practical limitations of a purely theoretical approach.
- The double coset space perspective in ECKConv may inspire approaches to geometric symmetry problems in molecular symmetry, protein structure analysis, and related domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to achieve continuous SE(3)-equivariant and scalable point cloud convolution within the intertwiner framework; solid theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers four task types (classification, registration, part segmentation, semantic segmentation) with thorough ablations and scalability comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear and rigorous; figures are intuitive; supplementary material is comprehensive.
- **Value**: ⭐⭐⭐⭐ Resolves a fundamental trade-off in the group convolution literature; represents a significant advance for equivariant network research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics](efficient_hybrid_se3-equivariant_visuomotor_flow_policy_via_spherical_harmonics_.md)
- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [\[AAAI 2026\] Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis](../../AAAI2026/3d_vision/graph_smoothing_for_enhanced_local_geometry_learning_in_point_cloud_analysis.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->
