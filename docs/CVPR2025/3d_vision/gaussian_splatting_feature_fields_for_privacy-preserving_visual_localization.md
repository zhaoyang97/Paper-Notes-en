---
title: >-
  [Paper Note] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization
description: >-
  [CVPR 2025][3D Vision][Visual Localization] This paper proposes Gaussian Splatting Feature Fields (GSFFs), which combine the explicit geometry of 3DGS with implicit feature fields. Through self-supervised contrastive learning, they train scale-aware 3D features and 2D encoders, and leverage Delaunay-graph-based spatial clustering to convert features into segmentation labels, achieving high-accuracy non-privacy and privacy-preserving visual localization.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Visual Localization"
  - "Privacy Preserving"
  - "Gaussian Splatting Feature Fields"
  - "Self-Supervised Learning"
  - "Pose Optimization"
date: 2026-05-08
content_hash: 9a1dfb726097b471
---

# Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization

**Conference**: CVPR 2025  
**arXiv**: [2507.23569](https://arxiv.org/abs/2507.23569)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Visual Localization, Privacy Preserving, Gaussian Splatting Feature Fields, Self-Supervised Learning, Pose Optimization

## TL;DR
This paper proposes Gaussian Splatting Feature Fields (GSFFs), which combine the explicit geometry of 3DGS with implicit feature fields. Through self-supervised contrastive learning, they train scale-aware 3D features and 2D encoders, and leverage Delaunay-graph-based spatial clustering to convert features into segmentation labels, achieving high-accuracy non-privacy and privacy-preserving visual localization.

## Background & Motivation

1. **Background**: Visual localization (VL) is the task of estimating the 6DoF camera pose from which an image was taken, which is crucial for autonomous driving and robot navigation. Mainstream methods establish 2D-3D correspondences based on feature matching, offering the highest accuracy but risking privacy leaks—detailed images can be reconstructed from feature descriptors.

2. **Limitations of Prior Work**: (a) Feature matching methods present privacy concerns, as feature descriptors of a scene can be exploited for image reconstruction attacks; (b) SegLoc achieves privacy preservation by quantizing features into segmentation labels, but its segmentation is learned in 2D without multi-view consistency guarantees; (c) NeRF-based feature fields (such as SSL-Nif) ensure multi-view consistency and high accuracy, but do not support privacy preservation and suffer from slow rendering speeds.

3. **Key Challenge**: Privacy preservation demands removing high-dimensional features (replacing them with low-dimensional labels), but traditional segmentation labels are learned in 2D and lack 3D consistency. Meanwhile, 3D-consistent feature field methods rely on high-dimensional features, failing to satisfy privacy requirements.

4. **Goal**: How to simultaneously achieve: (i) a 3D-consistent feature representation, (ii) high-accuracy pose optimization, and (iii) privacy preservation (retaining localization capability even after removing high-dimensional features)?

5. **Key Insight**: Leverage the explicit geometry of 3DGS (finite set of Gaussians) and its fast differentiable rendering characteristics—explicit geometry allows direct spatial clustering on Gaussians to naturally convert features into labels, and differentiable rendering supports both feature-level and segmentation-level pose refinement.

6. **Core Idea**: Use a triplane to encode scale-aware 3D Gaussian features, aligning them with a 2D encoder through contrastive learning; use spectral clustering on a Delaunay graph to generate spatial prototypes, converting features into segmentation labels to achieve privacy-preserving localization.

## Method

### Overall Architecture
The 3D scene representation is constructed based on Gaussian Opacity Fields. During the training stage, the GSFFs feature field (triplane) and the 2D feature encoder are jointly optimized so that the rendered feature map $F^{3D}$ aligns with the encoder-extracted $F^{2D}$ under a contrastive loss. A Gaussian graph is constructed via Delaunay triangulation, and spectral clustering is applied to generate prototypes, further regularized by a prototypical contrastive loss. During localization, given a query image, an initial pose is first obtained via image retrieval, and then the final pose is acquired through feature-metric (or segmentation-metric) pose refinement.

### Key Designs

1. **Triplane-based Scale-Aware 3D Feature Field**:

    - **Function**: Associate each 3D Gaussian with a feature vector that accounts for its spatial scale.
    - **Mechanism**: Three orthogonal 2D feature planes $H_{xy}, H_{xz}, H_{yz} \in \mathbb{R}^{R \times R \times D}$ are placed at the world origin. For each Gaussian $\mathcal{G}_i$, it is projected onto the three planes to obtain three 2D Gaussians $\mathcal{G}_i^{xy}, \mathcal{G}_i^{xz}, \mathcal{G}_i^{yz}$. An RBF kernel (parameterized by the covariance of the 2D Gaussian) is used to sample features on the planes, and the three-way average yields the volumetric feature $\mathbf{g}_i^{3D}$. Large Gaussians aggregate features from larger areas, while small Gaussians aggregate from smaller areas, naturally achieving scale awareness. The features are rendered onto the image plane via alpha-blending.
    - **Design Motivation**: Storing high-dimensional features independently for each Gaussian is too costly. The Triplane shares parameters, and the sampling kernel parameterized by the Gaussian covariance achieves scale awareness, keeping the total parameter count far smaller than storing features per Gaussian.

2. **Delaunay Graph-based Prototypical Feature Regularization**:

    - **Function**: Structure the feature space to provide segmentation labels for privacy-preserving localization.
    - **Mechanism**: Delaunay triangulation is performed on Gaussian centers to construct a sparse graph, and spectral clustering is executed on the Laplacian matrix of this graph to divide Gaussians into $K$ groups. The mean of all Gaussian features within each group serves as the prototype $\mathbf{p}_k$. A prototypical contrastive loss $L_{PRO}$ is introduced to encourage pixel-aligned 2D/3D feature pairs to be close to the same prototype. Cross-view consistency is also introduced—pixel correspondences across different views are established using depth maps and poses, and feature pairs are randomly swapped in the contrastive loss to enhance view invariance. After training, each Gaussian is assigned a hard label $k^*=\text{argmax}_k(\mathbf{l}_{ik})$, and the feature field and color information are discarded, retaining only geometry + labels.
    - **Design Motivation**: Directly applying K-means to features ignores spatial structure. The Delaunay graph naturally encodes the adjacency relationships of Gaussians, and spectral clustering on this graph considers spatial continuity, resulting in more geometrically meaningful labels. The conversion from features to labels is the core of privacy preservation.

3. **Feature-Level / Segmentation-Level Pose Refinement**:

    - **Function**: Iteratively optimize to obtain an accurate pose given a query image and an initial pose.
    - **Mechanism**: **Feature Mode (GSFFs-PR Feature)**: Extract query image 2D features $F^{2D}$, render 3D features $F^{3D}$ from the current pose, and minimize $P^*=\min_{P \in SE(3)} \|F^{2D}-F^{3D}(P,\mathcal{G})\|_2^2$, updating the pose on the se(3) Lie algebra through explicit backpropagation of the rasterizer. **Privacy Mode (GSFFs-PR Privacy)**: Replace features with segmentation labels and minimize $P^*=\min_{P \in SE(3)} CE(S^{2D}, S_P^{3D})$, using a cross-entropy loss to align 2D and rendered 3D segmentation maps. Only geometry and labels are preserved in the scene, discarding all color and feature information.
    - **Design Motivation**: Explicit backpropagation of the rasterizer is more accurate than matching + PnP baselines (no RANSAC outlier issues). Although segmentation-level optimization has slightly lower accuracy, it is completely privacy-safe—only coarse-grained geometry and discrete labels are retained, preventing any reconstruction of image detail.

### Loss & Training
The total loss consists of three components: $L_{NCE}$ (contrastive loss aligning pixel-level 2D/3D features), $L_{PRO}$ (prototypical contrastive loss encouraging features to match spatial prototypes), and $L_{CE}$ (cross-entropy loss optimizing segmentation consistency). Cross-view consistency is achieved by establishing correspondences via depth re-projection and randomly swapping feature pairs. The Sinkhorn-Knopp algorithm is used for optimal transport to solve the feature-to-prototype assignment.

## Key Experimental Results

### Main Results

| Dataset | Method | Position Error (cm) ↓ | Rotation Error (°) ↓ | 5cm/5° Recall (%) ↑ |
|--------|------|-----------|-----------|-----------|
| **7Scenes (avg of 6 scenes, excl. Stairs)** | | | | |
| | HLoc (SBM) | ~1.0 | ~0.18 | ~99% |
| | ACE + GS-CPR | ~0.7 | ~0.25 | - |
| | SSL-Nif (RBM) | ~1.3 | ~0.37 | ~83% |
| | GS-CPR (RBM) | ~0.9 | ~0.29 | - |
| | **GSFFs-PR Feature** | **~0.7** | **~0.29** | **~94%** |
| | **GSFFs-PR Privacy** | ~1.2 | ~0.44 | ~89% |
| **Cambridge Landmarks** | | | | |
| | HLoc | 4-15 | 0.2-0.3 | - |
| | SegLoc (Privacy) | 30-134 | 0.71-2.78 | - |
| | **GSFFs-PR Feature** | **4-8** | **0.12-0.25** | - |
| | **GSFFs-PR Privacy** | **7-26** | **0.12-0.62** | - |

### Ablation Study

| Configuration | Chess Position (cm) ↓ | Chess Rotation (°) ↓ | Description |
|------|--------------|-------------|------|
| Full (GSFFs-PR Feature) | **0.4** | **0.19** | Full model |
| w/o Triplane (per-Gaussian feat) | 0.5 | 0.22 | Triplane sharing is effective |
| w/o Scale-awareness | 0.6 | 0.25 | Scale information is important |
| w/o Cross-view consistency | 0.5 | 0.21 | Multi-view regularization helps |
| w/o Prototypical regularization | 0.5 | 0.20 | Prototypical structuring is effective |
| Pre-trained features (DINOv2) | 0.6 | 0.24 | Self-supervised learned features are superior |

### Key Findings
- **Privacy mode accuracy is close to feature mode**: On 7Scenes, GSFFs Privacy is only about 40% worse in position error than Feature, but far superior to the only comparable privacy method, SegLoc.
- **Self-supervised features outperform pre-trained counterparts**: Features learned by GSFFs themselves are more accurate for localization than DINOv2, because localization requires local discriminative features rather than semantic features.
- **Explicit backpropagation outperforms PnP post-processing**: GSFFs directly optimizes the pose via the rasterizer, avoiding cumulative errors in the matching -> PnP pipeline.
- **The Stairs scene is the only failure case** (25cm error), due to repetitive structures making features indistinguishable.
- On the large-scale Cambridge Landmarks dataset, GSFFs Privacy significantly outperforms SegLoc (King's: 14cm vs 30cm), proving that 3D-consistent segmentation is superior to 2D segmentation.

## Highlights & Insights
- **Natural transition from features to segmentation**: Through spatial clustering to generate prototypes -> soft assignment -> hard assignment, the transition from feature fields to privacy-safe scene representations is highly natural, without requiring additional training of privacy-specific models.
- **Scale-aware Triplane features**: Utilizing the Gaussian covariance as RBF kernel parameters achieves scale-awareness, which is more accurate than simple coordinate lookup. This paradigm of directly integrating Gaussian geometric attributes into feature extraction is worth generalizing.
- **Spectral clustering on Delaunay graphs**: Constructing a spatial graph via Delaunay triangulation on the Gaussian point cloud is much more efficient than clustering on a full density matrix, while successfully retaining local geometric structure.
- **Cross-view consistency training strategy**: Establishing correspondences via depth re-projection and swapping feature pairs in the contrastive loss is a simple but highly effective way to enhance multi-view invariance.

## Limitations & Future Work
- Fails in repetitive texture scenes (e.g., Stairs), requiring stronger global discriminability or integration of semantic information.
- The accuracy of the privacy mode is still about 40% lower than the feature mode, and the label granularity (choice of $K$ value) significantly impacts accuracy.
- The Triplane resolution is fixed; large-scale scenes may require higher resolutions or hierarchical representations.
- Initial pose estimate relies on image retrieval; if retrieval fails, refinement cannot converge.
- Robustness against adversarial privacy attacks was not evaluated—is it truly impossible to reconstruct scene information from geometry + labels?
- Learning the feature field and encoder requires independent training per scene; cross-scene generalization remains unexplored.

## Related Work & Insights
- **vs SegLoc**: SegLoc learns segmentation in 2D with no 3D consistency guarantee and projects sparse SfM points; GSFFs learns 3D-consistent features on 3DGS and converts them to segmentation, using dense rendered alignment. GSFFs Privacy significantly outperforms SegLoc on Cambridge.
- **vs SSL-Nif**: Both utilize self-supervised feature fields + pose refinement, but SSL-Nif is based on NeRF (slow rendering) and lacks privacy protection, while GSFFs is based on 3DGS (fast rendering) and supports a privacy mode.
- **vs GS-CPR**: GS-CPR uses pre-trained features + matching + PnP; GSFFs uses self-supervised feature learning + direct pose refinement, which is more accurate.
- The concept of spatial prototypes can be transferred to other 3DGS-based tasks like semantic segmentation and scene understanding.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The ideas of scale-aware feature fields and spatial prototypical clustering for privacy preservation are novel, though the pose refinement framework itself is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on multiple datasets, contrasting both privacy and non-privacy modes, though ablation studies could be more detailed.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is detailed with a clear logical flow from feature learning to segmentation conversion.
- **Value**: ⭐⭐⭐⭐ Privacy-preserving localization is an important real-world challenge; GSFFs offers a solid balance between accuracy and privacy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Feature-Preserving Mesh Decimation for Normal Integration](feature-preserving_mesh_decimation_for_normal_integration.md)
- [\[CVPR 2026\] ULF-Loc: Unbiased Landmark Feature for Robust Visual Localization with 3D Gaussian Splatting](../../CVPR2026/3d_vision/ulf-loc_unbiased_landmark_feature_for_robust_visual_localization_with_3d_gaussia.md)
- [\[CVPR 2026\] AsymLoc: Towards Asymmetric Feature Matching for Efficient Visual Localization](../../CVPR2026/3d_vision/asymloc_towards_asymmetric_feature_matching_for_efficient_visual_localization.md)
- [\[CVPR 2025\] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting](feat2gs_probing_visual_foundation_models_with_gaussian_splatting.md)
- [\[CVPR 2025\] 3D Dental Model Segmentation with Geometrical Boundary Preserving](3d_dental_model_segmentation_with_geometrical_boundary_preserving.md)

</div>

<!-- RELATED:END -->
