---
title: >-
  [Paper Note] Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][visual relocalization] SplatHLoc proposes a hierarchical visual relocalization framework based on Feature Gaussian Splatting (FGS). By combining adaptive viewpoint retrieval that synthesizes virtual views closer to the query perspective with a hybrid feature matching strategy (rendered features for coarse matching + semi-dense matcher for fine matching), the method achieves new state-of-the-art accuracy on both indoor and outdoor benchmarks.
tags:
  - CVPR 2026
  - 3D Vision
  - visual relocalization
  - Gaussian splatting
  - feature matching
  - novel view synthesis
  - hierarchical localization
date: 2026-05-08
content_hash: f604ad2c5bcce6d4
---

# Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2603.29185](https://arxiv.org/abs/2603.29185)
**Code**: [https://hqitao.github.io/SplatHLoc](https://hqitao.github.io/SplatHLoc)
**Area**: 3D Vision
**Keywords**: visual relocalization, Gaussian splatting, feature matching, novel view synthesis, hierarchical localization

## TL;DR

SplatHLoc proposes a hierarchical visual relocalization framework based on Feature Gaussian Splatting (FGS). By combining adaptive viewpoint retrieval that synthesizes virtual views closer to the query perspective with a hybrid feature matching strategy (rendered features for coarse matching + semi-dense matcher for fine matching), the method achieves new state-of-the-art accuracy on both indoor and outdoor benchmarks.

## Background & Motivation

1. **Background**: Visual relocalization aims to estimate the 6-DoF camera pose within a known scene and serves as a foundation for robotics navigation, AR, and autonomous driving. Mainstream approaches fall into three categories: structure-based methods (SfM point cloud + PnP), regression-based methods (direct pose or scene coordinate regression), and rendering-based methods (NeRF/GS-guided rendering). Among these, hierarchical methods such as HLoc have been widely adopted due to their modular design and scalability to large scenes.

2. **Limitations of Prior Work**: Hierarchical methods rely on feature matching against retrieved database images. However, uneven database image distribution leaves certain regions sparsely observed, causing large viewpoint discrepancies between retrieved images and the query, which leads to unreliable matching. Rendering-based methods can synthesize novel viewpoints but often produce artifacts such as floaters, making features extracted from rendered images unstable for matching.

3. **Key Challenge**: A feature gap exists between rendered features (directly rendered via FGS) and features extracted by image encoders. Rendered features carry multi-view prior knowledge and are suitable for coarse-level matching, but domain discrepancy with query features makes them unsuitable for precise pixel-level matching. Existing methods such as STDLoc overlook this asymmetry.

4. **Goal**: (1) Address the large viewpoint discrepancy caused by sparse database images; (2) bridge the feature gap between rendered and query features.

5. **Key Insight**: The authors observe that FGS-rendered features perform better in the coarse matching stage (due to multi-view prior knowledge and reduced error accumulation), while features extracted by image encoders are more effective in the fine matching stage (due to their ability to model precise geometric relationships).

6. **Core Idea**: Combine adaptive viewpoint retrieval using FGS to synthesize virtual near-view references with a hybrid matching strategy—rendered features for coarse matching and a semi-dense matcher for fine matching—simultaneously addressing sparse observation and the feature gap.

## Method

### Overall Architecture

SplatHLoc is a hierarchical relocalization framework that takes a query image as input and outputs a 6-DoF pose. The overall pipeline consists of three stages: (1) adaptive coarse-to-fine viewpoint retrieval—candidate images are first retrieved from the database, and if the number of inlier matches is insufficient, FGS is used to render virtual viewpoints for a second retrieval round; (2) hybrid feature matching—rendered features establish coarse correspondences while a semi-dense matcher refines them, yielding 2D-2D matches that are lifted to 2D-3D via a rendered depth map; (3) an initial pose is estimated via RANSAC-PnP and then iteratively refined through render-and-match cycles.

### Key Designs

1. **Feature Gaussian Splatting (FGS) Scene Representation**:

    - **Function**: Serves as a unified scene representation capable of simultaneously rendering color images, depth maps, and feature maps.
    - **Mechanism**: Each Gaussian primitive in standard 3DGS is augmented with a feature vector $\mathbf{f}_i \in \mathbb{R}^d$. For efficiency, a low-dimensional feature map $F_r^{\text{low}} \in \mathbb{R}^{C' \times H \times W}$ ($C'=64$) is rendered and then decoded to high-dimensional $C=256$ via a $3\times3$ convolutional decoder. The training loss is a weighted sum of photometric loss $\mathcal{L}_{\text{rgb}}$ and feature loss $\mathcal{L}_{\text{feat}}$.
    - **Design Motivation**: The low-dimensional rendering plus decoder design significantly reduces FGS map size (353 MB vs. 904 MB for STDLoc) and GPU memory requirements (4 GB vs. 12 GB), while cutting training time by a factor of three.

2. **Adaptive Coarse-to-Fine Viewpoint Retrieval**:

    - **Function**: Identify the reference image whose viewpoint is closest to the query.
    - **Mechanism**: In the coarse stage, MixVPR retrieves top-$k_1$ candidates using global descriptors; SuperPoint+LightGlue performs geometric verification on each pair to select the best match. If the inlier count $N$ falls below a threshold (indicating insufficient co-visibility), the fine stage is triggered: $k_2$ random perturbations (rotation $a°$, translation $b$ m) are applied to the coarse pose, virtual keyframes are rendered from FGS, and a second round of retrieval and geometric verification selects the best virtual viewpoint, where $k_3 < k_1 \leq 10 < k_2 \leq 150$.
    - **Design Motivation**: Unlike GPVK-VL, which pre-synthesizes a large number of virtual views, this approach synthesizes views on demand only when initial retrieval quality is insufficient. The progressively narrowing search space preserves computational efficiency.

3. **Hybrid Coarse-to-Fine Feature Matching**:

    - **Function**: Establish precise 2D-2D correspondences between the query image and the rendered reference image.
    - **Mechanism**: In the coarse matching stage, a similarity matrix between query features $F_t$ and rendered features $F_r^{\text{high}}$ is computed at $H/8 \times W/8$ resolution; mutual nearest neighbor (MNN) filtering after bidirectional softmax yields coarse correspondences $\mathcal{C}_{q,r}^c$. In the fine matching stage, the semi-dense matcher JamMa extracts fine features from both the query and rendered images at $H/2 \times W/2$ resolution; guided by coarse correspondences, local windows are cropped and matched to produce sub-pixel-level correspondences $\mathcal{C}_{q,r}^f$.
    - **Design Motivation**: Experiments confirm that rendered features outperform semi-dense coarse features in the coarse matching stage (due to multi-view knowledge), while the semi-dense matcher outperforms rendered features in the fine matching stage (due to precise geometric modeling). This complementary hybrid strategy outperforms either approach alone.

### Loss & Training

FGS training follows standard 3DGS with loss $\mathcal{L} = \mathcal{L}_{\text{rgb}} + \gamma \mathcal{L}_{\text{feat}}$, where $\gamma=1$. $\mathcal{L}_{\text{rgb}}$ combines L1 and D-SSIM losses ($\lambda=0.2$), and $\mathcal{L}_{\text{feat}}$ is the L1 loss between rendered features and SuperPoint-extracted features. Each scene is trained for 30K steps.

## Key Experimental Results

### Main Results

Relocalization accuracy on the 7-Scenes indoor dataset (median translation/rotation error):

| Method | Avg. Error (cm/°) | Type |
|--------|-------------------|------|
| SplatHLoc (Ours) | **0.55/0.17** | GS rendering |
| STDLoc | 0.76/0.24 | GS rendering |
| LoGS | 0.76/0.24 | GS rendering |
| ACE+GS-CPR | 0.78/0.25 | GS rendering |
| HLoc (SP+SG) | 3.31/1.08 | Structure-based |

Results on the Cambridge Landmarks outdoor dataset:

| Method | Avg. Error (cm/°) |
|--------|-------------------|
| SplatHLoc (Ours) | **9/0.13** |
| STDLoc | 10/0.14 |
| LoGS | 10/0.20 |

### Ablation Study

Ablation on the 7-Scenes Stairs scene (low texture):

| Configuration | Error (cm/°) | RR@[5cm,5°] | Note |
|---------------|--------------|-------------|------|
| Baseline (MixVPR + SP+LG) | 1.82/0.49 | 75.5% | Baseline |
| + Adaptive Retrieval | 1.57/0.45 | 80.5% | +5% from adaptive retrieval |
| + Hybrid Matcher | 1.14/0.33 | 84.0% | +8.5% from hybrid matching |
| Full (both combined) | **1.03/0.30** | **91.9%** | Complementary; total +16.4% |

### Key Findings

- The hybrid matcher contributes most: combining rendered-feature coarse matching with semi-dense fine matching improves RR@[2cm,2°] on 7-Scenes from 91.46% to 93.84% (with JamMa).
- The reversed configuration (semi-dense coarse + rendered fine) degrades performance, confirming the core observation that rendered features suit coarse matching while extracted features suit fine matching.
- SplatHLoc's iterative refinement runs nearly 2× faster than STDLoc, as only low-dimensional features are rendered and decoded rather than directly rendering high-dimensional feature maps.
- The FGS map is only 39% the size of STDLoc's (353 MB vs. 904 MB), with training time reduced to one-third.

## Highlights & Insights

- The observation on the **complementarity between rendered and extracted features** is particularly insightful—rendered features naturally exhibit multi-view consistency but suffer from domain gap, while extracted features are geometrically precise but lack cross-view information. This insight directly motivates the hybrid matching strategy.
- **On-demand virtual view synthesis** is a practical design choice—FGS rendering is triggered only when retrieval quality is insufficient, avoiding the storage and computational overhead of pre-synthesizing large numbers of virtual views.
- The low-dimensional rendering plus decoder design is transferable to other tasks requiring FGS feature rendering, such as semantic SLAM and 3D scene understanding.

## Limitations & Future Work

- Performance depends on Gaussian map quality, which degrades when training images are sparse.
- The authors suggest replacing COLMAP initialization with 3D reconstruction foundation models.
- Performance in dynamic scenes has not been evaluated.

## Related Work & Insights

- **vs. STDLoc**: STDLoc applies rendered features in both coarse and fine matching stages, overlooking the rendered-query feature gap. SplatHLoc uses rendered features only in the coarse stage and delegates fine matching to a dedicated matcher, which is more principled.
- **vs. GS-CPR**: GS-CPR employs MASt3R for matching but operates at low resolution with high computational cost; SplatHLoc uses the lightweight JamMa for fine matching, balancing accuracy and efficiency.
- **vs. GPVK-VL**: Pre-synthesizing large numbers of virtual keyframes incurs heavy storage overhead; SplatHLoc's on-demand synthesis is more efficient.

## Rating

- Novelty: ⭐⭐⭐⭐ The insight behind hybrid matching is novel and practical, though the overall framework is a composition of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three standard datasets, detailed ablations, runtime analysis, and map size comparisons are all provided.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured with natural motivation derivation.
- Value: ⭐⭐⭐⭐ Achieves comprehensive state-of-the-art results in visual relocalization with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)
- [\[CVPR 2026\] Scaling View Synthesis Transformers (SVSM)](scaling_view_synthesis_transformers.md)
- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)
- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)

<!-- RELATED:END -->
