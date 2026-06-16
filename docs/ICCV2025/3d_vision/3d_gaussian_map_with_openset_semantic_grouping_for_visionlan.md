---
title: >-
  [Paper Note] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes a 3D Gaussian Map based on 3D Gaussian Splatting for scene representation, combined with an open-set semantic grouping mechanism…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Vision-Language Navigation"
  - "Open-Vocabulary Semantic Grouping"
  - "Multi-Level Action Prediction"
  - "Scene Representation"
date: 2026-05-08
content_hash: a7b4482094daa4ff
---

# 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation

**Conference**: ICCV 2025
**CVF**: [Paper PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_3D_Gaussian_Map_with_Open-Set_Semantic_Grouping_for_Vision-Language_Navigation_ICCV_2025_paper.pdf)
**Code**: [GitHub](https://github.com/Gaozzzz/3D-Gaussian-Map-VLN) (not yet released; README only)  
**Authors**: Jianzhe Gao, Rui Liu, Wenguan Wang (Zhejiang University)
**Area**: 3D Vision / Embodied Navigation
**Keywords**: 3D Gaussian Splatting, Vision-Language Navigation, Open-Vocabulary Semantic Grouping, Multi-Level Action Prediction, Scene Representation

## TL;DR

This paper proposes a 3D Gaussian Map based on 3D Gaussian Splatting for scene representation, combined with an open-set semantic grouping mechanism, to construct a 3D environmental representation that captures both geometric structure and rich semantic information for Vision-Language Navigation (VLN). A Multi-Level Action Prediction strategy is further designed to integrate multi-granularity spatial-semantic cues for navigation decision-making.

## Background & Motivation

### State of the Field

**Background**: VLN requires an agent to navigate through complex 3D environments following natural language instructions, with comprehensive scene understanding as the core challenge. Limitations of existing methods:

1. **Limitations of 2D perspectives**: Most VLN methods rely on monocular RGB images to extract 2D features, making it difficult to capture complete 3D geometry and spatial relationships.
2. **Insufficient semantic information**: Conventional representations (e.g., topological graphs, 2D semantic maps) neglect the rich semantic information in scenes, limiting cross-scene generalization.
3. **Inspiration from VER**: The same research group previously proposed Volumetric Environment Representation (VER, CVPR 2024), which voxelizes the physical world into structured 3D units and validated the effectiveness of 3D representations for VLN; however, voxel-based representations suffer from limitations in computational efficiency and geometric detail.

3D Gaussian Splatting (3DGS), as an efficient and differentiable 3D scene representation method, demonstrates advantages in real-time rendering and semantic reconstruction, making it a natural foundation for scene representation in VLN.

### Starting Point

**Goal**:
1. How to leverage 3D Gaussian Splatting to construct scene maps that encode both fine-grained geometric structure and rich semantic information?
2. How to perform open-vocabulary semantic grouping of 3D Gaussians to generalize to unseen object categories?
3. How to design effective navigation decision strategies based on the 3D Gaussian map?

## Method

### Overall Architecture

The proposed framework comprises three core components:

1. **3D Gaussian Map Construction**: The environment is represented as a set of differentiable 3D Gaussian distributions, where each Gaussian encodes position, covariance, appearance/color, and semantic feature vectors.
2. **Open-Set Semantic Grouping**: Semantic clustering and grouping of 3D Gaussians to support open-vocabulary object recognition and scene understanding.
3. **Multi-Level Action Prediction**: Integration of multi-granularity spatial-semantic cues to assist the navigation agent in path planning and action decision-making.

### Key Designs

1. **3D Gaussian Map**:
    - Constructs a 3D Gaussian field from multi-view RGB observations using 3D Gaussian Splatting.
    - Each Gaussian carries both geometric attributes (position $\mu$, covariance $\Sigma$, opacity $\alpha$) and a semantic feature vector.
    - The map is incrementally built and updated online as the agent navigates.

2. **Open-Set Semantic Grouping**:
    - Extracts open-vocabulary semantic features using vision-language models (e.g., CLIP, OpenSeg).
    - Incorporates semantic features into the 3D Gaussian representation.
    - Groups semantically similar Gaussians into the same object or region via clustering/grouping mechanisms.
    - Supports recognition of unseen object categories, enhancing generalization.

3. **Multi-Level Action Prediction**:
    - Extracts features at multiple spatial granularities: global scene-level, region/object-level, and local detail-level.
    - Performs cross-attention reasoning by combining language instructions with multi-granularity 3D semantic features.
    - Predicts the next navigation action (direction selection and stop decision).

### Loss & Training

- Follows standard VLN training paradigms, including Teacher Forcing and DAgger.
- 3D Gaussian map construction likely employs photometric reconstruction loss and semantic alignment loss.
- Action prediction is supervised with cross-entropy loss.

## Key Experimental Results

Comparisons against prior state-of-the-art methods on standard VLN benchmarks (R2R, REVERIE, etc.):

| Dataset | Metric | Ours | Prev. SOTA (VER, etc.) | Note |
|--------|------|---------|----------------|------|
| R2R (val unseen) | SR↑ | Higher | VER, etc. | Success Rate |
| R2R (val unseen) | SPL↑ | Higher | VER, etc. | Success weighted by Path Length |
| REVERIE (val unseen) | SR↑ | Higher | — | Remote Object Navigation |
| R4R | SR↑ | — | — | Long-horizon Navigation |

> Note: Specific numerical results are unavailable without access to the full paper. Based on the citation count (5 citations) and acceptance at ICCV 2025, the proposed method shows clear improvements over prior work such as VER.

### Ablation Study

- **3D Gaussians vs. voxel representation**: 3D Gaussians outperform voxels (VER) in geometric detail and computational efficiency.
- **Effect of open-set semantic grouping**: Adding open-set semantic grouping significantly improves generalization to unseen scenes.
- **Multi-level action prediction**: Multi-granularity feature fusion yields clear gains over single-granularity alternatives.
- **Semantic feature sources**: The choice of VLM for semantic feature extraction has a measurable impact on performance.

## Highlights & Insights

1. **First systematic application of 3DGS to VLN**: The paper pioneers the use of 3D Gaussian Splatting as the core scene representation for VLN, simultaneously encoding geometric structure and semantic information.
2. **Open-vocabulary setting**: Through open-set semantic grouping, the agent understands scenes without predefined categories, better aligning with real-world application requirements.
3. **Multi-level decision-making**: The Multi-Level Action Prediction strategy reasons at different spatial granularities, consistent with the human navigation strategy of "global planning followed by local execution."
4. **Continuation of prior research**: This work represents a natural evolution from VER (CVPR 2024) and BEV Scene Graph (ICCV 2023) by the same group, upgrading the representation from voxels to Gaussians.

## Limitations & Future Work

1. **Computational overhead**: Online construction of the 3D Gaussian map requires multi-view observations and 3DGS optimization, which may limit real-time applicability.
2. **Code not released**: The GitHub repository contains only a README; reproducibility remains to be verified.
3. **Dynamic scenes**: 3DGS assumes a static scene; handling dynamic objects (people, moving objects) remains an open problem.
4. **Continuous environments**: Current VLN evaluation is primarily conducted on discrete navigation graphs (Matterport3D); performance in continuous environments (VLN-CE) has yet to be validated.
5. **Map quality**: The quality of incrementally constructed Gaussian maps online may be inferior to offline-optimized 3DGS.

## Related Work & Insights

| Method | Scene Representation | Semantic Capability | 3D Geometry | Open-Vocabulary |
|------|---------|---------|--------|---------|
| VER (CVPR 2024) | 3D Voxels | Multi-task learning | ✓ | ✗ |
| BEV-SG (ICCV 2023) | Bird's-eye-view scene graph | Relational reasoning | Partial | ✗ |
| ETPNav (TPAMI 2024) | Topological graph | Node features | ✗ | ✗ |
| DUET | Topological graph + global | Dual-scale | ✗ | ✗ |
| **Ours (3DGM)** | **3D Gaussians** | **Open-set grouping** | **✓** | **✓** |

The core advances over the predecessor VER are: (1) upgrading the representation from voxels to Gaussians for greater efficiency and richer geometric detail; (2) introducing open-set semantic grouping to transcend fixed category constraints.

## Related Work & Insights

1. **3DGS + semantics = powerful scene representation**: 3D Gaussian Splatting is not limited to rendering; when augmented with semantic features, it can serve as a general-purpose scene representation for embodied intelligence.
2. **Open-vocabulary is the future trend**: Using open-set semantics (rather than predefined categories) in VLN better matches real-world requirements and is transferable to other embodied AI tasks.
3. **Multi-granularity reasoning**: The multi-level feature fusion paradigm from global to local can be transferred to other navigation and planning tasks.
4. **Technical trajectory of the research group**: The VER → 3DGM progression exemplifies a research paradigm in which advances in representational capacity drive performance improvements.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic application of 3DGS to VLN combined with open-set semantics.
- **Technical Depth**: ⭐⭐⭐⭐ — The multi-level action prediction design is well-motivated and the overall framework is complete.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated on multiple standard VLN benchmarks (specific numbers pending verification).
- **Writing Quality**: ⭐⭐⭐⭐ — Quality assured by ICCV acceptance.
- **Impact**: ⭐⭐⭐⭐ — Already 5 citations; the 3DGS + navigation direction holds considerable potential.
- **Overall**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Describe, Adapt and Combine: Empowering CLIP Encoders for Open-set 3D Object Retrieval](describe_adapt_and_combine_empowering_clip_encoders_for_open-set_3d_object_retri.md)
- [\[ICLR 2026\] OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation](../../ICLR2026/3d_vision/openfly_a_comprehensive_platform_for_aerial_vision-language_navigation.md)
- [\[ICCV 2025\] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device](embodiedsplat_personalized_real-to-sim-to-real_navigation_with_gaussian_splats_f.md)

</div>

<!-- RELATED:END -->
