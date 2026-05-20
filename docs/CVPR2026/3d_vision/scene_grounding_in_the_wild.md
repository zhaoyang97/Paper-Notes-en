---
title: >-
  [Paper Note] Scene Grounding In the Wild
description: >-
  [CVPR 2026][3D Vision][Scene Grounding] This paper proposes a semantic feature-based inverse optimization framework that aligns in-the-wild local 3D reconstructions (SfM) to a complete pseudo-synthetic reference model (e…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Scene Grounding"
  - "3D Reconstruction"
  - "Gaussian Splatting"
  - "Semantic Features"
  - "Cross-Domain Alignment"
date: 2026-05-08
content_hash: 06052b3d0abb3aa8
---

# Scene Grounding In the Wild

**Conference**: CVPR 2026
**arXiv**: [2603.26584](https://arxiv.org/abs/2603.26584)  
**Code**: [https://tau-vailab.github.io/SceneGround/](https://tau-vailab.github.io/SceneGround/)  
**Area**: 3D Vision
**Keywords**: Scene Grounding, 3D Reconstruction, Gaussian Splatting, Semantic Features, Cross-Domain Alignment

## TL;DR

This paper proposes a semantic feature-based inverse optimization framework that aligns in-the-wild local 3D reconstructions (SfM) to a complete pseudo-synthetic reference model (e.g., Google Earth Studio). By leveraging DINOv2 features and robust optimization, the method addresses large domain gaps and achieves globally consistent fusion of non-overlapping local reconstructions.

## Background & Motivation

1. **Background**: Reconstructing 3D scenes from unstructured photo collections is a core challenge in computer vision. Classical SfM and modern learning-based methods (DUSt3R, MASt3R, VGGT, etc.) can reconstruct scenes from large-scale image sets, provided sufficient visual overlap exists between viewpoints.
2. **Limitations of Prior Work**: Large-scale real-world image collections often suffer from severe viewpoint bias—tourists, for example, predominantly photograph the façade of Milan Cathedral rather than its rear. This causes SfM to produce multiple disconnected local reconstructions, or to incorrectly merge non-overlapping regions.
3. **Key Challenge**: The absence of overlap invalidates feature-matching-based geometric correspondence. Tools such as Google Earth Studio can render complete scene coverage, but their rendered images differ substantially from real photographs in appearance (domain gap), rendering traditional photometric losses ineffective for alignment.
4. **Goal**: Ground in-the-wild local reconstructions to a complete reference model, achieving globally consistent alignment.
5. **Key Insight**: Despite large appearance differences, real photographs and pseudo-synthetic renderings share the same scene semantics. The key insight is that semantic features extracted by foundation models such as DINOv2 remain consistent across domains, enabling a semantics-based inverse optimization.
6. **Core Idea**: Distill semantic features into a 3DGS reference model, then optimize a 6DoF+scale transformation by minimizing the L1 loss between rendered and real-image semantic features, augmented with Least Trimmed Squares (LTS) robust optimization to handle outliers.

## Method

### Overall Architecture

**Inputs**: (1) A 3DGS reference model built from Google Earth Studio renderings, with distilled DINOv2 features; (2) a set of in-the-wild real images and their SfM reconstruction (termed a meta-image). **Output**: A global 6DoF+scale transformation $T$ aligning the meta-image to the reference model. The reference model parameters are frozen; only the 7-parameter transformation (SE3 + scale) is optimized. Gradients from a semantic feature loss computed via differentiable rendering are back-propagated to update $T$.

### Key Designs

1. **Semantic Features as a Substitute for Photometric Loss**

    - **Function**: Provides an effective cross-domain supervision signal.
    - **Mechanism**: DINOv2 feature vectors are distilled onto each Gaussian in the 3DGS model (following the Feature 3DGS paradigm), enabling the model to render both RGB images and feature maps. During optimization, an L1 loss $L_{sem}$ compares the rendered feature map against DINOv2 features extracted from real images. Because DINOv2 captures scene semantics rather than appearance details, the features remain consistent even when pseudo-synthetic and real images differ substantially in color and illumination.
    - **Design Motivation**: Traditional photometric losses (as used in iNeRF) fail entirely under large color discrepancies and low-quality reference models (ablation: $\Delta R = 6.48°$ vs. $2.48°$ for the proposed method). Comparisons against LSeg and DINOv2+DVT further confirm that raw DINOv2 features yield the best alignment performance.

2. **Least Trimmed Squares Robust Optimization**

    - **Function**: Handles outliers in rendered and real images (floaters, occlusions, etc.).
    - **Mechanism**: The alignment is formulated as $\hat{T} = \arg\min_T \varphi(\mathcal{L}(T|\mathcal{I}, \mathcal{M}))$, where the robust function $\varphi$ applies LTS—at each optimization iteration, images whose $L_{sem}$ values exceed the median of the previous round are discarded. This prevents rendered views occluded by floaters, or real photographs with transient objects or heavy occlusion, from corrupting the optimization.
    - **Design Motivation**: Outliers are unavoidable in in-the-wild collections (pedestrian occlusions, reference-model floaters, etc.). LTS's adaptive selection outperforms both fixed truncation and IRLS (verified by ablation).

3. **Multi-Initialization Support and Puzzle-Piece Global Alignment**

    - **Function**: Accommodates diverse initialization methods and aligns multiple local reconstructions to the reference model individually.
    - **Mechanism**: Compatible initializations include COLMAP, gDLS+++, and SuperPoint+LightGlue. Each meta-image is aligned to the reference model independently, assembling the full scene in a puzzle-piece fashion. This avoids the complexity of jointly optimizing all transformations simultaneously.
    - **Design Motivation**: Different initialization methods have complementary strengths and weaknesses (gDLS+++ is most stable but requires special setup; COLMAP is most general but yields larger initial errors). The inverse optimization consistently improves upon any initialization.

### Loss & Training

- **Loss Function**: L1 distance $L_{sem}$ on DINOv2 features, robustified via LTS.
- **Optimization**: Gradient descent over the 7-parameter transformation (6DoF SE3 + 1 scale).
- **Reference Model**: 3DGS with frozen parameters; only $T$ is updated.
- The 3DGS reference model is built from Google Earth Studio renderings using COLMAP, exploiting GPS coordinates.

## Key Experimental Results

### Main Results

WikiEarth benchmark (32 meta-images, 23 scenes):

| Method | ΔR° ↓ | ΔT ↓ | MTA% ↑ | O% ↓ | Failures |
|--------|-------|------|--------|------|----------|
| COLMAP | 4.99 | 0.12 | 66 | 12 | 0/32 |
| **Ours (COLMAP init)** | **2.48** | **0.12** | **81** | **0** | — |
| gDLS+++ | 2.86 | 0.12 | 78 | 6 | 1/32 |
| **Ours (gDLS+++ init)** | **2.69** | **0.13** | **84** | **3** | — |
| SP+LG | 3.74 | 0.25 | 74 | 15 | 5/32 |
| **Ours (SP+LG init)** | **3.13** | **0.24** | **81** | **7** | — |

Comparison against feed-forward 3D models (geodesic rotation error):

| Method | ΔR_I↔M° ↓ | ΔR_I↔I° ↓ |
|--------|-----------|-----------|
| DUSt3R | 54.40 | 29.27 |
| MASt3R | 24.18 | 12.52 |
| VGGT | 51.69 | 24.63 |
| π³ | 68.46 | 45.80 |
| **Ours (COLMAP init)** | **2.59** | **1.48** |

The proposed method achieves errors an order of magnitude lower than feed-forward models.

### Ablation Study

| Method | ΔR° ↓ | ΔT ↓ | MTA% ↑ | O% ↓ |
|--------|-------|------|--------|------|
| **Ours (full)** | **2.48** | **0.12** | **81** | **0** |
| Photometric Loss | 6.48 | 0.38 | 72 | 22 |
| LSeg | 4.78 | 0.34 | 62 | 19 |
| DINOv2 + DVT | 2.86 | 0.14 | 78 | 0 |
| w/o LTS | 3.78 | 0.19 | 69 | 3 |
| Fixed LTS | 2.78 | 0.14 | 72 | 0 |
| IRLS | 3.51 | 0.18 | 72 | 3 |

### Key Findings

- Photometric loss is nearly unusable in the cross-domain setting ($\Delta R$: 6.48° vs. 2.48°; O%: 22%), confirming the necessity of semantic features.
- Raw DINOv2 features outperform DVT-enhanced and LSeg segmentation features—fine-grained spatial semantics, rather than semantic category labels, are what scene alignment requires.
- LTS adaptive outlier detection is critical: removing LTS drops MTA from 81% to 69% and raises O% from 0% to 3%.
- Feed-forward 3D models (DUSt3R, MASt3R, VGGT, π³) completely fail in the non-overlapping setting, with errors ranging from 24° to 68°, versus 2.59° for the proposed method—an order-of-magnitude gap.
- The method generalizes to reference models constructed from drone footage, not just Google Earth.

## Highlights & Insights

- **Semantics as a Cross-Domain Bridge**: The paper elegantly exploits the observation that images from different domains share scene semantics, distilling DINOv2 features into 3DGS to enable differentiable rendering combined with semantic comparison. This strategy is transferable to any cross-domain 3D alignment scenario.
- **iNeRF Framework Upgraded to 3DGS**: Replacing NeRF with 3DGS yields real-time rendering speed, making inverse optimization iterations substantially more efficient, while extending the scope from single-frame pose estimation to global transformation alignment.
- **Value of the WikiEarth Benchmark**: The first dataset providing ground-truth alignment between pseudo-synthetic reference models and real-world reconstructions, filling an important evaluation gap.
- The stress-test against state-of-the-art feed-forward models is highly convincing—DUSt3R/MASt3R/VGGT are nearly entirely ineffective in the non-overlapping setting, underscoring the necessity of external reference models.

## Limitations & Future Work

- The method relies on external data sources such as Google Earth Studio, whose availability and quality vary by region.
- Each meta-image is aligned independently, leaving potential constraints among multiple meta-images unexploited.
- The quality of Google Earth Studio models is limited (low-resolution textures, coarse geometry), which may be insufficient for fine-grained alignment in certain scenes.
- The WikiEarth benchmark is primarily composed of European landmarks such as cathedrals, limiting scene diversity.
- Promising directions for improvement include: jointly optimizing transformations for multiple meta-images; leveraging LLMs/VLMs for scene–image semantic matching to assist initialization; and extending the framework to indoor or non-landmark scenes.

## Related Work & Insights

- **vs. iNeRF**: iNeRF optimizes single-frame camera pose using a photometric loss in controlled environments; this work optimizes a global transformation using a semantic loss over in-the-wild image collections. Key upgrades are the replacement of NeRF with 3DGS and the introduction of LTS robustification.
- **vs. DUSt3R/MASt3R/VGGT**: These feed-forward models excel when overlap is available, but collapse in the non-overlapping setting due to the absence of global geometric constraints. This demonstrates that end-to-end approaches from the LLM era cannot fully replace the classical reference-model-plus-inverse-optimization paradigm.
- **vs. GaussReg, NeRF2NeRF**: These methods perform registration between two 3D scenes but do not address the large appearance discrepancy inherent in cross-domain (synthetic-to-real) alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing semantic features into inverse optimization for cross-domain scene grounding is a natural yet effective contribution; the WikiEarth benchmark is a valuable addition to the field.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple initializations, comprehensive comparison against feed-forward models, detailed ablations, and drone-footage generalization experiments make for a very solid evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated, visualizations are rich, and the method description is well-organized.
- **Value**: ⭐⭐⭐⭐ — Strongly practical: the framework provides a solution to large-scale fragmented scene reconstruction with direct applications to cultural heritage preservation and urban modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Affostruction: 3D Affordance Grounding with Generative Reconstruction](affostruction_3d_affordance_grounding_with_generative_reconstruction.md)
- [\[CVPR 2026\] DROID-W: DROID-SLAM in the Wild](droid-slam_in_the_wild.md)
- [\[CVPR 2026\] GaussFusion: Improving 3D Reconstruction in the Wild with A Geometry-Informed Video Generator](gaussfusion_improving_3d_reconstruction_in_the_wild_with_a_geometry-informed_vid.md)
- [\[CVPR 2026\] GLINT: Modeling Scene-Scale Transparency via Gaussian Radiance Transport](glint_modeling_scene-scale_transparency_via_gaussian_radiance_transport.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](lumimotion_gaussian_relighting_dynamics.md)

</div>

<!-- RELATED:END -->
