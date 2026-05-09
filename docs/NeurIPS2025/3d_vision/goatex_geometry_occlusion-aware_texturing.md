---
title: >-
  [Paper Note] GOATex: Geometry & Occlusion-Aware Texturing
description: >-
  [NeurIPS 2025][3D Vision][3D texture generation] GOATex proposes the first occlusion-aware 3D mesh texturing framework. It decomposes meshes into visibility layers ordered from outermost to innermost via a ray-casting-based hit-level mechanism, applies a two-stage visibility control strategy combining normal flipping and residual face clustering, and performs visibility-weighted blending in UV space—achieving high-quality texture generation for both exterior surfaces and occluded interior surfaces.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D texture generation
  - occlusion awareness
  - diffusion models
  - UV texture blending
  - multi-layer texturing
date: 2026-05-08
content_hash: 736535c7165d888f
---

# GOATex: Geometry & Occlusion-Aware Texturing

**Conference**: NeurIPS 2025
**arXiv**: [2511.23051](https://arxiv.org/abs/2511.23051)
**Code**: Project page available; code to be confirmed
**Area**: 3D Vision
**Keywords**: 3D texture generation, occlusion awareness, diffusion models, UV texture blending, multi-layer texturing

## TL;DR
GOATex proposes the first occlusion-aware 3D mesh texturing framework. It decomposes meshes into visibility layers ordered from outermost to innermost via a ray-casting-based hit-level mechanism, applies a two-stage visibility control strategy combining normal flipping and residual face clustering, and performs visibility-weighted blending in UV space—achieving high-quality texture generation for both exterior surfaces and occluded interior surfaces.

## Background & Motivation

**Background**: Existing 3D mesh texturing methods (e.g., TEXTure, SyncMVD, Paint3D, TEXGen) predominantly rely on text-to-image diffusion models, generating textures by rendering multi-view images and back-projecting them into UV space. These methods perform well on externally visible surfaces.

**Limitations of Prior Work**: Existing methods are fundamentally limited to surfaces visible from external viewpoints. For occluded interior surfaces—such as building interiors, car dashboards, or tent inner walls—they either leave these regions untextured or apply heuristic approaches such as Voronoi filling, resulting in low-quality interior textures with visible seams and color inconsistencies.

**Key Challenge**: The multi-view render-and-back-project paradigm inherently lacks access to occluded geometry, while UV-space inpainting approaches lack geometric context, making it difficult to generate semantically coherent interior textures.

**Goal**: (1) How to systematically identify and expose occluded interior surfaces? (2) How to preserve overall structural consistency when exposing interior geometry? (3) How to seamlessly blend textures generated independently across multiple layers?

**Key Insight**: The mesh is treated as an onion-like layered structure. Ray casting is used to compute a "hit level" for each face, and texturing proceeds by exposing and processing layers from the outside in.

**Core Idea**: A ray-based occlusion-aware layering mechanism, combined with two-stage visibility control and weighted UV blending, enables unified high-quality texture generation for both interior and exterior surfaces.

## Method

### Overall Architecture
The input consists of an untextured 3D mesh and a text prompt. The pipeline proceeds in five stages: (1) **Superface construction**—aggregating fine-grained triangular faces into coherent regions; (2) **Hit level assignment**—assigning a visibility depth level to each superface via multi-view ray casting; (3) **Visibility control**—a two-stage strategy that progressively exposes interior geometry layer by layer; (4) **Texture synthesis**—generating textures for each layer using a depth-conditioned diffusion model; (5) **UV blending**—soft blending weighted by visibility confidence. The output is a complete UV texture map.

### Key Designs

1. **Superface Construction and Hit Level Assignment**:

    - **Function**: Aggregates large numbers of small triangular faces into low-curvature coherent superfaces, then assigns each superface a hit level representing its visibility depth.
    - **Mechanism**: Xatlas is used to over-segment the mesh into superfaces. Rays are cast from multiple viewpoints, and the intersection order $k$ for each ray–face pair is recorded. The influence weight of each ray is modulated by the cosine similarity between the ray direction and the face normal: $W(f,k) = \sum_{r \in R_k(f)} \max(-n(f) \cdot d(r), 0)$. The final hit level of a superface is determined by the intersection order with the highest cumulative weight: $H(SF_i) = \arg\max_k \sum_{f \in SF_i} W(f,k)$.
    - **Design Motivation**: Assigning hit levels directly to individual faces causes adjacent faces within the same planar region to be assigned to different layers, breaking semantic consistency. Superface aggregation ensures that coherent regions are processed uniformly.

2. **Two-Stage Visibility Control Strategy**:

    - **Function**: Addresses the sparsity and fragmentation of face sets when exposing interior geometry layer by layer.
    - **Mechanism**:
        - **Residual face clustering**: Rather than rendering only the faces belonging to the current layer, the method renders the full set of faces not yet textured: $F_k^{res} = F - \bigcup_{i=1}^{k-1} F_i^{init}$, analogous to peeling an onion.
        - **Normal flipping + backface culling**: Normals of already-textured faces are flipped so that they are culled at the current viewpoint, exposing the untextured interior faces. The final rendered face set is $F_k = F_k^{res} \cup (\overline{F} - \overline{F_k^{res}})$.
    - **Design Motivation**: Rendering faces strictly by layer level results in very sparse and fragmented face sets for deep hit levels, producing depth maps that fall outside the natural distribution (OOD) expected by diffusion models. Residual clustering increases density, while normal flipping preserves the overall shape.

3. **Weighted UV-Space Texture Blending**:

    - **Function**: Seamlessly merges textures generated independently per hit level into the final texture map.
    - **Mechanism**: For each viewpoint $v$ and hit level $k$, a UV-space weight $W_k^{(v)}$ is computed based on the cosine similarity between the view direction and the face normal. Cross-level normalization is performed using a masked softmax: $\overline{W_k} = \frac{e^{W_k} \odot \mathcal{M}_k}{\sum_{j=1}^{H} e^{W_j} \odot \mathcal{M}_j}$. The final texture is $\text{UV}_F = \sum_{k=1}^{H} \overline{W_k} \odot \text{UV}_k$.
    - **Design Motivation**: Naive overwriting or uniform averaging produces blurred boundaries or artifacts at interior–exterior transitions. Soft blending based on visibility confidence yields smooth inter-layer transitions.

### Loss & Training
GOATex requires no fine-tuning of pretrained diffusion models. It employs pretrained Stable Diffusion 1.5 with ControlNet as the texture generator, substantially reducing computational cost while preserving generalization capability. The method also supports separate text prompts for exterior and interior surfaces (dual-prompt), enabling fine-grained control over layered appearance.

## Key Experimental Results

### Dataset and Evaluation
A curated set of 226 high-quality meshes with complex interior geometry is selected from Objaverse/Objaverse-XL, spanning 12 object categories (houses, cars, buses, tents, etc.). Due to the absence of ground-truth interior textures, evaluation is conducted via user studies and GPT-based A/B preference testing.

### Main Results (Preference Rate % for GOATex vs. Baselines)

| Baseline | Human Preference (GOATex) | GPT-4o-mini | GPT-4o | GPT-4.1 | GPT-o3 |
|----------|--------------------------|-------------|--------|---------|--------|
| TEXTure | >60% | ~55% | ~52% | ~58% | ~55% |
| SyncMVD | >60% | ~55% | ~53% | ~57% | ~55% |
| Paint3D | >65% | ~70% | ~68% | ~72% | ~75% |
| TEXGen | >65% | ~72% | ~70% | ~75% | ~78% |

Human evaluators consistently and strongly prefer GOATex across all comparisons, with the advantage being particularly pronounced in interior texture quality.

### Ablation Study (Win Rate % When Incrementally Adding Components over SyncMVD Baseline)

| Configuration | GPT-4o-mini | GPT-4o | GPT-4.1 | GPT-o3 | Average |
|---------------|-------------|--------|---------|--------|---------|
| Hit level assignment | 82.50 | 66.67 | 75.00 | 77.50 | 75.68 |
| + Superface construction | 84.62 | 70.00 | 75.86 | 89.74 | 80.27 |
| + Soft UV blending | 79.49 | 82.05 | 90.00 | 95.00 | 86.49 |
| + Residual face clustering | 77.50 | 72.50 | 88.89 | 84.84 | 81.17 |
| + Normal flipping (full model) | 86.84 | 92.31 | 86.67 | 97.50 | **91.16** |

### Key Findings
- Normal flipping combined with backface culling is the most critical component, raising the overall win rate from 81.17% to 91.16%. Without it, residual clustering alone slightly degrades performance, as occluded faces cannot be correctly exposed.
- GPT-4.1 and GPT-o3 correlate more strongly with human evaluators (Pearson $r = 0.43$ and $0.34$, respectively); inter-GPT agreement (Cohen's $\kappa = 0.54$) exceeds inter-human agreement ($\kappa = 0.31$).
- GPT evaluators tend to favor smoother interior results, assigning slightly lower scores to explicit interior textures compared to human evaluators.

## Highlights & Insights
- **First occlusion-aware texture generation framework**: The hit level concept is concise yet effective, reformulating the seemingly complex interior texturing problem as a layer-by-layer rendering task. The normal-flipping trick cleverly exploits the backface culling mechanism of standard rendering pipelines to expose interior faces.
- **Zero fine-tuning**: The method achieves substantial improvements purely through geometric operations (layering, normal flipping, weighted blending) without any fine-tuning of diffusion models, resulting in low deployment overhead.
- **Dual-prompt control**: The ability to specify distinct text prompts for interior and exterior surfaces introduces a new control dimension for 3D asset creation, with potential applications in game development and VR scene production.

## Limitations & Future Work
- Hit level assignment is based solely on geometric visibility and does not account for semantic consistency. In complex geometries such as thin walls or nested cavities, semantically unified regions may be assigned to different levels, causing texture discontinuities at boundaries.
- The method depends on the quality of superface segmentation; Xatlas-based segmentation does not always guarantee semantic correctness.
- Computation time scales linearly with the number of hit levels, as each level requires a full diffusion inference pass. For a mesh with 5,000 faces, hit level assignment takes approximately 2 minutes and texture synthesis an additional 2 minutes.
- Future work could incorporate semantic segmentation priors to improve hit level assignment, or integrate multi-level blending directly into the denoising process to enhance cross-layer consistency.

## Related Work & Insights
- **vs. TEXTure / SyncMVD**: These methods handle exterior textures through iterative painting or multi-view synchronization, but rely on Voronoi filling for occluded regions. GOATex's layered rendering mechanism addresses interior texturing at a fundamental level.
- **vs. Paint3D / TEXGen**: These methods perform inpainting or generation in UV space, but UV space lacks geometric context and high-quality interior texture training data is scarce. GOATex operates in image space, leveraging the stronger image generation priors of diffusion models.
- **vs. SDS-based optimization methods (PaintIt, DreamMat)**: These optimize UV maps via score distillation sampling, which is computationally expensive and similarly unable to handle occluded regions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First method to systematically address interior texturing of 3D meshes; the hit level concept and normal-flipping trick are creative contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluation combines human studies and multiple GPT models with comprehensive ablations; traditional quantitative metrics are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is clear, method description progresses logically, and figures are abundant.
- **Value**: ⭐⭐⭐⭐ Addresses a genuine practical pain point; dual-prompt functionality has clear application value, though applicability is limited to closed or semi-closed objects.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion](geocomplete_geometry-aware_diffusion_for_reference-driven_image_completion.md)
- [\[CVPR 2026\] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](../../CVPR2026/3d_vision/seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](../../ICCV2025/3d_vision/occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[ICCV 2025\] Geometry Distributions](../../ICCV2025/3d_vision/geometry_distributions.md)
- [\[NeurIPS 2025\] TAPIP3D: Tracking Any Point in Persistent 3D Geometry](tapip3d_tracking_any_point_in_persistent_3d_geometry.md)

<!-- RELATED:END -->
