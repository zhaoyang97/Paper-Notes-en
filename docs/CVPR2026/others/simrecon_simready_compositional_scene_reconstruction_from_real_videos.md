---
title: >-
  [Paper Note] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos
description: >-
  [CVPR 2026][compositional scene reconstruction] This paper proposes SimRecon, a framework that automatically constructs simulation-ready compositional 3D scenes from real videos via a three-stage "perception → generation → simulation" pipeline. The core innovations are Active Viewpoint Optimization (AVO), which identifies the optimal projection viewpoint for single-object generation, and the Scene Graph Synthesizer (SGS), which guides physically plausible hierarchical assembly.
tags:
  - CVPR 2026
  - compositional scene reconstruction
  - simulation-ready
  - scene graph
  - active viewpoint optimization
  - physical assembly
date: 2026-05-08
content_hash: 6ccf3be03f87ef55
---

# SimRecon: SimReady Compositional Scene Reconstruction from Real Videos

**Conference**: CVPR 2026
**arXiv**: [2603.02133](https://arxiv.org/abs/2603.02133)
**Code**: [https://xiac20.github.io/SimRecon/](https://xiac20.github.io/SimRecon/)
**Area**: Other
**Keywords**: compositional scene reconstruction, simulation-ready, scene graph, active viewpoint optimization, physical assembly

## TL;DR

This paper proposes SimRecon, a framework that automatically constructs simulation-ready compositional 3D scenes from real videos via a three-stage "perception → generation → simulation" pipeline. The core innovations are Active Viewpoint Optimization (AVO), which identifies the optimal projection viewpoint for single-object generation, and the Scene Graph Synthesizer (SGS), which guides physically plausible hierarchical assembly.

## Background & Motivation

**Background**: 3D scene reconstruction follows three main paradigms — holistic neural reconstruction (3DGS/NeRF, non-interactive), manually or procedurally constructed simulators (AI2-THOR, ProcTHOR), and emerging compositional reconstruction (decomposing individual objects from multi-view input).

**Limitations of Prior Work**:
   - Holistic reconstruction lacks object boundaries and complete geometry, making it unsuitable for simulation and interaction.
   - Manual/procedural simulators are costly to build and yield unrealistic layouts.
   - Existing compositional reconstruction methods (DPRecon, InstaScene) rely on heuristic viewpoint selection, causing distortion in generated objects, and their outputs remain visual representations rather than simulation-ready scenes.

**Key Challenge**: Two critical gaps exist between real video and simulation-ready scenes — a **visual fidelity** gap in the "perception → generation" stage and a **physical plausibility** gap in the "generation → simulation" stage.

**Key Insight**: Rather than redesigning the entire pipeline, the paper designs two **bridging modules** to address the core challenges at each stage transition.

**Core Idea**: Active Viewpoint Optimization acquires the projection with maximum information gain as the generation condition, while the Scene Graph Synthesizer guides hierarchical physical assembly.

## Method

### Overall Architecture

SimRecon adopts a three-stage pipeline with an object-centric spatial representation as the unified interface:

1. **Perception Stage**: Semantic reconstruction from video input (2DGS + semantic segmentation) to obtain pose, scale, and semantic label for each object.
2. **Generation Stage**: AVO identifies the optimal viewpoint projection, which serves as the conditioning input for a single-object generative model (Rodin) to produce complete geometry and texture.
3. **Simulation Stage**: SGS constructs a scene graph to guide hierarchical assembly within a physics simulator (Blender/Isaac Sim).

### Key Designs

1. **Object-Centric Scene Representation**:

    - Function: Represents the scene as $\mathcal{S}_\text{comp} = \{o_1, o_2, ..., o_L\}$, a set of discrete object primitives.
    - Each object contains **intrinsic attributes** (spatial $T_i \in SE(3)$, appearance $\mathcal{M}_i, \mathcal{T}_i$, physical $l_i, \text{mat}_i, m_i$) and **relational attributes** (support/attachment relations in the scene graph).
    - Design Motivation: Provides a unified interface across all three stages, with attributes populated progressively.

2. **Active Viewpoint Optimization (AVO)**:

    - Function: Searches 3D space for the projection viewpoint with maximum information gain for each object.
    - Mechanism: Formulates viewpoint selection as an information-theoretic problem $IG(v) = H(X|v_0) - H(X|v)$, using cumulative opacity rendered by 3DGS as a differentiable proxy for information gain:
    $\max_v IG(v) = \max_v A(v) = \max_v \sum_{p \in \mathcal{P}_\text{obj}(v)} \alpha(p,v)$
    - Depth regularization prevents viewpoint collapse onto the object surface: $L_\text{depth}(v) = \frac{\lambda_\text{depth}}{|\mathcal{P}_\text{obj}(v)|}\sum_p (D(p,v) - d_\text{target}(s_i))^2$
    - **Iterative Viewpoint Expansion**: After each viewpoint is selected, the effective opacity of already-covered Gaussians is multiplicatively decayed as $\alpha_i^{(k)} = \alpha_i^{(k-1)} \cdot (1 - \text{clip}(\alpha_i'(v_k^*), 0, 1))$, directing subsequent iterations toward unobserved regions.
    - Distinction from heuristic methods: Instead of relying on manually defined viewpoint sampling strategies, AVO performs gradient-based optimization directly in 3D space.

3. **Scene Graph Synthesizer (SGS)**:

    - Function: Infers support/attachment physical relations between objects and constructs a globally consistent scene graph.
    - **Regionalized Inference**: DBSCAN clusters objects into spatial regions; AVO obtains the optimal observation viewpoint for each region, and a VLM (Qwen2.5-VL) infers local subgraphs $\mathcal{G}_k$.
    - **Online Graph Merging**: BFS traversal incrementally merges subgraphs, detects conflicting edges (missing paths or hierarchical contradictions), and resolves conflicts by acquiring arbitration viewpoints and re-querying the VLM.
    - **Hierarchical Physical Assembly**: BFS proceeds from base nodes (floor/walls); support relations are resolved via gravity settlement simulation, while attachment relations are anchored with fixed constraints.
    - Design Motivation: Naive object placement leads to floating or interpenetrating objects, necessitating an understanding of physical dependencies.

### Loss & Training

The framework does not involve end-to-end training; each stage employs independently pretrained models (2DGS, SceneSplat, Rodin). AVO optimization takes approximately 30 seconds per object.

## Key Experimental Results

### Main Results — ScanNet Compositional 3D Reconstruction

| Method | CD↓ | F-Score↑ | NC↑ | PSNR↑ | SSIM↑ | LPIPS↓ | MUSIQ↑ | Time |
|--------|-----|----------|-----|-------|-------|--------|--------|------|
| Gen3DSR | 11.69 | 30.19 | 70.50 | 19.26 | 0.886 | 0.425 | 60.94 | 17min |
| DPRecon | 9.26 | 46.12 | 78.28 | 21.97 | 0.913 | 0.257 | 71.49 | 10h42m |
| InstaScene | 6.90 | 49.69 | 82.55 | 22.35 | 0.907 | 0.302 | 71.57 | 29min |
| **SimRecon** | **4.34** | **62.65** | **87.37** | **24.43** | **0.924** | **0.153** | **73.56** | 21min |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Max. 2D Visibility | Maximizes 2D pixel coverage only; viewpoints are insufficiently informative. |
| w/o $L_\text{depth}$ | Viewpoints collapse onto the object surface, rendering projections invalid. |
| Full AVO | Information gain maximization + depth regularization → optimal viewpoints. |
| Global Infer. (SGS) | Single global inference misses objects and relations. |
| Naive Merging (SGS) | Simple merging without conflict resolution produces contradictory relations. |
| **Full SGS** | Regionalized inference + online conflict resolution → consistent scene graph. |

### Key Findings
- AVO reduces CD (Chamfer Distance) by 37% compared to InstaScene (4.34 vs. 6.90), demonstrating that viewpoint quality is critical for generation outcomes.
- Maximizing 2D visibility does not equate to maximizing 3D information — small objects may have high 2D coverage yet suffer from severe occlusion.
- SGS hierarchical assembly yields significantly better physical plausibility than the MCMC post-processing used in MetaScenes.
- The modular pipeline design allows individual components at each stage (reconstruction/generation/simulation) to be replaced independently.

## Highlights & Insights
- **Information-Theoretic Viewpoint Optimization**: Formulates viewpoint selection as information gain maximization and employs 3DGS opacity as a differentiable proxy.
- **Bridging Module Design Paradigm**: Rather than redesigning the full pipeline, the approach identifies transition bottlenecks and introduces targeted bridging solutions.
- **Scene Graph as Physical Scaffold**: Physical relation reasoning is decoupled from reconstruction; the scene graph both guides assembly and remains interpretable.
- **Decay Mechanism for Iterative Viewpoint Expansion**: Attenuating the contribution of already-covered regions after each viewpoint selection naturally directs attention toward unobserved areas.

## Limitations & Future Work
- Relies on a VLM (Qwen2.5-VL) for physical relation inference, which may produce erroneous relations.
- Validation is limited to 20 ScanNet scenes; larger-scale and outdoor settings remain untested.
- The generation stage depends on Rodin, which may underperform on complex, transparent, or specular objects.
- Conflict resolution in SGS requires additional VLM queries, potentially reducing efficiency in complex scenes.
- The quality of inferred physical attributes (mass, material) depends on the VLM and lacks quantitative validation.

## Rating
- Novelty: ⭐⭐⭐⭐ The information-theoretic viewpoint formulation in AVO and the online graph merging strategy in SGS are genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐ Limited to 20 ScanNet scenes; scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with intuitive illustrations of the three-stage pipeline.
- Value: ⭐⭐⭐⭐ Provides a practical end-to-end solution for the "video-to-simulation" problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] 4DGT: Learning a 4D Gaussian Transformer Using Real-World Monocular Videos](../../NeurIPS2025/others/4dgt_learning_a_4d_gaussian_transformer_using_realworld_mono.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[CVPR 2026\] V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos](v_nutri_nutrition_estimation_cooking_videos.md)
- [\[CVPR 2026\] AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments](adasformer_adaptive_serialized_transformers_for_monocular_semantic_scene_complet.md)
- [\[CVPR 2026\] SHREC: A Spectral Embedding-Based Approach for Ab-Initio Reconstruction of Helical Molecules](shrec_a_spectral_embeddingbased_approach_for_abini.md)

</div>

<!-- RELATED:END -->
