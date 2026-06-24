---
title: >-
  [Paper Note] Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding
description: >-
  [CVPR 2025][Multimodal VLM][Vision-Language Models] Galaxy-Walker is proposed as the first geometry-aware vision-language model framework. By performing random walks across Euclidean, spherical, and hyperbolic spaces to generate Geometry Prompts, coupled with a Mixture-of-Geometry-Experts adapter (Geometry Adapter), it substantially outperforms general VLMs and domain-specific models on galaxy attribute estimation (with $R^2$ up to 0.91) and morphological classification tasks…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Non-Euclidean Geometry"
  - "Astronomy"
  - "Mixture-of-Experts"
  - "Riemannian Manifolds"
date: 2026-05-08
content_hash: 408ddbbcb8543b1e
---

# Galaxy Walker: Geometry-aware VLMs For Galaxy-scale Understanding

**Conference**: CVPR 2025  
**arXiv**: [2503.18578](https://arxiv.org/abs/2503.18578)  
**Code**: None  
**Area**: Physics  
**Keywords**: Vision-Language Models, Non-Euclidean Geometry, Astronomy, Mixture-of-Experts, Riemannian Manifolds

## TL;DR
Galaxy-Walker is proposed as the first geometry-aware vision-language model framework. By performing random walks across Euclidean, spherical, and hyperbolic spaces to generate Geometry Prompts, coupled with a Mixture-of-Geometry-Experts adapter (Geometry Adapter), it substantially outperforms general VLMs and domain-specific models on galaxy attribute estimation (with $R^2$ up to 0.91) and morphological classification tasks (with an F1 score improvement of +0.17).

## Background & Motivation

**Background**: Modern VLMs (such as GPT-4o, Claude 3.5, etc.) perform exceptionally well in visual question answering, but their core architectures (patch embeddings, convolutional backbones, self-attention mechanisms) are entirely built within Euclidean spaces. Machine learning in astronomy has evolved from traditional supervised learning to cross-modal models like AstroCLIP.

**Limitations of Prior Work**: Applying VLMs to astronomical analysis leads to severe performance degradation—models like GPT-4o achieve $R^2 < 0.6$ in galaxy attribute estimation and F1 scores of only 0.4-0.7 in morphological classification. This is due to the universe's geometric structures naturally embodying non-Euclidean geometry: planetary orbits involve spherical spaces, black holes involve hyperbolic spaces, while existing VLMs cannot represent these geometric properties.

**Key Challenge**: The universe exhibits rich geometric diversity at different scales—locally it is a flat Euclidean space, galactic hierarchical relationships are best represented in hyperbolic spaces, and global similarities are ideal for spherical spaces. However, VLM patch embeddings and FFN layers all assume planar distances, neglecting spherical/hyperbolic distance relationships.

**Goal**: To design a geometry-aware VLM framework capable of simultaneously processing astronomical features in Euclidean, spherical, and hyperbolic spaces.

**Key Insight**: Introduce geometry awareness at two levels: (1) Input layer—generate geometry tokens as prompts by conducting random walks on physical graphs across multiple geometric spaces; (2) Feature layer—replace standard FFNs with Mixture-of-Geometry-Experts, adaptively routing to experts in different geometric spaces based on token characteristics.

**Core Idea**: Inject multi-space geometric priors at the input of the VLM (via random walks on Riemannian graphs) and employ three types of FFN experts (Euclidean, spherical, hyperbolic) at the feature processing stage to handle spatial anisotropy across different geometric properties.

## Method

### Overall Architecture
Galaxy-Walker is built upon a pre-trained VLM and comprises two core components: (1) Geometry Prompt module—starting from the physical coordinates of galaxies (right ascension/declination), graphs are constructed in Euclidean, spherical, and hyperbolic spaces respectively, and geometric feature tokens are learned via Riemannian GraphSAGE; (2) Geometry Adapter module—Mixture-of-Geometry-Experts FFNs are inserted every $k$ layers in the VLM's transformer blocks, coupled with a gating network to route tokens. The output end is equipped with two parallel heads: a Numeric Head (regression) and an LM Head (classification).

### Key Designs

1. **Geometry Prompt**:

    - Function: Injecting multi-space geometric priors into the input layer of the VLM
    - Mechanism: Starting from physical coordinates of galaxies, coordinates in three spaces are obtained via projection and exponential mapping $\mathbf{V}_\mathbb{M} = exp_o^c(proj(\mathbf{V}_{phy}))$. KNN graphs are constructed in each space, followed by learning geometric features via a two-layer Riemannian GraphSAGE: the first layer $\mathcal{F}_{\mathbb{E} \to \mathbb{M}}$ maps Euclidean features to the target manifold, and the second layer $\mathcal{F}_{\mathbb{M} \to \mathbb{M}}$ performs message passing on the manifold. Finally, three sets of geometric tokens $\mathbf{P}_\mathbb{E}, \mathbf{P}_\mathbb{H}, \mathbf{P}_\mathbb{S}$ are generated as visual prompts.
    - Design Motivation: Euclidean space captures local proximity relations of galaxies, spherical space captures global topological similarity, and hyperbolic space captures hierarchical evolutionary relations—the three complement each other to fully describe cosmic geometry.

2. **Geometry Adapter**:

    - Function: Adapting to the anisotropy of different geometric spaces at the feature processing level
    - Mechanism: Three types of FFN experts are designed: (a) Euclidean expert $\mathcal{F}_E$ using a standard FFN; (b) Spherical expert $\mathcal{F}_S$ which normalizes the output and multiplies it by a learnable curvature $\kappa$ to ensure the output lies on the unit sphere; (c) Hyperbolic expert $\mathcal{F}_H$ which transforms via logarithmic/exponential mapping of the Poincaré ball. Adaptive routing is executed via a gating network $G$: $y = \sum_{i \in \{E,S,H\}} G_i(x) \cdot \mathcal{F}_i(x)$.
    - Design Motivation: Different astronomical features correspond to different geometric properties—planetary orbital angular relationships are suited for spherical processing, gravitational hierarchies near black holes are suited for hyperbolic processing, and regular image features are processed in Euclidean space.

3. **Two-Stage Training**:

    - Function: Efficiently training the geometry-aware modules
    - Mechanism: Stage 1 independently trains the Geometry Prompt module (learning geometric representations via a galaxy attribute estimation task); Stage 2 freezes the attention blocks and trains only the Geometry Adapter FFN layers, projection layers, and the Numeric Head.
    - Design Motivation: Stage-wise training reduces optimization difficulty and preserves the language modeling ability of the pre-trained VLM.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{LM} + \lambda \mathcal{L}_{reg}$, where $\mathcal{L}_{LM}$ is the language modeling loss, and $\mathcal{L}_{reg}$ is the Smooth L1 loss for the regression task. Modality inputs are processed with fp32 precision, and a learnable scaling factor is applied after L2 normalization. The training data comprises attribute estimation for 84K galaxies and ~200K morphological classification samples.

## Key Experimental Results

### Main Results

| Method | Galaxy Attribute Estimation $R^2$ | Morphological Classification F1 |
|------|-------------------|------------|
| GPT-4o | < 0.6 | 0.4-0.7 |
| Claude 3.5 | < 0.6 | 0.4-0.7 |
| AstroCLIP | Moderate | Moderate |
| Galaxy-Walker | **0.52-0.91** | **+0.17 F1** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Euclidean Expert Only | Lower | Lacks non-Euclidean geometric information |
| Spherical + Hyperbolic Only | Gain | Non-Euclidean geometry makes an independent contribution |
| All Three Experts | Optimal | Three spaces are complementary |
| W/o Geometry Prompt | Decreased | Injection of geometric prior is crucial |
| W/o Gating Routing | Decreased | Uniform mixing is inferior to adaptive routing |

### Key Findings
- General VLMs perform poorly on astronomical tasks ($R^2 < 0.6$), illustrating that geometry awareness is indispensable.
- The three geometric spaces have different strengths: hyperbolic space offers the largest improvement for hierarchical structures (e.g., BAR features), while spherical space contributes significantly to global morphological recognition.
- Galaxy-Walker shows an F1 score gain of up to +0.17 on challenging features (such as BAR and SAC), demonstrating the value of non-Euclidean geometric modeling.
- Training only the Adapter parameters (while freezing attention blocks) is computationally efficient and does not damage the pre-trained representation.

## Highlights & Insights
- **Introducing Riemannian Geometry into VLM Architectures**: This is not only an application innovation for astronomy but also provides a general paradigm for processing non-Euclidean geometric data in VLMs. Fields such as medical imaging (spherical surfaces) and social networks (hierarchical structures) may also benefit.
- **Using Riemannian GraphSAGE for Geometric Prompting**: Performing cross-manifold message passing on graph neural networks ($\mathcal{F}_{\mathbb{E} \to \mathbb{M}}$) is an elegant combination of geometric deep learning and VLMs.
- **A Novel Application of MoE Architectures**: Traditional MoE routes by semantics or tasks, whereas this work routes by geometric spaces, mapping the inherent diversity mechanism of MoE to physical spatial diversity.

## Limitations & Future Work
- The experiments are verified only within the astronomical domain; the framework's effectiveness in other domains requiring non-Euclidean geometry (such as medical spherical surfaces or molecular graph structures) remains unexplored.
- The geometric spaces are limited to three (Euclidean, spherical, hyperbolic), leaving more general Riemannian manifolds or mixed-curvature spaces uninvestigated.
- The routing decisions of the gating network lack interpretability; it remains unclear which astronomical features are routed to which geometric experts.
- The scale of the training data (84K galaxies) is relatively small by astronomical standards, and larger-scale training may yield further improvements.

## Related Work & Insights
- **vs AstroCLIP**: AstroCLIP performs cross-modal galaxy feature interaction but remains confined to Euclidean space. Galaxy-Walker extends this to multiple geometric spaces, significantly outperforming it.
- **vs GeoCode/GeoGPT4V**: These augment the geometric awareness of VLMs via data augmentation but still operate within a Euclidean framework. Galaxy-Walker introduces non-Euclidean geometry at the architectural level.
- **vs Mixed-Curvature Space Learning (κ-GCN)**: κ-GCN performs mixed-curvature learning on graphs. Galaxy-Walker introduces a similar concept into the FFN layers of VLMs, uniting visual, textual, and geometric modalities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces systematic non-Euclidean geometric modeling into VLMs for the first time, carving out a completely new research direction.
- Experimental Thoroughness: ⭐⭐⭐ Astronomical experiments are thorough, but restricted to a single domain, lacking cross-domain validation.
- Writing Quality: ⭐⭐⭐⭐ Motives are clear, methodology is described in detail, and figures and tables are highly intuitive.
- Value: ⭐⭐⭐⭐ Possesses direct application value for AI in astronomy, offering important inspiration for geometric modeling in VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](../../ICCV2025/multimodal_vlm/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[CVPR 2025\] Locality-Aware Zero-Shot Human-Object Interaction Detection](locality-aware_zero-shot_human-object_interaction_detection.md)

</div>

<!-- RELATED:END -->
