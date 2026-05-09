---
title: >-
  [Paper Note] SceneForge: Enhancing 3D-text alignment with Structured Scene Compositions
description: >-
  [NeurIPS 2025][3D Vision][3D-text contrastive learning] This paper proposes SceneForge, a framework that composes individual 3D point cloud objects into multi-object scenes with explicit spatial relations, paired with LLM-refined compositional captions, to enhance data diversity and complexity for 3D-text contrastive learning, yielding consistent performance gains across multiple downstream tasks.
tags:
  - NeurIPS 2025
  - 3D Vision
  - 3D-text contrastive learning
  - compositional augmentation
  - point cloud scene composition
  - spatial relations
  - zero-shot classification
date: 2026-05-08
content_hash: b37180bf35fa8389
---

# SceneForge: Enhancing 3D-text alignment with Structured Scene Compositions

**Conference**: NeurIPS 2025
**arXiv**: [2509.15693](https://arxiv.org/abs/2509.15693)
**Code**: Not available
**Area**: 3D Vision
**Keywords**: 3D-text contrastive learning, compositional augmentation, point cloud scene composition, spatial relations, zero-shot classification

## TL;DR

This paper proposes SceneForge, a framework that composes individual 3D point cloud objects into multi-object scenes with explicit spatial relations, paired with LLM-refined compositional captions, to enhance data diversity and complexity for 3D-text contrastive learning, yielding consistent performance gains across multiple downstream tasks.

## Background & Motivation

Large-scale contrastive learning has fundamentally transformed vision-language modeling. The success of CLIP/ALIGN in 2D has motivated analogous 3D research (e.g., Uni3D, OmniBind, OpenShape). However, extending contrastive learning to 3D still faces core challenges:

**Scarcity of 3D-text data**: Compared to 2D image-text datasets, large-scale 3D-text datasets are extremely limited. Even the largest OpenShape dataset is far smaller in scale than its 2D counterparts.

**Limitations of existing augmentation methods**: 3D augmentation approaches such as PointCutMix and PointMixup either randomly mix points and destroy object semantics, or interpolate coordinates to produce unrealistic shapes.

**Lack of spatial relation modeling**: Existing 3D contrastive learning methods primarily handle single objects, leaving models without the ability to learn inter-object spatial relationships.

Two core insights motivate this work:

- **Inherent advantage of 3D**: Unlike 2D images, individual 3D point clouds can be freely composed into structured scenes without visual artifacts (no entanglement of background, lighting, or viewpoint).
- **Spatial controllability**: 3D data allows explicit control over object placement, which is difficult to achieve in 2D. Composed scenes can naturally be paired with text descriptions containing spatial relations (e.g., "A is above B").

## Method

### Overall Architecture

SceneForge is embedded as a data augmentation module into any 3D-text contrastive learning pipeline. Within each training batch, samples are marked as compositional with probability $\alpha$ (the remainder are treated as single-object samples). Marked samples are processed by the SceneForge module, which composes $K$ objects into a scene and generates the corresponding compositional text description.

### Key Designs

1. **3D Scene Forge**: Objects are placed sequentially according to spatial relations to ensure semantically coherent compositions.

    - Three spatial relations are defined: "over," "under," and "next to."
    - Object placement is governed by bounding box constraints:
        - "over" relation: the minimum z of $p_i$ is aligned above the maximum z of $p_{i-1}$: $\mathcal{P}(p_i, p_{i-1}, \text{"over"}) = \max_{\mathbf{z}}(p_{i-1}) - \min_{\mathbf{z}}(p_i)$
        - "next to" relation: a horizontal unit vector $\mathbf{d}$ is sampled in the xy-plane, and objects are aligned along that direction: $\mathcal{P}(p_i, p_{i-1}, \text{"next to"}) = (\max_{\mathbf{x} \in p_{i-1}} \langle \mathbf{x}, \mathbf{d} \rangle - \min_{\mathbf{y} \in p_i} \langle \mathbf{y}, \mathbf{d} \rangle) \mathbf{d}$
    - A fixed offset $\delta$ and Gaussian noise $\epsilon$ are added to prevent perfect alignment and introduce natural randomness.
    - The composed point cloud is finally downsampled to a target point count of $P = 10\text{k}$.

2. **Scene Caption Forge**: This component mirrors the 3D composition process by sequentially concatenating each object's description with the corresponding spatial relation, then refining the result using an LLM (Qwen2.5-7B-Instruct) to:

    - Correct grammar, punctuation, and sentence structure.
    - Preserve original semantics and spatial relations.
    - Enhance description diversity and fluency.
    - Additionally improve the imprecise BLIP-generated captions from the original OpenShape dataset.

3. **Training scheme design**:

    - **Loss partitioning**: Compositional samples participate only in the text–3D contrastive loss (since real-time rendering of 2D views is infeasible), while single-object samples participate in both text–3D and image–3D losses. The image–3D loss is scaled by $\frac{1}{1-\alpha}$ to balance gradient contributions:
    $$\mathcal{L} = \underbrace{\frac{1}{2}[\mathcal{L}_{3D \to txt} + \mathcal{L}_{txt \to 3D}]}_{\text{all } N \text{ samples}} + \frac{1}{1-\alpha} \underbrace{\frac{1}{2}[\mathcal{L}_{3D \to 2D} + \mathcal{L}_{2D \to 3D}]}_{\text{single-object only}}$$
    - **Augmentation constraints**: Objects to be composed are prohibited from translation augmentation (to avoid post-composition inconsistency); full rotation around the vertical axis is permitted, while rotations along other axes are restricted to preserve the semantics of "over"/"under."
    - **Model-agnosticism**: The approach is effective across three different encoders (OpenShape-PointBERT, Uni3D-G, ViT-Lens-G), with the CLIP encoder kept frozen throughout.

### Loss & Training

The standard InfoNCE contrastive loss is used:
$$\mathcal{L}_{m \to n}(\mathcal{S}) = -\frac{1}{|\mathcal{S}|} \sum_{i \in \mathcal{S}} \log \frac{\exp(\langle e_i^m, e_i^n \rangle / \tau)}{\sum_{j \in \mathcal{S}} \exp(\langle e_i^m, e_j^n \rangle / \tau)}$$

Training runs for 200 epochs with a global batch size of 1152, $\alpha = 0.5$, and a maximum composition count of $N = 3$. SceneForge requires only one additional GPU to run the lightweight LLM. A producer-consumer parallelism strategy is adopted: while training on batch $t$, batch $t+M$ is prepared concurrently.

## Key Experimental Results

### Main Results

**Zero-shot classification accuracy (Top-1%, ensemble; training set includes LVIS):**

| Method | LVIS | ModelNet | ScanObjNN | ScanNet | Avg. Δ |
|--------|------|----------|-----------|---------|--------|
| ULIP-2 | 50.6 | 84.7 | 51.5 | 38.9 | — |
| MixCon3D | 52.5 | 86.8 | 58.6 | 44.1 | — |
| OmniBind-L | 54.0 | 86.6 | 64.7 | 46.3 | — |
| Uni3D | 53.5 | 87.3 | 63.9 | 45.8 | — |
| **SF-Uni3D** | **54.7** | **88.2** | **65.2** | **49.4** | **+1.75** |

**ScanQA 3D visual question answering:**

| Method | B-4 | CIDEr | EM | ΔB-4 | ΔCIDEr |
|--------|-----|-------|----|------|--------|
| OmniBind-L + BLIP2 | 8.5 | 62.9 | 17.1 | — | — |
| Uni3D + BLIP2 | 7.5 | 58.3 | 16.4 | — | — |
| **SF-Uni3D + BLIP2** | **10.4** | **66.7** | **20.5** | **+2.9** | **+8.4** |

### Ablation Study

| Configuration | LVIS T1 | ModelNet T1 | ScanObjNN T1 | ScanNet T1 | Note |
|---------------|---------|-------------|--------------|-----------|------|
| N=1 (Uni3D baseline) | 53.5 | 87.3 | 63.9 | 45.8 | No composition |
| N=2 | 53.9 | 87.6 | 64.5 | 48.2 | Gains begin |
| **N=3** | **54.7** | **88.2** | **65.2** | **49.4** | **Best** |
| N=4 | ≈54 | ≈88 | ≈65 | ≈48 | Performance plateaus |
| N=5 | <54 | <88 | <65 | <47 | Excessive point cloud fragmentation |

**Comparison of 3D composition methods (N=2, Uni3D backbone):**

| Method | LVIS T1 | ModelNet T1 | ScanObjNN T1 | ScanNet T1 |
|--------|---------|-------------|--------------|-----------|
| Uni3D (no composition) | 53.5 | 87.3 | 63.9 | 45.8 |
| PointMixup | 39.2 | 78.7 | 41.4 | 30.2 |
| PointCutMix-K | 44.7 | 83.0 | 45.1 | 34.8 |
| PointCutMix-R | 53.5 | 87.1 | 64.1 | 47.5 |
| **SF-Uni3D (N=2)** | **53.9** | **87.6** | **64.5** | **48.2** |

### Key Findings

- **Optimal composition count is N=3**: Performance improves monotonically from 1 to 3, plateaus or declines at 4, and degrades across all metrics at 5. Under the fixed 10k-point budget, five objects cause significant fragmentation of geometric features.
- **α=0.5 is optimal**: An excessively high proportion of compositional samples (>0.5) compromises single-object understanding.
- PointMixup and PointCutMix-K substantially degrade performance by destroying object semantic integrity; PointCutMix-R, which randomly blends whole objects, performs slightly better but still falls short of structured composition.
- SF-Uni3D achieves the largest gains on spatial reasoning questions in ScanQA, confirming that multi-object training genuinely enhances spatial relationship understanding.
- **N-object cross-modal retrieval**: Prior models drop sharply below 50% at N=2, whereas the SceneForge N=3 model maintains above 70% at N=6.

## Highlights & Insights

- **Orthogonal and model-agnostic augmentation strategy**: SceneForge modifies neither model architecture nor loss functions, operating purely at the data level, and can be plug-and-play integrated into any 3D-text contrastive learning pipeline.
- **Empirical evidence that the whole exceeds the sum of its parts**: Multi-object compositional training improves not only multi-object scene understanding but also single-object classification performance, consistent with the regularization effect observed for CutMix/MixUp in 2D.
- **Unique advantage of 3D**: The approach exploits the inherently background-free and freely composable nature of 3D data—a property that 2D data augmentation cannot easily replicate.
- **Generalization of spatial relation understanding**: Training with only three simple relations (over/under/next to) generalizes to more complex spatial relations in ScanQA (e.g., "attached to," "sitting on").

## Limitations & Future Work

- Only three spatial relations (over/under/next to) are defined; richer relations (e.g., inside, behind) may yield further improvements.
- Real-time rendering of composed scenes from 2D viewpoints is infeasible, restricting compositional samples to the text–3D loss only and limiting the enhancement of 2D–3D alignment.
- LLM caption refinement introduces additional computational overhead (0–50% training slowdown), which, although mitigated through parallelization, remains a bottleneck.
- At N≥4, composed scenes become overly crowded, and the 10k-point budget is insufficient to represent the geometric detail of all objects.

## Related Work & Insights

This work is complementary to MixCon3D, which focuses on mixing multi-view renderings and point clouds for multimodal representation, whereas SceneForge targets data diversity through structured composition. Compared to OmniBind, which relies on multi-model ensembling, SceneForge achieves superior performance with a single model at lower inference cost. The structured data augmentation paradigm introduced here is generalizable to other 3D-text tasks, including 3D grounding and 3D captioning.

## Rating

- Novelty: ⭐⭐⭐⭐ Leverages the inherently composable nature of 3D data for structured augmentation; the idea is concise and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers classification, segmentation, VQA, retrieval, and fine-tuning; model-agnosticism is validated across three backbone networks; ablations are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and ablation designs are well-motivated, though some tables are dense.
- Value: ⭐⭐⭐⭐ Provides a low-cost, general-purpose strategy for improving 3D-text contrastive learning with broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment](va-gs_enhancing_the_geometric_representation_of_gaussian_splatting_via_view_alig.md)
- [\[CVPR 2026\] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](../../CVPR2026/3d_vision/dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)
- [\[NeurIPS 2025\] AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians](atlasgs_atlanta-world_guided_surface_reconstruction_with_implicit_structured_gau.md)
- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)
- [\[NeurIPS 2025\] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)

<!-- RELATED:END -->
