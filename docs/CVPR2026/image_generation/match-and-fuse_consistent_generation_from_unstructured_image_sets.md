---
title: >-
  [Paper Note] Match-and-Fuse: Consistent Generation from Unstructured Image Sets
description: >-
  [CVPR 2026][Image Generation][Set-to-set generation] Match-and-Fuse is proposed as the first training-free consistent generation method for unstructured image sets. Images are treated as nodes and image pairs as edges to construct a pairwise consistency graph. Multi-view Feature Fusion (MFF) and feature guidance are employed to manipulate internal features during diffusion inference, achieving set-level cross-image consistency with a DINO-MatchSim of 0.80, substantially outperforming all baselines.
tags:
  - CVPR 2026
  - Image Generation
  - Set-to-set generation
  - cross-image consistency
  - diffusion models
  - feature fusion
  - correspondences
  - training-free
  - zero-shot
date: 2026-05-08
content_hash: e1fc733b75bd4138
---

# Match-and-Fuse: Consistent Generation from Unstructured Image Sets

**Conference**: CVPR 2026
**arXiv**: [2511.22287](https://arxiv.org/abs/2511.22287)
**Area**: Image Generation / Consistent Generation
**Keywords**: Set-to-set generation, cross-image consistency, diffusion models, feature fusion, correspondences, training-free, zero-shot

## TL;DR
Match-and-Fuse is proposed as the first training-free consistent generation method for unstructured image sets. Images are treated as nodes and image pairs as edges to construct a pairwise consistency graph. Multi-view Feature Fusion (MFF) and feature guidance are employed to manipulate internal features during diffusion inference, achieving set-level cross-image consistency with a DINO-MatchSim of 0.80, substantially outperforming all baselines.

## Background & Motivation

**State of the Field**: Everyday visual experiences are organized as image collections (photo albums, product catalogs, property listings), yet generative AI primarily focuses on single images or videos, leaving set-level consistent generation largely unexplored.

**Core Challenges**: (a) Image collections lack the temporal continuity of video and thus provide no motion cues; (b) shared content may undergo large deformations; (c) shared elements must remain consistent while non-shared regions are allowed to vary freely.

**Limitations of Prior Work**: Edicho is limited to pairwise editing propagated from a single reference; IC-LoRA requires LoRA fine-tuning; FLUX Kontext lacks an explicit consistency mechanism; 3D/video editing methods rely on overly strong assumptions.

**Key Findings**: T2I diffusion models exhibit a **grid prior** — when multiple images are jointly generated on a concatenated canvas, consistency emerges spontaneously, yet it is incomplete and degrades rapidly as the number of images increases.

**Core Idea**: Model the image collection as a complete graph and exploit pairwise grid priors together with dense 2D correspondences to perform multi-view feature fusion and guidance at the feature level.

## Method

### Overall Architecture
The inputs are $N$ images, $\mathcal{P}^{shared}$ (a description of shared content), and $\mathcal{P}^{theme}$ (style/theme). In preprocessing, dense matches $M_{ij}$ (RoMA) are computed for all image pairs and a VLM generates per-image captions. During inference, joint denoising is performed over the pairwise consistency graph.

### Key Designs

1. **Pairwise Consistency Graph**:

    - Graph $G=(V,E)$: nodes represent images; edges connect all image pairs.
    - Each edge corresponds to a two-image grid latent encoding $z_{ij}^t = \text{concat}(z_i^t, z_j^t)$, paired with concatenated depth maps and a grid prompt.
    - After each denoising step, each node extracts and averages its own latent encoding version from all adjacent edges.
    - **Scalability**: Node degree is capped at 4 (random neighbors); full connectivity is used for $N \leq 5$, yielding linear complexity beyond that.

2. **Multi-view Feature Fusion (MFF)**:

    - **Core Finding**: The cosine similarity between features at matched locations is strongly correlated with visual consistency.
    - Pairwise fusion: $\mathbf{f}_i[\mathbf{c}] \leftarrow \frac{1}{2}(\mathbf{f}_i[\mathbf{c}] + \mathbf{f}_j[M_{ij}(\mathbf{c})])$ for all matched coordinates $\mathbf{c} \in \mathcal{C}_i$.
    - Extension to $N$ images: features are first averaged across adjacent edges $\bar{\mathbf{f}}_i = \frac{1}{|\delta(i)|}\sum_{e \in \delta(i)} \mathbf{f}_i^e$, then fused across all images.
    - Applied to K,V feature maps at selected layers of the DiT.

3. **Feature Guidance**:

    - Matching feature distance objective: $L_{guide} = \frac{1}{|E|}\sum_{\{i,j\}\in E}\frac{1}{|M_{ij}|}\sum_{\mathbf{c}\in M_{ij}}\|\mathbf{f}_i[\mathbf{c}] - \mathbf{f}_j[M_{ij}(\mathbf{c})]\|_2$
    - Gradients with respect to $z_i^{t-1}$ are computed for light refinement in latent space.
    - MFF can be interpreted as the analytical solution to this objective; Guidance corrects residual inconsistencies.
    - Gradient propagation through the model provides a wider receptive field, yielding robustness to sparse matches.

### Input Correspondences
Dense 2D matches are computed using RoMA; shared regions are identified automatically via confidence filtering, requiring no manual masks.

## Key Experimental Results

### Main Results: Consistency and Prompt Adherence

| Method | CLIP Score↑ | DreamSim↑ | DINO-MatchSim↑ |
|------|------------|-----------|----------------|
| FLUX Kontext | 0.65 | 0.78 | 0.57 |
| IC-LoRA | 0.65 | 0.71 | 0.65 |
| FLUX | 0.67 | 0.76 | 0.66 |
| Edicho | 0.65 | 0.81 | 0.72 |
| **Match-and-Fuse** | **0.66** | **0.85** | **0.80** |
| w/o Guidance | 0.66 | 0.82 | 0.76 |
| w/o MFF | 0.66 | 0.83 | 0.78 |
| w/o Pairwise Graph | 0.66 | 0.82 | 0.75 |

### User Study & VLM Evaluation (2AFC, Win Rate of Ours)

| Baseline | User Preference↑ | VLM Preference↑ |
|---------|----------|---------|
| vs Kontext | 88% | 82% |
| vs IC-LoRA | 90% | 92% |
| vs FLUX | 92% | 94% |
| vs Edicho | 83% | 78% |

### Metric Alignment with Human Judgment

| Metric | Agreement with Humans↑ |
|------|-------------|
| DreamSim | 84.3% |
| VLM | 84.9% |
| **DINO-MatchSim** | **91.4%** |

### Key Findings
- DINO-MatchSim of 0.80 substantially surpasses the strongest baseline Edicho at 0.72 (+11.1%).
- All three components (Graph, MFF, Guidance) are individually necessary.
- Match-and-Fuse with 9 images achieves higher consistency than baselines operating on only 2 images.
- Even when matches are as sparse as 10%, DINO-MatchSim remains above 0.76, demonstrating strong robustness.
- DINO-MatchSim achieves 91.4% alignment with human judgment, substantially exceeding DreamSim at 84.3%.

## Highlights & Insights
- **First set-to-set generation method**: Extends generative AI to image collections as a fundamental visual unit.
- **Elegant graph formulation**: The pairwise consistency graph enables local pairwise operations with global information propagation, and $O(N^2)$ complexity can be sparsified to $O(N)$.
- **Discovery and exploitation of the grid prior**: The spontaneous consistency emerging from T2I models under grid layouts is a key insight.
- **DINO-MatchSim metric**: Matched points in source images are used to localize corresponding positions in generated images, enabling patch-level similarity measurement that is more discriminative than global metrics.
- Entirely training-free, zero-shot, and requires no manual masks.

## Limitations & Future Work
- Relies on the quality of dense correspondences; regions with few or no matches may remain inconsistent.
- Depends on the base model's adherence to depth-conditioned inputs.
- FlowEdit integration requires per-edit hyperparameter tuning.
- $O(N^2)$ edge count still incurs overhead for large collections, despite sparsification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define and address set-to-set consistent generation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Quantitative evaluation + user study + VLM assessment + novel metric + ablation + extended applications.
- Writing Quality: ⭐⭐⭐⭐⭐ Polished figures, elegant formulations, and clear problem definition.
- Value: ⭐⭐⭐⭐ Applicable to creative workflows such as product advertising, character design, and storyboarding.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Organizing Unstructured Image Collections using Natural Language](organizing_unstructured_image_collections_using_natural_language.md)
- [\[CVPR 2026\] Cycle-Consistent Tuning for Layered Image Decomposition](cycle-consistent_tuning_for_layered_image_decomposition.md)
- [\[ICLR 2026\] Consistent Text-to-Image Generation via Scene De-Contextualization](../../ICLR2026/image_generation/consistent_text-to-image_generation_via_scene_de-contextualization.md)
- [\[AAAI 2026\] Infinite-Story: A Training-Free Consistent Text-to-Image Generation](../../AAAI2026/image_generation/infinite-story_a_training-free_consistent_text-to-image_gene.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](../../ICCV2025/image_generation/matchdiffusion_training-free_generation_of_match-cuts.md)

<!-- RELATED:END -->
