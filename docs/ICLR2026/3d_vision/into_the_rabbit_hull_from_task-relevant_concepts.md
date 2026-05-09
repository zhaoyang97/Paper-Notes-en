---
title: >-
  [Paper Note] Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry
description: >-
  [ICLR 2026][3D Vision][Interpretability] This paper trains a sparse autoencoder (SAE) on DINOv2 to extract a dictionary of 32,000 visual concepts, systematically investigates how different downstream tasks (classification / segmentation / depth estimation) selectively recruit subsets of these concepts, reveals that the geometry of the representation space goes beyond the Linear Representation Hypothesis (LRH), and proposes a novel Minkowski Representation Hypothesis (MRH) positing that tokens are superpositions of multiple convex mixtures.
tags:
  - ICLR 2026
  - 3D Vision
  - Interpretability
  - Visual Representation
  - Sparse Autoencoders
  - DINOv2
  - Concept Discovery
date: 2026-05-08
content_hash: 3226292dd7b2a1f7
---

# Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry

**Conference**: ICLR 2026
**arXiv**: [2510.08638](https://arxiv.org/abs/2510.08638)
**Code**: None
**Area**: 3D Vision
**Keywords**: Interpretability, Visual Representation, Sparse Autoencoders, DINOv2, Concept Discovery

## TL;DR

This paper trains a sparse autoencoder (SAE) on DINOv2 to extract a dictionary of 32,000 visual concepts, systematically investigates how different downstream tasks (classification / segmentation / depth estimation) selectively recruit subsets of these concepts, reveals that the geometry of the representation space goes beyond the Linear Representation Hypothesis (LRH), and proposes a novel Minkowski Representation Hypothesis (MRH) positing that tokens are superpositions of multiple convex mixtures.

## Background & Motivation

1. **Black-box nature of DINOv2 representations**: DINOv2, as a self-supervised visual foundation model, achieves strong performance across classification, segmentation, depth estimation, and robotic perception, yet what it internally encodes remains poorly understood. Elucidating its representational structure is essential for improving and controlling such models.

2. **Limitations of the Linear Representation Hypothesis**: Existing interpretability work largely assumes the LRH — that features are sparse superpositions of nearly orthogonal directions. Whether the LRH fully characterizes the representational structure of vision Transformers has not been rigorously verified.

3. **Need for concept-level explanations**: Attribution methods such as Grad-CAM can answer *where* a model looks, but not *what features* it computes. Concept-based explanation approaches require a systematic framework for feature extraction.

4. **Instability of naïve SAEs**: Standard sparse autoencoders produce inconsistent features across training runs, severely undermining interpretability. Stable dictionary learning methods are needed to yield reproducible concepts.

5. **Lack of task-specificity analysis**: Prior work rarely examines how different downstream tasks selectively recruit distinct subsets of concepts from a shared representation space, nor how those concepts are geometrically organized.

6. **Deep geometric structure of representations**: Preliminary observations reveal anisotropy, antipodal pairs, and low-rank task subspaces in the representation space — phenomena that simple sparse coding models cannot explain, motivating a more refined geometric hypothesis.

## Method

### Stable Sparse Autoencoder

The authors use DINOv2-B (with 4 register tokens) as the feature extractor and train a stable SAE on 1.4 million ImageNet-1K images. Key settings:

- **Dictionary size**: $c = 32{,}000$ atoms, far exceeding the embedding dimension $d = 768$ (overcomplete)
- **Sparsity**: each token activates only $k = 8$ concepts
- **Stability constraint**: each dictionary atom is constrained to lie within the convex hull of real activations ($\bm{D}_i \in \text{conv}(\bm{A})$), approximated via 128,000 k-means cluster centers
- **Parameterization**: $\bm{D} = \bm{S}\bm{C}$, where $\bm{S}$ is a row-stochastic matrix and $\bm{C}$ contains cluster centers
- **Encoder**: single-layer encoder with BatchTopK projection yielding sparse codes $\bm{Z}$
- **Reconstruction quality**: $R^2 > 88\%$

### Task-Specific Concept Analysis

Concept importance is defined by analyzing the interaction between a downstream task's linear probe $\bm{W}$ and the concept dictionary:

$$\text{Importance} = \mathbb{E}(\bm{Z}) \cdot \bm{W}' = \mathbb{E}(\bm{Z}) \cdot \bm{D}\bm{W}^{\mathsf{T}}$$

This score is the optimal faithfulness measure under linearity, grounded in standard C-Deletion and C-Insertion metrics.

### Three Main Findings

**Part I: Downstream tasks recruit different concepts**

- **Classification → "Elsewhere" concepts**: The most important concepts include not only target object features but also a novel class of "Elsewhere" concepts — activating in regions *other than* where the target object appears. These concepts implement a fuzzy spatial logic: "the object exists elsewhere, but the current token is not that object."
- **Segmentation → boundary concepts**: All top-50 most important concepts activate along object contours, forming a compact low-dimensional subspace.
- **Depth estimation → three monocular depth cues**: perspective geometry cues (vanishing lines), shading cues (illumination gradients), and frequency transition cues (texture detail changes).

**Part II: Concept geometry and statistics**

- Dictionary atoms are far from orthogonal (deviating from Grassmannian frames), exhibiting significant local coherence
- Antipodal pairs ($\cos\theta \approx -1$) exist, encoding semantically opposing concepts (e.g., "left/right," "black/white")
- Singular value spectra decay rapidly, indicating anisotropy in representation space
- Three anomalously dense concepts encode positional information (left/right/bottom), activating frequently across the entire dataset

**Part III: Minkowski Representation Hypothesis**

The paper proposes the MRH: the activation space is the Minkowski sum of multiple convex polytopes:

$$\mathcal{X} = \bigoplus_{i=1}^{m} \mathcal{P}_i, \quad \bm{x} = \sum_{i \in S} \bm{z}_i \mathcal{A}_{\mathcal{T}_i}$$

where each tile $\mathcal{T}_i$ corresponds to a group of prototypes (e.g., animal categories, colors, textures) and $\bm{z}_i$ are convex coefficients (lying on a simplex). This structure is naturally realized by multi-head attention: each head produces a convex combination of value vectors, and the summed multi-head outputs form a Minkowski sum.

## Key Experimental Results

### Task-Specific Concept Recruitment

| Dimension | Classification | Segmentation | Depth Estimation |
|-----------|---------------|-------------|-----------------|
| Concept coverage | Broadly distributed | Compact and local | Compact and local |
| Top-100 concept cohesion | Moderate | High (boundary subspace) | High (three cue types) |
| Feature spectrum decay | Moderate | Fast | Fast |
| Concept type | Elsewhere + object concepts | Boundary detectors | Perspective / shading / frequency |

### SAE Dictionary Geometry

| Metric | SAE Dictionary | Random Baseline | Grassmannian |
|--------|---------------|----------------|-------------|
| Inner product distribution tail | Heavy-tailed (high coherence) | Narrow | Narrowest (ideal orthogonality) |
| Singular value decay | Fastest (low effective rank) | Moderate | Slowest |
| Hoyer sparsity | ~0.25 (distributed coding) | ~0.20 | N/A |
| Antipodal pair frequency | Significantly present | Rarely present | Absent |

### Global Concepts in Register Tokens

The authors find that hundreds of concepts activate exclusively on register tokens, encoding global scene attributes (motion blur, lighting style, caustic reflections, lens effects), while the cls token has only one dedicated concept. This reveals that DINOv2 structurally separates local (patch token) and global (register token) information.

## Highlights & Insights

- **First large-scale visual concept dictionary**: 32,000 interpretable concepts paired with the largest interactive visualization demo, providing infrastructure for visual model interpretability
- **Discovery of "Elsewhere" concepts**: Reveals a counterintuitive classification mechanism in DINO based on "conditional negation," challenging assumptions underlying conventional heatmap-based explanations
- **MRH hypothesis**: Elegantly connects cognitive science (Gärdenfors' conceptual spaces) with the mathematical structure of multi-head attention, offering a theoretical framework beyond the LRH
- **Stable SAE method**: Convex hull constraints ensure dictionary atoms remain in-distribution, resolving the cross-run inconsistency of standard SAEs

## Limitations & Future Work

- The MRH remains primarily a qualitative hypothesis; rigorous quantitative criteria for determining whether representations "truly" obey a Minkowski sum structure are lacking
- Analysis is restricted to DINOv2-B; it is unclear whether the findings generalize to other vision Transformers (e.g., CLIP, MAE, SigLIP)
- The causal mechanism underlying "Elsewhere" concepts is not fully elucidated — it is uncertain whether this is an explicitly learned strategy or an artifact of training
- The convex hull–constrained SAE, while stable, may limit the expressiveness of certain concepts (directions outside the convex hull cannot be represented)
- No systematic comparison with other concept discovery methods (e.g., NMF, PCA, ICA) is provided

## Related Work & Insights

| Dimension | Ours | Bricken et al. 2023 (Anthropic SAE) | Cunningham et al. 2023 |
|-----------|------|--------------------------------------|----------------------|
| Domain | Vision (DINOv2) | Language (Claude) | Language models |
| Dictionary size | 32,000 | 4,096–131,072 | Thousands |
| Stability | Guaranteed via convex hull constraint | No stability guarantee | No stability guarantee |
| Geometric analysis | In-depth (Grassmannian, MRH) | Preliminary | Preliminary |
| Downstream task analysis | Three-task comparison | None | None |

| Dimension | Ours | Park et al. 2024 (LRH) | Gärdenfors 2004 |
|-----------|------|------------------------|-----------------|
| Hypothesis | MRH (Minkowski sum) | LRH (linear sparse superposition) | Conceptual spaces (convex regions) |
| Target | Vision Transformers | Language models | Cognitive science theory |
| Operationalization | SAE + convex geometry | SAE / probes | No computational implementation |
| Core structure | Sum of convex polytopes | Nearly orthogonal directions | Convex regions + prototypes |

## Rating

- ⭐⭐⭐⭐⭐ **Novelty**: The MRH hypothesis aligns cognitive science theory with the mathematical structure of attention mechanisms — highly original
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Large-scale dictionary + three downstream tasks + multi-dimensional geometric analysis + interactive demo — analysis is exceptionally comprehensive
- ⭐⭐⭐⭐ **Writing Quality**: Clear structure and superb figures; however, the paper is somewhat lengthy and the theoretical rigor of Part III has room for improvement
- ⭐⭐⭐⭐ **Value**: The concept dictionary and demo are directly applicable to explaining and debugging visual models; the MRH offers practical guidance for representation engineering

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [\[ICLR 2026\] Quantized Visual Geometry Grounded Transformer](quantized_visual_geometry_grounded_transformer.md)
- [\[ICLR 2026\] Joint Shadow Generation and Relighting via Light-Geometry Interaction Maps](joint_shadow_generation_and_relighting_via_light-geometry_interaction_maps.md)
- [\[ICLR 2026\] Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation](ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)
- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)

<!-- RELATED:END -->
