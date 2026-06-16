---
title: >-
  [Paper Note] Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry
description: >-
  [ICLR2026][3D Vision][DINOv2] By training a 32,000-unit Sparse Autoencoder dictionary on DINOv2, this work systematically analyzes how downstream tasks recruit distinct concepts…
tags:
  - "ICLR2026"
  - "3D Vision"
  - "DINOv2"
  - "Sparse Autoencoder"
  - "Linear Representation Hypothesis"
  - "Minkowski Representation Hypothesis"
  - "interpretability"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 6b9b43afdfe4b937
---

# Into the Rabbit Hull: From Task-Relevant Concepts in DINO to Minkowski Geometry

**Conference**: ICLR2026
**arXiv**: [2510.08638](https://arxiv.org/abs/2510.08638)  
**Code**: [kempnerinstitute.github.io/dinovision](https://kempnerinstitute.github.io/dinovision)  
**Area**: 3D Vision
**Keywords**: DINOv2, Sparse Autoencoder, Linear Representation Hypothesis, Minkowski Representation Hypothesis, interpretability, Vision Transformer

## TL;DR
By training a 32,000-unit Sparse Autoencoder dictionary on DINOv2, this work systematically analyzes how downstream tasks recruit distinct concepts, reveals that representational geometry deviates from the Linear Representation Hypothesis (LRH), and proposes the Minkowski Representation Hypothesis (MRH), which posits that token representations are Minkowski sums of multiple convex polytopes, with concepts defined by proximity to prototype points rather than linear directions.

## Background & Motivation
DINOv2, as a self-supervised visual foundation model, achieves strong performance on classification, segmentation, depth estimation, and robotic perception, yet the nature of its internal representations remains poorly understood. Existing interpretability methods largely rely on the Linear Representation Hypothesis (LRH), which assumes that internal features can be expressed as sparse superpositions of nearly orthogonal directions. However, large-scale experiments in this work reveal that LRH cannot fully account for the observed geometric phenomena:

- Dictionary atoms exhibit higher-than-expected coherence, deviating from ideal Grassmannian frames
- Representations simultaneously contain sparse features and dense features (e.g., positional encodings)
- Token embeddings within a single image form smooth low-dimensional manifolds, even after removing positional information
- Concepts recruited by different tasks form low-dimensional functional subspaces

These observations motivate the authors to go beyond LRH and propose a new representational hypothesis grounded in convex geometry.

## Core Problem
1. Which internal concepts in DINOv2 are recruited by different downstream tasks, and is there functional specialization?
2. What are the statistical properties and geometric structure of the learned concept dictionary, and do they conform to LRH predictions?
3. Can a more accurate representational geometry hypothesis be proposed to explain the observed phenomena beyond LRH?

## Method

### Step 1: Building the Concept Dictionary
A Stable SAE is trained on the penultimate layer of DINOv2-B (with 4 register tokens):

- Dictionary size $c = 32{,}000$, with each token activating $k = 8$ concepts
- Sparse coding is realized via BatchTopK projection
- Dictionary atoms are constrained to lie within the convex hull of real activations (parameterized via 128,000 k-means centroids as $\bm{D} = \bm{S}\bm{C}$, with $\bm{S}$ row-stochastic), ensuring stability and reproducibility
- Trained for 50 epochs on 1.4M ImageNet-1K images, achieving reconstruction fidelity $R^2 > 88\%$

### Step 2: Concept Recruitment Analysis for Downstream Tasks (Part I)

For linear probes $\bm{Y} = \bm{A}\bm{W}^T$, using the decomposition $\bm{A} \approx \bm{Z}\bm{D}$, concept importance is measured by $\mathbb{E}(\bm{Z})\bm{W}'$. Three tasks exhibit strikingly different concept recruitment patterns:

**Classification — "Elsewhere" Concepts**: The most important concepts include not only the target object itself but also a class of "Elsewhere" concepts — these activate on all tokens outside the object, yet their activation is conditioned on the object's presence. This implements a form of conditional negation logic: "the object exists elsewhere, but the current token is not the object." This distributed logic may support classification by implicitly delineating object boundaries or encoding contextual contrast.

**Segmentation — Border Concepts**: Nearly all top-50 concepts activate along object contours or spatial boundaries. These "border concepts" vary in visual appearance across categories but exhibit highly consistent spatial footprints, forming tight clusters in embedding space that constitute a low-dimensional subspace.

**Depth Estimation — Monocular Depth Cues**: Controlled image perturbation experiments identify three families of depth-relevant concepts: (i) projective geometry cues (vanishing lines, converging structures), (ii) shading cues (soft illumination gradients), and (iii) local frequency transitions (texture discontinuities resembling bokeh effects). This aligns with classical principles from visual neuroscience.

**Register Tokens — Global Scene Attributes**: Hundreds of concepts activate exclusively on register tokens, encoding global, non-local attributes such as motion blur, lighting style, caustic reflections, and lens effects.

### Step 3: Statistical and Geometric Analysis of Concepts (Part II)

- **Activation Statistics**: Concepts exhibit a frequency–energy trade-off (triangular envelope), but 3 anomalously dense-activating concepts are identified, encoding positional information (left/right/bottom), revealing a mixed representational regime where sparse and dense features coexist
- **Co-activation Spectrum**: The eigenvalues of $\bm{Z}^T\bm{Z}$ decay smoothly without obvious block structure, indicating that the concept space is high-dimensionally distributed rather than modular
- **Dictionary Geometry**:
    - The pairwise inner-product distribution has heavier tails than the random baseline, deviating from Grassmannian frames
    - The singular value spectrum decays sharply, indicating anisotropy and low effective rank
    - Antipodal pairs exist ($\bm{D}_i \approx -\bm{D}_j$), encoding semantic opposites (e.g., "left vs. right," "white vs. black")
    - Hoyer sparsity is well below 1.0, confirming that concepts are distributed rather than neuron-aligned

### Step 4: Minkowski Representation Hypothesis (Part III)

**Core Definition**: The activation space $\mathcal{X}$ is the Minkowski sum of multiple "tile polytopes":

$$\mathcal{X} = \bigoplus_{i=1}^{m} \mathcal{P}_i, \quad \mathcal{P}_i = \text{conv}(\mathcal{A}_{\mathcal{T}_i})$$

Each token is represented as the sum of convex combinations over a small set of active tiles: $\bm{x} = \sum_{i \in S} \bm{z}_i \mathcal{A}_{\mathcal{T}_i}$, where $\bm{z}_i \in \Delta^{|\mathcal{T}_i|}$ and $|S| \ll m$.

**Theoretical Support**:
1. **Cognitive Science**: In Gärdenfors' conceptual space theory, concepts are convex regions and prototypes are extreme points
2. **Natural Support from Attention Mechanisms**: Single-head attention outputs $\in \text{conv}(\bm{V})$ (softmax provides barycentric coordinates); affine transformations preserve convex structure; multi-head aggregation produces Minkowski sums via addition
3. **Formal Proofs**: Three lemmas/propositions rigorously demonstrate that multi-head attention naturally instantiates MRH structure

**Empirical Signals**:
- Linear interpolation rapidly exits the data support, whereas piecewise-linear geodesics on the kNN graph remain on the manifold
- Archetypal Analysis (a single-tile special case of MRH) matches SAE reconstruction quality using only ~10 prototypes
- The prototype coefficient matrix spontaneously exhibits block-sparse structure, consistent with the tile hypothesis of MRH

## Key Experimental Results
- SAE dictionary size: 32,000 concept atoms, reconstruction $R^2 > 88\%$
- Classification recruits far more concepts than segmentation or depth estimation
- Top-100 concepts within a task show significantly higher cosine similarity than random subsets, with faster eigenvalue spectrum decay
- Positional information in the final layer is compressed into a 2D subspace
- Archetypal Analysis achieves SAE-level reconstruction error with only 10 prototypes
- Co-activation and geometric similarity are only weakly correlated ($r = 0.28$, $R^2 = 0.08$)

## Highlights & Insights
- **Large-Scale Interpretability Resource**: The paper releases the largest interactive interpretability browser for visual foundation models (32K concept browser)
- **Discovery of "Elsewhere" Concepts**: Reveals a counterintuitive classification mechanism — supporting classification decisions via conditional activation over non-object regions
- **Theoretical Elegance of MRH**: Convex geometric organization is derived naturally from the mathematical structure of multi-head attention, with a complete chain of lemma–proposition proofs
- **Interdisciplinary Perspective**: Organically integrates cognitive science (Gärdenfors conceptual spaces), discrete geometry (Minkowski sums), and deep learning interpretability
- **Profound Implications for Steering**: MRH predicts a natural upper bound on concept manipulation (saturation upon reaching the prototype), explaining the plateau and reversal phenomena observed in current SAE probing

## Limitations & Future Work
- MRH currently remains a hypothesis; empirical signals constitute "compatible evidence" rather than rigorous proof, and multiple geometric hypotheses could produce similar phenomena
- Minkowski decomposition is inherently non-identifiable (Proposition 2); the original generative factors cannot be uniquely recovered from single-layer activations alone
- Analysis is concentrated on a single model, DINOv2-B, with no validation of generalization to other ViT variants or larger models
- The Archetypal Analysis comparison is conducted only at the single-image token level, lacking cross-image global validation
- The three-family taxonomy for depth estimation is incomplete — some concepts exhibit mixed sensitivity

## Related Work & Insights

| Method / Hypothesis | Core Claim | Relation to This Work |
|---|---|---|
| LRH + SAE | Representations = sparse superpositions of near-orthogonal directions | Starting point of this work; limitations revealed through experiments |
| Sparse Autoencoder | Overcomplete dictionary learning | This work uses Stable SAE as an analysis tool |
| k-Deep Simplex / SpaDE | Representations lie within a convex hull | Pioneer work aligned with the MRH perspective |
| Gärdenfors Conceptual Spaces | Concepts = convex regions | Cognitive science theoretical foundation of MRH |
| Park et al. 2025 | Convex polytope encoding of concepts in language models | Parallel finding in the NLP domain |

### Further Insights
- **Rethinking SAE Interpretability**: If MRH holds, current concept extraction methods based on linear directions may only approximate projections of convex structure; concept extraction methods that are aware of architectural structure (e.g., attention weights) need to be developed
- **New Paradigm for Concept Manipulation**: Shifting from unbounded scaling along directions to moving toward prototype points, with a natural saturation point, may yield more stable steering strategies
- **Connection to 3D Vision**: The three-family structure of depth estimation concepts indicates that DINOv2 spontaneously learns classical monocular depth cues, providing theoretical guidance for unsupervised 3D representation learning
- **Geometric Interpretation of Multi-Head Attention**: Each head corresponds to a convex combination within a concept tile, providing a new geometric language for understanding and designing attention mechanisms

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (MRH is an entirely new representational hypothesis with a uniquely interdisciplinary perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Large-scale analysis is thorough, but empirical support for MRH remains preliminary)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, rigorous theoretical derivations, and polished figures)
- Value: ⭐⭐⭐⭐⭐ (Far-reaching impact on the interpretability field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DINO Eats CLIP: Adapting Beyond Knowns for Open-set 3D Object Retrieval](../../CVPR2026/3d_vision/dino_eats_clip_adapting_beyond_knowns_for_open-set_3d_object_retrieval.md)
- [\[CVPR 2026\] 3D-Aware Multi-Task Learning with Cross-View Correlations for Dense Scene Understanding](../../CVPR2026/3d_vision/3d-aware_multi-task_learning_with_cross-view_correlations_for_dense_scene_unders.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[ICLR 2026\] Quantized Visual Geometry Grounded Transformer](quantized_visual_geometry_grounded_transformer.md)
- [\[AAAI 2026\] TOSC: Task-Oriented Shape Completion for Open-World Dexterous Grasp Generation from Partial Point Clouds](../../AAAI2026/3d_vision/tosc_task-oriented_shape_completion_for_open-world_dexterous_grasp_generation_fr.md)

</div>

<!-- RELATED:END -->
