---
title: >-
  [Paper Note] Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval
description: >-
  [AAAI 2026][Image Generation][Text-3D Retrieval] This paper proposes H2ARN, which embeds text and 3D point cloud data in the Lorentz hyperbolic space. It addresses hierarchical representation collapse via a hierarchical ordering loss (entailment cones), and mitigates redundancy-induced saliency dilution via contribution-aware hyperbolic aggregation. The method achieves state-of-the-art performance on Text-3D retrieval and introduces the T3DR-HIT v2 dataset, which is 2.6× larger than its predecessor.
tags:
  - AAAI 2026
  - Image Generation
  - Text-3D Retrieval
  - Hyperbolic Space
  - Hierarchical Alignment
  - Entailment Cones
  - Contribution-Aware Aggregation
date: 2026-05-08
content_hash: 3fbc803f834afca9
---

# Hyperbolic Hierarchical Alignment Reasoning Network for Text-3D Retrieval

**Conference**: AAAI 2026
**arXiv**: [2511.11045](https://arxiv.org/abs/2511.11045)
**Code**: [https://github.com/liwrui/H2ARN](https://github.com/liwrui/H2ARN)
**Area**: Image Generation
**Keywords**: Text-3D Retrieval, Hyperbolic Space, Hierarchical Alignment, Entailment Cones, Contribution-Aware Aggregation

## TL;DR

This paper proposes H2ARN, which embeds text and 3D point cloud data in the Lorentz hyperbolic space. It addresses hierarchical representation collapse via a hierarchical ordering loss (entailment cones), and mitigates redundancy-induced saliency dilution via contribution-aware hyperbolic aggregation. The method achieves state-of-the-art performance on Text-3D retrieval and introduces the T3DR-HIT v2 dataset, which is 2.6× larger than its predecessor.

## Background & Motivation

With the explosive growth of 3D data, text-3D retrieval has become increasingly important. Existing methods face two fundamental challenges:

### Challenge 1: Hierarchical Representation Collapse (HRC)

Both text and 3D data exhibit natural **tree-structured hierarchies**:
- Semantic level: from abstract concepts to fine-grained details (e.g., "ceramic vase" → "trophy-shaped vase with handles" → "surface engraved with patterns")
- Geometric level: from global structure to local parts (entire scene → object → handles, textures)

**Root Cause**: In such hierarchies, the number of nodes grows **exponentially** with depth, whereas the volume of Euclidean space and conventional Riemannian spaces grows at most **polynomially**. Embedding an exponentially growing tree structure into a polynomially growing space inevitably produces a **crowding effect**, compressing semantically distinct but structurally similar samples into nearby positions.

### Challenge 2: Redundancy-Induced Saliency Dilution (RISD)

Real-world 3D data contains substantial redundancy (scanning artifacts, decorative textures), and text descriptions include non-discriminative elements (prepositions, function words). **Existing methods commonly use mean pooling** to aggregate local features, implicitly assuming equal contributions from all parts. As a result, critical geometric and semantic cues are averaged into redundant noise, weakening the ability to distinguish hard negative samples.

## Method

### Overall Architecture

H2ARN consists of two main modules:
1. **Structural Context Encoder** (Euclidean space): extracts and enriches contextual representations of local features
2. **Hyperbolic Hierarchical Alignment Module** (hyperbolic space): performs contribution-aware aggregation and hierarchical alignment in hyperbolic space

### Key Designs

#### 1. **Lorentz Hyperbolic Space Embedding**

The Lorentz model is chosen over the Poincaré ball because it provides an **isometric embedding** in the $(d+1)$-dimensional Minkowski space, preserving distances exactly and supporting stable closed-form geodesic operations.

Hyperboloid definition:
$$\mathbb{H}_c^d = \{\mathbf{u} \in \mathbb{R}^{d+1} : \langle \mathbf{u}, \mathbf{u} \rangle_{\mathcal{L}} = -\frac{1}{c}, u_{d+1} > 0\}$$

Lorentz inner product: $\langle \mathbf{u}, \mathbf{v} \rangle_{\mathcal{L}} = \langle \tilde{\mathbf{u}}, \tilde{\mathbf{v}} \rangle_E - u_{d+1}v_{d+1}$

Lorentz distance (geodesic length):
$$d_{\mathbb{H}}(\mathbf{u}, \mathbf{v}) = \frac{1}{\sqrt{c}} \text{arccosh}(-c \langle \mathbf{u}, \mathbf{v} \rangle_{\mathcal{L}})$$

**Key advantage of hyperbolic space**: Volume grows **exponentially** with radius ($\sim e^{r\sqrt{c}}$), naturally suited for embedding tree-structured hierarchies. Points near the origin represent the most abstract concepts, while points farther from the origin correspond to more concrete instances.

#### 2. **Contribution-Aware Hyperbolic Aggregation**

**Core Idea**: The Lorentz distance is used to measure each local feature's contribution to the global semantics, rather than applying simple mean pooling.

Procedure:
1. Compute an initial anchor $\bar{z} = \frac{1}{L}\sum_{i=1}^{L} z_i$ (Euclidean mean pooling)
2. Project the anchor and all leaf nodes into hyperbolic space via the exponential map $\exp_\mathbf{o}^c$
3. Compute the Lorentz distance from each leaf node to the anchor
4. Derive contribution weights $\omega_i$ via softmax over negative distances
5. Perform weighted summation in Euclidean space: $z^\star = \sum_{i=1}^{L} \omega_i z_i$
6. Map the result back to hyperbolic space to obtain the final global representation $\mathbf{h} = \exp_\mathbf{o}^c(z^\star)$

**Key property**: Since the norm of $z^\star$ is smaller than that of any individual $z_i$, its hyperbolic image naturally lies closer to the origin, capturing a more abstract and denoised global concept. A learnable modality-specific scaling factor $\alpha$ is introduced to prevent numerical overflow in the exponential map.

#### 3. **Dual Geometric Loss**

**Multi-positive contrastive loss $\mathcal{L}_{cont}$**: Similarity is defined based on negative Lorentz distance, using a symmetric InfoNCE loss:
$$s(i,j) = -d_{\mathbb{H}}(\mathbf{h}_{t,i}, \mathbf{h}_{p,j}) / \tau$$

$$\mathcal{L}_{cont} = \frac{1}{2}(\mathcal{L}_{t \to p} + \mathcal{L}_{p \to t})$$

**Hierarchical ordering loss $\mathcal{L}_{ord}$**: Entailment cones encode the partial order relation "text entails 3D."

A hyperbolic cone is defined for each text embedding $\mathbf{h}_t$:
- Cone axis: $\mathbf{h}_t$
- Half-aperture angle **shrinks** as the point moves farther from the origin: $\phi(\mathbf{h}_t) = \arcsin\left(\frac{2K}{\sqrt{c}\|\tilde{\mathbf{h}}_t\|_E}\right)$

No penalty is imposed when the paired 3D embedding $\mathbf{h}_p$ lies inside the cone; otherwise, the penalty is proportional to the angular deviation:
$$\mathcal{L}_{ord} = \max(0, \theta(\mathbf{h}_t, \mathbf{h}_p) - \phi(\mathbf{h}_t))$$

This asymmetric geometric constraint forces text embeddings to occupy more generic "ancestor" positions.

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{cont} + \lambda \mathcal{L}_{ord}$

### Loss & Training

- Text encoder: CLIP; point cloud encoder: DGCNN
- Shared latent dimension $d=512$
- Curvature $c$ and scaling factor $\alpha$ are both learnable (positivity enforced via log-parameterization)
- AdamW optimizer, learning rate $2 \times 10^{-3}$, $\lambda=0.2$, $\tau=0.07$, $K=0.1$
- 100 epochs, batch size 256

## Key Experimental Results

### Main Results

**T3DR-HIT v2 Dataset**

| Method | Backbone | Text→PC R@1 | Text→PC R@5 | PC→Text R@1 | Rsum |
|--------|----------|------------|------------|------------|------|
| RMARN (CLIP+PointNet) | 16 heads, 6 layers | 7.6 | 25.2 | 6.5 | 127.3 |
| RMARN (CLIP+DGCNN) | 32 heads, 8 layers | 13.4 | 38.3 | 18.4 | 220.3 |
| **H2ARN (Ours)** | 64 heads, 6 layers | **16.4** | **44.5** | **19.6** | **238.5** |

**T3DR-HIT Original Dataset**

| Method | Text→PC R@1 | Text→PC R@5 | Text→PC R@10 | Rsum |
|--------|------------|------------|-------------|------|
| RMARN (best config) | 31 | 61 | 69 | 161 |
| **H2ARN (Ours)** | **32** | **63** | **73** | **168** |

### Ablation Study

| Configuration | Text→PC R@1 | PC→Text R@1 | Rsum | Notes |
|---------------|------------|------------|------|-------|
| Full H2ARN | 16.4 | 19.6 | **238.5** | |
| w/o $\mathcal{L}_{ord}$ | 15.3 | 18.4 | 229.6 | Ordering loss is critical |
| w/o aggregation | 15.2 | 16.9 | 233.5 | Contribution-aware aggregation is effective |
| w/o both | 14.3 | 14.5 | 222.0 | Severe performance degradation |
| Euclidean + mean pooling | 10.1 | 12.5 | 196.3 | Hyperbolic space advantage is clear |
| Euclidean + contribution aggregation | 12.5 | 14.2 | 215.1 | Aggregation is effective even in Euclidean space |
| **Hyperbolic full (H2ARN)** | **16.4** | **19.6** | **238.5** | Optimal |

### Key Findings

1. **Decisive advantage of hyperbolic space**: Rsum improves by **42.2 points** from Euclidean mean pooling (196) to full H2ARN (238)
2. **Importance of $\mathcal{L}_{ord}$**: Removing the hierarchical ordering loss causes an Rsum drop of approximately 9 points
3. **Contribution-aware aggregation is effective even in Euclidean space** (196→215), demonstrating its independent value in addressing RISD
4. **64-head 6-layer** attention configuration achieves the best Rsum (238.5) across configurations, though different configurations lead on individual metrics
5. **T3DR-HIT v2 scaling is effective**: The 2.6× data expansion increases task difficulty and validates the model's scalability

## Highlights & Insights

- The motivation for applying **hyperbolic space to cross-modal retrieval** is rigorously argued, grounded in the fundamental contradiction between the exponential growth of tree hierarchies and the polynomial growth of Euclidean space
- The **entailment cone** design elegantly encodes the linguistic intuition that "text is more abstract than 3D" as a geometric constraint
- **Contribution-aware aggregation** naturally distinguishes salient from redundant features using hyperbolic distances, without requiring additional supervision
- Learnable curvature and scaling parameters enhance model flexibility

## Limitations & Future Work

- Only RMARN is used as a comparison baseline; comparisons with a broader set of methods are lacking
- The T3DR-HIT dataset remains relatively small-scale (8,935 pairs) even after expansion
- Supplementary text for fine-grained artifact data is generated by LLaVA, which may introduce hallucinations
- The current framework supports only text and point cloud modalities; other 3D representations such as meshes and multi-view images are not addressed
- Hyperbolic space operations (exponential map, arccosh, etc.) incur higher computational cost than their Euclidean counterparts

## Related Work & Insights

- Introducing hyperbolic space into cross-modal retrieval is an emerging trend; this work is the first to apply it to the Text-3D setting
- RMARN employs Riemannian attention but still operates in Euclidean or low-curvature spaces; H2ARN's constant negative curvature Lorentz model represents a more principled departure
- The concept of entailment cones originates from hierarchical relation modeling in natural language processing; its application to cross-modal settings constitutes a key novelty
- This work offers insights for fine-grained and mixed-granularity methods in image-text retrieval: hyperbolic space may provide advantages in handling multi-granularity alignment

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of hyperbolic space, entailment cones, and contribution-aware aggregation is novel, though each component has prior precedents
- **Experimental Thoroughness**: ⭐⭐⭐ — Ablation studies are thorough, but the number of comparison baselines is insufficient
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is rigorously argued, mathematical derivations are complete, and figures are clear
- **Value**: ⭐⭐⭐⭐ — Represents a significant advance in Text-3D retrieval; the hyperbolic geometry methodology is generalizable to other cross-modal tasks

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] HierLoc: Hyperbolic Entity Embeddings for Hierarchical Visual Geolocation](../../ICLR2026/image_generation/hierloc_hyperbolic_entity_embeddings_for_hierarchical_visual_geolocation.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](../../ICCV2025/image_generation/hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[AAAI 2026\] TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs](truthfulrag_resolving_factual-level_conflicts_in_retrieval-augmented_generation_.md)
- [\[AAAI 2026\] Breaking the Modality Barrier: Generative Modeling for Accurate Molecule Retrieval from Mass Spectra](breaking_the_modality_barrier_generative_modeling_for_accurate_molecule_retrieva.md)

<!-- RELATED:END -->
