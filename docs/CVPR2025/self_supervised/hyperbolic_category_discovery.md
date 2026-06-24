---
title: >-
  [Paper Note] Hyperbolic Category Discovery
description: >-
  [CVPR 2025][Self-Supervised Learning][Generalized Category Discovery] This work proposes the HypCD framework, shifting representation learning in Generalized Category Discovery (GCD) from Euclidean/spherical spaces to hyperbolic space (the Poincaré ball model). Capitalizing on the property of hyperbolic space where the volume grows exponentially—making it naturally suitable for encoding hierarchical structures—this work proposes hybrid distance-angle similarity learning and a…
tags:
  - "CVPR 2025"
  - "Self-Supervised Learning"
  - "Generalized Category Discovery"
  - "Hyperbolic Space"
  - "Poincaré Ball"
  - "Hierarchy"
  - "Hybrid Similarity"
date: 2026-05-08
content_hash: 5ae18312a6dc286a
---

# Hyperbolic Category Discovery

**Conference**: CVPR 2025  
**arXiv**: [2504.06120](https://arxiv.org/abs/2504.06120)  
**Code**: [https://visual-ai.github.io/hypcd/](https://visual-ai.github.io/hypcd/)  
**Area**: Self-Supervised Learning  
**Keywords**: Generalized Category Discovery, Hyperbolic Space, Poincaré Ball, Hierarchy, Hybrid Similarity

## TL;DR
This work proposes the HypCD framework, shifting representation learning in Generalized Category Discovery (GCD) from Euclidean/spherical spaces to hyperbolic space (the Poincaré ball model). Capitalizing on the property of hyperbolic space where the volume grows exponentially—making it naturally suitable for encoding hierarchical structures—this work proposes hybrid distance-angle similarity learning and a hyperbolic classifier. It achieves an accuracy improvement on CUB for SelEx from 69.1% to 71.8%, and on ImageNet-100 from 87.1% to 88.3%.

## Background & Motivation

**Background**: Generalized Category Discovery (GCD) requires simultaneously identifying known classes and discovering novel classes when only partial class labels are available. Existing methods learn representations in Euclidean or spherical spaces and perform label assignment using semi-supervised k-means.

**Limitations of Prior Work**: Euclidean space possesses polynomial volume growth while spherical space has a constant volume; neither is suitable for encoding hierarchical structures ubiquitous in reality (e.g., the hierarchical relationship "Animal $\rightarrow$ Bird $\rightarrow$ Sparrow"). Hierarchical relationships among fine-grained categories (part compositions, scale changes) cannot be efficiently encoded by these spaces.

**Key Challenge**: The core challenge of GCD lies in distinguishing fine-grained categories, which naturally possess hierarchical structures (e.g., sharing parts but differing in details). Euclidean space cannot naturally express such tree-like nested relationships.

**Goal**: Utilize a geometric space better suited for hierarchical data to enhance the discriminative ability of fine-grained categories in GCD.

**Key Insight**: The volume of hyperbolic space grows exponentially with distance, which is naturally suited for embedding tree-like hierarchical structures—the leaf nodes of a tree spontaneously obtain a larger "workspace" in hyperbolic space to preserve discriminability.

**Core Idea**: Migrate representation learning in GCD to the Poincaré ball hyperbolic space, replace the single distance metric with a hybrid distance-angle similarity, and leverage the exponential volume growth to better encode the hierarchical structures of fine-grained categories.

## Method

### Overall Architecture
Integrate a hyperbolic space module on top of any GCD baseline: (1) project Euclidean features extracted by DINO/ViT onto the Poincaré ball via exponential mapping; (2) perform contrastive learning using hybrid distance-angle similarity in hyperbolic space; (3) replace the Euclidean MLP with a hyperbolic classifier for parametric methods; (4) perform inference using standard Euclidean k-means (as the hierarchical knowledge learned via hyperbolic training has already been transferred to the features).

### Key Designs

1. **Hyperbolic Space Mapping**:

    - **Function**: Map Euclidean features into the Poincaré ball model.
    - **Mechanism**: Project Euclidean embeddings onto the Poincaré ball with curvature $-c$ using the exponential map $\exp_o^c(z)$. Crucially, a feature clipping operation is introduced to prevent gradient vanishing/explosion—since the gradient tends to infinity when embeddings approach the boundary of the ball, a clipping value $r$ is used to control the maximum norm.
    - **Design Motivation**: The conformal property of the Poincaré ball model preserves angular information after mapping, while the hyperbolic distance behaves like Euclidean distance near the origin and grows rapidly near the boundary.

2. **Hybrid Distance-Angle Similarity**:

    - **Function**: Simultaneously utilize both distance and angular information for contrastive learning in hyperbolic space.
    - **Mechanism**: Define two types of similarity—distance-based similarity $\mathcal{S}_d = -\mathcal{D}_\mathbb{H}$ (negative hyperbolic distance) and angular similarity (cosine similarity, which remains valid in hyperbolic space due to conformality). The overall loss is formulated as $\mathcal{L}_{hrep} = \alpha_d \mathcal{L}_{dis} + (1-\alpha_d) \mathcal{L}_{ang}$, where $\alpha_d \approx 0.5$ is found to be optimal.
    - **Design Motivation**: Relying solely on either distance or angle is insufficient. Distance captures hierarchical depth (points farther from the origin lie deeper in the hierarchy), while angle captures similarity within the same layer (sibling classes under the same parent node are closer in angle). The two are complementary.

3. **Hyperbolic Classifier (HypFFN)**:

    - **Function**: Replace the Euclidean MLP for classification in hyperbolic space.
    - **Mechanism**: Utilize Möbius addition and Möbius scalar multiplication to replace the addition and multiplication operations of standard linear layers: $v_i = w \otimes_c z_i^\mathbb{H}$. Safekeeping projection is applied to prevent numerical overflow.
    - **Design Motivation**: Standard linear layers in Euclidean space assume flat geometry, which is invalid in the curved hyperbolic space.

### Loss & Training
A hyperbolic version of the standard contrastive learning loss, utilizing supervised contrastive learning for labeled data and self-supervised contrastive learning for unlabeled data. During inference, k-means can be directly applied in the Euclidean space—experiments indicate that the hierarchical knowledge acquired during hyperbolic training is effectively transferred back to the Euclidean representations.

## Key Experimental Results

### Main Results (DINO backbone)

| Baseline Method | Dataset | Original All Acc | +HypCD All Acc | Gain |
|---------|--------|-------------|---------------|------|
| SPTNet | CUB | 65.8% | 68.2% | +2.4% |
| SelEx | CUB | 69.1% | 71.8% | +2.7% |
| SPTNet | Stanford Cars | 59.0% | 62.1% | +3.1% |
| SelEx | Stanford Cars | 60.8% | 63.4% | +2.6% |
| SPTNet | CIFAR-100 | 81.3% | 82.9% | +1.6% |
| SelEx | ImageNet-100 | 87.1% | 88.3% | +1.2% |

Consistent improvements are observed across all baselines and datasets, with more pronounced gains on fine-grained datasets (CUB, Cars).

### Ablation Study

| Configuration | Description |
|------|------|
| Distance similarity only | Inferior to hybrid |
| Angle similarity only | Inferior to hybrid |
| $\alpha_d = 0.5$ | Optimal balance |
| No feature clipping | Gradient explosion/vanishing |
| Euclidean k-means (inference) | Performance maintained, hyperbolic knowledge successfully transferred |

### Key Findings
- Hyperbolic space exhibits a greater advantage on fine-grained datasets (CUB +2.7%, Cars +3.1%), as fine-grained categories possess more prominent hierarchical structures.
- The representations generated during hyperbolic training remain effective even when projected back to Euclidean space for k-means, indicating that hierarchical structure information has been encoded within the features.
- Hybrid similarity outperforms any single metric, and $\alpha_d = 0.5$ indicates that distance and angular information are equally important.
- The framework serves as a plug-and-play module that can enhance various GCD baselines (SPTNet, SelEx), demonstrating generalizability.

## Highlights & Insights
- **Geometric space selection is crucial for GCD**: While prior GCD research primarily focused on loss functions and training strategies, this work presents the first systematic study on how the geometric properties of the representation space affect performance.
- **Practicality of "Hyperbolic during Training, Euclidean during Inference"**: This avoids the computational complexity of hyperbolic space during inference, while retaining the benefits of hierarchical encoding.
- **Physical intuition behind hybrid distance-angle similarity**: Distance characterizes the "depth in the hierarchical tree", while angle represents the "position in the same layer." This decomposition is highly intuitive.

## Limitations & Future Work
- The curvature $c$ of the Poincaré ball is manually set; adaptive curvature learning could yield better results.
- Using Euclidean k-means at inference is a trade-off; hyperbolic k-means might yield further improvements.
- Validated only on visual GCD, leaving textual or multi-modal GCD unexplored.
- The evaluation of hierarchical structures lacks explicit hierarchical metrics (such as quantitative analysis of hierarchical consistency).

## Related Work & Insights
- **vs SimGCD/SPTNet**: These methods operate in Euclidean/spherical spaces; HypCD yields consistent and significant improvements (+2-3%) by modifying the geometry.
- **vs Hyperbolic Representation Learning (HYPE, etc.)**: HYPE is designed for graph embedding; HypCD introduces hyperbolic space to the GCD task for the first time, designing a task-specific hybrid similarity for GCD.
- **vs Metric Learning**: Traditional metric learning employs Euclidean distance; hyperbolic distance offers theoretical advantages in fine-grained differentiation (exponential volume growth $\rightarrow$ more space to distinguish similar categories).

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce hyperbolic space into GCD, with an elegantly designed hybrid distance-angle similarity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple baselines, datasets, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear intuitive explanations of hyperbolic space.
- Value: ⭐⭐⭐⭐ A plug-and-play enhancement module for GCD, making a significant contribution to fine-grained category discovery.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MOS: Modeling Object-Scene Associations in Generalized Category Discovery](mos_modeling_object-scene_associations_in_generalized_category_discovery.md)
- [\[NeurIPS 2025\] Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](../../NeurIPS2025/self_supervised/consistent_supervised-unsupervised_alignment_for_generalized_category_discovery.md)
- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](../../ICCV2025/self_supervised/a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[CVPR 2026\] Decouple Your Discovery and Memory in Continual Generalized Category Discovery](../../CVPR2026/self_supervised/decouple_your_discovery_and_memory_in_continual_generalized_category_discovery.md)

</div>

<!-- RELATED:END -->
