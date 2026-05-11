---
title: >-
  [Paper Note] Intrinsic Concept Extraction Based on Compositional Interpretability
description: >-
  [CVPR 2026][Image Generation][Concept Extraction] HyperExpress introduces a novel task termed Compositional Interpretability-based Intrinsic Concept Extraction (CI-ICE). By leveraging the hierarchical modeling capacity o…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Concept Extraction"
  - "Hyperbolic Space"
  - "Compositional Interpretability"
  - "Diffusion Models"
  - "Concept Disentanglement"
date: 2026-05-08
content_hash: 3237562b3d49a87f
---

# Intrinsic Concept Extraction Based on Compositional Interpretability

**Conference**: CVPR 2026
**arXiv**: [2603.11795](https://arxiv.org/abs/2603.11795)
**Code**: None
**Area**: Image Generation
**Keywords**: Concept Extraction, Hyperbolic Space, Compositional Interpretability, Diffusion Models, Concept Disentanglement

## TL;DR

HyperExpress introduces a novel task termed Compositional Interpretability-based Intrinsic Concept Extraction (CI-ICE). By leveraging the hierarchical modeling capacity of hyperbolic space and a horospherical projection module, it extracts composable object-level and attribute-level concepts from a single image, enabling invertible decomposition of complex visual concepts.

## Background & Motivation

**Background**: Unsupervised concept extraction (UCE) aims to extract human-interpretable visual concepts from a single image. Existing methods such as ConceptExpress and AutoConcept are limited to object-level concepts, while ICE can extract attribute-level concepts but does not account for composability.

**Core Problem**:
   - Existing methods focus solely on concept disentanglement while neglecting composability, preventing the extracted concepts from being reliably recombined to reconstruct the original image.
   - CCE considers composability but requires multiple images containing the same concepts.
   - Euclidean space fails to capture the hierarchical structure and relational dependencies between object-level and attribute-level concepts.

**Goal**: This work proposes the CI-ICE task and the HyperExpress method, which learns concept hierarchies in hyperbolic space and ensures the composability of the concept embedding space via horospherical projection.

## Method

### Overall Architecture

HyperExpress consists of two core components: **concept learning** and **concept refinement**. It first applies the first stage of ICE to localize salient objects and obtain masks and textual descriptions. Concepts are then learned through hyperbolic contrastive learning and entailment learning modules, followed by composability optimization via horospherical projection. Given an image containing $N$ objects, each with $M$ attributes, the goal is to learn $(M+1) \times N$ concept tokens and their corresponding embedding vectors.

### Key Designs

#### 1. Hyperbolic Text Encoder

CLIP-encoded text embeddings are mapped onto the Poincaré ball via the exponential map, with learnable weights $W$ introduced to learn the mapping from the standard encoder space to the tangent space. This endows concept embeddings with natural hierarchy: concepts closer to the center of the ball represent more abstract object-level concepts, while those near the boundary represent more concrete attribute-level concepts.

#### 2. Hyperbolic Contrastive Learning Module (HCL)

The hierarchical modeling capacity of hyperbolic space is exploited to distinguish object-level from attribute-level concepts:

- **Object–Attribute Discrimination**: A hyperbolic triplet loss enforces that the distance between an object-level concept anchor and its corresponding object embedding is smaller than that to attribute embeddings.
- **Inter-Attribute Discrimination**: An attribute-level triplet loss maintains appropriate distances between different attributes within the same attribute category.
- **Mechanism**: Concepts at different levels of the hierarchy are naturally positioned at different locations within hyperbolic space.

#### 3. Hyperbolic Entailment Learning Module (HEL)

Entailment relationships between objects and attributes are established in the Lorentz model:

- If concept $i$ entails concept $j$, the spatial angle $\theta(v_i, v_j)$ is smaller than the entailment cone half-angle $\omega(v_i)$.
- The entailment loss ensures that attribute concepts fall within the entailment cone of their corresponding object concept.
- Cone half-angles and spatial angles are computed via the transformation from the Poincaré ball to the Lorentz model.

#### 4. Horospherical Projection Module (HP)

The concept embedding space is mapped onto a composable submanifold:

- Trained on anchors, the module identifies $n$ geodesic directions that maximize variance after projection.
- The isometric property ensures that the learned hierarchical structure and inter-concept relationships are preserved.
- The projected submanifold inherits the zero-curvature property of horospheres, supporting vector addition and enabling concept composition.
- Rotation operations are realized via orthogonal matrices.

### Loss & Training

The total loss comprises four terms: reconstruction loss (diffusion model denoising), hyperbolic triplet loss (object-level and attribute-level), Wasserstein attention alignment loss, and hyperbolic entailment loss, each weighted by a corresponding $\lambda$ coefficient.

## Key Experimental Results

### Main Results

**UCEBench Performance Comparison (Table 1)**:

| Method | SIM^I (%) | SIM^C (%) | ACC^1 (%) | ACC^3 (%) |
|------|-----------|-----------|-----------|-----------|
| Break-A-Scene | 0.627 | 0.773 | 0.174 | 0.282 |
| ConceptExpress | 0.689 | 0.784 | 0.263 | 0.385 |
| AutoConcept | 0.690 | 0.770 | 0.350 | 0.520 |
| ICE | 0.738 | 0.822 | 0.325 | 0.518 |
| **HyperExpress** | **0.699** | **0.786** | **0.504** | **0.736** |

**ICBench Performance Comparison (Table 2)**:

| Method | SIM^T-T_obj | SIM^T-T_mat | SIM^T-T_color | SIM^T-V_obj | SIM^T-V_mat | SIM^V-T_color |
|------|------------|------------|--------------|------------|------------|--------------|
| ICE | 0.249 | 0.101 | 0.093 | 0.264 | 0.208 | 0.215 |
| **HyperExpress** | **0.280** | **0.115** | **0.098** | **0.305** | **0.211** | **0.222** |

### Ablation Study

| HCL | HEL | HP | SIM^I | SIM^C | ACC^1 | ACC^3 |
|-----|-----|-----|-------|-------|-------|-------|
| Y | N | N | 0.625 | 0.769 | 0.326 | 0.509 |
| Y | Y | N | 0.688 | 0.771 | 0.330 | 0.518 |
| Y | N | Y | 0.621 | 0.765 | 0.348 | 0.522 |
| Y | Y | Y | **0.699** | **0.786** | **0.504** | **0.736** |

### Key Findings

- HyperExpress substantially outperforms baselines on ACC^1 and ACC^3 (0.504 vs. 0.350), at the cost of a slight decrease in SIM^I relative to ICE.
- The three modules exhibit strong synergy: using HCL alone yields ACC^3 = 0.509, while the complete model achieves 0.736 (+44.6%).
- HP contributes most to composability, while HEL provides the largest gain in SIM^I.

## Highlights & Insights

1. **Novel Task Formulation**: CI-ICE simultaneously requires disentanglement and composability, addressing an underexplored research gap.
2. **Effective Application of Hyperbolic Geometry**: The Poincaré ball provides natural hierarchical modeling for concept-level organization.
3. **Theoretical Guarantee via Isometric Projection**: The isometric property of HP ensures that previously learned concept relationships are not distorted.
4. **Interpretable Composition Paths**: For example, "robot" + "metal" + "gold" → "golden robot made of metal".

## Limitations & Future Work

1. SIM^I performance falls short of ICE, indicating a fidelity cost imposed by the composability constraint.
2. The method depends on the first-stage object localization of ICE.
3. Computations in hyperbolic space introduce additional complexity.
4. Evaluation is conducted solely on the D1 dataset.
5. Attribute types are restricted to color and material.

## Related Work & Insights

- **ICE**: The direct predecessor; performs intrinsic concept extraction from a single image but does not address composability.
- **CCE**: Provides a theoretical framework for composability but requires multiple images.
- **HoroPCA**: Inspires the design of the horospherical projection module.
- **Insights**: Hyperbolic space warrants broader exploration in visual concept learning.

## Rating

| Dimension | Score (1–5) | Notes |
|------|-----------|------|
| Novelty | 4 | Novel task formulation and innovative application of hyperbolic space |
| Technical Depth | 4 | Rigorous mathematical framework with theoretical proofs |
| Experimental Thoroughness | 3 | Limited datasets and baselines |
| Writing Quality | 4 | Clear and logical presentation |
| Value | 3 | Task is relatively academic in nature |
| Overall | 3.6 | |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multiagent_dialogue_framework_for_c.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
