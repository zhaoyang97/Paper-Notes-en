---
title: >-
  [Paper Note] CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability
description: >-
  [CVPR 2026][Concept Extraction] This paper introduces the CI-ICE task and the HyperExpress method, which leverages the hierarchical modeling capacity of hyperbolic space (Poincaré ball) to extract composable object-level and attribute-level intrinsic concepts. By applying Horosphere projection to enforce compositionality in the concept embedding space, HyperExpress achieves an ACC₁ of 0.504 on UCEBench, a 55% improvement over ICE (0.325).
tags:
  - CVPR 2026
  - Concept Extraction
  - Compositionality
  - Hyperbolic Space
  - Poincaré Ball
  - Horosphere Projection
  - Diffusion Models
date: 2026-05-08
content_hash: ef98bbf6448fee3a
---

# CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability

**Conference**: CVPR 2026
**arXiv**: [2603.11795](https://arxiv.org/abs/2603.11795)
**Code**: N/A
**Area**: Interpretability / Concept Extraction
**Keywords**: Concept Extraction, Compositionality, Hyperbolic Space, Poincaré Ball, Horosphere Projection, Diffusion Models

## TL;DR

This paper introduces the CI-ICE task and the HyperExpress method, which leverages the hierarchical modeling capacity of hyperbolic space (Poincaré ball) to extract composable object-level and attribute-level intrinsic concepts. By applying Horosphere projection to enforce compositionality in the concept embedding space, HyperExpress achieves an ACC₁ of 0.504 on UCEBench, a 55% improvement over ICE (0.325).

## Background & Motivation

**State of the Field**: Unsupervised Concept Extraction (UCE) aims to extract human-interpretable visual concepts (objects, colors, materials) from a single image, serving as a key tool for model interpretability. ConceptExpress and AutoConcept extract concepts from single images, while ICE further enables the separation of object-level and attribute-level concepts.

**Limitations of Prior Work**: (1) ConceptExpress and AutoConcept extract only object-level concepts and cannot disentangle attributes such as color or material; (2) although ICE separates object and attribute concepts, it does not guarantee compositionality—the extracted concepts cannot be recombined to reconstruct the original complex concept; (3) CCE accounts for compositionality but requires multiple images sharing the same concepts, limiting its practicality.

**Root Cause**: Concept "disentanglement" ≠ concept "compositionality"—existing methods focus solely on disentanglement while ignoring the compositional structure of the concept space, rendering concept decomposition irreversible and uninterpretable.

**Paper Goals**: To extract intrinsic visual concepts from a single image that are both hierarchically disentangled (object-level vs. attribute-level) and composable (capable of being recombined to reconstruct the original concept).

**Starting Point**: The hierarchical modeling capacity inherent to hyperbolic space is exploited for concept learning, while the zero-curvature property of horospheres is utilized to guarantee compositionality.

**Core Idea**: Hierarchical concept relationships are learned within the Poincaré ball, and concepts are projected onto horospheres to enforce linear composability.

## Method

### Overall Architecture

HyperExpress addresses CI-ICE along two dimensions: **concept learning** (Hyperbolic Contrastive Learning, HCL; Hyperbolic Entailment Learning, HEL) and **concept optimization** (Horosphere Projection, HP). Given an image containing $N$ objects each with $M$ attributes, the first stage of ICE is used to localize objects and obtain masks $\mathcal{M}$ and text descriptions $\mathcal{T}^{anchor}$; subsequently, $(M+1) \cdot N$ concept token embeddings are learned.

### Key Designs

1. **Hyperbolic Contrastive Learning (HCL)**: Token embeddings are mapped onto the Poincaré ball via a CLIP encoder, learnable weights $W$, and the exponential map $\exp_0(\cdot)$. A hyperbolic triplet loss is applied in two steps to discriminate: (a) object-level vs. attribute-level concepts ($\mathcal{L}^{obj}_{triplet,k} = \max(0, d_{\mathbb{D}}(v_k^{anchor}, v_k^{obj}) - d_{\mathbb{D}}(v_k^{anchor}, v_k^{att}) + \gamma)$); and (b) different attributes belonging to the same object. **Design Motivation**: In hyperbolic space, semantically dissimilar concepts naturally reside farther apart, making it more suitable for hierarchical structure modeling than Euclidean space.

2. **Hyperbolic Entailment Learning (HEL)**: Entailment relationships between objects and attributes are established in the Lorentz model—attribute concepts should fall within the entailment cone of their corresponding object concept. The entailment loss is $\mathcal{L}_{entail,k} = \max(0, \cos(\omega(v_k^{obj})) - \cos(\theta(v_k^{obj}, v_k^{att})))$, where $\omega$ denotes the entailment cone half-angle and $\theta$ the spatial angle. **Design Motivation**: This geometrically encodes hierarchical associations such as "metal is an attribute of robot," making object–attribute relationships explicitly represented in the geometry.

3. **Horosphere Projection (HP)**: The primary objective is to guarantee compositionality. $n$ geodesic directions that maximize the variance of projected data are identified, and an orthogonal matrix $Q$ is used to rotate embeddings onto the composable submanifold. Key mathematical properties: the projection is distance-preserving ($d_{\mathbb{H}}(\pi(x), \pi(y)) = d_{\mathbb{H}}(x,y)$), preserving the learned hierarchical structure and entailment relationships; horospheres inherit Euclidean properties, enabling concepts to satisfy linear composition: $R([V_i] \cup [V_j]) = w_i R([V_i]) + w_j R([V_j])$.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{recon} + \lambda_{triplet} \mathcal{L}_{triplet} + \lambda_{attention} \mathcal{L}_{attention} + \lambda_{entail} \mathcal{L}_{entail}$. $\mathcal{L}_{recon}$ is the diffusion model denoising reconstruction loss; $\mathcal{L}_{triplet}$ comprises both object-level and attribute-level triplet losses; $\mathcal{L}_{attention}$ is a Wasserstein attention alignment loss that aligns T2I attention to masked regions; $\mathcal{L}_{entail}$ is the entailment loss. The method is implemented on top of Stable Diffusion.

## Key Experimental Results

### Main Results (UCEBench)

| Method | SIM_I (%) | SIM_C (%) | ACC₁ (%) | ACC₃ (%) |
|---|---|---|---|---|
| Break-A-Scene | 0.627 | 0.773 | 0.174 | 0.282 |
| ConceptExpress | 0.689 | 0.784 | 0.263 | 0.385 |
| AutoConcept | 0.690 | 0.770 | 0.350 | 0.520 |
| ICE | 0.738 | 0.822 | 0.325 | 0.518 |
| **HyperExpress** | 0.699 | 0.786 | **0.504** | **0.736** |

### Ablation Study (D1 Dataset)

| HCL | HEL | HP | SIM_I | SIM_C | ACC₁ | ACC₃ |
|---|---|---|---|---|---|---|
| ✔ | ✗ | ✗ | 0.625 | 0.769 | 0.326 | 0.509 |
| ✔ | ✔ | ✗ | 0.688 | 0.771 | 0.330 | 0.518 |
| ✔ | ✗ | ✔ | 0.621 | 0.765 | 0.348 | 0.522 |
| ✔ | ✔ | ✔ | **0.699** | **0.786** | **0.504** | **0.736** |

### Key Findings

- **Substantial gains on ACC metrics**: ACC₁ improves from 0.325 (ICE) to 0.504 (+55%), and ACC₃ from 0.518 to 0.736 (+42%), demonstrating a qualitative leap in concept disentanglement attributable to compositionality.
- **All three modules are indispensable**: The full HCL+HEL+HP configuration approximately doubles ACC₃ compared to HCL alone (0.509 → 0.736).
- **HP contributes most**: Removing HP causes ACC₃ to drop from 0.736 to 0.518, confirming that Horosphere projection is the key to compositionality.
- **Trade-off on SIM metrics**: SIM_I and SIM_C are slightly lower than ICE (0.699 vs. 0.738), indicating that the compositionality constraint moderately limits single-concept reconstruction fidelity.

## Highlights & Insights

- Positioning "compositionality" as the central objective of concept extraction represents a task-definition-level innovation—concept decomposition should be invertible.
- Applying hyperbolic space to visual concept extraction is a novel angle; its hierarchical modeling capacity naturally aligns with the object–attribute hierarchy.
- The mathematical properties of Horosphere projection are elegant: it is distance-preserving while guaranteeing compositionality—hyperbolic space preserves hierarchy, and the zero-curvature submanifold preserves linear combination.
- Qualitative composition paths are intuitive: "robot" + "metal" + "gold" → "golden metal robot."

## Limitations & Future Work

- **SIM trade-off**: A tension exists between compositionality and single-concept reconstruction fidelity; SIM_I is approximately 5% below ICE.
- **Predefined object/attribute counts**: $N$ and $M$ must be specified in advance, which reduces flexibility in complex scenes.
- **Inference efficiency not analyzed**: The computational overhead of hyperbolic operations and Horosphere projection in high-dimensional embedding spaces is not discussed.
- **Evaluated solely on Stable Diffusion**: Generalizability to other T2I models (e.g., DALL-E, Imagen) remains to be verified.

## Related Work & Insights

- **vs. ICE**: ICE disentangles objects and attributes but does not guarantee compositionality, making composition paths difficult to interpret; HyperExpress achieves reversible decomposition and recombination via hyperbolic space and HP projection.
- **vs. CCE**: CCE accounts for compositionality but requires multiple images and operates in Euclidean space, making it ill-suited to capture hierarchical relationships.
- **vs. ConceptExpress / Break-A-Scene**: These methods extract only object-level concepts and cannot separate attributes.
- **Insights**: The application of hyperbolic space to visual concept modeling warrants further exploration; compositionality as a core metric for interpretability has broad applicability.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The task formulation (CI-ICE) is innovative, the method design (hyperbolic space + Horosphere projection) is mathematically elegant with clear motivation, and ACC metrics achieve substantial improvements (+55%). The three-module design cleanly decouples responsibilities: HCL handles hierarchy, HEL handles entailment, and HP handles compositionality. Points are deducted for the SIM trade-off and for validation on only a single T2I model.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](../../ACL2026/interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[CVPR 2026\] Measuring the (Un)Faithfulness of Concept-Based Explanations](measuring_the_unfaithfulness_of_concept-based_explanations.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](../../NeurIPS2025/interpretability/toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)

<!-- RELATED:END -->
