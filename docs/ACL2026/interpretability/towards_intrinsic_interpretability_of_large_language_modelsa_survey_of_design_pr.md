---
title: >-
  [Paper Note] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures
description: >-
  [ACL 2026][Interpretability][Intrinsic interpretability] Ours systematically reviews the latest progress in the intrinsic interpretability of LLMs, categorizing existing methods into five major design paradigms (Functional Transparency, Concept Alignment, Representational Decomposability, Explicit Modularity, and Latent Sparse Induction), while discussing open challenges and future directions.
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Intrinsic interpretability"
  - "Large Language Models"
  - "Design paradigm taxonomy"
  - "Modular architecture"
  - "Sparse induction"
date: 2026-05-08
content_hash: f1b571a8dba20c1b
---

# Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures

**Conference**: ACL 2026  
**arXiv**: [2604.16042](https://arxiv.org/abs/2604.16042)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Intrinsic interpretability, Large Language Models, Design paradigm taxonomy, Modular architecture, Sparse induction

## TL;DR

Ours systematically reviews the latest progress in the intrinsic interpretability of LLMs, categorizing existing methods into five major design paradigms (Functional Transparency, Concept Alignment, Representational Decomposability, Explicit Modularity, and Latent Sparse Induction), while discussing open challenges and future directions.

## Background & Motivation

**Background**: Large Language Models have achieved significant success across various NLP tasks, but the opacity of their internal mechanisms (black-box nature) hinders trustworthy deployment, especially in high-risk fields such as healthcare and law. Existing surveys on explainable AI primarily focus on post-hoc explanation methods, such as LIME, SHAP, Sparse Autoencoders, and causal intervention.

**Limitations of Prior Work**: Post-hoc explanation methods explain already trained models through external approximations, leading to a "fidelity gap"—a fundamental bias between the explanation and the model's true computation. Even causal intervention methods (e.g., ROME), while exhibiting stronger local fidelity, provide an explanation granularity that is too fine to aggregate into a coherent understanding of the model's overall behavior.

**Key Challenge**: Historically, intrinsically interpretable models (e.g., linear models, decision trees) have been far inferior to black-box large models in terms of expressive power, leading the "interpretability vs. performance" trade-off to be viewed as irreconcilable. However, recent research suggests that this trade-off is being broken by embedding inductive biases such as modularity, sparsity, and disentanglement into modern architectures.

**Goal**: To provide a unified taxonomic framework for intrinsic interpretability methods, systematically organize design principles, clarify the advantages and disadvantages of each method and their applicable scenarios, and point out future research directions.

**Key Insight**: Unlike post-hoc explanation surveys that start from "tools," this paper starts from "design principles," focusing on how to build transparency from the architecture and training process.

**Core Idea**: Organize intrinsic interpretability methods into five design paradigms, each representing a different "source of transparency."

## Method

### Overall Architecture

Rather than starting from "post-hoc explanation tools," this paper uses the "source of transparency" as the main axis to organize intrinsic interpretability methods into a five-paradigm taxonomic system: Functional Transparency, Concept Alignment, Representational Decomposability, Explicit Modularity, and Latent Sparse Induction. These five paradigms address different facets of the same question—whether interpretability should be embedded into the computation process itself, the representation space, or the network structure. The design principles of the first three paradigms are expanded below (the latter two are discussed in the experimental comparison table), with the training cost spectrum summarized at the end.

### Key Designs

**1. Functional Transparency: Making every step of computation inherently readable**

This is the most direct source of transparency—if the computation process itself is transparent, external approximation tools are no longer needed for post-hoc explanation. Representative methods include Generalized Additive Models (GAMs) and their extensions (GA2M, EBMs, GAMI-Net), which use additivity constraints to isolate and visualize the contribution of each feature; Self-Explaining Neural Networks (SENN) decompose predictions into basis concepts and corresponding relevance scores; B-cos networks ensure the forward computation is equivalent to a linear explanation through weight-input alignment transforms; Kolmogorov-Arnold Networks (KANs) replace fixed activations with learnable spline functions, making the shape functions on every edge readable. The cost is that additivity constraints limit modeling capability, and whether KANs can scale to large-scale LLMs remains unverified.

**2. Concept Alignment: Binding internal representations to human concepts**

Concepts are the basic units of human thought; therefore, aligning the model's intermediate representations with human-understandable concepts often yields the most natural explanations. Concept Bottleneck Models (CBMs) force the prediction of a set of human-defined concepts in an intermediate layer, then make final predictions based solely on these concepts; CB-LLM applies this idea to LLMs, maintaining performance while introducing bottlenecks through hybrid bottlenecks and adversarial training; Label-free CBM uses CLIP to automatically discover concepts, bypassing manual annotation; Codebook Features use vector quantization to obtain discretized concept encodings. The primary risk is that concept definitions usually require domain experts, and residual channels in hybrid bottlenecks may leak information, bypassing the bottleneck and weakening explanation fidelity.

**3. Representational Decomposability: Extracting independent readable components at the representation level**

This paradigm does not change the overall architecture but introduces decomposition structures within the representation space. The Backpack language model learns multiple "sense vectors" for each word and combines them using contextual weights, allowing the tracking of which specific sense is activated for a word in its current context; CoCoMix predicts continuous concepts during training and mixes them into representations, ensuring concept-level information remains traceable throughout the forward pass. Its advantage is local modification and compatibility with existing architectures, while the cost is the additional inference overhead introduced by mechanisms like sense vectors in Backpack.

### Loss & Training

As this is a survey, it does not involve specific training. However, it summarizes the training cost characteristics of each paradigm: Functional Transparency and Concept Alignment methods have low-to-medium training costs; Explicit Modularity (MoE) methods have medium-to-high costs; Latent Sparse Induction (e.g., $L_0$ regularization) has extremely high costs.

## Key Experimental Results

### Main Results

Comprehensive comparison table (excerpt from Table 1):

| Category | Representative Methods | Source of Interpretability | Training Cost | Inference Cost | Performance Impact |
|---------|---------|------------|---------|---------|---------|
| Functional Transparency | KANs, B-cos LMs | Shape functions / Linear explanation | Med-High | Med-High | ≈ Baseline |
| Concept Alignment | CB-LLM, CBMs | Concept scores | High | Low | ↓ or ≈ |
| Representational Decomb. | Backpack, CoCoMix | Sense vectors / Continuous concepts | Med | High | ↓ or ≈ |
| Explicit Modularity | MoE-X, MONET | Sparse experts / Monosemantic experts | Low-High | Low-Med | ≈ or ↑ |
| Sparse Induction | Weight-Sparse, GLU | Sparse circuits / Activation paths | Extreme/Low | Low | ↓ or ≈ |

### Ablation Study

Comparison of the interpretability-performance trade-off across paradigms:

| Paradigm | Fidelity | Granularity | Scalability | Performance Retention |
|------|-------|------|---------|---------|
| Functional Transparency | Highest | Feature-level | Poor | Medium |
| Concept Alignment | High | Concept-level | Medium | Medium |
| Representational Decomb. | Medium | Word/Concept-level | Medium | Medium |
| Explicit Modularity | Medium | Expert/Routing-level | Good | Good |
| Sparse Induction | Med-High | Circuit/Neuron-level | Good | Medium |

### Key Findings

- Explicit Modularity (MoE-type methods) is the most promising paradigm due to its advantages in scalability and performance retention.
- Functional Transparency methods offer the highest fidelity but the worst scalability, making them difficult to apply directly to LLMs with billions of parameters.
- Concept Alignment methods rely on manual concept definitions; CB-LLM has begun exploring automated concept discovery, but it remains in the early stages.
- Weight-sparse models produced by $L_0$ regularization offer circuit-level interpretability but incur extremely high training costs (approx. 3x standard training).
- GLU/SwiGLU represents "free" sparse induction—already used by nearly all modern LLMs, though its interpretability potential is not yet fully exploited.

## Highlights & Insights

- The **five-paradigm taxonomic framework** is highly clear and practical—unifying fragmented literature under common design principles, making it easy for researchers to position their work and identify research gaps.
- The **argument that "interpretability does not necessarily sacrifice performance"** is powerful—methods like MoE-X and B-cos LMs demonstrate that carefully designed inductive biases can provide interpretability while maintaining performance.
- The **potential for cross-paradigm combinations** is explicitly identified—for example, combining Concept Alignment with Explicit Modularity (Concept Bottleneck + MoE) or Representational Decomposability with Sparse Induction opens up vast research spaces.

## Limitations & Future Work

- Most intrinsic interpretability methods have only been validated on small-to-medium scale models; whether they can scale to 10B/100B parameter LLMs remains uncertain.
- There is a lack of unified interpretability evaluation metrics—definitions and measurement standards for "interpretability" vary across different methods.
- Research on internal interpretability for multimodal large models is almost non-existent.
- Future directions include: combining interpretability with safety alignment, interpretable reasoning chain tracking, and interpretability analysis of dynamic sparse activations.

## Related Work & Insights

- **vs. Post-hoc explanation surveys (Madsen et al., 2022; Zhao et al., 2024)**: These surveys focus on tools for analyzing already trained models (e.g., probing, attention visualization), whereas ours focuses on building transparency from the design level.
- **vs. Mechanistic Interpretability (Sharkey et al., 2025)**: Mechanistic interpretability is the direction in post-hoc methods closest to intrinsic interpretability, but it remains "reverse engineering" rather than "forward design."

## Rating

- Novelty: ⭐⭐⭐⭐ The five-paradigm taxonomic framework is a new contribution, but the survey itself does not propose a new method.
- Experimental Thoroughness: ⭐⭐⭐ Survey paper; lacks original experiments, but the comparative organization in Table 1 is of high reference value.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear classification and comprehensive coverage; suitable as an introductory guide to the field.
- Value: ⭐⭐⭐⭐ Provides a much-needed structural framework for the rapidly growing field of intrinsic interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](../../ICML2026/interpretability/prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ICLR 2026\] Medical Interpretability and Knowledge Maps of Large Language Models](../../ICLR2026/interpretability/medical_interpretability_and_knowledge_maps_of_large_language_models.md)

</div>

<!-- RELATED:END -->
