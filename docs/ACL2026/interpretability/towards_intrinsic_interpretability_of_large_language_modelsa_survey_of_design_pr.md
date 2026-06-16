---
title: >-
  [Paper Note] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures
description: >-
  [ACL 2026][Interpretability][Paper Note] This paper provides a systematic survey of recent advances in the intrinsic interpretability of LLMs. It categorizes existing methods into five design paradigms (functional transparency, concept alignment, representational decomposability, explicit modularization, and latent sparse induction) and discusses open challen
tags:
  - ACL 2026
  - Interpretability
date: 2026-05-08
content_hash: 9d16825326cd92dc
---
# Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures

**Conference**: ACL 2026  
**arXiv**: [2604.16042](https://arxiv.org/abs/2604.16042)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Intrinsic Interpretability, Large Language Models, Design Paradigm Taxonomy, Modular Architectures, Sparse Induction

## TL;DR

This paper provides a systematic survey of recent advances in the intrinsic interpretability of LLMs. It categorizes existing methods into five design paradigms (functional transparency, concept alignment, representational decomposability, explicit modularization, and latent sparse induction) and discusses open challenges and future directions.

## Background & Motivation

**Background**: Large Language Models (LLMs) have achieved remarkable success across various NLP tasks, but their internal "black-box" nature hinders trusted deployment, especially in high-stakes domains like healthcare and law. Existing surveys on Explainable AI (XAI) primarily focus on post-hoc explanation methods, such as LIME, SHAP, Sparse Autoencoders (SAEs), and causal interventions.

**Limitations of Prior Work**: Post-hoc methods explain pre-trained models through external approximations, leading to a "fidelity gap"—a fundamental bias exists between the explanation and the model's actual computation. While causal interventions (e.g., ROME) offer stronger local fidelity, their granular nature makes it difficult to aggregate findings into a coherent understanding of overall model behavior.

**Key Challenge**: Historically, intrinsically interpretable models (e.g., linear models, decision trees) were significantly less expressive than black-box LLMs, leading to a perceived irreconcilable trade-off between interpretability and performance. However, recent research suggests that this trade-off is being challenged by embedding inductive biases—such as modularity, sparsity, and disentanglement—directly into modern architectures.

**Goal**: To provide a unified taxonomic framework for intrinsic interpretability methods, systematically review design principles, clarify the strengths and weaknesses of various approaches, and identify future research directions.

**Key Insight**: Instead of approaching interpretability via "tools" as in post-hoc surveys, this paper focuses on "design principles" to construct transparency from the architecture and training process themselves.

**Core Idea**: The paper organizes intrinsic interpretability methods into five design paradigms, each representing a distinct "source of transparency."

## Method

### Overall Architecture

Rather than focusing on post-hoc tools, the paper utilizes the "source of transparency" as the main axis to organize intrinsic methods into five paradigms: functional transparency, concept alignment, representational decomposability, explicit modularization, and latent sparse induction. These paradigms address different facets of the same question—whether interpretability should be embedded in the computation process, the representation space, or the network structure. The design principles for the first three paradigms are detailed below, while the latter two are discussed in the comparison table. Training cost spectrums for each paradigm are also summarized.

### Key Designs

**1. Functional Transparency: Making every computational step readable**

As the most direct source of transparency, this paradigm ensures the computation process itself is human-readable, removing the need for post-hoc approximations. Representative methods include Generalized Additive Models (GAMs) and their extensions (GA2M, EBMs, GAMI-Net), which use additivity constraints to isolate and visualize the contribution of each feature. Self-Explaining Neural Networks (SENN) decompose predictions into basis concepts and relevance scores. B-cos networks use weight-input alignment transforms to make forward computation equivalent to a linear explanation. Kolmogorov-Arnold Networks (KANs) replace fixed activations with learnable splines, making the shape function on every edge readable. The trade-off is that additivity constraints limit modeling capacity, and the scalability of KANs to large-scale LLMs remains unverified.

**2. Concept Alignment: Binding internal representations to human concepts**

Since concepts are the fundamental units of human thought, aligning intermediate representations with human-understandable concepts provides natural explanations. Concept Bottleneck Models (CBMs) force the prediction of a set of human-defined concepts in intermediate layers before making a final prediction. CB-LLM applies this to LLMs using hybrid bottlenecks and adversarial training to maintain performance. Label-free CBMs use CLIP to automatically discover concepts, bypassing manual annotation. Codebook Features utilize vector quantization to obtain discretized concept encodings. The primary limitation is that concept definition often requires domain experts, and residual channels in hybrid bottlenecks may leak information, bypassing the bottleneck and weakening fidelity.

**3. Representational Decomposability: Extracting independent readable components in representation space**

This paradigm introduces decomposition structures into the representation space without altering the overall back-end architecture. Backpack language models learn multiple "sense vectors" for each word and combine them using contextual weights, allowing the model to track which specific sense is activated in a given context. CoCoMix predicts continuous concepts during training and mixes them into representations, ensuring concept-level information remains traceable throughout the forward pass. This approach is locally non-intrusive and compatible with existing architectures, though mechanisms like Backpack's sense vectors incur additional inference overhead.

### Loss & Training

As a survey, this paper does not detail specific training procedures but summarizes the cost profiles of each paradigm: Functional transparency and concept alignment have Low-to-Medium training costs; explicit modularization (MoE) has Medium-to-High costs; and latent sparse induction (e.g., $L_0$ regularization) has extremely high costs.

## Key Experimental Results

### Main Results

Summary comparison (adapted from Table 1):

| Category | Representative Methods | Source of Interpretability | Training Cost | Inference Cost | Performance Impact |
|---------|---------|------------|---------|---------|---------|
| Functional Transparency | KANs, B-cos LMs | Shape functions / Linear explanation | Med-High | Med-High | ≈ Baseline |
| Concept Alignment | CB-LLM, CBMs | Concept scores | High | Low | ↓ or ≈ |
| Representational Decom. | Backpack, CoCoMix | Sense vectors / Continuous concepts | Medium | High | ↓ or ≈ |
| Explicit Modularization | MoE-X, MONET | Sparse/Monosemantic experts | Low-High | Low-Med | ≈ or ↑ |
| Sparse Induction | Weight-Sparse, GLU | Sparse circuits / Activation paths | Extreme/Low | Low | ↓ or ≈ |

### Ablation Study

Interpretability-performance trade-off comparison by paradigm:

| Paradigm | Fidelity | Granularity | Scalability | Perf. Retention |
|------|-------|------|---------|---------|
| Functional Transparency | Highest | Feature-level | Poor | Medium |
| Concept Alignment | High | Concept-level | Medium | Medium |
| Representational Decom. | Medium | Word/Concept-level | Medium | Medium |
| Explicit Modularization | Medium | Expert/Router-level | Good | Good |
| Sparse Induction | Med-High | Circuit/Neuron-level | Good | Medium |

### Key Findings

- Explicit modularization (MoE-based methods) offers the best balance of scalability and performance retention, making it the most promising current paradigm.
- Functional transparency provides the highest fidelity but struggles with scalability, making it difficult to apply to models with billions of parameters.
- Concept alignment depends heavily on manual definitions; while CB-LLM explores automatic discovery, it remains in the early stages.
- $L_0$ regularization produces weight-sparse models with interpretable circuits, but training costs are prohibitive (approx. 3x standard training).
- GLU/SwiGLU layers represent "free" sparse induction—nearly all modern LLMs use them, yet their interpretability potential remains under-explored.

## Highlights & Insights

- **The five-paradigm taxonomy** is clear and practical, unifying disparate literature under common principles to help researchers identify research gaps.
- **The argument that "interpretability does not necessarily sacrifice performance"** is compelling—methods like MoE-X and B-cos LMs demonstrate that well-designed inductive biases can provide transparency while maintaining or even improving performance.
- **Potential for cross-paradigm combinations** is explicitly highlighted, such as combining concept alignment with explicit modularization (Concept Bottleneck + MoE) or combining representational decomposability with sparse induction.

## Limitations & Future Work

- Most intrinsic interpretability methods have only been validated on small-to-medium scale models; scalability to 10B+ or 100B+ parameter LLMs remains uncertain.
- There is a lack of unified evaluation metrics for interpretability across different paradigms.
- Research into the intrinsic interpretability of multimodal LLMs is virtually non-existent.
- Future directions include: integrating interpretability with safety alignment, tracking interpretable reasoning chains, and analyzing the interpretability of dynamic sparse activations.

## Related Work & Insights

- **vs. Post-hoc Surveys (Madsen et al., 2022; Zhao et al., 2024)**: These focus on tools for analyzing pre-trained models (e.g., probing, attention visualization), whereas this paper focuses on building transparency through design.
- **vs. Mechanistic Interpretability (Sharkey et al., 2025)**: Mechanistic interpretability is the post-hoc direction closest to intrinsic methods, but it typically involves "reverse engineering" rather than "forward design."

## Rating

- Novelty: ⭐⭐⭐⭐ (The five-paradigm framework is a significant contribution, though the survey itself does not propose a new method.)
- Experimental Thoroughness: ⭐⭐⭐ (Survey paper with no original experiments, but Table 1's meta-analysis is highly valuable.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Exceptionally clear classification and comprehensive coverage; serves as an excellent entry guide to the field.)
- Value: ⭐⭐⭐⭐ (Provides a much-needed structured framework for the rapidly growing field of intrinsic interpretability.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](../../ICML2026/interpretability/prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)

</div>

<!-- RELATED:END -->
