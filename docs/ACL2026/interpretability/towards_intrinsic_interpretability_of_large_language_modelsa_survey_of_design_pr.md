---
title: >-
  [Paper Note] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures
description: >-
  [ACL 2026][Interpretability][Intrinsic Interpretability] This paper provides a systematic survey of recent progress in the intrinsic interpretability of LLMs. It categorizes existing methods into five major design paradi…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Intrinsic Interpretability"
  - "Large Language Models"
  - "Taxonomy of Design Paradigms"
  - "Modular Architecture"
  - "Sparse Induction"
date: 2026-05-08
content_hash: b8ab7d24b3a77b9d
---

# Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures

**Conference**: ACL 2026  
**arXiv**: [2604.16042](https://arxiv.org/abs/2604.16042)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Intrinsic Interpretability, Large Language Models, Taxonomy of Design Paradigms, Modular Architecture, Sparse Induction

## TL;DR

This paper provides a systematic survey of recent progress in the intrinsic interpretability of LLMs. It categorizes existing methods into five major design paradigms (functional transparency, concept alignment, representational decomposability, explicit modularity, and latent sparse induction) and discusses open challenges and future research directions.

## Background & Motivation

**Background**: Large Language Models (LLMs) have achieved remarkable success across various NLP tasks, but the opacity of their internal mechanisms (black-box nature) hinders trustworthy deployment, particularly in high-risk domains such as healthcare and law. Existing Explainable AI (XAI) surveys primarily focus on post-hoc explanation methods, such as LIME, SHAP, Sparse Autoencoders (SAEs), and causal interventions.

**Limitations of Prior Work**: Post-hoc explanation methods interpret pre-trained models through external approximations, leading to a "fidelity gap"—a fundamental discrepancy between the explanation and the model's actual computation. Even causal intervention methods (e.g., ROME), while exhibiting stronger local fidelity, suffer from a granularity that is too fine to aggregate into a coherent understanding of the model's overall behavior.

**Key Challenge**: Historically, intrinsically interpretable models (e.g., linear models, decision trees) have lacked the expressive power of black-box LLMs, leading to the perception of an irreconcilable trade-off between interpretability and performance. However, recent research suggests that this trade-off is being challenged by embedding inductive biases such as modularity, sparsity, and disentanglement directly into modern architectures.

**Goal**: This paper aims to provide a unified classification framework for intrinsic interpretability methods, systematically organize design principles, clarify the advantages and disadvantages of different approaches, and identify future research directions.

**Key Insight**: Unlike surveys on post-hoc explanations that start from "tools," this paper approaches the problem from "design principles," focusing on how to build transparency from the architecture and training process level.

**Core Idea**: The paper organizes intrinsic interpretability methods into five design paradigms, each representing a different "source of transparency."

## Method

### Overall Architecture

The classification system proposed in this paper includes five design paradigms that introduce interpretability into LLMs at different levels:

### Key Designs

1.  **Functional Transparency**:
    - **Function**: Ensures that each computational step of the model is inherently interpretable.
    - **Mechanism**: Includes Generalized Additive Models (GAMs) and their extensions (GA2Ms, EBMs, GAMI-Net), which use additivity constraints to visualize individual feature contributions; Self-Explaining Neural Networks (SENN) that decompose into basis concepts and relevance scores; B-cos networks that produce linear explanations through weight-input alignment; and Kolmogorov-Arnold Networks (KANs), which replace fixed activation functions with learnable splines.
    - **Design Motivation**: This is the most direct source of interpretability—if the computation itself is transparent, external tools are unnecessary. Limitations include additivity constraints that restrict modeling capacity and the unverified scalability of KANs for large-scale LLMs.

2.  **Concept Alignment**:
    - **Function**: Aligns internal model representations with human-understandable concepts.
    - **Mechanism**: Concept Bottleneck Models (CBMs) force the prediction of human-defined concepts in intermediate layers before making the final prediction; CB-LLM extends this to LLMs via hybrid bottlenecks and adversarial training to maintain performance; Label-free CBM uses CLIP for automated concept discovery; Codebook Features implement discretized concept encoding via vector quantization.
    - **Design Motivation**: Concepts are the fundamental units of human thought. Aligning representations with concepts produces natural explanations. However, concept definition requires domain expertise, and residual channels (in hybrid CBMs) may leak information that bypasses the bottleneck.

3.  **Representational Decomposability**:
    - **Function**: Makes model representations decomposable into independent, interpretable components.
    - **Mechanism**: Backpack Language Models learn multiple "sense vectors" for each word, combined via contextual weights; CoCoMix predicts continuous concepts during training and mixes them into representations, maintaining traceability at the concept level.
    - **Design Motivation**: These methods introduce decomposition structures at the representation level without changing the overall architecture. The advantage of Backpack is the ability to track which sense of a word is activated, though it incurs higher inference overhead.

### Loss & Training

While this is a survey, it summarizes the training cost characteristics of each paradigm: Functional transparency and concept alignment have low-to-medium training costs; explicit modularity (MoE) has medium-to-high costs; and latent sparse induction (e.g., $L_0$ regularization) involves extremely high costs.

## Key Experimental Results

### Main Results

Summary comparison (Selected from Table 1):

| Method Category | Representative Methods | Source of Interpretability | Training Cost | Inference Cost | Performance Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Functional Transparency | KANs, B-cos LMs | Shape Functions / Linear Exp | Med-High | Med-High | ≈ Baseline |
| Concept Alignment | CB-LLM, CBMs | Concept Scores | High | Low | ↓ or ≈ |
| Repr. Decomposability | Backpack, CoCoMix | Sense Vectors / Cont. Concepts | Med | High | ↓ or ≈ |
| Explicit Modularity | MoE-X, MONET | Sparse/Monosemantic Experts | Low-High | Low-Med | ≈ or ↑ |
| Sparse Induction | Weight-Sparse, GLU | Sparse Circuits / Act. Paths | Very High/Low | Low | ↓ or ≈ |

### Ablation Study

Comparison of interpretability-performance trade-offs across paradigms:

| Paradigm | Fidelity | Granularity | Scalability | Perf. Retention |
| :--- | :--- | :--- | :--- | :--- |
| Functional Transparency | Highest | Feature-level | Poor | Medium |
| Concept Alignment | High | Concept-level | Medium | Medium |
| Repr. Decomposability | Medium | Word/Concept-level | Medium | Medium |
| Explicit Modularity | Medium | Expert/Routing-level | Good | Good |
| Sparse Induction | Medium-High | Circuit/Neuron-level | Good | Medium |

### Key Findings

- Explicit modularity (MoE-based methods) shows the most promise in terms of scalability and performance retention.
- Functional transparency offers the highest fidelity but the poorest scalability, making it difficult to apply to LLMs with billions of parameters.
- Concept alignment methods rely on manual concept definitions; although CB-LLM explores automated discovery, it remains in the early stages.
- $L_0$ regularization produces weight-sparse models with interpretable circuits, but training costs are extremely high (~3x standard training).
- GLU/SwiGLU provides "free" sparse induction—widely used in modern LLMs, though its interpretability potential is under-explored.

## Highlights & Insights

- The **five-paradigm classification framework** is clear and practical, unifying disparate literature under common design principles to help researchers locate research gaps.
- The **argument that "interpretability does not necessarily sacrifice performance"** is compelling—methods like MoE-X and B-cos LMs demonstrate that carefully designed inductive biases can provide interpretability while maintaining performance.
- The **potential for cross-paradigm combinations** is explicitly identified, such as combining concept alignment with explicit modularity (Concept Bottleneck + MoE) or representational decomposability with sparse induction.

## Limitations & Future Work

- Most intrinsic interpretability methods have only been validated on small to medium models; it remains uncertain if they scale to 100B+ parameter LLMs.
- There is a lack of unified evaluation metrics for interpretability across different definitions and standards.
- Research on the intrinsic interpretability of multimodal LLMs is almost non-existent.
- Future directions include: combining interpretability with safety alignment, tracking interpretable reasoning chains, and analyzing the interpretability of dynamic sparse activations.

## Related Work & Insights

- **vs. Post-hoc Explanation Surveys (Madsen et al., 2022; Zhao et al., 2024)**: These surveys focus on analyzing pre-trained models using tools like probing or attention visualization, whereas this paper focuses on building transparency by design.
- **vs. Mechanistic Interpretability (Sharkey et al., 2025)**: Mechanistic interpretability is the post-hoc direction closest to intrinsic interpretability, but it remains "reverse engineering" rather than "forward design."

## Rating

- Novelty: ⭐⭐⭐⭐ (The five-paradigm framework is a new contribution, although the survey itself does not propose a new method.)
- Experimental Thoroughness: ⭐⭐⭐ (As a survey, it lacks original experiments, but the comparison in Table 1 is highly valuable.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear classification and comprehensive coverage; suitable as an introductory guide to the field.)
- Value: ⭐⭐⭐⭐ (Provides a much-needed structured framework for the rapidly growing field of intrinsic interpretability.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](../../ICML2026/interpretability/prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)

</div>

<!-- RELATED:END -->
