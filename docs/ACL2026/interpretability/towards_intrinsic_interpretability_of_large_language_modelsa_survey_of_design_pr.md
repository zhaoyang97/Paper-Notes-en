---
title: >-
  [Paper Note] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures
description: >-
  [ACL 2026][Intrinsic interpretability] This paper presents a systematic survey of recent advances in intrinsic interpretability of LLMs, organizing existing methods into five design paradigms (functional transparency, concept alignment, representational decomposability, explicit modularity, and latent sparsity induction), and discusses open challenges and future directions.
tags:
  - ACL 2026
  - Intrinsic interpretability
  - large language models
  - design paradigm taxonomy
  - modular architecture
  - latent sparsity induction
date: 2026-05-08
content_hash: 917c43dad89eb444
---

# Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures

**Conference**: ACL 2026
**arXiv**: [2604.16042](https://arxiv.org/abs/2604.16042)
**Code**: None
**Area**: Interpretability
**Keywords**: Intrinsic interpretability, large language models, design paradigm taxonomy, modular architecture, latent sparsity induction

## TL;DR

This paper presents a systematic survey of recent advances in intrinsic interpretability of LLMs, organizing existing methods into five design paradigms (functional transparency, concept alignment, representational decomposability, explicit modularity, and latent sparsity induction), and discusses open challenges and future directions.

## Background & Motivation

**Background**: Large language models have achieved remarkable success across a wide range of NLP tasks, yet the opacity of their internal mechanisms (black-box nature) hinders trustworthy deployment, particularly in high-stakes domains such as healthcare and law. Existing surveys on explainable AI primarily focus on post-hoc explanation methods, such as LIME, SHAP, sparse autoencoders, and causal intervention techniques.

**Limitations of Prior Work**: Post-hoc explanation methods approximate already-trained models via external tools, introducing a "faithfulness gap"—a fundamental discrepancy between the explanation and the model's true computation. Even causal intervention approaches (e.g., ROME), while offering stronger local faithfulness, operate at too fine a granularity to be aggregated into a coherent understanding of overall model behavior.

**Key Challenge**: Historically, intrinsically interpretable models (e.g., linear models, decision trees) were far less expressive than black-box large models, framing interpretability vs. performance as an irreconcilable trade-off. Recent work, however, suggests that embedding inductive biases such as modularity, sparsity, and disentanglement into modern architectures can break this trade-off.

**Goal**: To provide a unified taxonomic framework for intrinsic interpretability methods, systematically review their underlying design principles, clarify the strengths and limitations of each approach, and identify directions for future research.

**Key Insight**: Unlike post-hoc surveys that take "tools" as the entry point, this paper starts from "design principles," focusing on how transparency can be built into architectures and training processes.

**Core Idea**: Intrinsic interpretability methods are organized into five design paradigms, each representing a distinct "source of transparency."

## Method

### Overall Architecture

The proposed taxonomy comprises five design paradigms that introduce interpretability into LLMs from different perspectives:

### Key Designs

1. **Functional Transparency**:

    - Function: Ensures that each computational step of the model is itself interpretable.
    - Mechanism: Includes generalized additive models (GAMs) and their extensions (GA2M, EBMs, GAMI-Net), which use additive constraints to visualize each feature's contribution; self-explaining neural networks (SENNs) that decompose predictions into basis concepts and relevance scores; B-cos networks that produce linear explanations through weight-input alignment transformations; and Kolmogorov-Arnold Networks (KANs) that replace fixed activation functions with learnable spline functions.
    - Design Motivation: The most direct source of interpretability—if the computation itself is transparent, external explanation tools are unnecessary. The limitation is that additive constraints restrict modeling capacity, and the applicability of KANs to large-scale LLMs has yet to be validated.

2. **Concept Alignment**:

    - Function: Aligns internal model representations with human-understandable concepts.
    - Mechanism: Concept bottleneck models (CBMs) enforce prediction of human-defined concepts at intermediate layers before making final predictions; CB-LLM extends this to LLMs via a hybrid bottleneck combined with adversarial training to preserve performance; Label-free CBM leverages CLIP to automatically discover concepts; Codebook Features achieve discretized concept encoding through vector quantization.
    - Design Motivation: Concepts are the basic units of human cognition; aligning model representations with concepts yields the most natural explanations. However, concept definition requires domain expertise, and residual channels in hybrid CBMs may leak information that bypasses the bottleneck.

3. **Representational Decomposability**:

    - Function: Enables model representations to be decomposed into independent, interpretable components.
    - Mechanism: Backpack language models learn multiple "sense vectors" per word, combined via context-dependent weights; CoCoMix predicts continuous concepts during training and mixes them into representations, maintaining concept-level traceability.
    - Design Motivation: Introduces decomposition structure at the representation level without altering the overall architecture. Backpack's advantage lies in its ability to trace which sense of each word is activated, though at higher inference cost.

### Loss & Training

As a survey, this paper does not involve specific training procedures. However, it summarizes the training cost characteristics of each paradigm: functional transparency and concept alignment methods incur low-to-medium training costs; explicit modularity (MoE) methods incur medium-to-high costs; and latent sparsity induction methods (e.g., $L_0$ regularization) incur extremely high costs.

## Key Experimental Results

### Main Results

Summary comparison table (excerpted from Table 1):

| Category | Representative Methods | Source of Interpretability | Training Cost | Inference Cost | Performance Impact |
|---|---|---|---|---|---|
| Functional Transparency | KANs, B-cos LMs | Shape functions / linear explanations | Medium–High | Medium–High | ≈ Baseline |
| Concept Alignment | CB-LLM, CBMs | Concept scores | High | Low | ↓ or ≈ |
| Representational Decomposability | Backpack, CoCoMix | Sense vectors / continuous concepts | Medium | High | ↓ or ≈ |
| Explicit Modularity | MoE-X, MONET | Sparse experts / monosemantic experts | Low–High | Low–Medium | ≈ or ↑ |
| Sparsity Induction | Weight-Sparse, GLU | Sparse circuits / activation paths | Extremely High / Low | Low | ↓ or ≈ |

### Ablation Study

Interpretability–performance trade-off comparison across paradigms:

| Paradigm | Faithfulness | Granularity | Scalability | Performance Retention |
|---|---|---|---|---|
| Functional Transparency | Highest | Feature-level | Poor | Medium |
| Concept Alignment | High | Concept-level | Medium | Medium |
| Representational Decomposability | Medium | Word / concept-level | Medium | Medium |
| Explicit Modularity | Medium | Expert / routing-level | Good | Good |
| Sparsity Induction | Medium–High | Circuit / neuron-level | Good | Medium |

### Key Findings

- Explicit modularity (MoE-based methods) offers the greatest advantages in scalability and performance retention, making it currently the most promising paradigm.
- Functional transparency methods achieve the highest faithfulness but the poorest scalability, making direct application to billion-parameter LLMs difficult.
- Concept alignment methods rely on manually defined concepts; CB-LLM has begun exploring automatic concept discovery, but this remains in an early stage.
- Weight-sparse models induced via $L_0$ regularization, while yielding interpretable circuits, incur extremely high training costs (approximately 3× standard training).
- GLU/SwiGLU constitutes "free" sparsity induction—already adopted by nearly all modern LLMs—yet its interpretability potential remains largely unexplored.

## Highlights & Insights

- The **five-paradigm taxonomy** is notably clear and practical—it unifies a fragmented literature under shared design principles, helping researchers situate their own work and identify research gaps.
- The **argument that interpretability need not sacrifice performance** is compelling—methods such as MoE-X and B-cos LMs demonstrate that carefully designed inductive biases can provide interpretability while preserving model performance.
- The paper explicitly identifies **the potential of cross-paradigm combinations**—for instance, combining concept alignment with explicit modularity (concept bottleneck + MoE), or representational decomposability with sparsity induction, opening broad avenues for future research.

## Limitations & Future Work

- Most intrinsic interpretability methods have only been validated on small-to-medium-scale models; whether they scale to tens or hundreds of billions of parameters remains uncertain.
- There is no unified evaluation metric for interpretability—different methods adopt inconsistent definitions and measures of "interpretability."
- Research on intrinsic interpretability for multimodal large models is nearly absent.
- Future directions include: integrating interpretability with safety alignment, interpretable reasoning chain tracing, and interpretability analysis of dynamic sparse activation.

## Related Work & Insights

- **vs. Post-hoc explanation surveys (Madsen et al., 2022; Zhao et al., 2024)**: These surveys focus on tools for analyzing already-trained models (e.g., probes, attention visualization), whereas this paper focuses on building transparency at the design level.
- **vs. Mechanistic interpretability (Sharkey et al., 2025)**: Mechanistic interpretability is the post-hoc approach most closely related to intrinsic interpretability, but it remains "reverse engineering" rather than "forward design."

## Rating

- Novelty: ⭐⭐⭐⭐ The five-paradigm taxonomy is an original contribution, though the survey itself does not propose new methods.
- Experimental Thoroughness: ⭐⭐⭐ As a survey paper, no original experiments are conducted; however, the comparative synthesis in Table 1 is highly informative.
- Writing Quality: ⭐⭐⭐⭐⭐ The taxonomy is clear, coverage is comprehensive, and the paper serves well as an introductory guide to the field.
- Value: ⭐⭐⭐⭐ Provides a much-needed structured framework for the rapidly growing field of intrinsic interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Experiments or Outcomes? Probing Scientific Feasibility in Large Language Models](experiments_or_outcomes_probing_scientific_feasibility_in_large_language_models.md)
- [\[ACL 2026\] Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models](revitalizing_black-box_interpretability_actionable_interpretability_for_llms_via.md)
- [\[CVPR 2026\] CI-ICE: Intrinsic Concept Extraction Based on Compositional Interpretability](../../CVPR2026/interpretability/ciice_intrinsic_concept_extraction_compositional.md)
- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](../../NeurIPS2025/interpretability/the_trilemma_of_truth_in_large_language_models.md)

</div>

<!-- RELATED:END -->
